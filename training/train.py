"""
Training — stage 1: load the frozen snapshot from MinIO and apply the
TEMPORAL split recorded in its manifest.

Split roles:
  train : the model learns here (fits its weights)
  val   : we TUNE here (early stopping, decision threshold) -- the model
          does not learn weights from it
  test  : touched ONCE at the very end for an honest number; no decisions

Split is by TIME, not random: train = earliest, val = middle, test = latest,
mirroring production (learn on the past, predict the future).

Run:
    pip install boto3 pandas pyarrow
    python training/train.py                 # uses the latest snapshot
    python training/train.py --snapshot-id 20260716T102233Z_117650f2472b
"""

import argparse
import io
import json
import os

import numpy as np
import pandas as pd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", default=None, help="default: latest in the bucket")
    p.add_argument("--s3-endpoint", default=os.environ.get("S3_ENDPOINT", "http://127.0.0.1:9000"))
    p.add_argument("--s3-key", default=os.environ.get("MINIO_ROOT_USER", "minioadmin"))
    p.add_argument("--s3-secret", default=os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin_pwd"))
    p.add_argument("--bucket", default="datasets")
    p.add_argument("--dataset-name", default="cardops_fraud")
    p.add_argument("--neg-per-pos", type=int, default=20,
                   help="negatives kept per positive in TRAIN (val/test untouched)")
    p.add_argument("--target-recall", type=float, default=0.80,
                   help="recall the rules engine guarantees; we cut FPs at it")
    p.add_argument("--rules-precision", type=float, default=0.05,
                   help="ASSUMED precision of the rules engine at the target recall")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_URI", "http://127.0.0.1:5000"))
    p.add_argument("--experiment", default="cardops_fraud")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--artifact-dir", default="training/artifacts",
                   help="where to save model+calibrator+schema for the serving API")
    return p.parse_args()


def s3_client(args):
    import boto3
    return boto3.client(
        "s3", endpoint_url=args.s3_endpoint,
        aws_access_key_id=args.s3_key, aws_secret_access_key=args.s3_secret,
        region_name="us-east-1")


def resolve_snapshot_id(s3, args):
    if args.snapshot_id:
        return args.snapshot_id
    # snapshot ids start with a UTC timestamp, so lexical max == newest
    resp = s3.list_objects_v2(Bucket=args.bucket, Prefix=f"{args.dataset_name}/")
    ids = {k["Key"].split("/")[1] for k in resp.get("Contents", [])}
    if not ids:
        raise SystemExit(f"no snapshots under s3://{args.bucket}/{args.dataset_name}/")
    return sorted(ids)[-1]


def load_snapshot(args):
    s3 = s3_client(args)
    snap_id = resolve_snapshot_id(s3, args)
    prefix = f"{args.dataset_name}/{snap_id}"
    manifest = json.loads(
        s3.get_object(Bucket=args.bucket, Key=f"{prefix}/manifest.json")["Body"].read())
    parquet = s3.get_object(Bucket=args.bucket, Key=f"{prefix}/dataset.parquet")["Body"].read()
    df = pd.read_parquet(io.BytesIO(parquet))
    df["authorized_at"] = pd.to_datetime(df["authorized_at"])
    return df, manifest, snap_id


def temporal_split(df, manifest):
    """Cut by the dates recorded in the manifest. train < val < test in time."""
    train_end = pd.Timestamp(manifest["split"]["train_end"])
    val_end = pd.Timestamp(manifest["split"]["val_end"])
    t = df["authorized_at"]
    train = df[t <= train_end]
    val = df[(t > train_end) & (t <= val_end)]
    test = df[t > val_end]
    return train, val, test


def undersample(train, neg_per_pos, seed):
    """Drop negatives from TRAIN only, keeping every positive.
    Returns the balanced frame plus (rate_before, rate_after) for context."""
    pos = train[train["label"] == 1]
    neg = train[train["label"] == 0]
    n_keep = min(len(neg), neg_per_pos * len(pos))
    neg_s = neg.sample(n=n_keep, random_state=seed)
    out = pd.concat([pos, neg_s]).sample(frac=1, random_state=seed).reset_index(drop=True)
    rate_before = len(pos) / len(train)
    rate_after = len(pos) / len(out)
    return out, rate_before, rate_after


CAT = ["channel", "entry_mode", "country", "mcc", "risk_segment"]

CATBOOST_PARAMS = dict(iterations=800, depth=6, learning_rate=0.05, eval_metric="AUC")


def coerce_types(df, features):
    """ClickHouse Decimal -> pandas object; force numerics to float, cats to str."""
    df = df.copy()
    for c in features:
        if c in CAT:
            df[c] = df[c].astype(str)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def train_and_calibrate(train_bal, val, features, seed=42):
    """Train CatBoost (early stop on val), then isotonic-calibrate on val.

    Undersampling inflated the training prior ~13x, so raw probabilities are
    biased high. Isotonic learns raw->true mapping from val's REAL distribution.
    """
    from catboost import CatBoostClassifier, Pool
    from sklearn.isotonic import IsotonicRegression

    tb = coerce_types(train_bal, features)
    va = coerce_types(val, features)
    model = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed, verbose=0)
    model.fit(Pool(tb[features], tb["label"], cat_features=CAT),
              eval_set=Pool(va[features], va["label"], cat_features=CAT),
              early_stopping_rounds=50)

    p_raw = model.predict_proba(va[features])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw, va["label"].values)
    p_cal = iso.predict(p_raw)
    return model, iso, p_raw, p_cal


def pick_threshold(y, p, target_recall):
    """Highest threshold (=> highest precision) that still keeps recall >= target."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def evaluate_at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall, "precision": precision}


def reliability_figure(y, p, n_bins=10):
    """Observed fraud rate vs mean predicted prob, per quantile bin (test)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    xs, ys = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    hi = max(xs + ys + [1e-6])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, hi], [0, hi], "--", color="gray", label="perfect")
    ax.plot(xs, ys, "o-", color="#1D9E75", label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title("Calibration on test")
    ax.legend()
    fig.tight_layout()
    return fig


def log_to_mlflow(args, manifest, snap_id, model, iso, features, metrics, thr, best_iter, y_test, p_test):
    import json
    import os
    import tempfile

    import joblib
    import mlflow
    import mlflow.catboost

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "snapshot_id": snap_id,
            "dataset_sha256": manifest["sha256"][:16],
            "transaction_cutoff": manifest["recipe"]["transaction_cutoff"],
            "maturity_window_days": manifest["recipe"]["maturity_window_days"],
            "merchant_rate_granularity": manifest["recipe"]["merchant_rate_granularity"],
            "neg_per_pos": args.neg_per_pos,
            "seed": args.seed,
            "target_recall": args.target_recall,
            "rules_precision": args.rules_precision,
            "threshold": round(thr, 6),
            "best_iteration": best_iter,
            **{f"cb_{k}": v for k, v in CATBOOST_PARAMS.items()},
        })
        mlflow.log_metrics(metrics)
        try:
            mlflow.catboost.log_model(model, "model")
        except Exception as e:
            # e.g. MLflow 3.x client vs 2.x server (logged-models endpoint 404):
            # fall back to saving the raw .cbm as a plain artifact.
            print(f"  (flavored model log failed: {type(e).__name__}; saving raw .cbm instead)")
            with tempfile.TemporaryDirectory() as md:
                pth = os.path.join(md, "model.cbm")
                model.save_model(pth)
                mlflow.log_artifact(pth, "model")
        try:
            mlflow.log_figure(reliability_figure(y_test, p_test), "calibration_curve.png")
        except Exception as e:
            print(f"  (calibration figure skipped: {e})")
        with tempfile.TemporaryDirectory() as d:
            joblib.dump(iso, os.path.join(d, "calibrator.joblib"))
            with open(os.path.join(d, "feature_schema.json"), "w") as f:
                json.dump({"features": features, "cat_features": CAT,
                           "threshold": thr, "label": "label"}, f, indent=2)
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
            mlflow.log_artifacts(d, "artifacts")
    print(f"logged run to MLflow experiment '{args.experiment}' at {args.mlflow_uri}")


def save_artifacts_local(art_dir, model, iso, features, thr):
    """Persist the SERVING CONTRACT: model + calibrator + feature schema."""
    import joblib
    os.makedirs(art_dir, exist_ok=True)
    model.save_model(os.path.join(art_dir, "model.cbm"))
    joblib.dump(iso, os.path.join(art_dir, "calibrator.joblib"))
    with open(os.path.join(art_dir, "feature_schema.json"), "w") as f:
        json.dump({"features": features, "cat_features": CAT,
                   "threshold": thr, "label": "label"}, f, indent=2)


def describe(name, part):
    n = len(part)
    pos = int(part["label"].sum())
    rate = pos / n if n else 0.0
    lo, hi = part["authorized_at"].min(), part["authorized_at"].max()
    print(f"  {name:<5} rows={n:>7,}  pos={pos:>5}  rate={rate*100:5.3f}%  [{lo}  ..  {hi}]")


def main():
    args = get_args()
    df, manifest, snap_id = load_snapshot(args)
    print(f"snapshot   : {snap_id}")
    print(f"rows total : {len(df):,}  features={len(manifest['feature_columns'])}")

    train, val, test = temporal_split(df, manifest)
    print("temporal split:")
    describe("train", train)
    describe("val", val)
    describe("test", test)

    # sanity: the three windows must not overlap in time
    assert train["authorized_at"].max() <= val["authorized_at"].min()
    assert val["authorized_at"].max() <= test["authorized_at"].min()
    print("check      : no time overlap between splits  OK")

    # ---- undersample TRAIN ONLY -------------------------------------
    train_bal, rate_before, rate_after = undersample(train, args.neg_per_pos, args.seed)
    print(f"\nundersampling train (val/test untouched), {args.neg_per_pos} neg per pos:")
    print(f"  train before : {len(train):>7,} rows  fraud {rate_before*100:6.3f}%")
    print(f"  train after  : {len(train_bal):>7,} rows  fraud {rate_after*100:6.3f}%")
    print(f"  val (real)   : {len(val):>7,} rows  fraud {val['label'].mean()*100:6.3f}%")
    print(f"  test (real)  : {len(test):>7,} rows  fraud {test['label'].mean()*100:6.3f}%")
    print(f"\n  -> the model now sees fraud ~{rate_after/rate_before:.0f}x more often than reality.")
    print("     Its raw probabilities will be inflated -> isotonic calibration fixes that.")

    # ---- train CatBoost + isotonic calibration ----------------------
    features = manifest["feature_columns"]
    model, iso, p_raw, p_cal = train_and_calibrate(train_bal, val, features, args.seed)
    actual = val["label"].mean()
    print(f"\ntrained CatBoost (best iteration {model.get_best_iteration()} of 800 max)")
    print("calibration check on val:")
    print(f"  actual fraud rate    : {actual*100:6.3f}%")
    print(f"  mean RAW prob        : {p_raw.mean()*100:6.3f}%   <- inflated by undersampling")
    print(f"  mean CALIBRATED prob : {p_cal.mean()*100:6.3f}%   <- back to the real scale")
    print("\n  ranking is unchanged (isotonic is monotonic); only the SCALE is fixed.")

    # ---- threshold at target recall (chosen on VAL) -----------------
    thr = pick_threshold(val["label"].values, p_cal, args.target_recall)
    print(f"\nthreshold @ recall {args.target_recall} (picked on val) = {thr:.4f} calibrated prob")

    # ---- persist serving artifacts locally --------------------------
    save_artifacts_local(args.artifact_dir, model, iso, features, thr)
    print(f"saved serving artifacts -> {args.artifact_dir}/ (model, calibrator, feature_schema)")

    # ---- honest evaluation on TEST ----------------------------------
    te = coerce_types(test, features)
    p_test = iso.predict(model.predict_proba(te[features])[:, 1])
    ev = evaluate_at_threshold(test["label"].values, p_test, thr)
    alerts = ev["tp"] + ev["fp"]
    print("\nTEST at that operating point:")
    print(f"  recall    : {ev['recall']*100:5.1f}%   (caught {ev['tp']} of {ev['tp']+ev['fn']} frauds)")
    print(f"  precision : {ev['precision']*100:5.1f}%   ({alerts} alerts, {ev['fp']} false positives)")
    if ev["precision"]:
        print(f"  alerts per real fraud : {1/ev['precision']:.1f}")

    # ---- FP reduction vs an ASSUMED rules operating point -----------
    rp = args.rules_precision
    rules_fp = ev["tp"] / rp - ev["tp"] if rp else float("inf")
    reduction = (1 - ev["fp"] / rules_fp) if rules_fp > 0 else float("nan")
    if rules_fp > 0:
        print(f"\n  vs rules @ same recall (assumed precision {rp:.0%}):")
        print(f"    rules -> ~{rules_fp:,.0f} false positives; model -> {ev['fp']}")
        print(f"    => {reduction*100:.0f}% fewer false positives at the same recall")

    # ---- calibration reliability on TEST (unseen, honest) -----------
    actual_test = test["label"].mean()
    print(f"\ncalibration on TEST: mean pred {p_test.mean()*100:.3f}%  vs  actual {actual_test*100:.3f}%")

    # ---- log the whole thing to MLflow ------------------------------
    if not args.no_mlflow:
        from sklearn.metrics import average_precision_score, roc_auc_score
        yv, yt = val["label"].values, test["label"].values
        metrics = {
            "val_roc_auc": roc_auc_score(yv, p_raw),
            "val_pr_auc": average_precision_score(yv, p_raw),
            "test_roc_auc": roc_auc_score(yt, p_test),
            "test_pr_auc": average_precision_score(yt, p_test),
            "test_recall": ev["recall"],
            "test_precision": ev["precision"],
            "test_tp": ev["tp"], "test_fp": ev["fp"], "test_fn": ev["fn"],
            "test_alerts_per_fraud": (1 / ev["precision"]) if ev["precision"] else 0.0,
            "fp_reduction_vs_rules": reduction,
            "test_mean_pred": float(p_test.mean()),
            "test_actual_rate": float(actual_test),
        }
        log_to_mlflow(args, manifest, snap_id, model, iso, features,
                      metrics, thr, model.get_best_iteration(), yt, p_test)


if __name__ == "__main__":
    main()