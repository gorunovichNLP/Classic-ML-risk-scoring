"""
Model benchmark: CatBoost (the chosen production model) vs baselines
(LightGBM, Logistic Regression, linear SVM).

Point: JUSTIFY the CatBoost choice with numbers, not assert it.
  * GBMs (CatBoost, LightGBM) take categoricals + NaN natively.
  * Linear models (LogReg, SVM) get a one-hot + impute + scale pipeline.
  * SVM here is LINEAR (LinearSVC); a kernel SVM is O(n^2+) and does not
    finish at this scale -- itself a finding worth stating.

Metrics:
  * roc_auc / pr_auc  -- ranking quality (scale-invariant, so undersampling's
    probability inflation doesn't affect them).
  * prec@recN         -- THE business metric: precision at a target recall.
    Recall is the rules engine's job; the model exists to cut FALSE POSITIVES
    at that recall. Higher precision here = fewer false alarms.

Selection is on VAL. TEST stays reserved for the final chosen-model number.

Run:
    pip install catboost lightgbm scikit-learn boto3 pandas pyarrow
    python training/benchmark.py --target-recall 0.80
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from train import load_snapshot, temporal_split, undersample

CAT = ["channel", "entry_mode", "country", "mcc", "risk_segment"]


def coerce_types(df, features):
    """ClickHouse Decimal -> Python Decimal -> pandas 'object', which LightGBM
    rejects. Force numerics to float and categoricals to str."""
    df = df.copy()
    for c in features:
        if c in CAT:
            df[c] = df[c].astype(str)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def precision_at_recall(y, s, target):
    """Best precision achievable while keeping recall >= target."""
    prec, rec, _ = precision_recall_curve(y, s)
    mask = rec >= target
    return float(prec[mask].max()) if mask.any() else 0.0


def linear_pipeline(estimator, num_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])


def run_benchmark(train_bal, val, features, seed=42, target_recall=0.80):
    train_bal = coerce_types(train_bal, features)
    val = coerce_types(val, features)
    num_cols = [c for c in features if c not in CAT]
    ytr = train_bal["label"].values
    yva = val["label"].values
    rows = []

    def record(name, s, dt):
        rows.append((name, roc_auc_score(yva, s), average_precision_score(yva, s),
                     precision_at_recall(yva, s, target_recall), dt))

    # ---- CatBoost (native cats + NaN) -------------------------------
    from catboost import CatBoostClassifier, Pool
    t0 = time.time()
    cat = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                             eval_metric="AUC", random_seed=seed, verbose=0)
    cat.fit(Pool(train_bal[features], ytr, cat_features=CAT),
            eval_set=Pool(val[features], yva, cat_features=CAT),
            early_stopping_rounds=50)
    record("CatBoost", cat.predict_proba(val[features])[:, 1], time.time() - t0)

    # ---- LightGBM (native cats + NaN) -------------------------------
    import lightgbm as lgb
    Xtr = train_bal[features].copy()
    Xva = val[features].copy()
    for c in CAT:
        Xtr[c] = Xtr[c].astype("category")
        Xva[c] = Xva[c].astype("category")
    t0 = time.time()
    lg = lgb.LGBMClassifier(n_estimators=800, num_leaves=31, learning_rate=0.05,
                            random_state=seed, verbose=-1)
    lg.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    record("LightGBM", lg.predict_proba(Xva)[:, 1], time.time() - t0)

    # ---- Logistic Regression (one-hot + impute + scale) -------------
    t0 = time.time()
    lr = linear_pipeline(LogisticRegression(max_iter=2000), num_cols)
    lr.fit(train_bal[features], ytr)
    record("LogReg", lr.predict_proba(val[features])[:, 1], time.time() - t0)

    # ---- Linear SVM (one-hot + impute + scale; decision_function) ---
    t0 = time.time()
    sv = linear_pipeline(LinearSVC(C=1.0), num_cols)
    sv.fit(train_bal[features], ytr)
    record("LinearSVC", sv.decision_function(val[features]), time.time() - t0)

    col = f"prec@rec{int(target_recall*100)}"
    df = pd.DataFrame(rows, columns=["model", "roc_auc", "pr_auc", col, "fit_s"])
    df = df.sort_values(col, ascending=False).reset_index(drop=True)
    for c in ("roc_auc", "pr_auc", col):
        df[c] = df[c].round(4)
    df["fit_s"] = df["fit_s"].round(1)
    return df, col


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", default=None)
    p.add_argument("--neg-per-pos", type=int, default=20)
    p.add_argument("--target-recall", type=float, default=0.80,
                   help="the recall the rules engine guarantees; we cut FPs at it")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s3-endpoint", default="http://127.0.0.1:9000")
    p.add_argument("--s3-key", default="minioadmin")
    p.add_argument("--s3-secret", default="minioadmin_pwd")
    p.add_argument("--bucket", default="datasets")
    p.add_argument("--dataset-name", default="cardops_fraud")
    return p.parse_args()


def main():
    args = get_args()
    df, manifest, snap_id = load_snapshot(args)
    features = manifest["feature_columns"]
    train, val, test = temporal_split(df, manifest)
    train_bal, _, rate_after = undersample(train, args.neg_per_pos, args.seed)
    print(f"snapshot {snap_id} | train_bal={len(train_bal):,} (fraud {rate_after*100:.2f}%) "
          f"| val={len(val):,}")
    print(f"benchmarking on val | target recall = {args.target_recall}\n")

    table, col = run_benchmark(train_bal, val, features, args.seed, args.target_recall)
    print(table.to_string(index=False))
    winner = table.iloc[0]
    print(f"\nwinner by {col}: {winner['model']}  "
          f"(precision {winner[col]:.3f} at recall {args.target_recall})")
    print(f"-> ~{1/winner[col]:.1f} alerts per real fraud at that recall "
          f"(lower = fewer false positives).")


if __name__ == "__main__":
    main()