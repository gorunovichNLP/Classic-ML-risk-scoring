"""
SHAP explainability for the CatBoost fraud model -- for business & governance.

Answers "why does the model flag an operation?" via per-feature contributions.
Global importance (mean |SHAP|) shows what drives predictions overall; we check
it lines up with REAL fraud signals (velocity, geo, merchant rate, amount
deviation) and not artifacts.

Uses CatBoost's native ShapValues -- robust with categorical features (the
shap library's beeswarm needs numeric encoding for string cats).

Run:
    pip install catboost matplotlib boto3 pandas pyarrow
    python training/shap_analysis.py
"""

import argparse
import os

import numpy as np
import pandas as pd

from train import (CAT, coerce_types, load_snapshot, temporal_split,
                   train_and_calibrate, undersample)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", default=None)
    p.add_argument("--neg-per-pos", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=int, default=5000, help="rows to explain")
    p.add_argument("--out", default="shap_importance.png")
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
    train_bal, _, _ = undersample(train, args.neg_per_pos, args.seed)
    model, iso, _, _ = train_and_calibrate(train_bal, val, features, args.seed)

    from catboost import Pool
    sample = coerce_types(test, features)
    if len(sample) > args.sample:
        sample = sample.sample(args.sample, random_state=args.seed)

    # native SHAP: shape (n, n_features + 1); last column is the base value
    shap = model.get_feature_importance(Pool(sample[features], cat_features=CAT),
                                        type="ShapValues")
    contribs = shap[:, :-1]
    mean_abs = np.abs(contribs).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    print("global feature importance (mean |SHAP|, higher = more influence):\n")
    for i in order:
        f = features[i]
        direction = ""
        if f not in CAT:                       # direction only makes sense for numerics
            fv = pd.to_numeric(sample[f], errors="coerce").values
            m = ~np.isnan(fv)
            if m.sum() > 10 and np.std(fv[m]) > 0:
                r = np.corrcoef(fv[m], contribs[m, i])[0, 1]
                direction = ("↑ higher -> more fraud" if r > 0.05 else
                             "↓ higher -> less fraud" if r < -0.05 else "~ mixed")
        print(f"  {mean_abs[i]:8.4f}  {f:26s} {direction}")

    # horizontal bar chart for slides
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = order[:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([features[i] for i in top], mean_abs[top], color="#1D9E75")
    ax.set_xlabel("mean |SHAP|  (impact on the fraud score)")
    ax.set_title("What drives the fraud model")
    fig.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\nsaved chart -> {args.out}")
    print("sanity check: the top features should be real fraud signals")
    print("(velocity, merchant_fraud_rate_pit, amount_z, ip_mismatch, geo) -- not IDs or noise.")


if __name__ == "__main__":
    main()