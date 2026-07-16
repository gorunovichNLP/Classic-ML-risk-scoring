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
    p.add_argument("--seed", type=int, default=42)
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
    print("     Its raw probabilities will be inflated -> isotonic calibration (next) fixes that.")


if __name__ == "__main__":
    main()