"""
Build an immutable training snapshot.

Flow:
  1. run the feature-matrix SQL against ClickHouse  -> pandas DataFrame
  2. compute a content hash + a "recipe" manifest (cutoffs, window, split)
  3. upload dataset.parquet + manifest.json to MinIO bucket `datasets`,
     under a snapshot id that never gets overwritten.

Training later reads the SNAPSHOT (not live ClickHouse), so the same
recipe always yields the same file -> reproducible experiments.

Run:
    pip install clickhouse-connect pandas pyarrow boto3
    python pipeline/build_snapshot.py
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import os

import pandas as pd

MATURITY_DAYS = 90


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sql", default="clickhouse/05_feature_matrix.sql")
    p.add_argument("--ch-host", default=os.environ.get("CH_HOST", "127.0.0.1"))
    p.add_argument("--ch-port", type=int, default=int(os.environ.get("CH_PORT", "8123")))
    p.add_argument("--ch-user", default=os.environ.get("CH_USER", "cardops"))
    p.add_argument("--ch-password", default=os.environ.get("CH_PASSWORD", "cardops_pwd"))
    p.add_argument("--ch-db", default=os.environ.get("CH_DB", "cardops"))
    p.add_argument("--s3-endpoint", default=os.environ.get("S3_ENDPOINT", "http://127.0.0.1:9000"))
    p.add_argument("--s3-key", default=os.environ.get("MINIO_ROOT_USER", "minioadmin"))
    p.add_argument("--s3-secret", default=os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin_pwd"))
    p.add_argument("--bucket", default="datasets")
    p.add_argument("--dataset-name", default="cardops_fraud")
    p.add_argument("--dry-run", action="store_true", help="build locally, skip MinIO upload")
    return p.parse_args()


def main():
    args = get_args()
    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host=args.ch_host, port=args.ch_port, username=args.ch_user,
        password=args.ch_password, database=args.ch_db)

    # snapshot boundaries (the "as of now" and the maturity cutoff)
    now_ts, tx_cutoff = client.query(
        f"SELECT max(authorized_at), max(authorized_at) - toIntervalDay({MATURITY_DAYS}) "
        "FROM cardops.transactions"
    ).result_rows[0]

    print(f"now_ts    = {now_ts}")
    print(f"tx_cutoff = {tx_cutoff}  (maturity window {MATURITY_DAYS}d)")
    print("running feature-matrix query ...")

    sql = open(args.sql, encoding="utf-8").read()
    df = client.query_df(sql)

    n = len(df)
    n_pos = int(df["label"].sum())
    fraud_rate = n_pos / n if n else 0.0
    print(f"rows       : {n:,}")
    print(f"positives  : {n_pos:,}  ({fraud_rate*100:.3f}%)")

    # ---- suggested TEMPORAL split (70 / 15 / 15 by time) -------------
    ts = pd.to_datetime(df["authorized_at"])
    train_end = ts.quantile(0.70)
    val_end = ts.quantile(0.85)
    print(f"split      : train <= {train_end} < val <= {val_end} < test")

    # ---- freeze parquet + hash --------------------------------------
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    snap_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + sha[:12]

    feature_cols = [c for c in df.columns
                    if c not in ("transaction_id", "customer_id", "authorized_at", "label")]
    manifest = {
        "snapshot_id": snap_id,
        "dataset_name": args.dataset_name,
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
        "source": f"clickhouse://{args.ch_db}",
        "sql_file": args.sql,
        "recipe": {
            "now_ts": str(now_ts),
            "transaction_cutoff": str(tx_cutoff),
            "maturity_window_days": MATURITY_DAYS,
            "merchant_rate_granularity": "weekly",
        },
        "rows": n,
        "positives": n_pos,
        "fraud_rate": round(fraud_rate, 6),
        "split": {"train_end": str(train_end), "val_end": str(val_end)},
        "feature_columns": feature_cols,
        "label_column": "label",
        "sha256": sha,
    }

    if args.dry_run:
        print("dry-run: manifest below, skipping upload.\n")
        print(json.dumps(manifest, indent=2))
        return

    # ---- upload to MinIO --------------------------------------------
    import boto3
    s3 = boto3.client(
        "s3", endpoint_url=args.s3_endpoint,
        aws_access_key_id=args.s3_key, aws_secret_access_key=args.s3_secret,
        region_name="us-east-1")
    prefix = f"{args.dataset_name}/{snap_id}"
    s3.put_object(Bucket=args.bucket, Key=f"{prefix}/dataset.parquet", Body=data)
    s3.put_object(Bucket=args.bucket, Key=f"{prefix}/manifest.json",
                  Body=json.dumps(manifest, indent=2).encode("utf-8"))
    print(f"\nuploaded s3://{args.bucket}/{prefix}/dataset.parquet")
    print(f"uploaded s3://{args.bucket}/{prefix}/manifest.json")
    print(f"snapshot_id = {snap_id}")


if __name__ == "__main__":
    main()