-- ============================================================
-- Feature pipeline, block 1: velocity + deviation (ClickHouse)
--
-- DEMO version: computes the features for the single customer with the
-- biggest fraud episode and prints their timeline, so you can eyeball
-- that every window looks STRICTLY BACKWARD (current row never leaks
-- into its own feature).
--
-- The window logic here is exactly what we'll materialise for ALL
-- customers in the next step.
--
-- Run:
--   docker compose exec -T clickhouse clickhouse-client \
--       --user cardops --password cardops_pwd \
--       --format PrettyCompact < clickhouse/02_features_velocity_deviation.sql
-- ============================================================

WITH
-- pick the customer with the most fraud-labelled operations
target AS (
    SELECT t.customer_id AS cid
    FROM cardops.fraud_labels fl
    INNER JOIN cardops.transactions t USING (transaction_id)
    GROUP BY t.customer_id
    ORDER BY count() DESC
    LIMIT 1
),
-- raw window columns (one pass over that customer's ops)
feat AS (
    SELECT
        transaction_id,
        customer_id,
        authorized_at,
        amount,
        country,
        auth_result,
        -- velocity: how many ops in the trailing window (INCLUDING self for now)
        count()      OVER w1h  AS c1h,
        count()      OVER w24h AS c24h,
        count()      OVER w7d  AS c7d,
        -- previous op timestamp (single row before -> frame excludes current)
        any(authorized_at) OVER wprev AS prev_ts,
        -- trailing stats over the last 20 ops BEFORE this one
        avg(amount)              OVER w20 AS prior_mean,
        stddevPop(amount)        OVER w20 AS prior_std,
        avg(auth_result = 'declined') OVER w20 AS prior_decline
    FROM cardops.transactions
    WHERE customer_id = (SELECT cid FROM target)
    WINDOW
        -- RANGE frames are time-based: offset is in SECONDS of the ORDER BY value.
        -- toUInt32(DateTime) = unix seconds -> a numeric column RANGE can range over.
        w1h  AS (PARTITION BY customer_id ORDER BY toUInt32(authorized_at) RANGE BETWEEN 3600   PRECEDING AND CURRENT ROW),
        w24h AS (PARTITION BY customer_id ORDER BY toUInt32(authorized_at) RANGE BETWEEN 86400  PRECEDING AND CURRENT ROW),
        w7d  AS (PARTITION BY customer_id ORDER BY toUInt32(authorized_at) RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW),
        -- ROWS frames are position-based: "1 PRECEDING" cleanly excludes current row.
        wprev AS (PARTITION BY customer_id ORDER BY authorized_at, transaction_id ROWS BETWEEN 1  PRECEDING AND 1 PRECEDING),
        w20   AS (PARTITION BY customer_id ORDER BY authorized_at, transaction_id ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
)
SELECT
    f.authorized_at,
    f.amount,
    f.country,
    f.auth_result                                                        AS result,
    f.c1h  - 1                                                           AS vel_1h,   -- minus self
    f.c24h - 1                                                           AS vel_24h,
    f.c7d  - 1                                                           AS vel_7d,
    if(f.prev_ts = toDateTime(0), NULL,
       dateDiff('second', f.prev_ts, f.authorized_at))                   AS secs_since_prev,
    round((f.amount - f.prior_mean) / nullIf(f.prior_std, 0), 2)         AS amount_z,
    round(f.prior_decline, 3)                                            AS decline_rate_20,
    if(fl.transaction_id != 0, 1, 0)                                     AS is_fraud   -- for eyeballing only
FROM feat f
LEFT JOIN cardops.fraud_labels fl ON fl.transaction_id = f.transaction_id
ORDER BY f.authorized_at;