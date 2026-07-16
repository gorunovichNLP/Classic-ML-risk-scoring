-- ============================================================
-- Feature pipeline, block 2: point-in-time merchant fraud rate
-- via weekly snapshots + ASOF JOIN (ClickHouse)
--
-- Idea: a merchant's fraud rate CHANGES over time. Taking "the rate as
-- of now" and joining it to a past operation leaks the future. Instead
-- we snapshot the rate at the end of every week, using ONLY what was
-- known by then (fraud counts iff reported_at <= week end), then ASOF
-- JOIN each operation to the last completed week before it.
--
-- Run:
--   docker compose exec -T clickhouse clickhouse-client \
--       --user cardops --password cardops_pwd --multiquery \
--       --format PrettyCompact < clickhouse/03_merchant_pit_asof.sql
-- ============================================================

-- ---------- build the weekly snapshots ------------------------------
DROP TABLE IF EXISTS cardops.merchant_weekly_stats;

CREATE TABLE cardops.merchant_weekly_stats
ENGINE = MergeTree
ORDER BY (merchant_id, as_of_date)
AS
WITH per_tx AS (
    SELECT
        t.merchant_id                             AS merchant_id,
        toMonday(t.authorized_at)                 AS aw,   -- Monday of the auth week
        -- Monday of the week the fraud became KNOWN (else far future = never known)
        if(fl.transaction_id != 0, toMonday(fl.reported_at), toDate('2099-01-01')) AS rw
    FROM cardops.transactions t
    LEFT JOIN cardops.fraud_labels fl ON fl.transaction_id = t.transaction_id
)
SELECT
    merchant_id,
    -- snapshot becomes available at the START of the following week (= end of week k)
    toDateTime(addWeeks(aw, k + 1))                       AS as_of_date,
    count()                                               AS n_tx,
    countIf(rw <= addWeeks(aw, k))                        AS n_fraud_known,
    -- Bayesian shrink toward the global prior (~0.4%) so low-volume
    -- merchants don't produce noisy 0% / 100% rates
    (countIf(rw <= addWeeks(aw, k)) + 500 * 0.004) / (count() + 500) AS fraud_rate_pit
FROM per_tx
-- each op is "recent" for 13 weeks (~90 days): it feeds snapshots k..k+12
ARRAY JOIN range(0, 13) AS k
GROUP BY merchant_id, as_of_date;


-- ---------- demo 1: one merchant's rate EVOLVING week by week --------
SELECT
    merchant_id,
    as_of_date,
    n_tx,
    n_fraud_known,
    round(fraud_rate_pit, 5) AS rate
FROM cardops.merchant_weekly_stats
WHERE merchant_id = (
    SELECT merchant_id FROM cardops.merchant_weekly_stats
    ORDER BY n_fraud_known DESC LIMIT 1
)
ORDER BY as_of_date
LIMIT 20;


-- ---------- demo 2: ASOF pulls the PREVIOUS week's rate -------------
-- Note rate_as_of is always strictly BEFORE authorized_at: the op never
-- sees its own week's (incomplete) snapshot. Early ops get NULL = cold
-- start (no history yet), which the model treats as missing.
SELECT
    t.authorized_at,
    t.merchant_id,
    ms.as_of_date                        AS rate_as_of,
    round(ms.fraud_rate_pit, 5)          AS merchant_fraud_rate_pit
FROM cardops.transactions t
ASOF LEFT JOIN cardops.merchant_weekly_stats ms
    ON t.merchant_id = ms.merchant_id
   AND t.authorized_at >= ms.as_of_date       -- ASOF: greatest as_of_date <= authorized_at
WHERE t.merchant_id = (
    SELECT merchant_id FROM cardops.merchant_weekly_stats
    ORDER BY n_fraud_known DESC LIMIT 1
)
ORDER BY t.authorized_at
LIMIT 20;