-- ============================================================
-- Feature pipeline, block 3: label assembly (ClickHouse)
--
-- Turns raw fraud_labels into a clean 0/1 target, respecting time:
--   NOW        = the snapshot moment = latest transaction we've seen
--   tx_cutoff  = NOW - 90 days (the maturity window)
--
--   label = 1  : a fraud record exists AND was reported by NOW
--   label = 0  : NO fraud record  (the anti-join: nothing to attach)
--   dropped    : authorized_at > tx_cutoff  -> too young, truth not in yet
--
-- The single maturity gate (authorized_at <= tx_cutoff) does two jobs at
-- once: it removes the immature tail, and it guarantees every "0" has
-- had a full 90-day window to prove otherwise.
--
-- Run (PowerShell):
--   Get-Content -Raw clickhouse/04_labels_assembly.sql | docker compose exec -T clickhouse `
--       clickhouse-client --user cardops --password cardops_pwd --multiquery --format PrettyCompact
-- ============================================================

-- ---------- 1. the snapshot boundaries ------------------------------
SELECT
    max(authorized_at)                        AS now_ts,
    max(authorized_at) - toIntervalDay(90)    AS tx_cutoff
FROM cardops.transactions;


-- ---------- 2. how the data gets carved -----------------------------
WITH bounds AS (
    SELECT max(authorized_at) - toIntervalDay(90) AS tx_cutoff
    FROM cardops.transactions
)
SELECT 'total transactions'                 AS bucket, count() AS n
FROM cardops.transactions
UNION ALL
SELECT 'dropped: immature tail (<90d old)', count()
FROM cardops.transactions t CROSS JOIN bounds b
WHERE t.authorized_at > b.tx_cutoff
UNION ALL
SELECT 'qualified for dataset',             count()
FROM cardops.transactions t CROSS JOIN bounds b
WHERE t.authorized_at <= b.tx_cutoff;


-- ---------- 3. label distribution among qualified rows --------------
-- This is the reusable label logic the final matrix will carry.
WITH bounds AS (
    SELECT max(authorized_at)                     AS now_ts,
           max(authorized_at) - toIntervalDay(90) AS tx_cutoff
    FROM cardops.transactions
)
SELECT
    label,
    count()                                              AS n,
    round(100.0 * count() / sum(count()) OVER (), 3)     AS pct
FROM (
    SELECT
        -- anti-join: LEFT JOIN, and "no matched fraud row" => label 0
        if(fl.transaction_id != 0 AND fl.reported_at <= b.now_ts, 1, 0) AS label
    FROM cardops.transactions t
    CROSS JOIN bounds b
    LEFT JOIN cardops.fraud_labels fl ON fl.transaction_id = t.transaction_id
    WHERE t.authorized_at <= b.tx_cutoff                 -- maturity gate
)
GROUP BY label
ORDER BY label;