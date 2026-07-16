-- ============================================================
-- Feature pipeline, final: assemble the flat training matrix
--
-- One row per QUALIFIED operation. Combines:
--   block 1  window features (velocity / deviation)   [innermost]
--   block 3  label + maturity gate                    [middle joins]
--   block 2  point-in-time merchant rate via ASOF     [outer join]
-- plus a handful of raw features.
--
-- identifiers (transaction_id, customer_id, authorized_at) are kept for
-- reference and for the TEMPORAL split later -- they are NOT model features.
--
-- Executed by pipeline/build_snapshot.py (not run by hand).
-- ============================================================

SELECT
    f.transaction_id,
    f.customer_id,
    f.authorized_at,
    -- ---- raw / simple features ----
    f.amount,
    toHour(f.authorized_at)                                          AS hour,
    f.channel,
    f.entry_mode,
    f.is_ecommerce,
    f.is_recurring,
    f.is_tokenized,
    f.country,
    if(isNotNull(f.ip_country) AND f.ip_country != f.country, 1, 0)  AS ip_mismatch,
    -- ---- window features (block 1) ----
    f.vel_1h,
    f.vel_24h,
    f.vel_7d,
    f.secs_since_prev,
    f.amount_z,
    f.decline_rate_20,
    -- ---- merchant static ----
    f.mcc,
    f.risk_segment,
    -- ---- point-in-time merchant fraud rate (block 2, ASOF) ----
    ms.fraud_rate_pit                                                AS merchant_fraud_rate_pit,
    -- ---- label (block 3) ----
    f.label
FROM (
    SELECT
        w.transaction_id AS transaction_id,
        w.customer_id    AS customer_id,
        w.authorized_at  AS authorized_at,
        w.merchant_id    AS merchant_id,
        w.amount         AS amount,
        w.channel        AS channel,
        w.entry_mode     AS entry_mode,
        w.is_ecommerce   AS is_ecommerce,
        w.is_recurring   AS is_recurring,
        w.is_tokenized   AS is_tokenized,
        w.country        AS country,
        w.ip_country     AS ip_country,
        w.c1h - 1  AS vel_1h,
        w.c24h - 1 AS vel_24h,
        w.c7d  - 1 AS vel_7d,
        if(w.prev_ts = toDateTime(0), NULL,
           dateDiff('second', w.prev_ts, w.authorized_at))            AS secs_since_prev,
        round((w.amount - w.prior_mean) / nullIf(w.prior_std, 0), 4)  AS amount_z,
        round(w.prior_decline, 4)                                     AS decline_rate_20,
        m.mcc                                                         AS mcc,
        m.risk_segment                                                AS risk_segment,
        if(fl.transaction_id != 0
           AND fl.reported_at <= (SELECT max(authorized_at) FROM cardops.transactions),
           1, 0)                                                      AS label
    FROM (
        SELECT
            transaction_id, customer_id, authorized_at, merchant_id, amount,
            channel, entry_mode, is_ecommerce, is_recurring, is_tokenized,
            country, ip_country, auth_result,
            count()            OVER w1h  AS c1h,
            count()            OVER w24h AS c24h,
            count()            OVER w7d  AS c7d,
            any(authorized_at) OVER wprev AS prev_ts,
            avg(amount)        OVER w20  AS prior_mean,
            stddevPop(amount)  OVER w20  AS prior_std,
            avg(auth_result = 'declined') OVER w20 AS prior_decline
        FROM cardops.transactions
        WINDOW
            w1h  AS (PARTITION BY customer_id ORDER BY toUInt32(authorized_at) RANGE BETWEEN 3600   PRECEDING AND CURRENT ROW),
            w24h AS (PARTITION BY customer_id ORDER BY toUInt32(authorized_at) RANGE BETWEEN 86400  PRECEDING AND CURRENT ROW),
            w7d  AS (PARTITION BY customer_id ORDER BY toUInt32(authorized_at) RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW),
            wprev AS (PARTITION BY customer_id ORDER BY authorized_at, transaction_id ROWS BETWEEN 1  PRECEDING AND 1 PRECEDING),
            w20   AS (PARTITION BY customer_id ORDER BY authorized_at, transaction_id ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
    ) w
    LEFT JOIN cardops.fraud_labels fl ON fl.transaction_id = w.transaction_id
    LEFT JOIN cardops.merchants     m ON m.merchant_id     = w.merchant_id
    WHERE w.authorized_at <= (SELECT max(authorized_at) - toIntervalDay(90) FROM cardops.transactions)
) f
ASOF LEFT JOIN cardops.merchant_weekly_stats ms
    ON f.merchant_id = ms.merchant_id
   AND f.authorized_at >= ms.as_of_date