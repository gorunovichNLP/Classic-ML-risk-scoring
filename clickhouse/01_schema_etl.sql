-- ============================================================
-- ClickHouse: analytical copy + ETL from Postgres.
--
-- ClickHouse PULLS from Postgres via the built-in postgresql() table
-- function. This is the "analytical copy via ETL" arrow from the legend.
--
-- IMPORTANT — networking: this runs container-to-container inside the
-- docker network, so Postgres is reached as  postgres:5432  (its INTERNAL
-- port). The host remap 55432 does NOT apply here — that is host-only.
--
-- Run:
--   docker compose exec -T clickhouse clickhouse-client \
--       --user cardops --password cardops_pwd --multiquery \
--       < clickhouse/01_schema_etl.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS cardops;

-- ---------- dimensions ------------------------------------------------
CREATE TABLE IF NOT EXISTS cardops.mcc_dictionary
(
    mcc         String,
    description String,
    category    LowCardinality(String),
    base_risk   UInt8
) ENGINE = MergeTree ORDER BY mcc;

CREATE TABLE IF NOT EXISTS cardops.customers
(
    customer_id  UInt64,
    home_country LowCardinality(String),
    home_city    Nullable(String),
    segment      LowCardinality(String),
    signup_at    DateTime,
    home_lat     Nullable(Decimal(9,6)),
    home_lon     Nullable(Decimal(9,6))
) ENGINE = MergeTree ORDER BY customer_id;

CREATE TABLE IF NOT EXISTS cardops.cards
(
    card_id     UInt64,
    customer_id UInt64,
    issued_at   DateTime
) ENGINE = MergeTree ORDER BY card_id;

CREATE TABLE IF NOT EXISTS cardops.devices
(
    device_id     UInt64,
    device_type   LowCardinality(String),
    first_seen_at DateTime
) ENGINE = MergeTree ORDER BY device_id;

CREATE TABLE IF NOT EXISTS cardops.merchants
(
    merchant_id      UInt64,
    mcc              String,
    merchant_country LowCardinality(String),
    descriptor       String,
    risk_segment     LowCardinality(String)
) ENGINE = MergeTree ORDER BY merchant_id;

-- ---------- facts -----------------------------------------------------
-- ORDER BY (customer_id, authorized_at): physical ordering that makes
-- per-customer time-window scans (velocity / deviation features) cheap.
CREATE TABLE IF NOT EXISTS cardops.transactions
(
    transaction_id UInt64,
    customer_id    UInt64,
    card_id        UInt64,
    device_id      Nullable(UInt64),
    merchant_id    UInt64,
    amount         Decimal(14,2),
    currency       LowCardinality(String),
    country        LowCardinality(String),
    city           Nullable(String),
    channel        LowCardinality(String),
    entry_mode     LowCardinality(String),
    is_ecommerce   UInt8,
    is_recurring   UInt8,
    is_tokenized   UInt8,
    auth_result    LowCardinality(String),
    response_code  Nullable(String),
    authorized_at  DateTime,
    lat            Nullable(Decimal(9,6)),
    lon            Nullable(Decimal(9,6)),
    ip_country     Nullable(String)
) ENGINE = MergeTree ORDER BY (customer_id, authorized_at);

CREATE TABLE IF NOT EXISTS cardops.fraud_labels
(
    label_id       UInt64,
    transaction_id UInt64,
    label_type     LowCardinality(String),
    source         LowCardinality(String),
    reported_at    DateTime
) ENGINE = MergeTree ORDER BY transaction_id;

-- ============================================================
-- ETL: pull each table from Postgres.
-- Idempotent: TRUNCATE then re-load, so this file can be re-run.
-- Column order in each SELECT matches the CH table above.
--
-- If a timestamp ever refuses to cast, wrap it, e.g.:
--   toDateTime(authorized_at)
-- ============================================================

TRUNCATE TABLE cardops.mcc_dictionary;
INSERT INTO cardops.mcc_dictionary
SELECT * FROM postgresql('postgres:5432', 'cardops', 'mcc_dictionary', 'cardops', 'cardops_pwd');

TRUNCATE TABLE cardops.customers;
INSERT INTO cardops.customers
SELECT * FROM postgresql('postgres:5432', 'cardops', 'customers', 'cardops', 'cardops_pwd');

TRUNCATE TABLE cardops.cards;
INSERT INTO cardops.cards
SELECT * FROM postgresql('postgres:5432', 'cardops', 'cards', 'cardops', 'cardops_pwd');

TRUNCATE TABLE cardops.devices;
INSERT INTO cardops.devices
SELECT * FROM postgresql('postgres:5432', 'cardops', 'devices', 'cardops', 'cardops_pwd');

TRUNCATE TABLE cardops.merchants;
INSERT INTO cardops.merchants
SELECT * FROM postgresql('postgres:5432', 'cardops', 'merchants', 'cardops', 'cardops_pwd');

TRUNCATE TABLE cardops.transactions;
INSERT INTO cardops.transactions
SELECT * FROM postgresql('postgres:5432', 'cardops', 'transactions', 'cardops', 'cardops_pwd');

TRUNCATE TABLE cardops.fraud_labels;
INSERT INTO cardops.fraud_labels
SELECT * FROM postgresql('postgres:5432', 'cardops', 'fraud_labels', 'cardops', 'cardops_pwd');