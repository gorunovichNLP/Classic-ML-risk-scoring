-- ============================================================
-- CardOperationScoring — OLTP schema (PostgreSQL)
--
-- Operational source of truth. The ML dataset is assembled LATER
-- from these tables, so the golden rule for this schema is:
--   nothing here may bake in information that is only known
--   AFTER a transaction happens.
-- The single exception is `fraud_labels`, and there "the future"
-- is timestamped (reported_at) so the feature pipeline can respect
-- the observation window and avoid label leakage.
-- ============================================================

-- ---------- Reference: MCC dictionary -----------------------
-- Static dictionary. base_risk is an intrinsic category attribute
-- (e.g. gambling is riskier than groceries), NOT a rate derived
-- from our own fraud labels.
CREATE TABLE mcc_dictionary (
    mcc          CHAR(4)   PRIMARY KEY,
    description  TEXT      NOT NULL,
    category     TEXT      NOT NULL,
    base_risk    SMALLINT  NOT NULL DEFAULT 0   -- 0..3
);

-- ---------- Customers (STATIC profile only) -----------------
-- Deliberately no "typical spend" / "usual hours" columns:
-- those are DERIVED aggregates and must be computed point-in-time
-- in the feature pipeline, otherwise they leak the future.
CREATE TABLE customers (
    customer_id  BIGINT       PRIMARY KEY,
    home_country CHAR(2)      NOT NULL,          -- ISO-3166 alpha-2
    home_city    TEXT,
    segment      TEXT         NOT NULL,          -- 'mass','affluent','private'
    signup_at    TIMESTAMPTZ  NOT NULL
);

-- ---------- Cards (1 customer -> N cards) -------------------
-- Enables the "unique cards per device" fraud feature.
CREATE TABLE cards (
    card_id      BIGINT       PRIMARY KEY,
    customer_id  BIGINT       NOT NULL REFERENCES customers(customer_id),
    issued_at    TIMESTAMPTZ  NOT NULL
);

-- ---------- Devices -----------------------------------------
-- first_seen_at is an immutable past fact once observed, so
-- "days since first seen" is safe to use at auth time.
CREATE TABLE devices (
    device_id     BIGINT       PRIMARY KEY,
    device_type   TEXT         NOT NULL,         -- 'mobile','web','pos','atm'
    first_seen_at TIMESTAMPTZ  NOT NULL
);

-- ---------- Merchants ---------------------------------------
-- risk_segment is the network's own static tier. The merchant's
-- historical fraud rate is NOT stored here — it is computed
-- point-in-time in the feature pipeline.
CREATE TABLE merchants (
    merchant_id      BIGINT   PRIMARY KEY,
    mcc              CHAR(4)  NOT NULL REFERENCES mcc_dictionary(mcc),
    merchant_country CHAR(2)  NOT NULL,
    descriptor       TEXT     NOT NULL,
    risk_segment     TEXT     NOT NULL           -- 'low','medium','high'
);

-- ---------- Transactions (authorization fact table) ---------
-- Every column is available AT authorization time.
-- NOTE: there is intentionally NO is_fraud column here.
-- The label lives in fraud_labels and arrives with a delay.
CREATE TABLE transactions (
    transaction_id BIGINT        PRIMARY KEY,
    customer_id    BIGINT        NOT NULL REFERENCES customers(customer_id),
    card_id        BIGINT        NOT NULL REFERENCES cards(card_id),
    device_id      BIGINT        REFERENCES devices(device_id),  -- nullable (some channels have none)
    merchant_id    BIGINT        NOT NULL REFERENCES merchants(merchant_id),
    amount         NUMERIC(14,2) NOT NULL,
    currency       CHAR(3)       NOT NULL DEFAULT 'RUB',
    country        CHAR(2)       NOT NULL,        -- where the operation happened
    city           TEXT,
    channel        TEXT          NOT NULL,        -- 'mobile','web','pos','atm','ecom'
    entry_mode     TEXT          NOT NULL,        -- 'card_present','cnp'
    is_ecommerce   BOOLEAN       NOT NULL DEFAULT FALSE,
    is_recurring   BOOLEAN       NOT NULL DEFAULT FALSE,
    is_tokenized   BOOLEAN       NOT NULL DEFAULT FALSE,
    auth_result    TEXT          NOT NULL,        -- 'approved','declined'
    response_code  TEXT,                          -- decline reason if any
    authorized_at  TIMESTAMPTZ   NOT NULL
);

-- Indexes chosen to match the feature-pipeline access patterns:
--   per-customer time-ordered scans -> velocity & deviation features
--   per-device   time-ordered scans -> device/cards features
--   plain time index                -> temporal split cutoffs
CREATE INDEX ix_tx_customer_time ON transactions (customer_id, authorized_at);
CREATE INDEX ix_tx_device_time   ON transactions (device_id, authorized_at);
CREATE INDEX ix_tx_time          ON transactions (authorized_at);

-- ---------- Fraud labels (the ONLY future-aware table) ------
-- A row exists ONLY for transactions confirmed fraud/disputed.
-- Absence of a row = (potentially) legitimate.
-- reported_at = authorized_at + delay (the label maturity delay).
-- This is what powers:
--   * the observation / maturity window
--   * label_cutoff_date strictly < transaction_cutoff_date
--   * "negative = no fraud within window" as a temporal anti-join
CREATE TABLE fraud_labels (
    label_id       BIGINT       PRIMARY KEY,
    transaction_id BIGINT       NOT NULL UNIQUE REFERENCES transactions(transaction_id),
    label_type     TEXT         NOT NULL,         -- 'confirmed_fraud','chargeback','dispute'
    source         TEXT         NOT NULL,         -- 'manual_investigation','chargeback','customer_report'
    reported_at    TIMESTAMPTZ  NOT NULL
);

CREATE INDEX ix_fraud_reported ON fraud_labels (reported_at);