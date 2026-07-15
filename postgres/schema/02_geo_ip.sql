-- ============================================================
-- Migration 02: raw geo + IP signals that CANNOT be derived
-- from the existing columns.
--   * customers.home_lat/home_lon  -> distance-from-home features
--   * transactions.lat/lon         -> impossible-travel (speed between ops)
--   * transactions.ip_country      -> CNP mismatch (card country vs IP country)
-- All additive; safe on empty tables.
-- ============================================================

ALTER TABLE customers
    ADD COLUMN home_lat NUMERIC(9,6),
    ADD COLUMN home_lon NUMERIC(9,6);

ALTER TABLE transactions
    ADD COLUMN lat        NUMERIC(9,6),
    ADD COLUMN lon        NUMERIC(9,6),
    ADD COLUMN ip_country CHAR(2);   -- nullable: only meaningful for CNP/ecom