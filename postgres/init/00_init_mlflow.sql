-- Runs once, on first cluster initialization
-- (files in /docker-entrypoint-initdb.d are executed alphabetically).
--
-- We keep MLflow's metadata in its OWN role + database, separate from the
-- operational `cardops` DB. In a bank you never let the tracking server
-- share a service account with operational data — same principle here.

CREATE ROLE mlflow WITH LOGIN PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;