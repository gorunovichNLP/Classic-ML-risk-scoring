Поднять докер: docker compose up -d --build
Опустить докер: docker-compose down
Проверить, что с контейнером все ок:  docker compose ps
Проверка каждого сервиса по отдельности:
# Postgres: должны быть ОБЕ базы — cardops и mlflow
docker compose exec postgres psql -U cardops -d cardops -c "\l"

# ClickHouse: версия через HTTP-порт
curl http://localhost:8123/ping        # -> Ok.
docker compose exec clickhouse clickhouse-client --user cardops --password cardops_pwd -q "SELECT version()"

# MinIO: открой в браузере http://localhost:9001
#   логин minioadmin / minioadmin_pwd -> увидишь бакеты mlflow-artifacts и datasets

# MLflow: открой http://localhost:5000 -> должен подняться UI трекинга
curl http://localhost:5000/health      # -> OK

Логи: docker compose logs mlflow
docker compose logs clickhouse

накатить миграции: docker compose exec -T postgres psql -U cardops -d cardops < postgres/schema/02_geo_ip.sql
или: type postgres\schema\02_geo_ip.sql | docker compose exec -T postgres psql -U cardops -d cardops

type clickhouse/01_schema_etl.sql | docker compose exec -T clickhouse clickhouse-client --user cardops --password cardops_pwd --multiquery

type clickhouse/02_features_velocity_deviation.sql | docker compose exec -T clickhouse clickhouse-client --user cardops --password cardops_pwd --format PrettyCompact