# Build context is the REPO ROOT (see docker-compose: context: .).
FROM python:3.11-slim

# libgomp1: CatBoost's runtime needs OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/

# artifacts (model + calibrator + schema) are MOUNTED at runtime, not baked in,
# so a new model version = restart, no image rebuild.
ENV ARTIFACT_DIR=/artifacts
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]