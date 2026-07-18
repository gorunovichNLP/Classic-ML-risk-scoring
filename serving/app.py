"""
FastAPI inference service for the CatBoost fraud model.

Loads the SERVING CONTRACT from disk -- model + isotonic calibrator +
threshold + feature schema -- and scores one operation at a time.

The request is validated against the feature schema via Pydantic. That schema
IS the contract between training and serving: it protects against train/serve
skew (our "no feature store" mitigation). The response is a CALIBRATED
probability (real risk) plus a binary decision at the chosen threshold.

Run:
    pip install fastapi uvicorn catboost joblib pandas
    uvicorn serving.app:app --host 0.0.0.0 --port 8080
    # docs at http://localhost:8080/docs
"""

import json
import os
from typing import Optional

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from pydantic import create_model

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "training/artifacts")

# ---- load the contract ------------------------------------------------
with open(os.path.join(ARTIFACT_DIR, "feature_schema.json"), encoding="utf-8") as f:
    SCHEMA = json.load(f)
FEATURES = SCHEMA["features"]
CAT = set(SCHEMA["cat_features"])
THRESHOLD = SCHEMA["threshold"]

model = CatBoostClassifier()
model.load_model(os.path.join(ARTIFACT_DIR, "model.cbm"))
calibrator = joblib.load(os.path.join(ARTIFACT_DIR, "calibrator.joblib"))

# ---- build the request model from the schema (the contract) ----------
# categoricals -> required str; numerics -> optional float (None = cold-start NaN)
_fields = {}
for feat in FEATURES:
    _fields[feat] = (str, ...) if feat in CAT else (Optional[float], None)
Operation = create_model("Operation", **_fields)

app = FastAPI(title="CardOperationScoring — fraud scoring")


def _dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES), "threshold": THRESHOLD}


@app.post("/score")
def score(op: Operation):
    row = pd.DataFrame([_dump(op)])[FEATURES]           # enforce training column order
    for c in FEATURES:                                   # mirror training coercion
        if c in CAT:
            row[c] = row[c].astype(str)
        else:
            row[c] = pd.to_numeric(row[c], errors="coerce").astype("float64")

    raw = float(model.predict_proba(row)[0, 1])          # inflated by undersampling
    prob = float(calibrator.predict([raw])[0])           # calibrated -> real risk
    return {
        "fraud_probability": round(prob, 6),
        "flag": bool(prob >= THRESHOLD),
        "threshold": THRESHOLD,
        "raw_score": round(raw, 6),
    }