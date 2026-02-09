from typing import Dict, Any
import pandas as pd
import joblib
from fastapi import FastAPI
from pathlib import Path

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "Saved_models" / "model_lr_2026-02-03.joblib"


bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])

    missing = set(FEATURES) - set(df.columns)
    extra = set(df.columns) - set(FEATURES)

    if missing:
        return {"error": f"Missing features: {missing}"}

    if extra:
        df = df[FEATURES]  # drop extras silently

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]

    return {
        "churn": int(prediction),
        "churn_probability": float(proba)
    }
