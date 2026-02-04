from typing import Dict, Any
import pandas as pd
import joblib
from fastapi import FastAPI

app = FastAPI()



bundle = joblib.load('D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\Saved_models\model_lr_2026-02-03.joblib')
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
