from typing import Dict, Any
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel, Field, create_model

app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML model to predict customer churn",
    version="1.0"
)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "Saved_models" / "model_lr_2026-02-03.joblib"


bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
def validate_features(features: dict):
    if set(features.keys()) != set(FEATURES):
        raise ValueError("Invalid input schema")
BASE_DIR = Path(__file__).resolve().parent.parent
X_TEST_PATH = BASE_DIR / "data" / "X_test" / "X_test_2026-02-04.csv"
try:
    X_test = pd.read_csv(X_TEST_PATH)
    example_payload = X_test.iloc[0].to_dict()
except FileNotFoundError:
    # Fallback for Docker / production
    example_payload = {feature: 0.0 for feature in FEATURES}
def build_input_model(columns):
    fields = {}
    for col in columns:
        fields[col] = (
            float,
            Field(
                ...,
                description=f"Value for feature '{col}'",
                example=example_payload.get(col, 0.0)
            )
        )

    return create_model("ChurnInput", **fields)


ChurnInput = build_input_model(FEATURES)


def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]
    return {
        "churn": int(prediction),
        "churn_probability": float(proba)
    }

@app.post(
    "/predict",
    summary="Predict customer churn",
    description="Enter customer feature values to get churn prediction.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": example_payload
                }
            }
        }
    },
)
def predict_endpoint(features: ChurnInput):
    try:
        return predict(features.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))