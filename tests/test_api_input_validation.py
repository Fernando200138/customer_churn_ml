import pandas as pd
import joblib
import pytest

bundle = joblib.load(
    r"D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\Saved_models\model_lr_2026-02-03.joblib"
)

model = bundle["model"]
MODEL_COLUMNS = bundle["features"]

X_test = pd.read_csv(
    r"D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\X_test\X_test_2026-02-03.csv"
)
#X_test = X_test.drop(columns=["Unnamed: 0"])


def validate_features(features: dict, columns: list):
    feature_keys = set(features.keys())
    expected_keys = set(columns)

    if feature_keys != expected_keys:
        missing = expected_keys - feature_keys
        extra = feature_keys - expected_keys
        raise ValueError(
            f"Invalid features. Missing: {missing}, Extra: {extra}"
        )


@pytest.fixture
def sample_valid_features():
    return X_test.iloc[0].to_dict()
#print(MODEL_COLUMNS)
#print()

#print(X_test.columns)

def test_valid_input_passes(sample_valid_features):
    validate_features(sample_valid_features, MODEL_COLUMNS)


def test_api_schema_matches_training():
    assert MODEL_COLUMNS == list(X_test.columns)
