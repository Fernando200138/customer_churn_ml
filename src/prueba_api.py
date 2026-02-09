import pandas as pd
import requests
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
X_TEST_PATH = BASE_DIR / "data" / "X_test" / "X_test_2026-02-03.csv"
X_test = pd.read_csv(X_TEST_PATH)
url = "http://127.0.0.1:8000/predict"


row = X_test.iloc[0].to_dict()

response = requests.post(url, json=row)

print(response.status_code)
print(response.json())