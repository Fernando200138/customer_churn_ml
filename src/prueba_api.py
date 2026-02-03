import pandas as pd
import requests

X_test = pd.read_csv("D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\X_test\X_test_2026-02-03.csv")

url = "http://127.0.0.1:8001/predict"


row = X_test.iloc[0].to_dict()

response = requests.post(url, json=row)

print(response.json())
