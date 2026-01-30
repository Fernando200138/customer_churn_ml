# This file is used to test any important change in the project
import pandas as pd
import numpy as np
import sys

print("Python %s on %s" % (sys.version, sys.platform))
sys.path.extend(
    ["D:\\Documentos\\Vida_profesional\\Coding\\Projects\\customer_churn_ml"]
)
import src.data as data
from src.data import deeper_clean

#if __name__ == "__main__":
    #path_file = r"D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\raw\raw.csv"
    #df = data.load_data(path_file)
    # df = data.clean_basic_issues(df) # This tests clean_basic_issues
    #print(df.head())
def sample_df():
    return pd.DataFrame({
        "customerID": ["A", "B", "B"],
        "gender": ["Male", "Female", "Female"],
        "MonthlyCharges": [20.0, 30.0, 30.0],
        "TotalCharges": ["100", " ", "200"]
    })


def test_deeper_clean_drops_customer_id():
    df = sample_df()

    result = deeper_clean(df)

    assert "customerID" not in result.columns

