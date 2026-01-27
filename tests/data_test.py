# This file is used to test any important change in the project
import pandas as pd
import numpy as np
import sys

print("Python %s on %s" % (sys.version, sys.platform))
sys.path.extend(
    ["D:\\Documentos\\Vida_profesional\\Coding\\Projects\\customer_churn_ml"]
)
import src.data as data

if __name__ == "__main__":
    path_file = r"D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\raw\raw.csv"
    df = data.load_data(path_file)
    df = data.clean_basic_issues(df)
    print(df.head())
