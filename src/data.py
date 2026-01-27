# This file will deal with raw data. It will have functions to load clean and so on.
import numpy as np
import pandas as pd


def load_data(path_file):
    # Path file must be read as a raw string
    try:
        df = pd.read_csv(path_file)  # reads file
        return df
    except Exception as e:
        print(f"An error eoccured:{e}")


def clean_basic_issues(df):
    df = df.dropna(axis=0)  # Drops rows with nan values
    df = df.drop_duplicates()  # drops duplicates
    return df


path_file = (
    r"D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\raw\raw.csv"
)

# print(load_data(path_file))
# df=load_data(path_file=path_file)

# print(clean_basic_issues(df))
