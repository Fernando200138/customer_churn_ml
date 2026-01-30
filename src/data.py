# This file will deal with raw data. It will have functions to load clean and so on.
import numpy as np
import pandas as pd


def load_data(path_file):
    # Path file must be read as a raw string
    try:
        df = pd.read_csv(path_file)  # reads file
        return df
    except Exception as e:
        print(f"An error eccured:{e}")


def clean_basic_issues(df):
    df = df.dropna(axis=0)  # Drops rows with nan values
    df = df.drop_duplicates()  # drops duplicates
    return df
def deeper_clean(df, OneHotEncoding = False):
    df = df.dropna(axis=0)  # Drops rows with nan values
    df = df.drop_duplicates()  # drops duplicates
    df = df.drop(
        "customerID", axis=1
    )  # we pretty much know this variable is not related to churn
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    if OneHotEncoding == True:
        Column_names = df.columns.tolist()
        Column_names.remove("MonthlyCharges")
        Column_names.remove("TotalCharges")
        # print(Column_names)
        df_encoded = pd.get_dummies(df, columns=Column_names, drop_first=True, dtype=int)
        return df_encoded
    else:
        return df
"""
This function will create certain features for our dataset.
First we want to create a count of subscribed services
The Services we have are MultipleLines, PhoneService, OnlySecurity, OnlineBackup,
Deviceprotection,TechSupport,StreamingTV, StreamingMovies,PaperlessBilling. So for
our first feature we will count these services.
"""
def feature_engineering(df):
    services = ['MultipleLines','PhoneService','OnlineBackup',
                'Deviceprotection','TechSupport','StreamingTV',
                'StreamingMovies','PaperlessBilling']
    df['CountServices']= (df[services]=='Yes').sum(axis=1)

    return df





path_file = (
    r"D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\raw\raw.csv"
)

# print(load_data(path_file))
# df=load_data(path_file=path_file)

# print(clean_basic_issues(df))
