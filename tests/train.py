# In this file we are going to add a few training models.
# Our target variable is going to be churn. Right now it's a little difficult to define exactly what to predict
# I guess we can predict the probability a customer will churn.
# We will use logistic regression.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.data as data
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression


def basic_logistic_regression(path_file):
    df = data.load_data(path_file)
    df = data.clean_basic_issues(df)
    # In this model we will do feature selection manually
    df = df.drop(
        "customerID", axis=1
    )  # we pretty much know this variable is not related to churn
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # let's first  transform the categorical data into dummy variblaes using one-hot encoding
    Column_names = df.columns.tolist()
    Column_names.remove("MonthlyCharges")
    Column_names.remove("TotalCharges")
    # print(Column_names)
    df_encoded = pd.get_dummies(df, columns=Column_names, drop_first=True, dtype=int)
    # Now we can split the the data into training data and test data
    X = df_encoded.drop("Churn_Yes", axis=1)
    y = df_encoded["Churn_Yes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # We had to use a Standard Scaler because the model didn't converge using the normal feature
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)

    print(f"The Accuracy is: {score}")


if __name__ == "__main__":
    path_file = r"D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\raw\raw.csv"
    print(basic_logistic_regression(path_file=path_file))
