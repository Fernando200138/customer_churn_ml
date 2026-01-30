# In this file we are going to add a few training models.
# Our target variable is going to be churn. Right now it's a little difficult to define exactly what to predict
# I guess we can predict the probability a customer will churn.
# We will use logistic regression.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.data as data
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression


def basic_logistic_regression(path_file, plot_precison_recall=False):
    df = data.load_data(path_file)
    df = data.clean_basic_issues(df)
    # In this model we will do feature selection manually
    df_encoded = data.deeper_clean(df,OneHotEncoding=True)
    # Now we can split the data into training data and test data
    X = df_encoded.drop("Churn_Yes", axis=1)
    y = df_encoded["Churn_Yes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # We had to use a Standard Scaler because the model didn't converge using the normal feature
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced'))
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    y_pred = pipe.predict(X_test)
    y_scores = pipe.predict_proba(X_test)[:,1] # predict proba for the positive class
    precision,recall, thresholds = precision_recall_curve(y_test,y_scores)
    ap = average_precision_score(y_test,y_scores)
    print(f"The Accuracy is: {score}")
    print('Classification Report: ')
    print(classification_report(y_test,y_pred))
    print(f'ROC AUC score:{roc_auc_score(y_test,y_pred)}')
    if plot_precison_recall:
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Logistic Regression (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

'''
We have the option to use a balanced dataset or and imbalanced one. Out dataset is
imbalanced out of the box. When we use it as it is we get an accuracy of 0.78, when we 
balance it we get a lower accuracy (0.73). We need to understand what is the right 
bussiness metric to look for. What we want is to catch the most amount of churners, even at
the expense of flagging loyal customers. False Negative would be missing a churner while a
False positive would be flagging a loyal customer.
Recall minimized the amount of false negatives so that is the best metric to look for. 
In the imbalanced dataset set we get a recall of 0.89 for class 0 (no churn) and 0.51 for
class 1 (churn). This means we are classifying churners 51% percent of the time. 
Meanwhile in the balanced dataset we get a recall of 0.70 for class 0 and a 0.80 for class 1. 
This that we are catching must churners even though we are less accurate at classifying 
no-churners.
In conclusion, we will try to optimize for recall
'''

if __name__ == "__main__":
    path_file = r"D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\raw\raw.csv"
    print(basic_logistic_regression(path_file=path_file))
