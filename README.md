# Customer Churn Machine Learning Project
![Example image](Images/Customer_Churn.png)

## 1. Project Overview

Customer churn is a critical business challenge in subscription-based industries. 
Retaining customers is significantly more cost-effective than acquiring new ones.

This project builds, evaluates, and deploys a machine learning model to predict whether a 
customer will churn in the next billing cycle.

The project covers the complete ML lifecycle:

Data preprocessing and feature engineering

Model selection and evaluation

Imbalance handling

Hyperparameter tuning

REST API development with FastAPI

Containerization with Docker

Cloud deployment using Google Cloud Run
## 2. Dataset

We used the Telco Customer Churn dataset from Kaggle.

The dataset includes:

* Customer demographics

* Account information

* Service subscriptions

* Billing data

* Churn status

### Target Variable

Churn (binary classification):

* 1 → Customer churned

* 0 → Customer did not churn

The model outputs:

* A binary prediction

* The probability of churn
## 3. Data Preprocessing & Feature Engineering
### Handling Categorical Variables

Most features were categorical. We applied:

* OneHotEncoding to categorical variables

* Removal of CustomerID (non-informative identifier)

* Retained demographic variables such as gender

### Numerical Scaling

* Applied StandardScaler to numerical features to ensure proper scaling for Logistic Regression.

### Custom Feature

* Engineered a feature counting the number of subscribed services per customer.

### Class Imbalance Handling

The dataset is imbalanced (non-churners outnumber churners ~3:1).

To address this:

* Used class_weight="balanced" in Logistic Regression

* Used stratified 80/20 train-test split to preserve class distribution

This ensures the model pays appropriate attention to minority class (churners).
## 4. Modeling Approach

Models evaluated:

* Logistic Regression

* Random Forest

* XGBoost

After comparison, Logistic Regression was selected due to:

* Strong recall performance

* Stability

* Interpretability

* Lower risk of overfitting

Hyperparameter tuning was performed using:

* GridSearchCV

* RandomizedSearchCV

## 5. Model Evaluation
### Train/Test Split

* 80% training

* 20% testing

* Stratified split to preserve class distribution

### Why Not Accuracy?

Accuracy can be misleading with imbalanced data.

If 95% of customers do not churn, predicting all customers as non-churners
yields high accuracy but zero business value.
## 6. Performance Metrics
| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Non-Churn(1) | 0.91      | 0.73   | 0.81     |
| Churn (1)    | 0.51      | 0.80   | 0.62     |

### Confusion Matrix

Interpretation:

* 80% of churners are correctly identified

* 20% of churners are missed (false negatives)

* 27% of non-churners are incorrectly flagged as churners

Because recall was prioritized, the model accepts a moderate number 
of false positives in exchange for capturing most churners.

# 7. API Development

A REST API was developed using FastAPI.

The API:

* Loads the trained model

* Accepts customer feature inputs

Returns:

* prediction (0 or 1)

* churn_probability

## 8. Dockerization

The application was containerized using Docker to ensure:

* Reproducibility

* Dependency isolation

* Portability

* Cloud compatibility
## 10. Cloud Deployment

The Docker image was deployed using Google Cloud Run, enabling serverless hosting.

🔗 Live API Documentation:

https://customer-churn-476995771316.us-central1.run.app/docs

API Interface:![Example image](Images\Final_output1.jpg)



Prediction Output:![Example image](Images\Final_output2.jpg)


Users can modify input parameters and receive real-time churn predictions.
## 12. Limitations

* Dataset appears to represent a single billing-cycle snapshot

* No temporal or longitudinal behavior data

* No threshold optimization performed

* Business cost matrix not explicitly incorporated

