import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

def train_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "processed")
    x_path = os.path.join(data_path, "X_train.csv")
    y_path = os.path.join(data_path, "y_train.csv")


    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()  # Para convertir a Series si es DataFrame

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    preds = clf.predict(X)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    mlflow.log_param("classifier", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(clf, "model")
    joblib.dump(clf, os.path.join(base_dir, "model.pkl"))
