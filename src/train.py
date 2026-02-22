import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_PATH = "data/processed/train.csv"
MODEL_PATH = "models/model.pkl"
FEATURE_PATH = "models/features.pkl"


def train_model():

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # ----------------------------
    # Load Data
    # ----------------------------
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # ----------------------------
    # Train Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # MLflow Setup
    # ----------------------------
    mlflow.set_experiment("telco_churn_experiment")

    with mlflow.start_run():

        # ----------------------------
        # Model
        # ----------------------------
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )

        model.fit(X_train, y_train)

        # ----------------------------
        # Evaluation
        # ----------------------------
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, name="churn_model")

        print(f"Model trained successfully.")
        print(f"Accuracy: {accuracy}")

    # ----------------------------
    # Save Model Locally
    # ----------------------------
    joblib.dump(model, MODEL_PATH)

    # VERY IMPORTANT: Save feature names
    joblib.dump(X.columns.tolist(), FEATURE_PATH)

    print("Model and feature schema saved successfully.")


if __name__ == "__main__":
    train_model()