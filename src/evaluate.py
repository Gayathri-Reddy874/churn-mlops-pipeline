import pandas as pd
import joblib
from sklearn.metrics import classification_report

DATA_PATH = "data/processed/train.csv"
MODEL_PATH = "models/model.pkl"

def evaluate_model():

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)

    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate_model()