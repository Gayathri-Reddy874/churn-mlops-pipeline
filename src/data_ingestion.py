import pandas as pd
import os

RAW_PATH = "data/raw/telco.csv"
PROCESSED_PATH = "data/processed/train.csv"

def ingest_data():

    df = pd.read_csv(RAW_PATH)

    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Fix TotalCharges column
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Data ingestion completed.")

if __name__ == "__main__":
    ingest_data()