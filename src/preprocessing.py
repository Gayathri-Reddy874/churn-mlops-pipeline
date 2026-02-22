import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROCESSED_PATH = "data/processed/train.csv"

def preprocess_data():

    df = pd.read_csv(PROCESSED_PATH)

    # Convert target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    df.to_csv(PROCESSED_PATH, index=False)
    print("Preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()