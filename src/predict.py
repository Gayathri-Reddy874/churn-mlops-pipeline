import pandas as pd
import joblib

def predict(input_dict):

    model = joblib.load("models/model.pkl")
    feature_names = joblib.load("models/features.pkl")

    df = pd.DataFrame([input_dict])

    # One-hot encode input
    df = pd.get_dummies(df)

    # Align columns with training features
    df = df.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(df)[0]

    return prediction