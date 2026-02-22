from src.drift import detect_drift
from src.train import train_model

def retrain_if_needed():

    drift, features = detect_drift(
        "data/processed/train.csv",
        "data/production/new_data.csv"
    )

    if drift:
        print("Drift detected in:", features)
        train_model()
        return True

    return False