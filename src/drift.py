import pandas as pd
import numpy as np

def detect_drift(reference_path, current_path, threshold=0.2):

    ref = pd.read_csv(reference_path)
    curr = pd.read_csv(current_path)

    numeric_cols = ref.select_dtypes(include=np.number).columns

    drift_detected = False
    drift_features = []

    for col in numeric_cols:
        ref_mean = ref[col].mean()
        curr_mean = curr[col].mean()

        if ref_mean == 0:
            continue

        diff = abs(ref_mean - curr_mean) / abs(ref_mean)

        if diff > threshold:
            drift_detected = True
            drift_features.append(col)

    return drift_detected, drift_features