import streamlit as st
import pandas as pd
from src.predict import predict
from src.retrain import retrain_if_needed

st.title("📊 Telco Churn Prediction (End-to-End ML Pipeline)")

st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure", 0, 72, 12)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)

if st.button("Predict"):

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # NOTE: For demo, ensure features match trained dataset columns

    prediction = predict(input_data)

    if prediction == 1:
        st.error("Customer Likely to Churn")
    else:
        st.success("Customer Likely to Stay")

st.sidebar.markdown("---")

if st.sidebar.button("Check Drift & Retrain"):
    retrain_if_needed()
    st.sidebar.success("Drift check completed.")