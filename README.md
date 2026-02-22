📌 Project Overview

This project implements an end-to-end Customer Churn Prediction MLOps Pipeline.

It simulates a real-world ML production lifecycle using automation and CI/CD.

⚙ Architecture Diagram
Raw Data (IBM Telco Dataset)
        ↓
Data Ingestion
        ↓
Preprocessing
        ↓
Model Training (RandomForest)
        ↓
MLflow Logging
        ↓
Model Saved (models/model.pkl)
        ↓
Streamlit App (Inference UI)
        ↓
Production Data (new_data.csv)
        ↓
GitHub Actions (CI/CD)
        ↓
Automatic Retraining
🚀 How to Run Locally
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/churn-mlops-pipeline.git
cd churn-mlops-pipeline
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Training Pipeline
python src/data_ingestion.py
python src/preprocessing.py
python src/train.py
4️⃣ Launch Streamlit App
streamlit run app.py
🔁 CI/CD Automation

This project uses GitHub Actions.

✅ Trigger Condition

The pipeline automatically runs when:

data/production/new_data.csv

is modified and pushed to GitHub.

📊 Model Accuracy

Model: RandomForestClassifier

Accuracy: XX%

Evaluated using train-test split

(Replace XX with your actual accuracy)

🧠 Tech Stack

Python

Scikit-learn

Pandas

MLflow

Streamlit

GitHub Actions

Docker (for Hugging Face deployment)