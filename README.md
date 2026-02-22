# 📊 Telco Churn Prediction – End-to-End MLOps Pipeline

---

## 📌 Project Overview

This project demonstrates a complete End-to-End MLOps pipeline for Telco Customer Churn Prediction.

The system automatically:

- Ingests raw telecom customer data  
- Performs preprocessing & feature engineering  
- Trains a machine learning model  
- Tracks experiments using MLflow  
- Deploys a Streamlit inference app  
- Automatically retrains when new production data is pushed  
- Runs CI/CD using GitHub Actions  

This simulates a real-world production ML lifecycle.

---

## ⚙ Architecture Diagram

```
Raw Data (IBM Telco Dataset)
        │
        ▼
Data Ingestion
        │
        ▼
Preprocessing
        │
        ▼
Model Training (RandomForest)
        │
        ├── MLflow Logging
        ├── Model Saved (models/model.pkl)
        │
        ▼
Streamlit App (Inference UI)
        │
        ▼
Production Data (new_data.csv)
        │
        ▼
GitHub Actions (CI/CD)
        │
        ▼
Automatic Retraining
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/churn-mlops-pipeline.git
cd churn-mlops-pipeline
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Pipeline

```bash
python src/data_ingestion.py
python src/preprocessing.py
python src/train.py
```

### 4️⃣ Launch Streamlit App

```bash
streamlit run app.py
```

---

## 🔁 CI/CD Automation

This project includes a GitHub Actions workflow.

### Trigger Condition

The pipeline automatically runs when:

```
data/production/new_data.csv
```

is modified and pushed to GitHub.

---

## 📊 Model Performance

- Algorithm: Random Forest Classifier  
- Accuracy: ~79%  
- Evaluation metric: Accuracy score  
- Model tracking: MLflow  

---

## 🧠 Tech Stack

- Python  
- Scikit-learn  
- Pandas  
- MLflow  
- Streamlit  
- GitHub Actions  
- Evidently (Drift Monitoring)  
- Docker (Optional)

---

## 📦 Dataset

IBM Telco Customer Churn Dataset  
Binary classification problem (Churn / Not Churn)

---

## 💡 Key Highlights

✔ End-to-end ML lifecycle  
✔ CI/CD automation  
✔ Production simulation  
✔ Model retraining pipeline  
✔ Deployment-ready structure  