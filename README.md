# Behavioral Churn Prediction Platform

## 📌 Project Overview
This project builds an end-to-end churn prediction system for a subscription-based business using machine learning.  
The core focus is on modeling **behavioral friction**—latent user frustration signals—rather than relying only on raw demographic or usage data.

---

## ❓ Problem Statement
Customer churn is costly for subscription-based businesses.  
The challenge is not only predicting churn, but identifying **early warning signals** that allow businesses to intervene before customers leave.

This project addresses that by:
- Engineering friction-based proxy features
- Building an interpretable churn prediction model
- Aligning model decisions with business goals

---

## 📊 Dataset
- **Source:** Telco Customer Churn dataset  
- **Rows:** ~7,000 customers  
- **Target:** `Churn` (Yes / No)

Each row represents a customer account with service, billing, and contract information.

---

## 🧠 Approach

### 1. Exploratory Data Analysis
- Analyzed churn patterns across tenure, contract type, charges, and services
- Identified high-risk segments such as early-tenure and fiber users

### 2. Feature Engineering (Key Contribution)
Since direct friction data was unavailable, **behavioral proxy features** were engineered, including:
- Early tenure risk indicators
- Cost vs value mismatch metrics
- Support access gaps
- Service complexity scores
- Payment friction signals

These features capture **latent frustration** that leads to churn.

### 3. Modeling
- Trained a **Logistic Regression** model as the final choice
- Evaluated using **ROC–AUC** and recall (churn-sensitive metrics)
- Selected a custom classification threshold (0.4) to prioritize churn detection

---

## 📈 Model Performance
- **ROC–AUC:** ~0.8  
- **Recall (Churners):** Prioritized over raw accuracy  
- The model reliably ranks churn-risk customers for early intervention

---

## 💼 Business Impact
The system enables businesses to:
- Identify high-risk customers early
- Understand *why* a customer is at risk
- Target retention efforts efficiently
- Reduce churn-related revenue loss

---

## ▶️ How to Run
1. Explore data and features:
   - `01_eda.ipynb`
   - `02_feature_engineering.ipynb`
2. Train and evaluate model:
   - `03_modeling.ipynb`
3. Load the trained model:
   ```python
   import joblib
   model = joblib.load("models/final_churn_model.pkl")
