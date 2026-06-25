# 📉 Customer Churn Prediction System

An end-to-end machine learning pipeline to predict telecom customer churn, with a web dashboard for real-time predictions.

## 🏆 Results

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **XGBoost** ✓ | **87%** | **0.83** | **0.92** |
| Random Forest | 84% | 0.79 | 0.89 |
| Logistic Regression | 79% | 0.74 | 0.84 |

## ✨ Features

- **End-to-end ML pipeline** — data loading → preprocessing → training → evaluation → serving
- **SMOTE oversampling** to handle class imbalance (+12% F1 improvement)
- **Hyperparameter tuning** via 5-fold Stratified Cross-Validation
- **Model comparison** — Logistic Regression, Random Forest, XGBoost baselines
- **Feature importance visualization** — identifies top churn-driving factors
- **Web dashboard** (Flask) — live prediction with risk scoring UI

## 🛠 Tech Stack

- Python · Scikit-learn · XGBoost · Pandas · NumPy
- SMOTE (imbalanced-learn) · Matplotlib · Seaborn
- Flask (web dashboard)

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (downloads Telco dataset or use your own CSV)
python train.py

# Launch web dashboard
python app.py
# Open http://localhost:5000
```

## 📁 Structure

```
customer-churn-prediction/
├── train.py           # ML pipeline (preprocessing → SMOTE → train → evaluate → plot)
├── app.py             # Flask web server
├── requirements.txt
├── templates/
│   └── index.html     # Interactive prediction dashboard
├── notebooks/
│   └── exploration.ipynb
└── static/
    ├── feature_importance.png
    ├── confusion_matrix.png
    └── model_comparison.png
```

## 📊 Key Insights

- **Contract type** is the strongest churn predictor — month-to-month customers churn 3× more
- **Tenure < 12 months** + **monthly charges > $70** = highest churn risk segment
- SMOTE increased minority class recall from 58% → 74%

---
Made by [Priti Walunj](https://github.com/Priti2127)
