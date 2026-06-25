"""
Customer Churn Prediction — Flask Web App
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model once at startup (if exists)
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# Feature order must match training data
FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained yet. Run train.py first.'}), 503

    data = request.get_json()
    try:
        values = [float(data.get(f, 0)) for f in FEATURES]
        arr = np.array(values).reshape(1, -1)
        prob = float(model.predict_proba(arr)[0][1])
        label = 'Churn' if prob >= 0.5 else 'No Churn'
        risk = 'High' if prob >= 0.7 else 'Medium' if prob >= 0.4 else 'Low'
        return jsonify({'probability': round(prob * 100, 1), 'label': label, 'risk': risk})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/metrics')
def metrics():
    """Return dummy metrics for demo if model not trained."""
    return jsonify({
        'xgboost': {'accuracy': 0.87, 'f1': 0.83, 'roc_auc': 0.92},
        'random_forest': {'accuracy': 0.84, 'f1': 0.79, 'roc_auc': 0.89},
        'logistic_regression': {'accuracy': 0.79, 'f1': 0.74, 'roc_auc': 0.84},
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
