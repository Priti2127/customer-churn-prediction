"""
Customer Churn Prediction Pipeline
Telecom dataset · XGBoost · SMOTE · Cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & preprocess data ─────────────────────────────────────────────────

def load_and_preprocess(path: str = "data/telecom_churn.csv") -> tuple:
    """Load telecom churn dataset and return X, y arrays."""
    df = pd.read_csv(path)

    # Drop customer ID if present
    df.drop(columns=[c for c in df.columns if 'id' in c.lower() or 'customerid' in c.lower()], errors='ignore', inplace=True)

    # Encode target
    if df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0}).fillna(df['Churn'].astype(int))

    # Handle TotalCharges blank strings (common in telecom datasets)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Label-encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y, df


# ── 2. Apply SMOTE ─────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train, random_state: int = 42):
    """Balance classes with SMOTE oversampling."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


# ── 3. Train & compare models ─────────────────────────────────────────────────

def train_models(X_train, y_train, X_test, y_test):
    """Train Logistic Regression, Random Forest, XGBoost; return results dict."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss', random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        X_tr = X_train_sc if name == 'Logistic Regression' else X_train
        X_te = X_test_sc if name == 'Logistic Regression' else X_test

        # Cross-validation F1
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1').mean()

        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]

        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'roc_auc': roc_auc_score(y_test, proba),
            'cv_f1': cv_f1,
            'preds': preds,
            'proba': proba,
        }
        print(f"  {name:20s} | Acc: {results[name]['accuracy']:.3f} | F1: {results[name]['f1']:.3f} | AUC: {results[name]['roc_auc']:.3f} | CV-F1: {cv_f1:.3f}")

    return results, scaler


# ── 4. Feature importance ──────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, top_n: int = 15, save_path: str = "static/feature_importance.png"):
    """Plot top-N feature importances for tree models."""
    importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis', ax=ax)
    ax.set_title('Top Churn-Driving Features', fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved feature importance → {save_path}")
    return importances


def plot_confusion_matrix(y_test, preds, save_path: str = "static/confusion_matrix.png"):
    """Styled confusion matrix heatmap."""
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title('Confusion Matrix — XGBoost', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


def plot_model_comparison(results: dict, save_path: str = "static/model_comparison.png"):
    """Bar chart comparing accuracy, F1, AUC across models."""
    metrics = ['accuracy', 'f1', 'roc_auc']
    labels = list(results.keys())
    x = np.arange(len(labels))
    width = 0.25
    colors = ['#4361ee', '#7209b7', '#f72585']

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[m][metric] for m in labels]
        bars = ax.bar(x + i * width, vals, width, label=metric.upper(), color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9, color='white')

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', color='white')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#252525', labelcolor='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved model comparison → {save_path}")


# ── 5. Main ───────────────────────────────────────────────────────────────────

def run_pipeline(data_path: str = "data/telecom_churn.csv"):
    print("\n=== Customer Churn Prediction Pipeline ===\n")

    print("[1/5] Loading and preprocessing data...")
    X, y, df = load_and_preprocess(data_path)
    print(f"  Dataset: {df.shape[0]} rows × {df.shape[1]} cols | Churn rate: {y.mean():.1%}")

    print("[2/5] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("[3/5] Applying SMOTE to balance classes...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("[4/5] Training and evaluating models...")
    results, scaler = train_models(X_train_res, y_train_res, X_test, y_test)

    best = max(results, key=lambda k: results[k]['f1'])
    print(f"\n  ✓ Best model: {best} (F1 = {results[best]['f1']:.3f})")
    print("\n" + classification_report(y_test, results[best]['preds'], target_names=['No Churn', 'Churn']))

    print("[5/5] Saving plots and model...")
    import os
    os.makedirs('static', exist_ok=True)
    plot_feature_importance(results[best]['model'], list(X.columns))
    plot_confusion_matrix(y_test, results[best]['preds'])
    plot_model_comparison(results)
    joblib.dump(results[best]['model'], 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n  Model saved → model.pkl")
    print("=== Pipeline complete ===\n")

    return results


if __name__ == '__main__':
    run_pipeline()
