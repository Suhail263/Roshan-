"""
CKD Random Forest ML Training Pipeline
Trains on UCI Chronic Kidney Disease dataset (or synthetic equivalent)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report, confusion_matrix)
from sklearn.impute import SimpleImputer
import pickle
import os
import json

np.random.seed(42)

def generate_ckd_dataset(n=500):
    """Generate a realistic CKD dataset matching UCI feature distributions"""
    
    # CKD positive cases (60%)
    n_ckd = int(n * 0.6)
    n_healthy = n - n_ckd

    def ckd_samples(n):
        return {
            'age': np.random.normal(55, 15, n).clip(2, 90),
            'bp': np.random.normal(80, 20, n).clip(50, 180),
            'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n, p=[0.3,0.3,0.2,0.1,0.1]),
            'al': np.random.choice([0,1,2,3,4,5], n, p=[0.1,0.15,0.2,0.25,0.2,0.1]),
            'su': np.random.choice([0,1,2,3,4,5], n, p=[0.2,0.2,0.2,0.15,0.15,0.1]),
            'bgr': np.random.normal(170, 80, n).clip(70, 490),
            'bu': np.random.normal(70, 40, n).clip(10, 400),
            'sc': np.random.normal(4, 3, n).clip(0.4, 76),
            'sod': np.random.normal(130, 15, n).clip(100, 163),
            'pot': np.random.normal(5, 2, n).clip(2.5, 47),
            'hemo': np.random.normal(10, 3, n).clip(3.1, 17.8),
            'pcv': np.random.normal(32, 10, n).clip(9, 54),
            'wc': np.random.normal(9000, 4000, n).clip(2200, 26400),
            'rc': np.random.normal(3.5, 1, n).clip(2.1, 8),
            'htn': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'dm': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'appet': np.random.choice([0, 1], n, p=[0.3, 0.7]),  # 0=good, 1=poor
            'pe': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'ane': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'class': np.ones(n, dtype=int)
        }

    def healthy_samples(n):
        return {
            'age': np.random.normal(40, 15, n).clip(2, 90),
            'bp': np.random.normal(70, 10, n).clip(50, 130),
            'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n, p=[0.05,0.1,0.2,0.35,0.3]),
            'al': np.random.choice([0,1,2,3,4,5], n, p=[0.7,0.15,0.08,0.04,0.02,0.01]),
            'su': np.random.choice([0,1,2,3,4,5], n, p=[0.8,0.1,0.05,0.03,0.01,0.01]),
            'bgr': np.random.normal(110, 25, n).clip(70, 300),
            'bu': np.random.normal(35, 12, n).clip(10, 100),
            'sc': np.random.normal(1.0, 0.3, n).clip(0.4, 5),
            'sod': np.random.normal(140, 5, n).clip(120, 163),
            'pot': np.random.normal(4.2, 0.5, n).clip(2.5, 7),
            'hemo': np.random.normal(14, 1.5, n).clip(9, 17.8),
            'pcv': np.random.normal(44, 5, n).clip(25, 54),
            'wc': np.random.normal(7500, 2000, n).clip(2200, 15000),
            'rc': np.random.normal(5, 0.6, n).clip(3, 8),
            'htn': np.random.choice([0, 1], n, p=[0.85, 0.15]),
            'dm': np.random.choice([0, 1], n, p=[0.9, 0.1]),
            'appet': np.random.choice([0, 1], n, p=[0.9, 0.1]),
            'pe': np.random.choice([0, 1], n, p=[0.95, 0.05]),
            'ane': np.random.choice([0, 1], n, p=[0.9, 0.1]),
            'class': np.zeros(n, dtype=int)
        }

    ckd_df = pd.DataFrame(ckd_samples(n_ckd))
    healthy_df = pd.DataFrame(healthy_samples(n_healthy))
    df = pd.concat([ckd_df, healthy_df], ignore_index=True).sample(frac=1, random_state=42)

    # Introduce ~10% missing values realistically
    for col in ['bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']:
        mask = np.random.random(len(df)) < 0.08
        df.loc[mask, col] = np.nan

    return df


def train_and_save():
    print("=" * 60)
    print("CKD Random Forest Training Pipeline")
    print("=" * 60)

    df = generate_ckd_dataset(600)
    print(f"Dataset shape: {df.shape}")
    print(f"CKD cases: {df['class'].sum()}, Healthy: {(df['class']==0).sum()}")

    # Save dataset
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/ckd_dataset.csv', index=False)

    # Features & target
    feature_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot',
                    'hemo','pcv','wc','rc','htn','dm','appet','pe','ane']
    X = df[feature_cols]
    y = df['class']

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=feature_cols)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Random Forest with tuned hyperparameters
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(rf, X_imputed, y, cv=5, scoring='accuracy')

    print(f"\n{'='*40}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*40}")
    print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"ROC-AUC:         {roc:.4f}")
    print(f"CV Score (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Healthy','CKD'])}")

    # Feature importance
    importances = dict(zip(feature_cols, rf.feature_importances_.tolist()))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    print("Top 10 Feature Importances:")
    for feat, imp in list(sorted_imp.items())[:10]:
        bar = '█' * int(imp * 50)
        print(f"  {feat:8s}: {bar} {imp:.4f}")

    # Save model artifacts
    os.makedirs('saved', exist_ok=True)
    model_artifacts = {
        'model': rf,
        'imputer': imputer,
        'feature_cols': feature_cols,
        'metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(roc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std())
        },
        'feature_importance': sorted_imp,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    with open('saved/ckd_model.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)

    # Save metrics JSON for API
    with open('saved/model_metrics.json', 'w') as f:
        json.dump({
            'metrics': model_artifacts['metrics'],
            'feature_importance': sorted_imp,
            'confusion_matrix': model_artifacts['confusion_matrix']
        }, f, indent=2)

    print(f"\n✅ Model saved to saved/ckd_model.pkl")
    print(f"✅ Metrics saved to saved/model_metrics.json")
    return model_artifacts


if __name__ == '__main__':
    train_and_save()
