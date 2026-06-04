"""ML prediction engine - loads trained Random Forest model"""
import pickle
import numpy as np
import pandas as pd
import os
import json

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'saved', 'ckd_model.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'saved', 'model_metrics.json')

_artifacts = None

def load_model():
    global _artifacts
    if _artifacts is None:
        with open(MODEL_PATH, 'rb') as f:
            _artifacts = pickle.load(f)
        print("✅ CKD model loaded successfully")
    return _artifacts

def get_recommendations(result, confidence, input_data):
    recs = []
    if result == 'CKD':
        recs.append("⚠️ Consult a nephrologist immediately for comprehensive kidney evaluation.")
        if input_data.get('sc', 0) > 1.2:
            recs.append("🔬 High serum creatinine detected. Kidney filtration may be compromised.")
        if input_data.get('bu', 0) > 40:
            recs.append("🧪 Elevated blood urea. Restrict protein intake and increase hydration.")
        if input_data.get('htn', 0) == 1:
            recs.append("💊 Manage hypertension aggressively with prescribed medications.")
        if input_data.get('dm', 0) == 1:
            recs.append("🩺 Diabetic nephropathy risk. Maintain strict blood glucose control.")
        if input_data.get('hemo', 15) < 12:
            recs.append("💉 Anemia detected. Iron supplements and EPO therapy may be required.")
        recs.append("🥗 Follow a low-sodium, low-potassium, kidney-friendly diet.")
        recs.append("💧 Monitor fluid intake carefully as advised by your doctor.")
        if confidence > 0.85:
            recs.append("🚨 High confidence CKD prediction. Immediate medical attention strongly recommended.")
    else:
        recs.append("✅ No CKD indicators detected in current assessment.")
        recs.append("💧 Stay well hydrated — drink 2-3 liters of water daily.")
        recs.append("🥦 Maintain a balanced diet rich in fruits, vegetables, and whole grains.")
        recs.append("🏃 Regular exercise (30 min/day) supports kidney health.")
        recs.append("🩺 Schedule annual kidney function tests as preventive care.")
        if input_data.get('htn', 0) == 1:
            recs.append("⚠️ Monitor blood pressure regularly — hypertension is a CKD risk factor.")
        if input_data.get('dm', 0) == 1:
            recs.append("⚠️ Control blood sugar levels to prevent future kidney damage.")
    return recs

def get_risk_level(result, confidence):
    if result == 'CKD':
        if confidence >= 0.85: return 'Critical'
        if confidence >= 0.70: return 'High'
        return 'Medium'
    else:
        if confidence >= 0.90: return 'Low'
        return 'Medium'

def predict(input_data: dict):
    arts = load_model()
    model = arts['model']
    imputer = arts['imputer']
    feature_cols = arts['feature_cols']

    # Build feature vector
    features = []
    for col in feature_cols:
        val = input_data.get(col, np.nan)
        features.append(float(val) if val not in (None, '', 'nan') else np.nan)

    X = pd.DataFrame([features], columns=feature_cols)
    X_imputed = imputer.transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=feature_cols)

    prediction = model.predict(X_imputed)[0]
    probabilities = model.predict_proba(X_imputed)[0]
    confidence = float(max(probabilities))

    result = 'CKD' if prediction == 1 else 'Healthy'
    risk_level = get_risk_level(result, confidence)
    recommendations = get_recommendations(result, confidence, input_data)
    ckd_probability = float(probabilities[1])

    return {
        'result': result,
        'confidence': round(confidence * 100, 2),
        'ckd_probability': round(ckd_probability * 100, 2),
        'risk_level': risk_level,
        'recommendations': recommendations,
        'feature_importance': arts['feature_importance']
    }

def get_model_metrics():
    with open(METRICS_PATH, 'r') as f:
        return json.load(f)
