import joblib
import pandas as pd

def predict_patient(model_path: str, patient_dict: dict):
    model = joblib.load(model_path)
    X = pd.DataFrame([patient_dict])

    stage = model.predict(X)[0]
    risk = None
    if hasattr(model, "predict_proba"):
        risk = float(model.predict_proba(X).max())

    return stage, risk