"""Minimal FastAPI service scoring a single employee's attrition risk."""
from fastapi import FastAPI
import joblib
import pandas as pd

from src.main import MODELS

app = FastAPI(title="HR Attrition API")
_bundle = None


@app.on_event("startup")
def _load():
    global _bundle
    _bundle = joblib.load(MODELS / "attrition_model.joblib")


@app.post("/score")
def score(employee: dict):
    X = pd.get_dummies(pd.DataFrame([employee]))
    X = X.reindex(columns=_bundle["columns"], fill_value=0)
    model = _bundle["model"]
    feats = _bundle["scaler"].transform(X) if _bundle.get("needs_scaling") else X
    prob = float(model.predict_proba(feats)[0, 1])
    flag = prob >= _bundle.get("threshold", 0.5)
    return {"attrition_probability": prob, "flag_for_retention": bool(flag)}
