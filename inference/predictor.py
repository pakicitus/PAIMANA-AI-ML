
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# MODEL PACKAGE LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = (BASE_DIR / ".." / "models").resolve()

cost_final_model = joblib.load(MODELS_DIR / "cost_final_model.joblib")
schedule_final_model = joblib.load(MODELS_DIR / "schedule_final_model.joblib")
cox_model = joblib.load(MODELS_DIR / "cox_model.joblib")

feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")
cox_feature_columns = joblib.load(MODELS_DIR / "cox_feature_columns.joblib")

with open(MODELS_DIR / "model_manifest.json", "r") as f:
    package_metadata = json.load(f)


# ============================================================
# UNIFIED PREDICTION FUNCTION
# ============================================================

def predict_project_row(project_row):
    """
    Takes one raw project feature row and returns
    all downstream model outputs.

    Accepted input:
        - dict
        - pandas Series
        - pandas DataFrame
    """

    # --------------------------------------------------------
    # 1. Convert input into one-row DataFrame
    # --------------------------------------------------------
    if isinstance(project_row, pd.Series):
        row = project_row.to_frame().T.copy()

    elif isinstance(project_row, dict):
        row = pd.DataFrame([project_row])

    elif isinstance(project_row, pd.DataFrame):
        row = project_row.copy()

    else:
        raise TypeError(
            "project_row must be a dict, Series, or DataFrame"
        )

    # --------------------------------------------------------
    # 2. Make sure all 47 XGBoost features exist
    # --------------------------------------------------------
    missing_features = [
        col for col in feature_columns
        if col not in row.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    xgb_row = row[feature_columns].copy()

    # --------------------------------------------------------
    # 3. Cost risk
    # --------------------------------------------------------
    cost_probability = float(
        cost_final_model.predict_proba(xgb_row)[0, 1]
    )

    # --------------------------------------------------------
    # 4. Schedule risk
    # --------------------------------------------------------
    schedule_probability = float(
        schedule_final_model.predict_proba(xgb_row)[0, 1]
    )

    # --------------------------------------------------------
    # 5. Cox risk
    # --------------------------------------------------------
    cox_row = row[cox_feature_columns].copy()

    # Cox model cannot accept missing values.
    # Use the model's training means for missing inputs.
    for col in cox_feature_columns:

        if cox_row[col].isna().any():

            if hasattr(cox_model, "_norm_mean"):
                fill_value = float(
                    cox_model._norm_mean[col]
                )
            else:
                fill_value = 0.0

            cox_row[col] = cox_row[col].fillna(fill_value)

    cox_risk = float(
        cox_model.predict_partial_hazard(cox_row).iloc[0]
    )

    # --------------------------------------------------------
    # 6. Normalize Cox risk approximately to 0-1
    # --------------------------------------------------------
    cox_risk_probability = float(
        cox_risk / (1.0 + cox_risk)
    )

    # --------------------------------------------------------
    # 7. Composite risk score
    # --------------------------------------------------------
    composite_score = (
        0.40 * cost_probability +
        0.40 * schedule_probability +
        0.20 * cox_risk_probability
    ) * 100

    composite_score = float(
        np.clip(composite_score, 0, 100)
    )

    # --------------------------------------------------------
    # 8. Risk tier
    # --------------------------------------------------------
    if composite_score < 25:
        risk_tier = "LOW"

    elif composite_score < 50:
        risk_tier = "WATCH"

    elif composite_score < 75:
        risk_tier = "ELEVATED"

    else:
        risk_tier = "CRITICAL"

    # --------------------------------------------------------
    # 9. Backend-friendly output
    # --------------------------------------------------------
    return {
        "cost_risk_probability": round(
            cost_probability, 6
        ),

        "schedule_risk_probability": round(
            schedule_probability, 6
        ),

        "cox_risk": round(
            cox_risk, 6
        ),

        "cox_risk_probability": round(
            cox_risk_probability, 6
        ),

        "composite_risk_score": round(
            composite_score, 4
        ),

        "risk_tier": risk_tier,

        "model_version": (
            package_metadata["package_version"]
            if "package_version" in package_metadata
            else "1.0.0"
        )
    }
