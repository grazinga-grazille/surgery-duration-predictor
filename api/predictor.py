"""
predictor.py — Model loading and prediction logic.

Responsible for:
  1. Loading all .pkl artifacts once at startup
  2. Exposing a predict() function the API endpoint can call
"""

import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
# api/ is one level below the project root, so we go up one level to find models/
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# ── Text cleaning ──────────────────────────────────────────────────────────────
CUSTOM_STOP_WORDS = [
    "removal", "extraction", "total", "non", "diagnostic", "performed",
    "procedure", "status", "post", "bilateral", "left", "right", "well",
    "tolerated", "bx", "fx", "tx", "dx", "hx", "sx", "ex",
]

def clean_text(txt: str) -> str:
    txt = str(txt).lower()
    txt = re.sub(r"[^a-z ]", " ", txt)
    return " ".join(w for w in txt.split() if w not in CUSTOM_STOP_WORDS)


# ── Artifact loading ───────────────────────────────────────────────────────────
# @lru_cache ensures this function only runs once — artifacts are loaded into
# memory at startup and reused for every subsequent request.
@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    def _load(name):
        with open(MODELS_DIR / name, "rb") as f:
            return pickle.load(f)
        
    return {
        "rf_model": _load("rf_model.pkl"),
        "tfidf": _load("tfidf.pkl"),
        "svd": _load("svd.pkl"),
        "feature_columns": _load("feature_columns.pkl"),
        "categorical_values": _load("categorical_values.pkl"),
        "valid_combinations": _load("valid_combinations.pkl"),
        "procedure_examples": _load("procedure_examples.pkl"),
        "model_stats": _load("model_stats.pkl"),
        "test_results": _load("test_results.pkl"),
    }


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(
    surgical_priority: int,
    patient_type: str,
    room: str,
    specialty: str,
    procedure_description: str,
) -> float:
    """
    Run the model and return predicted duration in minutes.
    """
    artifacts = load_artifacts()

    #   1. Clean the procedure_description text using clean_text()
    procedure_description = clean_text(procedure_description)
    #   2. Transform it with tfidf, then svd
    tfidf_vector = artifacts["tfidf"].transform([procedure_description])
    svd_vector = artifacts["svd"].transform(tfidf_vector)
    #   3. Build the numerical and categorical DataFrames
    svd_df = pd.DataFrame(svd_vector, columns=[f"SVD_{i}" for i in range(svd_vector.shape[1])])
    X_num = pd.DataFrame({
        "SurgicalPriority": [surgical_priority],
    })
    #   4. One-hot encode the categorical features
    X_cat = pd.DataFrame({
        "PatientType": [patient_type],
        "Roomdescription": [room],
        "ProcedureSpecialtyDescription": [specialty],
    })
    X_cat_encoded = pd.get_dummies(X_cat, columns=X_cat.columns, drop_first=True, dtype=int)
    #   5. Concatenate everything and reindex to feature_columns
    X = pd.concat([X_num, X_cat_encoded, svd_df], axis=1).reindex(
        columns=artifacts["feature_columns"], fill_value=0
    )
    #   6. Run rf_model.predict(), exponentiate the result, return as float
    return float(np.exp(artifacts["rf_model"].predict(X)[0]))
