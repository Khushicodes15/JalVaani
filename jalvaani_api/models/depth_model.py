"""
JalVaani Day 1 — Stacking Ensemble depth predictor.
Loads jalvaani_model_best.pkl (XGBoost + RF → Ridge, ~2.5 GB).
"""
from pathlib import Path

import joblib
import numpy as np

_model = None
_feature_names: list = []

SAVED = Path(__file__).parent.parent / "saved_models"


def load() -> bool:
    global _model, _feature_names
    _model = joblib.load(SAVED / "jalvaani_model_best.pkl")
    _feature_names = joblib.load(SAVED / "jalvaani_features_best.pkl")
    return True


def predict(feature_vector: np.ndarray) -> float:
    if _model is None:
        raise RuntimeError("Depth model not loaded")
    return float(_model.predict(feature_vector.reshape(1, -1))[0])


def get_feature_names() -> list:
    return _feature_names
