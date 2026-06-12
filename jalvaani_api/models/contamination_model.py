"""
JalVaani Day 3 — Contamination risk classifiers.
Three LogisticRegression pipelines (fluoride, arsenic, nitrate).

Note on interpretation:
  - Fluoride (AUC 0.80 leave-states-out): genuinely predictive, geology-driven.
  - Arsenic (AUC 0.41), Nitrate (AUC 0.50): state-aggregate labels limit
    cross-state generalization; treat these scores with caution.
  - Probabilities reflect state-level contamination prevalence more than
    well-level chemistry.
"""
from pathlib import Path

import joblib
import numpy as np

_fluoride = None
_arsenic = None
_nitrate = None
_feature_names: list = []

SAVED = Path(__file__).parent.parent / "saved_models"


def load() -> bool:
    global _fluoride, _arsenic, _nitrate, _feature_names
    _fluoride = joblib.load(SAVED / "jalvaani_fluoride_model.pkl")
    _arsenic = joblib.load(SAVED / "jalvaani_arsenic_model.pkl")
    _nitrate = joblib.load(SAVED / "jalvaani_nitrate_model.pkl")
    _feature_names = joblib.load(SAVED / "jalvaani_classification_features.pkl")
    return True


def get_feature_names() -> list:
    return _feature_names


def _label(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.35:
        return "medium"
    return "low"


def predict(feature_vector: np.ndarray) -> dict:
    """
    Returns probabilities and risk labels for fluoride, arsenic, nitrate.

    overall_risk_score = max(F, A, N) — the worst contaminant drives the headline.
    weighted_avg_risk  = 0.4·F + 0.4·A + 0.2·N — retained as secondary signal.
    risk_drivers       = list of contaminants classified as "high" (prob ≥ 0.6).
    """
    if _fluoride is None:
        raise RuntimeError("Contamination models not loaded")
    x = feature_vector.reshape(1, -1)
    f = float(_fluoride.predict_proba(x)[0][1])
    a = float(_arsenic.predict_proba(x)[0][1])
    n = float(_nitrate.predict_proba(x)[0][1])
    drivers = [name for name, p in [("fluoride", f), ("arsenic", a), ("nitrate", n)]
               if _label(p) == "high"]
    return {
        "fluoride": {"risk_level": _label(f), "probability": round(f, 4)},
        "arsenic": {"risk_level": _label(a), "probability": round(a, 4)},
        "nitrate": {"risk_level": _label(n), "probability": round(n, 4)},
        "overall_risk_score": round(max(f, a, n), 4),
        "weighted_avg_risk": round(0.4 * f + 0.4 * a + 0.2 * n, 4),
        "risk_drivers": drivers,
    }
