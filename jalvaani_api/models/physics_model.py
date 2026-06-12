"""
JalVaani Day 2 — Physics-Guided NN (GroundwaterNet Version B).
Used for physics consistency check: compare ensemble vs physics prediction.
If they differ by > 2 mbgl, flag as unusual hydrogeological conditions.
"""
from pathlib import Path

import joblib
import numpy as np
import torch

from .jalvaani_model_architecture import GroundwaterNet

_model = None
_scaler = None
_yscaler = None

SAVED = Path(__file__).parent.parent / "saved_models"


def load() -> bool:
    global _model, _scaler, _yscaler
    _model = GroundwaterNet(n_features=45)
    state_dict = torch.load(
        SAVED / "jalvaani_physicsnn_day2_final.pth",
        map_location="cpu",
        weights_only=True,
    )
    _model.load_state_dict(state_dict)
    _model.eval()
    _scaler = joblib.load(SAVED / "jalvaani_scaler_day2.pkl")
    _yscaler = joblib.load(SAVED / "jalvaani_yscaler_day2.pkl")
    return True


def predict(feature_vector: np.ndarray) -> float:
    """Predict depth using physics-guided NN. feature_vector must be 45-d."""
    if _model is None:
        raise RuntimeError("Physics model not loaded")
    x = _scaler.transform(feature_vector.reshape(1, -1)).astype(np.float32)
    t = torch.from_numpy(x)
    with torch.no_grad():
        y_scaled = _model(t).item()
    return float(_yscaler.inverse_transform([[y_scaled]])[0][0])
