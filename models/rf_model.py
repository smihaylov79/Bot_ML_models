# models/rf_model.py
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from utils.target_encoding import encode_target


def _compute_sample_weights(y: pd.Series) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    inv_freq = {cls: 1.0 / c for cls, c in zip(classes, counts)}
    return np.array([inv_freq[v] for v in y])


def train_rf(train_df: pd.DataFrame, params: dict | None = None):
    X = train_df.drop(columns=["target"])
    y = encode_target(train_df["target"])

    default_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "bootstrap": True,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    model_params = {**default_params, **(params or {})}
    sample_weight = _compute_sample_weights(y)

    base_model = RandomForestClassifier(**model_params)
    base_model.fit(X, y, sample_weight=sample_weight)

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X, y, sample_weight=sample_weight)

    return calibrated
