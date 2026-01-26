# models/lgbm_model.py
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from utils.target_encoding import encode_target


def _compute_sample_weights(y: pd.Series) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    inv_freq = {cls: 1.0 / c for cls, c in zip(classes, counts)}
    return np.array([inv_freq[v] for v in y])


def train_lgbm(train_df: pd.DataFrame, params: dict | None = None):
    X = train_df.drop(columns=["target"])
    y = encode_target(train_df["target"])

    default_params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,
        "max_depth": -1,
        "num_leaves": 31,
        "n_estimators": 300,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "is_unbalance": True,
    }

    model_params = {**default_params, **(params or {})}
    sample_weight = _compute_sample_weights(y)

    base_model = lgb.LGBMClassifier(**model_params)
    base_model.fit(X, y, sample_weight=sample_weight)

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X, y, sample_weight=sample_weight)

    return calibrated
