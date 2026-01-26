# models/xgb_model.py
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from utils.target_encoding import encode_target


def _compute_sample_weights(y: pd.Series) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    inv_freq = {cls: 1.0 / c for cls, c in zip(classes, counts)}
    return np.array([inv_freq[v] for v in y])


def train_xgb(train_df: pd.DataFrame, params: dict):
    X = train_df.drop(columns=["target"])
    y = encode_target(train_df["target"])

    default_params = {
        "objective": "multi:softprob",  # probabilities, not hard labels
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    model_params = {**default_params, **params}
    sample_weight = _compute_sample_weights(y)

    base_model = xgb.XGBClassifier(**model_params)
    base_model.fit(X, y, sample_weight=sample_weight)

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X, y, sample_weight=sample_weight)

    return calibrated
