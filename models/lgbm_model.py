import lightgbm as lgb
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from utils.target_encoding import encode_target


def train_lgbm(train_df: pd.DataFrame, params: dict | None = None):
    """
    Trains a LightGBM multiclass classifier.
    """

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
    }

    model_params = {**default_params, **(params or {})}

    base_model = lgb.LGBMClassifier(**model_params)
    base_model.fit(X, y)

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X, y)

    return calibrated
