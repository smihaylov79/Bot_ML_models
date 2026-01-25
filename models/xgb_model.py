import xgboost as xgb
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from utils.target_encoding import encode_target


def train_xgb(train_df: pd.DataFrame, params: dict):
    """
    Trains an XGBoost model using the given hyperparameters.
    train_df must contain 'target' column.
    """

    X = train_df.drop(columns=["target"])

    y = encode_target(train_df["target"])

    default_params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    # Merge defaults with Optuna params
    model_params = {**default_params, **params}

    base_model = xgb.XGBClassifier(**model_params)
    base_model.fit(X, y)

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X, y)

    return calibrated
