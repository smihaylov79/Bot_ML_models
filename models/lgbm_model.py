import lightgbm as lgb
import pandas as pd
from utils.target_encoding import encode_target


def train_lgbm(train_df: pd.DataFrame, params: dict | None = None):
    """
    Trains a LightGBM multiclass classifier.
    """

    X = train_df.drop(columns=["target"])
    y = train_df["target"]

    y = encode_target(train_df["target"])

    default_params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,
        "max_depth": -1,
        "num_leaves": 31,
        "n_estimators": 400,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    model_params = {**default_params, **(params or {})}

    model = lgb.LGBMClassifier(**model_params)
    model.fit(X, y)

    return model
