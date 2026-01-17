from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils.target_encoding import encode_target


def train_rf(train_df: pd.DataFrame, params: dict | None = None):
    """
    Trains a RandomForest classifier.
    train_df must contain 'target'.
    """

    X = train_df.drop(columns=["target"])
    y = train_df["target"]

    y = encode_target(train_df["target"])

    default_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "bootstrap": True,
        "n_jobs": -1,
    }

    model_params = {**default_params, **(params or {})}

    model = RandomForestClassifier(**model_params)
    model.fit(X, y)

    return model
