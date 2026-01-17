import numpy as np
import pandas as pd
from evaluation.metrics import f1_score, accuracy, precision, recall


def walk_forward_backtest(model_fn, df, metric="f1", train_ratio=0.7, step=200):
    """
    model_fn: function that takes train_df and returns a trained model
    df: dataframe with features + target
    metric: evaluation metric ("f1", "accuracy", "precision", "recall")
    train_ratio: initial training window
    step: number of rows to advance per iteration
    """

    metrics_map = {
        "f1": f1_score,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    metric_fn = metrics_map[metric]

    scores = []
    n = len(df)

    start_train = int(n * train_ratio)

    for start in range(start_train, n - step, step):
        train_df = df.iloc[:start]
        test_df = df.iloc[start:start + step]

        model = model_fn(train_df)

        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        preds = model.predict(X_test)
        score = metric_fn(preds, y_test)

        scores.append(score)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))
