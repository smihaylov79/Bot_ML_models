from evaluation.metrics import f1_score, accuracy, precision, recall


def evaluate_model(model, test_df, metric="f1"):
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    preds = model.predict(X_test)

    metrics_map = {
        "f1": f1_score,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    return metrics_map[metric](preds, y_test)


def evaluate_all(models: dict, train_df, test_df, metric="f1"):
    """
    models: dict of {name: train_fn}
    train_fn: function(train_df) -> model
    """

    results = {}

    for name, train_fn in models.items():
        model = train_fn(train_df)
        score = evaluate_model(model, test_df, metric)
        results[name] = score

    return results


def select_best_model(results: dict):
    """
    results: {model_name: score}
    """
    return max(results, key=results.get)
