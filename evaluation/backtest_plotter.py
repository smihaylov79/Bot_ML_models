import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics import f1_score, confusion_matrix


# ---------------------------------------------------------
# 1. Equity Curve Plot
# ---------------------------------------------------------
def plot_equity_curve(df, preds, future_n=2, title="Equity Curve"):
    """
    df: dataframe with 'close' and 'target'
    preds: model predictions aligned with df.index
    """

    df = df.copy()
    df["pred"] = preds

    # Future returns
    df["future_ret"] = df["close"].shift(-future_n) / df["close"] - 1

    # Strategy return
    df["strategy_ret"] = df["future_ret"] * df["pred"]

    # Equity curve
    df["equity"] = (1 + df["strategy_ret"].fillna(0)).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["equity"], label="Strategy Equity", color="blue")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 2. Rolling F1 Score Plot
# ---------------------------------------------------------
def plot_rolling_f1(df, preds, window=200, title="Rolling F1 Score"):
    df = df.copy()
    df["pred"] = preds

    f1_values = []

    for i in range(window, len(df)):
        y_true = df["target"].iloc[i - window:i]
        y_pred = df["pred"].iloc[i - window:i]
        f1_values.append(f1_score(y_pred, y_true))

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[window:], f1_values, label="Rolling F1", color="purple")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 3. Confusion Matrix Heatmap
# ---------------------------------------------------------
def plot_confusion_matrix(df, preds, title="Confusion Matrix"):
    cm = confusion_matrix(preds, df["target"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["-1", "0", "1"],
        yticklabels=["-1", "0", "1"]
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 4. Model Comparison Bar Chart
# ---------------------------------------------------------
def plot_model_comparison(scores: dict, title="Model Comparison"):
    """
    scores: {"xgb": 0.55, "rf": 0.48, "lgbm": 0.52}
    """

    names = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=values, palette="viridis")
    plt.title(title)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 5. Feature Importance Plot (XGB / LGBM)
# ---------------------------------------------------------
def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """
    Works for XGBoost and LightGBM models.
    """

    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature_importances_. Skipping plot.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[idx],
        y=np.array(feature_names)[idx],
        palette="magma"
    )
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
