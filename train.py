import matplotlib
matplotlib.use("TkAgg")

from collections import Counter

import joblib
from pathlib import Path

import pandas as pd

from data_loader.mt5_loader import load_data
from features.feature_engineering import build_features
from evaluation.metrics import f1_score
from models.model_registry import train_all_models
from utils.params_io import load_best_params

from models.model_registry import MODEL_REGISTRY
from evaluation.backtest_plotter import ( plot_equity_curve, plot_rolling_f1, plot_confusion_matrix, plot_feature_importance )
from utils.target_encoding import decode_target
from utils.config import TIMEFRAME, DAYS, SYMBOL

SAVE_DIR = Path("models/saved")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def train(symbol, timeframe, days):
    print("Loading optimized parameters...")
    best_params = load_best_params()

    indicator_params = {
        k: v for k, v in best_params.items()
        if k not in ["xgb", "rf", "lgbm"]
    }

    model_params = {
        "xgb": best_params.get("xgb", {}),
        "rf": best_params.get("rf", {}),
        "lgbm": best_params.get("lgbm", {}),
    }

    print("Loading MT5 data...")
    df_raw = load_data(symbol, timeframe, days)

    print("Building features...")
    df = build_features(df_raw, params=indicator_params)

    # Train/test split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print("Training all models...")
    trained_models = train_all_models(train_df, model_params)

    print("Evaluating models...")
    scores = {}

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    for name, model in trained_models.items():
        preds = model.predict(X_test)

        preds = decode_target(preds)


        print("Predicted class distribution:", Counter(preds))
        print("True class distribution:", Counter(y_test))

        score = f1_score(preds, y_test)
        scores[name] = score
        print(f"{name} F1 score: {score:.4f}")

    best_model_name = max(scores, key=scores.get)
    best_model = trained_models[best_model_name]

    print(f"\nBest model: {best_model_name} (F1={scores[best_model_name]:.4f})")

    # 📊 PLOTTING SECTION #
    preds = best_model.predict(X_test)
    print("Generating backtest plots...")
    plot_equity_curve(test_df, preds, future_n=2, title=f"Equity Curve - {best_model_name}")
    plot_rolling_f1(test_df, preds, window=200, title=f"Rolling F1 Score - {best_model_name}")
    plot_confusion_matrix(test_df, preds, title=f"Confusion Matrix - {best_model_name}")
    plot_feature_importance(best_model, X_test.columns, title=f"Feature Importance - {best_model_name}")

    # --------------

    save_path = SAVE_DIR / f"{best_model_name}_best.pkl"
    joblib.dump(best_model, save_path)

    print(f"Saved best model to {save_path}")

    return best_model_name, save_path


if __name__ == "__main__":
    train(SYMBOL, TIMEFRAME, DAYS)
