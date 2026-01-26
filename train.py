import matplotlib
import numpy as np

from models.lgbm_model import train_lgbm
from models.rf_model import train_rf
from models.xgb_model import train_xgb

matplotlib.use("TkAgg")

from collections import Counter

import joblib
from pathlib import Path

import pandas as pd

from data_loader.mt5_loader import load_data
from features.feature_engineering import build_features
from evaluation.metrics import f1_score
from models.model_registry import train_all_models, get_model
from utils.params_io import load_best_params

from models.model_registry import MODEL_REGISTRY
from evaluation.backtest_plotter import ( plot_equity_curve, plot_rolling_f1, plot_confusion_matrix, plot_feature_importance )
from utils.target_encoding import decode_target
from utils.config import TIMEFRAME, DAYS, SYMBOL, START_DATE, END_DATE
from features.regime.regime_detector import RegimeDetector
import pickle


SAVE_DIR = Path("models/saved")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def save_active_model(model):
    save_path = SAVE_DIR / "active_model.pkl"
    joblib.dump(model, save_path)
    print(f"Saved active model to {save_path}")


def train(symbol, timeframe, days, start_date, end_date):
    print("Loading optimized parameters...")
    best_params = load_best_params()

    # Extract model name
    model_name = best_params["model_name"]

    # Extract indicator parameters
    indicator_params = best_params["indicators"]

    # Extract the CLEAN model parameters
    # These were stored under "model_params" in the objective
    model_params = best_params["model_params"]

    print(f"Training model: {model_name}")
    print("Indicator params:", indicator_params)
    print("Model params:", model_params)

    # Load raw data
    df_raw = load_data(symbol, timeframe, days, start_date, end_date)

    # Build features
    df_feat = build_features(
        df_raw,
        params=indicator_params,
        future_n=20
    )

    # Train the correct model
    if model_name == "xgb":
        model = train_xgb(df_feat, model_params)
    elif model_name == "rf":
        model = train_rf(df_feat, model_params)
    else:
        model = train_lgbm(df_feat, model_params)

    # Save model
    save_active_model(model)

    print("Model training complete.")
    return model

#
# def train(symbol, timeframe, days, start_date, end_date):
#     print("Loading optimized parameters...")
#     best_params = load_best_params()
#
#     indicator_params = {
#         k: v for k, v in best_params.items()
#         if k not in ["xgb", "rf", "lgbm"]
#     }
#
#     model_params = {
#         "xgb": best_params.get("xgb", {}),
#         "rf": best_params.get("rf", {}),
#         "lgbm": best_params.get("lgbm", {}),
#     }
#
#     print("Loading MT5 data...")
#     df_raw = load_data(symbol, timeframe, days, start_date=start_date, end_date=end_date)
#
#     print("Building features...")
#     df = build_features(df_raw, params=indicator_params)
#
#     # print("Detecting market regimes...")
#
#     # regime_detector = RegimeDetector(bb_expansion_percentile=70, bb_crush_percentile=30, atr_crush_percentile=30,
#     #                                  ema_slope_threshold=0.0, )
#
#     # Fit on train only to avoid leakage
#     # split = int(len(df) * 0.8)
#     # train_df = df.iloc[:split].copy()
#     # test_df = df.iloc[split:].copy()
#
#     # train_df["regime"] = regime_detector.fit_transform(train_df)
#     # test_df["regime"] = regime_detector.transform(test_df)
#
#     # Train/test split
#     split = int(len(df) * 0.8)
#     train_df = df.iloc[:split]
#     test_df = df.iloc[split:]
#
#     print("Training all models...")
#     trained_models = train_all_models(train_df, model_params)
#
#     print("Evaluating models...")
#     scores = {}
#
#     X_test = test_df.drop(columns=["target"])
#     y_test = test_df["target"]
#
#     for name, model in trained_models.items():
#         preds = model.predict(X_test)
#
#         preds = decode_target(preds)
#
#         print("Predicted class distribution:", Counter(preds))
#         print("True class distribution:", Counter(y_test))
#
#         score = f1_score(preds, y_test)
#         scores[name] = score
#         print(f"{name} F1 score: {score:.4f}")
#
#         # Shift predictions forward by 1 bar
#         preds_shifted = np.roll(preds, 1)
#         preds_shifted[0] = 0  # first value invalid → set to no-trade
#
#         score_shifted = f1_score(preds_shifted, y_test)
#         print(f"{name} SHIFTED F1 score: {score_shifted:.4f}")
#
#         # Flip predictions
#         preds_flipped = preds.copy()
#         preds_flipped[preds == -1] = 1
#         preds_flipped[preds == 1] = -1
#
#         score_flipped = f1_score(preds_flipped, y_test)
#         print(f"{name} FLIPPED F1 score: {score_flipped:.4f}")
#
#     best_model_name = max(scores, key=scores.get)
#     best_model = trained_models[best_model_name]
#
#     print(f"\nBest model: {best_model_name} (F1={scores[best_model_name]:.4f})")
#
#     # 📊 PLOTTING SECTION #
#     preds = best_model.predict(X_test)
#     preds_shifted = np.roll(preds, 1)
#     preds_shifted[0] = 0  # first value invalid → set to no-trade
#     preds_flipped = preds.copy()
#     preds_flipped[preds == -1] = 1
#     preds_flipped[preds == 1] = -1
#     print("Generating backtest plots...")
#     plot_equity_curve(test_df, preds, future_n=2, title=f"Equity Curve - {best_model_name}")
#     plot_rolling_f1(test_df, preds, window=200, title=f"Rolling F1 Score - {best_model_name}")
#     plot_confusion_matrix(test_df, preds, title=f"Confusion Matrix - {best_model_name} - normal")
#     plot_confusion_matrix(test_df, preds_shifted, title=f"Confusion Matrix - {best_model_name} - shifted")
#     plot_confusion_matrix(test_df, preds_flipped, title=f"Confusion Matrix - {best_model_name} - flipped")
#     plot_feature_importance(best_model, X_test.columns, title=f"Feature Importance - {best_model_name}")
#
#     # --------------
#     save_path = SAVE_DIR / "active_model.pkl"
#     joblib.dump(best_model, save_path)
#     print(f"Saved active model to {save_path}")
#
#     return best_model_name, save_path


if __name__ == "__main__":
    train(SYMBOL, TIMEFRAME, DAYS, START_DATE, END_DATE)
