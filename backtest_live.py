import matplotlib
matplotlib.use("TkAgg")
import joblib
from pathlib import Path
from collections import Counter

import pandas as pd


from data_loader.mt5_loader import load_data
from features.feature_engineering import build_features
from utils.target_encoding import decode_target
from evaluation.metrics import f1_score
from evaluation.backtest_plotter import (
    plot_equity_curve,
    plot_confusion_matrix,
    plot_rolling_f1
)

MODEL_PATH = Path("models/saved/active_model.pkl")

def backtest_live(symbol, timeframe, days, start_date, end_date):
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading MT5 data...")
    df_raw = load_data(symbol, timeframe, days, start_date, end_date)

    print("Building features...")
    df = build_features(df_raw)

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    print("Generating predictions...")
    preds = model.predict(X)
    preds = decode_target(preds)

    # Flip signal (based on your discovery)
    preds = preds

    print("Predicted class distribution:", Counter(preds))
    print("True class distribution:", Counter(y))

    score = f1_score(preds, y)
    print(f"Live backtest F1 score: {score:.4f}")

    print("Plotting results...")
    plot_equity_curve(df, preds, future_n=2, title="Live Backtest Equity Curve")
    plot_rolling_f1(df, preds, window=200, title="Live Backtest Rolling F1")
    plot_confusion_matrix(df, preds, title="Live Backtest Confusion Matrix")

    return score

if __name__ == "__main__":
    from utils.config import SYMBOL, TIMEFRAME, DAYS, START_DATE, END_DATE

    backtest_live(SYMBOL, TIMEFRAME, DAYS, START_DATE, END_DATE)
