# run.py
import importlib

import os
import pickle
from datetime import datetime
from pathlib import Path

import joblib

from backtesting.real_backtest import backtest_live_real
from optimization.optimize_indicators import run_optimization
from train import train
from backtesting import real_backtest
from utils import config
from utils.config import *
from utils.edit_config import edit_config
from trade import live_trading_loop


def show_active_model():
    model_path = Path("models/saved/active_model.pkl")
    if not os.path.exists(model_path):
        print("Active model: NONE")
        return

    try:
        with open(model_path, "rb") as f:
            model = joblib.load(model_path)


        model_name = model.__class__.__name__
        modified = model_path.stat().st_mtime
        timestamp = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Active model: {model_name} (saved {timestamp})")
    except:
        print("Active model: (could not read file)")


def print_config():
    print("\n=== Current Configuration ===")
    for key in dir(config):
        if key.isupper():
            print(f"{key:20} = {getattr(config, key)}")
    print()


def run_live_trading():
    print("\n=== LIVE TRADING ===")
    confirm = input("Start live trading? (yes/no): ").strip().lower()
    if confirm == "yes":
        live_trading_loop()
    else:
        print("Cancelled.")


def main():
    while True:
        print("\n==============================")
        print("      Trading System Menu     ")
        print("==============================")
        show_active_model()
        print_config()

        print("1. Optimize indicators")
        print("2. Train model")
        print("3. Backtest")
        print("4. Edit config values")
        print("5. Live Trading")
        print("0. Exit")

        choice = input("\nSelect an option: ").strip()

        if choice == "1":
            run_optimization(SYMBOL, TIMEFRAME, DAYS, START_DATE, END_DATE, NUMBER_TRIALS)
        elif choice == "2":
            train(SYMBOL, TIMEFRAME, DAYS, START_DATE, END_DATE)
        elif choice == "3":
            backtest_live_real(SYMBOL, TIMEFRAME, START_DATE, END_DATE, SL_ATR_MULT, TP_ATR_MULT,
                               CONF_THRESHOLD, ATR_THRESHOLD, CONTRACT_SIZE, LEVERAGE, MARGIN_LIMIT)
        elif choice == "4":
            edit_config()
        elif choice == "5":
            run_live_trading()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
