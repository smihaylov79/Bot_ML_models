import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from data_loader.mt5_loader import load_data
from diagnostics.regime_features import compute_trend_strength
from features.feature_engineering import build_features
from utils.params_io import load_best_params
from utils.target_encoding import decode_target
from utils.config import SYMBOL, TIMEFRAME, INITIAL_BALANCE, POSITION_SIZE, START_DATE, END_DATE

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "saved" / "active_model.pkl"
# PARAMS_PATH = Path("utils/best_params.json")

#
# def load_best_params(p):
#         return joblib.load(p)


def prepare_data(symbol, timeframe, start_date, end_date):
    print("Loading MT5 data...")
    df_raw = load_data(symbol=symbol, timeframe=timeframe,
                       start_date=start_date, end_date=end_date)

    print("Loading best params...")
    best_params = load_best_params()

    print("Building features...")
    df = build_features(df_raw.copy(), best_params)

    # Drop rows with NaNs from indicators
    df = df.dropna().copy()
    # df["hour"] = df.index.hour
    # df["weekday"] = df.index.day_name()
    # df["atr_norm"] = df["atr"] / df["close"]
    # df["trend_strength"] = compute_trend_strength(df["close"])  # from regime_features

    return df, best_params


def load_model():
    print("Loading model...")
    return joblib.load(MODEL_PATH)

def generate_signals(model, df):
    X = df.drop(columns=["target"]) if "target" in df.columns else df.copy()
    preds = model.predict(X)
    preds = decode_target(preds)  # -> -1, 0, 1
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    return preds, conf


def backtest_hedging(df, signals, conf, sl_mult=1.5, tp_mult=2.5,
                     initial_balance=INITIAL_BALANCE,
                     position_size=POSITION_SIZE, conf_threshold=0.55, atr_norm_threshold=0.5, contr_size=1, lev=20, marg_limit=0.5):
    balance = initial_balance
    equity_curve = []
    open_trades = []  # list of dicts
    trade_log = []
    used_margin = 0

    prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atr = df["atr"].values
    index = df.index

    atr_norm = (df["atr"] / df["close"]).values


    for i in range(1, len(df)):
        price = prices[i]
        bar_high = highs[i]
        bar_low = lows[i]
        bar_time = index[i]

        # 1) Update open trades (check SL/TP)
        still_open = []
        for trade in open_trades:
            if trade["direction"] == 1:  # long
                tp = trade["entry_price"] + tp_mult * trade["atr"]
                sl = trade["entry_price"] - sl_mult * trade["atr"]

                hit_tp = bar_high >= tp
                hit_sl = bar_low <= sl

                if hit_tp or hit_sl:
                    exit_price = tp if hit_tp else sl
                    pnl_points = (exit_price - trade["entry_price"])
                    pnl = pnl_points * trade["size"]
                    balance += pnl

                    used_margin -= trade['margin']

                    trade["exit_time"] = bar_time
                    trade["exit_price"] = exit_price
                    trade["pnl"] = pnl
                    trade["pnl_points"] = pnl_points
                    trade["holding_bars"] = i - trade["entry_index"]
                    trade_log.append(trade)
                else:
                    still_open.append(trade)

            elif trade["direction"] == -1:  # short
                tp = trade["entry_price"] - tp_mult * trade["atr"]
                sl = trade["entry_price"] + sl_mult * trade["atr"]

                hit_tp = bar_low <= tp
                hit_sl = bar_high >= sl

                if hit_tp or hit_sl:
                    exit_price = tp if hit_tp else sl
                    pnl_points = (trade["entry_price"] - exit_price)
                    pnl = pnl_points * trade["size"]
                    balance += pnl

                    used_margin -= trade['margin']

                    trade["exit_time"] = bar_time
                    trade["exit_price"] = exit_price
                    trade["pnl"] = pnl
                    trade["pnl_points"] = pnl_points
                    trade["holding_bars"] = i - trade["entry_index"]
                    trade_log.append(trade)
                else:
                    still_open.append(trade)

        open_trades = still_open

        # 2) Open new trade if signal != 0
        sig = signals[i]
        c = conf[i]
        vol = atr_norm[i]

        # Skip low-volatility trades
        if vol < atr_norm_threshold:
            continue

        if c < conf_threshold:
            continue
        if sig != 0 and not np.isnan(atr[i]) and atr[i] > 0:
            trade_margin = (price * position_size * contr_size) / lev
            max_allowed_margin = balance * marg_limit
            if used_margin + trade_margin > max_allowed_margin:
                continue
            trade = {
                "entry_time": bar_time,
                "entry_index": i,
                "entry_price": price,
                "direction": sig,  # 1 long, -1 short
                "size": position_size,
                "atr": atr[i],
                'confidence': c,
                'atr_norm': vol,
                'margin': trade_margin,
            }
            open_trades.append(trade)
            used_margin += trade_margin

        # 3) Track equity
        equity_curve.append({"time": bar_time, "equity": balance})

    # Close remaining trades at last price (optional)
    last_price = prices[-1]
    last_time = index[-1]
    for trade in open_trades:
        if trade["direction"] == 1:
            pnl_points = last_price - trade["entry_price"]
        else:
            pnl_points = trade["entry_price"] - last_price
        pnl = pnl_points * trade["size"]
        balance += pnl

        trade["exit_time"] = last_time
        trade["exit_price"] = last_price
        trade["pnl"] = pnl
        trade["pnl_points"] = pnl_points
        trade["holding_bars"] = len(df) - trade["entry_index"]
        trade_log.append(trade)

    equity_df = pd.DataFrame(equity_curve).set_index("time")
    trades_df = pd.DataFrame(trade_log)

    return balance, equity_df, trades_df

def print_backtest_summary(final_balance, trades_df, initial_balance, tf):
    timeframe_map = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440
    }

    tf = timeframe_map.get(tf)

    n_trades = len(trades_df)
    n_long = (trades_df["direction"] == 1).sum()
    n_short = (trades_df["direction"] == -1).sum()
    n_wins = (trades_df["pnl"] > 0).sum()
    n_losses = (trades_df["pnl"] < 0).sum()
    avg_hold = trades_df["holding_bars"].mean() if n_trades > 0 else 0
    long_winning = ((trades_df["direction"] == 1) & (trades_df["pnl"] > 0)).sum()
    short_winning = ((trades_df["direction"] == -1) & (trades_df["pnl"] > 0)).sum()

    print("\n=== Backtest Summary ===")
    print(f"Initial balance: {initial_balance:.2f}")
    print(f"Final balance:   {final_balance:.2f}")
    print(f"Net PnL:         {final_balance - initial_balance:.2f} ({((final_balance - initial_balance) / initial_balance)*100:.2f}%)")
    print(f"Total trades:    {n_trades}")
    print(f"  Long trades:   {n_long} ({long_winning} won - {long_winning / n_long * 100:.2f}%)")
    print(f"  Short trades:  {n_short} ({short_winning} won - {short_winning / n_short * 100:.2f}%)")
    print(f"  Wins:          {n_wins} ({(n_wins / n_trades)*100:.2f}%)")
    print(f"  Losses:        {n_losses}")
    print(f"Avg holding bars:{avg_hold:.2f} ({int((avg_hold*tf)//60):02d}:{int((avg_hold*tf)%60):02d})")
