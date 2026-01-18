import time
from pathlib import Path
import joblib
import MetaTrader5 as mt5
import numpy as np

from data_loader.mt5_loader import load_data, load_live_bars  # your load_data
from features.feature_engineering import build_features
from backtesting.backtest_engine import generate_signals, load_model  # your generate_signals
from utils.config import (
    SYMBOL, TIMEFRAME,
    SL_ATR_MULT, TP_ATR_MULT,
    CONF_THRESHOLD, ATR_THRESHOLD,
    CONTRACT_SIZE, LEVERAGE, MARGIN_LIMIT,
    POSITION_SIZE,
)


def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    print("MT5 initialized.")


def calc_required_margin(symbol, volume, direction):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick failed: {mt5.last_error()}")

    price = tick.ask if direction == mt5.ORDER_TYPE_BUY else tick.bid

    margin = mt5.order_calc_margin(direction, symbol, volume, price)
    if margin is None:
        raise RuntimeError(f"order_calc_margin failed: {mt5.last_error()}")

    return margin


def has_enough_margin(symbol, volume, direction):
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError(f"account_info failed: {mt5.last_error()}")

    required = calc_required_margin(symbol, volume, direction)
    max_allowed = acc.equity * MARGIN_LIMIT

    return acc.margin + required <= max_allowed


def place_order(symbol, direction, price, atr_value):
    volume = POSITION_SIZE
    sl = None
    tp = None

    if direction == 1:  # long
        sl = price - SL_ATR_MULT * atr_value
        tp = price + TP_ATR_MULT * atr_value
        order_type = mt5.ORDER_TYPE_BUY
    else:  # short
        sl = price + SL_ATR_MULT * atr_value
        tp = price - TP_ATR_MULT * atr_value
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "ML_live",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result is None:
        print("order_send returned None:", mt5.last_error())
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order failed:", result.retcode, result.comment)
        return False
    else:
        print(f"Order placed: {direction}, price={price}, sl={sl}, tp={tp}")
        return True


def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol '{symbol}' not found in MT5.")

    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol '{symbol}'.")
    print(f"Symbol '{symbol}' selected.")


def live_trading_loop(poll_seconds=300, lookback_days=5):
    initialize_mt5()
    ensure_symbol(SYMBOL)
    model = load_model()

    last_bar_time = None

    while True:
        try:
            df = load_live_bars(SYMBOL, TIMEFRAME, n_bars=500)
            df_feat = build_features(df)
            if df_feat.empty:
                time.sleep(poll_seconds)
                continue

            # work on last completed bar
            last_row = df_feat.iloc[-1:]
            bar_time = last_row.index[-1]

            # only act on new bar
            if last_bar_time is not None and bar_time <= last_bar_time:
                time.sleep(poll_seconds)
                continue

            last_bar_time = bar_time

            signals, conf = generate_signals(model, df_feat)
            sig = signals[-1]
            c = conf[-1]

            price = df["close"].iloc[-1]
            atr_value = df_feat["atr"].iloc[-1]
            atr_norm = atr_value / price

            # filters (same as backtest)
            if atr_norm < ATR_THRESHOLD:
                print(bar_time, "ATR filter, no trade.")
                time.sleep(poll_seconds)
                continue

            if c < CONF_THRESHOLD or sig == 0 or np.isnan(atr_value) or atr_value <= 0:
                print(bar_time, "No valid signal.")
                time.sleep(poll_seconds)
                continue

            direction = mt5.ORDER_TYPE_BUY if sig == 1 else mt5.ORDER_TYPE_SELL

            if not has_enough_margin(SYMBOL, POSITION_SIZE, direction):
                print(bar_time, "Not enough margin, skipping trade.")
                time.sleep(poll_seconds)
                continue

            # execute
            place_order(SYMBOL, sig, price, atr_value)

        except Exception as e:
            print("Error in live loop:", e)

        time.sleep(poll_seconds)
