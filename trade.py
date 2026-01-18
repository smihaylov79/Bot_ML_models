import time
from pathlib import Path
import joblib
import MetaTrader5 as mt5
import numpy as np

from data_loader.mt5_loader import load_data  # your load_data
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


# def load_active_model():
#     model_path = Path("models/active_model.pkl")
#     if not model_path.exists():
#         raise FileNotFoundError("No active_model.pkl found.")
#     model = joblib.load(model_path)
#     print("Loaded model:", model.__class__.__name__)
#     return model


def calc_required_margin(symbol, volume):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info failed: {mt5.last_error()}")

    # If broker provides margin_initial → use it
    if info.margin_initial > 0:
        return info.margin_initial * volume

    # Otherwise fallback to margin_rate
    if info.margin_rate > 0:
        # margin = price * contract_size * volume * margin_rate
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if tick else info.last
        return price * info.trade_contract_size * volume * info.margin_rate

    raise RuntimeError("No margin info available for this symbol.")


def has_enough_margin(symbol, volume):
    acc  = mt5.account_info()
    if acc is None:
        raise RuntimeError(f"account_info failed: {mt5.last_error()}")

    # very simple approximation, same as backtest:
    required = calc_required_margin(symbol, volume)
    # margin_for_trade = (price * position_size * CONTRACT_SIZE) / LEVERAGE
    max_allowed_margin = acc.equity * MARGIN_LIMIT

    # used_margin is info.margin
    return acc.margin + required <= max_allowed_margin


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
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order failed:", result.retcode, result.comment)
    else:
        print(f"Order placed: {direction}, price={price}, sl={sl}, tp={tp}")


def live_trading_loop(poll_seconds=300, lookback_days=5):
    initialize_mt5()
    model = load_model()

    last_bar_time = None

    while True:
        try:
            df = load_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=lookback_days)
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

            if not has_enough_margin(price, POSITION_SIZE):
                print(bar_time, "Not enough margin, skipping trade.")
                time.sleep(poll_seconds)
                continue

            # execute
            place_order(SYMBOL, sig, price, atr_value)

        except Exception as e:
            print("Error in live loop:", e)

        time.sleep(poll_seconds)
