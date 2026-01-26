import pandas as pd
import numpy as np

from features.indicators import (
    ema, rsi, atr, volatility_std, momentum,
    stochastic_oscillator, macd, bollinger_bands,
    candle_components
)


def add_basic_price_features(df):
    close = df["close"]
    df["return_1"] = close.pct_change(1)
    df["return_2"] = close.pct_change(2)
    df["return_5"] = close.pct_change(5)
    df["high_low_range"] = df["high"] - df["low"]
    df["close_open"] = df["close"] - df["open"]
    df["rolling_return_3"] = close.pct_change(3)
    df["rolling_return_6"] = close.pct_change(6)
    df["rolling_return_12"] = close.pct_change(12)
    return df


def add_volatility_features(df, params):
    close = df["close"]
    df["atr"] = atr(df["high"], df["low"], close, params.get("atr_window", 14))
    df["vol_std"] = volatility_std(close.pct_change(), params.get("vol_window", 20))
    return df


def add_momentum_features(df, params):
    close = df["close"]
    df["rsi"] = rsi(close, params.get("rsi_window", 14))
    df["momentum"] = momentum(close, params.get("momentum_window", 10))

    k, d = stochastic_oscillator(
        df["high"], df["low"], close,
        params.get("stoch_k", 14),
        params.get("stoch_d", 3)
    )
    df["stoch_k"] = k
    df["stoch_d"] = d
    return df


def add_trend_features(df, params):
    close = df["close"]
    df["ema_fast"] = ema(close, params.get("ema_fast", 10))
    df["ema_slow"] = ema(close, params.get("ema_slow", 20))
    df["ema_slope_fast"] = df["ema_fast"].diff()
    df["ema_slope_slow"] = df["ema_slow"].diff()
    return df


def add_optional_advanced_features(df, params):
    close = df["close"]

    _, _, hist = macd(
        close,
        params.get("macd_fast", 12),
        params.get("macd_slow", 26),
        params.get("macd_signal", 9)
    )
    df["macd_hist"] = hist

    lower, mid, upper = bollinger_bands(close, params.get("bb_window", 20))
    df["bb_pos"] = (close - mid) / (upper - lower + 1e-9)
    df["bb_width"] = (upper - lower) / (mid.replace(0, np.nan))

    uw, lw, body, ratio = candle_components(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["wick_upper"] = uw
    df["wick_lower"] = lw
    df["body_size"] = body
    df["body_ratio"] = ratio

    return df


# def add_target(df, future_n=2, threshold=0.0):
#     future_price = df["close"].shift(-future_n)
#     ret = (future_price - df["close"]) / df["close"]
#     df["target"] = 0
#     df.loc[ret > threshold, "target"] = 1
#     df.loc[ret < -threshold, "target"] = -1
#     return df


def add_regime_features(df):
    # --- TIME REGIMES ---
    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek

    # --- VOLATILITY REGIMES ---
    df["atr_norm"] = df["atr"] / df["close"]
    df["vol_compression"] = df["atr"] / df["atr"].rolling(20).mean()
    df["vol_shock"] = df["atr"].pct_change()

    # --- TREND REGIMES ---
    df["ma_fast"] = ema(df["close"], 10)
    df["ma_slow"] = ema(df["close"], 30)
    df["trend_strength"] = df["ma_fast"] - df["ma_slow"]
    df["trend_slope"] = df["trend_strength"].diff()
    df["ma_dist"] = (df["close"] - df["ma_slow"]) / df["ma_slow"]

    # --- BOLLINGER REGIMES ---
    df["bb_width_norm"] = df["bb_width"] / df["close"]
    df["bb_squeeze"] = (
        df["bb_width_norm"] < df["bb_width_norm"].rolling(100).quantile(0.2)
    ).astype(int)

    # --- RETURN REGIMES ---
    df["ret_abs"] = df["return_1"].abs()
    df["ret_rolling_std"] = df["return_1"].rolling(20).std()
    df["ret_zscore"] = (
        df["return_1"] - df["return_1"].rolling(20).mean()
    ) / df["return_1"].rolling(20).std()

    return df


def add_tp_sl_target(df, sl_mult=2.0, tp_mult=2.0, future_n=20):
    df = df.copy()

    # Compute SL/TP levels
    sl_price = df["close"] - df["atr"] * sl_mult
    tp_price = df["close"] + df["atr"] * tp_mult

    # Correct future window alignment
    future_high = df["high"].rolling(future_n).max().shift(-future_n)
    future_low  = df["low"].rolling(future_n).min().shift(-future_n)

    # Initialize target
    df["target"] = 0

    df.loc[future_high >= tp_price, "target"] = 1
    df.loc[future_low  <= sl_price, "target"] = -1

    return df


def build_features(df, params=None, future_n=20, sl_mult=2.0, tp_mult=2.0):
    df = df.copy()
    params = params or {}

    # 1. Add all features
    df = add_basic_price_features(df)
    df = add_volatility_features(df, params)
    df = add_momentum_features(df, params)
    df = add_trend_features(df, params)
    df = add_optional_advanced_features(df, params)
    df = add_regime_features(df)

    # 2. Drop NaNs BEFORE computing target
    df = df.dropna().copy()

    # 3. Compute target AFTER features
    df = add_tp_sl_target(df, sl_mult=sl_mult, tp_mult=tp_mult, future_n=future_n)

    # 4. Drop rows where target is NaN (end of dataset)
    df = df.dropna(subset=["target"])

    return df


