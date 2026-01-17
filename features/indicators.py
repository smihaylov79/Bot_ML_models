import numpy as np
import pandas as pd


def ema(series: pd.Series, window: int = 14) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/window, adjust=False).mean()

    rs = gain_ema / (loss_ema + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/window, adjust=False).mean()


def volatility_std(returns: pd.Series, window: int = 14) -> pd.Series:
    return returns.rolling(window=window).std()


def momentum(series: pd.Series, window: int = 10) -> pd.Series:
    return series - series.shift(window)


def stochastic_oscillator(high: pd.Series,
                          low: pd.Series,
                          close: pd.Series,
                          k_window: int = 14,
                          d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = k.rolling(window=d_window).mean()
    return k, d


def macd(series: pd.Series,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series,
                    window: int = 20,
                    num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return lower, ma, upper


def candle_components(open_: pd.Series,
                      high: pd.Series,
                      low: pd.Series,
                      close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    body = (close - open_).abs()
    range_ = (high - low).replace(0, np.nan)
    body_ratio = body / range_

    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    return upper_wick, lower_wick, body, body_ratio
