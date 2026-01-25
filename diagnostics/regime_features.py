import pandas as pd


def compute_volatility(price: pd.Series, window: int = 14) -> pd.Series:
    """
    Simple volatility proxy: rolling standard deviation of returns.
    """
    returns = price.pct_change()
    vol = returns.rolling(window, min_periods=3).std()
    return vol


def compute_trend_strength(price: pd.Series, ma_window: int = 50) -> pd.Series:
    """
    Trend proxy: distance from moving average.
    """
    ma = price.rolling(ma_window).mean()
    return price - ma
