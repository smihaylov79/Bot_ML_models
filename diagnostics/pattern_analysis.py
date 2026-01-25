import pandas as pd

def add_basic_labels(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["is_win"] = (trades["profit"] > 0).astype(int)
    return trades


def analyze_by_hour(trades: pd.DataFrame) -> pd.Series:
    trades = trades.copy()
    trades["hour"] = trades["open_time"].dt.hour
    return trades.groupby("hour")["is_win"].mean().sort_index()


def analyze_by_weekday(trades: pd.DataFrame) -> pd.Series:
    trades = trades.copy()
    trades["weekday"] = trades["open_time"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    s = trades.groupby("weekday")["is_win"].mean()
    return s.reindex(order).dropna()


def analyze_by_duration_quintile(trades: pd.DataFrame) -> pd.Series:
    trades = trades.copy()
    q = pd.qcut(trades["duration"], 5, duplicates="drop")
    return trades.groupby(q, observed=False)["is_win"].mean()


def analyze_by_direction(trades: pd.DataFrame) -> pd.Series:
    return trades.groupby("direction")["is_win"].mean()


def analyze_by_volatility_quintile(trades, vol_series):
    trades = trades.copy()

    trades["volatility"] = vol_series.reindex(
        trades["open_time"],
        method="nearest",
        tolerance=pd.Timedelta("10min")
    ).values

    # Drop NaNs
    valid = trades.dropna(subset=["volatility"])
    if len(valid) < 10:
        return pd.Series(dtype=float)

    try:
        q = pd.qcut(valid["volatility"], 5, duplicates="drop")
        return valid.groupby(q, observed=False)["is_win"].mean()
    except Exception:
        return pd.Series(dtype=float)


def analyze_by_trend_quintile(trades: pd.DataFrame, trend_series: pd.Series) -> pd.Series:
    """
    trend_series: pd.Series indexed by datetime (e.g. price - MA, or slope)
    """
    trades = trades.copy()

    # Ensure timezone alignment
    if trend_series.index.tz is None:
        trend_series = trend_series.tz_localize("UTC")
    else:
        trend_series = trend_series.tz_convert("UTC")

    if trades["open_time"].dt.tz is None:
        trades["open_time"] = trades["open_time"].dt.tz_localize("UTC")
    else:
        trades["open_time"] = trades["open_time"].dt.tz_convert("UTC")

    # Align using nearest timestamp
    trades["trend_strength"] = trend_series.reindex(
        trades["open_time"],
        method="nearest",
        tolerance=pd.Timedelta("10min")
    ).values

    valid = trades.dropna(subset=["trend_strength"])
    if len(valid) < 10:
        return pd.Series(dtype=float)

    try:
        q = pd.qcut(valid["trend_strength"], 5, duplicates="drop")
        return valid.groupby(q, observed=False)["is_win"].mean()
    except Exception:
        return pd.Series(dtype=float)

