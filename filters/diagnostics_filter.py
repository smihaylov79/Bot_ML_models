import pandas as pd

from diagnostics.regime_features import compute_trend_strength


def apply_diagnostics_filter(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out trades based on diagnostics insights.
    Returns a filtered trades DataFrame.
    """
    trades = trades.copy()

    # Time filters
    trades["hour"] = trades["entry_time"].dt.hour
    trades["weekday"] = trades["entry_time"].dt.day_name()

    # Duration filter
    # duration_threshold = 40  # minutes
    # trades = trades[trades["holding_bars"] >= duration_threshold]

    # Hour filter: avoid 3–6
    # trades = trades[~trades["hour"].between(3, 6)]

    # Weekday filter: avoid Monday and Friday
    # trades = trades[~trades["weekday"].isin(["Monday", "Friday"])]

    # Volatility filter: avoid lowest 2 quintiles
    # if "volatility" in trades.columns:
        # vol_q = pd.qcut(trades["volatility"], 5, labels=False, duplicates="drop")
        # trades = trades[vol_q >= 1]

    # Trend filter: avoid near-zero trend zones
    # if "trend_strength" in trades.columns:
    #     trades = trades[trades["trend_strength"].abs() > 1.5]

    return trades


def apply_diagnostics_mask(df):
    mask = (
        (df.index.hour < 3) | (df.index.hour > 6)
    ) & (~df.index.day_name().isin(["Monday", "Friday"])) & \
        ((df["atr"] / df["close"]) >= 0.0004) & \
        (compute_trend_strength(df["close"]).abs() > 2.5)
    return df[mask]


def recompute_equity_from_trades(trades_df, initial_balance):
    """
    Rebuild equity curve using only the filtered trades.
    """
    balance = initial_balance
    equity_curve = []

    # Sort trades by exit_time to simulate chronological PnL accumulation
    trades_df = trades_df.sort_values("exit_time")

    for _, trade in trades_df.iterrows():
        balance += trade["pnl"]
        equity_curve.append({"time": trade["exit_time"], "equity": balance})

    equity_df = pd.DataFrame(equity_curve).set_index("time")
    return balance, equity_df
