import pandas as pd


def build_equity_curve(trades: pd.DataFrame, starting_balance: float) -> pd.Series:
    """
    Build a step-wise equity curve from closed trades.
    Each step occurs at trade close time.
    """
    # Ensure we only use closed trades and sort them
    trades = trades.sort_values("close_time").copy()

    equity_values = [starting_balance]
    timestamps = [trades["close_time"].iloc[0]]  # start at first close

    for _, row in trades.iterrows():
        equity_values.append(equity_values[-1] + row["profit"])
        timestamps.append(row["close_time"])

    equity = pd.Series(equity_values, index=pd.to_datetime(timestamps))
    equity = equity.sort_index()
    return equity
