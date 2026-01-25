# mt5_data/account_history.py

from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd


def init_mt5():
    """Initialize connection to MetaTrader 5 terminal."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed, error code: {mt5.last_error()}")


def load_raw_account_history(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Load raw deals history from MT5 between start and end.
    Returns a raw DataFrame directly from MT5.
    """
    init_mt5()

    deals = mt5.history_deals_get(start, end)
    if deals is None or len(deals) == 0:
        raise RuntimeError(f"No deals returned from MT5 for range {start} → {end}")

    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms")

    return df


def normalize_deals_to_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw MT5 deals into clean closed trades.
    MT5 stores entries and exits separately, so we reconstruct trades.
    """

    # Keep only entry=0 (open) and entry=1 (close)
    opens = df[df["entry"] == 0].copy()
    closes = df[df["entry"] == 1].copy()

    # Sort by time to ensure correct pairing
    opens = opens.sort_values("time")
    closes = closes.sort_values("time")

    trades = []

    # Use position_id to match open and close operations
    for pos_id in opens["position_id"].unique():
        open_deal = opens[opens["position_id"] == pos_id]
        close_deal = closes[closes["position_id"] == pos_id]

        if len(open_deal) == 0 or len(close_deal) == 0:
            continue  # skip incomplete trades

        open_row = open_deal.iloc[0]
        close_row = close_deal.iloc[-1]  # last close event

        open_time = pd.to_datetime(open_row["time"])
        close_time = pd.to_datetime(close_row["time"])

        trade = {"ticket": pos_id, "symbol": open_row["symbol"],
                 "direction": "BUY" if open_row["type"] == 0 else "SELL", "volume": open_row["volume"],
                 "open_time": open_time, "close_time": close_time, "price_open": open_row["price"],
                 "price_close": close_row["price"], "commission": open_row["commission"] + close_row["commission"],
                 "swap": open_row["swap"] + close_row["swap"], "profit": close_row["profit"],
                 "duration": (close_time - open_time).total_seconds() / 60}

        trades.append(trade)

    return pd.DataFrame(trades)


def get_starting_balance(start, end):
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    # Get all balance operations in the period
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return None

    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())

    # Filter only balance operations
    balance_ops = df[df["type"] == mt5.DEAL_TYPE_BALANCE]

    if len(balance_ops) > 0:
        # The first balance operation in the period
        first_balance = balance_ops.sort_values("time").iloc[0]["profit"]
        return first_balance

    # If no balance ops exist, fallback to account_info
    acc = mt5.account_info()
    return acc.balance
