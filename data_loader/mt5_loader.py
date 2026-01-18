import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from utils.config import SYMBOL, TIMEFRAME, LOCAL_TZ


def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    print("MT5 initialized successfully.")


def load_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=None, start_date=None, end_date=None):
    initialize_mt5()
    # utc_to = datetime.utcnow()
    # utc_from = utc_to - timedelta(days=days)

    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }

    tf = timeframe_map.get(timeframe)
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # --- NEW LOGIC --- # If start_date and end_date are provided → use them
    if start_date is not None and end_date is not None:
        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
    else:
        if days is None:
            raise ValueError("Either days or (start_date and end_date) must be provided.")
        utc_to = datetime.utcnow()
        utc_from = utc_to - timedelta(days=days)
        rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)

    # rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
    if rates is None:
        raise RuntimeError(f"Failed to load data for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert(LOCAL_TZ)
    return df


def get_latest_tick(symbol=SYMBOL):
    initialize_mt5()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Failed to get latest tick for {symbol}: {mt5.last_error()}")
    return tick



