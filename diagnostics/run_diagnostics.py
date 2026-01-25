from datetime import datetime, timedelta

from data_loader.account_hystory import load_raw_account_history, normalize_deals_to_trades
from data_loader.mt5_loader import load_data
from diagnostics.pattern_analysis import (
    add_basic_labels,
    analyze_by_hour,
    analyze_by_weekday,
    analyze_by_duration_quintile,
    analyze_by_direction,
    analyze_by_volatility_quintile,
    analyze_by_trend_quintile,
)
from diagnostics.regime_features import compute_volatility, compute_trend_strength
from diagnostics.viz import plot_pattern_summary

# trades: your normalized trades DataFrame
# price_df: DataFrame with 'close' indexed by datetime
end = datetime.now()
start = end - timedelta(days=7)
symbol = '[SP500]'
timeframe = 'M5'
raw = load_raw_account_history(start, end)
trades = normalize_deals_to_trades(raw)

trades = add_basic_labels(trades)
price_df = load_data(symbol, timeframe, start_date=start, end_date=end)
price_df = price_df.tz_convert("UTC")
trades["open_time"] = trades["open_time"].dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
trades["close_time"] = trades["close_time"].dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")

#
#
#
# print(price_df.head())
# print(price_df.tail())
# print(price_df.index.dtype)
# print(price_df.isna().sum())
# print(len(price_df))

price = price_df["close"]
vol = compute_volatility(price)
trend = compute_trend_strength(price)

by_hour = analyze_by_hour(trades)
by_weekday = analyze_by_weekday(trades)
by_duration = analyze_by_duration_quintile(trades)
by_direction = analyze_by_direction(trades)

# print("Volatility head:")
# print(vol.head())
# print("Volatility NaNs:", vol.isna().sum())


by_vol = analyze_by_volatility_quintile(trades, vol)
by_trend = analyze_by_trend_quintile(trades, trend)


plot_pattern_summary(by_hour, by_weekday, by_duration, by_direction, by_vol, by_trend)
