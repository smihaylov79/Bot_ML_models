from datetime import datetime, timedelta

from analytics.dashboard import generate_dashboard
from analytics.utils import build_equity_curve
from data_loader.account_hystory import load_raw_account_history, normalize_deals_to_trades, get_starting_balance

end = datetime.now()
start = end - timedelta(days=7)

raw = load_raw_account_history(start, end)
trades = normalize_deals_to_trades(raw)
# print(trades.head())
# print(len(trades), "closed trades reconstructed")

starting_balance = get_starting_balance(start, end)  # or read from MT5
equity_curve = build_equity_curve(trades, starting_balance)

# print(equity_curve.head())
# print(equity_curve.tail())

from analytics.performance import generate_performance_report

report = generate_performance_report(trades, equity_curve)

# print(report)
generate_dashboard(trades, equity_curve)

