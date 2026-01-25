# analytics/performance.py

from dataclasses import dataclass
import pandas as pd
import numpy as np
from .metrics import (
    compute_total_return, compute_cagr, compute_max_drawdown,
    compute_sharpe, compute_sortino, compute_profit_factor
)

@dataclass
class PerformanceReport:
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    sortino: float
    volatility: float
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_r: float
    exposure_time: float

    equity_curve: pd.Series
    trades: pd.DataFrame
    drawdown_series: pd.Series


def generate_performance_report(trades: pd.DataFrame, equity: pd.Series):
    # Ensure equity is a Series
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]

    returns = equity.pct_change().dropna()

    total_return = compute_total_return(equity)
    cagr = compute_cagr(equity, days=len(equity))
    max_dd, dd_series = compute_max_drawdown(equity)
    sharpe = compute_sharpe(returns)
    sortino = compute_sortino(returns)
    volatility = returns.std() * np.sqrt(252)
    win_rate = (trades["profit"] > 0).mean()
    profit_factor = compute_profit_factor(trades)
    expectancy = trades["profit"].mean()

    # If you track risk per trade, avg_r becomes meaningful
    avg_r = np.nan

    # Exposure time = total duration of trades / total time span
    exposure_time = trades["duration"].sum() / (len(equity) * 24 * 60)

    return PerformanceReport(
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        volatility=volatility,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_r=avg_r,
        exposure_time=exposure_time,
        equity_curve=equity,
        trades=trades,
        drawdown_series=dd_series
    )
