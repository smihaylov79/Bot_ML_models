import numpy as np


def compute_total_return(equity):
    return equity.iloc[-1] / equity.iloc[0] - 1


def compute_cagr(equity, days):
    years = days / 365
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def compute_max_drawdown(equity):
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return dd.min(), dd


def compute_sharpe(returns, risk_free=0.0):
    return (returns.mean() - risk_free) / returns.std() * np.sqrt(252)


def compute_sortino(returns, risk_free=0.0):
    downside = returns[returns < 0].std()
    return (returns.mean() - risk_free) / downside * np.sqrt(252)


def compute_profit_factor(trades):
    gross_profit = trades[trades["profit"] > 0]["profit"].sum()
    gross_loss = abs(trades[trades["profit"] < 0]["profit"].sum())

    if gross_loss == 0:
        return float("inf")

    return gross_profit / gross_loss
