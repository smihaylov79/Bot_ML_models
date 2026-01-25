from analytics.performance import PerformanceReport


def generate_weekly_report(report: PerformanceReport):
    return {
        "total_return": report.total_return,
        "max_drawdown": report.max_drawdown,
        "sharpe": report.sharpe,
        "win_rate": report.win_rate,
        "profit_factor": report.profit_factor,
    }
