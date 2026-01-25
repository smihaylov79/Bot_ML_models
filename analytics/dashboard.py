from .performance import generate_performance_report
from .plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_pnl_distribution,
    plot_duration_distribution
)


def generate_dashboard(trades, equity_curve, output_path="dashboard.html"):
    report = generate_performance_report(trades, equity_curve)

    # Generate plots
    fig_equity = plot_equity_curve(report)
    fig_drawdown = plot_drawdown(report)
    fig_pnl = plot_pnl_distribution(report)
    fig_duration = plot_duration_distribution(report)

    # Build HTML
    html = f"""
    <html>
    <head>
        <title>Trading Performance Dashboard</title>
    </head>
    <body>
        <h1>Trading Performance Dashboard</h1>

        <h2>Summary</h2>
        <p>Total Return: {report.total_return:.2%}</p>
        <p>Max Drawdown: {report.max_drawdown:.2%}</p>
        <p>Sharpe Ratio: {report.sharpe:.2f}</p>
        <p>Win Rate: {report.win_rate:.2%}</p>
        <p>Profit Factor: {report.profit_factor:.2f}</p>

        <h2>Equity Curve</h2>
        {fig_equity.to_html()}

        <h2>Drawdown</h2>
        {fig_drawdown.to_html()}

        <h2>PnL Distribution</h2>
        {fig_pnl.to_html()}

        <h2>Duration Distribution</h2>
        {fig_duration.to_html()}
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
