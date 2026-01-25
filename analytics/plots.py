import plotly.graph_objects as go
import plotly.express as px


def plot_equity_curve(report):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=report.equity_curve.index,
        y=report.equity_curve.values,
        mode="lines",
        name="Equity"
    ))
    fig.update_layout(title="Equity Curve")
    return fig


def plot_drawdown(report):
    dd = report.drawdown_series

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode="lines",
        name="Drawdown",
        line=dict(color="red")
    ))

    fig.update_layout(
        title="Drawdown (Underwater Curve)",
        xaxis_title="Time",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        template="plotly_white"
    )

    return fig


def plot_pnl_distribution(report):
    trades = report.trades

    fig = px.histogram(
        trades,
        x="profit",
        nbins=50,
        title="PnL Distribution",
        labels={"profit": "Profit per Trade"},
        template="plotly_white"
    )

    fig.update_layout(
        bargap=0.05
    )

    return fig


def plot_duration_distribution(report):
    trades = report.trades

    fig = px.histogram(
        trades,
        x="duration",
        nbins=50,
        title="Trade Duration Distribution (minutes)",
        labels={"duration": "Duration (minutes)"},
        template="plotly_white"
    )

    fig.update_layout(
        bargap=0.05
    )

    return fig
