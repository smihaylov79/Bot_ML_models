import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_bar(series: pd.Series, title: str, ylabel: str = "Win rate"):
    plt.figure(figsize=(8, 4))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()


def plot_pattern_summary(
    by_hour: pd.Series,
    by_weekday: pd.Series,
    by_duration: pd.Series,
    by_direction: pd.Series,
    by_vol: pd.Series | None = None,
    by_trend: pd.Series | None = None,
):
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    by_hour.plot(kind="bar")
    plt.title("Win rate by hour")
    plt.xticks(rotation=0)

    plt.subplot(2, 3, 2)
    by_weekday.plot(kind="bar")
    plt.title("Win rate by weekday")
    plt.xticks(rotation=45)

    plt.subplot(2, 3, 3)
    by_direction.plot(kind="bar")
    plt.title("Win rate by direction")
    plt.xticks(rotation=0)

    plt.subplot(2, 3, 4)
    by_duration.plot(kind="bar")
    plt.title("Win rate by duration quintile")
    plt.xticks(rotation=45)

    if by_vol is not None and len(by_vol) > 0:
        plt.subplot(2, 3, 5)
        by_vol.plot(kind="bar")
        plt.title("Win rate by volatility quintile")
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "No volatility data", ha="center")

    if by_trend is not None and len(by_trend) > 0:
        plt.subplot(2, 3, 6)
        by_trend.plot(kind="bar")
        plt.title("Win rate by trend quintile")
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "No trend data", ha="center")

    plt.tight_layout()
    plt.show()
