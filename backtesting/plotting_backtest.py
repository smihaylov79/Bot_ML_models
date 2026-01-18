import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def plot_equity_and_trades(df, equity_df, trades_df, title="Backtest"):
    fig, (ax_price, ax_eq) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Price
    ax_price.plot(df.index, df["close"], label="Close", color="black", linewidth=1)

    # Mark trades
    for _, tr in trades_df.iterrows():
        color = "green" if tr["direction"] == 1 else "red"
        ax_price.scatter(tr["entry_time"], tr["entry_price"],
                         marker="^" if tr["direction"] == 1 else "v",
                         color=color, s=30)
        ax_price.scatter(tr["exit_time"], tr["exit_price"],
                         marker="x", color=color, s=30)

    ax_price.set_title(f"{title} - Price & Trades")
    ax_price.legend()

    # Equity
    ax_eq.plot(equity_df.index, equity_df["equity"], label="Equity", color="blue")
    ax_eq.set_title("Equity Curve")
    ax_eq.legend()

    plt.tight_layout()
    plt.show()
