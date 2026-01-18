from backtesting.backtest_engine import prepare_data, load_model, generate_signals, backtest_hedging, \
    print_backtest_summary
from backtesting.plotting_backtest import plot_equity_and_trades
from utils.config import (SYMBOL, TIMEFRAME, START_DATE, END_DATE, INITIAL_BALANCE, POSITION_SIZE,
                          SL_ATR_MULT, TP_ATR_MULT, CONF_THRESHOLD, ATR_THRESHOLD,
                          MARGIN_LIMIT, LEVERAGE, CONTRACT_SIZE)


def backtest_live_real(symbol, timeframe, start_date, end_date, sl_mult, tp_mult, conf_threshold, atr_threshold, contr_size, lev, m_limit):
    df, _ = prepare_data(symbol, timeframe, start_date, end_date)
    model = load_model()
    signals, conf = generate_signals(model, df)

    final_balance, equity_df, trades_df = backtest_hedging(
        df, signals, conf,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        initial_balance=INITIAL_BALANCE,
        position_size=POSITION_SIZE,
        conf_threshold=conf_threshold,
        atr_norm_threshold=atr_threshold,
        contr_size=contr_size,
        lev=lev,
        marg_limit=m_limit
    )

    print_backtest_summary(final_balance, trades_df, INITIAL_BALANCE, TIMEFRAME)
    plot_equity_and_trades(df, equity_df, trades_df,
                           title=f"{symbol} {timeframe}")

if __name__ == "__main__":
    backtest_live_real(SYMBOL, TIMEFRAME, START_DATE, END_DATE, SL_ATR_MULT, TP_ATR_MULT,
                       CONF_THRESHOLD, ATR_THRESHOLD, CONTRACT_SIZE, LEVERAGE, MARGIN_LIMIT)
