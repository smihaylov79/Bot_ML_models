import numpy as np
import pandas as pd

from backtesting.backtest_engine import backtest_hedging
from evaluation.metrics import f1_score, accuracy, precision, recall


def walk_forward_backtest(model_fn, df, train_ratio=0.7, step=200):
    scores = []
    n = len(df)

    start_train = int(n * train_ratio)

    for start in range(start_train, n - step, step):
        train_df = df.iloc[:start]
        test_df = df.iloc[start:start + step]

        # 1. Train model
        model = model_fn(train_df)

        # 2. Predict on validation fold
        X_test = test_df.drop(columns=["target"])
        preds = model.predict(X_test)

        # 3. Convert predictions to signals
        signals = preds  # already -1, 0, 1

        # 4. Run a mini backtest on this fold
        try:
            final_balance, _, trades_df = backtest_hedging(
                test_df,
                signals,
                conf=np.ones(len(signals)),
                sl_mult=2,
                tp_mult=2,
                initial_balance=1000,
                position_size=0.1,
                conf_threshold=0.0,
                atr_norm_threshold=0.0,
                contr_size=1,
                lev=1,
                marg_limit=1e9
            )
        except Exception:
            scores.append(0.0)
            continue

        # 5. Compute Profit Factor
        if trades_df.empty or "pnl" not in trades_df.columns:
            scores.append(0.0)
            continue

        gross_profit = trades_df[trades_df.pnl > 0].pnl.sum()
        gross_loss = trades_df[trades_df.pnl < 0].pnl.sum()

        if gross_loss == 0:
            pf = 10.0
        else:
            pf = gross_profit / abs(gross_loss)

        scores.append(pf)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))
