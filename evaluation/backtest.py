# evaluation/backtest.py
import numpy as np
import pandas as pd
from typing import Tuple

from backtesting.backtest_engine import backtest_hedging
from utils.target_encoding import decode_target


def _compute_profit_factor(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or "pnl" not in trades_df.columns:
        return 0.0

    gross_profit = trades_df[trades_df.pnl > 0].pnl.sum()
    gross_loss = trades_df[trades_df.pnl < 0].pnl.sum()

    if gross_loss == 0:
        return 10.0
    return float(gross_profit / abs(gross_loss))


def walk_forward_backtest(
    model_fn,
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    step: int = 200,
    conf_threshold: float = 0.55,
    atr_norm_threshold: float = 0.0,
    unseen_ratio: float = 0.1,
):
    from collections import Counter
    import numpy as np

    scores = []
    fold_stats = []   # store diagnostics for Optuna
    n = len(df)

    if n < step * 2:
        return 0.0, 0.0, fold_stats

    start_train = int(n * train_ratio)
    unseen_start = int(n * (1.0 - unseen_ratio))

    # --- Walk-forward folds ---
    for start in range(start_train, unseen_start - step, step):
        train_df = df.iloc[:start]
        test_df = df.iloc[start:start + step]

        model = model_fn(train_df)

        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"].values

        try:
            probs = model.predict_proba(X_test)
        except Exception:
            scores.append(0.0)
            continue

        classes = model.classes_
        max_idx = np.argmax(probs, axis=1)

        # encoded predictions (0,1,2)
        encoded_preds = np.array([classes[i] for i in max_idx])

        # decode to trading labels (-1,0,1)
        signals = decode_target(encoded_preds)

        conf = np.max(probs, axis=1)

        # === DIAGNOSTICS ===
        pred_dist = Counter(signals)
        actual_dist = Counter(y_test)

        mean_conf = float(np.mean(conf))
        max_conf = float(np.max(conf))
        min_conf = float(np.min(conf))

        # Per-class correctness
        correct_1 = int(((signals == 1) & (y_test == 1)).sum())
        correct_0 = int(((signals == 0) & (y_test == 0)).sum())
        correct_m1 = int(((signals == -1) & (y_test == -1)).sum())

        total_1 = int((y_test == 1).sum())
        total_0 = int((y_test == 0).sum())
        total_m1 = int((y_test == -1).sum())

        # Store fold stats
        fold_stats.append({
            "pred_dist": dict(pred_dist),
            "actual_dist": dict(actual_dist),
            "mean_conf": mean_conf,
            "max_conf": max_conf,
            "min_conf": min_conf,
            "correct_1": correct_1,
            "correct_0": correct_0,
            "correct_-1": correct_m1,
            "total_1": total_1,
            "total_0": total_0,
            "total_-1": total_m1,
        })

        # Print diagnostics
        # print("\n=== FOLD DEBUG ===")
        # print("Predicted:", pred_dist)
        # print("Actual:", actual_dist)
        # print("Mean conf:", mean_conf)
        # print("Max conf:", max_conf)
        # print("Min conf:", min_conf)
        # print("Correct 1:", correct_1, "/", total_1)
        # print("Correct 0:", correct_0, "/", total_0)
        # print("Correct -1:", correct_m1, "/", total_m1)

        # Run backtest
        try:
            final_balance, _, trades_df = backtest_hedging(
                test_df,
                signals,
                conf=conf,
                sl_mult=2,
                tp_mult=2,
                initial_balance=1000,
                position_size=0.1,
                conf_threshold=conf_threshold,
                atr_norm_threshold=atr_norm_threshold,
                contr_size=1,
                lev=1,
                marg_limit=1e9,
            )
        except Exception:
            scores.append(0.0)
            continue

        # print("Trades executed:", len(trades_df))

        pf = _compute_profit_factor(trades_df)
        scores.append(pf)

    wf_pf = float(np.mean(scores)) if scores else 0.0

    # --- Unseen validation ---
    if unseen_start <= start_train or unseen_start >= n - step:
        return wf_pf, 0.0, fold_stats

    train_df = df.iloc[:unseen_start]
    test_df = df.iloc[unseen_start:]

    model = model_fn(train_df)
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"].values

    try:
        probs = model.predict_proba(X_test)
        classes = model.classes_
        max_idx = np.argmax(probs, axis=1)
        signals = np.array([classes[i] for i in max_idx])
        conf = np.max(probs, axis=1)

        _, _, trades_df = backtest_hedging(
            test_df,
            signals,
            conf=conf,
            sl_mult=2,
            tp_mult=2,
            initial_balance=1000,
            position_size=0.1,
            conf_threshold=conf_threshold,
            atr_norm_threshold=atr_norm_threshold,
            contr_size=1,
            lev=1,
            marg_limit=1e9,
        )
        unseen_pf = _compute_profit_factor(trades_df)
    except Exception:
        unseen_pf = 0.0

    return wf_pf, unseen_pf, fold_stats


#
# def walk_forward_backtest(
#     model_fn,
#     df: pd.DataFrame,
#     train_ratio: float = 0.7,
#     step: int = 200,
#     conf_threshold: float = 0.55,
#     atr_norm_threshold: float = 0.0,
#     unseen_ratio: float = 0.1,
# ) -> Tuple[float, float]:
#     scores = []
#     n = len(df)
#     if n < step * 2:
#         return 0.0, 0.0
#
#     start_train = int(n * train_ratio)
#     unseen_start = int(n * (1.0 - unseen_ratio))
#
#     # --- Walk-forward folds ---
#     for start in range(start_train, unseen_start - step, step):
#         train_df = df.iloc[:start]
#         test_df = df.iloc[start:start + step]
#
#         model = model_fn(train_df)
#
#         X_test = test_df.drop(columns=["target"])
#         try:
#             probs = model.predict_proba(X_test)
#         except Exception:
#             scores.append(0.0)
#             continue
#
#         classes = model.classes_
#         max_idx = np.argmax(probs, axis=1)
#         signals = np.array([classes[i] for i in max_idx])
#         conf = np.max(probs, axis=1)
#
#         try:
#             final_balance, _, trades_df = backtest_hedging(
#                 test_df,
#                 signals,
#                 conf=conf,
#                 sl_mult=2,
#                 tp_mult=2,
#                 initial_balance=1000,
#                 position_size=0.1,
#                 conf_threshold=conf_threshold,
#                 atr_norm_threshold=atr_norm_threshold,
#                 contr_size=1,
#                 lev=1,
#                 marg_limit=1e9,
#             )
#         except Exception:
#             scores.append(0.0)
#             continue
#
#         pf = _compute_profit_factor(trades_df)
#         scores.append(pf)
#
#     wf_pf = float(np.mean(scores)) if scores else 0.0
#
#     # --- Unseen validation segment ---
#     if unseen_start <= start_train or unseen_start >= n - step:
#         return wf_pf, 0.0
#
#     train_df = df.iloc[:unseen_start]
#     test_df = df.iloc[unseen_start:]
#
#     model = model_fn(train_df)
#     X_test = test_df.drop(columns=["target"])
#
#     try:
#         probs = model.predict_proba(X_test)
#         classes = model.classes_
#         max_idx = np.argmax(probs, axis=1)
#         signals = np.array([classes[i] for i in max_idx])
#         conf = np.max(probs, axis=1)
#
#         _, _, trades_df = backtest_hedging(
#             test_df,
#             signals,
#             conf=conf,
#             sl_mult=2,
#             tp_mult=2,
#             initial_balance=1000,
#             position_size=0.1,
#             conf_threshold=conf_threshold,
#             atr_norm_threshold=atr_norm_threshold,
#             contr_size=1,
#             lev=1,
#             marg_limit=1e9,
#         )
#         unseen_pf = _compute_profit_factor(trades_df)
#     except Exception:
#         unseen_pf = 0.0
#
#     return wf_pf, unseen_pf
