# optimization/run_optimization.py
import optuna

from data_loader.mt5_loader import load_data
from optimization.objective import create_objective
from utils.config import SYMBOL, DAYS, START_DATE, END_DATE, NUMBER_TRIALS, TIMEFRAME
from utils.params_io import save_best_params


def run_optimization(symbol, timeframe, days, start_date, end_date, n_trials=50):
    print("Loading MT5 data...")
    df_raw = load_data(symbol, timeframe, days, start_date, end_date)

    print("Preparing objective...")
    objective = create_objective(df_raw)

    print("Starting optimization...")
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{symbol}_{timeframe}_opt",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    print("\n===== Optimization Complete =====")
    print(f"Best Score (combined PF): {study.best_value:.4f}")

    best = study.best_trial.user_attrs

    print(f"Best Model: {best['model_name']}")
    print(f"Walk-forward PF: {best.get('wf_pf', 0.0):.4f}")
    print(f"Unseen PF: {best.get('unseen_pf', 0.0):.4f}")
    print("Indicator Params:", best["indicators"])
    print(f"{best['model_name']} Params:", best.get(best["model_name"], {}))

    save_best_params(study)

    return study




if __name__ == "__main__":
    run_optimization(SYMBOL, TIMEFRAME, DAYS, START_DATE, END_DATE, NUMBER_TRIALS)
