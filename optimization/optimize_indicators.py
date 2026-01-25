import optuna
from data_loader.mt5_loader import load_data
from optimization.objective import create_objective
from utils.params_io import save_best_params
from utils.config import TIMEFRAME, DAYS, SYMBOL, START_DATE, END_DATE, NUMBER_TRIALS


def run_optimization(symbol, timeframe, days, start_date, end_date, n_trials=50):
    print("Loading MT5 data...")
    df_raw = load_data(symbol, timeframe, days, start_date, end_date)

    print("Preparing objective...")
    objective = create_objective(df_raw)

    print("Starting optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n===== Optimization Complete =====")
    print(f"Best Score (PF): {study.best_value:.4f}")

    # Extract structured params from user_attrs
    best = study.best_trial.user_attrs

    print(f"Best Model: {best['model_name']}")
    print("Indicator Params:", best["indicators"])
    print(f"{best['model_name']} Params:", best.get(best["model_name"], {}))

    # Save structured params
    save_best_params(study)

    return study



if __name__ == "__main__":
    run_optimization(SYMBOL, TIMEFRAME, DAYS, START_DATE, END_DATE, NUMBER_TRIALS)
