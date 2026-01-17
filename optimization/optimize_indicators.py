import optuna
from data_loader.mt5_loader import load_data
from optimization.objective import create_objective
from utils.params_io import save_best_params
from utils.config import TIMEFRAME, DAYS, SYMBOL


def run_optimization(symbol, timeframe, days, n_trials=50):
    print("Loading MT5 data...")
    df_raw = load_data(symbol, timeframe, days)

    print("Preparing objective...")
    objective = create_objective(df_raw)

    print("Starting optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n===== Optimization Complete =====")
    print("Best Score:", study.best_value)
    print("Best Params:", study.best_params)

    save_best_params(study.best_params)

    return study


if __name__ == "__main__":
    run_optimization(SYMBOL, TIMEFRAME, DAYS, n_trials=20)
