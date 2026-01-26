# optimization/search_space.py
def indicator_search_space(trial):
    return {
        "rsi_window": trial.suggest_int("rsi_window", 5, 30),
        "ema_fast": trial.suggest_int("ema_fast", 5, 20),
        "ema_slow": trial.suggest_int("ema_slow", 20, 60),
        "atr_window": trial.suggest_int("atr_window", 5, 30),
        "stoch_k": trial.suggest_int("stoch_k", 5, 30),
        "stoch_d": trial.suggest_int("stoch_d", 2, 10),
        "bb_window": trial.suggest_int("bb_window", 10, 40),
        "momentum_window": trial.suggest_int("momentum_window", 3, 20),
        "macd_fast": trial.suggest_int("macd_fast", 5, 20),
        "macd_slow": trial.suggest_int("macd_slow", 20, 40),
        "macd_signal": trial.suggest_int("macd_signal", 5, 15),
    }


def xgb_search_space(trial):
    return {
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 600),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.0, 5.0),
        "min_child_weight": trial.suggest_float("xgb_min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
    }


def rf_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 600),
        "max_depth": trial.suggest_int("rf_max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        "bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False]),
        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
    }


def lgbm_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("lgbm_num_leaves", 20, 200),
        "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lgbm_lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lgbm_lambda_l2", 0.0, 5.0),
        "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 10, 100),
        "is_unbalance": True,
    }
