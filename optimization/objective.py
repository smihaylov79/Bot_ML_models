# optimization/objective.py
from features.feature_engineering import build_features
from evaluation.backtest import walk_forward_backtest
from models.lgbm_model import train_lgbm
from models.rf_model import train_rf
from models.xgb_model import train_xgb
from optimization.search_space import (
    indicator_search_space,
    xgb_search_space,
    rf_search_space,
    lgbm_search_space,
)


def create_objective(df_raw):

    def objective(trial):
        # Force XGBoost only
        model_name = "xgb"

        # Indicator parameters
        indicator_params = indicator_search_space(trial)

        # XGBoost hyperparameters (clean keys!)
        model_params = xgb_search_space(trial)

        # Build features
        df_feat = build_features(
            df_raw,
            params=indicator_params,
            future_n=20,
        )

        # Define model function
        model_fn = lambda train_df: train_xgb(train_df, model_params)

        # Walk-forward + unseen validation
        wf_pf, unseen_pf, fold_stats = walk_forward_backtest(
            model_fn=model_fn,
            df=df_feat,
            train_ratio=0.7,
            step=200,
            conf_threshold=0.0,
            atr_norm_threshold=0.0,
            unseen_ratio=0.1,
        )

        # Combined score
        score = 0.7 * wf_pf + 0.3 * unseen_pf

        # Convert numpy keys to Python ints for JSON serialization
        clean_stats = []
        for fs in fold_stats:
            clean_stats.append({
                "pred_dist": {int(k): int(v) for k, v in fs["pred_dist"].items()},
                "actual_dist": {int(k): int(v) for k, v in fs["actual_dist"].items()},
                "mean_conf": float(fs["mean_conf"]),
                "max_conf": float(fs["max_conf"]),
                "min_conf": float(fs["min_conf"]),
                "correct_1": int(fs["correct_1"]),
                "correct_0": int(fs["correct_0"]),
                "correct_-1": int(fs["correct_-1"]),
                "total_1": int(fs["total_1"]),
                "total_0": int(fs["total_0"]),
                "total_-1": int(fs["total_-1"]),
            })

        # Store clean attributes
        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("indicators", indicator_params)
        trial.set_user_attr("model_params", model_params)   # <-- IMPORTANT
        trial.set_user_attr("wf_pf", wf_pf)
        trial.set_user_attr("unseen_pf", unseen_pf)
        trial.set_user_attr("fold_stats", clean_stats)

        return score

    return objective



# def create_objective(df_raw):
#
#     def objective(trial):
#         # model_name = trial.suggest_categorical("model_name", ["xgb", "rf", "lgbm"])
#         model_name = "xgb"
#
#         indicator_params = indicator_search_space(trial)
#
#         if model_name == "xgb":
#             model_params = xgb_search_space(trial)
#             model_fn = lambda train_df: train_xgb(train_df, model_params)
#         elif model_name == "rf":
#             model_params = rf_search_space(trial)
#             model_fn = lambda train_df: train_rf(train_df, model_params)
#         else:
#             model_params = lgbm_search_space(trial)
#             model_fn = lambda train_df: train_lgbm(train_df, model_params)
#
#         df_feat = build_features(
#             df_raw,
#             params=indicator_params,
#             future_n=20,
#         )
#
#         wf_pf, unseen_pf = walk_forward_backtest(
#             model_fn=model_fn,
#             df=df_feat,
#             train_ratio=0.7,
#             step=200,
#             conf_threshold=0.55,
#             atr_norm_threshold=0.0,
#             unseen_ratio=0.1,
#         )
#
#         score = 0.7 * wf_pf + 0.3 * unseen_pf
#
#         trial.set_user_attr("model_name", model_name)
#         trial.set_user_attr("indicators", indicator_params)
#         trial.set_user_attr("model_params", model_params)
#         trial.set_user_attr("wf_pf", wf_pf)
#         trial.set_user_attr("unseen_pf", unseen_pf)
#
#         return score
#
#     return objective
