from features.feature_engineering import build_features
from evaluation.backtest import walk_forward_backtest
from models.lgbm_model import train_lgbm
from models.rf_model import train_rf
from models.xgb_model import train_xgb
from optimization.search_space import indicator_search_space, xgb_search_space, rf_search_space, \
    lgbm_search_space


def create_objective(df_raw):

    def objective(trial):
        # 1. Choose model
        model_name = trial.suggest_categorical("model_name", ["xgb", "rf", "lgbm"])

        # 2. Sample indicator params
        indicator_params = indicator_search_space(trial)

        # 3. Sample model params
        if model_name == "xgb":
            model_params = xgb_search_space(trial)
            model_fn = lambda train_df: train_xgb(train_df, model_params)

        elif model_name == "rf":
            model_params = rf_search_space(trial)
            model_fn = lambda train_df: train_rf(train_df, model_params)

        else:  # lgbm
            model_params = lgbm_search_space(trial)
            model_fn = lambda train_df: train_lgbm(train_df, model_params)

        # 4. Build features
        df_feat = build_features(
            df_raw,
            params=indicator_params,
            future_n=20,   # TP/SL target window
        )

        # 5. Evaluate using PF
        score = walk_forward_backtest(
            model_fn=model_fn,
            df=df_feat,
        )
        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("indicators", indicator_params)
        trial.set_user_attr(model_name, model_params)

        return score

    return objective

