from features.feature_engineering import build_features
from evaluation.backtest import walk_forward_backtest
from models.xgb_model import train_xgb
from optimization.search_space import indicator_search_space, model_search_space


def create_objective(df_raw):

    def objective(trial):
        indicator_params = indicator_search_space(trial)
        model_params = model_search_space(trial)

        df_feat = build_features(
            df_raw,
            params=indicator_params,
            future_n=2,
            threshold=0.0
        )

        score = walk_forward_backtest(
            model_fn=lambda train_df: train_xgb(train_df, model_params),
            df=df_feat,
            metric="f1"
        )

        return score

    return objective
