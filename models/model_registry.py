from models.xgb_model import train_xgb
from models.rf_model import train_rf
from models.lgbm_model import train_lgbm


MODEL_REGISTRY = {
    "xgb": train_xgb,
    "rf": train_rf,
    "lgbm": train_lgbm,
}


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]


def list_models():
    return list(MODEL_REGISTRY.keys())


def train_all_models(train_df, model_params: dict):
    """
    model_params: dict of dicts
    {
        "xgb": {...},
        "rf": {...},
        "lgbm": {...}
    }
    """
    trained = {}

    for name, train_fn in MODEL_REGISTRY.items():
        params = model_params.get(name, {})
        trained[name] = train_fn(train_df, params)

    return trained
