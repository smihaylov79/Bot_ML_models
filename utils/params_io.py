import json
from pathlib import Path

PARAMS_FILE = Path(__file__).resolve().parent / "best_params.json"


def save_best_params(study, path=PARAMS_FILE):
    best = study.best_trial.user_attrs

    best_params = {
        "model_name": best["model_name"],
        "indicators": best["indicators"],
        "xgb": best.get("xgb", {}),
        "rf": best.get("rf", {}),
        "lgbm": best.get("lgbm", {})
    }

    import json
    with open(path, "w") as f:
        json.dump(best_params, f, indent=4, ensure_ascii=False)

    print(f"Saved best parameters to {path}")


def load_best_params() -> dict:
    if not PARAMS_FILE.exists():
        print("No optimized params found, using defaults.")
        return {}
    with open(PARAMS_FILE, "r") as f:
        return json.load(f)
