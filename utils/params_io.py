import json
from pathlib import Path

PARAMS_FILE = Path(__file__).resolve().parent / "best_params.json"


def save_best_params(params: dict):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Saved optimized parameters to {PARAMS_FILE}")


def load_best_params() -> dict:
    if not PARAMS_FILE.exists():
        print("No optimized params found, using defaults.")
        return {}
    with open(PARAMS_FILE, "r") as f:
        return json.load(f)
