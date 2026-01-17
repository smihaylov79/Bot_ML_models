import pandas as pd

# Mapping for training (model-friendly)
ENCODE_MAP = {-1: 0, 0: 1, 1: 2}

# Mapping for predictions (trading-friendly)
DECODE_MAP = {0: -1, 1: 0, 2: 1}


def encode_target(y):
    """
    Convert trading labels (-1,0,1) to model labels (0,1,2).
    Works with pandas Series or numpy arrays.
    """
    return pd.Series(y).map(ENCODE_MAP).values


def decode_target(y):
    """
    Convert model predictions (0,1,2) back to trading labels (-1,0,1).
    """
    return pd.Series(y).map(DECODE_MAP).values
