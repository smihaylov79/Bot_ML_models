import MetaTrader5 as mt5
import pandas as pd

def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")
    print("MT5 initialized")


