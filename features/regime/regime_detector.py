import numpy as np
import pandas as pd

class RegimeDetector:
    def __init__(
        self,
        bb_expansion_percentile=70.0,
        bb_crush_percentile=30.0,
        atr_crush_percentile=30.0,
        ema_slope_threshold=0.0,
    ):
        self.bb_expansion_percentile = bb_expansion_percentile
        self.bb_crush_percentile = bb_crush_percentile
        self.atr_crush_percentile = atr_crush_percentile
        self.ema_slope_threshold = ema_slope_threshold

    def fit(self, df):
        self.bb_expansion_level = np.percentile(df["bb_width"], self.bb_expansion_percentile)
        self.bb_crush_level = np.percentile(df["bb_width"], self.bb_crush_percentile)
        self.atr_crush_level = np.percentile(df["atr"], self.atr_crush_percentile)
        return self

    def transform(self, df):
        regimes = pd.Series(index=df.index, dtype="int64")

        bb = df["bb_width"]
        atr = df["atr"]

        # EMA slope proxy for trend strength
        ema_slope = df["ema_slope_fast"] - df["ema_slope_slow"]

        # 0: no-trade / volatility crush
        cond_crush = (bb <= self.bb_crush_level) & (atr <= self.atr_crush_level)

        # 1: trend
        cond_trend = ema_slope.abs() >= self.ema_slope_threshold

        # 3: volatility expansion
        cond_expansion = bb >= self.bb_expansion_level

        regimes[cond_crush] = 0
        regimes[cond_trend & ~cond_crush] = 1
        regimes[cond_expansion & ~cond_crush & ~cond_trend] = 3

        # 2: range (default)
        regimes[regimes.isna()] = 2

        return regimes.astype(int)

    def fit_transform(self, df):
        return self.fit(df).transform(df)
