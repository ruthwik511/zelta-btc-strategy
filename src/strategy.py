"""
ZeltaHybridStrategy - final strategy for the Zelta BTC/USDT brief (2018-2022).

Two sleeves netted into ONE account (no leverage, |exposure| <= 1x equity):

1) DIP-BUY / SPIKE-FADE sleeve  (built from the friend's 5-minute features)
   * LONG  when price is more than 7% below the 48-bar (4h) high
           ("Breakout Distance"/"Rolling High-Low Extremes" feature)
   * SHORT when price is more than 7% above the 48-bar low
   * falling-knife filter: skip a long if the last 1h fell more than 5%
     (skip a short if the last 1h rose more than 5%)
   * hold for 12 bars (1h) after the latest trigger; exit early once price is
     back to its 48-bar EMA ("EMA-48 Deviation" feature crosses 0)
   * size = min(1, 2.0 / realized 24h volatility) -> smaller bets in panics

2) TREND sleeve (used whenever sleeve 1 is flat)
   * equal-weight vote of 5 classic trend systems (each +1/-1/0):
       EMA 50/200 on 4h, EMA 10/30 on 1d, Donchian 55/20 on 12h,
       Supertrend(20, 4) on 12h, Donchian 40/20 on 2h
   * exposure = 0.4 x vote x min(1, 60% / realized 7-day volatility)

Execution: signals use completed 5m candles only and trade at the NEXT 5m
open, 0.10% cost per executed side, exposure re-sized only when it flips,
goes flat, or drifts by 25% of equity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zelta_engine import (EngineConfig, align_to, ema, resample_ohlcv,
                          run_exposure_backtest, supertrend)


@dataclass(frozen=True)
class StrategyParams:
    # dip-buy / spike-fade sleeve
    window: int = 48                # 48 x 5m = 4h, same window as the feature table
    dip_threshold: float = 0.07     # long when close < 48-bar high * (1 - 7%)
    spike_threshold: float = 0.07   # short when close > 48-bar low * (1 + 7%)
    knife_filter: float = 0.05      # skip entries during a >5% 1h free-fall / melt-up
    hold_bars: int = 12             # stay in for 1h after the latest trigger
    exit_ema_dev: float = 0.0       # take profit when price is back at the 48-EMA
    mr_vol_target: float = 2.0      # size = min(1, 200% / realized 24h vol)
    # trend sleeve
    trend_weight: float = 0.4
    trend_vol_target: float = 0.6
    # execution
    fee: float = 0.001
    min_rebalance: float = 0.25
    initial_capital: float = 100_000.0


class ZeltaHybridStrategy:
    def __init__(self, params: StrategyParams = StrategyParams()):
        self.p = params

    # ------------------------------------------------------------------ features
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        p, c = self.p, df["close"]
        f = pd.DataFrame(index=df.index)
        f["dist_high"] = c / df["high"].rolling(p.window).max() - 1
        f["dist_low"] = c / df["low"].rolling(p.window).min() - 1
        f["ema_dev"] = c / c.ewm(span=p.window, adjust=False).mean() - 1
        f["ret_1h"] = c / c.shift(12) - 1
        h1 = resample_ohlcv(df, "1h")
        lr = np.log(h1["close"]).diff()
        f["rv_24h"] = align_to(lr.rolling(24).std() * np.sqrt(24 * 365), df.index, "1h")
        f["rv_7d"] = align_to(lr.rolling(24 * 7).std() * np.sqrt(24 * 365), df.index, "1h")
        return f

    # ------------------------------------------------------------- trend sleeve
    @staticmethod
    def _donchian(d: pd.DataFrame, n: int, m: int) -> pd.Series:
        hi_n = d["high"].rolling(n).max().shift(1).to_numpy()
        lo_n = d["low"].rolling(n).min().shift(1).to_numpy()
        hi_m = d["high"].rolling(m).max().shift(1).to_numpy()
        lo_m = d["low"].rolling(m).min().shift(1).to_numpy()
        c = d["close"].to_numpy()
        pos, p = np.zeros(len(d)), 0
        for i in range(len(d)):
            if p == 0:
                p = 1 if c[i] > hi_n[i] else (-1 if c[i] < lo_n[i] else 0)
            elif p == 1:
                p = -1 if c[i] < lo_n[i] else (0 if c[i] < lo_m[i] else 1)
            else:
                p = 1 if c[i] > hi_n[i] else (0 if c[i] > hi_m[i] else -1)
            pos[i] = p
        return pd.Series(pos, d.index)

    def trend_vote(self, df: pd.DataFrame) -> pd.DataFrame:
        bars = {"2h": resample_ohlcv(df, "2h"), "4h": resample_ohlcv(df, "4h"),
                "12h": resample_ohlcv(df, "12h"), "1D": resample_ohlcv(df, "1D")}
        systems = {
            "ema_4h_50_200": ("4h", np.sign(ema(bars["4h"]["close"], 50) - ema(bars["4h"]["close"], 200))),
            "ema_1d_10_30": ("1D", np.sign(ema(bars["1D"]["close"], 10) - ema(bars["1D"]["close"], 30))),
            "donchian_12h_55_20": ("12h", self._donchian(bars["12h"], 55, 20)),
            "supertrend_12h_20_4": ("12h", supertrend(bars["12h"], 20, 4.0)),
            "donchian_2h_40_20": ("2h", self._donchian(bars["2h"], 40, 20)),
        }
        out = pd.DataFrame({k: align_to(s, df.index, tf).fillna(0.0) for k, (tf, s) in systems.items()})
        out["vote"] = out.mean(axis=1)
        return out

    # ---------------------------------------------------------- dip-buy sleeve
    def dip_sleeve(self, df: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
        p = self.p
        long_trig = (f["dist_high"] < -p.dip_threshold) & ~(f["ret_1h"] < -p.knife_filter)
        short_trig = (f["dist_low"] > p.spike_threshold) & ~(f["ret_1h"] > p.knife_filter)
        lt = long_trig.fillna(False).to_numpy(bool)
        st = short_trig.fillna(False).to_numpy(bool)
        ed = f["ema_dev"].to_numpy()
        size = np.nan_to_num(np.clip(p.mr_vol_target / f["rv_24h"].to_numpy(), 0, 1.0), nan=0.0)
        pos = np.zeros(len(df))
        side, last, sz = 0, -10**9, 0.0
        for i in range(len(pos)):
            if side != 0:
                still = i - last < p.hold_bars
                if still:
                    if side == 1 and not lt[i] and ed[i] >= p.exit_ema_dev:
                        still = False
                    if side == -1 and not st[i] and ed[i] <= -p.exit_ema_dev:
                        still = False
                if (side == 1 and lt[i]) or (side == -1 and st[i]):
                    last, still = i, True
                if not still:
                    side = 0
                    continue
                pos[i] = side * sz
                continue
            if lt[i]:
                side, last, sz = 1, i, size[i]
                pos[i] = sz
            elif st[i]:
                side, last, sz = -1, i, size[i]
                pos[i] = -sz
        return pd.Series(pos, df.index)

    # ---------------------------------------------------------------- combined
    def target_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        f = self.features(df)
        tv = self.trend_vote(df)
        trend = p.trend_weight * tv["vote"] * (p.trend_vol_target / f["rv_7d"]).clip(upper=1.0).fillna(0.0)
        dip = self.dip_sleeve(df, f)
        exposure = dip.where(dip != 0, trend)
        return pd.DataFrame({"exposure": exposure, "dip_sleeve": dip, "trend_sleeve": trend,
                             "trend_vote": tv["vote"]}).join(f)

    def backtest(self, df_full: pd.DataFrame, start: str, end: str, fee: float | None = None,
                 delay_bars: int = 0, signals: pd.DataFrame | None = None):
        """Signals are computed on the full history (so indicators are warm),
        then one account trades only inside [start, end]."""
        sig = self.target_exposure(df_full) if signals is None else signals
        expo = sig["exposure"].shift(delay_bars).fillna(0.0) if delay_bars else sig["exposure"]
        window = df_full.loc[start:end]
        cfg = EngineConfig(fee=self.p.fee if fee is None else fee, initial_capital=self.p.initial_capital)
        return run_exposure_backtest(window, expo, cfg, min_rebalance=self.p.min_rebalance), sig
