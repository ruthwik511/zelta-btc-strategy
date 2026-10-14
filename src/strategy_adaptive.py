"""
ZeltaAdaptiveStrategy - regime-adaptive version of ZeltaHybridStrategy.

Why: after 2022 BTC volatility roughly halved and the market trended up.
The fixed 7% dip trigger almost stopped firing, the trend sleeve kept shorting
pullbacks in a bull market, and its 40% cap left most of the rally on the table.

Three changes, each driven by data the strategy can see at the time:
  1. Volatility-scaled triggers: dip / spike threshold = max(floor, k x sigma_4h),
     sigma_4h = recent std of 5m log returns scaled to a 4-hour horizon.
  2. Constant-risk trend sizing: exposure = vote x min(1, target_vol / 7-day vol)
     x trend_weight, so calm markets get a larger position than wild ones.
  3. Bull / bear regime: when the daily close is above its 200-day EMA, no shorts
     (trend or spike-fade), and optionally a minimum vol-scaled long ("bull floor").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy import StrategyParams, ZeltaHybridStrategy
from zelta_engine import align_to, ema, resample_ohlcv


@dataclass(frozen=True)
class AdaptiveParams(StrategyParams):
    dip_mode: str = "vol"           # "fixed" (original 7%) or "vol" (k x sigma)
    dip_k: float = 4.0
    dip_floor: float = 0.03
    knife_ratio: float = 0.7        # knife filter = 0.7 x threshold (5% / 7% in the original)
    sigma_days: int = 7
    regime_days: int = 200
    shorts_only_in_bear: bool = True
    bull_floor: float = 0.0         # minimum vol-scaled LONG when above the regime EMA
    bear_floor: float = 0.0         # minimum vol-scaled SHORT when below the regime EMA
    pb_drop: float = 0.0            # bull-market pullback buy: enter when close is this far below
    pb_days: int = 14               #   its pb_days high (0 = off), exit near the high again
    pb_exit: float = 0.02           #   (within pb_exit), after pb_hold_days, or if the regime
    pb_hold_days: int = 30          #   turns bear


# Final pick (stage 9): chosen on 2018-2022 + 2023-2024 only; 2025-2026 untouched.
FINAL_PARAMS = AdaptiveParams(
    dip_mode="fixed",            # keep the 7% panic trigger: smaller vol-scaled dips did not pay
    shorts_only_in_bear=True,    # no shorts while BTC is above its 200-day EMA
    regime_days=200,
    trend_weight=0.7, trend_vol_target=0.4,
    pb_drop=0.16, pb_days=7, pb_hold_days=7,   # bull market: buy a 16% fall from the 7-day high
)


class ZeltaAdaptiveStrategy(ZeltaHybridStrategy):
    def __init__(self, params: AdaptiveParams = AdaptiveParams()):
        super().__init__(params)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = super().features(df)
        p = self.p
        r5 = np.log(df["close"]).diff()
        f["sigma_4h"] = r5.rolling(288 * p.sigma_days).std() * np.sqrt(48)
        d1 = resample_ohlcv(df, "1D")
        bull = (d1["close"] > ema(d1["close"], p.regime_days)).astype(float)
        f["bull"] = align_to(bull, df.index, "1D").fillna(0.0)
        if p.pb_drop > 0:
            f[f"dist_high_{p.pb_days}d"] = df["close"] / df["high"].rolling(288 * p.pb_days).max() - 1
        return f

    def thresholds(self, f: pd.DataFrame) -> np.ndarray:
        p = self.p
        if p.dip_mode == "fixed":
            return np.full(len(f), p.dip_threshold)
        return np.maximum(p.dip_floor, p.dip_k * f["sigma_4h"].to_numpy())

    def dip_sleeve(self, df: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
        p = self.p
        thr = self.thresholds(f)
        knife = p.knife_filter if p.dip_mode == "fixed" else p.knife_ratio * thr
        dh, dl, r1 = f["dist_high"].to_numpy(), f["dist_low"].to_numpy(), f["ret_1h"].to_numpy()
        with np.errstate(invalid="ignore"):
            lt = (dh < -thr) & ~(r1 < -knife)
            st = (dl > thr) & ~(r1 > knife)
            if p.shorts_only_in_bear:
                st &= f["bull"].to_numpy() < 0.5
        lt = np.nan_to_num(lt, nan=0).astype(bool)
        st = np.nan_to_num(st, nan=0).astype(bool)
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

    def trend_sleeve(self, f: pd.DataFrame, vote: pd.Series) -> pd.Series:
        p = self.p
        vf = (p.trend_vol_target / f["rv_7d"]).clip(upper=1.0).fillna(0.0)
        trend = p.trend_weight * vote * vf
        bull = f["bull"] > 0.5
        if p.shorts_only_in_bear:
            trend = trend.where(~bull, trend.clip(lower=0.0))
        if p.bull_floor > 0:
            trend = trend.where(~bull, np.maximum(trend, p.bull_floor * vf))
        if p.bear_floor > 0:
            trend = trend.where(bull, np.minimum(trend, -p.bear_floor * vf))
        if p.pb_drop > 0:
            trend = np.maximum(trend, self.pullback_sleeve(f) * vf)
        return trend

    def pullback_sleeve(self, f: pd.DataFrame) -> pd.Series:
        p = self.p
        dist = f[f"dist_high_{p.pb_days}d"].to_numpy()
        bull = f["bull"].to_numpy() > 0.5
        hold = 288 * p.pb_hold_days
        pos = np.zeros(len(f))
        active, start = False, 0
        for i in range(len(pos)):
            if active:
                if (not bull[i]) or dist[i] > -p.pb_exit or i - start >= hold:
                    active = False
                    continue
                pos[i] = 1.0
            elif bull[i] and dist[i] < -p.pb_drop:
                active, start = True, i
                pos[i] = 1.0
        return pd.Series(pos, f.index)

    def target_exposure(self, df: pd.DataFrame, cache: dict | None = None) -> pd.DataFrame:
        """`cache` may hold precomputed 'features' and 'vote' (they depend only on
        sigma_days / regime_days), which makes parameter scans much faster."""
        if cache is not None and "features" in cache:
            f, vote = cache["features"], cache["vote"]
        else:
            f = self.features(df)
            vote = self.trend_vote(df)["vote"]
            if cache is not None:
                cache["features"], cache["vote"] = f, vote
        trend = self.trend_sleeve(f, vote)
        dip = self.dip_sleeve(df, f)
        exposure = dip.where(dip != 0, trend)
        return pd.DataFrame({"exposure": exposure, "dip_sleeve": dip, "trend_sleeve": trend,
                             "trend_vote": vote}).join(f)
