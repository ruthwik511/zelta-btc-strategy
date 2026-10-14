"""
Zelta BTC/USDT strategy engine (Claude rebuild).

Design rules (no look-ahead):
  * Every signal is computed from COMPLETED bars only.
  * A signal computed at the close of bar t is executed at the OPEN of bar t+1.
  * Protective stops are checked inside each bar with its high/low; if a bar
    gaps through the stop, the fill is the (worse) open price.
  * 0.10% fee/slippage on every executed side (entry and exit), per the brief.
  * Equity is marked to market every hour; drawdown is reported both on hourly
    closes and on the intrabar worst case (low for longs / high for shorts).
"""
from __future__ import annotations

import glob
import math
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class BinanceKlines:
    """Loads Binance public-data monthly kline zips into one clean DataFrame."""

    @staticmethod
    def load(folder: str) -> pd.DataFrame:
        frames = []
        for path in sorted(glob.glob(os.path.join(folder, "*.zip"))):
            with zipfile.ZipFile(path) as zf:
                with zf.open(zf.namelist()[0]) as fh:
                    df = pd.read_csv(fh, header=None, names=KLINE_COLS, dtype=str)
            df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        for col in KLINE_COLS[:11]:
            df[col] = pd.to_numeric(df[col])
        ot = df["open_time"].astype("int64").to_numpy()
        ot = np.where(ot > 10**14, ot // 1000, ot)  # 2025+ files use microseconds
        df.index = pd.to_datetime(ot, unit="ms", utc=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        return df[["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_base"]]


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "quote_volume": "sum", "trades": "sum", "taker_base": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])


# ----------------------------------------------------------------------------
# Indicators (all causal)
# ----------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + g / l)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, 1)  # raw true range smoothed with alpha=1 -> TR itself
    a = lambda x: pd.Series(x, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    atr_n = a(tr)
    pdi = 100 * a(plus_dm) / atr_n
    mdi = 100 * a(minus_dm) / atr_n
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """Returns +1 / -1 supertrend direction computed on completed bars."""
    a = atr(df, n).to_numpy()
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    direction = np.zeros(len(c))
    for i in range(1, len(c)):
        if np.isnan(a[i]):
            continue
        fub[i] = ub[i] if (np.isnan(fub[i - 1]) or ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (np.isnan(flb[i - 1]) or lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        prev = direction[i - 1] if direction[i - 1] != 0 else 1
        if prev == 1 and c[i] < flb[i]:
            direction[i] = -1
        elif prev == -1 and c[i] > fub[i]:
            direction[i] = 1
        else:
            direction[i] = prev
    return pd.Series(direction, index=df.index)


def align_to(higher: pd.Series, base_index: pd.DatetimeIndex, higher_bar: str) -> pd.Series:
    """Map a higher-timeframe series to the base index WITHOUT look-ahead:
    a higher bar's value becomes visible only after that bar has closed."""
    shifted = higher.copy()
    shifted.index = shifted.index + pd.Timedelta(higher_bar)  # available at bar close
    return shifted.reindex(base_index, method="ffill")


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
@dataclass
class EngineConfig:
    fee: float = 0.001             # 0.10% per executed side
    initial_capital: float = 100_000.0
    leverage_cap: float = 1.0      # max |notional| / equity at entry
    stop_atr: float = 0.0          # 0 = no protective stop
    trail: bool = True             # trailing ATR stop (True) or fixed initial stop
    max_hold_bars: int = 0         # 0 = unlimited
    dd_brake: float = 0.0          # 0 = off; liquidate and pause after this drawdown from episode peak
    brake_pause_bars: int = 0


@dataclass
class BacktestResult:
    equity: pd.Series
    equity_worst: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    config: EngineConfig = field(repr=False)


def run_backtest(df: pd.DataFrame, target: pd.Series, cfg: EngineConfig,
                 atr_series: pd.Series | None = None) -> BacktestResult:
    """target[t] in [-1, 1] = desired exposure decided at the close of bar t,
    executed at the open of bar t+1."""
    # plain Python lists are much faster than numpy scalars inside the bar loop
    o = df["open"].to_numpy(float).tolist()
    h = df["high"].to_numpy(float).tolist()
    l = df["low"].to_numpy(float).tolist()
    c = df["close"].to_numpy(float).tolist()
    tg = np.nan_to_num(target.reindex(df.index).to_numpy(float)).tolist()
    av = (atr_series.reindex(df.index).to_numpy(float) if atr_series is not None
          else np.full(len(c), np.nan)).tolist()
    n = len(c)
    fee = cfg.fee
    isnan = math.isnan

    cash = cfg.initial_capital
    units = 0.0
    side = 0
    stop = np.nan
    entry_i = -1
    entry_px = 0.0
    entry_fee = 0.0
    entry_units = 0.0
    blocked_side = 0           # after a stop-out, block same-side re-entry until signal resets
    peak = cfg.initial_capital
    pause_until = -1

    eq = [0.0] * n
    eq_worst = [0.0] * n
    pos_arr = [0] * n
    trades = []
    nan = float("nan")

    def close_position(i, px, reason):
        nonlocal cash, units, side, stop, entry_i
        f = abs(units) * px * fee
        cash += units * px - f
        gross = units * (px - entry_px)
        trades.append((entry_i, i, side, entry_px, px, gross, entry_fee + f, gross - entry_fee - f, reason))
        units = 0.0
        side = 0
        stop = nan
        entry_i = -1

    stop = nan
    stop_atr = cfg.stop_atr
    lev_cap = cfg.leverage_cap
    for i in range(n):
        if i > 0:
            want = tg[i - 1]
            want_side = 1 if want > 0 else (-1 if want < 0 else 0)
            if blocked_side != 0 and want_side != blocked_side:
                blocked_side = 0
            if blocked_side != 0 and want_side == blocked_side:
                want, want_side = 0.0, 0
            if i <= pause_until:
                want, want_side = 0.0, 0
            # --- execute at open[i]
            if side != 0 and (want_side != side):
                close_position(i, o[i], "signal")
            if side == 0 and want_side != 0:
                equity_now = cash
                notional = min(abs(want), lev_cap) * equity_now
                units = want_side * notional / o[i]
                entry_fee = notional * fee
                cash -= units * o[i] + entry_fee
                side = want_side
                entry_px = o[i]
                entry_i = i
                if stop_atr > 0 and not isnan(av[i - 1]):
                    stop = o[i] - side * stop_atr * av[i - 1]
            # --- intrabar protective stop
            if side == 1 and l[i] <= stop:
                close_position(i, min(o[i], stop), "stop")
                blocked_side = 1
            elif side == -1 and h[i] >= stop:
                close_position(i, max(o[i], stop), "stop")
                blocked_side = -1
            # --- time stop (exit at close of bar)
            if side != 0 and cfg.max_hold_bars > 0 and i - entry_i >= cfg.max_hold_bars:
                close_position(i, c[i], "time")
        # --- mark to market
        e = cash + units * c[i]
        worst_px = l[i] if units > 0 else (h[i] if units < 0 else c[i])
        eq[i] = e
        eq_worst[i] = cash + units * worst_px
        pos_arr[i] = side
        # --- trailing stop update with the completed bar
        if side != 0 and stop_atr > 0 and cfg.trail and not isnan(av[i]):
            new_stop = c[i] - side * stop_atr * av[i]
            if isnan(stop):
                stop = new_stop
            else:
                stop = max(stop, new_stop) if side == 1 else min(stop, new_stop)
        # --- drawdown brake
        if e > peak:
            peak = e
        if cfg.dd_brake > 0 and e < peak * (1 - cfg.dd_brake) and i + 1 < n:
            if side != 0:
                close_position(i, c[i], "dd_brake")
                eq[i] = cash
            pause_until = i + cfg.brake_pause_bars
            peak = cash

    idx = df.index
    if side != 0:  # close any open position at the final close for reporting
        close_position(n - 1, c[-1], "end_of_test")
        eq[-1] = cash
        eq_worst[-1] = min(eq_worst[-1], cash)
    tr = pd.DataFrame(trades, columns=["entry_i", "exit_i", "side", "entry_price", "exit_price",
                                       "gross_pnl", "fees", "net_pnl", "exit_reason"])
    if len(tr):
        tr["entry_time"] = idx[tr["entry_i"].to_numpy()]
        tr["exit_time"] = idx[tr["exit_i"].to_numpy()]
        tr["hold_hours"] = (tr["exit_time"] - tr["entry_time"]).dt.total_seconds() / 3600
        tr["side"] = tr["side"].map({1: "LONG", -1: "SHORT"})
        tr = tr.drop(columns=["entry_i", "exit_i"])
    return BacktestResult(pd.Series(eq, idx), pd.Series(eq_worst, idx), pd.Series(pos_arr, idx), tr, cfg)


def run_exposure_backtest(df: pd.DataFrame, target: pd.Series, cfg: EngineConfig,
                          min_rebalance: float = 0.25) -> BacktestResult:
    """Portfolio engine for fractional, signed exposure (e.g. trend sleeve +
    dip-buy sleeve netted together). target[t] = desired exposure as a fraction
    of equity in [-leverage_cap, leverage_cap], decided at the close of bar t and
    traded at the open of bar t+1. Exposure is re-sized only when it changes
    sign, goes flat, or drifts by more than `min_rebalance` of equity, to keep
    costs realistic. A "trade" (for the 14 metrics) is one position episode:
    from leaving flat to returning flat (or flipping side)."""
    o = df["open"].to_numpy(float).tolist()
    h = df["high"].to_numpy(float).tolist()
    l = df["low"].to_numpy(float).tolist()
    c = df["close"].to_numpy(float).tolist()
    tg = np.clip(np.nan_to_num(target.reindex(df.index).to_numpy(float)),
                 -cfg.leverage_cap, cfg.leverage_cap).tolist()
    n = len(c)
    fee = cfg.fee
    cash = cfg.initial_capital
    units = 0.0
    eq = [0.0] * n
    eq_worst = [0.0] * n
    pos_arr = [0.0] * n
    trades = []
    ep = None  # open episode: [entry_i, side, entry_px, gross, fees]

    def end_episode(i, px, reason):
        nonlocal ep
        if ep is not None:
            trades.append((ep[0], i, ep[1], ep[2], px, ep[3], ep[4], ep[3] - ep[4], reason))
            ep = None

    for i in range(n):
        if i > 0:
            # mark previous close -> this open
            if units != 0.0 and ep is not None:
                ep[3] += units * (o[i] - c[i - 1])
            want = tg[i - 1]
            equity_open = cash + units * o[i]
            cur = units * o[i] / equity_open if equity_open > 0 else 0.0
            cur_side = (units > 0) - (units < 0)
            want_side = (want > 0) - (want < 0)
            need = (want_side != cur_side) or (want_side != 0 and abs(want - cur) >= min_rebalance)
            if need:
                if cur_side != 0 and want_side != cur_side:  # close / flip
                    f = abs(units) * o[i] * fee
                    cash += units * o[i] - f
                    ep[4] += f
                    end_episode(i, o[i], "signal")
                    units = 0.0
                    equity_open = cash
                new_units = want * equity_open / o[i]
                delta = new_units - units
                if delta != 0.0:
                    f = abs(delta) * o[i] * fee
                    cash -= delta * o[i] + f
                    if ep is None:
                        ep = [i, want_side, o[i], 0.0, 0.0]
                    ep[4] += f
                    units = new_units
            if units != 0.0 and ep is not None:
                ep[3] += units * (c[i] - o[i])
        e = cash + units * c[i]
        worst_px = l[i] if units > 0 else (h[i] if units < 0 else c[i])
        eq[i] = e
        eq_worst[i] = cash + units * worst_px
        pos_arr[i] = units * c[i] / e if e > 0 else 0.0
    if units != 0.0:
        f = abs(units) * c[-1] * fee
        cash += units * c[-1] - f
        ep[4] += f
        end_episode(n - 1, c[-1], "end_of_test")
        eq[-1] = cash
        eq_worst[-1] = min(eq_worst[-1], cash)
    idx = df.index
    tr = pd.DataFrame(trades, columns=["entry_i", "exit_i", "side", "entry_price", "exit_price",
                                       "gross_pnl", "fees", "net_pnl", "exit_reason"])
    if len(tr):
        tr["entry_time"] = idx[tr["entry_i"].to_numpy()]
        tr["exit_time"] = idx[tr["exit_i"].to_numpy()]
        tr["hold_hours"] = (tr["exit_time"] - tr["entry_time"]).dt.total_seconds() / 3600
        tr["side"] = tr["side"].map({1: "LONG", -1: "SHORT"})
        tr = tr.drop(columns=["entry_i", "exit_i"])
    return BacktestResult(pd.Series(eq, idx), pd.Series(eq_worst, idx), pd.Series(pos_arr, idx), tr, cfg)


# ----------------------------------------------------------------------------
# Metrics (the brief's 14 + extras)
# ----------------------------------------------------------------------------
def max_drawdown(e: pd.Series, worst: pd.Series | None = None) -> float:
    peak = e.cummax()
    dd = 1 - e / peak
    if worst is not None:
        dd = np.maximum(dd, 1 - worst / peak)
    return float(dd.max())


def metrics(res: BacktestResult, df: pd.DataFrame, start=None, end=None) -> dict:
    e = res.equity.loc[start:end]
    w = res.equity_worst.loc[start:end]
    e0 = res.equity.loc[:start].iloc[-2] if start is not None and len(res.equity.loc[:start]) > 1 else res.config.initial_capital
    t = res.trades
    if len(t):
        t = t[(t["entry_time"] >= (pd.Timestamp(start, tz="UTC") if start else t["entry_time"].min())) &
              (t["exit_time"] <= (pd.Timestamp(end, tz="UTC") + pd.Timedelta("1D") if end else t["exit_time"].max()))]
    daily = e.resample("1D").last().dropna()
    dr = daily.pct_change().fillna(daily.iloc[0] / e0 - 1)
    sharpe = float(np.sqrt(365) * dr.mean() / dr.std()) if dr.std() > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(dr, 0) ** 2))
    sortino = float(np.sqrt(365) * dr.mean() / downside) if downside > 0 else np.nan
    years = (e.index[-1] - e.index[0]).total_seconds() / (365.25 * 86400)
    total = e.iloc[-1] / e0 - 1
    yearly = e.resample("YE").last()
    prev = pd.concat([pd.Series([e0]), yearly.iloc[:-1]]).to_numpy()
    yr = pd.Series(yearly.to_numpy() / prev - 1, index=yearly.index.year)
    px = df["close"].loc[start:end]
    po = df["open"].loc[start:end]
    bh = (px.iloc[-1] * (1 - res.config.fee)) / (po.iloc[0] * (1 + res.config.fee)) - 1
    wins = t[t["net_pnl"] > 0] if len(t) else t
    losses = t[t["net_pnl"] <= 0] if len(t) else t
    gross_profit = float(t.loc[t["gross_pnl"] > 0, "gross_pnl"].sum()) if len(t) else 0.0
    gross_loss = float(t.loc[t["gross_pnl"] <= 0, "gross_pnl"].sum()) if len(t) else 0.0
    return {
        "gross_profit": gross_profit,
        "net_profit": float(e.iloc[-1] - e0),
        "total_trades": int(len(t)),
        "win_rate": float(len(wins) / len(t)) if len(t) else np.nan,
        "max_dd_close": max_drawdown(e),
        "max_dd_intrabar": max_drawdown(e, w),
        "gross_loss": gross_loss,
        "avg_win": float(wins["net_pnl"].mean()) if len(wins) else np.nan,
        "avg_loss": float(losses["net_pnl"].mean()) if len(losses) else np.nan,
        "buy_hold_return": float(bh),
        "largest_loss": float(t["net_pnl"].min()) if len(t) else np.nan,
        "largest_win": float(t["net_pnl"].max()) if len(t) else np.nan,
        "sharpe": sharpe,
        "sortino": sortino,
        "avg_hold_hours": float(t["hold_hours"].mean()) if len(t) else np.nan,
        "total_return": float(total),
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else np.nan,
        "avg_annual_return": float(yr.mean()),
        "yearly": {int(k): float(v) for k, v in yr.items()},
        "long_trades": int((t["side"] == "LONG").sum()) if len(t) else 0,
        "short_trades": int((t["side"] == "SHORT").sum()) if len(t) else 0,
        "fees_paid": float(t["fees"].sum()) if len(t) else 0.0,
        "time_in_market": float((res.position.loc[start:end] != 0).mean()),
    }
