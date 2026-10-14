"""
Run the Zelta BTC/USDT backtest locally in VS Code.

Default: download Binance monthly 5-minute candles from Aug 2017 through Sep 2026,
then run the original hybrid vs adaptive strategy comparisons and stress tests.

Usage:
    python run_backtest.py
    python run_backtest.py --skip-download
"""
from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zelta_engine import BinanceKlines, metrics, max_drawdown
from strategy import ZeltaHybridStrategy
from strategy_adaptive import ZeltaAdaptiveStrategy, FINAL_PARAMS

DATA_DIR = Path("data") / "BTCUSDT_5m"
START_MONTH = "2017-08"
END_MONTH = "2026-09"

BRIEF = ("2018-01-01", "2022-12-31 23:55")
AFTER = ("2023-01-01", "2026-09-30 23:55")
ALL = ("2018-01-01", "2026-09-30 23:55")

ROWS = [
    ("Gross profit (before fees)", "gross_profit", "usd"),
    ("Net profit (after fees)", "net_profit", "usd"),
    ("Total closed trades", "total_trades", "int"),
    ("Win rate", "win_rate", "pct"),
    ("Max drawdown (daily close)", "max_dd_daily", "pct"),
    ("Gross loss (before fees)", "gross_loss", "usd"),
    ("Average winning trade", "avg_win", "usd"),
    ("Average losing trade", "avg_loss", "usd"),
    ("BTC buy-and-hold return", "buy_hold_return", "pct"),
    ("Largest losing trade", "largest_loss", "usd"),
    ("Largest winning trade", "largest_win", "usd"),
    ("Sharpe ratio", "sharpe", "num"),
    ("Sortino ratio", "sortino", "num"),
    ("Average holding time (hours)", "avg_hold_hours", "num"),
    ("Average annual return", "avg_annual_return", "pct"),
    ("Compound annual growth (CAGR)", "cagr", "pct"),
    ("Max drawdown (intrabar worst case)", "max_dd_intrabar", "pct"),
]


def download_month(month: pd.Period) -> str:
    name = f"BTCUSDT-5m-{month.year}-{month.month:02d}.zip"
    path = DATA_DIR / name
    if path.exists() and path.stat().st_size > 0:
        return f"OK {name}"
    url = (
        "https://data.binance.vision/data/spot/monthly/klines/"
        f"BTCUSDT/5m/{name}"
    )
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=45) as response, open(path, "wb") as out:
            out.write(response.read())
        if path.stat().st_size == 0:
            path.unlink(missing_ok=True)
            return f"EMPTY {name}"
        return f"OK {name}"
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        path.unlink(missing_ok=True)
        return f"SKIP {name}: {exc}"


def prepare_data(skip_download: bool) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not skip_download:
        months = pd.period_range(START_MONTH, END_MONTH, freq="M")
        print(f"Checking/downloading {len(months)} monthly Binance files...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(download_month, months))
        ready = sum(s.startswith("OK ") for s in statuses)
        skipped = [s for s in statuses if s.startswith("SKIP ") or s.startswith("EMPTY ")]
        print(f"{ready} monthly files ready; {len(skipped)} unavailable.")
        for item in skipped[:8]:
            print(" ", item)
    if not list(DATA_DIR.glob("*.zip")):
        raise FileNotFoundError(
            f"No monthly .zip files found in {DATA_DIR.resolve()}. "
            "Run without --skip-download to fetch data."
        )
    df = BinanceKlines.load(str(DATA_DIR)).loc["2017-08-17":"2026-09-30 23:55"]
    if df.empty:
        raise ValueError("No candles loaded. Check the downloaded ZIP files.")
    print(f"Loaded {len(df):,} five-minute candles: {df.index[0]} to {df.index[-1]}")
    return df


def fmt(value, kind):
    if pd.isna(value):
        return "N/A"
    if kind == "usd":
        return f"{value:,.2f} USDT"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "pct":
        return f"{100 * value:.2f}%"
    return f"{value:.3f}"


def run(strategy, signals, df, window, **kwargs):
    result, _ = strategy.backtest(df, *window, signals=signals, **kwargs)
    metrics_out = metrics(result, df.loc[window[0]:window[1]])
    daily_equity = result.equity.resample("1D").last().dropna()
    metrics_out["max_dd_daily"] = max_drawdown(daily_equity)
    return result, metrics_out


def main():
    parser = argparse.ArgumentParser(description="Zelta BTC/USDT strategy backtest")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Use existing monthly Binance ZIP files in data/BTCUSDT_5m."
    )
    args = parser.parse_args()

    df = prepare_data(args.skip_download)
    final = ZeltaAdaptiveStrategy(FINAL_PARAMS)
    first = ZeltaHybridStrategy()

    print("\nBuilding signals. This may take a while for 5-minute data...")
    sig_final = final.target_exposure(df)
    sig_first = first.target_exposure(df)
    print("Signals ready.")

    _, m_brief = run(final, sig_final, df, BRIEF)
    _, m_after = run(final, sig_final, df, AFTER)
    table = pd.DataFrame({
        "Metric": [row[0] for row in ROWS],
        "2018-2022 (competition period)": [fmt(m_brief[k], t) for _, k, t in ROWS],
        "2023-Sep 2026 (out-of-sample period)": [fmt(m_after[k], t) for _, k, t in ROWS],
    })
    table.to_csv("metrics_by_period.csv", index=False)
    print("\n=== FINAL STRATEGY METRICS ===")
    print(table.to_string(index=False))

    periods = {
        "2018-2022 (brief)": BRIEF,
        "2023-2024": ("2023-01-01", "2024-12-31 23:55"),
        "Jan 2025-Sep 2026": ("2025-01-01", "2026-09-30 23:55"),
    }
    period_rows = []
    for name, window in periods.items():
        _, mf = run(first, sig_first, df, window)
        _, ma = run(final, sig_final, df, window)
        period_rows.append({
            "Period": name,
            "BTC buy & hold (total)": f"{100 * ma['buy_hold_return']:.2f}%",
            "First version avg annual return": f"{100 * mf['avg_annual_return']:.2f}%",
            "First version max drawdown": f"{100 * mf['max_dd_daily']:.2f}%",
            "Final version avg annual return": f"{100 * ma['avg_annual_return']:.2f}%",
            "Final version max drawdown": f"{100 * ma['max_dd_daily']:.2f}%",
            "Final Sharpe": f"{ma['sharpe']:.3f}",
        })
    period_table = pd.DataFrame(period_rows)
    period_table.to_csv("period_comparison.csv", index=False)
    print("\n=== PERIOD COMPARISON ===")
    print(period_table.to_string(index=False))

    r_final, _ = final.backtest(df, *ALL, signals=sig_final)
    r_first, _ = first.backtest(df, *ALL, signals=sig_first)
    win = df.loc[ALL[0]:ALL[1]]
    btc = 100_000 * win["close"].resample("1D").last() / win["open"].iloc[0]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(btc, lw=1.2, label="BTC buy and hold")
    ax.plot(r_first.equity.resample("1D").last(), lw=1.4, label="First version")
    ax.plot(r_final.equity.resample("1D").last(), lw=2.0, label="Final adaptive version")
    ax.axvline(pd.Timestamp("2023-01-01", tz="UTC"), color="black", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("Account value, USDT (log scale)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("100,000 USDT, Jan 2018-Sep 2026 (dashed line = end of competition period)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("equity_curve.png", dpi=160)
    plt.show()
    print("\nEnd values:")
    print(f"  Final adaptive: {r_final.equity.iloc[-1]:,.2f} USDT")
    print(f"  First version:  {r_first.equity.iloc[-1]:,.2f} USDT")
    print(f"  BTC buy/hold:   {btc.iloc[-1]:,.2f} USDT")

    years = pd.DataFrame({"Final version": {**m_brief["yearly"], **m_after["yearly"]}})
    close_y = df["close"].resample("YE").last()
    open_y = df["open"].resample("YE").first()
    years["BTC"] = [
        close_y.loc[str(y)].iloc[0] / open_y.loc[str(y)].iloc[0] - 1
        for y in years.index
    ]
    yearly = (years * 100).round(2)
    yearly.to_csv("calendar_year_returns.csv")
    print("\n=== CALENDAR-YEAR RETURNS (%) ===")
    print(yearly.to_string())

    stress_cases = [
        ("0.10% fee (base)", {}),
        ("0.15% fee", {"fee": 0.0015}),
        ("0.20% fee", {"fee": 0.002}),
        ("orders 5 min late", {"delay_bars": 1}),
        ("orders 15 min late", {"delay_bars": 3}),
    ]
    stress_rows = []
    for label, kwargs in stress_cases:
        result, _ = final.backtest(df, *BRIEF, signals=sig_final, **kwargs)
        m = metrics(result, df.loc[BRIEF[0]:BRIEF[1]])
        dd = max_drawdown(result.equity.resample("1D").last().dropna())
        stress_rows.append({
            "Scenario": label,
            "Average annual return": f"{100 * m['avg_annual_return']:.2f}%",
            "Max drawdown (daily)": f"{100 * dd:.2f}%",
            "Sharpe": f"{m['sharpe']:.3f}",
        })
    stress = pd.DataFrame(stress_rows)
    stress.to_csv("stress_test.csv", index=False)
    print("\n=== STRESS TEST (2018-2022) ===")
    print(stress.to_string(index=False))
    print("\nSaved: metrics_by_period.csv, period_comparison.csv, calendar_year_returns.csv, stress_test.csv, equity_curve.png")


if __name__ == "__main__":
    main()
