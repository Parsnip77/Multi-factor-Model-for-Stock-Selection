"""
analyze_main.py
---------------
Phase 2 entry script: load Phase-1 Parquet outputs, compute forward returns,
run single-factor IC analysis for every alpha, and report the effective alphas.

Usage
-----
    python analyze_main.py
"""

import sys
import pathlib
import warnings

import pandas as pd

# Allow importing from src/ without installing the package
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)

from targets import calc_forward_return
from ic_analyzer import calc_ic, calc_ic_metrics, plot_ic
from backtester import LayeredBacktester

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
DATA_DIR  = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
FORWARD_DAYS = 1          # d-day forward return
IC_MEAN_THRESHOLD = 0.02  # minimum |IC mean| to keep a factor
ICIR_THRESHOLD = 0.30     # minimum |ICIR| to keep a factor
SHOW_PLOTS = True         # set False to suppress interactive charts


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _sep(char: str = "-", width: int = 62) -> None:
    print(char * width)


def _step(msg: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {msg}")
    print(f"{'=' * 62}")


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _info(msg: str) -> None:
    print(f"  [  ]  {msg}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # Step 1: Load Parquet files
    # ------------------------------------------------------------------
    _step("Step 1 / 4  —  Loading Parquet files")

    prices_path  = DATA_DIR / "prices.parquet"
    factors_path = DATA_DIR / "factors_clean.parquet"

    for p in (prices_path, factors_path):
        if not p.exists():
            print(f"\n  [ERROR] File not found: {p}")
            print("  Please run `python data_preparation_main.py` first.\n")
            sys.exit(1)

    prices_df  = pd.read_parquet(prices_path)
    factors_df = pd.read_parquet(factors_path)

    _ok(f"prices.parquet   : {prices_df.shape[0]:>7,} rows | cols: {prices_df.columns.tolist()}")
    _ok(f"factors_clean    : {factors_df.shape[0]:>7,} rows | "
        f"factors: {[c for c in factors_df.columns if c not in ('trade_date', 'ts_code')]}")

    # ------------------------------------------------------------------
    # Step 2: Compute forward returns
    # ------------------------------------------------------------------
    _step(f"Step 2 / 4  —  Computing {FORWARD_DAYS}-day forward return")

    target_df = calc_forward_return(prices_df, d=FORWARD_DAYS)
    valid_count = target_df["forward_return"].notna().sum()
    _ok(f"target shape     : {target_df.shape[0]:>7,} rows  "
        f"(non-NaN: {valid_count:,})")

    # ------------------------------------------------------------------
    # Step 3: Single-factor IC analysis
    # ------------------------------------------------------------------
    _step("Step 3 / 4  —  Single-factor IC analysis")

    PLOTS_DIR.mkdir(exist_ok=True)
    alpha_cols = [c for c in factors_df.columns if c not in ("trade_date", "ts_code")]

    effective_alphas: list[str] = []

    for alpha in alpha_cols:
        _sep()
        print(f"  Factor: {alpha}")
        _sep()

        single_factor = factors_df[["trade_date", "ts_code", alpha]].copy()

        # 3.1 Compute IC time series
        ic_series = calc_ic(single_factor, target_df)

        # 3.2 IC metrics summary
        metrics = calc_ic_metrics(ic_series)
        _info(f"  IC Mean : {metrics['ic_mean']:>+.4f}")
        _info(f"  IC Std  : {metrics['ic_std']:>.4f}")
        _info(f"  ICIR    : {metrics['icir']:>+.4f}")

        # 3.3 Plot IC chart and save to plots/
        save_path = PLOTS_DIR / f"{alpha}_ic.png"
        plot_ic(ic_series, factor_name=alpha, show=SHOW_PLOTS, save_path=save_path)
        _info(f"  IC chart saved.")

        # 3.4 Layered backtest
        bt = LayeredBacktester(single_factor, target_df, plots_dir=PLOTS_DIR)
        perf = bt.run_backtest()
        _info("  Backtest metrics:")
        print(perf.to_string())
        bt_save = PLOTS_DIR / f"{alpha}_backtest.png"
        bt.plot(show=SHOW_PLOTS)
        _info(f"  Backtest plot saved.")

        # 3.5 Selection criteria
        if abs(metrics["ic_mean"]) > IC_MEAN_THRESHOLD and abs(metrics["icir"]) > ICIR_THRESHOLD:
            effective_alphas.append(alpha)
            print(f"  >>> SELECTED  (|IC mean| > {IC_MEAN_THRESHOLD:.0%}  &  |ICIR| > {ICIR_THRESHOLD})")
        else:
            print(f"      not selected  (threshold: |IC mean| > {IC_MEAN_THRESHOLD:.0%}  &  |ICIR| > {ICIR_THRESHOLD})")

    # ------------------------------------------------------------------
    # Step 4: Report effective alphas
    # ------------------------------------------------------------------
    _step("Step 4 / 4  —  Results summary")

    if effective_alphas:
        _ok(f"Effective alphas ({len(effective_alphas)} / {len(alpha_cols)}):")
        for name in effective_alphas:
            print(f"      * {name}")
    else:
        _info("No alpha passed the selection criteria.")
        _info(f"  Criteria: |IC mean| > {IC_MEAN_THRESHOLD:.0%}  AND  |ICIR| > {ICIR_THRESHOLD}")

    print(f"\n{'=' * 62}")
    print("  Phase 2 factor analysis complete.")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
