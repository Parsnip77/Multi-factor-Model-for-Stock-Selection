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
from io import StringIO

import pandas as pd

# Allow importing from src/ without installing the package
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)

from targets import calc_forward_return
from ic_analyzer import calc_ic, calc_ic_metrics, plot_ic
from backtester import LayeredBacktester
from factor_combiner import rolling_linear_combine
from net_backtester import NetReturnBacktester

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
DATA_DIR  = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
FORWARD_DAYS = 5          # d-day forward return
IC_MEAN_THRESHOLD = 0.015  # minimum |IC mean| to keep a factor
ICIR_THRESHOLD = 0.20      # minimum |ICIR| to keep a factor
SHOW_PLOTS = False        # set False to suppress interactive charts
COMBINE_WINDOW = 3        # rolling OLS training window (trading days)


# -----------------------------------------------------------------------
# Report buffer – accumulates all output for result.txt
# -----------------------------------------------------------------------
_buf = StringIO()

RESULT_FILE = ROOT / "result.txt"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _p(msg: str = "") -> None:
    """Print to stdout and append to the report buffer."""
    print(msg)
    _buf.write(msg + "\n")


def _sep(char: str = "-", width: int = 62) -> None:
    _p(char * width)


def _step(msg: str) -> None:
    _p(f"\n{'=' * 62}")
    _p(f"  {msg}")
    _p(f"{'=' * 62}")


def _ok(msg: str) -> None:
    _p(f"  [OK]  {msg}")


def _info(msg: str) -> None:
    _p(f"  [  ]  {msg}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # Step 1: Load Parquet files
    # ------------------------------------------------------------------
    _step("Step 1 / 6  —  Loading Parquet files")

    prices_path  = DATA_DIR / "prices.parquet"
    factors_path = DATA_DIR / "factors_clean.parquet"

    for p in (prices_path, factors_path):
        if not p.exists():
            _p(f"\n  [ERROR] File not found: {p}")
            _p("  Please run `python data_preparation_main.py` first.\n")
            sys.exit(1)

    prices_df  = pd.read_parquet(prices_path)
    factors_df = pd.read_parquet(factors_path)

    _ok(f"prices.parquet   : {prices_df.shape[0]:>7,} rows | cols: {prices_df.columns.tolist()}")
    _ok(f"factors_clean    : {factors_df.shape[0]:>7,} rows | "
        f"factors: {[c for c in factors_df.columns if c not in ('trade_date', 'ts_code')]}")

    # ------------------------------------------------------------------
    # Step 2: Compute forward returns
    # ------------------------------------------------------------------
    _step(f"Step 2 / 6  —  Computing {FORWARD_DAYS}-day forward return")

    target_df = calc_forward_return(prices_df, d=FORWARD_DAYS)
    valid_count = target_df["forward_return"].notna().sum()
    _ok(f"target shape     : {target_df.shape[0]:>7,} rows  "
        f"(non-NaN: {valid_count:,})")

    # ------------------------------------------------------------------
    # Step 3: Single-factor IC analysis
    # ------------------------------------------------------------------
    _step("Step 3 / 6  —  Single-factor IC analysis")

    PLOTS_DIR.mkdir(exist_ok=True)
    alpha_cols = [c for c in factors_df.columns if c not in ("trade_date", "ts_code")]

    effective_alphas: list[str] = []
    effective_alphas_ic_mean: dict[str, float] = {}   # to detect reverse factors

    for alpha in alpha_cols:
        _sep()
        _p(f"  Factor: {alpha}")
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
        bt = LayeredBacktester(single_factor, target_df, forward_days=FORWARD_DAYS, plots_dir=PLOTS_DIR)
        perf = bt.run_backtest()
        _info("  Backtest metrics:")
        _p(perf.to_string())
        bt.plot(show=SHOW_PLOTS)
        _info(f"  Backtest plot saved.")

        # 3.5 Selection criteria
        if abs(metrics["ic_mean"]) > IC_MEAN_THRESHOLD and abs(metrics["icir"]) > ICIR_THRESHOLD:
            effective_alphas.append(alpha)
            effective_alphas_ic_mean[alpha] = metrics["ic_mean"]
            direction = "reverse" if metrics["ic_mean"] < 0 else "forward"
            _p(f"  >>> SELECTED  [{direction}]  (|IC mean| > {IC_MEAN_THRESHOLD:.1%}  &  |ICIR| > {ICIR_THRESHOLD})")
        else:
            _p(f"      not selected  (threshold: |IC mean| > {IC_MEAN_THRESHOLD:.1%}  &  |ICIR| > {ICIR_THRESHOLD})")

    # ------------------------------------------------------------------
    # Step 4: Report effective alphas
    # ------------------------------------------------------------------
    _step("Step 4 / 6  —  Results summary")

    if effective_alphas:
        _ok(f"Effective alphas ({len(effective_alphas)} / {len(alpha_cols)}):")
        for name in effective_alphas:
            _p(f"      * {name}")
    else:
        _info("No alpha passed the selection criteria.")
        _info(f"  Criteria: |IC mean| > {IC_MEAN_THRESHOLD:.1%}  AND  |ICIR| > {ICIR_THRESHOLD}")

    # ------------------------------------------------------------------
    # Step 5: Synthetic factor via rolling OLS combination
    # ------------------------------------------------------------------
    _step("Step 5 / 6  —  Synthetic factor (rolling OLS + symmetric orthogonalization)")

    _info(f"Effective alphas (IC/ICIR screen, for reference): {effective_alphas}")
    _info(f"Using ALL {len(alpha_cols)} factors for OLS combination: {alpha_cols}")
    _info(f"Rolling window = {COMBINE_WINDOW} trading days  |  orthogonalize = True")

    # Use cross-sectional percentile rank of forward_return as the OLS
    # regression target.  This removes the market-wide daily move (beta)
    # from the label, so the model learns relative stock selection ability
    # rather than fitting to the overall market direction.
    target_flat = target_df.reset_index()[["trade_date", "ts_code", "forward_return"]]
    target_cs_rank = target_flat.copy()
    target_cs_rank["forward_return"] = target_cs_rank.groupby("trade_date")[
        "forward_return"
    ].rank(pct=True)
    _info("  OLS target: cross-sectional pct-rank of forward_return (per trade_date)")

    synth_df = rolling_linear_combine(
        factors_df,
        target_cs_rank,
        factor_cols=alpha_cols,
        window=COMBINE_WINDOW,
        orthogonalize=True,
    )
    _ok(f"Synthetic factor : {synth_df.shape[0]:>7,} rows  "
        f"(dates: {synth_df['trade_date'].nunique()})")

    bt_synth = LayeredBacktester(synth_df, target_df, forward_days=FORWARD_DAYS, plots_dir=PLOTS_DIR)
    perf_synth = bt_synth.run_backtest()
    _info("  Backtest metrics (synthetic factor):")
    _p(perf_synth.to_string())
    bt_synth.plot(show=SHOW_PLOTS)
    _info("  Synthetic backtest plot saved.")

    # ------------------------------------------------------------------
    # Step 6: Net-return backtest with transaction costs
    # ------------------------------------------------------------------
    _step("Step 6 / 6  —  Net-return backtest (with transaction costs)")

    _info(f"cost_rate = {0.002:.2%}  |  forward_days = {FORWARD_DAYS}  |  top 20%")
    nb = NetReturnBacktester(
        synth_df,
        prices_df,
        forward_days=FORWARD_DAYS,
        cost_rate=0.002,
        plots_dir=PLOTS_DIR,
    )
    net_summary = nb.run_backtest()
    _info("  Net-return performance summary:")
    _p(net_summary.to_string())
    nb.plot(show=SHOW_PLOTS)
    _info("  Net-return chart saved.")

    _p(f"\n{'=' * 62}")
    _p("  Phase 2 factor analysis complete.")
    _p(f"{'=' * 62}\n")

    # ------------------------------------------------------------------
    # Write report buffer to result.txt
    # ------------------------------------------------------------------
    RESULT_FILE.write_text(_buf.getvalue(), encoding="utf-8")
    print(f"  Report saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
