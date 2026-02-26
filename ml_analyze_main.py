"""
ml_analyze_main.py
------------------
Stage 3 pipeline: LightGBM alpha synthesis + dual backtest + report generation.

Workflow
--------
1.  Load ``factors_clean.parquet`` (15 cleaned alphas) and ``prices.parquet``.
2.  Compute 5-day forward returns via ``calc_forward_return``.
3.  Merge factors + target into a wide flat table; drop NaN rows.
4.  Run walk-forward cross-validation (WalkForwardSplitter).
5.  Per fold: train AlphaLGBM, predict out-of-sample ML alpha scores.
6.  Concatenate all fold predictions; apply 3-day rolling-mean smoothing.
7.  IC analysis of the ML synthetic factor (IC series, metrics, IC chart).
8.  Backtest with LayeredBacktester  → layered NAV chart + performance table.
9.  Backtest with NetReturnBacktester → net-return NAV chart + metrics.
10. Plot average feature importance across folds.
11. Plot SHAP beeswarm on last-fold test sample.
12. Write text summary to ``result_ml.txt``.

Usage
-----
    python ml_analyze_main.py
    # Key statistics are printed to stdout and also written to result_ml.txt.
"""

from __future__ import annotations

import pathlib
import sys
import textwrap
import warnings
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project modules
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from targets import calc_forward_return
from ic_analyzer import calc_ic, calc_ic_metrics, plot_ic
from backtester import LayeredBacktester
from net_backtester import NetReturnBacktester
from ml_data_prep import WalkForwardSplitter
from lgbm_model import AlphaLGBM

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path("data")
PLOTS_DIR = pathlib.Path("plots")
RESULT_FILE = pathlib.Path("result_ml.txt")

FORWARD_DAYS: int = 1
TRAIN_MONTHS: int = 24
VAL_MONTHS: int = 6
TEST_MONTHS: int = 6
EMBARGO_DAYS: int = 1        # must be >= FORWARD_DAYS to prevent target leakage
SHAP_SAMPLE_SIZE: int = 300  # number of rows to subsample for SHAP (speed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(msg: str, buf: StringIO) -> None:
    """Echo to stdout and accumulate into the string buffer for the report."""
    print(msg)
    buf.write(msg + "\n")


def _section(title: str, buf: StringIO, width: int = 70) -> None:
    _print("\n" + "=" * width, buf)
    _print(f"  {title}", buf)
    _print("=" * width, buf)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    report_buf = StringIO()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    _section("Stage 3 · LightGBM Alpha Synthesis", report_buf)

    factors_path = DATA_DIR / "factors_clean.parquet"
    prices_path = DATA_DIR / "prices.parquet"

    for p in (factors_path, prices_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run data_preparation_main.py first to generate Parquet files."
            )

    _print("1. Loading factors_clean.parquet and prices.parquet ...", report_buf)
    factors_df = pd.read_parquet(factors_path)
    prices_df = pd.read_parquet(prices_path)

    # Normalise: flatten MultiIndex if present
    if isinstance(factors_df.index, pd.MultiIndex):
        factors_df = factors_df.reset_index()
    if isinstance(prices_df.index, pd.MultiIndex):
        prices_df = prices_df.reset_index()

    _print(f"   factors shape : {factors_df.shape}", report_buf)
    _print(f"   prices  shape : {prices_df.shape}", report_buf)

    # -----------------------------------------------------------------------
    # 2. Compute forward returns
    # -----------------------------------------------------------------------
    _print(f"\n2. Computing {FORWARD_DAYS}-day forward returns ...", report_buf)
    target_df = calc_forward_return(prices_df, d=FORWARD_DAYS)
    target_flat = target_df.reset_index()[["trade_date", "ts_code", "forward_return"]]

    # -----------------------------------------------------------------------
    # 3. Merge factors + target into wide table
    # -----------------------------------------------------------------------
    _print("\n3. Merging factors and target labels ...", report_buf)

    # Identify alpha feature columns (everything except the key columns)
    key_cols = {"trade_date", "ts_code"}
    feature_cols = [c for c in factors_df.columns if c not in key_cols]

    df_merged = pd.merge(
        factors_df[["trade_date", "ts_code"] + feature_cols],
        target_flat,
        on=["trade_date", "ts_code"],
        how="inner",
    ).dropna(subset=feature_cols + ["forward_return"])

    df_merged = df_merged.sort_values("trade_date").reset_index(drop=True)

    # Cross-sectional percentile rank of forward_return per trade_date.
    # Using rank-based targets removes market-wide beta from the training
    # signal: the model learns to rank stocks within each day rather than
    # predicting absolute return levels, which are heavily contaminated by
    # the market's daily move.  Raw forward_return is kept for backtesting.
    df_merged["cs_rank_return"] = df_merged.groupby("trade_date")[
        "forward_return"
    ].rank(pct=True)

    _print(f"   merged shape  : {df_merged.shape}", report_buf)
    _print(f"   feature cols  : {feature_cols}", report_buf)
    _print(
        "   training target : cs_rank_return  "
        "(cross-sectional pct-rank of forward_return, per trade_date)",
        report_buf,
    )

    # -----------------------------------------------------------------------
    # 4. Walk-forward cross-validation
    # -----------------------------------------------------------------------
    _section("Walk-Forward Training", report_buf)

    splitter = WalkForwardSplitter(
        train_months=TRAIN_MONTHS,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        embargo_days=EMBARGO_DAYS,
    )

    estimated_folds = splitter.n_splits(df_merged)
    _print(
        f"   Settings : train={TRAIN_MONTHS}m  val={VAL_MONTHS}m  "
        f"test={TEST_MONTHS}m  embargo={EMBARGO_DAYS}d",
        report_buf,
    )
    _print(f"   Estimated folds : {estimated_folds}", report_buf)

    if estimated_folds == 0:
        raise RuntimeError(
            "Not enough data for even one fold. "
            "Consider reducing train_months / val_months / test_months."
        )

    all_predictions: list[pd.DataFrame] = []
    fold_importances: list[pd.DataFrame] = []
    last_X_test: pd.DataFrame | None = None

    for fold_idx, (train_mask, val_mask, test_mask) in enumerate(
        splitter.split(df_merged), start=1
    ):
        df_train = df_merged[train_mask]
        df_val = df_merged[val_mask]
        df_test = df_merged[test_mask]

        X_train = df_train[feature_cols]
        y_train = df_train["cs_rank_return"]   # cross-sectional rank target
        X_val = df_val[feature_cols]
        y_val = df_val["cs_rank_return"]       # cross-sectional rank target
        X_test = df_test[feature_cols]

        train_dates = df_train["trade_date"]
        test_dates = df_test["trade_date"]

        _print(
            f"\n--- Fold {fold_idx} ---\n"
            f"  Train : {train_dates.min()} → {train_dates.max()}  "
            f"(n={len(X_train):,})\n"
            f"  Val   : {df_val['trade_date'].min()} → "
            f"{df_val['trade_date'].max()}  (n={len(X_val):,})\n"
            f"  Test  : {test_dates.min()} → {test_dates.max()}  "
            f"(n={len(X_test):,})",
            report_buf,
        )

        ml_model = AlphaLGBM()
        ml_model.train(X_train, y_train, X_val, y_val)

        best_iter = ml_model.model.best_iteration_
        _print(f"  Best iteration : {best_iter}", report_buf)

        y_pred = ml_model.predict(X_test)

        df_pred_chunk = df_test[["trade_date", "ts_code"]].copy()
        df_pred_chunk["ml_alpha"] = y_pred
        all_predictions.append(df_pred_chunk)

        fold_importances.append(ml_model.get_feature_importance())

        last_X_test = X_test
        last_model = ml_model

    # -----------------------------------------------------------------------
    # 6. Concatenate predictions and smooth with rolling mean
    # -----------------------------------------------------------------------
    _section("Assembling Final ML Alpha", report_buf)
    final_alpha_df = pd.concat(all_predictions, ignore_index=True)

    # Apply a 3-day rolling mean per stock to reduce day-to-day signal noise
    # and lower portfolio turnover.  Stocks with fewer than 3 history points
    # (the first 2 trading days of each stock's prediction window) receive NaN
    # and are dropped so the backtester operates on fully-smoothed scores only.
    final_alpha_df = final_alpha_df.sort_values(
        ["ts_code", "trade_date"]
    ).reset_index(drop=True)
    final_alpha_df["ml_alpha"] = (
        final_alpha_df.groupby("ts_code")["ml_alpha"]
        .transform(lambda s: s.rolling(window=3, min_periods=3).mean())
    )
    final_alpha_df = final_alpha_df.dropna(subset=["ml_alpha"]).reset_index(drop=True)

    _print(f"   final_alpha_df shape : {final_alpha_df.shape}  (after 3-day smoothing)", report_buf)
    _print(
        f"   date range : {final_alpha_df['trade_date'].min()} → "
        f"{final_alpha_df['trade_date'].max()}",
        report_buf,
    )

    # -----------------------------------------------------------------------
    # 7. IC analysis of the ML synthetic factor
    # -----------------------------------------------------------------------
    _section("IC Analysis — ML Synthetic Factor", report_buf)

    ic_series = calc_ic(final_alpha_df, target_flat)
    ic_metrics = calc_ic_metrics(ic_series)

    _print(f"   IC Mean : {ic_metrics['ic_mean']:>+.4f}", report_buf)
    _print(f"   IC Std  : {ic_metrics['ic_std']:>.4f}", report_buf)
    _print(f"   ICIR    : {ic_metrics['icir']:>+.4f}", report_buf)

    ic_path = PLOTS_DIR / "ml_alpha_ic.png"
    plot_ic(ic_series, factor_name="ml_alpha", show=False, save_path=ic_path)
    plt.close("all")
    _print(f"\n   IC chart saved: {ic_path}", report_buf)

    # -----------------------------------------------------------------------
    # 8. Layered backtest
    # -----------------------------------------------------------------------
    _section("Layered Backtest (LayeredBacktester)", report_buf)

    bt = LayeredBacktester(
        final_alpha_df,
        target_flat,
        num_groups=5,
        rf=0.03,
        forward_days=FORWARD_DAYS,
        plots_dir=PLOTS_DIR,
    )
    perf_table = bt.run_backtest()
    bt.plot(show=False)

    _print("\nLayered Backtest Performance:\n", report_buf)
    _print(perf_table.to_string(), report_buf)

    # -----------------------------------------------------------------------
    # 8. Net return backtest
    # -----------------------------------------------------------------------
    _section("Net Return Backtest (NetReturnBacktester)", report_buf)

    nb = NetReturnBacktester(
        final_alpha_df,
        prices_df,
        forward_days=FORWARD_DAYS,
        cost_rate=0.002,
        rf=0.03,
        plots_dir=PLOTS_DIR,
    )
    net_summary = nb.run_backtest()
    nb.plot(show=False)

    _print("\nNet Return Backtest Summary:\n", report_buf)
    _print(net_summary.to_string(), report_buf)

    # -----------------------------------------------------------------------
    # 9. Average feature importance across folds
    # -----------------------------------------------------------------------
    _section("Feature Importance (Average Across Folds)", report_buf)

    avg_importance = (
        pd.concat(fold_importances, ignore_index=True)
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    _print("\nAverage Feature Importance (gain):\n", report_buf)
    _print(avg_importance.to_string(index=False), report_buf)

    # Plot average importance
    fig_imp, ax_imp = plt.subplots(figsize=(8, max(4, len(avg_importance) * 0.45)))
    ax_imp.barh(
        avg_importance["feature"][::-1],
        avg_importance["importance"][::-1],
    )
    ax_imp.set_title(f"Average Feature Importance (gain) — {estimated_folds} Folds")
    ax_imp.set_xlabel("Mean Importance (gain)")
    fig_imp.tight_layout()
    imp_path = PLOTS_DIR / "feature_importance.png"
    fig_imp.savefig(imp_path, dpi=150)
    plt.close(fig_imp)
    _print(f"\n   Saved: {imp_path}", report_buf)

    # -----------------------------------------------------------------------
    # 10. SHAP analysis (last fold test sample)
    # -----------------------------------------------------------------------
    _section("SHAP Analysis (Last Fold Test Sample)", report_buf)

    if last_X_test is not None and last_model is not None:
        sample_size = min(SHAP_SAMPLE_SIZE, len(last_X_test))
        X_shap = last_X_test.sample(n=sample_size, random_state=42)
        _print(f"   SHAP sample size : {sample_size} rows", report_buf)
        try:
            shap_path = PLOTS_DIR / "shap_beeswarm.png"
            last_model.plot_shap(X_shap, save_path=shap_path)
            plt.close("all")
            _print(f"   Saved: {shap_path}", report_buf)
        except ImportError as e:
            _print(f"   SHAP skipped: {e}", report_buf)

    # -----------------------------------------------------------------------
    # 11. Write report
    # -----------------------------------------------------------------------
    _section("Report Written", report_buf)
    report_text = report_buf.getvalue()
    RESULT_FILE.write_text(report_text, encoding="utf-8")
    print(f"\nFull report saved to: {RESULT_FILE}")
    print("Charts saved to:      plots/")


if __name__ == "__main__":
    main()
