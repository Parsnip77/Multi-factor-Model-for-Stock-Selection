"""
data_preparation_main.py — Phase 1 end-to-end pipeline for the multi-factor model.

Execution order:
    1. Verify that the local SQLite database exists.
    2. Load raw data via DataEngine.
    2.5 Compute tradable mask (suspended / delisted / limit-up-down).
    3. Compute raw alpha factors via Alpha101.
    4. Clean alpha factors via FactorCleaner; override non-tradable cells to NaN.
    5. Export four Parquet files to ./data/:
         prices.parquet        — raw OHLCV + adj_factor + tradable, keyed (trade_date, ts_code)
         meta.parquet          — market-cap, industry, PE, PB, keyed (trade_date, ts_code)
         factors_raw.parquet   — raw alpha values, keyed (trade_date, ts_code)
         factors_clean.parquet — cleaned alpha values; non-tradable cells are NaN
                                 (keyed (trade_date, ts_code))

All four tables share the same logical primary key (trade_date, ts_code) and
can be joined freely on those two columns.

Usage:
    python data_preparation_main.py
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress FutureWarnings from pandas/pyarrow version-compatibility notices.
warnings.filterwarnings("ignore", category=FutureWarning)

# Allow imports from src/ regardless of where the script is invoked from.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402  (must come after sys.path tweak)
from alphas import Alpha101
from data_loader import DataEngine
from preprocessor import FactorCleaner


# ------------------------------------------------------------------
# Console helpers
# ------------------------------------------------------------------

def _step(msg: str) -> None:
    """Print a prominent step banner."""
    print(f"\n{'=' * 62}\n  {msg}\n{'=' * 62}")


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _info(msg: str) -> None:
    print(f"  [--]  {msg}")


def _err(msg: str) -> None:
    print(f"  [ERR] {msg}")


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reset MultiIndex (date, code) to flat columns (trade_date, ts_code).
    Preserves row order; safe to call on frames with or without extra columns.
    """
    return (
        df.reset_index()
        .rename(columns={"date": "trade_date", "code": "ts_code"})
    )


def _compute_tradable(df_price: pd.DataFrame) -> pd.Series:
    """Return a boolean Series (MultiIndex date×code) marking tradable stock-days.

    A stock-day is considered NOT tradable under any of these three conditions:

    1. Suspended  — trading volume is zero (no transactions occurred).
    2. Delisted   — closing price is NaN or exactly zero (stock no longer exists).
    3. Limit hit  — daily price change exceeds 9.5 % AND the closing price is
                    flush against the intraday high (limit-up) or low (limit-down).
                    Entering a position at limit price is practically impossible
                    due to zero or near-zero available liquidity on that side.

    Returns
    -------
    pd.Series[bool]
        True  = tradable on that date.
        False = suspended / delisted / at limit — should be excluded from
                factor scoring and backtesting.
    """
    close = df_price["close"]
    high  = df_price["high"]
    low   = df_price["low"]
    vol   = df_price["vol"]

    # 1. Suspended: zero volume
    suspended = vol == 0

    # 2. Delisted: missing or zero close price
    delisted = close.isna() | (close == 0)

    # 3. Limit-up / limit-down:
    #    pct_change is computed per stock (unstack → pct_change → stack).
    #    The 9.5% threshold (vs. the formal 10%) gives a small safety margin
    #    for ST stocks whose limit is ±5%, and for minor float precision.
    pct_chg = (
        close.unstack("code")
             .pct_change()
             .stack(future_stack=True)
             .reindex(close.index)
    )
    large_move = pct_chg.abs() > 0.095
    at_high    = (high - close).abs() <= high.abs() * 1e-4
    at_low     = (close - low).abs() <= low.abs()  * 1e-4
    limit_hit  = large_move & (at_high | at_low)

    not_tradable = suspended | delisted | limit_hit
    return ~not_tradable   # True = tradable


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def main() -> None:
    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve DB path the same way DataEngine does (relative to src/).
    db_path = (ROOT / "src" / config.DB_PATH).resolve()

    # ----------------------------------------------------------------
    # Step 1: Verify database exists
    # ----------------------------------------------------------------
    _step("Step 1 / 6  —  Database check")

    if not db_path.exists():
        _err(f"Database not found at: {db_path}")
        _err("Run DataEngine().init_db() then engine.download_data() first.")
        sys.exit(1)

    _ok(f"Database found : {db_path}")

    # ----------------------------------------------------------------
    # Step 2: Load raw data
    # ----------------------------------------------------------------
    _step("Step 2 / 6  —  Loading data from SQLite")

    engine = DataEngine()
    data = engine.load_data()

    df_price    = data["df_price"]     # MultiIndex (date, code), cols: open/high/low/close/vol/amount
    df_mv       = data["df_mv"]        # MultiIndex (date, code), col: total_mv
    df_industry = data["df_industry"]  # index = code, cols: name, industry
    df_adj      = data["df_adj"]       # MultiIndex (date, code), col: adj_factor

    _ok(f"daily_price  : {df_price.shape[0]:>7,} rows  |  cols: {df_price.columns.tolist()}")
    _ok(f"df_mv        : {df_mv.shape[0]:>7,} rows")
    _ok(f"stock_info   : {len(df_industry):>7,} stocks  "
        f"|  {df_industry['industry'].nunique()} industries")
    _ok(f"adj_factor   : {df_adj.shape[0]:>7,} rows")

    # Load pe / pb alongside total_mv for the meta table.
    # DataEngine.load_data() exposes only total_mv, so we query directly.
    conn = sqlite3.connect(str(db_path))
    df_basic = pd.read_sql(
        "SELECT code, date, pe, pb, total_mv FROM daily_basic", conn
    )
    conn.close()
    df_basic = df_basic.set_index(["date", "code"]).sort_index()
    _ok(f"daily_basic  : {df_basic.shape[0]:>7,} rows  |  cols: {df_basic.columns.tolist()}")

    # ----------------------------------------------------------------
    # Step 2.5: Compute tradable mask
    # ----------------------------------------------------------------
    _step("Step 2.5 / 6  —  Computing tradable mask  (suspended / delisted / limit-hit)")

    tradable_mask = _compute_tradable(df_price)   # Series[bool], MultiIndex (date, code)

    n_total       = len(tradable_mask)
    n_tradable    = tradable_mask.sum()
    n_suspended   = (df_price["vol"] == 0).sum()
    n_delisted    = (df_price["close"].isna() | (df_price["close"] == 0)).sum()
    _ok(f"Total stock-days  : {n_total:>10,}")
    _ok(f"Tradable          : {n_tradable:>10,}  ({n_tradable / n_total * 100:.2f}%)")
    _info(f"Suspended (vol=0) : {n_suspended:>10,}")
    _info(f"Delisted  (px=0)  : {n_delisted:>10,}")
    _info(f"Limit-hit         : {(n_total - n_tradable - n_suspended - n_delisted):>10,}  (approx)")

    # ----------------------------------------------------------------
    # Step 3: Compute raw alpha factors
    # ----------------------------------------------------------------
    _step("Step 3 / 6  —  Computing raw alpha factors  (Alpha101, forward-adj)")

    alpha_engine  = Alpha101(data, adj_type="forward")
    df_alphas_raw = alpha_engine.get_all_alphas()   # MultiIndex (date, code) × alpha cols

    alpha_cols = df_alphas_raw.columns.tolist()
    nan_pct    = df_alphas_raw.isna().mean().mean() * 100

    _ok(f"Factors      : {len(alpha_cols)}  [{', '.join(alpha_cols)}]")
    _ok(f"Shape        : {df_alphas_raw.shape[0]:,} rows × {df_alphas_raw.shape[1]} factors")
    _info(f"Avg NaN rate : {nan_pct:.2f}%  (expected; time-series warmup period)")

    # ----------------------------------------------------------------
    # Step 4: Clean alpha factors
    # ----------------------------------------------------------------
    _step("Step 4 / 6  —  Cleaning alpha factors  (FactorCleaner, MAD + neutralize)")

    cleaner         = FactorCleaner(data)
    df_alphas_clean = cleaner.process_all(df_alphas_raw)   # NaN filled with 0 internally

    # Re-apply tradable mask: override process_all's fillna(0) for non-tradable
    # stock-days so that downstream backtester dropna naturally excludes them.
    tradable_aligned = tradable_mask.reindex(df_alphas_clean.index, fill_value=False)
    df_alphas_clean.loc[~tradable_aligned] = np.nan

    zero_pct = (df_alphas_clean == 0.0).mean().mean() * 100
    nan_pct_clean = df_alphas_clean.isna().mean().mean() * 100
    _ok(f"Shape        : {df_alphas_clean.shape[0]:,} rows × {df_alphas_clean.shape[1]} factors")
    _info(f"Zero %       : {zero_pct:.2f}%  (tradable stocks excluded from neutralization → 0)")
    _info(f"NaN %        : {nan_pct_clean:.2f}%  (non-tradable stock-days → NaN)")

    # ----------------------------------------------------------------
    # Step 5: Export Parquet files
    # ----------------------------------------------------------------
    _step("Step 5 / 6  —  Exporting Parquet files  →  ./data/")

    # -- prices.parquet -------------------------------------------------
    # Raw OHLCV (vol renamed to volume) + adj_factor + tradable flag.
    df_prices = (
        df_price[["open", "high", "low", "close", "vol"]]
        .rename(columns={"vol": "volume"})
        .join(df_adj["adj_factor"], how="left")
    )
    df_prices["tradable"] = tradable_mask.reindex(df_prices.index, fill_value=False)
    prices_out  = _flatten(df_prices)
    prices_path = out_dir / "prices.parquet"
    prices_out.to_parquet(prices_path, index=False)
    _ok(
        f"prices.parquet        : {prices_out.shape[0]:>7,} rows  "
        f"|  cols: {prices_out.columns.tolist()}"
    )

    # -- meta.parquet ---------------------------------------------------
    # Daily fundamentals (pe, pb, total_mv) + static industry per stock.
    industry_series = df_industry["industry"]          # Series indexed by code
    df_meta = df_basic[["total_mv", "pe", "pb"]].copy()
    df_meta["industry"] = (
        df_meta.index.get_level_values("code").map(industry_series)
    )
    df_meta = df_meta[["total_mv", "industry", "pe", "pb"]]
    meta_out  = _flatten(df_meta)
    meta_path = out_dir / "meta.parquet"
    meta_out.to_parquet(meta_path, index=False)
    _ok(
        f"meta.parquet          : {meta_out.shape[0]:>7,} rows  "
        f"|  cols: {meta_out.columns.tolist()}"
    )

    # -- factors_raw.parquet --------------------------------------------
    raw_out  = _flatten(df_alphas_raw)
    raw_path = out_dir / "factors_raw.parquet"
    raw_out.to_parquet(raw_path, index=False)
    _ok(f"factors_raw.parquet   : {raw_out.shape[0]:>7,} rows  |  {len(alpha_cols)} alpha cols")

    # -- factors_clean.parquet ------------------------------------------
    # Non-tradable cells are NaN (not 0); backtester dropna excludes them.
    clean_out  = _flatten(df_alphas_clean)
    clean_path = out_dir / "factors_clean.parquet"
    clean_out.to_parquet(clean_path, index=False)
    _ok(f"factors_clean.parquet : {clean_out.shape[0]:>7,} rows  |  {len(alpha_cols)} alpha cols  "
        f"|  non-tradable cells = NaN")

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    _step("Step 6 / 6  —  Summary")
    _ok("Phase 1 data preparation complete.")
    _ok(f"All files saved to : {out_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()
