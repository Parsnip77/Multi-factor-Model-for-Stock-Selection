"""
data_preparation_main.py — Phase 1 end-to-end pipeline for the multi-factor model.

Execution order:
    1. Verify that the local SQLite database exists.
    2. Load raw data via DataEngine.
    3. Compute raw alpha factors via Alpha101.
    4. Clean alpha factors via FactorCleaner.
    5. Export four Parquet files to ./data/:
         prices.parquet        — raw OHLCV + adj_factor, keyed (trade_date, ts_code)
         meta.parquet          — market-cap, industry, PE, PB, keyed (trade_date, ts_code)
         factors_raw.parquet   — raw alpha values, keyed (trade_date, ts_code)
         factors_clean.parquet — cleaned alpha values, keyed (trade_date, ts_code)

All four tables share the same logical primary key (trade_date, ts_code) and
can be joined freely on those two columns.

Usage:
    python data_preparation_main.py
"""

import sqlite3
import sys
import warnings
from pathlib import Path

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
    _step("Step 1 / 5  —  Database check")

    if not db_path.exists():
        _err(f"Database not found at: {db_path}")
        _err("Run DataEngine().init_db() then engine.download_data() first.")
        sys.exit(1)

    _ok(f"Database found : {db_path}")

    # ----------------------------------------------------------------
    # Step 2: Load raw data
    # ----------------------------------------------------------------
    _step("Step 2 / 5  —  Loading data from SQLite")

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
    # Step 3: Compute raw alpha factors
    # ----------------------------------------------------------------
    _step("Step 3 / 5  —  Computing raw alpha factors  (Alpha101, forward-adj)")

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
    _step("Step 4 / 5  —  Cleaning alpha factors  (FactorCleaner, MAD + neutralize)")

    cleaner         = FactorCleaner(data)
    df_alphas_clean = cleaner.process_all(df_alphas_raw)   # NaN filled with 0

    zero_pct = (df_alphas_clean == 0.0).mean().mean() * 100
    _ok(f"Shape        : {df_alphas_clean.shape[0]:,} rows × {df_alphas_clean.shape[1]} factors")
    _info(f"Zero-fill %  : {zero_pct:.2f}%  (stocks excluded from neutralization → 0)")

    # ----------------------------------------------------------------
    # Step 5: Export Parquet files
    # ----------------------------------------------------------------
    _step("Step 5 / 5  —  Exporting Parquet files  →  ./data/")

    # -- prices.parquet -------------------------------------------------
    # Raw OHLCV (vol renamed to volume) joined with adj_factor.
    df_prices = (
        df_price[["open", "high", "low", "close", "vol"]]
        .rename(columns={"vol": "volume"})
        .join(df_adj["adj_factor"], how="left")
    )
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
    clean_out  = _flatten(df_alphas_clean)
    clean_path = out_dir / "factors_clean.parquet"
    clean_out.to_parquet(clean_path, index=False)
    _ok(f"factors_clean.parquet : {clean_out.shape[0]:>7,} rows  |  {len(alpha_cols)} alpha cols")

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print("  Phase 1 data preparation complete.")
    print(f"  All files saved to : {out_dir.resolve()}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
