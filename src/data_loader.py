"""
DataEngine: data acquisition and persistence layer for the multi-factor model.

Data flow:
    Tushare Pro API  -->  DataEngine.download_data()  -->  SQLite (stock_data.db)
    SQLite           -->  DataEngine.load_data()       -->  pandas DataFrames

Database tables:
    daily_price  : daily OHLCV bars           (PK: code, date)
    daily_basic  : daily fundamentals         (PK: code, date)
    stock_info   : static stock metadata      (PK: code)
"""

import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import tushare as ts

# Allow this module to be imported from notebooks/ or any working directory.
sys.path.insert(0, str(Path(__file__).parent))
import config


class DataEngine:
    """
    Central class for all data I/O in the project.

    Implements a "download once, read repeatedly" caching strategy:
    - download_data() fetches from Tushare and persists to SQLite.
      Stocks already present in the database are skipped automatically.
      Delete stock_data.db to force a full re-download.
    - load_data() reads from the local SQLite and returns structured DataFrames.

    Typical usage:
        engine = DataEngine()
        engine.init_db()         # idempotent; safe to call every time
        engine.download_data()   # skips already-cached stocks
        data = engine.load_data()
        df_price    = data["df_price"]     # MultiIndex (date, code)
        df_mv       = data["df_mv"]        # MultiIndex (date, code)
        df_industry = data["df_industry"]  # indexed by code
    """

    def __init__(self):
        if config.TUSHARE_TOKEN in ("", "your_tushare_token"):
            raise ValueError(
                "TUSHARE_TOKEN is not set. Edit src/config.py and replace "
                "'your_tushare_token' with your actual token."
            )
        self.pro = ts.pro_api(config.TUSHARE_TOKEN)

        # Resolve DB path relative to the location of this source file (src/).
        # config.DB_PATH = '../data/stock_data.db'  =>  project_root/data/stock_data.db
        self.db_path: str = str(
            (Path(__file__).parent / config.DB_PATH).resolve()
        )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create all tables if they do not already exist (idempotent)."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS daily_price (
                code   TEXT  NOT NULL,
                date   TEXT  NOT NULL,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                vol    REAL,
                PRIMARY KEY (code, date)
            );

            CREATE TABLE IF NOT EXISTS daily_basic (
                code      TEXT  NOT NULL,
                date      TEXT  NOT NULL,
                pe        REAL,
                pb        REAL,
                total_mv  REAL,
                PRIMARY KEY (code, date)
            );

            CREATE TABLE IF NOT EXISTS stock_info (
                code      TEXT  PRIMARY KEY,
                name      TEXT,
                industry  TEXT
            );
            """
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_constituents(self) -> List[str]:
        """Return list of ts_codes for the configured universe index."""
        end = datetime.now()
        start = end - timedelta(days=31)
        df = self.pro.index_weight(
            index_code=config.UNIVERSE_INDEX,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            raise RuntimeError(
                f"No constituents returned for index '{config.UNIVERSE_INDEX}'. "
                "Check Tushare permissions (index_weight requires ~2000 points) "
                "or try '399300.SZ' instead of '000300.SH' in config.py."
            )
        return df["con_code"].drop_duplicates().tolist()

    @staticmethod
    def _rename(df: pd.DataFrame) -> pd.DataFrame:
        """Map Tushare column names to project conventions."""
        return df.rename(columns={"ts_code": "code", "trade_date": "date"})

    # ------------------------------------------------------------------
    # Download & persist
    # ------------------------------------------------------------------

    def download_data(self) -> None:
        """
        Download data for all universe constituents and persist to SQLite.

        Skips stocks that already have any rows in daily_price (cache hit).
        To force a full refresh, delete data/stock_data.db and re-run.

        Steps:
            1. Fetch constituent list via index_weight.
            2. Fetch stock metadata (name, industry) via stock_basic.
            3. For each constituent: fetch daily OHLCV and daily fundamentals.
        """
        codes = self._get_constituents()
        print(f"Universe: {len(codes)} stocks  [{config.UNIVERSE_INDEX}]")
        print(f"Date range: {config.START_DATE} -> {config.END_DATE}")

        conn = sqlite3.connect(self.db_path)

        # ---- Step 1: Stock metadata (industry classification) ----
        print("Fetching stock metadata (name, industry)...")
        info_df = self.pro.stock_basic(
            fields="ts_code,name,industry",
            list_status="L",
        )
        if info_df is not None and not info_df.empty:
            info_df = info_df[info_df["ts_code"].isin(codes)].copy()
            info_df = self._rename(info_df)
            conn.execute("DELETE FROM stock_info")
            info_df.to_sql("stock_info", conn, if_exists="append", index=False)
            conn.commit()

        # ---- Step 2: Daily price + fundamentals per stock ----
        cached: set = set(
            pd.read_sql("SELECT DISTINCT code FROM daily_price", conn)[
                "code"
            ].tolist()
        )
        n = len(codes)
        skipped = 0

        for i, ts_code in enumerate(codes):
            if ts_code in cached:
                skipped += 1
                continue

            try:
                # OHLCV
                time.sleep(config.SLEEP_PER_CALL)
                price_df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields="ts_code,trade_date,open,high,low,close,vol",
                )

                # Fundamentals (PE, PB, total market cap)
                time.sleep(config.SLEEP_PER_CALL)
                basic_df = self.pro.daily_basic(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields="ts_code,trade_date,pe,pb,total_mv",
                )

                if price_df is not None and not price_df.empty:
                    price_df = self._rename(price_df)
                    price_df.to_sql(
                        "daily_price", conn, if_exists="append", index=False
                    )

                if basic_df is not None and not basic_df.empty:
                    basic_df = self._rename(basic_df)
                    basic_df.to_sql(
                        "daily_basic", conn, if_exists="append", index=False
                    )

            except Exception as exc:
                print(f"  [WARN] {ts_code}: {exc}")

            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"  Progress: {i + 1 - skipped} downloaded / {n} total")

        conn.commit()
        conn.close()
        print(f"Done. Skipped (already cached): {skipped}. DB: {self.db_path}")

    # ------------------------------------------------------------------
    # Load for analysis
    # ------------------------------------------------------------------

    def load_data(self) -> dict:
        """
        Read all data from SQLite into memory.

        Returns:
            dict:
                "df_price"    : DataFrame with MultiIndex (date, code),
                                columns = [open, high, low, close, vol]
                "df_mv"       : DataFrame with MultiIndex (date, code),
                                columns = [total_mv]
                "df_industry" : DataFrame indexed by code,
                                columns = [name, industry]
        """
        conn = sqlite3.connect(self.db_path)

        price_df = pd.read_sql("SELECT * FROM daily_price", conn)
        price_df = price_df.set_index(["date", "code"]).sort_index()

        mv_df = pd.read_sql(
            "SELECT code, date, total_mv FROM daily_basic", conn
        )
        mv_df = mv_df.set_index(["date", "code"]).sort_index()

        info_df = pd.read_sql(
            "SELECT * FROM stock_info", conn, index_col="code"
        )

        conn.close()

        return {
            "df_price": price_df,
            "df_mv": mv_df,
            "df_industry": info_df,
        }
