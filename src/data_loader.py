"""
DataEngine: data acquisition and persistence layer for the multi-factor model.

Data flow:
    Tushare Pro API  -->  DataEngine.download_data()  -->  SQLite (stock_data.db)
    SQLite           -->  DataEngine.load_data()       -->  pandas DataFrames

Database tables:
    daily_price  : daily OHLCV + amount bars  (PK: code, date)
    daily_basic  : daily fundamentals         (PK: code, date)
    stock_info   : static stock metadata      (PK: code)
    adj_factor   : price adjustment factors   (PK: code, date)
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
      daily_price and adj_factor maintain independent cache sets so that
      each can be supplemented without re-downloading the other.
      Delete stock_data.db to force a full re-download of everything.
    - load_data() reads from the local SQLite and returns structured DataFrames.

    Typical usage:
        engine = DataEngine()
        engine.init_db()         # idempotent; safe to call every time
        engine.download_data()   # skips already-cached stocks
        data = engine.load_data()
        df_price    = data["df_price"]     # MultiIndex (date, code)
        df_mv       = data["df_mv"]        # MultiIndex (date, code)
        df_industry = data["df_industry"]  # indexed by code
        df_adj      = data["df_adj"]       # MultiIndex (date, code)
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
        """
        Create all tables if they do not already exist (idempotent).

        Also runs schema migrations for databases built with an older version:
          - adds 'amount' column to daily_price if missing
          - creates adj_factor table if missing
        """
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS daily_price (
                code    TEXT  NOT NULL,
                date    TEXT  NOT NULL,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                vol     REAL,
                amount  REAL,
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

            CREATE TABLE IF NOT EXISTS adj_factor (
                code        TEXT  NOT NULL,
                date        TEXT  NOT NULL,
                adj_factor  REAL,
                PRIMARY KEY (code, date)
            );

            CREATE TABLE IF NOT EXISTS stock_st (
                code        TEXT  NOT NULL,
                start_date  TEXT  NOT NULL,
                end_date    TEXT
            );
            """
        )
        conn.commit()

        # Schema migrations – SQLite does not support ADD COLUMN IF NOT EXISTS,
        # so each migration is wrapped in try/except OperationalError.

        # daily_price: add 'amount' column
        try:
            conn.execute("ALTER TABLE daily_price ADD COLUMN amount REAL")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        # stock_info: add 'list_date' column
        try:
            conn.execute("ALTER TABLE stock_info ADD COLUMN list_date TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        # daily_basic: add 13 extended fundamental columns
        _new_basic_cols = [
            "turnover_rate REAL",
            "turnover_rate_f REAL",
            "volume_ratio REAL",
            "pe_ttm REAL",
            "ps REAL",
            "ps_ttm REAL",
            "dv_ratio REAL",
            "dv_ttm REAL",
            "total_share REAL",
            "float_share REAL",
            "free_share REAL",
            "circ_mv REAL",
        ]
        for col_def in _new_basic_cols:
            try:
                conn.execute(f"ALTER TABLE daily_basic ADD COLUMN {col_def}")
                conn.commit()
            except sqlite3.OperationalError:
                pass

        conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_constituents(self) -> List[str]:
        """Return the index constituents as of the backtest start date.

        Queries a 31-day window anchored at config.START_DATE so that the
        universe reflects the index composition at the *beginning* of the
        study period.  Using today's constituents would introduce look-ahead
        bias because the CSI 300 continuously replaces weaker stocks with
        stronger ones, leaking future information into the stock universe.
        """
        start_dt = datetime.strptime(config.START_DATE, "%Y%m%d")
        end_dt = start_dt + timedelta(days=31)
        df = self.pro.index_weight(
            index_code=config.UNIVERSE_INDEX,
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            raise RuntimeError(
                f"No constituents returned for index '{config.UNIVERSE_INDEX}' "
                f"around start date {config.START_DATE}. "
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

        Independent cache sets are maintained for daily_price, adj_factor, and
        stock_st so that each can be supplemented on subsequent runs without
        re-downloading the others.  Delete stock_data.db to force a full
        re-download of everything.

        Steps:
            1. Fetch constituent list via index_weight.
            2. Fetch stock metadata (name, industry, list_date) via stock_basic.
            3. For each constituent: fetch daily OHLCV+amount and extended
               daily fundamentals (15 fields).
            4. For each constituent: fetch adj_factor history.
            5. For each constituent: fetch ST status history via stock_st.
        """
        codes = self._get_constituents()
        print(f"Universe: {len(codes)} stocks  [{config.UNIVERSE_INDEX}]")
        print(f"Date range: {config.START_DATE} -> {config.END_DATE}")

        conn = sqlite3.connect(self.db_path)

        # ---- Step 1: Stock metadata (name, industry, list_date) ----
        print("Fetching stock metadata (name, industry, list_date)...")
        info_df = self.pro.stock_basic(
            fields="ts_code,name,industry,list_date",
            list_status="L",
        )
        if info_df is not None and not info_df.empty:
            info_df = info_df[info_df["ts_code"].isin(codes)].copy()
            info_df = self._rename(info_df)
            conn.execute("DELETE FROM stock_info")
            info_df.to_sql("stock_info", conn, if_exists="append", index=False)
            conn.commit()

        # ---- Step 2: Daily price + extended fundamentals per stock ----
        price_cached: set = set(
            pd.read_sql("SELECT DISTINCT code FROM daily_price", conn)[
                "code"
            ].tolist()
        )
        n = len(codes)
        skipped_price = 0

        _basic_fields = (
            "ts_code,trade_date,"
            "turnover_rate,turnover_rate_f,volume_ratio,"
            "pe,pe_ttm,pb,ps,ps_ttm,"
            "dv_ratio,dv_ttm,"
            "total_share,float_share,free_share,total_mv,circ_mv"
        )

        print("Downloading daily price + fundamentals...")
        for i, ts_code in enumerate(codes):
            if ts_code in price_cached:
                skipped_price += 1
                continue

            try:
                # OHLCV + amount
                time.sleep(config.SLEEP_PER_CALL)
                price_df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields="ts_code,trade_date,open,high,low,close,vol,amount",
                )

                # Extended fundamentals (15 fields)
                time.sleep(config.SLEEP_PER_CALL)
                basic_df = self.pro.daily_basic(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields=_basic_fields,
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
                print(f"  Progress: {i + 1 - skipped_price} downloaded / {n} total")

        conn.commit()
        print(f"Price done. Skipped (already cached): {skipped_price}.")

        # ---- Step 3: Adjustment factors per stock (independent cache) ----
        adj_cached: set = set(
            pd.read_sql("SELECT DISTINCT code FROM adj_factor", conn)[
                "code"
            ].tolist()
        )
        skipped_adj = 0

        print("Downloading adj_factor...")
        for i, ts_code in enumerate(codes):
            if ts_code in adj_cached:
                skipped_adj += 1
                continue

            try:
                time.sleep(config.SLEEP_PER_CALL)
                adj_df = self.pro.adj_factor(
                    ts_code=ts_code,
                    start_date=config.START_DATE,
                    end_date=config.END_DATE,
                    fields="ts_code,trade_date,adj_factor",
                )

                if adj_df is not None and not adj_df.empty:
                    adj_df = self._rename(adj_df)
                    adj_df.to_sql(
                        "adj_factor", conn, if_exists="append", index=False
                    )

            except Exception as exc:
                print(f"  [WARN] adj_factor {ts_code}: {exc}")

            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"  Adj progress: {i + 1 - skipped_adj} / {n}")

        conn.commit()
        print(f"Adj done. Skipped (already cached): {skipped_adj}.")

        # ---- Step 4: ST status history per stock (independent cache) ----
        st_cached: set = set(
            pd.read_sql("SELECT DISTINCT code FROM stock_st", conn)[
                "code"
            ].tolist()
        )
        skipped_st = 0

        print("Downloading ST status history...")
        for i, ts_code in enumerate(codes):
            if ts_code in st_cached:
                skipped_st += 1
                continue

            try:
                time.sleep(config.SLEEP_PER_CALL)
                st_df = self.pro.stock_st(ts_code=ts_code)

                if st_df is not None and not st_df.empty:
                    st_df = st_df.rename(columns={"ts_code": "code"})
                    # Keep only relevant columns; end_date may be null for ongoing ST.
                    # Drop rows with null start_date to satisfy NOT NULL constraint
                    # (Tushare sometimes returns incomplete ST records).
                    keep_cols = [c for c in ["code", "start_date", "end_date"] if c in st_df.columns]
                    st_df = st_df.dropna(subset=["start_date"])
                    if not st_df.empty:
                        st_df[keep_cols].to_sql(
                            "stock_st", conn, if_exists="append", index=False
                        )
                    else:
                        # All rows had null start_date; mark as cached with no ST history.
                        pd.DataFrame([{"code": ts_code, "start_date": "99991231", "end_date": "99991231"}]).to_sql(
                            "stock_st", conn, if_exists="append", index=False
                        )
                else:
                    # Insert a sentinel row so the stock is marked as cached
                    # (no ST history = always tradable from ST perspective).
                    pd.DataFrame([{"code": ts_code, "start_date": "99991231", "end_date": "99991231"}]).to_sql(
                        "stock_st", conn, if_exists="append", index=False
                    )

            except Exception as exc:
                print(f"  [WARN] stock_st {ts_code}: {exc}")

            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"  ST progress: {i + 1 - skipped_st} / {n}")

        conn.commit()
        conn.close()
        print(
            f"Done. Price skipped: {skipped_price}, adj skipped: {skipped_adj}, "
            f"ST skipped: {skipped_st}.  DB: {self.db_path}"
        )

    def fetch_latest_adj_factor(self, codes: List[str]) -> pd.Series:
        """
        Fetch the most recent adj_factor for each stock directly from Tushare API.

        Tries up to 10 consecutive calendar days back from today until data is
        returned for the requested codes.  Returns a pd.Series indexed by ts_code.

        Used for forward-adjustment: P_forward = P_raw * adj_factor / adj_factor_latest.
        """
        for offset in range(10):
            query_date = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
            try:
                time.sleep(config.SLEEP_PER_CALL)
                df = self.pro.adj_factor(
                    trade_date=query_date,
                    fields="ts_code,adj_factor",
                )
                if df is not None and not df.empty:
                    df = df[df["ts_code"].isin(codes)]
                    if not df.empty:
                        return df.set_index("ts_code")["adj_factor"]
            except Exception:
                continue
        raise RuntimeError("Could not fetch latest adj_factor within the last 10 days.")

    # ------------------------------------------------------------------
    # Load for analysis
    # ------------------------------------------------------------------

    def load_data(self) -> dict:
        """
        Read all data from SQLite into memory.

        Returns:
            dict:
                "df_price"    : DataFrame with MultiIndex (date, code),
                                columns = [open, high, low, close, vol, amount]
                "df_mv"       : DataFrame with MultiIndex (date, code),
                                columns = [total_mv]   (kept for backward compat)
                "df_basic"    : DataFrame with MultiIndex (date, code),
                                columns = [turnover_rate, turnover_rate_f,
                                           volume_ratio, pe, pe_ttm, pb, ps,
                                           ps_ttm, dv_ratio, dv_ttm,
                                           total_share, float_share, free_share,
                                           total_mv, circ_mv]
                "df_industry" : DataFrame indexed by code,
                                columns = [name, industry, list_date]
                "df_adj"      : DataFrame with MultiIndex (date, code),
                                columns = [adj_factor]
                "df_st"       : DataFrame with columns [code, start_date, end_date]
                                (one row per ST status interval per stock;
                                 end_date may be NULL for ongoing ST)
        """
        conn = sqlite3.connect(self.db_path)

        price_df = pd.read_sql("SELECT * FROM daily_price", conn)
        price_df = price_df.set_index(["date", "code"]).sort_index()

        # Full extended daily_basic (15 fundamental fields)
        basic_df = pd.read_sql("SELECT * FROM daily_basic", conn)
        basic_df = basic_df.set_index(["date", "code"]).sort_index()

        # Backward-compatible df_mv (total market cap only)
        mv_df = basic_df[["total_mv"]].copy()

        info_df = pd.read_sql(
            "SELECT * FROM stock_info", conn, index_col="code"
        )

        adj_df = pd.read_sql("SELECT * FROM adj_factor", conn)
        adj_df = adj_df.set_index(["date", "code"]).sort_index()

        st_df = pd.read_sql("SELECT code, start_date, end_date FROM stock_st", conn)

        conn.close()

        return {
            "df_price": price_df,
            "df_mv": mv_df,
            "df_basic": basic_df,
            "df_industry": info_df,
            "df_adj": adj_df,
            "df_st": st_df,
        }
