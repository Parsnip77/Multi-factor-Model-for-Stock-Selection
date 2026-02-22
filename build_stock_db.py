"""
Build local SQLite database with CSI 300 constituent daily prices and fundamentals.

Uses Tushare Pro to download:
- CSI 300 constituent list (index_weight)
- Daily bars: code, date, open, high, low, close, vol
- Daily fundamentals: pb, pe, total_mv

Requires TUSHARE_TOKEN in environment. Run:
  export TUSHARE_TOKEN=your_token
  python build_stock_db.py
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts


# CSI 300 index code (Shenzhen). Alternative: '000300.SH'
CSI300_INDEX_CODE = "399300.SZ"
DB_PATH = "stock_data.db"
YEARS_BACK = 3
# Throttle to avoid Tushare rate limit (e.g. 500/min for basic)
SLEEP_PER_CALL = 0.2


def get_tushare_pro():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise SystemExit(
            "TUSHARE_TOKEN not set. Set it in environment, e.g.: export TUSHARE_TOKEN=your_token"
        )
    return ts.pro_api(token)


def get_csi300_constituents(pro):
    """Return list of ts_code for CSI 300 constituents (latest available date)."""
    # Use recent month to get current constituents
    end = datetime.now()
    start = end - timedelta(days=31)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    df = pro.index_weight(
        index_code=CSI300_INDEX_CODE,
        start_date=start_str,
        end_date=end_str,
    )
    if df is None or df.empty:
        # Fallback: try single trade_date
        df = pro.index_weight(index_code=CSI300_INDEX_CODE, trade_date=end_str)
    if df is None or df.empty:
        raise SystemExit(
            "Failed to get CSI 300 constituents. Check index_code and Tushare permissions."
        )
    # con_code is the constituent stock ts_code
    codes = df["con_code"].drop_duplicates().tolist()
    return codes


def date_range_yyyymmdd(years_back):
    end = datetime.now()
    start = end - timedelta(days=365 * years_back)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def fetch_daily_and_basic(pro, ts_code, start_date, end_date):
    """Fetch daily bars and daily_basic for one stock, merge on trade_date."""
    time.sleep(SLEEP_PER_CALL)
    daily = pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,open,high,low,close,vol",
    )
    time.sleep(SLEEP_PER_CALL)
    basic = pro.daily_basic(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,pb,pe,total_mv",
    )
    if daily is None or daily.empty:
        return None
    if basic is None or basic.empty:
        # Keep price data, fill fundamentals with NaN
        basic = daily[["ts_code", "trade_date"]].copy()
        basic["pb"] = float("nan")
        basic["pe"] = float("nan")
        basic["total_mv"] = float("nan")
    daily = daily.merge(
        basic[["trade_date", "pb", "pe", "total_mv"]],
        on="trade_date",
        how="left",
    )
    return daily


def create_daily_price_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_price (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            vol REAL,
            pb REAL,
            pe REAL,
            total_mv REAL,
            PRIMARY KEY (code, date)
        )
        """
    )
    conn.commit()


def main():
    start_date, end_date = date_range_yyyymmdd(YEARS_BACK)
    print(f"Date range: {start_date} .. {end_date}")

    pro = get_tushare_pro()
    codes = get_csi300_constituents(pro)
    print(f"CSI 300 constituents: {len(codes)} stocks")

    conn = sqlite3.connect(DB_PATH)
    create_daily_price_table(conn)

    # Align column names with spec: code, date, open, high, low, close, vol, pb, pe, total_mv
    rename = {"ts_code": "code", "trade_date": "date"}

    for i, ts_code in enumerate(codes):
        try:
            df = fetch_daily_and_basic(pro, ts_code, start_date, end_date)
            if df is None or df.empty:
                continue
            df = df.rename(columns=rename)
            df = df[
                [
                    "code",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "pb",
                    "pe",
                    "total_mv",
                ]
            ]
            # Remove existing rows for this code so re-run is idempotent
            conn.execute("DELETE FROM daily_price WHERE code = ?", (ts_code,))
            df.to_sql(
                "daily_price",
                conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
            )
        except Exception as e:
            print(f"Skip {ts_code}: {e}")
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(codes)}")
            conn.commit()

    conn.commit()
    conn.close()
    print(f"Done. Database: {DB_PATH}")


if __name__ == "__main__":
    main()
