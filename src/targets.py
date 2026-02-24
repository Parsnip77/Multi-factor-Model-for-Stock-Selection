"""
targets.py
----------
Label generation module for forward returns.
"""

import pandas as pd


def calc_forward_return(prices_df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Calculate d-day forward return aligned to time T.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Flat DataFrame with columns [trade_date, ts_code, close, ...].
    d : int
        Number of trading days ahead to compute the return.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with MultiIndex (trade_date, ts_code) and a single
        column ``forward_return`` = (close_{T+d} - close_T) / close_T.
        Rows for which T+d falls outside the data range contain NaN.
    """
    close_wide = prices_df.pivot(index="trade_date", columns="ts_code", values="close")

    # shift(-d) brings future price back to current index position
    fwd_wide = close_wide.shift(-d) / close_wide - 1

    # Stack back to long form, preserving NaN for boundary dates
    fwd_long = (
        fwd_wide.stack(future_stack=True)
        .rename("forward_return")
        .reset_index()
    )
    fwd_long = fwd_long.set_index(["trade_date", "ts_code"])

    return fwd_long
