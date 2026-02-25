"""
net_backtester.py
-----------------
Long-only net-return backtester with transaction costs.

Models a practical, real-world implementation of a long-only factor strategy
using an overlapping portfolio construction to smooth turnover across the
holding period d:

    Overlapping weight at T:
        daily_w_T[s]  = 1/|top20%_T|  if stock s is in top 20% on day T
        overlap_w_T   = mean(daily_w_T, daily_w_{T-1}, ..., daily_w_{T-d+1})

    Return accounting:
        GrossRet_T    = sum_s( overlap_w_{T-1,s} * R_{T,s} )
        Turnover_T    = 0.5 * ||overlap_w_T - overlap_w_{T-1}||_1
        NetRet_T      = GrossRet_T - Turnover_T * cost_rate

Performance metrics
-------------------
Cumulative Return, Annualised Return, Annualised Volatility, Sharpe Ratio,
Maximum Drawdown, Average Daily Turnover, Breakeven Turnover.

Public API
----------
    nb = NetReturnBacktester(alpha_df, prices_df, forward_days=5,
                             cost_rate=0.0035, rf=0.03, plots_dir=None)
    summary = nb.run_backtest()   # pd.Series of performance metrics
    fig     = nb.plot(show=False) # cumulative NAV chart, saved to plots_dir
"""

from __future__ import annotations

import math
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


class NetReturnBacktester:
    """Long-only overlapping-portfolio backtester with transaction costs.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Flat DataFrame with columns [trade_date, ts_code, <factor_col>].
        Exactly one column other than trade_date / ts_code is expected.
    prices_df : pd.DataFrame
        Flat DataFrame with at least columns [trade_date, ts_code, close].
    forward_days : int
        Rebalancing cycle length (= number of overlapping buckets).
    cost_rate : float
        Round-trip transaction cost rate (default 0.0035 = 35 bps).
    rf : float
        Annual risk-free rate for Sharpe and breakeven calculations (default 0.03).
    plots_dir : path-like, optional
        Directory where the chart is saved.  If None, no file is written.
    top_pct : float
        Fraction of top-ranked stocks to hold long (default 0.2 = top 20%).
    """

    TRADING_DAYS_PER_YEAR: int = 252

    def __init__(
        self,
        alpha_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        forward_days: int = 1,
        cost_rate: float = 0.0035,
        rf: float = 0.03,
        plots_dir: Optional[pathlib.Path | str] = None,
        top_pct: float = 0.20,
    ) -> None:
        # Identify factor column
        key_cols = {"trade_date", "ts_code"}
        factor_cols = [c for c in alpha_df.columns if c not in key_cols]
        if len(factor_cols) != 1:
            raise ValueError(
                f"alpha_df must have exactly one factor column; got {factor_cols}"
            )
        self.factor_col = factor_cols[0]
        self.forward_days = forward_days
        self.cost_rate = cost_rate
        self.rf = rf
        self.top_pct = top_pct
        self.plots_dir = pathlib.Path(plots_dir) if plots_dir is not None else None

        # Store raw inputs; heavy computation deferred to run_backtest()
        self._alpha_df = alpha_df
        self._prices_df = prices_df

        # Cache
        self._net_ret: Optional[pd.Series] = None
        self._gross_ret: Optional[pd.Series] = None
        self._turnover: Optional[pd.Series] = None
        self._summary: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_return_series(self) -> None:
        """Compute net/gross return and turnover series (vectorised)."""
        if self._net_ret is not None:
            return

        # Pivot to wide form (dates × stocks)
        alpha_wide = self._alpha_df.pivot(
            index="trade_date", columns="ts_code", values=self.factor_col
        )
        close_wide = self._prices_df.pivot(
            index="trade_date", columns="ts_code", values="close"
        )

        # Align to common dates and stocks
        common_dates = alpha_wide.index.intersection(close_wide.index)
        common_stocks = alpha_wide.columns.intersection(close_wide.columns)
        alpha_wide = alpha_wide.reindex(index=common_dates, columns=common_stocks)
        close_wide = close_wide.reindex(index=common_dates, columns=common_stocks)

        # Day-over-day stock returns
        stock_ret = close_wide.pct_change()

        # Daily G5 (top top_pct%) equal-weight portfolio
        pct_rank = alpha_wide.rank(axis=1, pct=True)
        in_top = pct_rank.gt(1.0 - self.top_pct).astype(float)
        row_sum = in_top.sum(axis=1).replace(0, np.nan)
        daily_w = in_top.div(row_sum, axis=0).fillna(0.0)

        # Overlapping weight: rolling mean over forward_days buckets
        overlap_w = daily_w.rolling(window=self.forward_days, min_periods=self.forward_days).mean()

        # Previous-day weights (used to compute return and turnover)
        w_prev = overlap_w.shift(1)

        # Gross return: yesterday's weights applied to today's stock returns
        gross_ret = (w_prev.fillna(0.0) * stock_ret.fillna(0.0)).sum(axis=1)

        # Turnover: half the L1 norm of weight change
        turnover = 0.5 * (overlap_w.fillna(0.0) - w_prev.fillna(0.0)).abs().sum(axis=1)

        # Net return
        net_ret = gross_ret - turnover * self.cost_rate

        # Trim: skip first forward_days rows (incomplete rolling window)
        trim = self.forward_days
        self._gross_ret = gross_ret.iloc[trim:].dropna()
        self._turnover  = turnover.iloc[trim:].dropna()
        self._net_ret   = net_ret.iloc[trim:].dropna()

    @staticmethod
    def _perf_metrics(
        r: pd.Series,
        rf: float,
        tdy: int,
    ) -> dict:
        """Compute standard performance metrics from a daily-return series."""
        r = r.dropna()
        n = len(r)
        if n == 0:
            return {k: float("nan") for k in
                    ["Cum Return", "Ann Return", "Ann Vol", "Sharpe", "Max DD"]}
        cum_val = (1 + r).cumprod()
        cum_ret = cum_val.iloc[-1] - 1
        ann_ret = (1 + cum_ret) ** (tdy / n) - 1
        ann_vol = r.std() * math.sqrt(tdy)
        sharpe  = (ann_ret - rf) / ann_vol if ann_vol != 0 else float("nan")
        mdd     = (cum_val / cum_val.cummax() - 1).min()
        return {
            "Cum Return": cum_ret,
            "Ann Return": ann_ret,
            "Ann Vol":    ann_vol,
            "Sharpe":     sharpe,
            "Max DD":     mdd,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(self) -> pd.Series:
        """Execute the backtest and return a performance metrics Series.

        Returns
        -------
        pd.Series
            Index = metric names; includes Cum Return, Ann Return, Ann Vol,
            Sharpe, Max DD, Avg Daily Turnover, Breakeven Turnover.
        """
        self._build_return_series()

        tdy = self.TRADING_DAYS_PER_YEAR
        metrics = self._perf_metrics(self._net_ret, self.rf, tdy)

        # Gross metrics (needed for breakeven turnover)
        gross_metrics = self._perf_metrics(self._gross_ret, self.rf, tdy)
        ann_gross_ret = gross_metrics["Ann Return"]

        # Additional cost-aware metrics
        avg_turnover = self._turnover.mean()
        # Breakeven: max daily turnover sustainable while still earning rf net
        breakeven = (ann_gross_ret - self.rf) / (self.cost_rate * tdy) if self.cost_rate > 0 else float("nan")

        metrics["Avg Daily Turnover"]  = avg_turnover
        metrics["Breakeven Turnover"]  = breakeven

        self._summary = pd.Series(metrics).round(6)
        return self._summary

    def plot(self, show: bool = False) -> plt.Figure:
        """Plot the cumulative net return NAV curve.

        Saves to ``plots_dir/{factor_col}_net.png`` when ``plots_dir`` is set.

        Parameters
        ----------
        show : bool
            Whether to display the chart interactively (default False).

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._build_return_series()

        cum_nav = (1 + self._net_ret).cumprod()

        # Convert trade_date index to datetime
        try:
            dates = pd.to_datetime(cum_nav.index.astype(str), format="%Y%m%d")
        except Exception:
            dates = pd.to_datetime(cum_nav.index, errors="coerce")

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(
            f"Net Return (after costs)  —  {self.factor_col}\n"
            f"cost_rate={self.cost_rate:.2%},  d={self.forward_days}d overlapping",
            fontsize=11,
            fontweight="bold",
        )

        ax.plot(dates, cum_nav.values, color="#1565C0", linewidth=1.8, label="Net NAV")
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Cumulative NAV")
        ax.legend(loc="upper left", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # Summary annotation in the corner
        if self._summary is not None:
            txt = (
                f"Ann Ret : {self._summary['Ann Return']:+.2%}\n"
                f"Sharpe  : {self._summary['Sharpe']:.3f}\n"
                f"Max DD  : {self._summary['Max DD']:.2%}\n"
                f"Avg TO  : {self._summary['Avg Daily Turnover']:.4f}"
            )
            ax.text(
                0.88, 0.05, txt,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", alpha=0.15),
            )

        # Sparse x-axis ticks
        locator   = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.DateFormatter("%Y-%m")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        if self.plots_dir is not None:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.plots_dir / f"{self.factor_col}_net.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        return fig
