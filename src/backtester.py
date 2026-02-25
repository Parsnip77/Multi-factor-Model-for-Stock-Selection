"""
backtester.py
-------------
Layered (quantile) backtest module.

Workflow
--------
1. Cross-sectional binning  : rank stocks by factor each day, qcut into N groups
2. Portfolio weighting      : equal-weight within each group
3. Group return series      : groupby(['trade_date', 'group']).mean().unstack()
4. Long-short spread        : G_N - G_1
5. Performance metrics      : cumulative return, annualized return, vol, Sharpe, MDD

Public API
----------
    bt = LayeredBacktester(factor_df, target_df, num_groups=5, rf=0.03, plots_dir=None)
    perf_table = bt.run_backtest()   # pd.DataFrame: rows=groups+LS, cols=metrics
    fig        = bt.plot(show=True)  # cumulative NAV chart, saved to plots_dir
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


class LayeredBacktester:
    """Quantile-based layered backtester for a single alpha factor.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Flat DataFrame with columns [trade_date, ts_code, <factor_col>].
        Exactly one column other than trade_date / ts_code is expected.
    target_df : pd.DataFrame
        Forward-return labels.  Accepts either:
        - MultiIndex (trade_date, ts_code) with column ``forward_return``
        - Flat DataFrame with columns [trade_date, ts_code, forward_return]
    num_groups : int
        Number of quantile groups (default 5 → G1 .. G5).
    rf : float
        Annual risk-free rate used for Sharpe ratio (default 0.03).
    plots_dir : path-like, optional
        Directory where the backtest chart is saved.  If None, no file is written.
    """

    TRADING_DAYS_PER_YEAR: int = 252

    def __init__(
        self,
        factor_df: pd.DataFrame,
        target_df: pd.DataFrame,
        num_groups: int = 5,
        rf: float = 0.03,
        plots_dir: Optional[pathlib.Path | str] = None,
    ) -> None:
        # Identify factor column
        key_cols = {"trade_date", "ts_code"}
        factor_cols = [c for c in factor_df.columns if c not in key_cols]
        if len(factor_cols) != 1:
            raise ValueError(
                f"factor_df must have exactly one factor column; got {factor_cols}"
            )
        self.factor_col = factor_cols[0]
        self.num_groups = num_groups
        self.rf = rf
        self.plots_dir = pathlib.Path(plots_dir) if plots_dir is not None else None
        self.labels = [f"G{i}" for i in range(1, num_groups + 1)]

        # Normalise target_df to flat form
        if isinstance(target_df.index, pd.MultiIndex):
            target_flat = target_df.reset_index()[["trade_date", "ts_code", "forward_return"]]
        else:
            target_flat = target_df[["trade_date", "ts_code", "forward_return"]].copy()

        # Merge and drop rows with any NaN in the key columns
        merged = pd.merge(
            factor_df[["trade_date", "ts_code", self.factor_col]],
            target_flat,
            on=["trade_date", "ts_code"],
            how="inner",
        ).dropna(subset=[self.factor_col, "forward_return"])

        self._merged: pd.DataFrame = merged

        # Cache computed results
        self._group_ret: Optional[pd.DataFrame] = None
        self._perf_table: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bin_and_group_returns(self) -> pd.DataFrame:
        """Assign group labels and compute equal-weight group return series."""
        if self._group_ret is not None:
            return self._group_ret

        df = self._merged.copy()

        def _assign_group(x: pd.Series) -> pd.Categorical:
            ranked = x.rank(method="first")
            return pd.qcut(ranked, q=self.num_groups, labels=self.labels)

        df["group"] = df.groupby("trade_date")[self.factor_col].transform(_assign_group)

        group_ret = (
            df.groupby(["trade_date", "group"])["forward_return"]
            .mean()
            .unstack(level="group")
        )
        # Ensure columns are in G1 .. GN order
        group_ret = group_ret.reindex(columns=self.labels)

        self._group_ret = group_ret
        return self._group_ret

    @staticmethod
    def _calc_perf_metrics(r: pd.Series, rf: float, tdy: int) -> dict:
        """Compute five performance metrics from a daily-return series."""
        r = r.dropna()
        if len(r) == 0:
            return {
                "Cum Return": float("nan"),
                "Ann Return": float("nan"),
                "Ann Vol": float("nan"),
                "Sharpe": float("nan"),
                "Max DD": float("nan"),
            }

        cum_val = (1 + r).cumprod()
        cum_ret = cum_val.iloc[-1] - 1
        n = len(r)
        ann_ret = (1 + cum_ret) ** (tdy / n) - 1
        ann_vol = r.std() * math.sqrt(tdy)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else float("nan")
        drawdown = cum_val / cum_val.cummax() - 1
        mdd = drawdown.min()

        return {
            "Cum Return": cum_ret,
            "Ann Return": ann_ret,
            "Ann Vol": ann_vol,
            "Sharpe": sharpe,
            "Max DD": mdd,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(self) -> pd.DataFrame:
        """Run the full layered backtest and return a performance metrics table.

        Returns
        -------
        pd.DataFrame
            Rows = G1 .. GN + LS, columns = performance metric names.
            Numeric values are rounded to 4 decimal places for readability.
        """
        group_ret = self._bin_and_group_returns()

        # Long-short spread: top group minus bottom group
        ls = group_ret[self.labels[-1]] - group_ret[self.labels[0]]
        ls.name = "L-S"

        all_series = {col: group_ret[col] for col in self.labels}
        all_series["L-S"] = ls

        rows = {}
        for name, series in all_series.items():
            rows[name] = self._calc_perf_metrics(series, self.rf, self.TRADING_DAYS_PER_YEAR)

        self._perf_table = pd.DataFrame(rows).T.round(4)
        return self._perf_table

    def plot(self, show: bool = True) -> plt.Figure:
        """Plot cumulative NAV curves for all groups and the L-S spread.

        The figure is saved to ``plots_dir/{factor_name}_backtest.png`` when
        ``plots_dir`` is set.

        Parameters
        ----------
        show : bool
            Whether to display the chart interactively (default True).

        Returns
        -------
        matplotlib.figure.Figure
        """
        group_ret = self._bin_and_group_returns()

        ls = group_ret[self.labels[-1]] - group_ret[self.labels[0]]
        ls.name = "L-S"

        # Cumulative NAV (starts at 1.0)
        nav_df = (1 + group_ret).cumprod()
        nav_ls = (1 + ls).cumprod()

        # Convert trade_date index to datetime
        try:
            dates = pd.to_datetime(nav_df.index.astype(str), format="%Y%m%d")
        except Exception:
            dates = pd.to_datetime(nav_df.index, errors="coerce")

        fig, ax = plt.subplots(figsize=(13, 6))
        fig.suptitle(
            f"Layered Backtest  —  {self.factor_col}  "
            f"({self.num_groups} groups,  L-S = {self.labels[-1]} - {self.labels[0]})",
            fontsize=12,
            fontweight="bold",
        )

        # G1..GN: use a sequential colormap
        cmap = plt.get_cmap("RdYlGn", self.num_groups)
        for i, col in enumerate(self.labels):
            ax.plot(dates, nav_df[col].values, color=cmap(i), linewidth=1.4, label=col)

        # Long-short: thick black dashed line
        ax.plot(dates, nav_ls.values, color="black", linewidth=2.0,
                linestyle="--", label="L-S")

        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Cumulative NAV")
        ax.legend(loc="upper left", fontsize=9, ncol=self.num_groups + 1)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # Sparse x-axis ticks
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.DateFormatter("%Y-%m")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        if self.plots_dir is not None:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.plots_dir / f"{self.factor_col}_backtest.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        return fig
