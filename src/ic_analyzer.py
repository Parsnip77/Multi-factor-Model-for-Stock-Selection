"""
ic_analyzer.py
--------------
Factor IC (Information Coefficient) evaluation module.

Provides three public functions:
    calc_ic          -- cross-sectional Spearman IC time series
    calc_ic_metrics  -- summary statistics (mean, std, ICIR)
    plot_ic          -- IC bar chart + cumulative IC line chart
"""

from __future__ import annotations

import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates


def calc_ic(factors_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    """Compute daily cross-sectional Spearman IC between a single factor and forward return.

    Parameters
    ----------
    factors_df : pd.DataFrame
        Long-form DataFrame.  Must contain columns ``trade_date``, ``ts_code``,
        and exactly one factor value column.
    target_df : pd.DataFrame
        Long-form DataFrame with MultiIndex (trade_date, ts_code) and a column
        ``forward_return`` (output of ``calc_forward_return``).

    Returns
    -------
    pd.Series
        Time series of daily IC values indexed by ``trade_date``.
    """
    # Identify factor value column (everything except trade_date / ts_code)
    factor_cols = [c for c in factors_df.columns if c not in ("trade_date", "ts_code")]
    if len(factor_cols) != 1:
        raise ValueError(
            f"factors_df must contain exactly one factor column; got {factor_cols}"
        )
    factor_col = factor_cols[0]

    # Reset target index so we can merge on plain columns
    if isinstance(target_df.index, pd.MultiIndex):
        target_reset = target_df.reset_index()
    else:
        target_reset = target_df.copy()

    merged = pd.merge(
        factors_df[["trade_date", "ts_code", factor_col]],
        target_reset[["trade_date", "ts_code", "forward_return"]],
        on=["trade_date", "ts_code"],
        how="inner",
    ).dropna(subset=[factor_col, "forward_return"])

    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < 5:
            return float("nan")
        # Skip constant cross-sections to avoid ConstantInputWarning
        if group[factor_col].std() == 0 or group["forward_return"].std() == 0:
            return float("nan")
        return group[factor_col].corr(group["forward_return"], method="spearman")

    ic_series = (
        merged.groupby("trade_date")
        .apply(_spearman, include_groups=False)
        .rename(factor_col)
    )
    ic_series.index.name = "trade_date"
    return ic_series


def calc_ic_metrics(ic_series: pd.Series) -> dict:
    """Compute IC summary statistics.

    Parameters
    ----------
    ic_series : pd.Series
        Daily IC values, as returned by ``calc_ic``.

    Returns
    -------
    dict
        Keys: ``ic_mean``, ``ic_std``, ``icir``.
    """
    ic_clean = ic_series.dropna()
    ic_mean = ic_clean.mean()
    ic_std = ic_clean.std()
    icir = ic_mean / ic_std if ic_std != 0 else float("nan")
    return {"ic_mean": ic_mean, "ic_std": ic_std, "icir": icir}


def plot_ic(
    ic_series: pd.Series,
    factor_name: str = "",
    show: bool = True,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """Plot IC time series (bar) and cumulative IC (line) in two subplots.

    Parameters
    ----------
    ic_series : pd.Series
        Daily IC values indexed by trade_date.
    factor_name : str, optional
        Factor name used in the chart title.
    show : bool, optional
        Whether to call ``plt.show()`` automatically (default True).
    save_path : path-like, optional
        If provided, save the figure to this path before showing.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ic_clean = ic_series.dropna()
    cum_ic = ic_clean.cumsum()

    # Convert string index (e.g. "20220103") to datetime for proper axis ticks
    try:
        dates = pd.to_datetime(ic_clean.index.astype(str), format="%Y%m%d")
    except Exception:
        dates = pd.to_datetime(ic_clean.index, errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(
        f"IC Analysis  —  {factor_name}" if factor_name else "IC Analysis",
        fontsize=13,
        fontweight="bold",
    )

    # ---- IC time series bar chart ----------------------------------------
    ax_bar = axes[0]
    colors = ["#2196F3" if v >= 0 else "#F44336" for v in ic_clean.values]
    ax_bar.bar(dates, ic_clean.values, color=colors, width=1.5, alpha=0.8)
    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bar.axhline(ic_clean.mean(), color="#FF9800", linewidth=1.2,
                   linestyle="-", label=f"Mean = {ic_clean.mean():.4f}")
    ax_bar.set_ylabel("IC")
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_bar.set_title("IC Time Series", fontsize=10)

    # ---- Cumulative IC line chart ----------------------------------------
    ax_cum = axes[1]
    ax_cum.plot(dates, cum_ic.values, color="#4CAF50", linewidth=1.5)
    ax_cum.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_cum.fill_between(dates, 0, cum_ic.values,
                        where=cum_ic.values >= 0, alpha=0.15, color="#4CAF50")
    ax_cum.fill_between(dates, 0, cum_ic.values,
                        where=cum_ic.values < 0, alpha=0.15, color="#F44336")
    ax_cum.set_ylabel("Cumulative IC")
    ax_cum.set_title("Cumulative IC", fontsize=10)

    # ---- X-axis: sparse, readable date ticks ----------------------------
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter("%Y-%m")
    ax_cum.xaxis.set_major_locator(locator)
    ax_cum.xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
