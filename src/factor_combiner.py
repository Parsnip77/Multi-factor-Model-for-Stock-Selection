"""
factor_combiner.py
------------------
Rolling-window linear factor combination module.

Combines multiple alpha factors into a single synthetic factor using
a rolling OLS regression:

    Y_{t} = beta_1 * X_1_{t} + ... + beta_k * X_k_{t}

where Y is the forward return (or its cross-sectional rank) and X_i are
the individual factor values.  For each prediction date T, the model is
trained on [T-window, T-1] and used to score stocks on date T.  The
resulting cross-sectional scores are z-score normalised within each date.

Optionally, each cross-section's factor matrix is symmetrically
orthogonalised (Löwdin / symmetric orthogonalisation) before the OLS
step, which removes inter-factor collinearity and tends to improve the
stability of the regression coefficients.

Public API
----------
    synth_df = rolling_linear_combine(
        factors_df, target_df,
        factor_cols=['alpha042', 'alpha054', 'alpha038'],
        window=60,
        orthogonalize=True,   # set False to skip orthogonalization
    )
    # returns flat DataFrame: trade_date / ts_code / synthetic_factor
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def symmetric_orthogonalize(
    X: np.ndarray,
    min_eigenvalue: float = 1e-8,
) -> np.ndarray:
    """Apply Löwdin symmetric orthogonalization to a factor matrix.

    Transforms the columns of X into an orthonormal set while preserving
    the original column space as symmetrically as possible.  Useful for
    removing inter-factor collinearity before OLS regression.

    Parameters
    ----------
    X : np.ndarray, shape (N, K)
        Cross-sectional factor matrix (N stocks, K factors).  Each column
        should already be mean-centred (e.g. z-scored) before calling.
    min_eigenvalue : float
        Floor applied to eigenvalues of the covariance matrix to guard
        against near-singular matrices (default 1e-8).

    Returns
    -------
    np.ndarray, shape (N, K)
        Orthogonalized factor matrix whose columns satisfy X_orth.T @ X_orth ≈ N * I.
    """
    N = X.shape[0]
    C = X.T @ X / N                               # (K, K) sample covariance
    eigenvalues, V = np.linalg.eigh(C)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    C_inv_sqrt = V @ np.diag(eigenvalues ** -0.5) @ V.T
    return X @ C_inv_sqrt


def rolling_linear_combine(
    factors_df: pd.DataFrame,
    target_df: pd.DataFrame,
    factor_cols: List[str],
    window: int = 60,
    orthogonalize: bool = True,
) -> pd.DataFrame:
    """Synthesise a composite factor via a rolling OLS regression.

    Parameters
    ----------
    factors_df : pd.DataFrame
        Flat DataFrame with columns [trade_date, ts_code, *factor_cols, ...].
    target_df : pd.DataFrame
        Forward-return labels.  Accepts either:
        - MultiIndex (trade_date, ts_code) with column ``forward_return``
        - Flat DataFrame with columns [trade_date, ts_code, forward_return]
    factor_cols : list of str
        Names of the factor columns to combine (k >= 1).
    window : int
        Number of trading days used as the rolling training window (default 60).
    orthogonalize : bool
        If True (default), apply symmetric (Löwdin) orthogonalization to each
        cross-sectional factor matrix before the OLS step.  This removes
        inter-factor collinearity and stabilises regression coefficients.
        Set to False to use the raw factor matrix without transformation.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with columns [trade_date, ts_code, synthetic_factor].
        Only prediction dates are included (head window days are trimmed).
        Values are z-score normalised within each cross-section.
    """
    if len(factor_cols) < 1:
        raise ValueError("factor_cols must contain at least one factor name.")

    # Normalise target to flat form
    if isinstance(target_df.index, pd.MultiIndex):
        target_flat = target_df.reset_index()[["trade_date", "ts_code", "forward_return"]]
    else:
        target_flat = target_df[["trade_date", "ts_code", "forward_return"]].copy()

    # Merge factors with forward returns; keep only rows with complete data
    merged = pd.merge(
        factors_df[["trade_date", "ts_code"] + factor_cols],
        target_flat,
        on=["trade_date", "ts_code"],
        how="inner",
    ).dropna(subset=factor_cols + ["forward_return"])

    sorted_dates = sorted(merged["trade_date"].unique())
    n_dates = len(sorted_dates)

    if window >= n_dates:
        raise ValueError(
            f"window ({window}) must be smaller than the number of available dates "
            f"({n_dates})."
        )

    # Pre-index rows by date for fast slicing
    date_to_rows: dict = {d: merged[merged["trade_date"] == d] for d in sorted_dates}

    records: list[dict] = []

    for i in range(window, n_dates):
        pred_date = sorted_dates[i]
        train_dates = sorted_dates[i - window: i]

        # Build training matrix (optionally orthogonalize each cross-section)
        train_frames = [date_to_rows[d] for d in train_dates]
        train = pd.concat(train_frames, ignore_index=True)

        X_raw = train[factor_cols].values.astype(float)
        X_train = symmetric_orthogonalize(X_raw) if orthogonalize else X_raw
        y_train = train["forward_return"].values.astype(float)

        # OLS: find beta that minimises ||X_train @ beta - y_train||^2
        beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

        # Score today's stocks (apply the same transformation as training)
        today = factors_df[factors_df["trade_date"] == pred_date][
            ["ts_code"] + factor_cols
        ].dropna(subset=factor_cols)

        if today.empty:
            continue

        X_today_raw = today[factor_cols].values.astype(float)
        X_today = symmetric_orthogonalize(X_today_raw) if orthogonalize else X_today_raw
        scores = X_today @ beta

        for ts_code, score in zip(today["ts_code"].values, scores):
            records.append(
                {"trade_date": pred_date, "ts_code": ts_code, "synthetic_factor": score}
            )

    if not records:
        return pd.DataFrame(columns=["trade_date", "ts_code", "synthetic_factor"])

    result = pd.DataFrame(records)

    # Z-score normalise within each cross-section
    def _zscore(x: pd.Series) -> pd.Series:
        std = x.std()
        return (x - x.mean()) / std if std > 0 else pd.Series(0.0, index=x.index)

    result["synthetic_factor"] = result.groupby("trade_date")["synthetic_factor"].transform(
        _zscore
    )

    return result
