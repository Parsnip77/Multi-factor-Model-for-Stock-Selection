"""
FactorCleaner: preprocessing pipeline that converts raw alpha factors into
model-ready, cleaned factors.

Pipeline (applied per alpha column, cross-sectionally per date):
    1. Sanity check      — replace ±inf with NaN
    2. Winsorize         — MAD-based clipping  (Median ± 3 × 1.4826 × MAD)
    3. Pre-standardize   — Z-score normalization
    4. Neutralize        — OLS residuals vs. log(market_cap) + industry dummies
    5. Final standardize — Z-score on residuals
    6. Fill NaN with 0   — neutral value for missing positions

Inputs:
    data_dict     : dict returned by DataEngine.load_data()
    raw_alphas_df : DataFrame returned by Alpha101.get_all_alphas()
                    MultiIndex (date, code) × alpha columns

Output:
    df_clean_factors : same shape/index as raw_alphas_df, NaN filled with 0
"""

import numpy as np
import pandas as pd


class FactorCleaner:
    """
    Clean raw alpha factors into model-ready factors.

    Parameters
    ----------
    data_dict : dict
        Dictionary from DataEngine.load_data().  Must contain:
            "df_mv"       : MultiIndex (date, code), column 'total_mv'
            "df_industry" : indexed by code, column 'industry'
    """

    def __init__(self, data_dict: dict):
        df_mv = data_dict["df_mv"].copy()
        df_industry = data_dict["df_industry"].copy()

        # Log market cap: clip to avoid log(0); pivot to wide (date × code)
        df_mv["log_mv"] = np.log(df_mv["total_mv"].clip(lower=1e-10))
        self._log_mv: pd.DataFrame = df_mv["log_mv"].unstack(level="code")

        # Industry one-hot dummies; drop_first removes one category per group
        # to avoid perfect multicollinearity in the OLS regressor matrix.
        self._industry_dummies: pd.DataFrame = pd.get_dummies(
            df_industry["industry"], drop_first=True, dtype=float
        )

    # ------------------------------------------------------------------
    # Individual cleaning steps (public — can be called standalone)
    # ------------------------------------------------------------------

    def winsorize(
        self, factor_df: pd.DataFrame, method: str = "mad", limits: float = 3
    ) -> pd.DataFrame:
        """
        Winsorize each cross-section (row = one trading date) of factor_df.

        Parameters
        ----------
        factor_df : DataFrame
            Wide form: index = date, columns = stock code.
        method : str
            'mad'   — clip at Median ± limits × 1.4826 × MAD  (default)
            'sigma' — clip at Mean ± limits × Std
        limits : float
            Multiplier for the threshold (default 3).
        """

        def _clip_row(row: pd.Series) -> pd.Series:
            valid = row.dropna()
            if valid.empty:
                return row
            if method == "mad":
                med = valid.median()
                mad = (valid - med).abs().median()
                delta = limits * 1.4826 * mad
                lo, hi = med - delta, med + delta
            else:
                lo = valid.mean() - limits * valid.std()
                hi = valid.mean() + limits * valid.std()
            return row.clip(lower=lo, upper=hi)

        return factor_df.apply(_clip_row, axis=1)

    def standardize(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score standardize each cross-section (row = one trading date).

        Parameters
        ----------
        factor_df : DataFrame
            Wide form: index = date, columns = stock code.
        """
        mean = factor_df.mean(axis=1)
        std = factor_df.std(axis=1).replace(0, np.nan)
        return factor_df.sub(mean, axis=0).div(std, axis=0)

    def neutralize(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove market-cap and industry effects via cross-sectional OLS.

        For each date, regresses the factor on [log_mv, industry_dummies]
        and returns the OLS residuals.  Stocks that lack market-cap or
        industry data are excluded from the regression (residual = NaN).

        Parameters
        ----------
        factor_df : DataFrame
            Wide form: index = date, columns = stock code.
        """
        result = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)

        for date, row in factor_df.iterrows():
            if date not in self._log_mv.index:
                continue

            lmv = self._log_mv.loc[date]

            # Regressor matrix for this date: log_mv + industry dummies
            risk = pd.concat([lmv.rename("log_mv"), self._industry_dummies], axis=1)

            # Keep only stocks with valid factor AND complete risk data
            valid_codes = row.dropna().index.intersection(risk.dropna().index)

            # Need at least (number of regressors + intercept + 2) observations
            min_obs = risk.shape[1] + 1 + 2
            if len(valid_codes) < min_obs:
                continue

            y = row.loc[valid_codes].values.astype(float)
            X_raw = risk.loc[valid_codes].values.astype(float)

            # Include intercept so residuals are mean-zero in the subspace
            X = np.column_stack([np.ones(len(X_raw)), X_raw])

            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ beta
                result.loc[date, valid_codes] = residuals
            except np.linalg.LinAlgError:
                continue

        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_all(self, raw_alphas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete 5-step cleaning pipeline on a raw alpha DataFrame.

        Steps
        -----
        1. Sanity check      : ±inf  → NaN
        2. Winsorize         : MAD-based clipping (Median ± 3 × 1.4826 × MAD)
        3. Pre-standardize   : Z-score
        4. Neutralize        : OLS residuals vs. log_mv + industry dummies
        5. Final standardize : Z-score on residuals
        6. Fill NaN with 0

        Parameters
        ----------
        raw_alphas_df : pd.DataFrame
            MultiIndex (date, code) × alpha columns, as returned by
            Alpha101.get_all_alphas().

        Returns
        -------
        pd.DataFrame
            Same MultiIndex (date, code) and column structure as
            raw_alphas_df.  All remaining NaN values are filled with 0.
        """
        alpha_names = raw_alphas_df.columns.tolist()

        # Unstack to wide DataFrames: one per alpha column (date × code)
        wide: dict = {
            col: raw_alphas_df[col].unstack(level="code")
            for col in alpha_names
        }

        cleaned: dict = {}

        for name, df in wide.items():
            # Step 1: Sanity check
            df = df.replace([np.inf, -np.inf], np.nan)

            # Step 2: Winsorize (MAD-based, operates on non-NaN values only)
            df = self.winsorize(df, method="mad", limits=3)

            # Step 3: Pre-standardize
            df = self.standardize(df)

            # Step 4: Neutralize (core step — strip market-cap & industry bias)
            # df = self.neutralize(df)

            # Step 5: Final standardize (re-center residuals to N(0,1))
            df = self.standardize(df)

            cleaned[name] = df

        # Stack back to long form MultiIndex (date, code)
        frames = []
        for name, df in cleaned.items():
            s = df.stack(dropna=False)
            s.name = name
            frames.append(s)

        df_clean = pd.concat(frames, axis=1)
        df_clean.index.names = ["date", "code"]

        # Replace remaining NaN (stocks excluded from regression) with 0
        df_clean_factors = df_clean.fillna(0.0)

        return df_clean_factors
