"""
Alpha101: selected alpha factors from '101 Formulaic Alphas' (Kakushadze, 2015).

Reference: https://arxiv.org/abs/1601.00991
Formulas are taken verbatim from Appendix A of the paper.

Implemented alphas:
    Alpha#1  : (rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close),2.),5)) - 0.5)
    Alpha#3  : (-1 * correlation(rank(open), rank(volume), 10))
    Alpha#6  : (-1 * correlation(open, volume, 10))
    Alpha#12 : (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    Alpha#21 : ((((sum(close,8)/8) + stddev(close,8)) < (sum(close,2)/2)) ? -1 :
               (((sum(close,2)/2) < ((sum(close,8)/8) - stddev(close,8))) ? 1 :
               (((1 < (volume/adv20)) || ((volume/adv20) == 1)) ? 1 : -1)))
    Alpha#38 : ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    Alpha#40 : ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    Alpha#41 : (((high * low)^0.5) - vwap)
    Alpha#42 : (rank((vwap - close)) / rank((vwap + close)))
    Alpha#54 : ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    Alpha#72 : (rank(decay_linear(correlation(((high+low)/2), adv40, 8.93345), 10.1519)) /
               rank(decay_linear(correlation(Ts_Rank(vwap,3.72469),
                                             Ts_Rank(volume,18.5188), 6.86671), 2.95011)))
    Alpha#88 : min(rank(decay_linear(((rank(open)+rank(low))-(rank(high)+rank(close))),8.06882)),
               Ts_Rank(decay_linear(correlation(Ts_Rank(close,8.44728),
                                                Ts_Rank(adv60,20.6966),8.01266),6.65053),2.61957))
    Alpha#94 : ((rank(vwap - ts_min(vwap,11.5783)) ^
               Ts_Rank(correlation(Ts_Rank(vwap,19.6462),
                                   Ts_Rank(adv60,4.02992), 18.0926), 2.70756)) * -1)
    Alpha#98 : (rank(decay_linear(correlation(vwap,sum(adv5,26.4719),4.58418),7.18088)) -
               rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open),
                                                   rank(adv15),20.8187),8.62571),6.95668),8.07206)))
    Alpha#101: ((close - open) / ((high - low) + .001))
    Alpha_5_day_reversal: (close - delay(close,5)) / delay(close,5)

Conventions:
    - All DataFrames are in wide form: index = date (str), columns = stock code (str).
    - rank()    : cross-sectional — ranks across stocks (columns) at each date (row).
    - ts_rank() : time-series    — ranks today's value within the past d days, per stock.
    - corr()    : time-series    — rolling pairwise correlation per stock column.
    - vwap      : computed as amount * 10 / vol (amount in thousand CNY, vol in lots);
                  falls back to (high + low + close) / 3 when amount is unavailable.
    - vol and amount are never adjusted, regardless of adj_type.

Price adjustment (adj_type):
    'forward'  (default): P_adj = P_raw * adj_factor / adj_factor_latest
                          latest = last row of stored adj_factor (or caller-supplied Series).
    'backward'           : P_adj = P_raw * adj_factor
    'raw'                : no adjustment applied.

Raw factor values (including NaN and inf) are preserved; downstream cleaning
is handled by a separate processing layer (to be implemented).
"""

from this import d
import numpy as np
import pandas as pd


class Alpha101:
    """
    Compute selected alphas from '101 Formulaic Alphas'.

    Input : the dictionary returned by DataEngine.load_data().
    Internally all series are pivoted to wide form (dates × codes) for
    efficient vectorized operations.

    Parameters
    ----------
    data_dict : dict
        Dictionary from DataEngine.load_data().  Must contain 'df_price'
        and 'df_mv'; 'df_adj' is required for any adj_type other than 'raw'.
    adj_type : str, default 'forward'
        Price adjustment mode.  One of:
            'forward'  — P * adj_factor / adj_factor_latest (latest date in range)
            'backward' — P * adj_factor
            'raw'      — no adjustment
    latest_adj : pd.Series or None
        Optional pre-fetched latest adj_factor values indexed by ts_code,
        e.g. from DataEngine.fetch_latest_adj_factor(codes).
        When None, the last row of df_adj is used as the reference for
        forward adjustment.

    Usage:
        data = DataEngine().load_data()

        alpha = Alpha101(data)                          # forward adj (default)
        alpha = Alpha101(data, adj_type='backward')     # backward adj
        alpha = Alpha101(data, adj_type='raw')          # no adjustment

        # For a more up-to-date forward reference factor:
        latest = engine.fetch_latest_adj_factor(codes)
        alpha  = Alpha101(data, adj_type='forward', latest_adj=latest)

        df_a6   = alpha.alpha006()       # wide DataFrame: dates × codes
        df_all  = alpha.get_all_alphas() # long DataFrame: MultiIndex (date, code) × alphas
    """

    def __init__(
        self,
        data_dict: dict,
        adj_type: str = "forward",
        latest_adj: pd.Series = None,
    ):
        if adj_type not in ("forward", "backward", "raw"):
            raise ValueError(f"adj_type must be 'forward', 'backward', or 'raw'; got '{adj_type}'")

        df_price = data_dict["df_price"]   # MultiIndex (date, code)
        df_mv    = data_dict["df_mv"]      # MultiIndex (date, code)
        df_adj   = data_dict.get("df_adj") # MultiIndex (date, code), may be absent

        # ---- Unstack raw price series (wide form: dates × codes) ----
        open_raw  = df_price["open"].unstack("code")
        high_raw  = df_price["high"].unstack("code")
        low_raw   = df_price["low"].unstack("code")
        close_raw = df_price["close"].unstack("code")
        self.vol  = df_price["vol"].unstack("code")   # never adjusted

        # ---- Precise VWAP: amount (thousand CNY) × 10 / vol (lots) ----
        # amount (千元) × 1000 / (vol (手) × 100) = amount × 10 / vol  (元/股)
        if "amount" in df_price.columns:
            amount_wide = df_price["amount"].unstack("code")
            vwap_raw = (amount_wide * 10).div(self.vol)
            # Fallback to typical price where amount or vol is missing / zero
            fallback = (high_raw + low_raw + close_raw) / 3
            valid = amount_wide.notna() & (self.vol > 0)
            vwap_raw = vwap_raw.where(valid, fallback)
        else:
            vwap_raw = (high_raw + low_raw + close_raw) / 3

        # ---- Price adjustment ----
        if adj_type != "raw" and df_adj is not None and not df_adj.empty:
            adj_wide = df_adj["adj_factor"].unstack("code")
            # Reindex to price dates; forward-fill gaps (non-trading days in adj table)
            adj_wide = adj_wide.reindex(open_raw.index).ffill()

            if adj_type == "forward":
                if latest_adj is None:
                    # Use the last available row in the stored date range as reference
                    latest_row = adj_wide.iloc[-1]
                else:
                    latest_row = latest_adj.reindex(adj_wide.columns)
                adj_ratio = adj_wide.div(latest_row, axis=1)
            else:  # 'backward'
                adj_ratio = adj_wide

            self.open  = open_raw.mul(adj_ratio)
            self.high  = high_raw.mul(adj_ratio)
            self.low   = low_raw.mul(adj_ratio)
            self.close = close_raw.mul(adj_ratio)
            self.vwap  = vwap_raw.mul(adj_ratio)   # vwap tracks the same price scale
        else:
            self.open  = open_raw
            self.high  = high_raw
            self.low   = low_raw
            self.close = close_raw
            self.vwap  = vwap_raw

        self.returns  = self.close.pct_change()
        self.total_mv = df_mv["total_mv"].unstack("code")

    # ------------------------------------------------------------------
    # Operator / function library (paper Appendix A.1, Functions)
    # ------------------------------------------------------------------

    def _rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional rank at each date, normalized to (0, 1].
        rank(x) = cross-sectional rank  [paper definition]
        """
        return df.rank(axis=1, pct=True)

    def _delay(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """delay(x, d) = value of x d days ago."""
        return df.shift(d)

    def _delta(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """delta(x, d) = today's value of x minus the value d days ago."""
        return df.diff(d)

    def _corr(self, df1: pd.DataFrame, df2: pd.DataFrame, d: int) -> pd.DataFrame:
        """
        correlation(x, y, d) = time-series rolling correlation over d days.
        Applied column-wise (per stock).
        """
        return df1.rolling(d).corr(df2)

    def _cov(self, df1: pd.DataFrame, df2: pd.DataFrame, d: int) -> pd.DataFrame:
        """covariance(x, y, d) = time-series rolling covariance over d days."""
        return df1.rolling(d).cov(df2)

    def _stddev(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """stddev(x, d) = rolling standard deviation over d days."""
        return df.rolling(d).std()

    def _sum(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """sum(x, d) = rolling sum over d days."""
        return df.rolling(d).sum()

    def _product(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """product(x, d) = rolling product over d days."""
        return df.rolling(d).apply(np.prod, raw=True)

    def _ts_min(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """ts_min(x, d) = rolling minimum over d days."""
        return df.rolling(d).min()

    def _ts_max(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """ts_max(x, d) = rolling maximum over d days."""
        return df.rolling(d).max()

    def _ts_argmax(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """ts_argmax(x, d) = 0-based offset of the rolling maximum over d days."""
        return df.rolling(d).apply(np.argmax, raw=True)

    def _ts_argmin(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """ts_argmin(x, d) = 0-based offset of the rolling minimum over d days."""
        return df.rolling(d).apply(np.argmin, raw=True)

    def _ts_rank(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """
        ts_rank(x, d) = time-series rank of today's value within the past d days.
        Normalized to (0, 1]: rank 1/d (lowest) to 1 (highest).
        Ties are broken by position (first occurrence = lower rank).
        """

        def _rank_last(arr: np.ndarray) -> float:
            # Double argsort: fast ordinal rank without scipy dependency.
            temp = arr.argsort()
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(1, len(arr) + 1)
            return ranks[-1] / len(arr)

        return df.rolling(d).apply(_rank_last, raw=True)

    def _scale(self, df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
        """
        scale(x, a=1) = cross-sectional rescaling so that sum(abs(x)) = a at each date.
        """
        abs_sum = df.abs().sum(axis=1).replace(0, np.nan)
        return df.div(abs_sum, axis=0) * a

    def _decay_linear(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """
        decay_linear(x, d) = weighted moving average over d days.
        Weights: d, d-1, ..., 1 (most recent = d, oldest = 1), normalized to sum 1.
        """
        weights = np.arange(1, d + 1, dtype=float)   # [1, 2, ..., d]
        weights /= weights.sum()
        return df.rolling(d).apply(lambda x: np.dot(x, weights), raw=True)

    def _sign(self, df: pd.DataFrame) -> pd.DataFrame:
        """sign(x): +1, 0, or -1 element-wise."""
        return np.sign(df)

    def _log(self, df: pd.DataFrame) -> pd.DataFrame:
        """log(x): natural logarithm element-wise."""
        return np.log(df)

    def _abs(self, df: pd.DataFrame) -> pd.DataFrame:
        """abs(x): absolute value element-wise."""
        return df.abs()

    def _signed_power(self, df: pd.DataFrame, e: float) -> pd.DataFrame:
        """signedpower(x, a) = sign(x) * |x|^a."""
        return np.sign(df) * (np.abs(df) ** e)

    # ------------------------------------------------------------------
    # Alpha implementations
    # ------------------------------------------------------------------

    def alpha001(self) -> pd.DataFrame:
        """
        Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)

        For each stock: if today's return is negative, substitute stddev(returns, 20) for
        close; otherwise keep close. Apply SignedPower(x, 2) to the result, then find
        which of the past 5 days had the maximum value (Ts_ArgMax). Cross-sectionally
        rank that position index and subtract 0.5 to centre around zero.
        """
        std20 = self._stddev(self.returns, 20)
        inner = np.where(self.returns < 0, std20, self.close)
        inner = pd.DataFrame(inner, index=self.close.index, columns=self.close.columns)
        sp = self._signed_power(inner, 2.0)
        return self._rank(self._ts_argmax(sp, 5)) - 0.5

    def alpha003(self) -> pd.DataFrame:
        """
        Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))

        Negative rolling correlation between cross-sectional ranks of open price and
        volume over a 10-day window. Penalises stocks whose open rank and volume rank
        move together (momentum-and-volume agreement is faded).
        """
        return -1 * self._corr(self._rank(self.open), self._rank(self.vol), 10)

    def alpha006(self) -> pd.DataFrame:
        """
        Alpha#6: (-1 * correlation(open, volume, 10))

        Negative time-series correlation between open price and volume
        over a 10-day rolling window, per stock.
        High (positive) correlation is penalized → contrarian tilt.
        """
        return -1 * self._corr(self.open, self.vol, 10)

    def alpha012(self) -> pd.DataFrame:
        """
        Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))

        When volume increases (sign = +1), take a short position in the
        close move (mean-reversion). When volume falls, follow the close move.
        Combines volume direction with intraday price change.
        """
        return self._sign(self._delta(self.vol, 1)) * (-1 * self._delta(self.close, 1))

    # alpha021 outputs only ±1 signals, which is incompatible with the layered
    # backtest framework (cross-sectional ranking requires continuous values).
    # Disabled to avoid spurious IC / backtest results.
    #
    # def alpha021(self) -> pd.DataFrame:
    #     """
    #     Alpha#21: ((((sum(close,8)/8) + stddev(close,8)) < (sum(close,2)/2)) ? (-1*1) :
    #                (((sum(close,2)/2) < ((sum(close,8)/8) - stddev(close,8))) ? 1 :
    #                (((1 < (volume/adv20)) || ((volume/adv20) == 1)) ? 1 : (-1*1))))
    #     """
    #     adv20 = self.vol.rolling(20).mean()
    #     sum8 = self._sum(self.close, 8) / 8
    #     sum2 = self._sum(self.close, 2) / 2
    #     std8 = self._stddev(self.close, 8)
    #     vol_ratio = self.vol / adv20
    #     cond1 = (sum8 + std8) < sum2
    #     cond2 = sum2 < (sum8 - std8)
    #     cond3 = vol_ratio >= 1
    #     result = np.where(cond1, -1, np.where(cond2, 1, np.where(cond3, 1, -1)))
    #     return pd.DataFrame(result, index=self.close.index, columns=self.close.columns)

    def alpha038(self) -> pd.DataFrame:
        """
        Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))

        Stocks that are near their recent highs (high ts_rank) AND have a
        high close-to-open ratio are shorted (both ranks are high → product large → -1 applied).
        """
        ts_rnk = self._ts_rank(self.close, 10)
        return (-1 * self._rank(ts_rnk)) * self._rank(self.close / self.open)

    def alpha040(self) -> pd.DataFrame:
        """
        Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))

        Stocks with high volatility in the high price (large stddev rank) are shorted
        when high price and volume are positively correlated over the past 10 days.
        Combines dispersion penalty with momentum-volume agreement.
        """
        return (-1 * self._rank(self._stddev(self.high, 10))) * self._corr(self.high, self.vol, 10)

    def alpha041(self) -> pd.DataFrame:
        """
        Alpha#41: (((high * low)^0.5) - vwap)

        Geometric mean of the day's high and low minus vwap.
        Positive when the geometric mean exceeds the typical price;
        may signal upward intraday momentum.
        Uses precise vwap = amount * 10 / vol when available.
        """
        return np.power(self.high * self.low, 0.5) - self.vwap

    def alpha042(self) -> pd.DataFrame:
        """
        Alpha#42: (rank((vwap - close)) / rank((vwap + close)))

        Ratio of the cross-sectional rank of (vwap - close) to the rank of (vwap + close).
        Positive when close < vwap (stock slid below the day's average price); acts as a
        delay-0 mean-reversion signal — stocks that underperformed their vwap are favoured.
        """
        return self._rank(self.vwap - self.close) / self._rank(self.vwap + self.close)

    def alpha054(self) -> pd.DataFrame:
        """
        Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

        ^ denotes signedpower: sign(x)*|x|^e. For positive prices this reduces to
        the ordinary power. Denominator (low - high) <= 0; division by zero yields
        NaN naturally and is left for downstream cleaning.
        """
        num = -1 * (self.low - self.close) * self._signed_power(self.open, 5)
        den = (self.low - self.high) * self._signed_power(self.close, 5)
        return num / den

    def alpha072(self) -> pd.DataFrame:
        """
        Alpha#72: (rank(decay_linear(correlation((high+low)/2, adv40, 8.93345), 10.1519)) /
                   rank(decay_linear(correlation(Ts_Rank(vwap,3.72469),
                                                 Ts_Rank(volume,18.5188), 6.86671), 2.95011)))

        Non-integer window parameters are floored per the paper convention:
          correlation window 8.93345 → 8, decay_linear 10.1519 → 10
          Ts_Rank vwap 3.72469 → 3, Ts_Rank vol 18.5188 → 18
          correlation window 6.86671 → 6, decay_linear 2.95011 → 2
        adv40 approximated as 40-day rolling mean of share volume.
        """
        adv40 = self.vol.rolling(40).mean()
        mid = (self.high + self.low) / 2
        num = self._rank(self._decay_linear(self._corr(mid, adv40, 8), 10))
        ts_rank_vwap = self._ts_rank(self.vwap, 3)
        ts_rank_vol = self._ts_rank(self.vol, 18)
        den = self._rank(self._decay_linear(self._corr(ts_rank_vwap, ts_rank_vol, 6), 2))
        return num / den

    def alpha088(self) -> pd.DataFrame:
        """
        Alpha#88: min(rank(decay_linear(((rank(open)+rank(low))-(rank(high)+rank(close))), 8.06882)),
                      Ts_Rank(decay_linear(correlation(Ts_Rank(close,8.44728),
                                                       Ts_Rank(adv60,20.6966),8.01266),6.65053),2.61957))

        Element-wise minimum of two signals:
          (1) rank of decay-linear-smoothed difference between open/low ranks and high/close ranks
          (2) ts_rank of decay-linear-smoothed rolling correlation between time-series ranks of
              close and adv60
        Non-integer window parameters are floored per paper convention:
          decay_linear 8.06882→8, Ts_Rank close 8.44728→8, Ts_Rank adv60 20.6966→20,
          corr 8.01266→8, decay_linear 6.65053→6, outer Ts_Rank 2.61957→2
        adv60 approximated as 60-day rolling mean of share volume.
        """
        adv60 = self.vol.rolling(60).mean()
        signal1 = self._rank(
            self._decay_linear(
                (self._rank(self.open) + self._rank(self.low))
                - (self._rank(self.high) + self._rank(self.close)),
                8,
            )
        )
        signal2 = self._ts_rank(
            self._decay_linear(
                self._corr(self._ts_rank(self.close, 8), self._ts_rank(adv60, 20), 8),
                6,
            ),
            2,
        )
        return np.minimum(signal1, signal2)

    def alpha094(self) -> pd.DataFrame:
        """
        Alpha#94: ((rank(vwap - ts_min(vwap, 11.5783)) ^
                    Ts_Rank(correlation(Ts_Rank(vwap,19.6462),
                                        Ts_Rank(adv60,4.02992), 18.0926), 2.70756)) * -1)

        ^ denotes signedpower(base, exp).
        Non-integer window parameters are floored per the paper convention:
          ts_min 11.5783 → 11, Ts_Rank vwap 19.6462 → 19
          Ts_Rank adv60 4.02992 → 4, correlation 18.0926 → 18, outer Ts_Rank 2.70756 → 2
        adv60 approximated as 60-day rolling mean of share volume.
        """
        adv60 = self.vol.rolling(60).mean()
        ts_rank_vwap = self._ts_rank(self.vwap, 19)
        ts_rank_adv60 = self._ts_rank(adv60, 4)
        base = self._rank(self.vwap - self._ts_min(self.vwap, 11))
        exp = self._ts_rank(self._corr(ts_rank_vwap, ts_rank_adv60, 18), 2)
        return self._signed_power(base, exp) * -1

    def alpha098(self) -> pd.DataFrame:
        """
        Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5,26.4719), 4.58418), 7.18088)) -
                   rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open),
                                                                    rank(adv15),20.8187),8.62571),
                                             6.95668), 8.07206)))

        Difference of two ranked decay-linear signals:
          (1) rank of decay-smooth correlation between vwap and 26-day sum of adv5
          (2) rank of decay-smooth ts_rank of ts_argmin of correlation between
              open-rank and adv15-rank
        Non-integer window parameters are floored per paper convention:
          sum adv5 26.4719→26, corr 4.58418→4, decay_linear 7.18088→7,
          corr(open,adv15) 20.8187→20, Ts_ArgMin 8.62571→8, Ts_Rank 6.95668→6,
          decay_linear 8.07206→8
        adv5/adv15 approximated as 5/15-day rolling mean of share volume.
        """
        adv5  = self.vol.rolling(5).mean()
        adv15 = self.vol.rolling(15).mean()
        signal1 = self._rank(
            self._decay_linear(
                self._corr(self.vwap, self._sum(adv5, 26), 4),
                7,
            )
        )
        signal2 = self._rank(
            self._decay_linear(
                self._ts_rank(
                    self._ts_argmin(
                        self._corr(self._rank(self.open), self._rank(adv15), 20),
                        8,
                    ),
                    6,
                ),
                8,
            )
        )
        return signal1 - signal2

    def alpha101(self) -> pd.DataFrame:
        """
        Alpha#101: ((close - open) / ((high - low) + .001))

        Intraday momentum: how much the stock closed relative to where it opened,
        normalized by the day's price range. The +0.001 avoids division by zero
        on days with no price movement.
        """
        return (self.close - self.open) / (self.high - self.low + 0.001)

    def alpha_5_day_reversal(self) -> pd.DataFrame:
        """
        Alpha_5_day_reversal: (close - delay(close,5)) / delay(close,5)

        Reversal of the last 5 days' closing prices.
        """
        return (self.close - self._delay(self.close, 5)) / self._delay(self.close, 5)   

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def get_all_alphas(self) -> pd.DataFrame:
        """
        Compute every alpha* method and combine into a single long-form DataFrame.

        Returns:
            pd.DataFrame:
                Index   : MultiIndex (date, code)
                Columns : one per alpha (e.g. 'alpha006', 'alpha012', ...)
                Values  : raw factor values (NaN and inf preserved)
        """
        series: dict = {}
        for name in sorted(dir(self)):
            if not name.startswith("alpha"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            wide = method()                      # DataFrame: dates × codes
            stacked = wide.stack(dropna=False)   # Series: MultiIndex (date, code)
            stacked.name = name
            series[name] = stacked

        if not series:
            return pd.DataFrame()

        result = pd.concat(series, axis=1)
        result.index.names = ["date", "code"]
        return result
