"""Factor calculation routines for QuantPEGY.

This module implements the core investment factors that power the
QuantPEGY ranking model.  Specifically, it provides functions to
compute the PEGY ratio, momentum and volatility measures, z‑score
normalisation and the final composite factor score.  These
computations operate on pandas DataFrames and Series and are
designed to be robust in the face of missing or irregular data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Iterable, Mapping, Optional, Tuple

def compute_pegy_ratio(
    pe_ratio: pd.Series,
    earnings_growth: pd.Series,
    dividend_yield: pd.Series,
) -> pd.Series:
    """Compute the PEGY ratio for each stock.

    The PEGY ratio extends the classic PEG ratio by including the
    dividend yield in the denominator.  Mathematically it is defined
    as::

        PEGY = PE_ratio / (earnings_growth + dividend_yield)

    A lower PEGY ratio indicates a more attractive valuation when
    growth and yield are taken into account.  Missing or zero
    denominators yield NaN values.

    Parameters
    ----------
    pe_ratio : pd.Series
        Series indexed by ticker containing P/E ratios.
    earnings_growth : pd.Series
        Series indexed by ticker containing earnings growth rates (e.g., 0.1
        for 10% growth).  Should be expressed in fractional form.
    dividend_yield : pd.Series
        Series indexed by ticker containing dividend yields expressed
        as fractions (e.g., 0.02 for 2% yield).

    Returns
    -------
    pd.Series
        PEGY ratios indexed by ticker.
    """
    denom = earnings_growth + dividend_yield
    # Avoid division by zero
    denom = denom.replace(0, np.nan)
    ratio = pe_ratio / denom
    return ratio


def compute_momentum(
    price_df: pd.DataFrame,
    lookback: int = 252,
    column: str = "Adj Close",
) -> pd.Series:
    """Compute simple price momentum over a lookback window.

    Momentum is defined as the percentage change over the specified
    lookback period.  By default, a one year (252 trading days)
    lookback is used.  If a ticker has insufficient history, the
    momentum value will be NaN.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price data in wide format with dates on the index and tickers on
        the columns.  The DataFrame should contain at least the column
        specified by ``column`` (e.g., 'Adj Close' or 'Close').
    lookback : int, optional
        Number of observations to look back for computing momentum.
    column : str, optional
        Name of the price column to use for calculation.

    Returns
    -------
    pd.Series
        A Series indexed by ticker with momentum values expressed as
        fractional returns.
    """
    if column not in price_df.columns.get_level_values(0):
        raise KeyError(f"Price DataFrame is missing column '{column}'.")
    # Extract the price subtable for the specified column
    sub = price_df[column]
    # Compute percent change from lookback days ago to the most recent date
    # Compute momentum as return over the lookback window.  For each
    # ticker we compare the most recent price to the price lookback
    # periods ago.  Drop any tickers with insufficient history.  Use
    # .shift() on the full DataFrame rather than indexing individual
    # series to avoid warnings and ensure alignment.
    past = sub.shift(lookback).iloc[-1]
    current = sub.iloc[-1]
    momentum = (current - past) / past
    return momentum


def compute_volatility(
    price_df: pd.DataFrame,
    lookback: int = 63,
    column: str = "Adj Close",
) -> pd.Series:
    """Compute realised volatility over a lookback window.

    Volatility is computed as the annualised standard deviation of
    daily log returns over the specified period.  A lower volatility
    is considered more desirable for ranking purposes.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price data in wide format as returned by
        ``quantpegy.data.get_price_data().unstack()``.  Should contain
        at least the column specified by ``column``.
    lookback : int, optional
        Number of observations to include in the volatility window.
    column : str, optional
        Name of the price column to use for calculation.

    Returns
    -------
    pd.Series
        A Series indexed by ticker with annualised volatility values.
    """
    if column not in price_df.columns.get_level_values(0):
        raise KeyError(f"Price DataFrame is missing column '{column}'.")
    sub = price_df[column]
    # Take the last lookback observations
    # Calculate percentage change without forward filling.  The default
    # behaviour of DataFrame.pct_change is to forward fill missing
    # values.  In pandas >=2.0 this default will change, so we
    # explicitly specify ``fill_method=None`` to avoid deprecation
    # warnings and unintended behaviour.
    window = sub.tail(lookback + 1).pct_change(fill_method=None).dropna()
    # Compute daily standard deviation and annualise (sqrt of 252 trading days)
    vol = window.std() * np.sqrt(252)
    return vol


def _zscore(series: pd.Series) -> pd.Series:
    """Compute the z-score of a series.

    The z-score is (value - mean) / std deviation.  If the standard
    deviation is zero or NaN, the result will be NaN for all entries.
    """
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(index=series.index, data=np.nan)
    return (series - mean) / std
# Ensure unique ticker index
def compute_factor_scores(
    fundamental_df: pd.DataFrame,
    price_df: pd.DataFrame,
    factor_weights: Optional[Mapping[str, float]] = None,
    momentum_lookback: int = 252,
    vol_lookback: int = 63,
) -> pd.DataFrame:
    """Compute individual factor values and composite scores."""

    # 1️⃣ Remove any duplicate tickers from fundamentals
    fundamental_df = fundamental_df[~fundamental_df.index.duplicated(keep="first")]

    required_cols = {"pe_ratio", "earnings_growth", "dividend_yield"}
    if not required_cols.issubset(fundamental_df.columns):
        missing = required_cols - set(fundamental_df.columns)
        raise KeyError(f"Fundamental DataFrame is missing columns: {missing}")

    # 2️⃣ Compute PEGY ratio
    pegy = compute_pegy_ratio(
        fundamental_df["pe_ratio"],
        fundamental_df["earnings_growth"],
        fundamental_df["dividend_yield"],
    )

    # 3️⃣ Choose price column
    if "Adj Close" in price_df.columns.get_level_values(0):
        col = "Adj Close"
    elif "Close" in price_df.columns.get_level_values(0):
        col = "Close"
    else:
        raise KeyError("Price DataFrame must contain 'Adj Close' or 'Close' column.")

    # 4️⃣ Compute momentum & volatility
    momentum = compute_momentum(price_df, lookback=momentum_lookback, column=col)
    volatility = compute_volatility(price_df, lookback=vol_lookback, column=col)

    # 5️⃣ Drop duplicate tickers in momentum/volatility
    momentum = momentum[~momentum.index.duplicated(keep="first")]
    volatility = volatility[~volatility.index.duplicated(keep="first")]

    # 6️⃣ Align on common unique tickers only
    common = sorted(set(pegy.index) & set(momentum.index) & set(volatility.index))
    pegy, momentum, volatility = pegy.loc[common], momentum.loc[common], volatility.loc[common]

    # 7️⃣ Reset index uniqueness (safety)
    pegy.index = pegy.index.str.upper()
    momentum.index = momentum.index.str.upper()
    volatility.index = volatility.index.str.upper()

    pegy = pegy.groupby(pegy.index).mean()
    momentum = momentum.groupby(momentum.index).mean()
    volatility = volatility.groupby(volatility.index).mean()

    # 8️⃣ Final DataFrame with consistent unique index
    df = pd.DataFrame({
        "pegy": pegy,
        "momentum": momentum,
        "volatility": volatility,
    }).dropna(how="all")

    # 9️⃣ Compute z-scores
    z_pegy = -_zscore(df["pegy"])
    z_mom = _zscore(df["momentum"])
    z_vol = -_zscore(df["volatility"])

    df["z_pegy"] = z_pegy
    df["z_momentum"] = z_mom
    df["z_volatility"] = z_vol

    # 🔟 Weighting logic
    if factor_weights is None:
        weights = {"pegy": 1/3, "momentum": 1/3, "volatility": 1/3}
    else:
        weights = {k.lower(): float(v) for k, v in factor_weights.items()}
        total = sum(abs(v) for v in weights.values())
        weights = {k: v/total for k, v in weights.items()} if total != 0 else {"pegy": 1/3, "momentum": 1/3, "volatility": 1/3}

    df["composite_score"] = (
        weights.get("pegy", 0) * z_pegy +
        weights.get("momentum", 0) * z_mom +
        weights.get("volatility", 0) * z_vol
    )

    return df
