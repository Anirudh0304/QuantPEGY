"""Portfolio optimisation routines.

This module wraps the popular PyPortfolioOpt library to compute
efficient allocations.  If the library is unavailable, a simple
equal‑weight fallback is used.  Users can choose between several
optimisation targets such as maximum Sharpe ratio or minimum
volatility.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
except ImportError:
    EfficientFrontier = None  # type: ignore
    risk_models = None  # type: ignore
    expected_returns = None  # type: ignore

logger = logging.getLogger(__name__)


def equal_weight(tickers: Iterable[str]) -> Dict[str, float]:
    """Return an equal weight allocation for the given tickers."""
    tickers_list = list(tickers)
    n = len(tickers_list)
    if n == 0:
        return {}
    w = 1.0 / n
    return {t: w for t in tickers_list}


def optimise_weights(
    price_df: pd.DataFrame,
    tickers: Iterable[str],
    method: str = "max_sharpe",
    returns_frequency: str = "daily",
) -> Dict[str, float]:
    """Compute optimal weights for a set of tickers based on historical data.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price data in wide format (e.g., price_df['Adj Close']).  Must
        include the columns for the specified tickers.
    tickers : Iterable[str]
        The universe of assets to optimise.
    method : str, optional
        Optimisation objective.  Supported values include 'max_sharpe',
        'min_volatility' and 'equal'.  Defaults to 'max_sharpe'.
    returns_frequency : str, optional
        Determines the frequency used when annualising returns (default: daily).

    Returns
    -------
    Dict[str, float]
        Mapping of ticker to weight.  If optimisation fails, a naive
        equal weight allocation is returned.
    """
    tickers_list = list(tickers)
    # Sanity check
    if not tickers_list:
        return {}
    # If user requests equal weighting or we lack PyPortfolioOpt, fallback
    if method == "equal" or EfficientFrontier is None:
        return equal_weight(tickers_list)
    # Prepare returns data
    # Only use available tickers in price_df
    available = [t for t in tickers_list if t in price_df.columns]
    if not available:
        return equal_weight(tickers_list)
    # Compute expected returns and covariance.  Convert prices to numeric and
    # forward fill to avoid NaNs.  Drop rows with any remaining NaNs to
    # ensure PyPortfolioOpt receives clean numeric arrays.
    try:
        clean_prices = price_df[available].apply(pd.to_numeric, errors="coerce").ffill().dropna(how="any")
        if clean_prices.empty:
            return equal_weight(tickers_list)
        mu = expected_returns.mean_historical_return(clean_prices, frequency=252)
        Sigma = risk_models.sample_cov(clean_prices, frequency=252)
        ef = EfficientFrontier(mu, Sigma)
        if method == "max_sharpe":
            weights = ef.max_sharpe()
        elif method == "min_volatility":
            weights = ef.min_volatility()
        else:
            logger.warning("Unknown optimisation method '%s'; using equal weights.", method)
            return equal_weight(tickers_list)
        cleaned = ef.clean_weights()
        # Fill missing tickers with zero weight and normalise remaining weights
        total = sum(cleaned.get(t, 0.0) for t in tickers_list)
        if total == 0:
            return equal_weight(tickers_list)
        return {t: cleaned.get(t, 0.0)/total for t in tickers_list}
    except Exception as e:
        logger.warning("Portfolio optimisation failed: %s", e)
        return equal_weight(tickers_list)