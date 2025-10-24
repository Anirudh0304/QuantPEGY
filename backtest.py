"""Portfolio backtesting utilities.

Unlike more sophisticated libraries such as ``vectorbt`` or
``backtrader`` this module implements a lightweight vectorised
backtester.  It assumes that portfolio weights are determined at
discrete rebalancing dates and remain constant in between.

The primary function, ``backtest_portfolio``, takes a schedule of
weights and historical price data and returns a DataFrame with the
equity curve and several summary statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Tuple


def _compute_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute performance statistics for a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the portfolio.
    risk_free_rate : float, optional
        Daily risk free rate as a fraction (annual rate / 252).  Assumed
        zero by default.

    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics including annualised return,
        annualised volatility, Sharpe ratio and maximum drawdown.
    """
    if returns.empty:
        return {k: np.nan for k in ["cagr", "volatility", "sharpe", "max_drawdown"]}
    # Cumulative return
    cumulative = (1 + returns).cumprod()
    # CAGR
    n_periods = len(returns)
    years = n_periods / 252.0
    cagr = cumulative.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    # Volatility (annualised)
    vol = returns.std(ddof=0) * np.sqrt(252)
    # Sharpe ratio
    excess = returns - risk_free_rate
    sharpe = (excess.mean() / returns.std(ddof=0)) * np.sqrt(252) if returns.std(ddof=0) != 0 else np.nan
    # Max drawdown
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    max_dd = drawdowns.min()
    return {
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def backtest_portfolio(
    price_df: pd.DataFrame,
    weight_schedule: pd.DataFrame,
    column: str = "Adj Close",
    risk_free_rate: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Backtest a portfolio given a schedule of weights and price data.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price data with multi-index columns (price type, ticker).
    weight_schedule : pd.DataFrame
        DataFrame indexed by rebalancing dates with tickers as columns
        containing target weights.  Weights for dates outside the price
        index will be ignored.  It's assumed weights sum to one for
        each rebalancing date.
    column : str, optional
        The price column used for return computation (default: 'Adj Close').
    risk_free_rate : float, optional
        Annual risk free rate expressed as a fraction (e.g., 0.02 for 2%).
        Converted to a daily rate internally.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        A tuple containing a DataFrame with the equity curve and daily
        returns under the columns ``['portfolio_value', 'portfolio_returns']``
        and a dictionary of performance metrics.
    """
    # Extract price series
    if column not in price_df.columns.get_level_values(0):
        raise KeyError(f"Price DataFrame is missing column '{column}'.")
    price_wide = price_df[column]
    # Align weight schedule to price index
    weights = weight_schedule.reindex(price_wide.index).ffill().fillna(0.0)
    # Ensure only tickers present in price data are used
    common = price_wide.columns.intersection(weights.columns)
    price_wide = price_wide[common]
    weights = weights[common]
    # Normalise weights to sum to one (handle any numeric drift)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    # Compute daily returns of each asset
    asset_returns = price_wide.pct_change(fill_method=None).fillna(0.0)
    # Compute portfolio returns as dot product of weights and returns.  Forward fill
    # weights to apply the most recent rebalance across each period.
    port_rets = (weights.shift().ffill() * asset_returns).sum(axis=1)
    # Starting value 1
    equity_curve = (1 + port_rets).cumprod()
    df = pd.DataFrame({
        "portfolio_value": equity_curve,
        "portfolio_returns": port_rets,
    }, index=price_wide.index)
    daily_rf = risk_free_rate / 252.0
    metrics = _compute_performance_metrics(port_rets, risk_free_rate=daily_rf)
    return df, metrics