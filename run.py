"""Entry points for running the QuantPEGY pipeline.

This module provides high‑level functions to orchestrate the end‑to‑end
workflow: selecting a universe, downloading data, computing factors,
optionally adjusting factor weights via macro conditions or IC
analysis, ranking and selecting securities, optimising weights and
performing a backtest.  A command line interface is also provided
for convenience.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import get_universe, get_price_data, get_fundamental_metrics
from .factors import compute_factor_scores
from .macro import get_macro_data, classify_regime
from .optimizer import optimise_weights, equal_weight
from .backtest import backtest_portfolio
from .alpha_boost import compute_ic_weights

logger = logging.getLogger(__name__)


def _determine_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Determine rebalancing dates based on a pandas frequency string.

    Given a datetime index, this helper returns the last date of each
    period specified by ``freq`` (e.g., 'M' for month end).  It drops
    the first date to ensure there is a lookback period available for
    momentum and volatility calculations.
    """
    # Resample the index and take the last observation of each period
    # Support new pandas frequency naming (e.g., ME for month-end)
    if freq == "M":
        freq = "ME"
    rebals = index.to_series().resample(freq).last().dropna()
    # Remove the very first rebalancing date to allow for factor lookbacks
    if len(rebals) > 1:
        rebals = rebals.iloc[1:]
    return pd.DatetimeIndex(rebals)


def run_pipeline(
    region: str = "US",
    start: str = "2018-01-01",
    end: Optional[str] = None,
    top_n: int = 10,
    factor_weights: Optional[Dict[str, float]] = None,
    rebal_freq: str = "M",
    optimisation: str = "equal",
    use_macro: bool = False,
    use_ic_weighting: bool = False,
    alpha_lookforward: int = 21,
    risk_free_rate: float = 0.02,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Execute the QuantPEGY strategy and return portfolio equity and metrics.

    Parameters
    ----------
    region : str, optional
        Geographic region identifier (e.g., 'US', 'IN').
    start : str, optional
        Start date for data retrieval.
    end : str, optional
        End date for data retrieval.  Defaults to today.
    top_n : int, optional
        Number of stocks to include in the portfolio at each rebalance.
    factor_weights : dict, optional
        Weights for PEGY, momentum and volatility factors.  If
        ``use_ic_weighting`` is True this is used as an initial guess
        prior to IC adjustment.
    rebal_freq : str, optional
        Pandas offset alias specifying rebalancing frequency (e.g., 'M' for
        monthly, 'Q' for quarterly).
    optimisation : str, optional
        Optimisation method: 'equal', 'max_sharpe' or 'min_volatility'.
    use_macro : bool, optional
        Whether to fetch macro data and print the current regime.  The
        macro regime is not currently used to modify weights but is
        provided for informational purposes and to support future
        enhancements.
    use_ic_weighting : bool, optional
        Whether to adjust factor weights based on information
        coefficient analysis.
    alpha_lookforward : int, optional
        Number of days ahead used for IC calculation when
        ``use_ic_weighting`` is True.
    risk_free_rate : float, optional
        Annualised risk free rate used for Sharpe ratio calculation.
    verbose : bool, optional
        If True, prints progress and summary information.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        The equity curve DataFrame and performance metrics.
    """
    # Retrieve universe and data
    tickers = get_universe(region)
    if verbose:
        logger.info("Selected %d tickers for region %s.", len(tickers), region)
    prices = get_price_data(tickers, start=start, end=end, interval="1d")
    if prices.empty:
        raise RuntimeError("Failed to download price data. Ensure network connectivity and yfinance availability.")
    # Unstack to wide format (price_type, ticker)
    price_wide = prices.unstack(level="ticker")
        # Clean up duplicate dates/columns and normalize tickers
    price_wide = price_wide.loc[~price_wide.index.duplicated(keep='first')]
    price_wide = price_wide.loc[:, ~price_wide.columns.duplicated()]

    price_wide.columns = pd.MultiIndex.from_arrays([
        price_wide.columns.get_level_values(0),
        price_wide.columns.get_level_values(1).str.upper()
    ])
    # Fetch fundamental metrics
    fundamentals = get_fundamental_metrics(tickers)
    # Align fundamental tickers with price tickers
    # Align fundamental tickers with those available in the price data.  Use
    # uppercase tickers for consistency as yfinance may return tickers
    # with mixed case (e.g., 'Aapl').
    fundamental_index = fundamentals.index.str.upper()
    price_tickers = price_wide.columns.get_level_values(1).str.upper()
    fundamentals.index = fundamental_index
    price_wide.columns = pd.MultiIndex.from_arrays([
        price_wide.columns.get_level_values(0),
        price_tickers
    ])
    fundamentals = fundamentals.loc[fundamentals.index.intersection(price_wide.columns.get_level_values(1))]
    if fundamentals.empty:
        raise RuntimeError("No fundamental data available. Ensure yfinance is installed and network access is enabled.")
    # Compute factor scores
    factor_df = compute_factor_scores(fundamentals, price_wide, factor_weights=factor_weights)
    # Optionally adjust factor weights based on IC
    if use_ic_weighting:
        ic_weights = compute_ic_weights(factor_df, price_wide, lookforward=alpha_lookforward)
        # Merge user provided weights with IC weights (multiplicative blending)
        # Start with equal weighting if no initial weights provided
        base_weights = factor_weights or {"pegy": 1/3, "momentum": 1/3, "volatility": 1/3}
        # Normalise base
        total = sum(abs(v) for v in base_weights.values())
        if total == 0:
            base_norm = {k: 1/3 for k in base_weights}
        else:
            base_norm = {k: v/total for k, v in base_weights.items()}
        # Multiply and renormalise
        combined = {k: base_norm.get(k, 0) * ic_weights.get(k, 0) for k in base_norm}
        total_c = sum(combined.values())
        if total_c == 0:
            combined = {k: 1/3 for k in combined}
        else:
            combined = {k: v/total_c for k, v in combined.items()}
        # Recompute factor scores with new weights
        factor_df = compute_factor_scores(fundamentals, price_wide, factor_weights=combined)
    # Determine rebalancing dates
    rebalance_dates = _determine_rebalance_dates(price_wide.index, rebal_freq)
    # Build weight schedule
    weight_rows = []
    # Determine which price column to use ('Adj Close' preferred) once up front
    if 'Adj Close' in price_wide.columns.get_level_values(0):
        price_column_data_global = price_wide['Adj Close']
        price_column_name = 'Adj Close'
    elif 'Close' in price_wide.columns.get_level_values(0):
        price_column_data_global = price_wide['Close']
        price_column_name = 'Close'
    else:
        raise KeyError("Price data does not contain 'Adj Close' or 'Close' columns")

    for date in rebalance_dates:
        # On each rebalance date, rank by composite score
        ranked = factor_df.sort_values("composite_score", ascending=False)
        selected = list(ranked.head(top_n).index)
        # Compute weights via optimisation for the selected tickers using the chosen price column
        w = optimise_weights(price_column_data_global, selected, method=optimisation)
        # Build row with weights for all tickers (ensuring uppercase keys)
        row = {t: w.get(t, 0.0) for t in tickers}
        weight_rows.append((date, row))
    # Construct weight schedule DataFrame
    weight_schedule = pd.DataFrame({t: [row[t] for _, row in weight_rows] for t in tickers}, index=[d for d, _ in weight_rows])
    # Backtest using the same price column used for optimisation
    equity, metrics = backtest_portfolio(price_wide, weight_schedule, column=price_column_name, risk_free_rate=risk_free_rate)
    # Optionally fetch and print macro regime
    if use_macro:
        macro = get_macro_data(start=start, end=end)
        if not macro.empty:
            regime = classify_regime(macro)
            if verbose:
                logger.info("Current macro regime: %s", regime)
    # Print summary
    if verbose:
        logger.info("Backtest completed. CAGR: %.2f%%, Sharpe: %.2f, Max DD: %.2f%%", metrics.get("cagr", np.nan)*100, metrics.get("sharpe", np.nan), metrics.get("max_drawdown", np.nan)*100)
    return equity, metrics


def _parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the QuantPEGY strategy.")
    parser.add_argument("--region", type=str, default="US", help="Region key (US or IN).")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of stocks to hold.")
    parser.add_argument("--rebalance", type=str, default="M", help="Rebalance frequency (e.g., M, Q).")
    parser.add_argument("--optimisation", type=str, default="equal", choices=["equal", "max_sharpe", "min_volatility"], help="Optimisation method.")
    parser.add_argument("--use_macro", action="store_true", help="Fetch macro data and display regime.")
    parser.add_argument("--use_ic", action="store_true", help="Enable information coefficient based weighting.")
    parser.add_argument("--risk_free_rate", type=float, default=0.02, help="Annual risk free rate used for Sharpe ratio.")
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> None:
    """Entry point for command line execution."""
    ns = _parse_args(args)
    run_pipeline(
        region=ns.region,
        start=ns.start,
        end=ns.end,
        top_n=ns.top_n,
        rebal_freq=ns.rebalance,
        optimisation=ns.optimisation,
        use_macro=ns.use_macro,
        use_ic_weighting=ns.use_ic,
        risk_free_rate=ns.risk_free_rate,
    )

if __name__ == "__main__":
    main()