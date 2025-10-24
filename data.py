"""Data acquisition utilities for QuantPEGY.

This module handles selection of the stock universe, retrieval of
historical price data and scraping of fundamental metrics required
for computing the PEGY factor.  All external dependencies are
contained within this file so callers can remain agnostic to
whether ``yfinance`` or another backend is being used.

The functions defined here are intentionally robust – missing data
is gracefully handled and optional caching is provided.  The
universe definitions can be easily extended by editing the
``UNIVERSES`` constant.
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import logging
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    # ``yfinance`` is not installed in the sandbox environment.  The
    # presence of this variable allows clients to detect and handle the
    # missing dependency at runtime.
    yf = None  # type: ignore

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Universe definitions
#
# These lists contain representative tickers from the US (S&P 500 heavyweights)
# and India (NIFTY 50 constituents).  They are deliberately conservative to
# ensure that the code runs in constrained environments; feel free to expand
# them as needed.  Tickers ending with ``.NS`` refer to the National Stock
# Exchange of India.

UNIVERSES: Dict[str, List[str]] = {
    "US": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ",
        "V", "UNH", "PG", "MA", "TSLA", "HD", "BAC", "KO", "XOM", "PFE",
        "CVX", "T", "CSCO", "CMCSA", "ADBE", "ABT", "PEP", "COST", "NFLX",
        "AMD", "INTC", "CRM", "TMO", "ORCL", "ACN", "AVGO", "MCD", "NKE"
    ],
    "IN": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
        "KOTAKBANK.NS", "ITC.NS", "HCLTECH.NS", "ASIANPAINT.NS", "LT.NS",
        "MARUTI.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "AXISBANK.NS",
        "ONGC.NS", "HINDALCO.NS", "BAJAJFINSV.NS", "WIPRO.NS",
        "POWERGRID.NS", "TITAN.NS", "NTPC.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
        "COALINDIA.NS", "NESTLEIND.NS", "SBILIFE.NS"
    ],
}

# Aliases allow users to specify regions more flexibly.  For example
# ``IN`` and ``INDIA`` map to the same universe.
UNIVERSE_ALIASES: Dict[str, str] = {
    "US": "US",
    "USA": "US",
    "INDIA": "IN",
    "IN": "IN",
}


def get_universe(region: str) -> List[str]:
    """Return the list of tickers corresponding to a geographic region.

    Parameters
    ----------
    region : str
        A region key such as ``'US'`` or ``'IN'``.  Case insensitive.

    Returns
    -------
    List[str]
        A list of ticker symbols.

    Raises
    ------
    KeyError
        If the region is unknown.
    """
    key = region.strip().upper()
    if key not in UNIVERSE_ALIASES:
        raise KeyError(f"Unknown region '{region}'. Valid keys are {list(UNIVERSE_ALIASES.keys())}.")
    return UNIVERSES[UNIVERSE_ALIASES[key]].copy()


def _fetch_single_ticker_price(
    ticker: str, start: str, end: str, interval: str
) -> pd.DataFrame:
    """Helper to download price history for a single ticker using yfinance.

    Returns a DataFrame indexed by date with OHLCV columns.  If the
    download fails (for example because ``yfinance`` isn't installed or
    there is no network connectivity), an empty DataFrame is returned.
    """
    if yf is None:
        logger.warning("yfinance not installed; cannot download data for %s", ticker)
        return pd.DataFrame()
    try:
        # Explicitly disable auto-adjust and threading.  Recent yfinance
        # versions change the default of ``auto_adjust`` to True which
        # adjusts prices and drops the ``Adj Close`` field.  We set
        # ``auto_adjust=False`` to preserve the ``Adj Close`` column.  We
        # also disable ``threads`` to avoid intermittent "dictionary
        # changed size during iteration" errors during downloads.
        data = yf.download(ticker, start=start, end=end, interval=interval,
                           progress=False, auto_adjust=False, threads=False)
        if not data.empty:
            # Add a column to identify the ticker after we concat
            data["ticker"] = ticker
        return data
    except Exception as e:
        logger.warning("Failed to download price data for %s: %s", ticker, e)
        return pd.DataFrame()


def get_price_data(
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    max_workers: int = 5,
) -> pd.DataFrame:
    """Fetch OHLCV price history for multiple tickers in parallel.

    The returned DataFrame has a DateTimeIndex and a second level index
    indicating the ticker. Use ``df.xs(ticker, level='ticker')`` to access a single series.

    Notes
    -----
    If yfinance cannot be imported or if network access is unavailable,
    this function will quietly return an empty DataFrame.
    """
    if end is None:
        end = _dt.datetime.utcnow().strftime("%Y-%m-%d")

    tickers_list = list(tickers)
    if not tickers_list:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []

    # Fetch in parallel safely
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_fetch_single_ticker_price, t, start, end, interval)
            for t in tickers_list
        ]
        for fut in concurrent.futures.as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)

    # If no successful downloads, return empty DataFrame
    if not frames:
        logger.warning("No price data could be downloaded for any ticker.")
        return pd.DataFrame()

    # Concatenate results
    combined = pd.concat(frames, axis=0)

    # Safety cleanup
    combined = combined.reset_index()
    if "Date" not in combined.columns or "ticker" not in combined.columns:
        logger.error("Unexpected format returned from yfinance, columns=%s", combined.columns)
        return pd.DataFrame()

    # Ensure unique Date–Ticker combinations
    combined = combined.loc[~combined[["Date", "ticker"]].duplicated(keep="first")]

    # Set MultiIndex for compatibility
    combined = combined.set_index(["Date", "ticker"]).sort_index()

    # Drop duplicate columns if yfinance returns repeated OHLCV headers
    combined = combined.loc[:, ~combined.columns.duplicated()]

    return combined



def _fetch_single_ticker_fundamentals(ticker: str) -> Tuple[str, Dict[str, float]]:
    """Retrieve fundamental metrics for a single ticker.

    This function uses the ``yfinance.Ticker.info`` attribute to obtain
    trailing P/E ratio, earnings growth and dividend yield.  If a
    particular metric is unavailable, ``numpy.nan`` is returned for
    that field.
    """
    metrics: Dict[str, float] = {
        "pe_ratio": np.nan,
        "earnings_growth": np.nan,
        "dividend_yield": np.nan,
    }
    if yf is None:
        return ticker, metrics
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return ticker, metrics
    # P/E ratio
    pe = info.get("trailingPE") or info.get("forwardPE")
    metrics["pe_ratio"] = float(pe) if pe not in (None, 0) else np.nan
    # Earnings growth: use trailing annual earnings growth if available
    eg = (
        info.get("earningsQuarterlyGrowth")
        or info.get("earningsGrowth")
        or info.get("revenueGrowth")
    )
    metrics["earnings_growth"] = float(eg) if eg is not None else np.nan
    # Dividend yield: either explicit yield or compute from rate/price
    div_yield = info.get("dividendYield")
    if div_yield is not None:
        metrics["dividend_yield"] = float(div_yield)
    else:
        # fall back to rate / previousClose
        div_rate = info.get("dividendRate")
        price = info.get("previousClose")
        if div_rate and price:
            metrics["dividend_yield"] = float(div_rate) / float(price)
    return ticker, metrics


def get_fundamental_metrics(
    tickers: Iterable[str],
    max_workers: int = 5,
) -> pd.DataFrame:
    """Fetch fundamental metrics (P/E, earnings growth, dividend yield) for tickers.

    The returned DataFrame is indexed by ticker and has columns
    ``['pe_ratio', 'earnings_growth', 'dividend_yield']``.  Rows with
    entirely missing data are dropped.
    """
    tickers_list = list(tickers)
    if not tickers_list:
        return pd.DataFrame(columns=["pe_ratio", "earnings_growth", "dividend_yield"])
    records: List[Tuple[str, Dict[str, float]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_single_ticker_fundamentals, t) for t in tickers_list]
        for fut in concurrent.futures.as_completed(futures):
            records.append(fut.result())
    if not records:
        return pd.DataFrame(columns=["pe_ratio", "earnings_growth", "dividend_yield"])
    df = pd.DataFrame({ticker: metrics for ticker, metrics in records}).T
    df.index.name = "ticker"
    # Drop rows where all metrics are NaN
    df = df.dropna(how="all")
    return df
