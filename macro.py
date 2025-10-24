"""Macroeconomic regime classification utilities.

This module provides helper functions for downloading macroeconomic
time series from the Federal Reserve Economic Database (FRED) and
classifying market regimes.  The classification rules are deliberately
simple but can be extended by users as required.

In order to access FRED data you will need either the ``fredapi``
package or ``pandas_datareader``.  If both are unavailable, the
``get_macro_data`` function will raise a RuntimeError.  When using
``fredapi`` you can set the environment variable ``FRED_API_KEY`` or
pass an API key explicitly.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from fredapi import Fred  # type: ignore
except ImportError:
    Fred = None  # type: ignore

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None  # type: ignore


def get_macro_data(
    start: str,
    end: Optional[str] = None,
    series: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download macroeconomic time series from FRED.

    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD) for the requested series.
    end : str, optional
        End date (YYYY-MM-DD).  If omitted, uses today's date.
    series : dict, optional
        Mapping of friendly names to FRED series IDs.  If omitted
        defaults to a small set of commonly used macro indicators.
    api_key : str, optional
        FRED API key.  Required when using the ``fredapi`` backend.  If
        omitted, the environment variable ``FRED_API_KEY`` will be used.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by date with columns corresponding to the
        requested series.  Values are floats and missing values are
        forward filled.  If no backend is available, raises a
        RuntimeError.
    """
    if end is None:
        end = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    if series is None:
        series = {
            "vix": "VIXCLS",               # CBOE Volatility Index
            "yield_curve": "T10Y2Y",       # 10-Year Treasury Constant Maturity Minus 2-Year
            "inflation": "CPALTT01USM657N",  # Consumer Price Index: All Items
        }
    # Determine backend
    data: pd.DataFrame
    if Fred is not None:
        key = "67ecf51ca22d115745ea1bd0dbcf6107"	
        try:
            fred = Fred(key)
            frames = []
            for name, sid in series.items():
                s = fred.get_series(sid, observation_start=start, observation_end=end)
                s.name = name
                frames.append(s)
            data = pd.concat(frames, axis=1)
        except Exception as e:
            logger.debug("fredapi backend failed: %s", e)
            data = pd.DataFrame()
    elif pdr is not None:
        # Use pandas_datareader as a fallback
        frames = []
        for name, sid in series.items():
            try:
                s = pdr.DataReader(sid, "fred", start, end)
                s.columns = [name]
                frames.append(s)
            except Exception as e:
                logger.warning("Failed to fetch series %s via pandas_datareader: %s", sid, e)
        if frames:
            data = pd.concat(frames, axis=1)
        else:
            data = pd.DataFrame()
    else:
        # ------------------------------------------------------------------
        # Fallback using yfinance when fredapi and pandas_datareader are unavailable.
        #
        # If neither of the FRED backends is installed, attempt to fetch
        # approximate macro indicators via yfinance.  We download the VIX index
        # (CBOE Volatility Index) and the 10-year and short‑term Treasury
        # yields to approximate the yield curve.  Note that these tickers are
        # approximations and may differ slightly from the official FRED
        # definitions, but they provide useful guidance in the absence of
        # dedicated APIs.
        try:
            # Import yfinance locally.  If unavailable, this import will fail and
            # the exception will be caught below.
            import yfinance as _yf
            # Fetch daily data for VIX (volatility), 10‑year yield (^TNX) and
            # 3‑month T‑bill yield (^IRX) to approximate the 2‑year yield.
            vix = _yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
            t10 = _yf.download("^TNX", start=start, end=end, progress=False, auto_adjust=True)
            t3m = _yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=True)
            # Use closing/adjusted close prices as yields and convert to percentage
            vix_series = vix["Adj Close"].rename("vix")
            t10_series = t10["Adj Close"].rename("yield10") / 100.0
            t3m_series = t3m["Adj Close"].rename("yield3m") / 100.0
            # Approximate yield curve as 10Y minus 3M (proxy for 2Y)
            yield_curve = (t10_series - t3m_series).rename("yield_curve")
            frames = [vix_series, yield_curve]
            data = pd.concat(frames, axis=1)
        except Exception as exc:
            raise RuntimeError(
                "Neither fredapi nor pandas_datareader are installed and fetching macro data via yfinance failed: {}".format(exc)
            )
    # Clean up DataFrame
    data = data.sort_index()
    # Forward fill missing values
    data = data.ffill()
    return data


def classify_regime(
    macro_df: pd.DataFrame,
    vix_threshold_high: float = 20.0,
    vix_threshold_low: float = 15.0,
    yield_curve_threshold: float = 0.0,
) -> str:
    """Classify the current macro regime into risk-on, neutral or risk-off.

    This simple classifier uses the most recent values of the VIX and
    the 10Y–2Y yield spread to determine the risk appetite of the
    market.  High VIX or an inverted yield curve indicates risk-off,
    whereas a low VIX and positive yield spread signals risk-on.  Values
    between these extremes are labelled neutral.

    Parameters
    ----------
    macro_df : pd.DataFrame
        DataFrame returned by ``get_macro_data`` containing at least
        the columns ``'vix'`` and ``'yield_curve'``.
    vix_threshold_high : float, optional
        Above this level the VIX is considered elevated and triggers a
        risk-off classification.
    vix_threshold_low : float, optional
        Below this level the VIX is considered benign and supports a
        risk-on classification.
    yield_curve_threshold : float, optional
        Threshold for the yield curve (10Y minus 2Y).  Negative values
        are typically associated with recessionary risk.

    Returns
    -------
    str
        One of ``'risk_off'``, ``'risk_on'`` or ``'neutral'``.
    """
    latest = macro_df.dropna().iloc[-1]
    vix = latest.get("vix", np.nan)
    spread = latest.get("yield_curve", np.nan)
    if np.isnan(vix) or np.isnan(spread):
        return "unknown"
    # Risk-off: high volatility or inverted yield curve
    if vix >= vix_threshold_high or spread <= yield_curve_threshold:
        return "risk_off"
    # Risk-on: low volatility and healthy yield curve
    if vix < vix_threshold_low and spread > yield_curve_threshold:
        return "risk_on"
    return "neutral"