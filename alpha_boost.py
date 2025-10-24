"""Stable IC weighting for QuantPEGY — MultiIndex‑safe and robust."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def compute_ic_weights(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    lookforward: int = 21,
    column: str = "Adj Close",
) -> Dict[str, float]:
    """
    Compute Information Coefficient (IC) based factor weights.
    Works even when price_df has MultiIndex columns.
    """
    factor_cols = [
        c for c in factor_df.columns
        if not c.startswith("z_") and c not in ["composite_score"]
    ]
    if not factor_cols:
        return {"pegy": 1/3, "momentum": 1/3, "volatility": 1/3}

    # make a local copy to avoid mutating caller's DataFrame
    factor_df = factor_df.copy()

    # --- Handle MultiIndex columns safely for price_df ---
    if isinstance(price_df.columns, pd.MultiIndex):
        if column not in price_df.columns.get_level_values(0):
            raise KeyError(f"Price DataFrame missing level '{column}'")
        # select the requested top-level column, drop that level
        price_data = price_df.xs(column, axis=1, level=0, drop_level=True).copy()
    else:
        price_data = price_df.copy()

    # flatten any remaining MultiIndex columns and uppercase
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = ["_".join(map(str, c)).upper() for c in price_data.columns]
    else:
        price_data.columns = [str(c).upper() for c in price_data.columns]

    # --- Normalize and align tickers for factor_df.index ---
    if isinstance(factor_df.index, pd.MultiIndex):
        # collapse multiindex tuples into single ticker strings
        factor_df.index = factor_df.index.map(lambda tpl: "_".join(map(str, tpl)).upper())
    else:
        factor_df.index = factor_df.index.astype(str).str.upper()

    common_tickers = factor_df.index.intersection(price_data.columns)
    if common_tickers.empty:
        return {f: 1/len(factor_cols) for f in factor_cols}

    price_data = price_data[common_tickers]
    factor_df = factor_df.loc[common_tickers]

    # --- Compute forward returns ---
    fwd_returns = (
        price_data.pct_change(periods=lookforward, fill_method=None)
        .shift(-lookforward)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
    )

    ic_scores = {}
    for f in factor_cols:
        try:
            # Broadcast factor values across time
            f_matrix = pd.DataFrame(
                np.tile(factor_df[f].values, (len(fwd_returns), 1)),
                index=fwd_returns.index,
                columns=common_tickers,
            )
            # Flatten for correlation
            stacked = pd.DataFrame({
                "factor": f_matrix.values.ravel(),
                "returns": fwd_returns.values.ravel(),
            }).dropna()
            ic = stacked["factor"].corr(stacked["returns"])
            ic_scores[f] = float(ic) if not np.isnan(ic) else 0.0
        except Exception:
            ic_scores[f] = 0.0

    # --- Normalize absolute ICs to weights ---
    abs_ic = {k: abs(v) for k, v in ic_scores.items()}
    total = sum(abs_ic.values())
    if total == 0:
        return {k: 1.0 / len(abs_ic) for k in abs_ic}
    return {k: v / total for k, v in abs_ic.items()}
