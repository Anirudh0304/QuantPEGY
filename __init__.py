"""Top level package for the QuantPEGY project.

This package exposes a collection of modules that together build a
multi‑asset quantitative investment engine.  The primary goal of
QuantPEGY is to compute a hybrid value–growth factor (PEGY),
combine it with momentum and volatility features, classify the
macroeconomic backdrop, backtest candidate portfolios and
optimize final allocations.

Users of the library should typically start with the high level
``quantpegy.run`` module which orchestrates the full pipeline or
``quantpegy.streamlit_app`` for the interactive dashboard.
"""

__all__ = [
    "data",
    "factors",
    "macro",
    "backtest",
    "optimizer",
    "alpha_boost",
    "run",
    "streamlit_app",
]

from . import data, factors, macro, backtest, optimizer, alpha_boost, run