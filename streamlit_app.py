# # """Streamlit dashboard for the QuantPEGY strategy."""




# # @st.cache_data(show_spinner=True)
# # def run_strategy_cached(
# #     region: str,
# #     start: str,
# #     end: str,
# #     top_n: int,
# #     factor_weights: Dict[str, float],
# #     rebal_freq: str,
# #     optimisation: str,
# #     use_macro: bool,
# #     use_ic: bool,
# #     risk_free_rate: float,
# # ):
# #     """Cached wrapper around run_pipeline to avoid re-computation."""
# #     equity, metrics = run_pipeline(
# #         region=region,
# #         start=start,
# #         end=end,
# #         top_n=top_n,
# #         factor_weights=factor_weights,
# #         rebal_freq=rebal_freq,
# #         optimisation=optimisation,
# #         use_macro=use_macro,
# #         use_ic_weighting=use_ic,
# #         risk_free_rate=risk_free_rate,
# #         verbose=False,
# #     )
# #     return equity, metrics


# # def main() -> None:
# #     st.set_page_config(page_title="QuantPEGY Dashboard", layout="wide")

# #     st.title("QuantPEGY: Multi-Region Macro-Aware Factor Engine")
# #     st.markdown(
# #         "Use this dashboard to test PEGY-based portfolios across US and Indian markets. "
# #         "Adjust the parameters, run backtests, and visualize results interactively."
# #     )

# #     # Sidebar Configuration
# #     st.sidebar.header("Configuration")
# #     region = st.sidebar.selectbox("Region", ["US", "IN"])
# #     start_date = st.sidebar.date_input("Start Date", dt.date(2018, 1, 1))
# #     end_date = st.sidebar.date_input("End Date", dt.date.today())
# #     top_n = st.sidebar.slider("Number of Stocks", 5, 30, 10, 1)

# #     st.sidebar.subheader("Factor Weights")
# #     w_pegy = st.sidebar.slider("PEGY", 0.0, 1.0, 0.33, 0.01)
# #     w_mom = st.sidebar.slider("Momentum", 0.0, 1.0, 0.33, 0.01)
# #     w_vol = st.sidebar.slider("Volatility", 0.0, 1.0, 0.34, 0.01)
# #     total = w_pegy + w_mom + w_vol
# #     factor_weights = {
# #         "pegy": w_pegy / total if total > 0 else 1 / 3,
# #         "momentum": w_mom / total if total > 0 else 1 / 3,
# #         "volatility": w_vol / total if total > 0 else 1 / 3,
# #     }

# #     rebal_freq = st.sidebar.selectbox("Rebalance Frequency", ["M", "Q"], index=0)
# #     optimisation = st.sidebar.selectbox(
# #         "Optimisation", ["equal", "max_sharpe", "min_volatility"], index=1
# #     )
# #     use_macro = st.sidebar.checkbox("Show Macro Regime", value=False)
# #     use_ic = st.sidebar.checkbox("IC Weighting", value=False)
# #     risk_free_rate = st.sidebar.number_input(
# #         "Risk Free Rate (annual)", 0.0, 0.1, 0.02, 0.005, format="%.3f"
# #     )

# #     # Run Backtest
# #     if st.sidebar.button("Run Backtest"):
# #         with st.spinner("Running backtest..."):
# #             try:
# #                 equity, metrics = run_strategy_cached(
# #                     region=region,
# #                     start=start_date.strftime("%Y-%m-%d"),
# #                     end=end_date.strftime("%Y-%m-%d"),
# #                     top_n=top_n,
# #                     factor_weights=factor_weights,
# #                     rebal_freq=rebal_freq,
# #                     optimisation=optimisation,
# #                     use_macro=use_macro,
# #                     use_ic=use_ic,
# #                     risk_free_rate=risk_free_rate,
# #                 )

# #                 st.success("✅ Backtest completed successfully!")

# #                 # Equity Curve
# #                 st.subheader("Equity Curve")
# #                 y_col = (
# #                     "portfolio_value"
# #                     if "portfolio_value" in equity.columns
# #                     else "equity"
# #                 )
# #                 st.line_chart(equity[y_col])

# #                 # Summary Statistics
# #                 st.subheader("Summary Statistics")
# #                 st.dataframe(pd.DataFrame(metrics, index=["Metric"]).T)

# #                 # Download CSV
# #                 csv = equity.to_csv().encode("utf-8")
# #                 st.download_button(
# #                     "📊 Download Equity CSV",
# #                     data=csv,
# #                     file_name="quantpegy_equity.csv",
# #                     mime="text/csv",
# #                 )

# #             except Exception as e:
# #                 st.error(f"❌ An error occurred: {e}")
# #     else:
# #         st.info("Configure parameters on the left and click **Run Backtest** to begin.")


# # if __name__ == "__main__":
# #     main()
# """
# QuantPEGY Streamlit Dashboard — Final Polished Version

# Features:
# - PEGY, Momentum, Volatility sliders
# - Configurable region, rebalance, and optimisation
# - Interactive equity curve visualization
# - Auto-save results to /results folder
# - Formatted summary statistics with color indicators
# """

# # from __future__ import annotations
# # import os
# # import datetime as dt
# # import pandas as pd
# # import streamlit as st
# # from typing import Dict
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# # from quantpegy.run import run_pipeline

# # -------------------- CONFIG --------------------
# RESULTS_DIR = "results"
# os.makedirs(RESULTS_DIR, exist_ok=True)

# st.set_page_config(
#     page_title="QuantPEGY Dashboard",
#     layout="wide",
#     page_icon="📈",
# )

# # -------------------- CACHED RUNNER --------------------
# @st.cache_data(show_spinner=True)
# def run_strategy_cached(
#     region: str,
#     start: str,
#     end: str,
#     top_n: int,
#     factor_weights: Dict[str, float],
#     rebal_freq: str,
#     optimisation: str,
#     use_macro: bool,
#     use_ic: bool,
#     risk_free_rate: float,
# ):
#     equity, metrics = run_pipeline(
#         region=region,
#         start=start,
#         end=end,
#         top_n=top_n,
#         factor_weights=factor_weights,
#         rebal_freq=rebal_freq,
#         optimisation=optimisation,
#         use_macro=use_macro,
#         use_ic_weighting=use_ic,
#         risk_free_rate=risk_free_rate,
#         verbose=False,
#     )
#     return equity, metrics

# # -------------------- MAIN APP --------------------
# st.title("QuantPEGY: Multi-Region Macro-Aware Factor Engine")
# st.markdown(
#     "Use this dashboard to test PEGY-based portfolios across **US** and **Indian** markets. "
#     "Adjust the parameters, run backtests, and visualize results interactively."
# )

# # Sidebar Configuration
# st.sidebar.header("Configuration")
# region = st.sidebar.selectbox("Region", options=["US", "IN"], index=0)
# start_date = st.sidebar.date_input("Start Date", dt.date(2020, 1, 1))
# end_date = st.sidebar.date_input("End Date", dt.date.today())
# top_n = st.sidebar.slider("Number of Stocks", 5, 30, 10)

# st.sidebar.subheader("Factor Weights")
# w_pegy = st.sidebar.slider("PEGY", 0.0, 1.0, 0.33, 0.01)
# w_mom = st.sidebar.slider("Momentum", 0.0, 1.0, 0.33, 0.01)
# w_vol = st.sidebar.slider("Volatility", 0.0, 1.0, 0.34, 0.01)

# total_w = w_pegy + w_mom + w_vol
# factor_weights = {
#     "pegy": w_pegy / total_w if total_w else 1/3,
#     "momentum": w_mom / total_w if total_w else 1/3,
#     "volatility": w_vol / total_w if total_w else 1/3,
# }

# rebal_freq = st.sidebar.selectbox("Rebalance Frequency", ["M", "Q"], index=0)
# optimisation = st.sidebar.selectbox("Optimisation", ["equal", "max_sharpe", "min_volatility"], index=1)
# use_macro = st.sidebar.checkbox("Show Macro Regime", value=False)
# use_ic = st.sidebar.checkbox("IC Weighting", value=False)
# risk_free_rate = st.sidebar.number_input(
#     "Risk Free Rate (annual)", min_value=0.0, max_value=0.1, value=0.02, step=0.005, format="%.3f"
# )

# # -------------------- RUN BUTTON --------------------
# if st.sidebar.button("Run Backtest"):
#     with st.spinner("Running backtest... Please wait ⏳"):
#         try:
#             equity, metrics = run_strategy_cached(
#                 region=region,
#                 start=start_date.strftime("%Y-%m-%d"),
#                 end=end_date.strftime("%Y-%m-%d"),
#                 top_n=top_n,
#                 factor_weights=factor_weights,
#                 rebal_freq=rebal_freq,
#                 optimisation=optimisation,
#                 use_macro=use_macro,
#                 use_ic=use_ic,
#                 risk_free_rate=risk_free_rate,
#             )

#             st.success("✅ Backtest completed successfully!")

#             # ---------------- EQUITY CURVE ----------------
#             st.subheader(f"📈 Equity Curve ({region}, {optimisation.upper()})")
#             st.line_chart(equity["portfolio_value"])

#             # ---------------- SUMMARY STATISTICS ----------------
#             st.subheader("📊 Summary Statistics")

#             formatted = {
#                 "CAGR": f"{metrics.get('cagr', 0):.2%}",
#                 "Volatility": f"{metrics.get('volatility', 0):.2%}",
#                 "Sharpe Ratio": f"{metrics.get('sharpe', 0):.3f}",
#                 "Max Drawdown": f"{metrics.get('max_drawdown', 0):.2%}",
#             }

#             color = "green" if metrics.get("sharpe", 0) > 1 else "orange" if metrics.get("sharpe", 0) > 0.5 else "red"
#             st.markdown(f"**Sharpe Ratio:** <span style='color:{color}'>{metrics.get('sharpe', 0):.3f}</span>", unsafe_allow_html=True)

#             df_metrics = pd.DataFrame(list(formatted.items()), columns=["Metric", "Value"])
#             st.dataframe(df_metrics, use_container_width=True)

#             # ---------------- MACRO REGIME SECTION ----------------
#             if use_macro:
#                 st.subheader("🌎 Detected Macro Regime")
#                 from quantpegy.macro import get_macro_data, classify_regime

#                 macro = get_macro_data(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
#                 if not macro.empty:
#                     regime = classify_regime(macro)
#                     regime_colors = {
#                         "Expansion": "🟢",
#                         "Recession": "🔴",
#                         "Stagflation": "🟠",
#                         "Disinflation": "🔵",
#                     }
#                     icon = regime_colors.get(regime, "⚪")
#                     st.markdown(f"**Current Regime:** {icon} {regime}")
#                     st.line_chart(macro)
#                 else:
#                     st.warning("⚠️ Macro data not available for the selected period.")

#             # ---------------- IC WEIGHTING SECTION ----------------
#             if use_ic:
#                 st.subheader("🧠 Information Coefficient (IC) Weighted Factors")
#                 st.markdown(
#                     "IC weighting dynamically adjusts factor weights based on predictive power of each factor "
#                     "(correlation between factor rank and forward returns)."
#                 )

#                 # Re-run factor IC computation manually for display
#                 from quantpegy.alpha_boost import compute_ic_weights
#                 try:
#                     from quantpegy.data import get_universe, get_price_data, get_fundamental_metrics
#                     from quantpegy.factors import compute_factor_scores

#                     tickers = get_universe(region)
#                     prices = get_price_data(tickers, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
#                     fundamentals = get_fundamental_metrics(tickers)
#                     factor_df = compute_factor_scores(fundamentals, prices.unstack(level="ticker"))
#                     ic_weights = compute_ic_weights(factor_df, prices.unstack(level="ticker"))

#                     ic_df = pd.DataFrame(
#                         list(ic_weights.items()), columns=["Factor", "IC Weight"]
#                     ).sort_values("IC Weight", ascending=False)
#                     st.dataframe(ic_df, use_container_width=True)
#                 except Exception as e:
#                     st.warning(f"Could not compute IC weights: {e}")

#             # ---------------- SAVE RESULTS ----------------
#             ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
#             csv_path = os.path.join(RESULTS_DIR, f"equity_{region}_{ts}.csv")
#             json_path = os.path.join(RESULTS_DIR, f"summary_{region}_{ts}.json")

#             equity.to_csv(csv_path, index=True)
#             pd.Series(metrics).to_json(json_path)

#             st.success(f"Results saved to `{RESULTS_DIR}/` ✅")
#             st.download_button(
#                 label="📥 Download Equity CSV",
#                 data=equity.to_csv().encode("utf-8"),
#                 file_name=f"quantpegy_equity_{region}_{ts}.csv",
#                 mime="text/csv",
#             )

#         except Exception as e:
#             st.error(f"❌ An error occurred: {e}")
# else:
#     st.info("Configure parameters on the left and click **Run Backtest** to start.")

"""
QuantPEGY Streamlit Dashboard — Final Full Integration

Features:
- PEGY, Momentum, Volatility sliders
- Macro regime detection (via FRED API)
- IC-weighted factor analysis
- Auto-save results and CSV download
"""

from __future__ import annotations
import datetime as dt
import sys, os
from typing import Dict
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantpegy.run import run_pipeline
from quantpegy.macro import get_macro_data, classify_regime
from quantpegy.alpha_boost import compute_ic_weights
from quantpegy.data import get_universe, get_price_data, get_fundamental_metrics
from quantpegy.factors import compute_factor_scores

# ---------------- CONFIG ----------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

st.set_page_config(
    page_title="QuantPEGY Dashboard",
    layout="wide",
    page_icon="📈",
)

# ---------------- CACHED RUNNER ----------------
@st.cache_data(show_spinner=True)
def run_strategy_cached(region, start, end, top_n, factor_weights, rebal_freq,
                        optimisation, use_macro, use_ic, risk_free_rate):
    equity, metrics = run_pipeline(
        region=region,
        start=start,
        end=end,
        top_n=top_n,
        factor_weights=factor_weights,
        rebal_freq=rebal_freq,
        optimisation=optimisation,
        use_macro=use_macro,
        use_ic_weighting=use_ic,
        risk_free_rate=risk_free_rate,
        verbose=False,
    )
    return equity, metrics

# ---------------- UI ----------------
st.title("QuantPEGY: Multi-Region Macro-Aware Factor Engine")
st.markdown(
    "Use this dashboard to test PEGY-based portfolios across **US** and **Indian** markets. "
    "Adjust parameters, run backtests, and visualize results interactively."
)

st.sidebar.header("Configuration")
region = st.sidebar.selectbox("Region", ["US", "IN"], index=0)
start_date = st.sidebar.date_input("Start Date", dt.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())
top_n = st.sidebar.slider("Number of Stocks", 5, 30, 10)

st.sidebar.subheader("Factor Weights")
w_pegy = st.sidebar.slider("PEGY", 0.0, 1.0, 0.33, 0.01)
w_mom = st.sidebar.slider("Momentum", 0.0, 1.0, 0.33, 0.01)
w_vol = st.sidebar.slider("Volatility", 0.0, 1.0, 0.34, 0.01)
total_w = w_pegy + w_mom + w_vol

factor_weights = {
    "pegy": w_pegy / total_w if total_w else 1/3,
    "momentum": w_mom / total_w if total_w else 1/3,
    "volatility": w_vol / total_w if total_w else 1/3,
}

rebal_freq = st.sidebar.selectbox("Rebalance Frequency", ["M", "Q"], index=0)
optimisation = st.sidebar.selectbox("Optimisation", ["equal", "max_sharpe", "min_volatility"], index=1)
use_macro = st.sidebar.checkbox("Show Macro Regime", value=False)
use_ic = st.sidebar.checkbox("IC Weighting", value=False)
risk_free_rate = st.sidebar.number_input(
    "Risk Free Rate (annual)", min_value=0.0, max_value=0.1, value=0.02, step=0.005, format="%.3f"
)

# ---------------- RUN ----------------
if st.sidebar.button("Run Backtest"):
    with st.spinner("Running backtest... Please wait ⏳"):
        try:
            equity, metrics = run_strategy_cached(
                region=region,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                top_n=top_n,
                factor_weights=factor_weights,
                rebal_freq=rebal_freq,
                optimisation=optimisation,
                use_macro=use_macro,
                use_ic=use_ic,
                risk_free_rate=risk_free_rate,
            )

            st.success("✅ Backtest completed successfully!")
            st.subheader(f"📈 Equity Curve ({region}, {optimisation.upper()})")
            st.line_chart(equity["portfolio_value"])

            # ---- SUMMARY STATS ----
            st.subheader("📊 Summary Statistics")
            formatted = {
                "CAGR": f"{metrics.get('cagr', 0):.2%}",
                "Volatility": f"{metrics.get('volatility', 0):.2%}",
                "Sharpe Ratio": f"{metrics.get('sharpe', 0):.3f}",
                "Max Drawdown": f"{metrics.get('max_drawdown', 0):.2%}",
            }
            color = "green" if metrics.get("sharpe", 0) > 1 else "orange" if metrics.get("sharpe", 0) > 0.5 else "red"
            st.markdown(f"**Sharpe Ratio:** <span style='color:{color}'>{metrics.get('sharpe', 0):.3f}</span>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(list(formatted.items()), columns=["Metric", "Value"]))

            # ---- MACRO REGIME ----
            if use_macro:
                st.subheader("🌎 Detected Macro Regime")
                macro = get_macro_data(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
                if not macro.empty:
                    regime = classify_regime(macro)
                    emoji = {"risk_on": "🟢", "risk_off": "🔴", "neutral": "🟠", "unknown": "⚪"}
                    st.markdown(f"**Current Regime:** {emoji.get(regime, '⚪')} {regime.replace('_', ' ').title()}")
                    st.line_chart(macro)
                else:
                    st.warning("⚠️ Could not fetch macro data (check FRED or API connection).")

            # ---- IC WEIGHTING ----
            if use_ic:
                st.subheader("🧠 Information Coefficient (IC) Weighted Factors")
                st.markdown("IC weighting adjusts factor importance based on correlation with future returns.")
                try:
                    tickers = get_universe(region)
                    prices = get_price_data(tickers, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
                    fundamentals = get_fundamental_metrics(tickers)
                    factor_df = compute_factor_scores(fundamentals, prices.unstack(level='ticker'))
                    ic_weights = compute_ic_weights(factor_df, prices.unstack(level='ticker'))
                    ic_df = pd.DataFrame(list(ic_weights.items()), columns=["Factor", "IC Weight"]).sort_values("IC Weight", ascending=False)
                    st.dataframe(ic_df)
                except Exception as e:
                    st.warning(f"Could not compute IC weights: {e}")

            # ---- SAVE RESULTS ----
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(RESULTS_DIR, f"equity_{region}_{ts}.csv")
            json_path = os.path.join(RESULTS_DIR, f"summary_{region}_{ts}.json")
            equity.to_csv(csv_path)
            pd.Series(metrics).to_json(json_path)

            st.success(f"Results saved in `{RESULTS_DIR}/` ✅")
            st.download_button(
                label="📥 Download Equity CSV",
                data=equity.to_csv().encode("utf-8"),
                file_name=f"quantpegy_equity_{region}_{ts}.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

else:
    st.info("Configure parameters and click **Run Backtest** to start.")
