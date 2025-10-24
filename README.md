# QuantPEGY: Macro Regime-Aware Factor Investing Strategy

QuantPEGY is a dynamic factor-based portfolio construction engine that selects undervalued, fundamentally strong stocks using the PEGY (P/E-to-Growth + Dividend Yield) ratio. This version incorporates macroeconomic regime detection to adjust investment strategies contextually (risk-on, risk-off, neutral).

---

## 📁 File Structure

```
quantpegy/
├── __init__.py               # Package initializer
├── alpha_boost.py            # (To be added in Phase 2) Information Coefficient-based factor weighting
├── backtest.py               # Backtesting engine to simulate portfolio equity over time
├── data.py                   # Downloads price & fundamental data using yfinance
├── factors.py                # Computes PEGY, Momentum, and Volatility factor scores
├── macro.py                  # Fetches macroeconomic indicators and classifies market regime
├── optimizer.py              # Portfolio weight optimization: Equal, Max Sharpe, Min Volatility
├── run.py                    # Main entry-point for executing the full pipeline (CLI)
├── streamlit_app.py          # Interactive dashboard to visualize factor signals and portfolio performance
```

---

## How to Run the Project

Make sure you have Python 3.9+ and run the following setup:

```bash
# Step 1: Create virtual environment
python -m venv .venv
.\.venv\Scriptsctivate  # Windows

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run CLI pipeline (example with US region)
python -m quantpegy.run --region US --start 2023-01-01 --end 2024-01-01 --top_n 10 --optimisation max_sharpe --use_macro

# Step 4: Run the Streamlit Dashboard (optional)
streamlit run quantpegy/streamlit_app.py
```

---

## Parameters Available

| Parameter       | Description                                       | Example                |
|----------------|---------------------------------------------------|------------------------|
| `--region`      | Region to analyze (`US` or `IN`)                  | `--region US`          |
| `--start`       | Start date of analysis                            | `--start 2023-01-01`   |
| `--end`         | End date of analysis                              | `--end 2024-01-01`     |
| `--top_n`       | Number of top-ranked stocks to include            | `--top_n 10`           |
| `--optimisation`| Optimizer: equal, max_sharpe, min_volatility      | `--optimisation max_sharpe` |
| `--use_macro`   | Enables macro regime classification               | `--use_macro`          |
| `--use_ic`      | Enables IC-based factor weighting (⚠️ coming soon) | `--use_ic`             |

---

## 📈 Output & Results

After successful execution, you will see:

1. **Backtest Equity Curve**: 
   - Shows the growth of a $100 portfolio over the selected period.
   - Higher slope → stronger CAGR.

2. **Performance Metrics**:
   - `CAGR` (Compounded Annual Growth Rate): Portfolio return growth rate.
   - `Sharpe Ratio`: Return per unit risk (higher is better).
   - `Max Drawdown`: Largest drop from peak (lower is safer).

3. **Selected Stocks per Rebalance** (Streamlit only):
   - Stocks with highest composite score based on PEGY, Momentum, Volatility.

4. **Current Macro Regime**:
   - Based on VIX, Yield Curve, and Inflation.
   - Can be `risk_on`, `risk_off`, or `neutral`.

---

## 🧠 What's Unique?

- Combines valuation, momentum, and volatility in a single ranking system.
- Macro regime aware: Adapts strategy contextually based on economic signals.
- Future-ready with planned IC-weighted factor blending (Phase 2).

---

## 🏁 Next Phase

> Future versions will include dynamic **IC-weighted factors**, regime-specific strategies, and transaction cost modeling.


QuantPEGY Portfolio Engine | Developed by [Your Name]  
For queries or contributions, please contact: [your.email@example.com]
