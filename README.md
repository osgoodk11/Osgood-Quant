# Quantitative Momentum Portfolio

A systematic trend-following and momentum strategy built in Python, backtested from 2019–2024 and connected to live paper trading via the Alpaca Markets API.

Built as a personal quantitative finance project, with a custom execution layer inspired by the open-source [GS Quant](https://github.com/goldmansachs/gs-quant) library's architecture.

---

## Strategy Overview

The strategy holds up to 5 ETFs and rotates monthly based on two signals:

| Signal | Logic |
|--------|-------|
| **Trend filter** | Price > 60-day MA **and** 20-day MA > 60-day MA |
| **Momentum** | 63-day return ranking (top ranked ETFs get capital) |

If no ETF passes the trend filter, the strategy moves to 100% cash — a built-in bear market escape.

**Universe:** SPY · QQQ · IWM · TLT · GLD  
**Rebalance:** Monthly, last trading day  
**Transaction cost:** 5 bps per trade

---

## Position Sizing Methods

| Method | Description |
|--------|-------------|
| Equal-Weight (EW) | Capital split evenly across qualifying ETFs |
| Inverse-Volatility (IV) | More weight to lower-volatility ETFs |
| Volatility-Targeted (VT) | Scales position size to hit a 10% annualized vol target |

---

## Performance Metrics Computed

CAGR · Sharpe Ratio · Sortino Ratio · Calmar Ratio · Treynor Ratio · Information Ratio · Omega Ratio · Half-Kelly Criterion · VaR 95% · CVaR 95% · Win Rate · Max Drawdown

---

## Technical Indicators

Moving Average · EMA · Bollinger Bands · RSI (14-day EWMA) · MACD (12/26/9) · Realized Volatility · Rolling Beta · Pearson Correlation · Hurst Exponent (R/S analysis)

---

## Validation

- **Walk-forward test:** In-sample (2019–2021) vs. out-of-sample (2022–2024) to check for overfitting
- **Parameter sensitivity:** 3×3 grid of fast/slow moving average windows — Sharpe and CAGR heatmaps
- **Benchmark:** 60/40 portfolio (monthly-rebalanced 60% SPY / 40% TLT) + SPY buy-and-hold

---

## Project Structure

```
quant-momentum-portfolio/
├── quant_portfolio.ipynb       # Full strategy: research, backtest, analytics, live trading
├── quant_portfolio.html        # Rendered presentation (open in any browser)
└── gs_quant_alpaca/            # Custom execution layer
    ├── config.py               # AlpacaConfig — loads API keys from environment
    ├── client.py               # AlpacaBroker — wraps alpaca-py SDK
    └── portfolio.py            # Signal generation and rebalance logic
```

---

## Setup

### 1. Install dependencies
```bash
pip install yfinance alpaca-py pandas numpy matplotlib seaborn scipy jupyter ipykernel
```

### 2. Set Alpaca paper trading keys
Get free paper trading keys at [alpaca.markets](https://alpaca.markets) (no funding required).

Add to your shell profile (`~/.zshrc` or `~/.bashrc`):
```bash
export ALPACA_API_KEY_ID="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here"
```
Then reload: `source ~/.zshrc`

### 3. Register the Jupyter kernel
```bash
python3 -m ipykernel install --user --name=gs-quant-portfolio --display-name="GS Quant Portfolio"
```

### 4. Open the notebook
Open `quant_portfolio.ipynb` in VS Code or JupyterLab and run all cells top-to-bottom.

> **Note:** API keys are read from environment variables only — never hardcoded. Live trading requires an additional `ALPACA_ALLOW_LIVE_TRADING=1` env var as a safety gate.

---

## Quick View

To view results without running any code, open `quant_portfolio.html` in any browser — all charts and metrics are pre-rendered.

---

## Disclaimer

This project is for educational and research purposes only. It is not financial advice. The `gs_quant_alpaca` module is a custom extension written for this project and is **not** part of Goldman Sachs's official GS Quant library.
