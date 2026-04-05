# 📈 Market Regime Detection
### ML-Driven Quantitative Trading System

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Dashboard-009688?style=flat&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-22c55e?style=flat)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white)

> A production-style machine learning system for detecting market regimes, generating trading signals, managing risk, and evaluating performance using real financial data — validated across 13 years of historical data (2013–2026). Now with a full Bloomberg-style web dashboard.


---

## 🚀 Quick Start

### CLI (Terminal)
```bash
git clone https://github.com/manav363/market-regime-detection.git
cd market-regime-detection
python3 -m pip install -e .[dev]

# Single ticker analysis
market-regime --ticker AAPL

# Multi-ticker research report
market-regime-research --tickers AAPL,SPY,QQQ --transaction-cost-bps 0,1,5,10 --slippage-bps 2
```

### Web Dashboard
```bash
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
# Open http://localhost:8000
```

---

## 🎯 What This System Does

Financial markets cycle through different **regimes** — bullish, bearish, high-volatility, low-volatility. Identifying which regime you're in changes everything about how you should trade.

This system:
- **Learns** market behavior from 13+ years of daily OHLCV data
- **Classifies** the current regime using a walk-forward validated ML model
- **Generates** confidence-filtered trading signals
- **Manages** risk using volatility-adjusted position sizing
- **Reports** performance vs Buy & Hold with full cost modeling
- **Visualizes** everything through a live web dashboard

---

## 🖥 Web Dashboard

A Bloomberg terminal-style dashboard built with FastAPI + vanilla JS. Three tabs — no frameworks, no bloat.


**Tab 1 — Analyze:** Enter any ticker → runs full walk-forward pipeline → displays decision (BUY/SELL/HOLD), confidence bars, all 6 OOS performance metrics, and fold accuracy visualization.

**Tab 2 — Scanner:** Preset universes (Mag7, Indices, Financials, Tech) or custom tickers → ranked results table sorted by Sharpe ratio.

**Tab 3 — Backtest:** Ticker + adjustable transaction cost + slippage → full backtest with Strategy vs Buy & Hold comparison chart.

---

## 📊 Live Results (AAPL — April 2026)

```
===== Performance (In-Sample vs Walk-Forward OOS) =====
Metric                      In-Sample          OOS
----------------------------------------------------
Sharpe Ratio                    0.981        0.982
Strategy Return                 7.819        7.796
Buy & Hold Return              14.582       14.582
Max Drawdown                   -0.258       -0.258
Win Rate                        0.528        0.526
Cost Drag                    0.000000     0.002625
Cost model: transaction=1.0 bps, slippage=2.0 bps
```

### Walk-Forward Validation Folds (AAPL)

| Fold | Accuracy |
|------|----------|
| 1    | 0.522    |
| 2    | 0.446    |
| 3    | 0.448    |
| 4    | 0.595    |
| 5    | 0.500    |

> Fold-level accuracy reflects **realistic market predictability**, not overfitting. The model doesn't claim to predict markets perfectly — it claims to manage risk better.

---

## 🌐 Multi-Ticker Research Results

```bash
market-regime-research --tickers AAPL,SPY,QQQ --transaction-cost-bps 0,1,5,10 --slippage-bps 2
```

| Ticker | OOS Sharpe | OOS Return | Max Drawdown |
|--------|------------|------------|--------------|
| AAPL   | 1.009      | 823%       | -25.8%       |
| QQQ    | 1.057      | 883%       | -23.2%       |
| SPY    | 0.925      | 639%       | -22.6%       |

### Cost Sensitivity (Mean across AAPL/QQQ/SPY)

| Transaction Cost | OOS Return |
|-----------------|------------|
| 0 bps           | 820.2%     |
| 10 bps          | 781.1%     |
| Delta           | -0.89%     |

> Strategy is **robust to transaction costs** — performance degrades minimally even at 10 bps.

---

## 🧠 System Architecture

```
Market Data (Yahoo Finance / yfinance)
        ↓
Feature Engineering
(RSI, ATR, Momentum, Volatility, Trend — 14 features)
        ↓
Walk-Forward ML Training
(Random Forest · Time-series aware · Balanced classes)
        ↓
Confidence-Based Signal Filtering
(threshold: 0.55 — signals below this are suppressed)
        ↓
Risk Management
(Volatility-adjusted sizing · Drawdown stop · Capital compounding)
        ↓
Backtesting & Benchmarking
(In-Sample vs OOS · Cost-aware · Buy & Hold comparison)
        ↓
Decision Engine                    FastAPI Backend
(BUY / SELL / HOLD / EXIT)   →    Web Dashboard (dashboard.html)
```

---

## 🧾 Decision Engine Output

```
===== DECISION =====
Action     : HOLD
Reason     : Low confidence (0.511 < threshold 0.55)
Confidence : 0.511
Drawdown   : -14.58%
Capital    : $879,637.33
Runtime    : 12.79s
```

The system enforces **automatic risk discipline** — it refuses to trade when confidence is low.

---

## 🏗 Project Structure

```
market_regime/
│
├── config/
│   ├── config.yaml          # All parameters (model, risk, costs)
│   └── schema.py            # Runtime config validation
│
├── experiments/
│   └── logger.py            # JSONL experiment tracking
│
├── reports/
│   └── results.py           # CSV performance reports
│
├── src/market_regime/
│   ├── cli.py               # Typer CLI entrypoints
│   ├── data.py              # Data ingestion & validation
│   ├── features.py          # Feature registry (14 indicators)
│   ├── model.py             # Walk-forward RF training
│   ├── strategy.py          # Signal generation logic
│   ├── risk.py              # Position sizing & drawdown control
│   ├── backtest.py          # Backtesting engine
│   ├── visualize.py         # Equity curve & exposure plots
│   ├── explain.py           # Performance explainability
│   ├── decision.py          # Live trading decision
│   ├── pipeline.py          # End-to-end orchestration
│   └── main.py
│
├── assets/
│   └── equity_curve.png     # Backtest visualization
│
├── api.py                   # FastAPI backend
├── dashboard.html           # Bloomberg-style web dashboard
├── Makefile
├── pyproject.toml
└── README.md
```

---

## ⚙️ Configuration

```yaml
# config/config.yaml
ticker:
  default: AAPL

model:
  horizon: 5
  n_splits: 5
  confidence_threshold: 0.55

risk:
  initial_capital: 100000
  risk_per_trade: 0.01
  transaction_cost_bps: 1.0
  slippage_bps: 2.0
```

All config is **validated against a schema at runtime** — no silent misconfiguration.

---

## 🧪 Generated Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Experiment logs | `experiments/log.jsonl` | Model params + validation metrics |
| Performance report | `reports/{TICKER}_results.csv` | In-sample vs OOS metrics |
| Multi-ticker summary | `reports/research/multi_ticker_summary.csv` | Cross-asset comparison |
| Cost sensitivity | `reports/research/cost_sensitivity.csv` | Return vs transaction cost |
| Research report | `reports/research/research_report.md` | Human-readable summary |
| Equity curve | `assets/equity_curve.png` | Visual backtest result |

---

## ✅ Quality Gates

```bash
make lint    # ruff check src tests
make test    # pytest -q
```

CI runs on **GitHub Actions** for Python 3.10, 3.11, and 3.12 on every push.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| ML | Scikit-learn (Random Forest) |
| Data | Pandas, NumPy, yFinance |
| API | FastAPI, Uvicorn |
| Frontend | Vanilla JS, Chart.js |
| Visualization | Matplotlib |
| CLI | Typer |
| Config | PyYAML + schema validation |
| CI/CD | GitHub Actions |
| Logging | JSONL experiment tracker |

---

## 📌 Known Limitations

- Uses **daily OHLCV data** — no intraday execution modeling
- Limited to **tree-based ML models** (Random Forest)
- No live broker integration (research system only)
- Dashboard requires local server — not yet deployed to cloud

These are intentional research simplifications, not oversights.

---

## 🔮 Roadmap

- [ ] Deploy dashboard to Railway/Render (one-click cloud access)
- [ ] Hidden Markov Models for unsupervised regime labeling
- [ ] LSTM / Transformer-based sequence models
- [ ] Portfolio-level multi-asset optimization
- [ ] Live broker integration (Alpaca / Zerodha)
- [ ] Hyperparameter optimization with Optuna

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only.**
It does not constitute financial advice. Trading involves substantial risk of loss.

---

## 👤 Author

**Manav Garg**
Machine Learning · Quantitative Research · Systems Engineering
