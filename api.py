import matplotlib
matplotlib.use("Agg")   # prevents GUI popup that blocks Railway

import os, sys, re, traceback, yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

app = FastAPI(title="Market Regime Detection API", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    ticker: str = "AAPL"

class ScanRequest(BaseModel):
    tickers: List[str] = ["AAPL", "SPY", "QQQ"]
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 2.0

class BacktestRequest(BaseModel):
    ticker: str = "AAPL"
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 2.0

# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(ticker: str = "AAPL",
                transaction_cost_bps: float = 1.0,
                slippage_bps: float = 2.0) -> dict:
    config_path = os.path.join(ROOT, "config", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.setdefault("ticker",  {})["default"]            = ticker
    config.setdefault("risk",    {})["transaction_cost_bps"] = transaction_cost_bps
    config.setdefault("risk",    {})["slippage_bps"]         = slippage_bps
    config.setdefault("visualize", {})["show"] = False   # never show popup
    return config

# ── Safe float helper ─────────────────────────────────────────────────────────
def _f(d: dict, *keys, default: float = 0.0) -> float:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default

# ── Core pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(ticker: str,
                 transaction_cost_bps: float = 1.0,
                 slippage_bps: float = 2.0) -> dict:
    try:
        from market_regime.pipeline import run_analysis_for_ticker

        config = load_config(ticker, transaction_cost_bps, slippage_bps)
        result = run_analysis_for_ticker(ticker, config)

        # ── Unpack AnalysisResult dataclass ───────────────────────────────────
        oos = result.stats_oos          if isinstance(result.stats_oos,  dict) else {}
        dec = result.decision           if isinstance(result.decision,   dict) else {}
        ins = result.stats_in_sample    if isinstance(result.stats_in_sample, dict) else {}
        df  = result.df_oos_eval

        # ── Fold accuracies ───────────────────────────────────────────────────
        fold_accs_raw = ins.get("fold_accuracies", ins.get("cv_scores", []))
        fold_accs = [round(float(f), 4) for f in fold_accs_raw] if fold_accs_raw else []

        # ── Equity curve ──────────────────────────────────────────────────────
        equity_data = []
        if df is not None and not df.empty:
            try:
                close_col  = next((c for c in ["Close","close","price"]              if c in df.columns), None)
                equity_col = next((c for c in ["equity","Equity","portfolio_value",
                                               "capital","Capital","cum_return"]     if c in df.columns), None)
                if close_col and equity_col:
                    sample = df[[close_col, equity_col]].dropna().tail(500)
                    equity_data = [
                        {
                            "date":   str(idx.date()) if hasattr(idx, "date") else str(idx),
                            "price":  round(float(row[close_col]),  2),
                            "equity": round(float(row[equity_col]), 2),
                        }
                        for idx, row in sample.iterrows()
                    ]
            except Exception:
                pass

        return {
            "ticker": ticker.upper(),
            "decision": {
                "action":     str(dec.get("Action",     dec.get("action",     "HOLD"))),
                "reason":     str(dec.get("Reason",     dec.get("reason",     ""))),
                "confidence": _f(dec, "Confidence", "confidence"),
                "capital":    _f(dec, "Capital",    "capital"),
                "drawdown":   abs(_f(dec, "Drawdown", "drawdown")),
            },
            "performance": {
                "sharpe_ratio":    _f(oos, "Sharpe Ratio",    "sharpe_ratio",    "oos_sharpe"),
                "strategy_return": _f(oos, "Strategy Return", "strategy_return", "Gross Strategy Return"),
                "buyhold_return":  _f(oos, "BuyHold Return",  "buyhold_return",  "buy_hold_return"),
                "max_drawdown":    abs(_f(oos, "Max Drawdown", "max_drawdown")),
                "win_rate":        _f(oos, "Win Rate",        "win_rate"),
                "cost_drag":       _f(oos, "Cost Drag",       "cost_drag"),
            },
            "model": {
                "fold_accuracies": fold_accs,
                "mean_accuracy":   round(sum(fold_accs)/len(fold_accs), 4) if fold_accs else 0,
            },
            "equity_curve": equity_data,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        )

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    path = os.path.join(ROOT, "dashboard.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>dashboard.html not found next to api.py</h1>")
    with open(path) as f:
        return HTMLResponse(f.read())

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    return JSONResponse(run_pipeline(req.ticker))

@app.post("/scan")
async def scan(req: ScanRequest):
    results = []
    for ticker in req.tickers:
        try:
            results.append(run_pipeline(ticker, req.transaction_cost_bps, req.slippage_bps))
        except HTTPException as e:
            results.append({"ticker": ticker.upper(), "error": e.detail})
    results.sort(key=lambda x: x.get("performance", {}).get("sharpe_ratio", -999), reverse=True)
    return JSONResponse({"results": results})

@app.post("/backtest")
async def backtest(req: BacktestRequest):
    return JSONResponse(run_pipeline(req.ticker, req.transaction_cost_bps, req.slippage_bps))

@app.get("/health")
async def health():
    return {"status": "ok", "version": "4.0"}

@app.get("/debug/{ticker}")
async def debug(ticker: str):
    """Visit /debug/AAPL to inspect raw AnalysisResult keys."""
    try:
        from market_regime.pipeline import run_analysis_for_ticker
        config = load_config(ticker)
        r = run_analysis_for_ticker(ticker, config)
        return JSONResponse({
            "stats_oos":        {k: str(v)[:80] for k,v in r.stats_oos.items()}       if isinstance(r.stats_oos,        dict) else str(r.stats_oos)[:300],
            "stats_in_sample":  {k: str(v)[:80] for k,v in r.stats_in_sample.items()} if isinstance(r.stats_in_sample,  dict) else str(r.stats_in_sample)[:300],
            "decision":         {k: str(v)[:80] for k,v in r.decision.items()}         if isinstance(r.decision,          dict) else str(r.decision)[:300],
            "df_oos_columns":   list(r.df_oos_eval.columns)                            if r.df_oos_eval is not None else [],
        })
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()})