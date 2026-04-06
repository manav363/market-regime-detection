"""
Market Regime Detection — FastAPI Backend v3
Author: Manav Garg

Runs your existing `market-regime` CLI as a subprocess and parses its output.
Zero import chain issues — if the CLI works in your terminal, this works too.

Usage:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000
    open http://localhost:8000
"""

import os, sys, re, subprocess, traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

ROOT     = os.path.dirname(os.path.abspath(__file__))
VENV_PY  = os.path.join(ROOT, "venv", "bin", "python3")
SRC_PATH = os.path.join(ROOT, "src")

app = FastAPI(title="Market Regime Detection API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

# ── CLI runner ────────────────────────────────────────────────────────────────
def run_cli(ticker: str, timeout: int = 120) -> str:
    """
    Run `market-regime` with the ticker piped via stdin.
    Returns the full stdout+stderr output string.
    """
    # Try multiple possible CLI locations (local venv vs Railway)
    candidates = [
        os.path.join(ROOT, "venv", "bin", "market-regime"),
        "/usr/local/bin/market-regime",
        "/usr/bin/market-regime",
    ]
    import shutil
    cli = next((c for c in candidates if os.path.exists(c)), None) or shutil.which("market-regime")
    if not cli:
        raise FileNotFoundError("market-regime CLI not found")
    cmd = [cli]

    try:
        proc = subprocess.run(
            cmd if isinstance(cmd, list) else [cmd],
            input=f"{ticker}\n",
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": SRC_PATH},
        )
        output = proc.stdout + proc.stderr
        if not output.strip():
            raise ValueError("CLI produced no output. Check venv path.")
        return output
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail=f"Analysis timed out after {timeout}s")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"market-regime CLI not found at {CLI}.\nMake sure venv is activated and the package is installed."
        )

# ── Output parser ─────────────────────────────────────────────────────────────
def parse_output(output: str, ticker: str) -> dict:
    """
    Parse the structured text output of `market-regime`.

    Handles the exact format your CLI produces:

    Fold N Accuracy: 0.5223
    ...
    Metric                 In-Sample    OOS
    Sharpe Ratio               0.981  0.982
    Strategy Return            7.819  7.796
    BuyHold Return            14.582 14.582
    Max Drawdown              -0.258 -0.258
    Win Rate                   0.528  0.526
    Cost Drag               0.000000  0.002625
    ...
    ===== DECISION =====
    Action: HOLD
    Reason: Low confidence
    Confidence: 0.5111
    Drawdown: -0.1458
    Capital: 879635.31
    """

    def first_float(pattern, text, default=0.0):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                pass
        return default

    def oos_float(label, text, default=0.0):
        """Extract OOS (second) value from a two-column metrics table."""
        pattern = rf"{re.escape(label)}\s+([\-\d.]+)\s+([\-\d.]+)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(2))   # OOS is always column 2
            except (ValueError, IndexError):
                pass
        return default

    # ── Fold accuracies ───────────────────────────────────────────────────────
    fold_accs = [float(x) for x in re.findall(r"Fold \d+ Accuracy:\s*([\d.]+)", output)]

    # ── OOS performance metrics ───────────────────────────────────────────────
    # NOTE: "Strategy Return" appears after "Gross Strategy Return" in output
    # We use the last match for Strategy Return (the net one)
    sharpe   = oos_float("Sharpe Ratio",         output)
    # Find all Strategy Return matches, take the last (net) one
    sr_matches = re.findall(r"Strategy Return\s+([\-\d.]+)\s+([\-\d.]+)", output)
    strat_r  = float(sr_matches[-1][1]) if sr_matches else 0.0
    bh_r     = oos_float("BuyHold Return",       output)
    max_dd   = oos_float("Max Drawdown",         output)
    win_rate = oos_float("Win Rate",             output)
    cost     = oos_float("Cost Drag",            output)

    # ── Decision block ────────────────────────────────────────────────────────
    action     = (re.search(r"Action:\s*(\w+)",         output) or re.search(r"", "")).group(1) if re.search(r"Action:\s*(\w+)", output) else "HOLD"
    reason     = (re.search(r"Reason:\s*(.+)",          output) or re.search(r"", "")).group(1).strip() if re.search(r"Reason:\s*(.+)", output) else ""
    confidence = first_float(r"Confidence:\s*([\d.]+)", output)
    drawdown   = abs(first_float(r"Drawdown:\s*([-\d.]+)", output))
    capital    = first_float(r"Capital:\s*([\d.]+)",    output)

    return {
        "ticker": ticker.upper(),
        "decision": {
            "action":     action,
            "reason":     reason,
            "confidence": round(confidence, 4),
            "capital":    round(capital,    2),
            "drawdown":   round(drawdown,   4),
        },
        "performance": {
            "sharpe_ratio":    round(sharpe,   3),
            "strategy_return": round(strat_r,  3),
            "buyhold_return":  round(bh_r,     3),
            "max_drawdown":    round(abs(max_dd), 3),
            "win_rate":        round(win_rate,  3),
            "cost_drag":       round(cost,      6),
        },
        "model": {
            "fold_accuracies": [round(f, 4) for f in fold_accs],
            "mean_accuracy":   round(sum(fold_accs)/len(fold_accs), 4) if fold_accs else 0,
        },
        "equity_curve": [],   # chart populated from df — see /debug for full output
        "_raw": output[-1000:],  # last 1000 chars for debugging
    }

# ── Core pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(ticker: str, **kwargs) -> dict:
    output = run_cli(ticker)
    return parse_output(output, ticker)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    path = os.path.join(ROOT, "dashboard.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>Place dashboard.html next to api.py and restart.</h1>")
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
            results.append(run_pipeline(ticker))
        except Exception as e:
            results.append({"ticker": ticker.upper(), "error": str(e)})
    results.sort(key=lambda x: x.get("performance", {}).get("sharpe_ratio", -999), reverse=True)
    return JSONResponse({"results": results})

@app.post("/backtest")
async def backtest(req: BacktestRequest):
    return JSONResponse(run_pipeline(req.ticker,
                                     transaction_cost_bps=req.transaction_cost_bps,
                                     slippage_bps=req.slippage_bps))

@app.get("/health")
async def health():
    return {"status": "ok", "cli": CLI, "cli_exists": os.path.exists(CLI)}

@app.get("/debug/{ticker}")
async def debug(ticker: str):
    """Visit /debug/AAPL to see raw CLI output and what got parsed."""
    try:
        output = run_cli(ticker)
        parsed = parse_output(output, ticker)
        return JSONResponse({"raw_output": output, "parsed": parsed})
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()})