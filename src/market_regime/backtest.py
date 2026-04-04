import numpy as np
import pandas as pd
from typing import Dict
from .utils import setup_logging

logger = setup_logging("backtest")


def evaluate_performance(df: pd.DataFrame) -> Dict[str, float]:

    if "Capital" not in df.columns:
        return {}

    strategy_return = df.get("Strategy_Return", pd.Series(0.0, index=df.index)).fillna(0)
    gross_strategy_return = df.get("Gross_Strategy_Return", strategy_return).fillna(0)
    trading_cost = df.get("Trading_Cost", pd.Series(0.0, index=df.index)).fillna(0)
    turnover = df.get("Turnover", pd.Series(0.0, index=df.index)).fillna(0)

    sharpe = np.sqrt(252) * strategy_return.mean() / (strategy_return.std() + 1e-6)

    strat_ret = (1 + strategy_return).prod() - 1
    bh_ret = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1

    roll_max = df["Capital"].cummax()
    dd = df["Capital"] / roll_max - 1

    max_dd = dd.min()

    win = (strategy_return > 0).mean()
    gross_ret = (1 + gross_strategy_return).prod() - 1
    total_cost = trading_cost.sum()
    avg_turnover = turnover.mean()
    cost_drag = gross_ret - strat_ret

    return {
        "Sharpe Ratio": float(sharpe),
        "Gross Strategy Return": float(gross_ret),
        "Strategy Return": float(strat_ret),
        "Cost Drag": float(cost_drag),
        "Total Trading Cost": float(total_cost),
        "Average Turnover": float(avg_turnover),
        "BuyHold Return": float(bh_ret),
        "Max Drawdown": float(max_dd),
        "Win Rate": float(win)
    }
