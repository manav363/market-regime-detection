import pandas as pd
import numpy as np
from typing import Dict
from .utils import setup_logging

logger = setup_logging("decision")


def make_trading_decision(
    df: pd.DataFrame,
    capital: float,
    min_confidence: float = 0.6,
    max_drawdown: float = -0.15,
    prediction_col: str = "Prediction",
    confidence_col: str = "Confidence",
    position_col: str = "Position",
) -> Dict[str, object]:

    last = df.iloc[-1]

    confidence_raw = last.get(confidence_col, 0)
    prediction_raw = last.get(prediction_col, 0)
    position_raw = last.get(position_col, 0)

    confidence = 0.0 if pd.isna(confidence_raw) else float(confidence_raw)
    prediction = 0 if pd.isna(prediction_raw) else int(prediction_raw)
    position = 0 if pd.isna(position_raw) else int(position_raw)

    peak = df["Capital"].cummax().iloc[-1]
    drawdown = 0.0 if peak <= 0 or np.isclose(peak, 0.0) else capital / peak - 1

    decision = {
        "Confidence": confidence,
        "Drawdown": drawdown,
        "Capital": capital
    }

    if drawdown < max_drawdown:
        return {"Action": "EXIT", "Reason": "Max drawdown breached", **decision}

    if confidence < min_confidence:
        return {"Action": "HOLD", "Reason": "Low confidence", **decision}

    if prediction == 1 and position <= 0:
        return {"Action": "BUY", "Reason": "Bullish regime", **decision}

    if prediction == 0 and position >= 0:
        return {"Action": "SELL", "Reason": "Bearish regime", **decision}

    return {"Action": "HOLD", "Reason": "No change", **decision}
