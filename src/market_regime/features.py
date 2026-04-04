import numpy as np
import pandas as pd
from .utils import setup_logging

logger = setup_logging("features")


FEATURE_LIST = [
    "Return",
    "Log_Return",
    "Trend",
    "Volatility",
    "ATR",
    "Momentum",
    "RSI"
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        logger.warning("Empty DataFrame passed to add_features")
        return df

    logger.info("Generating technical features...")
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Trend"] = df["MA_20"] - df["MA_50"]

    df["Volatility"] = df["Return"].rolling(20).std()
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    df["Momentum"] = df["Close"] - df["Close"].shift(10)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    logger.info(f"Features generated. Shape: {df.shape}")
    return df