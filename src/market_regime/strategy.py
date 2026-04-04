import pandas as pd
from .utils import setup_logging

logger = setup_logging("strategy")


def generate_signals(
    df: pd.DataFrame,
    confidence_threshold: float = 0.55,
    prediction_col: str = "Prediction",
    confidence_col: str = "Confidence",
) -> pd.DataFrame:

    if prediction_col not in df.columns or confidence_col not in df.columns:
        logger.warning("Predictions missing. Skipping signals.")
        return df

    df = df.copy()
    prediction = df[prediction_col]
    confidence = df[confidence_col].fillna(0)

    df["Signal"] = 0

    # Only trade when model is confident
    trade_zone = confidence >= confidence_threshold

    # Long when expecting upside
    df.loc[
        (prediction == 1) & trade_zone,
        "Signal"
    ] = 1

    # Short when expecting downside
    df.loc[
        (prediction == 0) & trade_zone,
        "Signal"
    ] = -1

    # Regime-following position
    df["Position"] = df["Signal"].where(df["Signal"] != 0).ffill().fillna(0).astype(int)

    logger.info("Signals generated (trend-following regime logic)")

    return df
