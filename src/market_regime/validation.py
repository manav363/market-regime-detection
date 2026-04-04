import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_cols: list):

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.isnull().any().any():
        raise ValueError("NaN values detected in dataframe")

    if len(df) < 100:
        raise ValueError("Insufficient data length")