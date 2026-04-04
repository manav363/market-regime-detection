import yfinance as yf # type: ignore
import pandas as pd
import time
from .utils import setup_logging

logger = setup_logging("data")

def load_data(symbol: str, start: str = "2010-01-01", retries: int = 3) -> pd.DataFrame:
    """
    Download historical market data from Yahoo Finance.

    Args:
        symbol (str): Ticker symbol (e.g., 'AAPL').
        start (str): Start date for data download.
        retries (int): Number of retries for failed downloads.

    Returns:
        pd.DataFrame: DataFrame with Open, High, Low, Close, Volume.
    
    Raises:
        ValueError: If data cannot be downloaded or is insufficient.
    """
    df = pd.DataFrame()
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
            
            if df is not None and not df.empty:
                break
        except Exception as e:
            logger.warning(f"Download attempt {attempt+1} failed: {e}")
        
        logger.info(f"Download failed. Retrying... ({attempt+1}/{retries})")
        time.sleep(2)

    if df is None or df.empty:
        raise ValueError(f"Yahoo Finance unavailable for {symbol}")

    # Flatten columns if needed (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    # Check if columns exist
    if not all(col in df.columns for col in required_cols):
         # Try to map lower case if needed or check what happened. 
         # Yfinance usually returns Capitalized.
         pass
         
    df = df[required_cols]
    df.dropna(inplace=True)

    if len(df) < 300:
        raise ValueError("Not enough historical data (minimum 300 points required)")

    logger.info(f"Successfully loaded {len(df)} rows for {symbol}")
    return df
