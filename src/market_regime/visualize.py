import matplotlib.pyplot as plt
import pandas as pd
import os
from .utils import setup_logging
from typing import Optional

logger = setup_logging("visualize")

def plot_results(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot equity curve and market/position exposure.

    Args:
        df (pd.DataFrame): Data with 'Capital', 'Close', 'Position'.
        save_path (str, optional): Path to save the plot image.
    """
    if "Capital" not in df.columns:
        logger.warning("No Capital column to plot.")
        return

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["Capital"], label="Equity Curve")
    plt.legend()
    plt.title("Strategy Equity Curve")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(df.index, df["Close"], label="Market Price", color="gray", alpha=0.7)
    
    # Scale position for visibility if needed, or just show when we are in market
    # Position is 0 or +/- 1 usually. 
    # Let's plot Close price where we have position? 
    # Or plot position on secondary axis?
    # Original code: df["Position"] * df["Close"] -> This shows Close when Position=1, 0 otherwise.
    # If Position is -1 (short), it shows -Close. 
    
    plt.plot(df.index, df["Position"] * df["Close"], label="Position Exposure", alpha=0.6, linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        logger.info("Displaying plot...")
        plt.show()
    
    plt.close() # Free memory
