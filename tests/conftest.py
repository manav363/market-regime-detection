from typing import Callable

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> Callable[[int, int], pd.DataFrame]:
    def _build(rows: int = 900, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0003, 0.012, rows)
        close = 100 * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.0015, rows))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, rows)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, rows)))
        volume = rng.integers(1_000_000, 5_000_000, rows)
        index = pd.bdate_range("2018-01-01", periods=rows)
        return pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )

    return _build
