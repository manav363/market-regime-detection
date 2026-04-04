import numpy as np
import pandas as pd

import market_regime.model as model_module
from market_regime.backtest import evaluate_performance
from market_regime.features import add_features
from market_regime.model import train_model
from market_regime.risk import apply_risk_management


def test_train_model_emits_oos_predictions(synthetic_ohlcv, monkeypatch):
    raw_df = synthetic_ohlcv(rows=950, seed=11)
    feat_df = add_features(raw_df)
    monkeypatch.setattr(model_module, "log_experiment", lambda *args, **kwargs: None)

    modeled_df, _ = train_model(feat_df, horizon=5, n_splits=5, save_path=None)

    assert "InSample_Prediction" in modeled_df.columns
    assert "InSample_Confidence" in modeled_df.columns
    assert "OOS_Prediction" in modeled_df.columns
    assert "OOS_Confidence" in modeled_df.columns
    assert modeled_df["OOS_Confidence"].notna().sum() > 0
    assert modeled_df["OOS_Confidence"].isna().sum() > 0
    assert 0.1 < modeled_df["OOS_Confidence"].notna().mean() < 1.0


def test_risk_management_cost_and_turnover_accounting():
    # Position flips create turnover and therefore trading cost.
    df = pd.DataFrame(
        {
            "Position": [0, 1, 1, -1, -1, 0],
            "Volatility": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "Return": [0.0, 0.01, -0.02, 0.01, -0.01, 0.0],
            "Close": [100, 101, 99, 100, 99, 99],
        }
    )

    out = apply_risk_management(
        df,
        capital=100000,
        risk_per_trade=0.01,
        transaction_cost_bps=1.0,
        slippage_bps=2.0,
    )

    expected_turnover = np.array([0.0, 0.0, 1.0, 0.0, 2.0, 0.0])
    expected_cost = expected_turnover * 0.0003
    np.testing.assert_allclose(out["Turnover"].to_numpy(), expected_turnover, atol=1e-12)
    np.testing.assert_allclose(out["Trading_Cost"].to_numpy(), expected_cost, atol=1e-12)
    np.testing.assert_allclose(
        out["Strategy_Return"].to_numpy(),
        (out["Gross_Strategy_Return"] - out["Trading_Cost"]).to_numpy(),
        atol=1e-12,
    )

    stats = evaluate_performance(out)
    assert np.isclose(
        stats["Cost Drag"],
        stats["Gross Strategy Return"] - stats["Strategy Return"],
        atol=1e-12,
    )
    assert stats["Cost Drag"] >= -1e-12
