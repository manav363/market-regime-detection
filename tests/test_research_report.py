import json
from pathlib import Path

import pandas as pd

import market_regime.model as model_module
import market_regime.research as research_module
from market_regime.research import generate_research_report


def test_research_report_artifacts_are_reproducible(tmp_path, synthetic_ohlcv, monkeypatch):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "research_out"
    config_path.write_text(
        "\n".join(
            [
                "ticker:",
                "  default: AAA",
                "  universes:",
                "    - AAA",
                "    - BBB",
                "model:",
                "  horizon: 5",
                "  n_splits: 4",
                "  confidence_threshold: 0.55",
                "  n_estimators: 40",
                "  max_depth: 5",
                "  class_weight: balanced",
                "  random_state: 42",
                "risk:",
                "  initial_capital: 100000",
                "  risk_per_trade: 0.01",
                "  transaction_cost_bps: 1.0",
                "  slippage_bps: 2.0",
            ]
        ),
        encoding="utf-8",
    )

    def fake_load_data(symbol: str, start: str = "2010-01-01", retries: int = 3):
        del start, retries
        seed = 100 if symbol == "AAA" else 200
        return synthetic_ohlcv(rows=900, seed=seed)

    monkeypatch.setattr(research_module, "load_data", fake_load_data)
    monkeypatch.setattr(model_module, "log_experiment", lambda *args, **kwargs: None)

    artifacts_a = generate_research_report(
        config_path=str(config_path),
        output_dir=str(output_dir),
        tickers=["AAA", "BBB"],
        cost_scenarios=[0.0, 5.0],
        slippage_bps=2.0,
    )
    artifacts_b = generate_research_report(
        config_path=str(config_path),
        output_dir=str(output_dir),
        tickers=["AAA", "BBB"],
        cost_scenarios=[0.0, 5.0],
        slippage_bps=2.0,
    )

    for artifact_path in artifacts_a.values():
        assert Path(artifact_path).exists()

    summary = pd.read_csv(artifacts_a["summary_csv"])
    assert len(summary) == 4
    assert sorted(summary["ticker"].unique().tolist()) == ["AAA", "BBB"]
    assert set(summary["transaction_cost_bps"].tolist()) == {0.0, 5.0}
    assert "oos_strategy_return" in summary.columns

    # Deterministic artifacts: second run should produce identical file content.
    for key in artifacts_a:
        a_bytes = Path(artifacts_a[key]).read_bytes()
        b_bytes = Path(artifacts_b[key]).read_bytes()
        assert a_bytes == b_bytes

    manifest = json.loads(Path(artifacts_a["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["tickers"] == ["AAA", "BBB"]
    assert manifest["transaction_cost_bps"] == [0.0, 5.0]
