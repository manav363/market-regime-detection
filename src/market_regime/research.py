import json
import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import typer

from .data import load_data
from .pipeline import build_modeled_dataframe, evaluate_modeled_dataframe
from .utils import load_config, setup_logging


logger = setup_logging("research")
app = typer.Typer(add_completion=False)

METRIC_ORDER = [
    "Sharpe Ratio",
    "Gross Strategy Return",
    "Strategy Return",
    "Cost Drag",
    "Total Trading Cost",
    "Average Turnover",
    "BuyHold Return",
    "Max Drawdown",
    "Win Rate",
]


def _metric_slug(metric_name: str) -> str:
    return metric_name.lower().replace(" ", "_").replace("-", "_")


def _parse_csv_text(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_float_list(value: Optional[str]) -> List[float]:
    parsed: List[float] = []
    for part in _parse_csv_text(value):
        parsed.append(float(part))
    return parsed


def _coerce_tickers(config: Dict[str, Any], tickers: Optional[Iterable[str]]) -> List[str]:
    if tickers:
        return sorted({t.strip().upper() for t in tickers if t.strip()})

    ticker_cfg = config.get("ticker", {})
    universes = ticker_cfg.get("universes")
    if universes:
        return sorted({str(t).strip().upper() for t in universes if str(t).strip()})

    default_ticker = ticker_cfg.get("default", "AAPL")
    return [str(default_ticker).strip().upper()]


def _coerce_cost_scenarios(cost_scenarios: Optional[Iterable[float]]) -> List[float]:
    if cost_scenarios:
        return sorted({float(c) for c in cost_scenarios})
    return [0.0, 1.0, 5.0, 10.0]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def generate_research_report(
    config_path: str,
    output_dir: str,
    tickers: Optional[Iterable[str]] = None,
    cost_scenarios: Optional[Iterable[float]] = None,
    slippage_bps: Optional[float] = None,
) -> Dict[str, str]:
    config = load_config(config_path)
    selected_tickers = _coerce_tickers(config, tickers)
    selected_costs = _coerce_cost_scenarios(cost_scenarios)
    default_slippage = float(config.get("risk", {}).get("slippage_bps", 2.0))
    scenario_slippage = default_slippage if slippage_bps is None else float(slippage_bps)

    summary_rows: List[Dict[str, Any]] = []

    for ticker in selected_tickers:
        logger.info("Running research pipeline for %s", ticker)
        raw_df = load_data(ticker)
        modeled_df = build_modeled_dataframe(raw_df, config, save_model_path=None)

        for transaction_cost in selected_costs:
            result = evaluate_modeled_dataframe(
                ticker=ticker,
                modeled_df=modeled_df,
                config=config,
                transaction_cost_bps=float(transaction_cost),
                slippage_bps=scenario_slippage,
            )
            row: Dict[str, Any] = {
                "ticker": ticker,
                "transaction_cost_bps": float(transaction_cost),
                "slippage_bps": scenario_slippage,
                "evaluation_start": str(result.eval_start),
            }
            for metric in METRIC_ORDER:
                slug = _metric_slug(metric)
                row[f"in_sample_{slug}"] = float(result.stats_in_sample.get(metric, float("nan")))
                row[f"oos_{slug}"] = float(result.stats_oos.get(metric, float("nan")))
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["ticker", "transaction_cost_bps", "slippage_bps"], kind="mergesort"
    )
    numeric_columns = [c for c in summary_df.columns if c.startswith("oos_")]
    aggregate_columns = {
        "oos_sharpe_ratio": "mean",
        "oos_strategy_return": "mean",
        "oos_cost_drag": "mean",
        "oos_total_trading_cost": "mean",
        "oos_average_turnover": "mean",
        "oos_max_drawdown": "mean",
        "oos_win_rate": "mean",
    }
    available_aggregations = {k: v for k, v in aggregate_columns.items() if k in numeric_columns}

    cost_sensitivity_df = (
        summary_df.groupby(["transaction_cost_bps", "slippage_bps"], as_index=False)
        .agg(available_aggregations)
        .sort_values(["transaction_cost_bps", "slippage_bps"], kind="mergesort")
    )

    if "oos_strategy_return" in cost_sensitivity_df.columns and not cost_sensitivity_df.empty:
        baseline = float(cost_sensitivity_df["oos_strategy_return"].iloc[0])
        cost_sensitivity_df["delta_vs_baseline_oos_strategy_return"] = (
            cost_sensitivity_df["oos_strategy_return"] - baseline
        )

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "multi_ticker_summary.csv")
    sensitivity_path = os.path.join(output_dir, "cost_sensitivity.csv")
    report_path = os.path.join(output_dir, "research_report.md")
    manifest_path = os.path.join(output_dir, "manifest.json")

    summary_df.to_csv(summary_path, index=False, float_format="%.10f")
    cost_sensitivity_df.to_csv(sensitivity_path, index=False, float_format="%.10f")

    oos_cols = [
        c
        for c in [
            "ticker",
            "transaction_cost_bps",
            "slippage_bps",
            "oos_sharpe_ratio",
            "oos_strategy_return",
            "oos_cost_drag",
            "oos_max_drawdown",
            "oos_win_rate",
        ]
        if c in summary_df.columns
    ]
    oos_table = summary_df[oos_cols]
    sensitivity_table = cost_sensitivity_df.copy()

    report_lines = [
        "# Market Regime Research Report",
        "",
        "This report compares in-sample and walk-forward OOS performance across multiple tickers and transaction-cost scenarios.",
        "",
        f"- Tickers: {', '.join(selected_tickers)}",
        f"- Transaction cost scenarios (bps): {', '.join(str(v) for v in selected_costs)}",
        f"- Slippage (bps): {scenario_slippage}",
        "",
        "## OOS Summary By Ticker And Cost",
        _markdown_table(oos_table),
        "",
        "## Cost Sensitivity (Mean Across Tickers)",
        _markdown_table(sensitivity_table),
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    manifest = {
        "config_path": config_path,
        "tickers": selected_tickers,
        "transaction_cost_bps": selected_costs,
        "slippage_bps": scenario_slippage,
        "artifacts": {
            "summary_csv": summary_path,
            "sensitivity_csv": sensitivity_path,
            "report_md": report_path,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return {
        "summary_csv": summary_path,
        "sensitivity_csv": sensitivity_path,
        "report_md": report_path,
        "manifest_json": manifest_path,
    }


@app.command()
def run(
    config_path: str = typer.Option("src/market_regime/config/config.yaml"),
    output_dir: str = typer.Option("reports/research"),
    tickers: Optional[str] = typer.Option(
        None,
        help="Comma-separated tickers (defaults to config ticker.universes).",
    ),
    transaction_cost_bps: Optional[str] = typer.Option(
        None,
        help="Comma-separated transaction-cost scenarios in bps (default: 0,1,5,10).",
    ),
    slippage_bps: Optional[float] = typer.Option(
        None,
        help="Constant slippage in bps for each scenario (defaults to config risk.slippage_bps).",
    ),
) -> None:
    ticker_list = _parse_csv_text(tickers)
    cost_list = _parse_float_list(transaction_cost_bps)
    artifacts = generate_research_report(
        config_path=config_path,
        output_dir=output_dir,
        tickers=ticker_list or None,
        cost_scenarios=cost_list or None,
        slippage_bps=slippage_bps,
    )

    print("\n📊 Research artifacts generated")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
