import typer
import pandas as pd
import warnings
import time
from typing import Dict

from .utils import load_config, setup_logging
from .visualize import plot_results
from .explain import explain_results
from .pipeline import run_analysis_for_ticker
from .reports.results import save_results


warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)

app = typer.Typer()
logger = setup_logging("cli")


def _print_comparison(stats_in_sample: Dict[str, float], stats_oos: Dict[str, float]) -> None:
    preferred_order = [
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
    all_keys = [k for k in preferred_order if k in stats_in_sample or k in stats_oos]

    print("\n===== Performance (In-Sample vs Walk-Forward OOS) =====")
    print(f"{'Metric':<24} {'In-Sample':>12} {'OOS':>12}")
    print("-" * 52)

    for key in all_keys:
        is_value = stats_in_sample.get(key, float("nan"))
        oos_value = stats_oos.get(key, float("nan"))
        precision = 6 if "Cost" in key or "Turnover" in key else 3
        print(f"{key:<24} {is_value:>12.{precision}f} {oos_value:>12.{precision}f}")


@app.command()
def run(
    ticker: str = typer.Option(None),
    config_path: str = typer.Option("src/market_regime/config/config.yaml"),
    save_plot: bool = typer.Option(False)
):

    start = time.time()

    config = load_config(config_path)

    if not ticker:
        ticker = config["ticker"]["default"]
        ticker = typer.prompt("Ticker", default=ticker).upper()

    print(f"\n🚀 Analysis for {ticker}")
    print("=" * 30)

    try:
        risk_cfg = config.get("risk", {})
        transaction_cost_bps = risk_cfg.get("transaction_cost_bps", 1.0)
        slippage_bps = risk_cfg.get("slippage_bps", 2.0)
        result = run_analysis_for_ticker(
            ticker=ticker,
            config=config,
            save_model_path=f"models/{ticker}.joblib",
        )

        stats_in_sample = result.stats_in_sample
        stats_oos = result.stats_oos

        combined_stats = {f"InSample {k}": v for k, v in stats_in_sample.items()}
        combined_stats.update({f"OOS {k}": v for k, v in stats_oos.items()})
        combined_stats["Evaluation Start"] = str(result.eval_start)
        save_results(combined_stats, path=f"reports/{ticker}_results.csv")

        _print_comparison(stats_in_sample, stats_oos)
        print(f"\nCost model: transaction={transaction_cost_bps} bps, slippage={slippage_bps} bps")
        print(f"OOS evaluation start: {result.eval_start}")

        explain_results(stats_oos)

        plot_path = f"plots/{ticker}.png" if save_plot else None
        plot_results(result.df_oos_eval, save_path=plot_path)

        print("\n===== DECISION =====")

        for k, v in result.decision.items():
            print(f"{k}: {v}")

    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"❌ Error: {e}")

    print(f"\n⏱ Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    app()
