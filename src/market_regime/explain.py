from typing import Dict
from .utils import setup_logging

logger = setup_logging("explain")


def explain_results(stats: Dict[str, float]) -> None:

    print("\n🧾 PERFORMANCE EXPLANATION")
    print("--------------------------------------------------")

    sharpe = stats.get("Sharpe Ratio", 0)
    strat = stats.get("Strategy Return", 0) * 100
    bh = stats.get("BuyHold Return", 0) * 100
    drawdown = stats.get("Max Drawdown", 0) * 100
    win = stats.get("Win Rate", 0) * 100

    print("📈 Profitability:")
    print(f"  Strategy Return: {strat:.1f}%")
    print(f"  Buy & Hold:      {bh:.1f}%")

    if strat > bh:
        print("  → Strategy outperformed passive investing.")
    else:
        print("  → Strategy underperformed buy & hold.")

    print("\n🛡 Risk Control:")
    print(f"  Max Drawdown: {abs(drawdown):.1f}%")

    print("\n⚖️ Risk-Adjusted Performance:")
    print(f"  Sharpe Ratio: {sharpe:.2f}")

    if sharpe > 0.5:
        print("  → Good risk-adjusted performance.")
    elif sharpe > 0:
        print("  → Moderate performance.")
    else:
        print("  → Negative risk-adjusted returns.")

    print("\n🎯 Trade Accuracy:")
    print(f"  Win Rate: {win:.1f}%")

    print("\n🧠 Summary:")
    print("  The system applies ML-driven regime detection,")
    print("  volatility-adjusted risk management, and")
    print("  disciplined position sizing.")

    logger.info("Explanation printed.")
