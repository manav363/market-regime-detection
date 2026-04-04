from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .backtest import evaluate_performance
from .data import load_data
from .decision import make_trading_decision
from .features import add_features
from .model import train_model
from .risk import apply_risk_management
from .strategy import generate_signals


@dataclass
class AnalysisResult:
    ticker: str
    eval_start: pd.Timestamp
    stats_in_sample: Dict[str, float]
    stats_oos: Dict[str, float]
    decision: Dict[str, object]
    df_in_sample_eval: pd.DataFrame
    df_oos_eval: pd.DataFrame


def _read_model_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    return {
        "horizon": model_cfg.get("horizon", 5),
        "n_splits": model_cfg.get("n_splits", 5),
        "confidence_threshold": model_cfg.get("confidence_threshold", 0.55),
        "model_params": {
            "n_estimators": model_cfg.get("n_estimators", 300),
            "max_depth": model_cfg.get("max_depth", 6),
            "class_weight": model_cfg.get("class_weight", "balanced"),
            "random_state": model_cfg.get("random_state", 42),
        },
    }


def _read_risk_settings(
    config: Dict[str, Any],
    transaction_cost_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
) -> Dict[str, float]:
    risk_cfg = config.get("risk", {})
    return {
        "capital": float(risk_cfg.get("initial_capital", 100000)),
        "risk_per_trade": float(risk_cfg.get("risk_per_trade", 0.01)),
        "transaction_cost_bps": float(
            risk_cfg.get("transaction_cost_bps", 1.0)
            if transaction_cost_bps is None
            else transaction_cost_bps
        ),
        "slippage_bps": float(
            risk_cfg.get("slippage_bps", 2.0) if slippage_bps is None else slippage_bps
        ),
    }


def build_modeled_dataframe(
    raw_df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    save_model_path: Optional[str] = None,
) -> pd.DataFrame:
    df = add_features(raw_df)
    if len(df) < 300:
        raise ValueError("Insufficient data")

    model_settings = _read_model_settings(config)
    df, _ = train_model(
        df,
        horizon=model_settings["horizon"],
        n_splits=model_settings["n_splits"],
        model_params=model_settings["model_params"],
        save_path=save_model_path,
    )
    return df


def evaluate_modeled_dataframe(
    ticker: str,
    modeled_df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    transaction_cost_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
) -> AnalysisResult:
    model_settings = _read_model_settings(config)
    risk_settings = _read_risk_settings(
        config,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )

    df_in_sample = generate_signals(
        modeled_df,
        confidence_threshold=model_settings["confidence_threshold"],
        prediction_col="InSample_Prediction",
        confidence_col="InSample_Confidence",
    )
    df_in_sample = apply_risk_management(
        df_in_sample,
        capital=risk_settings["capital"],
        risk_per_trade=risk_settings["risk_per_trade"],
        transaction_cost_bps=risk_settings["transaction_cost_bps"],
        slippage_bps=risk_settings["slippage_bps"],
    )

    df_oos = generate_signals(
        modeled_df,
        confidence_threshold=model_settings["confidence_threshold"],
        prediction_col="OOS_Prediction",
        confidence_col="OOS_Confidence",
    )
    df_oos = apply_risk_management(
        df_oos,
        capital=risk_settings["capital"],
        risk_per_trade=risk_settings["risk_per_trade"],
        transaction_cost_bps=risk_settings["transaction_cost_bps"],
        slippage_bps=risk_settings["slippage_bps"],
    )

    eval_start = df_oos["OOS_Confidence"].first_valid_index()
    if eval_start is None:
        raise ValueError("No OOS predictions available. Increase data size or reduce n_splits.")

    df_in_sample_eval = df_in_sample.loc[eval_start:].copy()
    df_oos_eval = df_oos.loc[eval_start:].copy()

    stats_in_sample = evaluate_performance(df_in_sample_eval)
    stats_oos = evaluate_performance(df_oos_eval)

    decision_df = df_oos_eval
    decision_prediction_col = "OOS_Prediction"
    decision_confidence_col = "OOS_Confidence"

    if pd.isna(decision_df[decision_confidence_col].iloc[-1]):
        decision_df = df_in_sample_eval
        decision_prediction_col = "InSample_Prediction"
        decision_confidence_col = "InSample_Confidence"

    decision = make_trading_decision(
        decision_df,
        capital=decision_df["Capital"].iloc[-1],
        prediction_col=decision_prediction_col,
        confidence_col=decision_confidence_col,
    )

    return AnalysisResult(
        ticker=ticker,
        eval_start=eval_start,
        stats_in_sample=stats_in_sample,
        stats_oos=stats_oos,
        decision=decision,
        df_in_sample_eval=df_in_sample_eval,
        df_oos_eval=df_oos_eval,
    )


def run_analysis_for_ticker(
    ticker: str,
    config: Dict[str, Any],
    *,
    save_model_path: Optional[str] = None,
    transaction_cost_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
) -> AnalysisResult:
    raw_df = load_data(ticker)
    modeled_df = build_modeled_dataframe(raw_df, config, save_model_path=save_model_path)
    return evaluate_modeled_dataframe(
        ticker,
        modeled_df,
        config,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
