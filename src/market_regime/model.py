import numpy as np
import pandas as pd
import joblib
import os

from typing import Dict, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from .utils import setup_logging
from .features import FEATURE_LIST
from .experiments.logger import log_experiment


logger = setup_logging("model")


DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def _positive_class_confidence(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return P(class=1) even when a fold has only one observed class."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X), dtype=float)


def train_model(
    df: pd.DataFrame,
    horizon: int = 5,
    save_path: Optional[str] = "models/model.joblib",
    n_splits: int = 5,
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, RandomForestClassifier]:

    df = df.copy()

    if len(df) < 500:
        raise ValueError("Not enough data for training")

    logger.info("Creating target labels...")

    df["Future_Return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] > 0).astype(int)

    df.dropna(inplace=True)

    X = df[FEATURE_LIST]
    y = df["Target"]

    rf_params = DEFAULT_MODEL_PARAMS.copy()
    if model_params:
        rf_params.update({k: v for k, v in model_params.items() if v is not None})

    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    oos_prediction = pd.Series(np.nan, index=df.index, dtype=float)
    oos_confidence = pd.Series(np.nan, index=df.index, dtype=float)

    logger.info("Running walk-forward validation...")

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = RandomForestClassifier(**rf_params)
        fold_model.fit(X_train, y_train)

        fold_pred = fold_model.predict(X_test)
        fold_conf = _positive_class_confidence(fold_model, X_test)

        oos_prediction.iloc[test_idx] = fold_pred
        oos_confidence.iloc[test_idx] = fold_conf

        score = fold_model.score(X_test, y_test)
        scores.append(score)

        logger.info(f"Fold {i+1} Accuracy: {score:.4f}")

    log_experiment(
        params=rf_params,
        metrics={
            "cv_accuracy": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
            "splits": n_splits,
            "oos_coverage": float(oos_confidence.notna().mean()),
        }
    )

    logger.info("Training final model on full dataset...")

    model = RandomForestClassifier(**rf_params)
    model.fit(X, y)

    df["InSample_Prediction"] = model.predict(X)
    df["InSample_Confidence"] = _positive_class_confidence(model, X)

    df["OOS_Prediction"] = oos_prediction
    df["OOS_Confidence"] = oos_confidence

    # Backward-compatible generic prediction fields default to realistic OOS values.
    df["Prediction"] = df["OOS_Prediction"].fillna(df["InSample_Prediction"])
    df["Confidence"] = df["OOS_Confidence"].fillna(df["InSample_Confidence"])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)

    return df, model
