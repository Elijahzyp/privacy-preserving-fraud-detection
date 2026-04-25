from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import RANDOM_STATE


def evaluate_split(
    split_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    return {
        "split": split_name,
        "model": model_name,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        "predicted_positive": int(y_pred.sum()),
    }


def maybe_load_previous_baseline(previous_path: Path):
    if not previous_path.exists():
        return None
    with previous_path.open() as f:
        previous = json.load(f)
    return previous if "metrics" in previous else None


def summarize_comparison(previous_metrics: dict, current_metrics: dict) -> list[str]:
    comparisons = []
    for key in [("ridge_score", "test"), ("logistic_regression", "test")]:
        current = current_metrics.get(key)
        previous = previous_metrics.get(key)
        if not current or not previous:
            continue

        delta_f1 = current["f1"] - previous["f1"]
        delta_pr_auc = current["pr_auc"] - previous["pr_auc"]
        if abs(delta_f1) < 0.01 and abs(delta_pr_auc) < 0.01:
            verdict = "No material performance change."
        elif delta_f1 >= 0 and delta_pr_auc >= 0:
            verdict = "Performance improved slightly."
        else:
            verdict = "Performance changed modestly."

        model_label = "Ridge" if key[0] == "ridge_score" else "Logistic Regression"
        comparisons.append(
            f"{model_label}: F1 {previous['f1']:.4f} -> {current['f1']:.4f}, "
            f"PR AUC {previous['pr_auc']:.4f} -> {current['pr_auc']:.4f}. {verdict}"
        )
    return comparisons


def choose_he_eval_subset(
    df: pd.DataFrame,
    fraud_target: int = 32,
    non_fraud_target: int = 96,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    fraud_idx = df.index[df["isFraud"] == 1].to_numpy()
    non_fraud_idx = df.index[df["isFraud"] == 0].to_numpy()

    fraud_take = min(fraud_target, len(fraud_idx))
    non_fraud_take = min(non_fraud_target, len(non_fraud_idx))

    fraud_sel = rng.choice(fraud_idx, size=fraud_take, replace=False) if fraud_take else np.array([], dtype=int)
    non_fraud_sel = (
        rng.choice(non_fraud_idx, size=non_fraud_take, replace=False) if non_fraud_take else np.array([], dtype=int)
    )

    selected = np.concatenate([fraud_sel, non_fraud_sel])
    rng.shuffle(selected)
    return df.loc[selected].reset_index(drop=True)
