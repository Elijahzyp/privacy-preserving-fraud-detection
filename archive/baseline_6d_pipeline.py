from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CHUNK_SIZE = 250_000
NEG_POS_RATIO = 10
TEST_SIZE = 0.2
VAL_SIZE_WITHIN_TRAIN = 0.2
SELECTED_TYPES = ["TRANSFER", "CASH_OUT"]

NUMERIC_BASE_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "deltaOrig",
    "deltaDest",
]
NUMERIC_SCALED_FEATURES = [
    "amount_scaled",
    "oldbalanceOrg_scaled",
    "oldbalanceDest_scaled",
    "deltaOrig_scaled",
    "deltaDest_scaled",
]
BINARY_FEATURES = ["is_transfer"]
FEATURE_COLUMNS = NUMERIC_SCALED_FEATURES + BINARY_FEATURES

SUMMARY_JSON_PATH = ARTIFACT_DIR / "baseline_6d_summary.json"
FEATURE_ORDER_JSON_PATH = ARTIFACT_DIR / "feature_order_6d.json"
SCALER_PICKLE_PATH = ARTIFACT_DIR / "baseline_6d_scaler.pkl"
RIDGE_PICKLE_PATH = ARTIFACT_DIR / "baseline_6d_ridge.pkl"
LOGREG_PICKLE_PATH = ARTIFACT_DIR / "baseline_6d_logreg.pkl"
HE_FEATURES_PATH = ARTIFACT_DIR / "he_eval_scaled_features_6d.csv"
HE_META_PATH = ARTIFACT_DIR / "he_eval_meta_6d.csv"
HE_SCORES_PATH = ARTIFACT_DIR / "he_eval_plaintext_scores_6d.csv"


def scan_selected_type_counts(data_path: Path, chunk_size: int = CHUNK_SIZE) -> tuple[int, int]:
    selected_non_fraud = 0
    selected_fraud = 0

    usecols = ["type", "isFraud"]
    for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=chunk_size):
        chunk = chunk[chunk["type"].isin(SELECTED_TYPES)]
        if chunk.empty:
            continue

        fraud_count = int(chunk["isFraud"].sum())
        selected_fraud += fraud_count
        selected_non_fraud += int((chunk["isFraud"] == 0).sum())

    return selected_fraud, selected_non_fraud


def build_sample_dataframe(
    data_path: Path,
    negative_sample_prob: float,
    chunk_size: int = CHUNK_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    usecols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]

    sampled_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=chunk_size):
        chunk = chunk[chunk["type"].isin(SELECTED_TYPES)].copy()
        if chunk.empty:
            continue

        fraud_df = chunk[chunk["isFraud"] == 1]
        non_fraud_df = chunk[chunk["isFraud"] == 0]
        take_mask = rng.random(len(non_fraud_df)) < negative_sample_prob
        sampled_non_fraud_df = non_fraud_df.loc[take_mask]
        sampled_chunks.append(pd.concat([fraud_df, sampled_non_fraud_df], ignore_index=True))

    sample_df = pd.concat(sampled_chunks, ignore_index=True)
    sample_df = sample_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sample_df.insert(0, "sample_row_id", np.arange(len(sample_df), dtype=int))
    return sample_df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["deltaOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["deltaDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    return df


def add_scaled_feature_columns(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    scaled_values = scaler.transform(df[NUMERIC_BASE_FEATURES])
    scaled_df = pd.DataFrame(scaled_values, columns=NUMERIC_SCALED_FEATURES, index=df.index)
    return pd.concat([df, scaled_df], axis=1)


def evaluate_split(
    split_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "split": split_name,
        "model": model_name,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": cm.astype(int).tolist(),
        "predicted_positive": int(y_pred.sum()),
    }


def weighted_standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    return (X - mean) / scale, mean, scale


def fit_ridge_score_model(X: np.ndarray, y_pm1: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    Xz, x_mean, x_scale = weighted_standardize(X)
    X_aug = np.column_stack([Xz, np.ones(len(Xz))])
    reg = alpha * np.eye(X_aug.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_pm1.astype(float))

    weights_z = coef[:-1]
    bias_z = coef[-1]
    weights_raw = weights_z / x_scale
    bias_raw = bias_z - np.sum((weights_z * x_mean) / x_scale)
    return weights_raw, float(bias_raw)


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    max_iter: int = 500,
) -> tuple[np.ndarray, float, object]:
    X_aug = np.column_stack([X, np.ones(len(X))])
    y = y.astype(float)

    def objective(w: np.ndarray) -> float:
        z = X_aug @ w
        p = expit(z)
        eps = 1e-12
        loss = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)).mean()
        reg = 0.5 * l2 * np.sum(w[:-1] ** 2) / len(y)
        return float(loss + reg)

    def gradient(w: np.ndarray) -> np.ndarray:
        z = X_aug @ w
        p = expit(z)
        grad = (X_aug.T @ (p - y)) / len(y)
        grad[:-1] += l2 * w[:-1] / len(y)
        return grad

    init = np.zeros(X_aug.shape[1], dtype=float)
    result = minimize(objective, init, jac=gradient, method="L-BFGS-B", options={"maxiter": max_iter})
    return result.x[:-1], float(result.x[-1]), result


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


def maybe_load_previous_baseline(previous_path: Path) -> dict | None:
    if not previous_path.exists():
        return None
    with previous_path.open() as f:
        previous = json.load(f)
    if "metrics" not in previous:
        return None
    return previous


def summarize_comparison(previous_metrics: dict[str, dict], current_metrics: dict[str, dict]) -> list[str]:
    comparisons: list[str] = []
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
            (
                f"{model_label}: F1 {previous['f1']:.4f} -> {current['f1']:.4f}, "
                f"PR AUC {previous['pr_auc']:.4f} -> {current['pr_auc']:.4f}. {verdict}"
            )
        )
    return comparisons


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    fraud_count, selected_non_fraud_count = scan_selected_type_counts(DATA_PATH)
    target_negative_samples = NEG_POS_RATIO * fraud_count
    negative_sample_prob = min(1.0, target_negative_samples / selected_non_fraud_count)

    sample_df = build_sample_dataframe(DATA_PATH, negative_sample_prob=negative_sample_prob)
    sample_df = add_features(sample_df)

    train_full_df, test_df = train_test_split(
        sample_df,
        test_size=TEST_SIZE,
        stratify=sample_df["isFraud"],
        random_state=RANDOM_STATE,
    )
    train_df, val_df = train_test_split(
        train_full_df,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        stratify=train_full_df["isFraud"],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_BASE_FEATURES])

    train_df = add_scaled_feature_columns(train_df, scaler)
    val_df = add_scaled_feature_columns(val_df, scaler)
    test_df = add_scaled_feature_columns(test_df, scaler)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["isFraud"].to_numpy(dtype=int)
    X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_val = val_df["isFraud"].to_numpy(dtype=int)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df["isFraud"].to_numpy(dtype=int)

    ridge_w, ridge_b = fit_ridge_score_model(X_train, 2 * y_train - 1, alpha=1.0)
    ridge_val_score = X_val @ ridge_w + ridge_b
    ridge_test_score = X_test @ ridge_w + ridge_b
    ridge_val_pred = (ridge_val_score >= 0.0).astype(int)
    ridge_test_pred = (ridge_test_score >= 0.0).astype(int)

    logreg_w, logreg_b, logreg_result = fit_logistic_regression(X_train, y_train, l2=1.0, max_iter=500)
    logreg_val_logit = X_val @ logreg_w + logreg_b
    logreg_test_logit = X_test @ logreg_w + logreg_b
    logreg_val_prob = expit(logreg_val_logit)
    logreg_test_prob = expit(logreg_test_logit)
    logreg_val_pred = (logreg_val_prob >= 0.5).astype(int)
    logreg_test_pred = (logreg_test_prob >= 0.5).astype(int)

    metrics = [
        evaluate_split("validation", "ridge_score", y_val, ridge_val_pred, ridge_val_score),
        evaluate_split("test", "ridge_score", y_test, ridge_test_pred, ridge_test_score),
        evaluate_split("validation", "logistic_regression", y_val, logreg_val_pred, logreg_val_prob),
        evaluate_split("test", "logistic_regression", y_test, logreg_test_pred, logreg_test_prob),
    ]

    feature_order = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_base_features": NUMERIC_BASE_FEATURES,
        "binary_features": BINARY_FEATURES,
    }

    metrics_by_key = {(item["model"], item["split"]): item for item in metrics}

    previous_path = ARTIFACT_DIR / "baseline_model_artifacts.json"
    previous_summary = maybe_load_previous_baseline(previous_path)
    comparison_summary: list[str] | str
    if previous_summary is None:
        comparison_summary = "Previous 8D artifacts were not available for comparison."
    else:
        previous_metrics_by_key = {}
        for item in previous_summary.get("metrics", []):
            model_name = item.get("model", "")
            if model_name == "ridge_score_val":
                previous_metrics_by_key[("ridge_score", "validation")] = item
            elif model_name == "ridge_score_test":
                previous_metrics_by_key[("ridge_score", "test")] = item
            elif model_name == "logreg_val":
                previous_metrics_by_key[("logistic_regression", "validation")] = item
            elif model_name == "logreg_test":
                previous_metrics_by_key[("logistic_regression", "test")] = item
        comparison_summary = summarize_comparison(previous_metrics_by_key, metrics_by_key)
        if not comparison_summary:
            comparison_summary = "Previous 8D metrics file was present, but it did not contain comparable entries."

    he_eval_df = choose_he_eval_subset(test_df)
    he_eval_scaled_features_df = he_eval_df[FEATURE_COLUMNS].copy()
    he_eval_X = he_eval_scaled_features_df.to_numpy(dtype=float)
    he_eval_scores_df = pd.DataFrame(
        {
            "sample_row_id": he_eval_df["sample_row_id"],
            "isFraud": he_eval_df["isFraud"],
            "ridge_raw_score": he_eval_X @ ridge_w + ridge_b,
            "logistic_raw_logit": he_eval_X @ logreg_w + logreg_b,
            "logistic_probability": expit(he_eval_X @ logreg_w + logreg_b),
        }
    )
    he_eval_meta_df = he_eval_df[["sample_row_id", "step", "type", "isFraud"]].copy()

    he_eval_scaled_features_df.to_csv(HE_FEATURES_PATH, index=False)
    he_eval_meta_df.to_csv(HE_META_PATH, index=False)
    he_eval_scores_df.to_csv(HE_SCORES_PATH, index=False)

    summary = {
        "feature_order": feature_order,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "ridge_score": {
            "weights": ridge_w.tolist(),
            "bias": float(ridge_b),
        },
        "logistic_regression": {
            "weights": logreg_w.tolist(),
            "bias": float(logreg_b),
            "optimizer_success": bool(logreg_result.success),
            "optimizer_message": str(logreg_result.message),
        },
        "sample_summary": {
            "rows": int(len(sample_df)),
            "fraud": int(sample_df["isFraud"].sum()),
            "non_fraud": int((sample_df["isFraud"] == 0).sum()),
            "types": {str(key): int(value) for key, value in sample_df["type"].value_counts().to_dict().items()},
            "negative_to_positive_ratio": float((sample_df["isFraud"] == 0).sum() / max(sample_df["isFraud"].sum(), 1)),
        },
        "split_summary": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_fraud": int(train_df["isFraud"].sum()),
            "validation_fraud": int(val_df["isFraud"].sum()),
            "test_fraud": int(test_df["isFraud"].sum()),
        },
        "metrics": metrics,
        "he_eval_subset": {
            "rows": int(len(he_eval_df)),
            "fraud": int(he_eval_df["isFraud"].sum()),
            "non_fraud": int((he_eval_df["isFraud"] == 0).sum()),
            "feature_csv": str(HE_FEATURES_PATH),
            "meta_csv": str(HE_META_PATH),
            "plaintext_scores_csv": str(HE_SCORES_PATH),
        },
        "comparison_to_previous_8d": comparison_summary,
    }

    FEATURE_ORDER_JSON_PATH.write_text(json.dumps(feature_order, indent=2))
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2))

    with SCALER_PICKLE_PATH.open("wb") as f:
        pickle.dump(scaler, f)
    with RIDGE_PICKLE_PATH.open("wb") as f:
        pickle.dump({"weights": ridge_w, "bias": ridge_b}, f)
    with LOGREG_PICKLE_PATH.open("wb") as f:
        pickle.dump({"weights": logreg_w, "bias": logreg_b}, f)

    print("6D baseline complete")
    print(f"Sample rows: {len(sample_df):,} | fraud: {int(sample_df['isFraud'].sum()):,} | non-fraud: {int((sample_df['isFraud'] == 0).sum()):,}")
    print(f"Feature order: {FEATURE_COLUMNS}")
    for metric in metrics:
        print(
            f"{metric['split']} {metric['model']}: "
            f"precision={metric['precision']:.4f} "
            f"recall={metric['recall']:.4f} "
            f"f1={metric['f1']:.4f} "
            f"pr_auc={metric['pr_auc']:.4f} "
            f"predicted_positive={metric['predicted_positive']}"
        )
    print("Exported files:")
    for path in [
        SUMMARY_JSON_PATH,
        FEATURE_ORDER_JSON_PATH,
        SCALER_PICKLE_PATH,
        RIDGE_PICKLE_PATH,
        LOGREG_PICKLE_PATH,
        HE_FEATURES_PATH,
        HE_META_PATH,
        HE_SCORES_PATH,
    ]:
        print(f"- {path}")
    print("Suitability for HE next step: yes, the compact 6D feature vector keeps only scaled numeric inputs plus one binary indicator.")


if __name__ == "__main__":
    main()
