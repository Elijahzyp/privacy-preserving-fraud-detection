from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def weighted_standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    return (X - mean) / scale, mean, scale


def fit_ridge_score_model(
    X: np.ndarray,
    y_pm1: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
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
    result = minimize(
        objective,
        init,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    return result.x[:-1], float(result.x[-1]), result
