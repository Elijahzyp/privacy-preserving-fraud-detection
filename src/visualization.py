from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

FIGURE_DIR_DEFAULT = Path("artifacts/figures")

_MODEL_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#009688"]
_HE_COLOR = "#2196F3"
_NON_HE_COLOR = "#FF5722"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def plot_pr_curves(
    model_scores: dict[str, tuple[np.ndarray, bool]],
    y_true: np.ndarray,
    title: str = "Precision-Recall Curves",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    model_scores: {model_name: (score_array, he_compatible)}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, (scores, he_compat)) in enumerate(model_scores.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        auc = average_precision_score(y_true, scores)
        linestyle = "-" if he_compat else "--"
        suffix = "" if he_compat else " [non-HE]"
        ax.plot(recall, precision, linestyle=linestyle,
                color=_MODEL_COLORS[i % len(_MODEL_COLORS)], linewidth=2,
                label=f"{name}{suffix} (AUC={auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrices(
    model_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    n = len(model_preds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, preds) in zip(axes, model_preds.items()):
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Non-Fraud", "Fraud"],
                    yticklabels=["Non-Fraud", "Fraud"])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_model_comparison_bar(
    metrics_df: pd.DataFrame,
    he_compat_map: dict[str, bool],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    metrics_df: must have columns [model, f1, pr_auc], filtered to test split already.
    he_compat_map: {model_name: bool}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["f1", "pr_auc"]):
        models = metrics_df["model"].tolist()
        values = metrics_df[metric].tolist()
        colors = [_HE_COLOR if he_compat_map.get(m, True) else _NON_HE_COLOR for m in models]
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric.replace("_", "-").upper())
        ax.set_title(f"{metric.replace('_', '-').upper()} by Model")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    legend_elements = [
        mpatches.Patch(facecolor=_HE_COLOR, label="HE-compatible"),
        mpatches.Patch(facecolor=_NON_HE_COLOR, label="Non-HE baseline"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right")
    fig.suptitle("Model Performance Comparison (Test Set)", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_ratio_sweep(
    ratio_results: list[dict],
    save_path: Path | None = None,
) -> plt.Figure:
    """ratio_results: list of dicts with keys ratio, model, f1, pr_auc"""
    df = pd.DataFrame(ratio_results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["f1", "pr_auc"]):
        for i, model_name in enumerate(sorted(df["model"].unique())):
            sub = df[df["model"] == model_name].sort_values("ratio")
            ax.plot(sub["ratio"], sub[metric], marker="o", linewidth=2,
                    color=_MODEL_COLORS[i % len(_MODEL_COLORS)], label=model_name)
        ax.set_xlabel("NEG_POS_RATIO (non-fraud : fraud)")
        ax.set_ylabel(metric.replace("_", "-").upper())
        ax.set_title(f"{metric.replace('_', '-').upper()} vs Sampling Ratio")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Effect of Non-Fraud Sampling Ratio on Model Performance", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_he_timing_breakdown(
    timing_df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    timing_df: columns [model_name, config, encrypt_time_sec, inference_time_sec, decrypt_time_sec]
    """
    configs = timing_df["config"].unique()
    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5))
    if len(configs) == 1:
        axes = [axes]
    for ax, config in zip(axes, configs):
        sub = timing_df[timing_df["config"] == config].reset_index(drop=True)
        models = sub["model_name"].tolist()
        enc = sub["encrypt_time_sec"].tolist()
        inf = sub["inference_time_sec"].tolist()
        dec = sub["decrypt_time_sec"].tolist()
        x = range(len(models))
        ax.bar(x, enc, label="Encrypt", color="#2196F3", alpha=0.85)
        ax.bar(x, inf, bottom=enc, label="Infer", color="#4CAF50", alpha=0.85)
        bottom2 = [e + i for e, i in zip(enc, inf)]
        ax.bar(x, dec, bottom=bottom2, label="Decrypt", color="#FF9800", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Total Time (s, 128 samples)")
        ax.set_title(f"CKKS Config: {config}")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("HE Encrypted Inference Timing Breakdown", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_he_vs_plaintext_overhead(
    he_timing_df: pd.DataFrame,
    plaintext_times: dict[str, float],
    config: str = "fast_8192",
    n_samples: int = 128,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    he_timing_df: from he_timing_results.json, columns include model_name, config, total_runtime_sec
    plaintext_times: {model_name: total_time_sec for n_samples samples}
    Shows HE vs plaintext total time per sample, labeled with overhead factor.
    """
    sub = he_timing_df[he_timing_df["config"] == config].copy()
    models = sub["model_name"].tolist()

    he_per_sample    = [sub.loc[sub["model_name"] == m, "total_runtime_sec"].values[0] / n_samples * 1000
                        for m in models]
    plain_per_sample = [plaintext_times.get(m, float("nan")) / n_samples * 1000
                        for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_plain = ax.bar(x - width / 2, plain_per_sample, width,
                        label="Plaintext", color="#4CAF50", alpha=0.85)
    bars_he    = ax.bar(x + width / 2, he_per_sample,    width,
                        label=f"HE (CKKS {config})", color="#2196F3", alpha=0.85)

    # Overhead factor labels above HE bars
    for bar, he_t, pt in zip(bars_he, he_per_sample, plain_per_sample):
        if pt > 0 and not np.isnan(pt):
            factor = he_t / pt
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(he_per_sample) * 0.02,
                    f"×{factor:.0f}", ha="center", va="bottom", fontsize=9, color="#1565C0")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Time per sample (ms)")
    ax.set_title(f"Plaintext vs HE Encrypted Inference — Per-Sample Latency\n"
                 f"(CKKS config: {config}, {n_samples} samples total)")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_he_error_scatter(
    he_scores_df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    he_scores_df: columns [model_name, plaintext_score, decrypted_score]
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (model_name, sub) in enumerate(he_scores_df.groupby("model_name")):
        ax.scatter(sub["plaintext_score"], sub["decrypted_score"],
                   label=model_name, alpha=0.55, s=18,
                   color=_MODEL_COLORS[i % len(_MODEL_COLORS)])
    all_vals = pd.concat([he_scores_df["plaintext_score"], he_scores_df["decrypted_score"]])
    mn, mx = all_vals.min(), all_vals.max()
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="y = x (perfect)")
    ax.set_xlabel("Plaintext Score")
    ax.set_ylabel("Decrypted Score")
    ax.set_title("Plaintext vs Encrypted Inference Scores")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
