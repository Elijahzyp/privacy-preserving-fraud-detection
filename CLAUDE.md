# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Privacy-preserving fraud detection prototype using the PaySim synthetic financial transaction dataset. Implements baseline ML models designed for homomorphic encryption (HE) inference using the CKKS scheme (TenSEAL).

## Environment

Uses the conda environment `prompt_engineering` (Python 3.x, located at `/opt/anaconda3/envs/prompt_engineering`). This is the only environment with all required dependencies: pandas, numpy, scipy, scikit-learn, and tenseal.

```bash
conda activate prompt_engineering
```

Do **not** attempt to migrate to venv — TenSEAL installation outside conda is unreliable on this machine.

## Running the Notebooks

There is no build system. The project runs entirely in Jupyter:

```bash
# Launch Jupyter (after activating prompt_engineering)
jupyter notebook

# Or run notebooks headlessly (kernel_name flag is required — base Python won't have TenSEAL)
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=prompt_engineering fraud_baseline_6d.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=prompt_engineering he_inference_demo.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=prompt_engineering visualization.ipynb

# Standalone pipeline script (equivalent to training notebook, pre-HE)
python archive/baseline_6d_pipeline.py
```

## Notebook Execution Order

1. **`fraud_baseline_6d.ipynb`** — must run first; trains all models, runs ratio sweep + grid search + non-HE baselines, writes all artifacts to `artifacts/`
2. **`he_inference_demo.ipynb`** — loads artifacts; runs encrypted inference demo and exports HE timing results (no retraining)
3. **`visualization.ipynb`** — loads all artifacts (no training); generates poster-quality figures to `artifacts/figures/`

## Architecture

### Data Pipeline

```
Raw PaySim CSV (6.8M rows)
  → Filter: TRANSFER & CASH_OUT only (2.77M rows)
  → Stratified downsample: 10:1 non-fraud:fraud ratio (~90K rows)
  → Feature engineering: deltaOrig, deltaDest, is_transfer
  → Train/Val/Test split: 57,598 / 14,400 / 18,000
  → StandardScaler fit on training set → 6D scaled features
  → 4 HE-compatible models (Ridge, LogReg, LinearSVC, Poly2LogReg) + tuned variants
  → 2 non-HE upper-bound baselines (RandomForest, GradientBoosting)
  → Ratio sweep (NEG_POS_RATIO ∈ [5,10,20,30]) and hyperparameter grid search
  → Artifacts exported to artifacts/
```

### Feature Space

**6D base features (HE-friendly):** `amount_scaled`, `oldbalanceOrg_scaled`, `oldbalanceDest_scaled`, `deltaOrig_scaled`, `deltaDest_scaled`, `is_transfer`

**27D polynomial features:** `PolynomialFeatures(degree=2, include_bias=False)` expansion of the 6D base; pre-computed in plaintext before encryption.

### HE Inference Strategy

- **Linear models (Ridge, LogReg, LinearSVC):** encrypt 6D features once, evaluate weighted sums in encrypted CKKS space
- **Poly2LogReg:** expand 6D→27D in plaintext first, then encrypt and score — avoids expensive encrypted polynomial construction
- Evaluation subset: 128 test samples (32 fraud, 96 non-fraud) in `artifacts/he_eval_*.csv`

### Key Constants (`src/config.py`)

```python
RANDOM_STATE = 42
NEG_POS_RATIO = 10        # 10:1 non-fraud to fraud in training set
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20
SELECTED_TYPES = ["TRANSFER", "CASH_OUT"]
CHUNK_SIZE = 250_000      # CSV loading chunk size
```

### Source Modules (`src/`)

The project logic is extracted into importable Python modules (used by all notebooks):
- `src/config.py` — all shared constants and feature column lists
- `src/data.py` — CSV loading, sampling, feature engineering, scaling
- `src/models.py` — custom Ridge and LogReg fitting functions
- `src/evaluation.py` — `evaluate_split()`, HE eval subset selection
- `src/he_utils.py` — CKKS context builder, encrypted scoring, batch demo runner
- `src/visualization.py` — all matplotlib/seaborn plotting functions

### Artifacts

All outputs land in `artifacts/`:
- `*.pkl` — serialized model objects and scaler (load with `pickle`)
- `baseline_6d_summary.json` — full per-split metrics for base 4 models
- `baseline_model_comparison_summary.json` — compact test-set comparison
- `extended_model_comparison.json` — all models including tuned + non-HE baselines ← use this for dashboard
- `grid_search_results.json` — best hyperparams per model from CV/val sweep
- `ratio_sweep_results.json` — F1/PR-AUC vs NEG_POS_RATIO for LogReg + Poly2LogReg
- `he_timing_results.json` — per-model encrypt/infer/decrypt times across 3 CKKS configs
- `test_scores_6d.csv` — raw score columns for all models on test set (for PR curves)
- `test_labels.csv` — ground-truth labels for test set
- `feature_order_6d.json` — feature name ordering (required for correct inference)
- `he_eval_*.csv` — 128-sample HE evaluation subset with precomputed plaintext scores
- `figures/` — poster-quality PNGs at 300 DPI: `pr_curves.png`, `confusion_matrices.png`, `model_comparison_bar.png`, `ratio_sweep.png`, `he_timing_breakdown.png`, `he_vs_plaintext_overhead.png`, `he_error_scatter.png`

### Current Best Performance (Test Set)

| Model | HE-compatible | F1 | PR-AUC |
|-------|--------------|-----|--------|
| random_forest | No | 0.9566 | 0.9930 |
| gradient_boosting | No | 0.8822 | 0.9731 |
| poly2_logreg_tuned (C=10) | **Yes** | **0.8782** | **0.9638** |
| logreg_tuned (l2=0.01) | Yes | 0.7999 | 0.9042 |
| linear_svc_tuned (C=10) | Yes | 0.7438 | 0.8671 |
| ridge_tuned (alpha=0.01) | Yes | 0.3140 | 0.7335 |

`poly2_logreg_tuned` is the recommended HE inference model.

### CKKS Parameter Configurations

Three configs are swept in `he_inference_demo.ipynb`:
- `fast_8192`: poly_modulus_degree=8192 — fastest, sufficient precision (<1ms avg error)
- `balanced_16384`: poly_modulus_degree=16384 — balanced
- `precise_16384`: poly_modulus_degree=16384 — highest precision

### Dashboard Integration (TODO — assigned to teammate)

The dashboard should load artifacts directly — no retraining needed:
1. Load `artifacts/scaler_6d.pkl` and `artifacts/poly2_logreg_tuned.pkl` (or whichever model)
2. Load `artifacts/feature_order_6d.json` to ensure correct feature ordering
3. For plaintext inference: apply scaler → compute score with model weights
4. For HE inference demo: use `src/he_utils.py` → `build_ckks_context()` + `encrypted_linear_score()`
   - Recommend `fast_8192` config for interactive demo (fastest, ~1ms avg decryption error)
5. Reference `artifacts/extended_model_comparison.json` for displaying benchmark results
6. Reference `artifacts/figures/` for pre-generated poster charts

### Previous Baseline

`archive/baseline_model_artifacts_8d.json` contains the prior 8D feature baseline for comparison. The 6D baseline achieves comparable performance (Poly2LogReg: F1=0.8597, PR-AUC=0.9336).
