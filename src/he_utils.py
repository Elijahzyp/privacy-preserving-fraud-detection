from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
    TENSEAL_IMPORT_ERROR = None
except Exception as exc:
    ts = None
    TENSEAL_AVAILABLE = False
    TENSEAL_IMPORT_ERROR = exc

CKKS_PARAMETER_OPTIONS = [
    {
        "name": "fast_8192",
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale_bits": 40,
    },
    {
        "name": "balanced_16384",
        "poly_modulus_degree": 16384,
        "coeff_mod_bit_sizes": [60, 40, 40, 40, 60],
        "global_scale_bits": 40,
    },
    {
        "name": "precise_16384",
        "poly_modulus_degree": 16384,
        "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 60],
        "global_scale_bits": 40,
    },
]


def build_ckks_context(config: dict):
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL is not installed.")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=config["poly_modulus_degree"],
        coeff_mod_bit_sizes=config["coeff_mod_bit_sizes"],
    )
    context.global_scale = 2 ** config["global_scale_bits"]
    context.generate_galois_keys()
    return context


def encrypted_linear_score(context, features: np.ndarray, weights: np.ndarray, bias: float):
    enc_vec = ts.ckks_vector(context, features.tolist())
    enc_score = enc_vec.dot(weights.tolist())
    enc_score = enc_score + bias
    return enc_vec, enc_score


def run_ckks_batch_demo(
    X: np.ndarray,
    plaintext_scores: np.ndarray,
    weights: np.ndarray,
    bias: float,
    config: dict,
):
    if not TENSEAL_AVAILABLE:
        return None, None

    context = build_ckks_context(config)

    encrypt_time = 0.0
    inference_time = 0.0
    decrypt_time = 0.0
    ciphertext_sizes = []
    records = []

    for idx, (features, plain_score) in enumerate(zip(X, plaintext_scores)):
        t0 = time.perf_counter()
        enc_vec = ts.ckks_vector(context, features.tolist())
        encrypt_time += time.perf_counter() - t0

        t1 = time.perf_counter()
        enc_score = enc_vec.dot(weights.tolist())
        enc_score = enc_score + bias
        inference_time += time.perf_counter() - t1

        t2 = time.perf_counter()
        decrypted_score = float(enc_score.decrypt()[0])
        decrypt_time += time.perf_counter() - t2

        ciphertext_sizes.append(len(enc_score.serialize()) if hasattr(enc_score, "serialize") else math.nan)
        records.append(
            {
                "sample_index": idx,
                "plaintext_score": float(plain_score),
                "decrypted_score": decrypted_score,
                "absolute_error": abs(float(plain_score) - decrypted_score),
            }
        )

    results_df = pd.DataFrame(records)
    timing = {
        "encrypt_time_sec": encrypt_time,
        "inference_time_sec": inference_time,
        "decrypt_time_sec": decrypt_time,
        "average_abs_error": float(results_df["absolute_error"].mean()),
        "max_abs_error": float(results_df["absolute_error"].max()),
        "ciphertext_size_bytes": float(np.nanmean(ciphertext_sizes)) if ciphertext_sizes else math.nan,
        "total_runtime_sec": encrypt_time + inference_time + decrypt_time,
    }
    return results_df, timing
