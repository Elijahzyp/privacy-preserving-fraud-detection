from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    CHUNK_SIZE,
    NUMERIC_BASE_FEATURES,
    NUMERIC_SCALED_FEATURES,
    RANDOM_STATE,
    SELECTED_TYPES,
)


def scan_selected_type_counts(
    data_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[int, int]:
    selected_fraud = 0
    selected_non_fraud = 0
    for chunk in pd.read_csv(data_path, usecols=["type", "isFraud"], chunksize=chunk_size):
        chunk = chunk[chunk["type"].isin(SELECTED_TYPES)]
        if chunk.empty:
            continue
        selected_fraud += int(chunk["isFraud"].sum())
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

    sampled_chunks = []
    for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=chunk_size):
        chunk = chunk[chunk["type"].isin(SELECTED_TYPES)].copy()
        if chunk.empty:
            continue

        fraud_df = chunk[chunk["isFraud"] == 1]
        non_fraud_df = chunk[chunk["isFraud"] == 0]
        sampled_non_fraud_df = non_fraud_df.loc[
            rng.random(len(non_fraud_df)) < negative_sample_prob
        ]
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
