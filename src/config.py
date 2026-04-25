from __future__ import annotations

RANDOM_STATE = 42
CHUNK_SIZE = 250_000
NEG_POS_RATIO = 10
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20
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
