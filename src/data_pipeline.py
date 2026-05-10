import os
import pandas as pd
import numpy as np

COLS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
# Flat sensors that carry no degradation signal
FLAT_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}
USEFUL_SENSORS = [s for s in [f"s{i}" for i in range(1, 22)] if s not in FLAT_SENSORS]
RUL_CAP = 125


def load_cmapss(dataset: str = "FD001", data_dir: str = "data/nasa-dataset") -> dict:
    """Load and preprocess NASA C-MAPSS dataset. Returns dict with train/test DataFrames."""
    train_path = os.path.join(data_dir, f"train_{dataset}.txt")
    test_path  = os.path.join(data_dir, f"test_{dataset}.txt")
    rul_path   = os.path.join(data_dir, f"RUL_{dataset}.txt")

    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLS)
    test  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=COLS)
    rul   = pd.read_csv(rul_path, header=None, names=["final_rul"])

    # Compute piecewise-capped RUL for training set
    max_cycles = train.groupby("unit")["cycle"].transform("max")
    train["actual_rul"] = (max_cycles - train["cycle"]).clip(upper=RUL_CAP)

    # Normalize useful sensors using train statistics
    for col in USEFUL_SENSORS:
        mu, sigma = train[col].mean(), train[col].std()
        if sigma > 1e-6:
            train[col] = (train[col] - mu) / sigma
            test[col]  = (test[col]  - mu) / sigma

    # Attach final RUL to test set's last cycle
    rul["unit"] = range(1, len(rul) + 1)
    test_last = test.groupby("unit").tail(1).copy()
    test_last = test_last.merge(rul, on="unit")
    test_last["actual_rul"] = test_last["final_rul"].clip(upper=RUL_CAP)

    return {
        "train": train,
        "test": test,
        "test_last": test_last,
        "rul_ground_truth": rul,
        "useful_sensors": USEFUL_SENSORS,
    }
