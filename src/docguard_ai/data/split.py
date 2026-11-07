"""Decoupage train / val / test au niveau du groupe (pas de fuite entre splits)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_splits(df: pd.DataFrame, group_col: str, val_size: float, test_size: float,
                  seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    groups = np.array(sorted(df[group_col].unique()))
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, round(n * test_size)) if n >= 3 else 0
    n_val = max(1, round(n * val_size)) if n >= 3 else 0
    mapping = {}
    for i, g in enumerate(groups):
        mapping[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    out = df.copy()
    out["split"] = out[group_col].map(mapping)
    return out
