import pandas as pd

from docguard_ai.data.audit import split_leakage
from docguard_ai.data.split import assign_splits


def test_no_group_leakage_and_deterministic():
    df = pd.DataFrame({"sample_id": range(40), "group_id": [f"g{i // 2}" for i in range(40)]})
    a = assign_splits(df, "group_id", 0.15, 0.15, seed=42)
    b = assign_splits(df, "group_id", 0.15, 0.15, seed=42)
    assert not split_leakage(a) and a["split"].equals(b["split"])
    assert set(a["split"]) == {"train", "val", "test"}
