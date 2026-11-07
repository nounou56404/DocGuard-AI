"""Schema du manifeste d'images et validations."""
from __future__ import annotations

import pandas as pd

LABELS = ("non_scam", "scam")
SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = [
    "sample_id", "image_path", "label", "label_original", "source_dataset",
    "group_id", "split", "license", "width", "height",
]


def validate_manifest(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des erreurs (vide si le manifeste est valide)."""
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"colonnes manquantes : {missing}"]
    if df["sample_id"].duplicated().any():
        errors.append("sample_id non unique")
    bad = set(df["label"].dropna().unique()) - set(LABELS)
    if bad:
        errors.append(f"labels inconnus : {sorted(bad)}")
    if df["label"].isna().any():
        errors.append("labels manquants")
    splits = set(df["split"].dropna().unique()) - set(SPLITS) - {""}   # "" = a assigner (prepare)
    if splits:
        errors.append(f"splits inconnus : {sorted(splits)}")
    return errors
