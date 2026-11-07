"""Controles qualite : desequilibre, fichiers manquants/illisibles, doublons, fuite entre splits."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import pandas as pd

from docguard_ai.config import resolve


def audit_manifest(df: pd.DataFrame, check_images: bool = True) -> dict:
    counts = df["label"].value_counts().to_dict()
    report: dict = {"n_samples": len(df), "class_counts": counts}
    if len(counts) == 2:
        report["imbalance_ratio"] = round(max(counts.values()) / min(counts.values()), 3)
    if check_images:
        missing, unreadable, hashes = [], [], {}
        for sid, path in zip(df["sample_id"], df["image_path"]):
            p = Path(resolve(path))
            if not p.exists():
                missing.append(sid)
                continue
            if cv2.imread(str(p)) is None:
                unreadable.append(sid)
                continue
            hashes.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append(sid)
        dups = [v for v in hashes.values() if len(v) > 1]
        report.update(missing_files=missing, unreadable_files=unreadable,
                      duplicate_groups=dups)
    report["pii_check"] = "a faire manuellement : retirer toute donnee personnelle"
    return report


def split_leakage(df: pd.DataFrame, group_col: str = "group_id") -> list[str]:
    """Groupes presents dans plusieurs splits (doit etre vide)."""
    n = df.groupby(group_col)["split"].nunique()
    return n[n > 1].index.tolist()
