"""Construction du manifeste a partir de data/raw/<label>/*.{png,jpg,...}."""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from docguard_ai.data.schema import LABELS

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_manifest(raw_dir: str | Path, source_dataset: str = "local",
                   license_: str = "unknown") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows = []
    for label in LABELS:
        for p in sorted((raw_dir / label).glob("*")):
            if p.suffix.lower() not in IMAGE_EXT:
                continue
            img = cv2.imread(str(p))
            h, w = (img.shape[:2] if img is not None else (0, 0))
            sid = f"{label}_{p.stem}"
            rows.append({
                "sample_id": sid, "image_path": str(p), "label": label, "label_original": label,
                "source_dataset": source_dataset, "group_id": sid, "split": "",
                "license": license_, "width": w, "height": h,
            })
    return pd.DataFrame(rows)
