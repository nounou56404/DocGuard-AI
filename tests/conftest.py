"""Fixture : un mini-projet temporaire (configs + images synthetiques) pointe par DOCGUARD_ROOT."""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from docguard_ai.data.ingestion import build_manifest

REPO = Path(__file__).resolve().parents[1]


def _fake_doc(rng, scam: bool) -> np.ndarray:
    img = np.full((120, 160, 3), 255, np.uint8)
    for i in range(4 + 2 * scam):
        y = 15 + 18 * i
        cv2.line(img, (10, y), (int(60 + rng.randint(0, 90)), y), (0, 0, 0), 2)
    return img


@pytest.fixture()
def mini_project(tmp_path, monkeypatch):
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    shutil.copy(REPO / "params.yaml", tmp_path / "params.yaml")
    rng = np.random.RandomState(0)
    for label in ("scam", "non_scam"):
        d = tmp_path / "data" / "raw" / label
        d.mkdir(parents=True)
        for i in range(10):
            cv2.imwrite(str(d / f"doc{i:02d}.png"), _fake_doc(rng, label == "scam"))
    meta = tmp_path / "data" / "metadata"
    meta.mkdir(parents=True)
    build_manifest(tmp_path / "data" / "raw").to_csv(meta / "image_manifest.csv", index=False)
    monkeypatch.setenv("DOCGUARD_ROOT", str(tmp_path))
    return tmp_path
