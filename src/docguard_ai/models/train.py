"""Entrainement : delegue au module de l'approche (cnn.py, vit.py, ocr_text.py)."""
from __future__ import annotations

from docguard_ai.config import get_config, get_params
from docguard_ai.models._dispatch import load


def run(approach: str) -> None:
    load(approach).train(get_params(), get_config())
