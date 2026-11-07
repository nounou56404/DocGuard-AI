"""Registre des modeles ONNX disponibles (charges a la demande)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from deployment.inference import OnnxModel

_cache: dict[str, OnnxModel] = {}


def models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", "models"))


def available() -> list[dict]:
    cards = []
    for card in sorted(models_dir().glob("*/model_card.json")):
        if (card.parent / "model.onnx").exists():
            cards.append(json.loads(card.read_text()))
    return cards


def get_model(approach: str) -> OnnxModel | None:
    d = models_dir() / approach
    if not (d / "model.onnx").exists():
        return None
    if approach not in _cache:
        _cache[approach] = OnnxModel(d)
    return _cache[approach]
