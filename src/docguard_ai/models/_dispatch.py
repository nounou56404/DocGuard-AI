from __future__ import annotations

import importlib

MODULES = {"cnn": "docguard_ai.models.cnn", "vit": "docguard_ai.models.vit",
           "ocr": "docguard_ai.models.ocr_text"}


def load(approach: str):
    if approach not in MODULES:
        raise ValueError(f"approche inconnue : {approach} (attendu : {sorted(MODULES)})")
    return importlib.import_module(MODULES[approach])
