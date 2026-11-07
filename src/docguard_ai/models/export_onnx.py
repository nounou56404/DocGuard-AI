"""Export ONNX : delegue au module de l'approche, avec verification d'equivalence et model card."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from docguard_ai.config import get_config, get_params
from docguard_ai.models._dispatch import load


def check_equivalence(reference: np.ndarray, onnx_out: np.ndarray, atol: float) -> None:
    """Les sorties ONNX doivent correspondre a celles du modele d'origine."""
    if not np.allclose(reference, onnx_out, atol=atol):
        raise AssertionError(f"ecart ONNX max = {np.abs(reference - onnx_out).max():.2e} > {atol}")


def write_model_card(path: str | Path, *, name: str, version: str, approach: str,
                     input_size: list[int], mean: list[float], std: list[float],
                     data_version: str) -> None:
    card = {"name": name, "version": version, "approach": approach, "input_size": input_size,
            "mean": mean, "std": std, "labels": ["non_scam", "scam"],
            "data_version": data_version}
    Path(path).write_text(json.dumps(card, indent=2))


def run(approach: str) -> None:
    load(approach).export(get_params(), get_config())
