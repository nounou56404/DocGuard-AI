"""Chargement des configurations (configs/*.yaml et params.yaml).

La racine du projet est le dossier courant, ou la variable d'environnement
DOCGUARD_ROOT si elle est definie. Les etapes DVC s'executent depuis la racine.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(os.environ.get("DOCGUARD_ROOT", Path.cwd())).resolve()


def _load(relative: str) -> dict:
    with open(project_root() / relative, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config() -> dict:
    return _load("configs/config.yaml")


def get_params() -> dict:
    return _load("params.yaml")


def resolve(path: str | Path) -> Path:
    """Chemin relatif au projet -> chemin absolu."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p
