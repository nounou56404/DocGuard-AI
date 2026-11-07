"""Baselines que toute experience doit battre : classe majoritaire et tirage aleatoire."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from docguard_ai.config import get_config, get_params, resolve
from docguard_ai.models.evaluate import compute_metrics


def run() -> dict:
    cfg, params = get_config(), get_params()
    df = pd.read_csv(resolve(cfg["outputs"]["manifest"]))
    lm = params["prepare"]["label_map"]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    y_test = test["label"].map(lm).to_numpy()
    majority = int(train["label"].map(lm).mode().iloc[0])
    rng = np.random.RandomState(params["seed"])
    thr = params["evaluate"]["threshold"]
    out = {
        "majority": compute_metrics(y_test, np.full(len(y_test), float(majority)), thr),
        "random": compute_metrics(y_test, rng.rand(len(y_test)), thr),
    }
    path = resolve(cfg["outputs"]["baselines"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out
