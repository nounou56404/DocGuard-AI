"""Evaluation commune : memes metriques, meme jeu de test pour les trois approches.

Contrat : l'etape train ecrit models/<approche>/predictions_test.csv
avec les colonnes sample_id, y_true (0/1), score (probabilite de scam).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from docguard_ai.config import get_config, get_params, resolve
from docguard_ai.tracking import mlflow_utils


def compute_metrics(y_true, score, threshold: float = 0.5) -> dict:
    y_true, score = np.asarray(y_true), np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    rec = float(recall_score(y_true, pred, zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": rec,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, score)) if both else None,
        "recall_scam": rec,
    }


def run(approach: str) -> dict:
    cfg, params = get_config(), get_params()
    preds = pd.read_csv(resolve(f"models/{approach}/predictions_test.csv"))
    metrics = compute_metrics(preds["y_true"], preds["score"], params["evaluate"]["threshold"])
    out = resolve(f"reports/{approach}/metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    with mlflow_utils.start_run(approach, f"{approach}-evaluate", cfg=cfg) as mlf:
        mlflow_utils.log_metrics(mlf, {k: v for k, v in metrics.items() if v is not None})
    return metrics
