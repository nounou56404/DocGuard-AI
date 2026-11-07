"""Enveloppe MLflow : sans effet si mlflow n'est pas installe.

Conventions : experiences scam-a / scam-b / scam-c ; run <approche>-<variante>-<MMJJ> ;
metriques accuracy, precision, recall, f1, auc, recall_scam ;
registry scam-cnn / scam-vit / scam-ocr (Staging puis Production).
"""
from __future__ import annotations

import contextlib
import datetime
import os


def make_run_name(approach: str, variant: str) -> str:
    return f"{approach}-{variant}-{datetime.date.today():%m%d}"


@contextlib.contextmanager
def start_run(approach: str, run_name: str, params: dict | None = None,
              tags: dict | None = None, cfg: dict | None = None):
    try:
        import mlflow
    except ImportError:
        yield None
        return
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    experiments = (cfg or {}).get("mlflow", {}).get("experiments", {})
    mlflow.set_experiment(experiments.get(approach, f"scam-{approach}"))
    with mlflow.start_run(run_name=run_name, tags=tags or {}):
        if params:
            mlflow.log_params(_flatten(params))
        yield mlflow


def log_metrics(mlflow, metrics: dict, prefix: str = "") -> None:
    if mlflow is None:
        return
    for k, v in _flatten(metrics, prefix).items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, float(v))


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
