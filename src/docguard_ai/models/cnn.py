"""Personne A - CNN avec transfer learning et fine-tuning multi-echelles (Grad-CAM).

Contrat a respecter (voir docs/architecture.md) :
- train(params, config) : runs MLflow (tracking.mlflow_utils), ecrit models/<approche>/checkpoint
  et models/<approche>/predictions_test.csv (sample_id, y_true, score) ;
- export(params, config) : ecrit models/<approche>/model.onnx + model_card.json
  apres verification d'equivalence (models.export_onnx.check_equivalence).
"""
from __future__ import annotations


def train(params: dict, config: dict) -> None:
    raise NotImplementedError("A implementer par la personne A")


def export(params: dict, config: dict) -> None:
    raise NotImplementedError("A implementer par la personne A")
