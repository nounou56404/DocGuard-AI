"""Contrat d'API commun aux trois approches (gele apres le jour 1)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OcrBox(BaseModel):
    text: str
    box: list[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")
    suspect: bool = False


class PredictResponse(BaseModel):
    label: Literal["scam", "non_scam"]
    score: float = Field(..., ge=0.0, le=1.0, description="probabilite de la classe scam")
    model: str = Field(..., examples=["scam-cnn:v1"])
    heatmap: str | None = Field(None, description="PNG en base64 (Grad-CAM / attention)")
    ocr: list[OcrBox] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ModelInfo(BaseModel):
    approach: str
    name: str
    version: str
    input_size: list[int]
    data_version: str
