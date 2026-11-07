"""Schemas Pydantic communs (contrat gele apres le jour 1)."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class OcrBox(BaseModel):
    text: str
    box: List[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")
    suspect: bool = False


class PredictResponse(BaseModel):
    label: Literal["scam", "non_scam"]
    score: float = Field(..., ge=0.0, le=1.0)
    model: str = Field(..., examples=["scam-cnn:v3"])
    heatmap: Optional[str] = Field(None, description="PNG en base64")
    ocr: Optional[List[OcrBox]] = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ModelInfo(BaseModel):
    name: str
    version: str
    input_size: List[int]
    data_version: str
