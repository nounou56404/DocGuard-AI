"""Inference ONNX Runtime : charge models/<approche>/model.onnx + model_card.json."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from docguard_ai.vision.preprocessing import preprocess_image, to_model_input


class OnnxModel:
    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        self.card = json.loads((model_dir / "model_card.json").read_text())
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"),
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @property
    def tag(self) -> str:
        return f"{self.card['name']}:{self.card['version']}"

    def predict(self, image_bgr: np.ndarray) -> float:
        """Probabilite de la classe scam."""
        size = self.card["input_size"][0]
        x = to_model_input(preprocess_image(image_bgr, size), self.card["mean"], self.card["std"])
        out = self.session.run(None, {self.input_name: x})[0].reshape(-1)
        if out.size == 1:                       # sortie sigmoide / logit unique
            return float(1.0 / (1.0 + np.exp(-out[0])))
        e = np.exp(out - out.max())             # softmax sur [non_scam, scam]
        return float((e / e.sum())[1])


def decode_image(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
