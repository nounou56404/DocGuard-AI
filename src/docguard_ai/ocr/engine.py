"""Moteur OCR (personne C). Doit renvoyer des OcrBox du contrat.

A implementer avec pytesseract ou easyocr (params.yaml -> ocr.engine).
"""
from __future__ import annotations

from docguard_ai.contract.schemas import OcrBox


def flag_suspect(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def run_ocr(image_bgr, params: dict) -> list[OcrBox]:
    raise NotImplementedError("brancher pytesseract / easyocr ; utiliser vision.preprocessing."
                              "preprocess_for_ocr puis flag_suspect")
