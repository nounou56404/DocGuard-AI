"""Pretraitement OpenCV, identique a l'entrainement et a l'inference."""
from __future__ import annotations

import cv2
import numpy as np


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Angle d'inclinaison (degres) estime sur les pixels sombres (texte)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    return float(angle) if abs(angle) <= 15 else 0.0


def deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.3:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21) if img.ndim == 3 \
        else cv2.fastNlMeansDenoising(img, None, 5, 7, 21)


def resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Redimensionne en conservant le ratio, puis complete en blanc jusqu'a size x size."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size) + img.shape[2:], 255, dtype=img.dtype)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def preprocess_image(img: np.ndarray, size: int = 224, do_deskew: bool = True,
                     do_denoise: bool = True) -> np.ndarray:
    """Image BGR uint8 -> image BGR uint8 size x size (pour CNN / ViT)."""
    if do_deskew:
        img = deskew(img)
    if do_denoise:
        img = denoise(img)
    return resize_pad(img, size)


def preprocess_for_ocr(img: np.ndarray, binarize: bool = True, clahe: bool = True) -> np.ndarray:
    """Image BGR -> niveaux de gris, contraste (CLAHE), binarisation, redressement."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = deskew(gray)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if binarize:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)
    return gray


def to_model_input(img_bgr: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """BGR uint8 HxWx3 -> float32 1x3xHxW normalise (RGB), pret pour ONNX Runtime."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
