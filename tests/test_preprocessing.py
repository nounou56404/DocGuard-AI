import numpy as np

from docguard_ai.vision.preprocessing import preprocess_for_ocr, preprocess_image, to_model_input


def test_preprocess_image_shape_and_dtype():
    img = np.random.RandomState(0).randint(0, 255, (80, 120, 3), dtype=np.uint8)
    out = preprocess_image(img, 64)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8


def test_preprocess_for_ocr_is_binary():
    img = np.full((60, 90, 3), 200, np.uint8)
    out = preprocess_for_ocr(img)
    assert out.ndim == 2 and set(np.unique(out)) <= {0, 255}


def test_model_input_layout():
    x = to_model_input(np.zeros((32, 32, 3), np.uint8), [0.5] * 3, [0.5] * 3)
    assert x.shape == (1, 3, 32, 32) and x.dtype == np.float32
