import pytest
from pydantic import ValidationError

from docguard_ai.contract.schemas import OcrBox, PredictResponse


def test_predict_response_ok():
    r = PredictResponse(label="scam", score=0.93, model="scam-cnn:v1",
                        ocr=[OcrBox(text="virement urgent", box=[1, 2, 3, 4], suspect=True)])
    assert r.heatmap is None and r.ocr[0].suspect


def test_predict_response_rejects_bad_score():
    with pytest.raises(ValidationError):
        PredictResponse(label="scam", score=1.5, model="m")
