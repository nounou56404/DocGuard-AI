from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from deployment import state
from deployment.inference import decode_image
from docguard_ai.config import get_params
from docguard_ai.contract.schemas import PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...),  # noqa: B008
                  model: str = Query("cnn", description="cnn | vit | ocr"),
                  ) -> PredictResponse:
    net = await run_in_threadpool(state.get_model, model)
    if net is None:
        raise HTTPException(404, f"modele '{model}' indisponible (models/{model}/model.onnx)")
    img = decode_image(await file.read())
    if img is None:
        raise HTTPException(400, "image illisible")
    score = await run_in_threadpool(net.predict, img)
    thr = get_params().get("evaluate", {}).get("threshold", 0.5)
    # heatmap (Grad-CAM / attention) et ocr (boites) : a brancher via docguard_ai.vision.explain
    # et docguard_ai.ocr.engine ; None tant que non implementes.
    return PredictResponse(label="scam" if score >= thr else "non_scam", score=score,
                           model=net.tag, heatmap=None, ocr=None)
