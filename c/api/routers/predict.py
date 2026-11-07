from fastapi import APIRouter, File, UploadFile
from contract.schemas import PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    raise NotImplementedError("A implementer : pretraitement -> ONNX -> explicabilite")
