
from fastapi import APIRouter

from deployment import state
from docguard_ai.contract.schemas import ModelInfo

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    return [ModelInfo(approach=c["approach"], name=c["name"], version=c["version"],
                      input_size=c["input_size"], data_version=c["data_version"])
            for c in state.available()]
