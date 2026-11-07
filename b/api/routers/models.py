from typing import List
from fastapi import APIRouter
from contract.schemas import ModelInfo

router = APIRouter()


@router.get("/models", response_model=List[ModelInfo])
async def list_models() -> List[ModelInfo]:
    raise NotImplementedError("A implementer : lire model_card.json")
