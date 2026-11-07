from fastapi import APIRouter

from docguard_ai.contract.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()
