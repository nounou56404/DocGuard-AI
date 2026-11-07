"""API FastAPI asynchrone. Lancer : uvicorn deployment.app:app --reload  (Swagger : /docs)."""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deployment.routers import health, models, predict

app = FastAPI(title="DocGuard-AI", version="0.1.0",
              description="Detection de documents scam / non_scam")
app.add_middleware(CORSMiddleware,
                   allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(health.router)
app.include_router(models.router)
app.include_router(predict.router)
