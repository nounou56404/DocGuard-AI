from fastapi import FastAPI
from api.routers import health, models, predict

app = FastAPI(title="DocGuard-AI A - scam-cnn", version="0.1.0")
app.include_router(predict.router)
app.include_router(health.router)
app.include_router(models.router)
