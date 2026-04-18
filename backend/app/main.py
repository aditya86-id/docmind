from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()

Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
Path(settings.upload_path).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AI Document QA System",
    version="1.0.0",
    description="FastAPI backend for document question answering with LangChain and Chroma.",
)

if settings.cors_origins.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
