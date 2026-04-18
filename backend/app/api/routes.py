from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import Settings
from app.schemas import AskRequest, AskResponse, SourceDocument, UploadResponse
from app.services.document_service import (
    ensure_supported_file,
    load_documents,
    safe_session_name,
    save_upload_file,
    split_documents,
)
from app.services.qa_service import answer_question
from app.core.vectorstore import get_vectorstore

router = APIRouter(prefix="/api", tags=["document-qa"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
):
    settings = Settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    session_name = safe_session_name(session_id)
    upload_root = Path(settings.upload_path) / session_name
    upload_root.mkdir(parents=True, exist_ok=True)

    vectorstore = get_vectorstore(settings, session_name)
    total_chunks = 0
    uploaded_names: list[str] = []

    for file in files:
        ensure_supported_file(file.filename or "")
        destination = upload_root / (file.filename or "uploaded_file")
        await save_upload_file(file, destination)
        uploaded_names.append(destination.name)

        docs = load_documents(destination)
        chunks = split_documents(docs)
        if chunks:
            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)

    return UploadResponse(
        session_id=session_id,
        uploaded_files=uploaded_names,
        total_chunks_indexed=total_chunks,
        message=f"Indexed {len(uploaded_names)} file(s) into your session knowledge base.",
    )


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    settings = Settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")

    session_name = safe_session_name(payload.session_id)
    answer, sources = answer_question(
        settings=settings,
        session_id=session_name,
        question=payload.question,
        top_k=payload.top_k,
    )

    return AskResponse(
        session_id=payload.session_id,
        question=payload.question,
        answer=answer,
        sources=[SourceDocument(**src) for src in sources],
    )


@router.delete("/documents/clear/{session_id}")
def clear_session(session_id: str):
    settings = Settings()
    session_name = safe_session_name(session_id)
    chroma_dir = Path(settings.chroma_path) / session_name
    upload_dir = Path(settings.upload_path) / session_name

    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)

    return {"session_id": session_id, "message": "Session data cleared."}
