from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vectorstore(settings: Settings, session_id: str) -> Chroma:
    embeddings = get_embeddings(settings)
    session_dir = Path(settings.chroma_path) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=session_id,
        persist_directory=str(session_dir),
        embedding_function=embeddings,
    )
