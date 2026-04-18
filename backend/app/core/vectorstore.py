from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import Settings

if TYPE_CHECKING:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma


def get_embeddings(settings: Settings) -> "HuggingFaceEmbeddings":
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vectorstore(settings: Settings, session_id: str) -> "Chroma":
    from langchain_community.vectorstores import Chroma

    embeddings = get_embeddings(settings)
    session_dir = Path(settings.chroma_path) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=session_id,
        persist_directory=str(session_dir),
        embedding_function=embeddings,
    )
