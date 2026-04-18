from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from app.core.config import Settings


class LocalHashEmbeddings(Embeddings):
    """Lightweight offline embeddings to avoid external model downloads."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


def get_embeddings(settings: Settings) -> Embeddings:
    _ = settings.embedding_model
    return LocalHashEmbeddings()


def get_vectorstore(settings: Settings, session_id: str) -> Chroma:
    embeddings = get_embeddings(settings)
    session_dir = Path(settings.chroma_path) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=session_id,
        persist_directory=str(session_dir),
        embedding_function=embeddings,
    )
