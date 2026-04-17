from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def ensure_supported_file(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    return suffix


async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = await upload_file.read()
    destination.write_bytes(content)
    return destination


def load_documents(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    else:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

    for idx, doc in enumerate(docs):
        doc.metadata = {
            **doc.metadata,
            "source": file_path.name,
            "file_path": str(file_path),
            "chunk": idx,
        }
    return docs


def split_documents(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata.setdefault("source", chunk.metadata.get("file_path", "unknown"))
    return chunks


def safe_session_name(session_id: str) -> str:
    return _stable_hash(session_id)
