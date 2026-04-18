from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    session_id: str
    uploaded_files: list[str]
    total_chunks_indexed: int
    message: str


class AskRequest(BaseModel):
    session_id: str = Field(..., min_length=4)
    question: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=20)


class SourceDocument(BaseModel):
    source: str
    page: int | None = None
    chunk: int | None = None
    excerpt: str | None = None


class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[SourceDocument]
