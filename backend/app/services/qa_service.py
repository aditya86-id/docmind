from __future__ import annotations

from typing import List

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.vectorstore import get_vectorstore


def get_llm(settings: Settings) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def build_sources(docs: List[Document]) -> list[dict]:
    sources: list[dict] = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        chunk = doc.metadata.get("chunk_index", doc.metadata.get("chunk"))
        key = (source, page, chunk)

        if key in seen:
            continue
        seen.add(key)

        excerpt = doc.page_content.strip().replace("\n", " ")
        if len(excerpt) > 280:
            excerpt = excerpt[:280].rstrip() + "..."

        sources.append(
            {
                "source": source,
                "page": page,
                "chunk": chunk,
                "excerpt": excerpt,
            }
        )

    return sources


def answer_question(settings: Settings, session_id: str, question: str, top_k: int = 4) -> tuple[str, list[dict]]:
    vectorstore = get_vectorstore(settings, session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = get_llm(settings)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful document question-answering assistant. "
                "Use only the provided context to answer. "
                "If the answer is not in the context, say you do not know. "
                "Keep the answer concise and accurate.",
            ),
            (
                "human",
                "Question: {input}\n\nContext:\n{context}",
            ),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    result = qa_chain.invoke({"input": question})
    answer = result.get("answer", "").strip()
    context_docs = result.get("context", []) or []
    sources = build_sources(context_docs)
    return answer, sources
