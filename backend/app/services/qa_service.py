from __future__ import annotations
 
from typing import List
 
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
 
from app.core.config import Settings
from app.core.vectorstore import get_vectorstore
 
 
def get_llm(settings: Settings) -> ChatGroq:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
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
 
 
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
 
 
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
                "Question: {question}\n\nContext:\n{context}",
            ),
        ]
    )
 
    # Retrieve docs separately so we can return them as sources
    context_docs = retriever.invoke(question)
 
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "context": format_docs(context_docs),
    }).strip()
 
    sources = build_sources(context_docs)
    return answer, sources