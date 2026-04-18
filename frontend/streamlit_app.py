from __future__ import annotations

import os
import uuid

import requests
import streamlit as st


st.set_page_config(page_title="AI Document QA", page_icon="📄", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload documents first, then ask questions about them.",
        }
    ]

if "last_upload" not in st.session_state:
    st.session_state.last_upload = None


def api_url(path: str) -> str:
    return f"{BACKEND_URL}{path}"


def upload_files(uploaded_files):
    if not uploaded_files:
        return None

    files = []
    for file in uploaded_files:
        files.append(("files", (file.name, file.getvalue(), file.type or "application/octet-stream")))

    data = {"session_id": st.session_state.session_id}
    response = requests.post(api_url("/api/documents/upload"), data=data, files=files, timeout=300)
    response.raise_for_status()
    return response.json()


def ask_question(question: str, top_k: int = 4):
    payload = {
        "session_id": st.session_state.session_id,
        "question": question,
        "top_k": top_k,
    }
    response = requests.post(api_url("/api/ask"), json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def clear_session():
    requests.delete(api_url(f"/api/documents/clear/{st.session_state.session_id}"), timeout=120)
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Session cleared. Upload fresh documents to begin again.",
        }
    ]
    st.session_state.last_upload = None


st.title("📄 AI Document QA System")
st.caption("FastAPI backend + Streamlit frontend + LangChain retrieval")

with st.sidebar:
    st.subheader("Settings")
    st.write(f"Backend: `{BACKEND_URL}`")
    top_k = st.slider("Retriever top-k", min_value=1, max_value=10, value=4)
    st.write(f"Session ID: `{st.session_state.session_id[:8]}...`")
    if st.button("Clear session", use_container_width=True):
        try:
            clear_session()
            st.success("Session cleared.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to clear session: {exc}")

st.subheader("1) Upload documents")
uploaded_files = st.file_uploader(
    "Supported: PDF, TXT, MD",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("Index documents", use_container_width=True):
    try:
        if not uploaded_files:
            st.warning("Choose at least one file first.")
        else:
            with st.spinner("Uploading and indexing..."):
                result = upload_files(uploaded_files)
            st.session_state.last_upload = result
            st.success(result["message"])
            st.info(f"Indexed chunks: {result['total_chunks_indexed']}")
    except requests.HTTPError as exc:
        st.error(f"Upload failed: {exc.response.text}")
    except Exception as exc:
        st.error(f"Upload failed: {exc}")

if st.session_state.last_upload:
    st.write("Last upload:")
    st.json(st.session_state.last_upload)

st.divider()
st.subheader("2) Ask questions")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your uploaded documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Thinking..."):
            result = ask_question(question, top_k=top_k)

        answer = result["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

            sources = result.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        label = src["source"]
                        page = src.get("page")
                        chunk = src.get("chunk")
                        meta = []
                        if page is not None:
                            meta.append(f"page {page + 1}")
                        if chunk is not None:
                            meta.append(f"chunk {chunk}")
                        heading = f"**{label}**"
                        if meta:
                            heading += " — " + ", ".join(meta)
                        st.markdown(heading)
                        if src.get("excerpt"):
                            st.caption(src["excerpt"])
    except requests.HTTPError as exc:
        st.error(f"Answer request failed: {exc.response.text}")
    except Exception as exc:
        st.error(f"Answer request failed: {exc}")
