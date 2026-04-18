# AI Document QA System

A practical document question-answering project with:

- **FastAPI** backend
- **Streamlit** frontend
- **LangChain** for retrieval-augmented generation
- **Chroma** for persistent vector storage
- **OpenAI-compatible chat model** and embeddings through LangChain

## What it does

1. Upload PDF or text documents.
2. Documents are chunked and embedded.
3. Ask questions in a chat UI.
4. The backend retrieves relevant chunks and uses the LLM to answer.

## Folder structure

```text
ai_document_qa_system/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── services/
│   │   └── main.py
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 3) Install frontend dependencies

```bash
pip install -r frontend/requirements.txt
```

### 4) Configure environment variables

Copy `.env.example` to `.env` and set your API key.

```bash
cp .env.example .env
```

### 5) Start the backend

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6) Start the Streamlit frontend

```bash
streamlit run frontend/streamlit_app.py
```

## Notes

- The backend stores uploaded files and Chroma indexes inside `storage/`.
- Each browser session gets its own session ID so users do not overwrite each other.
- Supported file types in this starter: **.pdf, .txt, .md**.
- If the answer is not in the documents, the model is instructed to say so.
