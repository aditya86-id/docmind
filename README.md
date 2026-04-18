# AI Document QA System

A practical document question-answering project with:

- **FastAPI** backend
- **Streamlit** frontend
- **LangChain** for retrieval-augmented generation
- **Chroma** for persistent vector storage
- **Groq** for ultra-fast LLM inference (Llama 3.3 70B by default)
- **HuggingFace sentence-transformers** for local embeddings (no extra API key needed)

## What it does

1. Upload PDF or text documents.
2. Documents are chunked and embedded locally.
3. Ask questions in a chat UI.
4. The backend retrieves relevant chunks and uses Groq's LLM to answer.

## Folder structure

```text
docmind/
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
├── .gitignore
└── README.md
```

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install backend dependencies

> **Important:** Use `pip install`, not `uv pip install`. The `uv pip install` command may only audit already-installed packages without installing new ones.

```bash
pip install -r backend/requirements.txt
```

### 3) Install frontend dependencies

```bash
pip install -r frontend/requirements.txt
```

### 4) Configure environment variables

Copy `.env.example` to `.env` and set your Groq API key (free at https://console.groq.com).

```bash
cp .env.example .env
# Edit .env and set: GROQ_API_KEY=your_key_here
```

### 5) Start the backend

> **Important:** You must `cd` into `backend/` before running uvicorn. The app uses relative imports that require `app/` to be on the Python path directly. Running `uvicorn backend.app.main:app` from the project root will fail.

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6) Start the Streamlit frontend

Open a **second terminal** from the project root:

```bash
streamlit run frontend/streamlit_app.py
```

## Notes

- **Groq API key** is required. Get one free at https://console.groq.com.
- **Embeddings run locally** via HuggingFace `sentence-transformers` — no additional API key needed.
- The backend stores uploaded files and Chroma indexes inside `storage/`.
- Each browser session gets its own session ID so users do not overwrite each other.
- Supported file types: **.pdf, .txt, .md**.
- If the answer is not in the documents, the model is instructed to say so.
- **Never commit your `.env` file** — it is listed in `.gitignore`.