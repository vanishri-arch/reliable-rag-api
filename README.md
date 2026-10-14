# Reliable RAG API

A compact Retrieval-Augmented Generation API that ingests PDF/TXT/Markdown files, searches chunks with local sentence-transformer embeddings, answers with an LLM, returns evidence excerpts, and abstains when retrieval finds no sufficiently relevant evidence.

## Architecture

1. **Ingestion** — accepts `.pdf`, `.txt`, and `.md`; PDF text is extracted with `pypdf`.
2. **Chunking** — creates overlapping, bounded text chunks with stable IDs.
3. **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` runs locally.
4. **Vector search** — normalized embeddings and cosine similarity using NumPy. This keeps setup simple for a small assessment corpus; swap for FAISS/Chroma/Qdrant for larger corpora.
5. **Answer generation** — OpenAI-compatible chat completion with temperature 0 and an evidence-only prompt.
6. **Evidence and abstention** — response includes document, chunk ID, excerpt, similarity score; no retrieved evidence produces a fixed “I don't know…” answer without an LLM call.

## Requirements

- Python 3.11 recommended
- An OpenAI API key for answer generation (embedding model is local)
- Internet on first install/model download

## Local setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Put your own API key in `.env` as `OPENAI_API_KEY=...`. Never commit `.env` or paste the key into source code.

Start the API:

```powershell
uvicorn app.main:app --reload
```

Open interactive API docs at http://127.0.0.1:8000/docs and health at http://127.0.0.1:8000/health.

The first start downloads the embedding model and may take a few minutes.

## Quick start

1. Upload documents via Swagger `POST /documents` (choose one or more PDF, TXT, or Markdown files), or place files in `data/uploads/` and call `POST /ingest`.
2. Ask a question using `POST /ask`:

```json
{
  "question": "How many paid annual leave days do full-time employees receive?",
  "top_k": 4
}
```

Example response shape (the exact answer and similarity scores depend on the model and corpus):

```json
{
  "answer": "Full-time employees receive 18 days of paid annual leave per calendar year [company_handbook.md, company_handbook.md::chunk-0000].",
  "found_evidence": true,
  "sources": [
    {
      "document": "company_handbook.md",
      "chunk_id": "company_handbook.md::chunk-0000",
      "excerpt": "Full-time employees receive 18 days of paid annual leave per calendar year...",
      "score": 0.71
    }
  ]
}
```

Similarity scores above are illustrative, not measured test results. A score is a ranking heuristic, not a probability that an answer is correct.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check and indexed chunk count |
| POST | `/documents` | Upload PDF/TXT/MD files and index them |
| POST | `/ingest` | Index supported files already in `data/uploads/` |
| POST | `/ask` | Retrieve evidence and answer a question |

## Evaluation and tests

Fourteen example questions live in `data/evaluation/questions.json`: answerable questions, unanswerable questions, and a question requiring multiple pieces of evidence. Use these as a manual evaluation set against the sample handbook. They are not a claim of measured model accuracy.

Run automated unit tests:

```powershell
pytest -q
```

Tests cover chunking boundaries/IDs, invalid chunk settings, text-file ingestion, unsupported extensions, and the no-evidence abstention path. They do not call the live LLM or measure retrieval quality. Before submitting, run the tests locally and report the actual result; do not claim they passed unless they did.

## Docker

```powershell
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY
docker compose up --build
```

Persisted documents and the local index are mounted under `./data`.

## Design choices and trade-offs

- **NumPy cosine index:** minimal setup and transparent for small datasets; linear scan is slower than dedicated vector databases at scale.
- **Local embeddings:** avoids embedding API costs and keeps document text local for embedding; first run downloads a model.
- **LLM API:** simple hosted answer generation; requires a valid API key, network access, and incurs provider cost.
- **Similarity threshold:** configurable `MIN_RELEVANCE`; cosine similarity is not calibrated confidence. Tune it against domain-specific evaluation data to balance false positives and false abstentions.
- **Citations:** source excerpts and chunk IDs are returned separately from the generated answer so a client can inspect evidence. LLM-generated inline citations still require evaluation.
- **Persistence:** local JSON + NumPy files are appropriate for a demo, not concurrent multi-user production.
- **PDFs:** scanned/image-only PDFs need OCR, which is not included.
- **Security:** demo API has no authentication, rate limiting, malware scanning, or per-user isolation. Do not expose it publicly without those controls.
- **Reliability:** ingestion skips unreadable files in batch mode; the upload endpoint returns an error for a file it cannot parse. LLM provider errors return HTTP 502 without leaking credentials.

## Before production

Add authentication/authorization, upload size limits, file scanning, OCR if needed, structured logs/metrics, request timeouts, rate limits, a managed vector database, corpus versioning, and automated citation-faithfulness evaluation.
