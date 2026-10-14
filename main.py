from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from .config import settings
from .schemas import AskRequest, AskResponse, SourceEvidence, IngestResponse
from .ingestion import load_documents, read_document, SUPPORTED
from .retrieval import VectorIndex
from .answering import generate_answer, NO_ANSWER

app = FastAPI(title=settings.app_name, version="1.0.0",
              description="Document-grounded question answering with evidence and abstention.")
DATA_DIR = Path(settings.data_dir)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
index = VectorIndex(settings.embedding_model, DATA_DIR / "index",
                    settings.chunk_size, settings.chunk_overlap)

@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(index.chunks)}

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    docs = load_documents(UPLOAD_DIR)
    if not docs:
        raise HTTPException(status_code=400, detail="No readable PDF, TXT, or Markdown documents found in data/uploads.")
    count = index.ingest_texts(docs)
    return IngestResponse(files_processed=len(docs), chunks_added=count, documents=list(docs))

@app.post("/documents", response_model=IngestResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(status_code=400, detail=f"Unsupported file: {file.filename}. Use PDF, TXT, or MD.")
        safe_name = Path(file.filename or "upload").name
        destination = UPLOAD_DIR / safe_name
        content = await file.read()
        if not content:
            continue
        destination.write_bytes(content)
        saved.append(safe_name)
    if not saved:
        raise HTTPException(status_code=400, detail="Uploaded files were empty.")
    docs = {}
    for name in saved:
        try:
            docs[name] = read_document(UPLOAD_DIR / name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read {name}: {exc}") from exc
    count = index.ingest_texts(docs)
    return IngestResponse(files_processed=len(docs), chunks_added=count, documents=list(docs))

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    k = request.top_k or settings.top_k
    results = index.search(request.question, k)
    # Similarity is a heuristic, not a calibrated probability.
    evidence = [r for r in results if r["score"] >= settings.min_relevance]
    sources = [SourceEvidence(document=r["chunk"].document, chunk_id=r["chunk"].chunk_id,
                              excerpt=r["chunk"].text[:500], score=round(r["score"], 4))
               for r in evidence]
    if not evidence:
        return AskResponse(answer=NO_ANSWER, found_evidence=False, sources=[])
    try:
        answer = generate_answer(request.question, evidence, settings.openai_api_key, settings.openai_model)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Answer generation failed. Check LLM configuration and provider availability. ({type(exc).__name__})") from exc
    return AskResponse(answer=answer, found_evidence=True, sources=sources)
