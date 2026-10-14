from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)

class SourceEvidence(BaseModel):
    document: str
    chunk_id: str
    excerpt: str
    score: float

class AskResponse(BaseModel):
    answer: str
    found_evidence: bool
    sources: list[SourceEvidence]

class IngestResponse(BaseModel):
    files_processed: int
    chunks_added: int
    documents: list[str]
