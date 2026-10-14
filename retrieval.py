from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from .chunking import split_text, Chunk

class VectorIndex:
    """Small local cosine-similarity index. Suitable for demos/small corpora."""
    def __init__(self, model_name: str, data_dir: Path, chunk_size: int = 800, overlap: int = 120):
        self.model = SentenceTransformer(model_name)
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size, self.overlap = chunk_size, overlap
        self.chunks: list[Chunk] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self._load()

    def _load(self):
        meta = self.data_dir / "index.json"
        vec = self.data_dir / "vectors.npy"
        if meta.exists() and vec.exists():
            try:
                self.chunks = [Chunk(**x) for x in json.loads(meta.read_text(encoding="utf-8"))]
                self.vectors = np.load(vec)
            except Exception:
                self.chunks, self.vectors = [], np.empty((0, 0), dtype=np.float32)

    def ingest_texts(self, documents: dict[str, str]) -> int:
        added = []
        existing_names = {c.document for c in self.chunks}
        # Replace previously indexed chunks for a re-ingested filename.
        for name, content in documents.items():
            self.chunks = [c for c in self.chunks if c.document != name]
            new_chunks = split_text(content, name, self.chunk_size, self.overlap)
            if new_chunks:
                added.extend(new_chunks)
        self.chunks.extend(added)
        if self.chunks:
            self.vectors = self.model.encode([c.text for c in self.chunks], normalize_embeddings=True,
                                             convert_to_numpy=True).astype(np.float32)
        else:
            self.vectors = np.empty((0, 0), dtype=np.float32)
        self._save()
        return len(added)

    def _save(self):
        (self.data_dir / "index.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(self.data_dir / "vectors.npy", self.vectors)

    def search(self, question: str, top_k: int = 4) -> list[dict]:
        if not self.chunks or self.vectors.size == 0:
            return []
        q = self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.vectors @ q
        indices = np.argsort(scores)[::-1][:top_k]
        return [{"chunk": self.chunks[int(i)], "score": float(scores[int(i)])} for i in indices]
