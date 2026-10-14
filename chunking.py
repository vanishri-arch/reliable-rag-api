from dataclasses import dataclass
import re

@dataclass
class Chunk:
    document: str
    chunk_id: str
    text: str

def split_text(text: str, document: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Split text into overlapping word-boundary chunks with stable local IDs."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    words = text.split(" ")
    # Character-budget chunks avoid unexpectedly huge chunks from long words.
    chunks, start, idx = [], 0, 0
    while start < len(words):
        end, length = start, 0
        while end < len(words):
            extra = len(words[end]) + (1 if end > start else 0)
            if length + extra > chunk_size and end > start:
                break
            length += extra
            end += 1
        body = " ".join(words[start:end]).strip()
        if body:
            chunks.append(Chunk(document=document, chunk_id=f"{document}::chunk-{idx:04d}", text=body))
            idx += 1
        if end >= len(words):
            break
        # Move backwards by approximate overlap characters, always advancing at least one word.
        next_start, overlap_chars = end, 0
        while next_start > start + 1 and overlap_chars < overlap:
            next_start -= 1
            overlap_chars += len(words[next_start]) + 1
        start = max(start + 1, next_start)
    return chunks
