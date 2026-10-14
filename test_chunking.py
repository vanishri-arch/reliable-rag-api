import pytest
from app.chunking import split_text

def test_empty_text_returns_no_chunks():
    assert split_text("  ", "empty.txt") == []

def test_chunks_are_nonempty_and_have_ids():
    chunks = split_text("alpha beta gamma delta epsilon", "demo.txt", chunk_size=12, overlap=3)
    assert len(chunks) > 1
    assert all(c.text and c.chunk_id.startswith("demo.txt::chunk-") for c in chunks)

def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        split_text("hello world", "x.txt", chunk_size=10, overlap=10)
