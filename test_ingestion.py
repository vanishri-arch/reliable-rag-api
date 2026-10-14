from pathlib import Path
import pytest
from app.ingestion import read_document

def test_reads_text(tmp_path: Path):
    p = tmp_path / "guide.txt"
    p.write_text("Hello RAG", encoding="utf-8")
    assert read_document(p) == "Hello RAG"

def test_rejects_unsupported_file(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        read_document(p)
