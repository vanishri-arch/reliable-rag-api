from pathlib import Path
from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}

def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return path.read_text(encoding="utf-8", errors="replace").strip()

def load_documents(directory: Path) -> dict[str, str]:
    """Load supported files recursively; unreadable files are skipped by the caller."""
    result = {}
    if not directory.exists():
        return result
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            try:
                content = read_document(path)
                if content:
                    result[path.name] = content
            except Exception:
                # A single corrupt file must not prevent other documents from loading.
                continue
    return result
