"""
paper_loader.py
───────────────
Document ingestion pipeline for Papeer.

Strategy: Parent-Child chunking
  • Parent chunks  (≈1 500 chars) are saved to SQLite for full-context retrieval.
  • Child chunks   (≈300 chars)   are embedded and pushed to Qdrant for recall.
  • Every child carries its parent_id so the retrieval node can swap it back.

Supported sources
  • Local PDF / TXT / Markdown files
  • Arbitrary HTTPS URLs  (WebBaseLoader)
  • ArXiv papers          (by ID  OR  title-phrase search)
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Chunking parameters ───────────────────────────────────────────────────────

PARENT_CHUNK_SIZE = 1_500
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=0,           # parents should not overlap
    add_start_index=True,
)
_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
    add_start_index=True,
)
_md_parent_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown",
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=0,
    add_start_index=True,
)

# ── ArXiv helpers ─────────────────────────────────────────────────────────────

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")


def _extract_arxiv_id(query: str) -> str | None:
    m = _ARXIV_ID_RE.search(query)
    return re.sub(r"v\d+$", "", m.group(1)) if m else None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    """Resolve a bare ArXiv ID to its paper title via the Atom feed."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:
    """Full-text title search on ArXiv — returns the bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query={search_query}&max_results=1&sortBy=relevance"
    )
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for query: {query!r}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_pdf(arxiv_id: str) -> list[Document]:
    """Download an ArXiv PDF by bare ID and return raw (unsplit) Document list."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)
        docs = PyMuPDFLoader(str(tmp_path)).load()
        if not docs:
            raise ValueError(f"PyMuPDF could not load ArXiv PDF: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or _arxiv_api_lookup(arxiv_id)
        for doc in docs:
            doc.metadata["title"] = title
        return docs
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


# ── SQLite document store ─────────────────────────────────────────────────────

DOC_STORE_PATH = "doc_store.db"


def _init_doc_store(db_path: str = DOC_STORE_PATH) -> None:
    """Create the parent-chunk table if it does not exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_store (
                parent_id TEXT PRIMARY KEY,
                title     TEXT,
                text      TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def _doc_store(db_path: str = DOC_STORE_PATH):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def fetch_parent(parent_id: str, db_path: str = DOC_STORE_PATH) -> str | None:
    """Return the full parent text for a given parent_id, or None if missing."""
    with _doc_store(db_path) as conn:
        row = conn.execute(
            "SELECT text FROM document_store WHERE parent_id = ?", (parent_id,)
        ).fetchone()
    return row[0] if row else None


# ── Core chunking pipeline ────────────────────────────────────────────────────

def _chunk_and_store(
    raw_docs: list[Document],
    title: str,
    parent_splitter: RecursiveCharacterTextSplitter,
    db_path: str = DOC_STORE_PATH,
) -> list[Document]:
    """
    Split raw_docs into Parent chunks → store in SQLite.
    Split Parents into Child chunks → enrich with context → return for embedding.

    Returns a list of child Document objects ready to be pushed to Qdrant.
    """
    _init_doc_store(db_path)

    # Stamp the title on every raw page
    for doc in raw_docs:
        doc.metadata["title"] = title

    parent_chunks = parent_splitter.split_documents(raw_docs)
    child_docs: list[Document] = []

    with _doc_store(db_path) as conn:
        for parent in parent_chunks:
            parent_id = str(uuid.uuid4())

            # Persist parent in SQLite
            conn.execute(
                "INSERT OR REPLACE INTO document_store (parent_id, title, text) VALUES (?, ?, ?)",
                (parent_id, title, parent.page_content),
            )

            # Split into children
            children = _child_splitter.split_documents([parent])
            for child in children:
                # Context enrichment: prepend paper title to the child text
                child.page_content = f"[Paper: {title}] {child.page_content}"
                child.metadata["parent_id"] = parent_id
                child.metadata["title"] = title
                child_docs.append(child)

        conn.commit()

    return child_docs


# ── Public loaders ────────────────────────────────────────────────────────────

def load_pdf(file_path: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    raw = PyMuPDFLoader(file_path).load()
    title = (raw[0].metadata.get("title") or "").strip() or Path(file_path).stem
    return _chunk_and_store(raw, title, _parent_splitter, db_path)


def load_text(file_path: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    raw = TextLoader(file_path, encoding="utf-8").load()
    title = Path(file_path).stem
    return _chunk_and_store(raw, title, _parent_splitter, db_path)


def load_markdown(file_path: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    raw = TextLoader(file_path, encoding="utf-8").load()
    title = Path(file_path).stem
    return _chunk_and_store(raw, title, _md_parent_splitter, db_path)


def load_webpage(url: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    raw = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (raw[0].metadata.get("title") or url) if raw else url
    return _chunk_and_store(raw, title, _parent_splitter, db_path)


def load_arxiv(query: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    raw = _load_arxiv_pdf(arxiv_id)
    title = raw[0].metadata.get("title", arxiv_id)
    return _chunk_and_store(raw, title, _parent_splitter, db_path)


def load_document(source: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    """Dispatch to the correct loader based on URL scheme or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source, db_path)
    ext = Path(source).suffix.lower()
    dispatch = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md": load_markdown,
        ".markdown": load_markdown,
    }
    loader = dispatch.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext!r}")
    return loader(source, db_path)