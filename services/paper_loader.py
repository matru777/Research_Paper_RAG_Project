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


PARENT_CHUNK_SIZE = 1_500
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50
DOC_STORE_PATH = "doc_store.db"

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=0,
    add_start_index=True,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
    add_start_index=True,
)
md_parent_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown",
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=0,
    add_start_index=True,
)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")
def extract_arxiv_id(query: str) -> str | None:
    """Checks if the query already contains a raw ArXiv ID like '1706.03762'."""
    m = ARXIV_ID_RE.search(query)
    return re.sub(r"v\d+$", "", m.group(1)) if m else None

def arxiv_api_lookup(arxiv_id: str) -> str:
    """Resolves a bare ArXiv ID to its paper title via the Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id

def arxiv_search(query: str) -> str:
    """Full-text title search on ArXiv — returns the bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for query: {query!r}")
    return re.sub(r"v\d+$", "", m.group(1))

def load_arxiv_pdf(arxiv_id: str) -> list[Document]:
    """Downloads the PDF, saves to a temp file, extracts text, then cleans up."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)
        docs = PyMuPDFLoader(str(tmp_path)).load()
        if not docs:
            raise ValueError(f"Could not read PDF for ArXiv ID: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or arxiv_api_lookup(arxiv_id)
        for doc in docs:
            doc.metadata["title"] = title
        return docs
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

def init_doc_store(db_path: str = DOC_STORE_PATH):
    """Creates the parent-chunk table if it doesn't exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS document_store "
            "(parent_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, title TEXT, text TEXT NOT NULL)"
        )
        conn.commit()

@contextmanager
def doc_store(db_path: str = DOC_STORE_PATH):
    """Safe database connection that always closes, even on crash."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def fetch_parent(parent_id: str,session_id: str, db_path: str = DOC_STORE_PATH) -> str | None:
    """Fetches the full Parent text from SQLite during the Parent-Child swap."""
    with doc_store(db_path) as conn:
        row = conn.execute(
            "SELECT text FROM document_store WHERE parent_id = ? AND session_id = ?", (parent_id,session_id)
        ).fetchone()
    return row[0] if row else None

def chunk_and_store(
    raw_docs: list[Document],
    title: str,
    session_id: str,
    parent_splitter: RecursiveCharacterTextSplitter,
    db_path: str = DOC_STORE_PATH
) -> list[Document]:

    """
    The heart of the Parent-Child strategy:
    1. Split raw pages into big Parent chunks and save them to SQLite.
    2. Split Parents into tiny Children and inject the parent_id.
    3. Prepend the paper title to each Child for context enrichment.
    4. Return the Children for embedding into Qdrant.
    """

    init_doc_store(db_path)

    for doc in raw_docs:
        doc.metadata["title"] = title
    
    parent_chunks = parent_splitter.split_documents(raw_docs)
    child_docs=[]

    with doc_store(db_path) as conn:
        for parent in parent_chunks:
            parent_id = str(uuid.uuid4())

            conn.execute(
                "INSERT OR REPLACE INTO document_store (parent_id, session_id, title, text) VALUES (?, ?, ?, ?)",
                (parent_id, session_id, title, parent.page_content),
            )

            children = child_splitter.split_documents([parent])
            for child in children:
                # Add context enrichment and tracking metadata
                child.page_content = f"[Paper: {title}] {child.page_content}"
                child.metadata["parent_id"] = parent_id
                child.metadata["title"] = title
                child_docs.append(child)

        conn.commit()
    return child_docs

def load_pdf(file_path: str, session_id: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    raw = PyMuPDFLoader(file_path).load()
    title = (raw[0].metadata.get("title") or "").strip() or Path(file_path).stem
    return chunk_and_store(raw, title, session_id, parent_splitter, db_path)

def load_text(file_path: str, session_id: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    raw = TextLoader(file_path, encoding="utf-8").load()
    title = Path(file_path).stem
    return chunk_and_store(raw, title, session_id, parent_splitter, db_path)

def load_markdown(file_path: str, session_id: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    raw = TextLoader(file_path, encoding="utf-8").load()
    title = Path(file_path).stem
    return chunk_and_store(raw, title, session_id, md_parent_splitter, db_path)

def load_webpage(url: str, session_id: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    raw = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (raw[0].metadata.get("title") or url) if raw else url
    return chunk_and_store(raw, title, session_id, parent_splitter, db_path)

def load_arxiv(query: str, session_id: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    arxiv_id = extract_arxiv_id(query) or arxiv_search(query)
    raw = load_arxiv_pdf(arxiv_id)
    title = raw[0].metadata.get("title", arxiv_id)
    return chunk_and_store(raw, title, session_id, parent_splitter, db_path)
    
def load_document(source: str, session_id: str, db_path: str = DOC_STORE_PATH) -> list[Document]:
    """Auto-dispatches to the correct loader based on file extension or URL."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source, session_id, db_path)
    
    if extract_arxiv_id(source):
        return load_arxiv(source, session_id, db_path)

    ext = Path(source).suffix.lower()
    dispatch = {".pdf": load_pdf, ".txt": load_text, ".md": load_markdown, ".markdown": load_markdown}
    loader = dispatch.get(ext)
    
    if loader is None:
        raise ValueError(f"Unsupported file type or invalid ArXiv ID: {source}")
    
    return loader(source, session_id, db_path)