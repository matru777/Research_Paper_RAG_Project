import re
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import ParentDocument
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm
PARENT_CHUNK_SIZE = 1_500
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

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

def summarize_tables(md_text: str) -> str:
    """Finds markdown tables, summarizes them with LLM, and prepends the summary."""
    from core.config import get_fast_llm
    llm = get_fast_llm()
    
    # Regex to match a Markdown table
    table_pattern = re.compile(r'(\|.*\|[\r\n]+\|[-:| ]+\|[\r\n]+(?:\|.*\|[\r\n]+)+)', re.MULTILINE)
    
    def replacer(match):
        raw_table = match.group(1)
        try:
            summary = llm.invoke(f"Write a concise 1-2 sentence summary of the key findings in this table. Output only the summary.\n\n{raw_table}").content
            return f"**Table Summary:** {summary}\n\n{raw_table}"
        except Exception:
            return raw_table
            
    return table_pattern.sub(replacer, md_text)

def load_arxiv_pdf(arxiv_id: str) -> list[Document]:
    """Downloads the PDF, extracts text via Docling, then cleans up."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)
        
        md_text = pymupdf4llm.to_markdown(str(tmp_path))
        
        if not md_text:
            raise ValueError(f"Could not read PDF for ArXiv ID: {arxiv_id}")
            
        md_text = summarize_tables(md_text)
        title = arxiv_api_lookup(arxiv_id)
        docs = [Document(page_content=md_text, metadata={"title": title})]
        return docs
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

async def fetch_parent(parent_id: str, session_id: str) -> str|None:
    """Fetches the full Parent text from Postgres during the Parent-Child swap."""
    async with AsyncSessionLocal() as session:
        stmt = select(ParentDocument).where(
            ParentDocument.id == parent_id,
            ParentDocument.session_id == session_id
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()
    return doc.content if doc else None

async def chunk_and_store(
    raw_docs: list[Document],
    title: str,
    session_id: str,
    parent_splitter: RecursiveCharacterTextSplitter
) -> list[Document]:
    for doc in raw_docs:
        doc.metadata["title"] = title
    
    parent_chunks = parent_splitter.split_documents(raw_docs)
    child_docs = []
    
    async with AsyncSessionLocal() as session:
        for parent in parent_chunks:
            parent_id = str(uuid.uuid4())
            
            db_doc = ParentDocument(
                id=parent_id,
                session_id=session_id,
                content=parent.page_content,
                document_metadata={"title": title}
            )
            session.add(db_doc)
            children = child_splitter.split_documents([parent])
            for child in children:
                child.page_content = f"[Paper: {title}] {child.page_content}"
                child.metadata["parent_id"] = parent_id
                child.metadata["title"] = title
                child_docs.append(child)
        
        await session.commit()
        
    return child_docs

async def load_pdf(file_path: str, session_id: str) -> list[Document]:
    md_text = pymupdf4llm.to_markdown(file_path)
    
    md_text = summarize_tables(md_text)
    
    title = Path(file_path).stem
    raw = [Document(page_content=md_text, metadata={"title": title})]
    return await chunk_and_store(raw, title, session_id, md_parent_splitter)

async def load_text(file_path: str, session_id: str) -> list[Document]:
    raw = TextLoader(file_path, encoding="utf-8").load()
    title = Path(file_path).stem
    return await chunk_and_store(raw, title, session_id, parent_splitter)

async def load_markdown(file_path: str, session_id: str) -> list[Document]:
    raw = TextLoader(file_path, encoding="utf-8").load()
    title = Path(file_path).stem
    return await chunk_and_store(raw, title, session_id, md_parent_splitter)

async def load_webpage(url: str, session_id: str) -> list[Document]:
    raw = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (raw[0].metadata.get("title") or url) if raw else url
    return await chunk_and_store(raw, title, session_id, parent_splitter)

async def load_arxiv(query: str, session_id: str) -> list[Document]:
    arxiv_id = extract_arxiv_id(query) or arxiv_search(query)
    raw = load_arxiv_pdf(arxiv_id)
    title = raw[0].metadata.get("title", arxiv_id)
    return await chunk_and_store(raw, title, session_id, md_parent_splitter)

async def load_document(source: str, session_id: str) -> list[Document]:
    """Auto-dispatches to the correct loader based on file extension or URL."""
    if source.startswith(("http://", "https://")):
        return await load_webpage(source, session_id)
    
    if extract_arxiv_id(source):
        return await load_arxiv(source, session_id)
    ext = Path(source).suffix.lower()
    dispatch = {".pdf": load_pdf, ".txt": load_text, ".md": load_markdown, ".markdown": load_markdown}
    loader = dispatch.get(ext)
    
    if loader is None:
        raise ValueError(f"Unsupported file type or invalid ArXiv ID: {source}")
    
    return await loader(source, session_id)

