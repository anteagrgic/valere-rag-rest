from typing import List
from sklearn.datasets import fetch_20newsgroups
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils import ensure_dirs

from .config import settings
from .utils import ensure_dirs
from .vectorstore import add_documents

def load_20newsgroups_docs() -> List[Document]:
    ensure_dirs()
    data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
    docs: List[Document] = []
    for i, text in enumerate(data.data):
        if not text or not text.strip():
            continue
        meta = {
            "doc_id": i,
            "target": int(data.target[i]),
            "target_name": data.target_names[data.target[i]],
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def chunk_docs(docs: List[Document], chunk_size: int | None = None, chunk_overlap: int | None = None) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def ingest_20newsgroups(collection: str, recreate: bool = False, chunk_size: int | None = None, chunk_overlap: int | None = None) -> int:
    docs = load_20newsgroups_docs()
    chunks = chunk_docs(docs, chunk_size, chunk_overlap)
    count = add_documents(chunks, collection)
    return count
