import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

from .config import DOCS_DIR, INDEX_DIR, get_embeddings

EXPECTED_FILES = ["inland_rev.md", "labor_laws.md", "companies_act.md"]


def chunk_markdown(md_text: str, source: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["## ", "### ", "\n", " "]
    )
    return [Document(page_content=t, metadata={"source": source}) for t in splitter.split_text(md_text)]


def build_faiss_from_docs(docs_dir: str = DOCS_DIR, index_dir: str = INDEX_DIR) -> FAISS:
    docs: List[Document] = []
    for fname in EXPECTED_FILES:
        path = os.path.join(docs_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing document: {path}")
        with open(path, "r", encoding="utf-8") as f:
            md = f.read()
        docs.extend(chunk_markdown(md, source=f"docs/{fname}"))

    vs = FAISS.from_documents(docs, get_embeddings())
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)
    return vs


def load_or_build_index(docs_dir: str = DOCS_DIR, index_dir: str = INDEX_DIR) -> FAISS:
    index_path = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_path):
        return FAISS.load_local(index_dir, get_embeddings(), allow_dangerous_deserialization=True)
    return build_faiss_from_docs(docs_dir, index_dir)
