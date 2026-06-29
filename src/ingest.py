"""
Ingestion pipeline:
1. Load PDFs from disk
2. Split into overlapping chunks
3. Attach metadata (source file, page number, chunk id)

Decoupled from the vector store so chunks can be inspected/debugged
before embedding.
"""
import os
import glob
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src import config


def load_pdfs(pdf_dir: str = config.PDF_DIR) -> List[Document]:
    """Load every PDF in pdf_dir into LangChain Document objects (one per page)."""
    pdf_paths = glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Put at least one .pdf file there."
        )

    all_docs: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = os.path.basename(path)
        all_docs.extend(pages)

    print(f"[ingest] Loaded {len(all_docs)} pages from {len(pdf_paths)} PDF(s).")
    return all_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Recursively split documents into chunks, trying to break on paragraph,
    then sentence, then word boundaries (in that priority order).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    print(f"[ingest] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


def run_ingestion(pdf_dir: str = config.PDF_DIR) -> List[Document]:
    docs = load_pdfs(pdf_dir)
    chunks = chunk_documents(docs)
    return chunks


if __name__ == "__main__":
    chunks = run_ingestion()
    print("\nSample chunk:\n", chunks[0].page_content[:300])
    print("\nMetadata:", chunks[0].metadata)
