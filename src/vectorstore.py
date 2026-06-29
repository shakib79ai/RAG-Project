"""
Builds and persists a Chroma vector store from chunked documents,
using a local sentence-transformers embedding model (no API calls,
no per-token cost - runs on CPU or GPU on your own machine).
"""
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    BAAI/bge-small-en-v1.5 is a strong, lightweight open-source embedding
    model (~130MB) that runs comfortably on CPU.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # required for cosine similarity
    )


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed chunks locally and persist them to disk as a Chroma collection."""
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTORSTORE_DIR,
    )
    print(f"[vectorstore] Persisted {len(chunks)} chunks to {config.VECTORSTORE_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an already-persisted Chroma collection (no re-embedding)."""
    embeddings = get_embedding_model()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.VECTORSTORE_DIR,
    )


if __name__ == "__main__":
    from src.ingest import run_ingestion

    chunks = run_ingestion()
    build_vectorstore(chunks)
