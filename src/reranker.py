"""
Reranking step: takes the wider candidate set from hybrid retrieval and
re-scores it with a local cross-encoder reranker from sentence-transformers.
A cross-encoder looks at the query and each candidate document together
(rather than comparing independent embeddings), which consistently
improves precision@k over raw vector similarity.

Runs fully locally - no API key, no extra cost.
"""
from typing import List
from functools import lru_cache

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from src import config


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Cached so the model is loaded into memory only once per process."""
    return CrossEncoder(config.RERANKER_MODEL)


def rerank(query: str, candidates: List[Document], top_k: int = config.FINAL_TOP_K) -> List[Document]:
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored[:top_k]]
