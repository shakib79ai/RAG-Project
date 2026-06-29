"""
Hybrid retriever: combines dense (embedding) search with sparse (BM25)
keyword search, then merges the two ranked lists.

Why hybrid: dense embeddings are great at semantic similarity but often
miss exact technical terms (product codes, names, acronyms). BM25 is
the opposite. Combining both consistently beats either alone in practice.
"""
from typing import List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src import config


def build_bm25_retriever(chunks: List[Document]) -> BM25Retriever:
    """BM25 needs the raw chunk text in memory (it's not stored in Chroma)."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = config.SPARSE_TOP_K
    return retriever


def _normalize_scores(docs_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """Min-max normalize so dense and sparse scores are comparable before merging."""
    if not docs_with_scores:
        return []
    scores = [s for _, s in docs_with_scores]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(d, 1.0) for d, _ in docs_with_scores]
    return [(d, (s - lo) / (hi - lo)) for d, s in docs_with_scores]


class HybridRetriever:
    """
    Fuses dense vector search and BM25 sparse search using weighted score
    combination (a simplified, transparent alternative to RRF fusion).
    """

    def __init__(self, vectorstore: Chroma, bm25_retriever: BM25Retriever):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, top_k: int = config.FINAL_TOP_K * 3) -> List[Document]:
        # Dense search returns (doc, distance) - Chroma uses distance, so lower = more similar.
        dense_results = self.vectorstore.similarity_search_with_score(
            query, k=config.DENSE_TOP_K
        )
        dense_results = [(doc, -score) for doc, score in dense_results]
        dense_results = _normalize_scores(dense_results)

        sparse_docs = self.bm25_retriever.invoke(query)
        sparse_results = [
            (doc, 1.0 - (i / max(len(sparse_docs), 1)))
            for i, doc in enumerate(sparse_docs)
        ]

        combined_scores = {}
        for doc, score in dense_results:
            key = doc.metadata.get("chunk_id", doc.page_content[:50])
            combined_scores[key] = {
                "doc": doc,
                "score": config.HYBRID_DENSE_WEIGHT * score,
            }

        for doc, score in sparse_results:
            key = doc.metadata.get("chunk_id", doc.page_content[:50])
            if key in combined_scores:
                combined_scores[key]["score"] += config.HYBRID_SPARSE_WEIGHT * score
            else:
                combined_scores[key] = {
                    "doc": doc,
                    "score": config.HYBRID_SPARSE_WEIGHT * score,
                }

        ranked = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked[:top_k]]
