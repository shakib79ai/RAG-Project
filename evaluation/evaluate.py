"""
Evaluation harness using RAGAS, configured to use the same local Hugging
Face embedding model and HF LLM as the rest of the pipeline (RAGAS supports
plugging in any LangChain-compatible LLM/embeddings, not just OpenAI).

Measures:
  - Retrieval quality: context_precision, context_recall
  - Generation quality: faithfulness, answer_relevancy

Usage:
    Fill out evaluation/test_questions.json with question/ground_truth pairs,
    then run: python -m evaluation.evaluate
"""
import json
import os
from typing import List, Dict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from src.vectorstore import load_vectorstore, get_embedding_model
from src.retriever import HybridRetriever, build_bm25_retriever
from src.reranker import rerank
from src.ingest import run_ingestion
from src.agentic_rag import build_agentic_rag
from src.llm import get_llm


def load_test_set(path: str = "evaluation/test_questions.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(test_set_path: str = "evaluation/test_questions.json"):
    test_cases = load_test_set(test_set_path)

    chunks = run_ingestion()
    vectorstore = load_vectorstore()
    bm25 = build_bm25_retriever(chunks)
    hybrid = HybridRetriever(vectorstore, bm25)
    agent = build_agentic_rag(hybrid)

    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for case in test_cases:
        question = case["question"]
        ground_truth = case["ground_truth"]

        answer, top_chunks = agent.answer(question)
        contexts = [c.page_content for c in top_chunks] or [""]

        records["question"].append(question)
        records["answer"].append(answer)
        records["contexts"].append(contexts)
        records["ground_truth"].append(ground_truth)

    dataset = Dataset.from_dict(records)

    # RAGAS needs an LLM (for judging) and embeddings - reuse the same HF
    # backends as the main pipeline instead of defaulting to OpenAI.
    judge_llm = get_llm()
    judge_embeddings = get_embedding_model()

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=2, timeout=120),
    )

    df = results.to_pandas()
    os.makedirs("evaluation/results", exist_ok=True)
    df.to_csv("evaluation/results/ragas_results.csv", index=False)

    print("\n=== RAGAS Evaluation Summary ===")
    print(results)
    print("\nFull results saved to evaluation/results/ragas_results.csv")
    return results


if __name__ == "__main__":
    run_evaluation()
