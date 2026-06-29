"""
Central configuration for the Hugging Face RAG pipeline.
Loads environment variables and exposes them as constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./data/vectorstore")
PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")

# Embedding + reranker run locally via sentence-transformers (no API cost)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM mode: "api" (Hugging Face Inference API) or "local" (transformers pipeline)
LLM_MODE = os.getenv("LLM_MODE", "api")
# api-inference.huggingface.co is deprecated; route through router.huggingface.co/together
LLM_MODEL_API = os.getenv("LLM_MODEL_API", "Qwen/Qwen2.5-7B-Instruct-Turbo")
LLM_PROVIDER_URL = os.getenv("LLM_PROVIDER_URL", "https://router.huggingface.co/together/v1")
LLM_MODEL_LOCAL = os.getenv("LLM_MODEL_LOCAL", "microsoft/Phi-3-mini-4k-instruct")

# Chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Retrieval config
DENSE_TOP_K = 15         # candidates from vector search
SPARSE_TOP_K = 15        # candidates from BM25
FINAL_TOP_K = 5          # after reranking, how many chunks go to the LLM
HYBRID_DENSE_WEIGHT = 0.6
HYBRID_SPARSE_WEIGHT = 0.4

COLLECTION_NAME = "rag_documents_hf"
