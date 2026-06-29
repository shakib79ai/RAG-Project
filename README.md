# 📚 Agentic RAG Pipeline — Hugging Face Edition

A complete, production-style Retrieval-Augmented Generation (RAG) pipeline built
**entirely on open-source / Hugging Face models** — no OpenAI dependency. Features
hybrid retrieval, local cross-encoder reranking, a router + self-correction agentic
loop, RAGAS evaluation, and a Streamlit chat UI.

## Pipeline Animation

![Agentic RAG Pipeline](pipeline_animation.gif)

## 🏗️ Architecture

```
PDF(s)
  │
  ▼
[Ingestion]  →  Recursive chunking (800 tokens, 120 overlap)
  │
  ▼
[Embedding]  →  BAAI/bge-small-en-v1.5 (sentence-transformers, runs locally on CPU)
  │              →  Chroma (persisted vector store)
  ▼
[Hybrid Retrieval]  →  Dense (vector similarity) + Sparse (BM25 keyword)
  │                     weighted score fusion (60% dense / 40% sparse)
  ▼
[Reranking]  →  cross-encoder/ms-marco-MiniLM-L-6-v2 (local, no API call)
  │
  ▼
[Agentic Loop]  →  1. Router: does this need retrieval at all?
  │                 2. Sufficiency check: is retrieved context enough?
  │                    if not, rewrite query and retrieve again (max 1 retry)
  │                 3. Generate grounded answer with source citations
  ▼
[LLM]  →  Hugging Face Inference API (Qwen2.5-7B-Instruct-Turbo) or local transformers
  │        (e.g. Phi-3-mini) — switchable via .env
  ▼
[Streamlit UI]  →  Chat interface + document upload + live indexing + source viewer
```

## ✨ Key Features

- **100% open-source stack**: embeddings, reranker, and LLM can all run without
  any proprietary API (HF Inference API is optional/free-tier friendly; local mode
  needs no internet after first download).
- **Hybrid Search**: Dense (semantic) + BM25 (exact keyword match) — outperforms
  either method alone, especially on technical terms and named entities.
- **Local Cross-Encoder Reranking**: Re-scores retrieved candidates using the full
  query+document pair (not just independent embeddings), improving precision@k —
  with zero added API cost.
- **Explicit Agentic Loop**: Since most open-source instruct models don't reliably
  support native function/tool calling, this implements a transparent
  router → retrieve → self-correct → generate loop using plain prompting, instead
  of a black-box tool-calling agent.
- **Switchable LLM backend**: Toggle between Hugging Face Inference API (fast, no
  GPU needed) and fully local inference (`transformers` pipeline) via one config flag.
- **Evaluation Harness**: RAGAS metrics (faithfulness, answer relevancy, context
  precision, context recall) — using the same HF LLM/embeddings as the pipeline
  itself, not OpenAI.
- **Source-grounded answers**: Every answer cites the source file, page, and chunk
  ID it was generated from.

## 📂 Project Structure

```
rag-hf-project/
├── app.py                      # Streamlit chat UI
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py               # central configuration
│   ├── ingest.py                # PDF loading + chunking
│   ├── vectorstore.py           # local HF embeddings + Chroma persistence
│   ├── retriever.py             # hybrid (dense + BM25) retrieval
│   ├── reranker.py              # local cross-encoder reranking
│   ├── llm.py                   # HF Inference API / local transformers LLM wrapper
│   └── agentic_rag.py           # router + self-correction agentic loop
├── evaluation/
│   ├── evaluate.py              # RAGAS evaluation harness (HF-backed)
│   └── test_questions.json      # question/ground-truth test set
└── data/
    ├── pdfs/                    # put your source PDFs here
    └── vectorstore/             # persisted Chroma index (auto-generated)
```

## 🚀 Setup

```bash
git clone https://github.com/shakib79ai/RAG-Huggingface-Project.git
cd RAG-Huggingface-Project
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # fill in HUGGINGFACEHUB_API_TOKEN (free at huggingface.co/settings/tokens)
```

No OpenAI key required anywhere in this project.

## ▶️ Usage

**Option 1 — CLI ingestion + scripted run:**
```bash
python -m src.ingest          # sanity-check chunking
python -m src.vectorstore     # build + persist the local vector index
python -m src.agentic_rag     # quick test query from the terminal
```

**Option 2 — Streamlit app (recommended):**
```bash
streamlit run app.py
```
Then upload a PDF from the sidebar, click "Ingest & Index", and start chatting.

**Run evaluation:**
```bash
# First edit evaluation/test_questions.json with real Q&A pairs from your PDF
python -m evaluation.evaluate
```

## ⚙️ LLM Modes

Set in `.env`:

| `LLM_MODE` | Where it runs | Needs | Best for |
|---|---|---|---|
| `api` | Hugging Face Inference API | HF token, internet | Quick setup, no GPU |
| `local` | Your own machine via `transformers` | GPU recommended (CPU works for small models) | Offline use, no rate limits, full data privacy |

## 🧠 Design Decisions & Trade-offs

| Decision | Why |
|---|---|
| BAAI/bge-small-en-v1.5 for embeddings | Strong open-source MTEB benchmark score at a small (~130MB) footprint, runs on CPU |
| Local cross-encoder reranker over an API-based one | Keeps the whole pipeline free of paid dependencies while still meaningfully improving precision |
| Custom router/self-check loop instead of LangChain tool-calling agent | Most open-source instruct models lack robust function-calling support; explicit prompting is more reliable and debuggable |
| Hybrid (dense+BM25) over pure dense | Dense embeddings miss exact technical terms/codes; BM25 catches them |
| RAGAS with HF LLM/embeddings as judge | Keeps evaluation consistent with the actual pipeline backend rather than silently depending on OpenAI |

## 🔮 Possible Extensions

- Swap Qwen2.5-7B-Instruct-Turbo for a smaller/larger model depending on latency vs. quality needs
- Add 4-bit quantization (`bitsandbytes`) for running larger local models on consumer GPUs
- Swap Chroma for FAISS or Qdrant for larger-scale deployments
- Add conversation memory for multi-turn follow-up questions
- Deploy the Streamlit app + local LLM together in a single Docker container for fully offline use

## 🛠️ Tech Stack

`Python` · `LangChain` · `Hugging Face Transformers` · `sentence-transformers` ·
`ChromaDB` · `BM25` · `RAGAS` · `Streamlit`

---

*Built as a hands-on exploration of production RAG patterns using a fully
open-source stack: hybrid retrieval, local reranking, an explicit agentic
loop, and rigorous evaluation — without relying on any closed-source API.*
