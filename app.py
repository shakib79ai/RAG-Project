"""
Streamlit front-end for the Hugging Face Agentic RAG pipeline.

Run with:
    streamlit run app.py

Features:
  - Upload PDFs directly from the browser (re-ingest + re-index on demand)
  - Chat interface backed by the agentic RAG loop
  - Shows exactly which source chunks were used for each answer
"""
import os
import streamlit as st

from src import config
from src.ingest import load_pdfs, chunk_documents
from src.vectorstore import build_vectorstore, load_vectorstore
from src.retriever import HybridRetriever, build_bm25_retriever
from src.agentic_rag import build_agentic_rag

st.set_page_config(page_title="HF Agentic RAG", page_icon="📚", layout="wide")
st.title("📚 Agentic RAG — PDF Q&A (Hugging Face Edition)")
st.caption(
    f"sentence-transformers embeddings + BM25 hybrid search + 2cross-encoder reranking "
    f"+ {config.LLM_MODE.upper()} LLM ({config.LLM_MODEL_API if config.LLM_MODE=='api' else config.LLM_MODEL_LOCAL})"
)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

# ---------- Sidebar: ingestion ----------
with st.sidebar:
    st.header("⚙️ Document Setup")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("📥 Ingest & Index", use_container_width=True):
        os.makedirs(config.PDF_DIR, exist_ok=True)
        for f in uploaded_files:
            save_path = os.path.join(config.PDF_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())

        with st.spinner("Reading and chunking PDF(s)..."):
            docs = load_pdfs(config.PDF_DIR)
            chunks = chunk_documents(docs)

        with st.spinner("Generating local embeddings and saving to vector store..."):
            vectorstore = build_vectorstore(chunks)

        with st.spinner("Building hybrid retriever and agent (first run may download models)..."):
            bm25 = build_bm25_retriever(chunks)
            hybrid = HybridRetriever(vectorstore, bm25)
            st.session_state.agent = build_agentic_rag(hybrid)

        st.success(f"✅ Indexed {len(chunks)} chunks. You can ask questions now.")

    if st.button("🔁 Load Existing Vector Store", use_container_width=True):
        try:
            with st.spinner("Loading existing vector store..."):
                docs = load_pdfs(config.PDF_DIR)
                chunks = chunk_documents(docs)
                vectorstore = load_vectorstore()
                bm25 = build_bm25_retriever(chunks)
                hybrid = HybridRetriever(vectorstore, bm25)
                st.session_state.agent = build_agentic_rag(hybrid)
            st.success("✅ Vector store loaded.")
        except Exception as e:
            st.error(f"Failed to load vector store: {e}")

    st.divider()
    st.markdown(
        "**Pipeline:**\nPDF → Chunk → Hybrid Retrieval (Dense+BM25) → "
        "Cross-Encoder Rerank → Router/Self-Check Agent → HF LLM"
    )
    st.markdown(f"**LLM mode:** `{config.LLM_MODE}`  \n**Embedding model:** `{config.EMBEDDING_MODEL}`")

# ---------- Chat ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question about your document...")

if user_query:
    if st.session_state.agent is None:
        st.warning("⚠️ Please upload and index a PDF from the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.agent.answer(user_query)
                st.markdown(answer)
                if sources:
                    with st.expander("📎 Sources used"):
                        for doc in sources:
                            st.markdown(
                                f"**{doc.metadata.get('source_file', 'unknown')}** "
                                f"(page {doc.metadata.get('page', '?')}, "
                                f"{doc.metadata.get('chunk_id', '?')})"
                            )
                            st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
