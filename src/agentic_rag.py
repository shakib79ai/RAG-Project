"""
Agentic RAG layer for open-source / Hugging Face models.

Most open-source instruct models don't reliably support native function/tool
calling the way GPT-4-class models do, so instead of LangChain's tool-calling
agent, this module implements a transparent, explicit agent LOOP built from
plain prompting:

  1. ROUTER STEP   - ask the LLM whether this query needs document retrieval
                      at all (skips retrieval for greetings/small talk).
  2. RETRIEVE      - hybrid retrieval + reranking on the (possibly rewritten)
                      query.
  3. SUFFICIENCY CHECK (Self-RAG style) - ask the LLM if the retrieved
                      context is enough to answer. If not, it rewrites the
                      query and retrieves once more (max 2 attempts total).
  4. GENERATE      - final answer, grounded only in retrieved context, with
                      source citations.

This is more transparent/debuggable than a black-box agent and works with
any instruct-tuned HF model.
"""
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from src import config
from src.retriever import HybridRetriever
from src.reranker import rerank


ROUTER_PROMPT = """Decide whether answering the following user query requires looking up
information in a document, or whether it can be answered directly (e.g. greetings,
small talk, general knowledge unrelated to any document).

Query: "{query}"

Reply with exactly one word: RETRIEVE or DIRECT.
"""

SUFFICIENCY_PROMPT = """You are checking whether the retrieved context below is enough to
answer the user's question.

Question: {query}

Retrieved context:
{context}

If the context is sufficient, reply exactly: SUFFICIENT
If not, reply with a single improved/rephrased search query on one line, prefixed with
REWRITE: (for example: "REWRITE: pricing tiers for enterprise plan")
"""

ANSWER_PROMPT = """Answer the user's question using ONLY the context below. If the answer is
not contained in the context, say clearly: "This information was not found in the document."
Cite the source file and chunk id for any fact you use, in the format [source_file, chunk_id].

Context:
{context}

Question: {query}

Answer:
"""

DIRECT_PROMPT = """Answer the user's message directly and concisely. No document context is
needed for this.

Message: {query}

Answer:
"""


def _format_context(chunks: List[Document]) -> str:
    parts = []
    for doc in chunks:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        parts.append(f"[{source}, page {page}, {chunk_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class AgenticRAG:
    def __init__(self, llm, retriever: HybridRetriever, max_retries: int = 1):
        self.llm = llm
        self.retriever = retriever
        self.max_retries = max_retries

    def _call(self, prompt: str) -> str:
        response = self.llm.invoke([
            SystemMessage(content="You are a precise, helpful assistant."),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()

    def _needs_retrieval(self, query: str) -> bool:
        decision = self._call(ROUTER_PROMPT.format(query=query))
        return "RETRIEVE" in decision.upper()

    def _retrieve_and_rerank(self, query: str) -> List[Document]:
        candidates = self.retriever.retrieve(query)
        return rerank(query, candidates, top_k=config.FINAL_TOP_K)

    def answer(self, query: str) -> Tuple[str, List[Document]]:
        if not self._needs_retrieval(query):
            return self._call(DIRECT_PROMPT.format(query=query)), []

        current_query = query
        top_chunks: List[Document] = []

        for attempt in range(self.max_retries + 1):
            top_chunks = self._retrieve_and_rerank(current_query)
            context = _format_context(top_chunks)

            if attempt < self.max_retries:
                check = self._call(
                    SUFFICIENCY_PROMPT.format(query=query, context=context)
                )
                if check.upper().startswith("SUFFICIENT"):
                    break
                if check.upper().startswith("REWRITE:"):
                    current_query = check.split(":", 1)[1].strip()
                    continue
            break

        context = _format_context(top_chunks)
        final_answer = self._call(ANSWER_PROMPT.format(context=context, query=query))
        return final_answer, top_chunks


def build_agentic_rag(retriever: HybridRetriever) -> AgenticRAG:
    from src.llm import get_llm
    llm = get_llm()
    return AgenticRAG(llm=llm, retriever=retriever)


if __name__ == "__main__":
    from src.ingest import run_ingestion
    from src.vectorstore import load_vectorstore
    from src.retriever import build_bm25_retriever

    chunks = run_ingestion()
    vectorstore = load_vectorstore()
    bm25 = build_bm25_retriever(chunks)
    hybrid = HybridRetriever(vectorstore, bm25)

    agent = build_agentic_rag(hybrid)
    answer, sources = agent.answer("What is the main topic of this document?")
    print("\nANSWER:\n", answer)
    print("\nSOURCES USED:", [d.metadata.get("chunk_id") for d in sources])
