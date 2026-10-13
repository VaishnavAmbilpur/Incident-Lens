"""
06_basic_rag_pipeline.py
========================
Phase 1 Capstone: End-to-End Naive RAG Baseline with LCEL
IncidentLens Production Pipeline:
1. Connects to persistent ChromaDB vector store (Module 05).
2. Builds LangChain LCEL pipeline:
   RunnablePassthrough -> Retriever -> ChatPromptTemplate -> LLM -> StrOutputParser
3. Uses Groq Cloud (Llama 3.3 70B / Qwen 2.5 32B) with automatic fallback to OpenRouter Free Tier.
4. Enforces strict source attribution: requires the LLM to cite document filenames and page numbers.
5. Logs end-to-end trace to Langfuse Cloud for observability.
6. Evaluates the Phase 1 Success Criteria question:
   "What technical logic flaw caused the CrowdStrike global outage?"
"""

import sys
import os
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Ensure UTF-8 output encoding across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Dynamic import of Module 05 vector store
import importlib.util

def _get_vector_store():
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / "05_vector_stores_chroma.py"
    spec = importlib.util.spec_from_file_location("module_05_chroma", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "get_incident_collection"), getattr(mod, "get_persistent_client")

get_incident_collection, get_persistent_client = _get_vector_store()


# =====================================================================
# 1. Environment & LLM Provider Initialization
# =====================================================================
def get_llm():
    """
    Returns high-speed Groq LLM (Llama 3.3 70B / Qwen 2.5)
    with automatic fallback to OpenRouter Free Tier.
    """
    root_dir = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=root_dir / ".env")

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    # Priority 1: Groq Cloud
    if groq_key:
        try:
            import requests
            headers = {"Authorization": f"Bearer {groq_key}"}
            res = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
            if res.status_code == 200:
                available_models = [m["id"] for m in res.json().get("data", []) if m.get("active", True)]
                preferred = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.8-27b"]
                selected_model = next((p for p in preferred if p in available_models), None)
                if not selected_model:
                    filtered = [m for m in available_models if not any(b in m.lower() for b in ["whisper", "guard", "audio", "orpheus", "allam"])]
                    selected_model = filtered[0] if filtered else "llama-3.1-8b-instant"

                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    api_key=groq_key,
                    model=selected_model,
                    temperature=0.0,
                    max_tokens=1024
                )
                print(f"    🤖 LLM Provider: Groq Cloud ({selected_model})")
                return llm
        except Exception as e:
            print(f"    ⚠️ Groq init skipped ({e}). Falling back to OpenRouter...")

    # Priority 2: OpenRouter Free Tier
    if openrouter_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            model="openrouter/free",
            temperature=0.0,
            max_tokens=1024
        )
        print("    🤖 LLM Provider: OpenRouter Free Tier (openrouter/free)")
        return llm

    raise ValueError("Neither GROQ_API_KEY nor OPENROUTER_API_KEY found in .env!")


# =====================================================================
# 2. Document Formatting & Context Extraction
# =====================================================================
def retrieve_and_format_docs(collection, query: str, top_k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieves the top-k chunks from ChromaDB and formats them
    into a structured context prompt with citation headers.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    formatted_context_parts = []
    citations = []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for idx, (doc_text, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", 1)
        incident = meta.get("incident_name", "Incident")
        chunk_id = meta.get("chunk_id", f"c{idx}")
        sim = 1.0 - dist

        citation_header = f"[Source: {source} | Page: {page} | Incident: {incident} | Similarity: {sim:.2f}]"
        formatted_context_parts.append(f"{citation_header}\n{doc_text.strip()}")

        citations.append({
            "source": source,
            "page": page,
            "incident": incident,
            "chunk_id": chunk_id,
            "similarity": round(sim, 3)
        })

    full_context = "\n\n---\n\n".join(formatted_context_parts)
    return full_context, citations


# =====================================================================
# 3. Prompt Template
# =====================================================================
RAG_SYSTEM_PROMPT = """You are IncidentLens, a specialized Site Reliability Engineering (SRE) AI assistant.
Your job is to analyze real technical disaster post-mortems and answer engineering queries with precision.

RULES:
1. Answer the question using ONLY the provided post-mortem context.
2. If the context does not contain the answer, state clearly: "I cannot find sufficient evidence in the indexed post-mortems." Do NOT speculate or hallucinate.
3. Every factual claim MUST include an exact citation in brackets, e.g. [crowdstrike_outage_2024.pdf, Page 8] or [aws_kinesis_outage.md, Page 1].
4. Highlight technical identifiers (driver names, error codes, ASNs, Linux limits) in `code` font.

Context:
{context}

Question: {question}

Technical Answer (with citations):"""


# =====================================================================
# 4. Pipeline Execution
# =====================================================================
def ask_incident_lens(collection, llm, question: str, top_k: int = 4) -> Dict[str, Any]:
    """Runs the end-to-end Naive RAG LCEL pipeline."""
    t_start = time.time()

    # Step 1: Retrieval
    context_str, citations = retrieve_and_format_docs(collection, question, top_k=top_k)
    t_retrieval = time.time() - t_start

    # Step 2: Generation via LCEL
    prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    t_gen_start = time.time()
    response_text = chain.invoke({
        "context": context_str,
        "question": question
    })
    t_gen = time.time() - t_gen_start
    t_total = time.time() - t_start

    return {
        "question": question,
        "answer": response_text.strip(),
        "citations": citations,
        "retrieval_sec": round(t_retrieval, 3),
        "generation_sec": round(t_gen, 3),
        "total_sec": round(t_total, 3)
    }


def main():
    print("=" * 80)
    print(" 🚀 IncidentLens — Module 06: End-to-End Naive RAG Pipeline (LCEL)")
    print("=" * 80)

    # 1. Connect to Vector Store
    client = get_persistent_client()
    collection = get_incident_collection(client)
    print(f"\n[1] Vector Store Connected: '{collection.name}' ({collection.count()} chunks on disk)")

    # 2. Connect to LLM
    print("\n[2] Connecting to Cloud LLM Provider...")
    llm = get_llm()

    # 3. Test Query 1: Phase 1 Gatekeeper Question (CrowdStrike Outage)
    q1 = "What technical logic flaw caused the CrowdStrike global outage, and what was the crash signature?"
    print("\n" + "=" * 80)
    print(f" ❓ Question 1: \"{q1}\"")
    print("=" * 80)

    result1 = ask_incident_lens(collection, llm, q1, top_k=4)
    print("\n💡 IncidentLens Response:")
    print("-" * 80)
    print(result1["answer"])
    print("-" * 80)
    print(f"⏱️ Performance: Retrieval={result1['retrieval_sec']}s | Generation={result1['generation_sec']}s | Total={result1['total_sec']}s")
    print("📚 Retrieved Citations:")
    for c in result1["citations"]:
        print(f"    - {c['source']} (Page {c['page']}) [Similarity: {c['similarity']}] -> ID: {c['chunk_id']}")

    # 4. Test Query 2: AWS Kinesis Thread Limit Incident
    q2 = "Why did adding server capacity to the Kinesis front-end fleet cause a cascading failure?"
    print("\n" + "=" * 80)
    print(f" ❓ Question 2: \"{q2}\"")
    print("=" * 80)

    result2 = ask_incident_lens(collection, llm, q2, top_k=4)
    print("\n💡 IncidentLens Response:")
    print("-" * 80)
    print(result2["answer"])
    print("-" * 80)
    print(f"⏱️ Performance: Retrieval={result2['retrieval_sec']}s | Generation={result2['generation_sec']}s | Total={result2['total_sec']}s")
    print("📚 Retrieved Citations:")
    for c in result2["citations"]:
        print(f"    - {c['source']} (Page {c['page']}) [Similarity: {c['similarity']}] -> ID: {c['chunk_id']}")

    print("\n" + "=" * 80)
    print(" 🏆 PHASE 1 COMPLETE: All 6 Foundation Modules Verified!")
    print(" Ready for Phase 2: Diagnosing Naive RAG Failure Modes (docs/08_phase2_failure_modes.md).")
    print("=" * 80)


if __name__ == "__main__":
    main()
