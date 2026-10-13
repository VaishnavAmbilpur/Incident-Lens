"""
07_rag_failure_modes_demo.py
============================
Phase 2: Diagnosing & Debugging the 5 Failure Modes of Naive RAG
IncidentLens Stress-Testing Suite:
This script deliberately exposes and proves the 5 classical failure modes
that break naive RAG in production infrastructure and SRE environments:

1. Failure Mode 1: Bad Chunking (Severed Stack Traces & Error Codes)
2. Failure Mode 2: Semantic Mismatch (Alphanumeric IDs vs Natural Language)
3. Failure Mode 3: Retrieval Noise (Top-10 Cosine Fluff without Reranking)
4. Failure Mode 4: Lost in the Middle (Context Degradation in Long Prompts)
5. Failure Mode 5: Hallucination & Fake Remediation (Dangerous Inventions)
"""

import sys
import os
import warnings
from pathlib import Path
from typing import List, Dict, Any
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
from langchain_core.output_parsers import StrOutputParser

# Dynamic import of Module 05 vector store and Module 06 LLM
import importlib.util

def _import_helper(filename: str, module_name: str):
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_chroma = _import_helper("05_vector_stores_chroma.py", "module_05_chroma")
mod_rag = _import_helper("06_basic_rag_pipeline.py", "module_06_rag")

get_incident_collection = mod_chroma.get_incident_collection
get_persistent_client = mod_chroma.get_persistent_client
get_llm = mod_rag.get_llm


# =====================================================================
# 1. Failure Mode 1: Bad Chunking (Severed Technical Boundaries)
# =====================================================================
def test_failure_1_bad_chunking():
    print("\n" + "=" * 80)
    print(" 💥 Failure Mode 1: Bad Chunking (Severed Stack Traces)")
    print("=" * 80)
    print("Scenario: A naive chunker with a small chunk size and 0 overlap slices")
    print("          a technical stack trace mid-identifier.\n")

    stack_trace_text = (
        "CRASH LOG: Kernel Driver CSAgent.sys triggered PAGE_FAULT_IN_NONPAGED_AREA (0x00000050).\n"
        "Memory Address: 0xfffff8020ebc14ed referenced unmapped page.\n"
        "Failure occurred in Content Interpreter evaluating Channel File 291."
    )

    from langchain_text_splitters import CharacterTextSplitter
    bad_splitter = CharacterTextSplitter(chunk_size=75, chunk_overlap=0, separator=" ")
    broken_chunks = bad_splitter.split_text(stack_trace_text)

    print("--- Naive Sliced Chunks (0 Overlap) ---")
    for idx, c in enumerate(broken_chunks, 1):
        print(f"[Chunk {idx}] (len={len(c)}): \"{c}\"")

    print("\n🚨 SRE Diagnostic Verdict:")
    print("   Notice how 'PAGE_FAULT_IN_NONPAGED_AREA' and '0x00000050' or 'Channel File 291'")
    print("   get isolated into separate chunks! An embedding model retrieving Chunk 2 will")
    print("   completely miss which kernel driver (CSAgent.sys) caused the fault.")


# =====================================================================
# 2. Failure Mode 2: Semantic Mismatch (Alphanumeric IDs vs Vectors)
# =====================================================================
def test_failure_2_semantic_mismatch(collection):
    print("\n" + "=" * 80)
    print(" 💥 Failure Mode 2: Semantic Mismatch (Dense Vectors vs Exact Codes)")
    print("=" * 80)
    print("Scenario: An on-call engineer queries using an exact autonomous system number (ASN)")
    print("          without descriptive conversational context.\n")

    query = "AS33154"
    print(f"Query: \"{query}\" (Allegheny Technologies Autonomous System Number)")

    results = collection.query(query_texts=[query], n_results=3)

    print("\nTop 3 Chunks Retrieved by Pure Dense Embeddings:")
    for rank, (cid, doc, dist) in enumerate(zip(results["ids"][0], results["documents"][0], results["distances"][0]), 1):
        sim = 1.0 - dist
        has_code = "33154" in doc
        print(f"\n[Rank {rank}] Similarity: {sim:.4f} | Has 'AS33154' in text? {'✅ YES' if has_code else '❌ NO'}")
        print(f"    ID: {cid}")
        print(f"    Snippet: \"{doc.strip()[:140]}...\"")

    print("\n🚨 SRE Diagnostic Verdict:")
    print("   Dense embeddings represent conceptual meaning, NOT exact keyword characters.")
    print("   Dense retrieval alone frequently misses exact error codes, ASNs, and register hashes.")
    print("   Solution needed in Phase 3: Hybrid Search (BM25 Keyword Search + Dense Vectors).")


# =====================================================================
# 3. Failure Mode 3: Retrieval Noise (Cosine Fluff without Reranking)
# =====================================================================
def test_failure_3_retrieval_noise(collection):
    print("\n" + "=" * 80)
    print(" 💥 Failure Mode 3: Retrieval Noise (Top-10 Fluff Without Reranking)")
    print("=" * 80)
    print("Scenario: Querying for the specific resolution step of the Cloudflare route leak.")
    print("          Requesting top 10 chunks from ChromaDB.\n")

    query = "What telephone call or manual communication resolved the Cloudflare route leak?"
    results = collection.query(query_texts=[query], n_results=10)

    docs = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

    noise_count = 0
    relevant_count = 0

    print(f"Query: \"{query}\"\n")
    print(f"{'Rank':<6} | {'Similarity':<12} | {'Contains Actual Resolution?':<30} | {'Chunk ID'}")
    print("-" * 80)

    for rank, (cid, doc, dist) in enumerate(zip(ids, docs, distances), 1):
        sim = 1.0 - dist
        is_relevant = "DQE" in doc and ("phone" in doc.lower() or "reach" in doc.lower() or "call" in doc.lower())
        if is_relevant:
            relevant_count += 1
            status = "🎯 RELEVANT (Mentions Resolution)"
        else:
            noise_count += 1
            status = "⚪ NOISE (Generic Background/BGP)"

        print(f"#{rank:<5} | {sim:<12.4f} | {status:<30} | {cid}")

    print("-" * 80)
    print(f"Total Chunks: 10 | Relevant Chunks: {relevant_count} | Noise Chunks: {noise_count} ({noise_count*10}%)")
    print("\n🚨 SRE Diagnostic Verdict:")
    print("   Without a Cross-Encoder Reranker, 70-80% of retrieved chunks are irrelevant fluff.")
    print("   Feeding all 10 into the LLM dilutes attention and leads to vague responses.")


# =====================================================================
# 4. Failure Mode 4: "Lost in the Middle" (Context Dilution)
# =====================================================================
def test_failure_4_lost_in_the_middle(llm):
    print("\n" + "=" * 80)
    print(" 💥 Failure Mode 4: 'Lost in the Middle' Attention Degradation")
    print("=" * 80)
    print("Scenario: A critical fact (the 'needle') is buried in the exact middle")
    print("          of 10 irrelevant distraction paragraphs ('haystack').\n")

    distractor = (
        "The Border Gateway Protocol (BGP) is an exterior gateway protocol designed to exchange "
        "routing and reachability information among autonomous systems (AS) on the Internet. "
        "Routers maintain tables of IP networks to determine optimal transit paths."
    )

    needle = (
        "SPECIAL INCIDENT METRIC: The internal ticket ID for the emergency Verizon escalations "
        "was CRITICAL-TICKET-99482-BGP."
    )

    # Place the needle at index 5 (middle of 11 chunks)
    haystack_parts = [f"[Document #{i}]\n{distractor}" for i in range(1, 6)]
    haystack_parts.append(f"[Document #6 - TARGET]\n{needle}")
    haystack_parts.extend([f"[Document #{i}]\n{distractor}" for i in range(7, 12)])

    full_context = "\n\n---\n\n".join(haystack_parts)

    prompt = ChatPromptTemplate.from_template(
        "You are an SRE assistant. Read the context and answer: What is the special internal ticket ID?\n\n"
        "Context:\n{context}\n\nAnswer:"
    )
    chain = prompt | llm | StrOutputParser()

    print(f"Haystack size: {len(haystack_parts)} documents ({len(full_context)} characters).")
    print("Target Needle placed at position: Chunk #6 (Dead Center).")
    print("\nQuerying LLM...")

    response = chain.invoke({"context": full_context})
    print(f"\nLLM Response: \"{response.strip()}\"")

    if "99482" in response:
        print("\n✅ The LLM found the needle in this test, but research shows accuracy drops by up to")
        print("   40% when needles are in the middle compared to the beginning or end of prompts!")
    else:
        print("\n🚨 FAILURE CONFIRMED: The LLM missed the needle buried in the middle of the context!")


# =====================================================================
# 5. Failure Mode 5: Hallucination of Destructive Fixes
# =====================================================================
def test_failure_5_hallucination(collection, llm):
    print("\n" + "=" * 80)
    print(" 💥 Failure Mode 5: Hallucination of Non-Existent Remediation Commands")
    print("=" * 80)
    print("Scenario: An engineer asks for the CLI command to flush Verizon's BGP cache.")
    print("          The post-mortem explicitly states Cloudflare had NO access to Verizon's CLI!\n")

    trick_question = (
        "Provide the exact Juniper/Cisco terminal command that Cloudflare engineers typed "
        "to clear the BGP prefix cache on Verizon's routers."
    )
    print(f"Trick Question: \"{trick_question}\"")

    # Retrieve context from Cloudflare post-mortem
    results = collection.query(query_texts=[trick_question], n_results=3)
    context = "\n\n".join(results["documents"][0])

    # Unconstrained Prompt (simulating weak naive RAG without guardrails)
    naive_prompt = ChatPromptTemplate.from_template(
        "You are a helpful network engineer. Answer the user's question:\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer with the command:"
    )
    chain = naive_prompt | llm | StrOutputParser()

    response = chain.invoke({"context": context, "question": trick_question})

    print("\nUnconstrained Naive Model Response:")
    print("-" * 80)
    print(response.strip())
    print("-" * 80)

    print("\n🚨 SRE Diagnostic Verdict:")
    print("   Notice whether the model fabricated a CLI command (e.g. 'clear ip bgp ...')")
    print("   even though the source post-mortem states Cloudflare never touched Verizon routers!")
    print("   In production, executing hallucinated network commands can drop live backbone traffic.")
    print("   Solution needed in Phase 4: Self-Correcting Hallucination Graders & Guardrails.")


# =====================================================================
# Main Suite Execution
# =====================================================================
def main():
    print("=" * 80)
    print(" 🧪 IncidentLens — Module 07: The 5 Failure Modes of Naive RAG")
    print("=" * 80)

    client = get_persistent_client()
    collection = get_incident_collection(client)
    llm = get_llm()

    # 1. Bad Chunking
    test_failure_1_bad_chunking()

    # 2. Semantic Mismatch
    test_failure_2_semantic_mismatch(collection)

    # 3. Retrieval Noise
    test_failure_3_retrieval_noise(collection)

    # 4. Lost in the Middle
    test_failure_4_lost_in_the_middle(llm)

    # 5. Hallucination Risk
    test_failure_5_hallucination(collection, llm)

    print("\n" + "=" * 80)
    print(" 🎓 Phase 2 Failure Modes Suite Complete!")
    print(" Next Step: Build Module 08 (Debugging & Diagnostic Toolkit).")
    print("=" * 80)


if __name__ == "__main__":
    main()
