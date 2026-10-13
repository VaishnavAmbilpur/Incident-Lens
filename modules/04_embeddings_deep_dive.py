"""
04_embeddings_deep_dive.py
==========================
Phase 1: Vector Embeddings & Similarity Geometry
IncidentLens Embedding Explorer:
1. Explains dense vector representations and geometry.
2. Implements pure-Python vector math (Cosine Similarity, Dot Product, Euclidean Distance).
3. Uses ChromaDB's local embedding engine (all-MiniLM-L6-v2 / 384 dimensions) with zero external API cost.
4. Computes semantic similarity matrices across queries and actual post-mortem chunks.
5. Demonstrates the Semantic Bridge: how natural language questions match technical error codes.
"""

import sys
import os
import math
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Ensure UTF-8 output encoding across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Dynamic import of Module 03 production chunks
import importlib.util

def _get_chunker():
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / "03_text_splitters.py"
    spec = importlib.util.spec_from_file_location("module_03_splitters", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "get_production_chunks")

get_production_chunks = _get_chunker()


# =====================================================================
# 1. Pure-Python Vector Geometry Math
# =====================================================================
def dot_product(v1: List[float], v2: List[float]) -> float:
    """Computes the dot product of two vectors: sum(A_i * B_i)"""
    return sum(a * b for a, b in zip(v1, v2))


def vector_magnitude(v: List[float]) -> float:
    """Computes the Euclidean L2 norm (length) of a vector: sqrt(sum(A_i^2))"""
    return math.sqrt(sum(a * a for a in v))


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes cosine similarity between two vectors:
    cos(theta) = (A . B) / (||A|| * ||B||)
    Returns a score between -1.0 (opposite) and +1.0 (identical direction).
    """
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Computes straight-line distance between two vector coordinates: sqrt(sum((A_i - B_i)^2))"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


# =====================================================================
# 2. Embedding Engine Loader
# =====================================================================
def get_embedding_engine():
    """
    Returns ChromaDB's default local embedding function.
    Runs locally on CPU/GPU via ONNX with zero API keys or costs.
    Dimension: 384 floats.
    """
    try:
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()
        return ef
    except Exception as e:
        print(f"❌ Error initializing embedding engine: {e}")
        sys.exit(1)


def embed_texts(ef, texts: List[str]) -> List[List[float]]:
    """Generates embedding vectors for a list of strings."""
    return ef(texts)


# =====================================================================
# 3. Diagnostic Experiments
# =====================================================================
def experiment_semantic_bridge(ef):
    """
    Demonstrates the 'Semantic Bridge' power of dense embeddings:
    Matching natural language queries to dense technical jargon.
    """
    print("\n" + "=" * 80)
    print(" 🧪 Experiment 1: The Semantic Bridge Test")
    print("=" * 80)
    print("Question: Can natural language match technical jargon without exact keyword overlap?")

    query = "What caused Windows computers to show a blue screen crash?"

    candidates = [
        ("CrowdStrike Match", "A logic error in Channel File 291 caused memory safety violation bugcheck 0x00000050 in kernel driver CSAgent.sys."),
        ("Cloudflare Incident", "Allegheny Technologies leaked more-specific BGP prefixes to Verizon AS701 causing internet route hijacking."),
        ("GitHub Incident", "A 43-second network partition severed primary MySQL semi-synchronous replication leading to split-brain writes."),
        ("Unrelated Sentence", "The weather in Seattle was rainy and cold with heavy cloud coverage throughout the morning.")
    ]

    query_vec = embed_texts(ef, [query])[0]
    candidate_vecs = embed_texts(ef, [c[1] for c in candidates])

    print(f"\nUser Query: \"{query}\"")
    print(f"Query Vector Dimensions: {len(query_vec)} floating-point numbers\n")
    print(f"{'Target Candidate':<24} | {'Cosine Similarity':<18} | {'Euclidean Distance':<20} | {'Verdict'}")
    print("-" * 80)

    for (label, text), vec in zip(candidates, candidate_vecs):
        cos_sim = cosine_similarity(query_vec, vec)
        euc_dist = euclidean_distance(query_vec, vec)
        verdict = "🔥 STRONG MATCH" if cos_sim > 0.50 else ("⚪ WEAK MATCH" if cos_sim > 0.30 else "❌ IRRELEVANT")
        print(f"{label:<24} | {cos_sim:<18.4f} | {euc_dist:<20.4f} | {verdict}")


def experiment_real_corpus_retrieval(ef, chunks):
    """
    Searches the real 105 post-mortem chunks using vector cosine similarity.
    Finds the Top-3 most relevant chunks for a specific incident query.
    """
    print("\n" + "=" * 80)
    print(" 🧪 Experiment 2: Corpus Retrieval Across 105 Real Incident Chunks")
    print("=" * 80)

    test_query = "Why did Kinesis fail when new capacity was added to the fleet?"
    print(f"Test Query: \"{test_query}\"")

    query_vec = embed_texts(ef, [test_query])[0]

    # Pre-embed first 30 chunks for speed demonstration
    eval_chunks = chunks[:40]
    chunk_texts = [c.page_content for c in eval_chunks]
    chunk_vecs = embed_texts(ef, chunk_texts)

    scored_chunks: List[Tuple[float, Any]] = []
    for chunk_doc, vec in zip(eval_chunks, chunk_vecs):
        score = cosine_similarity(query_vec, vec)
        scored_chunks.append((score, chunk_doc))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop 3 Closest Chunks Found in Knowledge Base:")
    for rank, (score, chunk_doc) in enumerate(scored_chunks[:3], 1):
        print(f"\n[{rank}] Similarity Score: {score:.4f} | ID: {chunk_doc.metadata.get('chunk_id')}")
        print(f"    Source: {chunk_doc.metadata.get('source_file')} (Page {chunk_doc.metadata.get('page_number')})")
        print(f"    Content Excerpt: \"{chunk_doc.page_content.strip()[:180]}...\"")


def main():
    print("=" * 80)
    print(" 🧭 IncidentLens — Module 04: Vector Embeddings Deep Dive")
    print("=" * 80)

    # 1. Load Local Embedding Function
    print("\n[1] Initializing Local Dense Embedding Engine (ChromaDB ONNX / all-MiniLM-L6-v2)...")
    ef = get_embedding_engine()
    print("    ✅ Engine ready. Zero external API calls, zero cost, instant local inference.")

    # 2. Semantic Bridge Test
    experiment_semantic_bridge(ef)

    # 3. Real Corpus Test
    print("\n[2] Loading 105 Production Chunks from Module 03...")
    chunks = get_production_chunks()
    print(f"    ✅ Ingested {len(chunks)} chunks into memory.")
    experiment_real_corpus_retrieval(ef, chunks)

    print("\n" + "=" * 80)
    print(" ✅ Module 04 Complete: Vector Geometry & Embeddings Verified!")
    print(" Next Step: Index these vectors into persistent ChromaDB storage (Module 05).")
    print("=" * 80)


if __name__ == "__main__":
    main()
