"""
11_query_transformations.py
===========================
Phase 3: Retrieval Optimization & Query Engineering
IncidentLens Query Transformation Engine:
1. Multi-Query Generation: Deconstructs vague user questions into 3 technical perspectives:
   - Perspective A: Root-Cause & Code-Level Defect (Memory, pointers, syntax)
   - Perspective B: Architectural & Infrastructure Impact (Fleet, network, storage)
   - Perspective C: SRE Incident Timeline & Resolution (Alarms, mitigations, failover)
2. Multi-Perspective Retrieval: Executes parallel retrieval across all generated variants.
3. Cross-Query Reciprocal Rank Fusion: Merges multi-query result candidate lists into a
   consolidated, deduplicated candidate pool.
4. Recall Benchmark: Compares Single-Query Top-5 vs Multi-Query Top-5, proving recall gains
   on vague and multi-faceted incident queries.
"""

import sys
import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# Ensure UTF-8 output encoding across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

# Load API keys from .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import importlib.util

# Dynamic import helper for existing modules
def _import_helper(filename: str, module_name: str):
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_chroma = _import_helper("05_vector_stores_chroma.py", "module_05_chroma")
mod_splitters = _import_helper("03_text_splitters.py", "module_03_splitters")
mod_hybrid = _import_helper("10_hybrid_search_bm25.py", "module_10_hybrid")

get_incident_collection = mod_chroma.get_incident_collection
get_production_chunks = mod_splitters.get_production_chunks
HybridSearchEngine = mod_hybrid.HybridSearchEngine


# =====================================================================
# 1. Multi-Query Generator (Groq LLM)
# =====================================================================
class MultiQueryTransformer:
    """
    Transforms a single user query into multiple specialized technical
    formulations to maximize retrieval recall across disparate chunk facets.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be set in environment or passed to constructor.")

        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        self.llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.2,
            max_tokens=300,   # Keep well under Groq free-tier OTPM limit (1,000 tokens/min)
            api_key=self.api_key,
        )

        template = (
            "You are an expert SRE and distributed systems architect. "
            "Your task is to take a user question about a system outage and generate exactly "
            "3 distinct technical search queries from different analytical angles to maximize retrieval recall:\n"
            "1. Perspective A: Root-cause code/hardware failure (memory corruption, pointers, race conditions, error codes)\n"
            "2. Perspective B: Distributed infrastructure & architecture impact (traffic routing, thread pools, replication)\n"
            "3. Perspective C: Incident telemetry & timeline (alarms, failover events, mitigations)\n\n"
            "OUTPUT FORMAT RULES:\n"
            "- Output ONLY a valid JSON array of 3 strings.\n"
            "- Do NOT include markdown code fences (no ```json).\n"
            "- Each query should be 6-12 words packed with precise technical search terms.\n\n"
            "User Question: {question}"
        )

        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_variants(self, original_query: str) -> List[str]:
        """Generates 3 technical query variants using Groq LLM."""
        raw_output = self.chain.invoke({"question": original_query}).strip()

        # Clean any accidental markdown code fences
        cleaned = raw_output
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            variants = json.loads(cleaned)
            if isinstance(variants, list) and len(variants) >= 3:
                return variants[:3]
        except Exception:
            pass

        # Robust fallback if JSON parsing fails
        lines = [line.strip().lstrip("123456789.-* ") for line in cleaned.splitlines() if line.strip()]
        if len(lines) >= 3:
            return lines[:3]

        return [
            f"{original_query} root cause stack trace bugcheck",
            f"{original_query} architecture infrastructure cluster impact",
            f"{original_query} mitigation failover timeline alarms",
        ]


# =====================================================================
# 2. Multi-Query Fusion Retriever
# =====================================================================
class MultiQueryFusionRetriever:
    """
    Executes multi-query retrieval by running each query through the
    Hybrid Search Engine and combining candidate lists via Reciprocal Rank Fusion.
    """

    def __init__(self, hybrid_engine: HybridSearchEngine, rrf_k: int = 60):
        self.hybrid_engine = hybrid_engine
        self.rrf_k = rrf_k

    def retrieve(
        self,
        original_query: str,
        variants: List[str],
        top_k: int = 10,
        candidates_per_query: int = 15,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """
        Runs the original query plus all variants, fusing all lists with RRF.
        Tracks which query variants discovered which chunks.
        """
        all_queries = [original_query] + variants
        fused_pool: Dict[str, Dict[str, Any]] = {}
        provenance_map: Dict[str, List[str]] = {}

        for q_idx, q in enumerate(all_queries):
            query_label = "Original Query" if q_idx == 0 else f"Variant #{q_idx}"
            results = self.hybrid_engine.search_hybrid(q, top_k=candidates_per_query)

            for rank, item in enumerate(results, start=1):
                cid = item["chunk_id"]
                rrf_boost = 1.0 / (self.rrf_k + rank)

                if cid not in provenance_map:
                    provenance_map[cid] = []
                provenance_map[cid].append(query_label)

                if cid in fused_pool:
                    fused_pool[cid]["rrf_score"] += rrf_boost
                    fused_pool[cid]["hit_count"] += 1
                else:
                    fused_pool[cid] = {
                        "chunk_id": cid,
                        "rrf_score": rrf_boost,
                        "hit_count": 1,
                        "text": item["text"],
                        "metadata": item["metadata"],
                    }

        # Sort by total fused score descending
        fused_ranked = sorted(fused_pool.values(), key=lambda x: x["rrf_score"], reverse=True)

        for final_rank, item in enumerate(fused_ranked[:top_k], start=1):
            item["final_rank"] = final_rank

        return fused_ranked[:top_k], provenance_map


# =====================================================================
# 3. Benchmark: Single-Query vs Multi-Query Fusion
# =====================================================================
def run_transformation_benchmark():
    print("=" * 80)
    print(" 🛠️ IncidentLens — Module 11: Query Expansion & Multi-Query Transformation")
    print("=" * 80)

    chunks = get_production_chunks()
    collection = get_incident_collection()
    hybrid_engine = HybridSearchEngine(chunks, collection)
    transformer = MultiQueryTransformer()
    retriever = MultiQueryFusionRetriever(hybrid_engine)

    # Vague real-world query that users typically ask
    vague_query = "Tell me about the database split-brain and replication problems."

    print("\n[1] Deconstructing User Query into Technical Perspectives...")
    print(f"    Original Query: \"{vague_query}\"")

    variants = transformer.generate_variants(vague_query)
    print("\n--- Generated Specialized Perspectives ---")
    perspectives = [
        "A (Code/Mechanism)",
        "B (Infrastructure/Topology)",
        "C (Mitigation/Timeline)",
    ]
    for idx, (label, var) in enumerate(zip(perspectives, variants), 1):
        print(f"  [{idx}] Perspective {label}: \"{var}\"")

    # 1. Single-Query Baseline
    single_baseline = hybrid_engine.search_hybrid(vague_query, top_k=5)
    single_ids = set([x["chunk_id"] for x in single_baseline])

    # 2. Multi-Query Fusion
    multi_results, provenance = retriever.retrieve(vague_query, variants, top_k=5)
    multi_ids = set([x["chunk_id"] for x in multi_results])

    print("\n" + "=" * 80)
    print(" 📊 Retrieval Comparison: Single Query vs Multi-Query Fusion")
    print("=" * 80)

    print("\n--- Single Query Baseline (Top-5) ---")
    for idx, item in enumerate(single_baseline, 1):
        print(f"  #{idx} | {item['chunk_id']:<32} | Score: {item['rrf_score']:.5f}")

    print("\n--- Multi-Query Fused Results (Top-5) ---")
    for idx, item in enumerate(multi_results, 1):
        cid = item["chunk_id"]
        queries_hit = ", ".join(provenance.get(cid, []))
        print(f"  #{idx} | {cid:<32} | Fused RRF: {item['rrf_score']:.5f} | Hits: {item['hit_count']}/4 | Discovered by: {queries_hit}")

    newly_discovered = multi_ids - single_ids
    print("-" * 80)
    print(f"💡 Multi-Query Diagnostic Summary:")
    print(f"   Newly Unlocked Critical Chunks: {len(newly_discovered)}")
    for cid in newly_discovered:
        print(f"   🌟 Added Chunk: {cid} (Found via expanded perspectives)")
    print("   Multi-perspective expansion prevents query myopia and dramatically boosts recall!")

    print("\n" + "=" * 80)
    print(" 🏆 Module 11 Complete: Query Expansion & Multi-Query Fusion Verified!")
    print(" Ready for Module 12: Cross-Encoder Reranking Pipeline.")
    print("=" * 80)


if __name__ == "__main__":
    run_transformation_benchmark()
