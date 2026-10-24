"""
Stage 8 — Contextual retrieval pipeline.

This script enriches every chunk with a context sentence, re-embeds the enriched
chunks, indexes them in a separate collection, and re-measures Recall@5 against
the labelled questions. The goal is to close the Context Recall gap:

    Baseline (dense):     0.7500
    Baseline (hybrid):    0.8571
    Contextual target:    ≥0.85  (document-level recall)

Usage
-----
    python core/contextual_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Put the project root on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.chunking import Chunk, chunk_corpus
from core.contextual import Contextualizer, document_lookup
from core.embeddings import embed_chunks, embed_query
from core.ingest import Document, load_corpus, PROJECT
from core.vectorstore import Hit, VectorIndex
from core.evaluate import (
    Question, load_questions, check_labels, evaluate_retriever,
    compare_reports, compare_by_variant,
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ENRICHED_PATH = PROJECT / "data" / "enriched_chunks.json"
CONTEXTUAL_COLLECTION = "incidentlens_contextual"
CONTEXTUAL_DB_DIR = PROJECT / "data" / "chroma_db_contextual"


# --------------------------------------------------------------------------
# Enrich all chunks (with caching)
# --------------------------------------------------------------------------
def enrich_all_chunks(
    chunks: List[Chunk],
    documents_by_id: Dict[str, Document],
    limit: Optional[int] = None,
) -> List[Tuple[Chunk, str]]:
    """
    Enrich every chunk with a context sentence.

    Results are cached to data/enriched_chunks.json so a re-run does not
    repeat the LLM calls.

    Returns
    -------
    A list of (chunk, context_sentence) pairs.
    """
    cached: Dict[str, str] = {}
    if ENRICHED_PATH.is_file():
        try:
            data = json.loads(ENRICHED_PATH.read_text(encoding="utf-8"))
            cached = data.get("contexts", {})
            print(f" cached contexts : {len(cached)}")
        except (json.JSONDecodeError, KeyError):
            cached = {}

    target = chunks if limit is None else chunks[:limit]
    to_enrich = [c for c in target if c.chunk_id not in cached]

    print(f" total chunks   : {len(target)}")
    print(f" already cached  : {len(cached)}")
    print(f" to enrich       : {len(to_enrich)}")

    if to_enrich:
        contextualizer = Contextualizer()
        enriched = contextualizer.enrich(to_enrich, documents_by_id)
        report = contextualizer.report()
        print()
        print(f" enrichment done : {report.chunks_enriched} chunks")
        print(f" total latency   : {report.total_latency_ms:,.0f} ms")
        print(f" mean per chunk  : {report.mean_latency_per_chunk_ms:.0f} ms")

        for item in enriched:
            cached[item.chunk.chunk_id] = item.context_sentence

        # Persist the cache.
        ENRICHED_PATH.write_text(
            json.dumps({"contexts": cached}, indent=2), encoding="utf-8",
        )
        print(f" cache written   : {ENRICHED_PATH}")

    # Return pairs in the original chunk order.
    pairs: List[Tuple[Chunk, str]] = []
    for chunk in target:
        context = cached.get(chunk.chunk_id, "")
        pairs.append((chunk, context))
    return pairs


# --------------------------------------------------------------------------
# Build enriched Chunk objects
# --------------------------------------------------------------------------
def build_enriched_chunks(
    pairs: List[Tuple[Chunk, str]],
) -> List[Chunk]:
    """
    Create new Chunk objects whose text is the context sentence plus the
    original chunk text. The original text is preserved in metadata so the
    generator can strip the preamble before answering.
    """
    enriched: List[Chunk] = []
    for chunk, context in pairs:
        enriched_text = f"{context} {chunk.text}" if context else chunk.text
        enriched.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                position=chunk.position,
                text=enriched_text,
                char_count=len(enriched_text),
                token_count=chunk.token_count,
                start=chunk.start,
                end=chunk.end,
                boundary=chunk.boundary,
                has_overlap=chunk.has_overlap,
                metadata=dict(chunk.metadata),
            )
        )
    return enriched


# --------------------------------------------------------------------------
# Index enriched chunks
# --------------------------------------------------------------------------
def index_enriched_chunks(
    enriched_chunks: List[Chunk],
) -> VectorIndex:
    """
    Embed the enriched chunks and index them in a separate collection.
    """
    print()
    print("=" * 78)
    print(" STAGE 3 — EMBED ENRICHED CHUNKS")
    print("=" * 78)

    vectors, stats = embed_chunks(enriched_chunks, show_progress=True)
    print(f" chunks embedded : {stats.chunks}")
    print(f" vectors          : {len(vectors)} × {stats.dimension}")
    print(f" seconds          : {stats.seconds:.1f}")
    print(f" truncated        : {stats.truncated}")

    print()
    print("=" * 78)
    print(" STAGE 4 — INDEX ENRICHED CHUNKS")
    print("=" * 78)

    index = VectorIndex(
        collection=CONTEXTUAL_COLLECTION,
        db_dir=CONTEXTUAL_DB_DIR,
    )
    index.build(enriched_chunks, vectors)
    print(f" collection      : {CONTEXTUAL_COLLECTION}")
    print(f" vectors stored  : {index.count():,}")
    print(f" build time      : {index.build_seconds:.2f} s")
    return index


# --------------------------------------------------------------------------
# Retrieve from enriched index
# --------------------------------------------------------------------------
class ContextualRetriever:
    """Dense retrieval over the contextual (enriched) index."""

    def __init__(self, index: VectorIndex) -> None:
        self.index = index

    def search(self, question: str, top_k: int = 10) -> List[Hit]:
        query_vector = embed_query(question)
        return self.index.search(query_vector, top_k=top_k)

    def as_retriever(self):
        def retriever(question: str, top_k: int = 10) -> List[str]:
            hits = self.search(question, top_k=top_k)
            # Dedupe documents, keep first occurrence.
            seen: set = set()
            ordered: List[str] = []
            for hit in hits:
                if hit.document_id and hit.document_id not in seen:
                    seen.add(hit.document_id)
                    ordered.append(hit.document_id)
            return ordered
        return retriever


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print(" STAGE 8 — CONTEXTUAL RETRIEVAL PIPELINE")
    print("=" * 78)

    # --- Load corpus ---
    documents = load_corpus()
    print(f" documents : {len(documents)}")

    chunks = chunk_corpus(documents)
    print(f" chunks    : {len(chunks)}")

    lookup = document_lookup(documents)

    # --- Enrich chunks ---
    pairs = enrich_all_chunks(chunks, lookup)

    # --- Build enriched Chunk objects ---
    enriched_chunks = build_enriched_chunks(pairs)
    print(f" enriched chunks : {len(enriched_chunks)}")

    # --- Embed and index ---
    index = index_enriched_chunks(enriched_chunks)

    # --- Load questions ---
    questions = load_questions()
    document_ids = [d.document_id for d in documents]
    problems = check_labels(questions, document_ids)
    if problems:
        for problem in problems:
            print(f"   {problem}")
        return 1
    print(f" questions     : {len(questions)}")

    # --- Evaluate contextual retrieval ---
    print()
    print("=" * 78)
    print(" STAGE 5 — RETRIEVAL EVALUATION (contextual index)")
    print("=" * 78)

    retriever = ContextualRetriever(index)
    report = evaluate_retriever(
        questions, retriever.as_retriever(),
        name="contextual dense (stage 8)", verbose=True,
    )
    print_report(report)

    print()
    compare_reports([report], k=5)
    compare_by_variant([report], questions, k=5)

    # --- Compare with baseline ---
    print()
    print("=" * 78)
    print(" COMPARISON vs BASELINE")
    print("=" * 78)
    baseline_recall = report.metrics.get("recall", {}).get(5)
    print(f" baseline dense Recall@5 : 0.7500")
    print(f" contextual Recall@5     : "
          f"{'n/a' if baseline_recall is None else f'{baseline_recall:.4f}'}")
    if baseline_recall is not None:
        change = baseline_recall - 0.7500
        print(f" change                  : {change:+.4f} "
              f"({change / 0.7500 * 100:+.1f}% relative)")
        if baseline_recall >= 0.85:
            print(" ✅ Context Recall target (≥0.85) MET")
        else:
            print(" ⚠️  Context Recall target (≥0.85) NOT MET")

    return 0


def print_report(report) -> None:
    """Print one report in a fixed layout."""
    print("=" * 78)
    print(f" RETRIEVAL REPORT — {report.name}")
    print("=" * 78)
    print(f" questions        : {report.questions}")
    print()
    print(f" {'metric':<20} {'value':>10}")
    print(f" {'-' * 19:<20} {'-' * 10:>10}")
    for label, value in report.summary_rows():
        print(f" {label:<20} {value:>10}")
    if report.misses:
        print(f"\n MISSES (rank above 5 or not found): {len(report.misses)}")
        for miss in report.misses[:10]:
            rank = miss.rank if miss.rank is not None else "none"
            first = miss.retrieved[0] if miss.retrieved else "-"
            print(f"   {miss.qid:<5} rank={str(rank):<5} "
                  f"expected={miss.gold_document:<26} got={first}")


if __name__ == "__main__":
    sys.exit(main())
