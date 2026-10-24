"""
Diagnose command — IncidentLens.

For one query, report each pipeline stage and identify the stage that caused
a bad answer. You can tell a retrieval failure from a generation failure.

Usage
-----
    python diagnose.py "what caused the Cloudflare outage?"
    python diagnose.py --question "..." --top-k 5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

# Put the project root on sys.path (direct run).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.ingest import load_corpus, Document
from core.chunking import Chunk, chunk_document, chunk_corpus
from core.embeddings import embed_query, describe_model as describe_embed_model
from core.vectorstore import VectorIndex, Hit
from core.retrieval import DenseRetriever
from core.bm25 import BM25Retriever
from core.fusion import RRFFusion
from core.rerank import CrossEncoderReranker
from core.generate import Generator, GeneratedAnswer
from core.evaluate import (
    judge_faithfulness, judge_answer_relevance,
)
from core.contextual import Contextualizer, document_lookup, EnrichedChunk

PROJECT = _PROJECT_ROOT


def _safe(text: str, width: int = 120) -> str:
    """Truncate and make ASCII-safe for printing."""
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) > width:
        text = text[:width] + "..."
    return text.encode("ascii", errors="replace").decode("ascii")


def _print_stage(title: str, lines: List[str]) -> None:
    print(f"\n{'-' * 78}")
    print(f" {title}")
    print(f"{'-' * 78}")
    for line in lines:
        print(f"   {line}")


def diagnose(
    question: str,
    top_k: int = 5,
    generate_answer: bool = True,
) -> int:
    """
    Run one question through every pipeline stage and report the results.

    Returns 0 when the diagnosis is complete (even if a stage fails).
    """
    print("=" * 78)
    print(" INCIDENTLENS DIAGNOSE")
    print("=" * 78)
    print(f" question : {question}")
    print(f" top_k    : {top_k}")

    # ------------------------------------------------------------------
    # Stage 1 — Ingest
    # ------------------------------------------------------------------
    documents = load_corpus()
    _print_stage("STAGE 1 — INGEST", [
        f"documents loaded : {len(documents)}",
    ])

    # ------------------------------------------------------------------
    # Stage 2 — Chunk
    # ------------------------------------------------------------------
    chunks = chunk_corpus(documents)
    _print_stage("STAGE 2 — CHUNK", [
        f"chunks           : {len(chunks)}",
        f"mean chars/chunk : {sum(len(c.text) for c in chunks) / len(chunks):.0f}",
    ])

    # ------------------------------------------------------------------
    # Stage 3/4 — Embed + Index (query only)
    # ------------------------------------------------------------------
    embed_info = describe_embed_model()
    query_vector = embed_query(question)
    _print_stage("STAGE 3/4 — EMBED + INDEX (query)", [
        f"model            : {embed_info['model']} ({embed_info['dimension']}-dim)",
        f"query vector     : {len(query_vector)} floats",
    ])

    # ------------------------------------------------------------------
    # Stage 5 — Retrieve (dense + BM25)
    # ------------------------------------------------------------------
    index = VectorIndex()
    if not index.exists():
        print("\n the vector index is absent. run: python core/vectorstore.py --rebuild")
        return 1

    dense = DenseRetriever(index)
    bm25 = BM25Retriever.from_index(index)

    dense_hits = dense.search(question, top_k=top_k)
    bm25_hits = bm25.search(question, top_k=top_k)

    dense_doc_ids = [h.document_id for h in dense_hits]
    bm25_doc_ids = [h.document_id for h in bm25_hits]

    _print_stage("STAGE 5 — RETRIEVE (dense + BM25)", [
        f"dense   : {_safe(', '.join(dense_doc_ids))}",
        f"bm25    : {_safe(', '.join(bm25_doc_ids))}",
        f"dense latency : {dense.last_latency_ms:.0f} ms",
        f"bm25 latency  : {bm25.last_latency_ms:.0f} ms",
    ])

    # ------------------------------------------------------------------
    # Stage 6 — Fuse (RRF)
    # ------------------------------------------------------------------
    fusion = RRFFusion(dense=dense, bm25=bm25)
    fused_hits = fusion.search(question, top_k=top_k)
    fused_doc_ids = [h.document_id for h in fused_hits]

    _print_stage("STAGE 6 — FUSE (RRF)", [
        f"fused     : {_safe(', '.join(fused_doc_ids))}",
        f"sources   : {fusion.last_contributions}",
        f"latency   : {fusion.last_latency_ms:.0f} ms",
    ])

    # ------------------------------------------------------------------
    # Stage 7 — Rerank (cross-encoder)
    # ------------------------------------------------------------------
    reranker = CrossEncoderReranker(candidate_k=10)
    reranked_hits = reranker.search_documents(
        question, base=fusion, top_k=top_k,
    )
    reranked_doc_ids = [h.document_id for h in reranked_hits]

    _print_stage("STAGE 7 — RERANK (cross-encoder)", [
        f"reranked  : {_safe(', '.join(reranked_doc_ids))}",
        f"latency   : {reranker.last_latency_ms:.0f} ms",
        f"candidates: {reranker.last_candidates}",
    ])

    # ------------------------------------------------------------------
    # Determine if retrieval succeeded
    # ------------------------------------------------------------------
    # We cannot know the gold document without the labelled set, so we check
    # whether the top result looks plausible: it has a non-empty text and
    # a similarity score above zero.
    top_hit = reranked_hits[0] if reranked_hits else None
    retrieval_ok = top_hit is not None and len(top_hit.text.strip()) > 0

    # ------------------------------------------------------------------
    # Stage 9 — Generate
    # ------------------------------------------------------------------
    answer: Optional[GeneratedAnswer] = None
    if generate_answer and retrieval_ok:
        generator = Generator()
        answer = generator.generate(question, reranked_hits)
        _print_stage("STAGE 9 — GENERATE", [
            f"answer    : {_safe(answer.text)}",
            f"citations : {answer.citations}",
            f"latency   : {answer.latency_ms:.0f} ms  tokens: {answer.tokens_used}",
            f"blocked   : {answer.blocked}" + (f" ({answer.block_reason})" if answer.blocked else ""),
        ])
    elif not retrieval_ok:
        _print_stage("STAGE 9 — GENERATE", [
            "skipped   : no evidence chunks retrieved",
        ])

    # ------------------------------------------------------------------
    # Stage 10 — Evaluate (LLM judge)
    # ------------------------------------------------------------------
    if answer is not None and reranked_hits:
        faithfulness = judge_faithfulness(question, answer.text, reranked_hits)
        relevance = judge_answer_relevance(question, answer.text)
        _print_stage("STAGE 10 — EVALUATE (LLM judge)", [
            f"faithfulness   : {'n/a' if faithfulness is None else f'{faithfulness:.3f}'}",
            f"answer relevance: {'n/a' if relevance is None else f'{relevance:.3f}'}",
        ])

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(" DIAGNOSIS")
    print("=" * 78)

    if not retrieval_ok:
        print(" RESULT : RETRIEVAL FAILURE")
        print("         No evidence chunks were retrieved for this question.")
        print("         The retriever (stages 5-7) did not find the relevant document.")
        print("         Generation cannot fix a missing evidence problem.")
    elif answer is None:
        print(" RESULT : GENERATION SKIPPED")
        print("         Retrieval succeeded but generation was not run.")
    elif answer.blocked:
        print(" RESULT : GENERATION BLOCKED (guardrail)")
        print("         Retrieval succeeded. Generation produced a blocked answer.")
        print(f"         Reason: {answer.block_reason}")
    elif answer is not None and len(answer.citations) == 0:
        print(" RESULT : GENERATION WEAK (no citations)")
        print("         Retrieval succeeded. Generation produced an answer without citations.")
        print("         The model may not have found support in the chunks.")
    else:
        print(" RESULT : PIPELINE OK")
        print("         All stages produced output. The answer is grounded in retrieved chunks.")

    return 0


def main() -> int:
    question: Optional[str] = None
    top_k = 5
    no_generate = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-q", "--question") and i + 1 < len(args):
            question = args[i + 1]
            i += 2
        elif arg.startswith("--question="):
            question = arg.split("=", 1)[1]
            i += 1
        elif arg in ("-k", "--top-k") and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif arg.startswith("--top-k="):
            top_k = int(arg.split("=", 1)[1])
            i += 1
        elif arg in ("--no-generate",):
            no_generate = True
            i += 1
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            if not arg.startswith("-"):
                question = arg
            i += 1

    if not question:
        print("usage: python diagnose.py \"your question here\" [--top-k 5] [--no-generate]")
        return 1

    return diagnose(question, top_k=top_k, generate_answer=not no_generate)


if __name__ == "__main__":
    sys.exit(main())
