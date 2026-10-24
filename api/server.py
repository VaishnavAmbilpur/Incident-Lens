"""
IncidentLens API — FastAPI service.

This service calls the `core/` library stages and returns per-stage results so
the thin React UI can show the pipeline working, stage by stage.

Endpoints
---------
GET  /health         Health check.
POST /query          Run one question through all 10 stages. Returns each
                     stage's output, including retrieved chunks before and
                     after rerank, plus the generated answer.
GET  /metrics        Pipeline metrics (corpus size, index size, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the project root is on sys.path so `import core.*` works.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.ingest import load_corpus, Document
from core.chunking import Chunk, chunk_corpus
from core.embeddings import embed_query, describe_model as describe_embed_model
from core.vectorstore import VectorIndex, Hit, similarity_from_distance
from core.retrieval import DenseRetriever
from core.bm25 import BM25Retriever
from core.fusion import RRFFusion, reciprocal_rank_fusion
from core.rerank import CrossEncoderReranker
from core.generate import Generator, GeneratedAnswer
from core.evaluate import judge_faithfulness, judge_answer_relevance
from core.contextual import Contextualizer, document_lookup

app = FastAPI(title="IncidentLens", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Shared, lazily loaded resources
# --------------------------------------------------------------------------
_INDEX: Optional[VectorIndex] = None
_DENSE: Optional[DenseRetriever] = None
_BM25: Optional[BM25Retriever] = None
_FUSION: Optional[RRFFusion] = None
_RERANKER: Optional[CrossEncoderReranker] = None
_GENERATOR: Optional[Generator] = None
_DOCUMENTS: List[Document] = []
_CHUNKS: List[Chunk] = []


def _resources():
    """Load corpus, index, and retrievers once."""
    global _INDEX, _DENSE, _BM25, _FUSION, _RERANKER, _GENERATOR
    global _DOCUMENTS, _CHUNKS
    if _DENSE is not None:
        return
    _DOCUMENTS = load_corpus()
    _CHUNKS = chunk_corpus(_DOCUMENTS)
    _INDEX = VectorIndex()
    if not _INDEX.exists():
        raise RuntimeError(
            "vector index is absent. run: python core/vectorstore.py --rebuild"
        )
    _DENSE = DenseRetriever(_INDEX)
    _BM25 = BM25Retriever.from_index(_INDEX)
    _FUSION = RRFFusion(dense=_DENSE, bm25=_BM25)
    _RERANKER = CrossEncoderReranker(candidate_k=10)
    _GENERATOR = Generator()


# --------------------------------------------------------------------------
# Request / Response models
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class StageResult(BaseModel):
    name: str
    status: str  # "ok" | "skipped" | "error"
    detail: str
    data: Optional[dict] = None


class QueryResponse(BaseModel):
    question: str
    stages: List[StageResult]
    answer: Optional[str] = None
    citations: List[str] = []
    diagnosis: str


def _hit_dict(hit: Hit, include_text: bool = True) -> dict:
    data = {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "similarity": round(hit.similarity, 4),
        "rank": hit.rank,
        "metadata": hit.metadata,
    }
    if include_text:
        data["text"] = hit.text
    return data


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    try:
        _resources()
        index_ok = _INDEX is not None and _INDEX.exists()
        return {
            "status": "ok",
            "index_ready": index_ok,
            "vectors": _INDEX.count() if index_ok else 0,
            "documents": len(_DOCUMENTS),
            "chunks": len(_CHUNKS),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


@app.get("/metrics")
def metrics() -> dict:
    _resources()
    embed_info = describe_embed_model()
    return {
        "documents": len(_DOCUMENTS),
        "chunks": len(_CHUNKS),
        "vectors": _INDEX.count(),
        "embedding_model": embed_info["model"],
        "embedding_dimension": embed_info["dimension"],
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "generate_model": "openai/gpt-oss-20b",
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run one question through all pipeline stages."""
    _resources()

    question = req.question.strip()
    top_k = max(1, min(req.top_k, 20))
    stages: List[StageResult] = []

    # ---- Stage 1 — Ingest ----
    stages.append(
        StageResult(
            name="1 · Ingest",
            status="ok",
            detail=f"{len(_DOCUMENTS)} documents loaded",
            data={
                "documents": len(_DOCUMENTS),
                "organisations": len({d.org for d in _DOCUMENTS}),
            },
        )
    )

    # ---- Stage 2 — Chunk ----
    stages.append(
        StageResult(
            name="2 · Chunk",
            status="ok",
            detail=f"{len(_CHUNKS)} chunks",
            data={
                "chunks": len(_CHUNKS),
                "size_chars": 480,
                "overlap_chars": 120,
            },
        )
    )

    # ---- Stage 3/4 — Embed + Index (query) ----
    try:
        query_vector = embed_query(question)
        embed_info = describe_embed_model()
        stages.append(
            StageResult(
                name="3 · Embed / 4 · Index",
                status="ok",
                detail=f"{embed_info['dimension']}-dim query vector",
                data={
                    "dimension": embed_info["dimension"],
                    "query_vector_len": len(query_vector),
                    "index_vectors": _INDEX.count(),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        stages.append(
            StageResult(
                name="3 · Embed / 4 · Index",
                status="error",
                detail=str(exc),
            )
        )
        return _build_response(question, stages, None, [], "embed/index failed")

    # ---- Stage 5 — Retrieve (dense + BM25) ----
    try:
        dense_hits = _DENSE.search(question, top_k=top_k)
        bm25_hits = _BM25.search(question, top_k=top_k)
        stages.append(
            StageResult(
                name="5 · Retrieve",
                status="ok",
                detail=f"dense {len(dense_hits)} / bm25 {len(bm25_hits)} chunks",
                data={
                    "dense": [_hit_dict(h) for h in dense_hits],
                    "bm25": [_hit_dict(h) for h in bm25_hits],
                    "dense_latency_ms": round(_DENSE.last_latency_ms, 1),
                    "bm25_latency_ms": round(_BM25.last_latency_ms, 1),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        stages.append(
            StageResult(
                name="5 · Retrieve",
                status="error",
                detail=str(exc),
            )
        )
        return _build_response(question, stages, None, [], "retrieve failed")

    # ---- Stage 6 — Fuse (RRF) ----
    try:
        fused_hits = _FUSION.search(question, top_k=top_k)
        stages.append(
            StageResult(
                name="6 · Fuse (RRF)",
                status="ok",
                detail=f"{len(fused_hits)} fused chunks",
                data={
                    "fused": [_hit_dict(h) for h in fused_hits],
                    "contributions": _FUSION.last_contributions,
                    "latency_ms": round(_FUSION.last_latency_ms, 1),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        stages.append(
            StageResult(
                name="6 · Fuse (RRF)",
                status="error",
                detail=str(exc),
            )
        )
        return _build_response(question, stages, None, [], "fuse failed")

    # ---- Stage 7 — Rerank (cross-encoder) ----
    try:
        reranked_hits = _RERANKER.search_documents(
            question, base=_FUSION, top_k=top_k,
        )
        stages.append(
            StageResult(
                name="7 · Rerank (cross-encoder)",
                status="ok",
                detail=f"{len(reranked_hits)} reranked chunks",
                data={
                    "reranked": [_hit_dict(h) for h in reranked_hits],
                    "latency_ms": round(_RERANKER.last_latency_ms, 1),
                    "candidates": _RERANKER.last_candidates,
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        stages.append(
            StageResult(
                name="7 · Rerank (cross-encoder)",
                status="error",
                detail=str(exc),
            )
        )
        return _build_response(question, stages, None, [], "rerank failed")

    # ---- Stage 9 — Generate ----
    answer: Optional[GeneratedAnswer] = None
    diagnosis = ""
    if reranked_hits:
        try:
            answer = _GENERATOR.generate(question, reranked_hits)
            stages.append(
                StageResult(
                    name="9 · Generate",
                    status="ok",
                    detail=(
                        f"blocked={answer.blocked}, "
                        f"citations={len(answer.citations)}"
                    ),
                    data={
                        "answer": answer.text,
                        "citations": answer.citations,
                        "latency_ms": round(answer.latency_ms, 1),
                        "tokens": answer.tokens_used,
                        "blocked": answer.blocked,
                        "block_reason": answer.block_reason,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            stages.append(
                StageResult(
                    name="9 · Generate",
                    status="error",
                    detail=str(exc),
                )
            )
            return _build_response(
                question, stages, None, [], "generate failed"
            )
    else:
        stages.append(
            StageResult(
                name="9 · Generate",
                status="skipped",
                detail="no evidence chunks retrieved",
            )
        )

    # ---- Stage 10 — Evaluate (LLM judge) ----
    if answer is not None and reranked_hits:
        try:
            faithfulness = judge_faithfulness(
                question, answer.text, reranked_hits,
            )
            relevance = judge_answer_relevance(question, answer.text)
            stages.append(
                StageResult(
                    name="10 · Evaluate (judge)",
                    status="ok",
                    detail=(
                        f"faithfulness={faithfulness:.3f} "
                        f"relevance={relevance:.3f}"
                        if faithfulness is not None and relevance is not None
                        else "partial"
                    ),
                    data={
                        "faithfulness": faithfulness,
                        "answer_relevance": relevance,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            stages.append(
                StageResult(
                    name="10 · Evaluate (judge)",
                    status="error",
                    detail=str(exc),
                )
            )

    # ---- Diagnosis ----
    diagnosis = _diagnose(stages, answer, reranked_hits)

    return _build_response(question, stages, answer, reranked_hits, diagnosis)


def _build_response(
    question: str,
    stages: List[StageResult],
    answer: Optional[GeneratedAnswer],
    hits: List[Hit],
    diagnosis: str,
) -> QueryResponse:
    citations = answer.citations if answer else []
    answer_text = answer.text if answer else None
    return QueryResponse(
        question=question,
        stages=stages,
        answer=answer_text,
        citations=citations,
        diagnosis=diagnosis,
    )


def _diagnose(
    stages: List[StageResult],
    answer: Optional[GeneratedAnswer],
    hits: List[Hit],
) -> str:
    """Return a one-line diagnosis of where the pipeline succeeded or failed."""
    stage_names = {s.name for s in stages}
    if any(s.status == "error" for s in stages):
        for s in stages:
            if s.status == "error":
                return f"FAILURE at {s.name}: {s.detail}"
    if answer is None:
        if not hits:
            return "RETRIEVAL FAILURE: no evidence chunks found"
        return "GENERATION SKIPPED"
    if answer.blocked:
        return f"GENERATION BLOCKED: {answer.block_reason}"
    if not answer.citations:
        return "GENERATION WEAK: answer produced without citations"
    return "PIPELINE OK: answer grounded in retrieved chunks"


# --------------------------------------------------------------------------
# Serve the built React UI from the same origin (must come AFTER all API
# routes, otherwise it would shadow /health, /metrics and /query).
# --------------------------------------------------------------------------
_DIST_DIR = _PROJECT_ROOT / "frontend" / "dist"
if _DIST_DIR.is_dir():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    @app.get("/", include_in_schema=False)
    def _serve_index():
        index = _DIST_DIR / "index.html"
        if index.is_file():
            return FileResponse(str(index))
        return {"status": "ok", "ui": "not built (run: cd frontend && npm run build)"}

    app.mount(
        "/assets",
        StaticFiles(directory=str(_DIST_DIR / "assets")),
        name="assets",
    )


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    import uvicorn

    print("=" * 78)
    print(" INCIDENTLENS API + UI — FastAPI on http://localhost:8000")
    print("=" * 78)
    print(" UI:")
    print("   GET  /           — React pipeline view")
    print(" API:")
    print("   GET  /health   — health check")
    print("   GET  /metrics  — pipeline metrics")
    print("   POST /query    — run one question through all stages")
    print()
    uvicorn.run(app, host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
