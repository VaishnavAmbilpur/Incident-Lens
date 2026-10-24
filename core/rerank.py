"""
Stage 7 — Cross-encoder reranking.

This module puts the candidate chunks in a better order. It is the seventh stage.

What it does
------------
1. It takes the candidate list from stage 5 or stage 6.
2. It scores each candidate against the question with a cross-encoder.
3. It puts the candidates in the new order.
4. It measures the quality change and the latency cost.

Why a cross-encoder is different from the embedding model
--------------------------------------------------------
This is the central idea of the stage, and it is worth reading carefully.

    A BI-ENCODER encodes the question and the chunk SEPARATELY.
        vector(question) and vector(chunk) meet only at the end, as a dot product.
        The two texts never see each other.
        Consequence: the vectors can be computed in advance, so the search is fast.
        Cost: the model cannot see how the question and the chunk interact. The
        word "not" in the question is invisible to the comparison.

    A CROSS-ENCODER reads the question and the chunk TOGETHER in one pass.
        The attention layers see both texts at the same time.
        Consequence: the score is far more accurate.
        Cost: the score cannot be computed in advance. One forward pass is needed
        for each (question, chunk) pair. Therefore it is slow.

The design that follows from these two facts
--------------------------------------------
Neither model can do the whole job:

    The first stage needs SPEED     -> bi-encoder, over the whole corpus
    The last stage needs ACCURACY   -> cross-encoder, over a short list

Therefore the pipeline uses both. This is the standard two-stage design, and it
is the reason this module exists.

The cost, stated honestly
-------------------------
The cross-encoder adds one forward pass for EACH candidate. With 50 candidates
it adds 50 passes. On a CPU that is tens of milliseconds, and it does not scale
to a large corpus. Therefore the candidate count must be bounded, and the
latency must be measured, not assumed. This module measures it.

Usage
-----
    from core.fusion import RRFFusion
    from core.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    hits = reranker.search("who is the president", base=fusion, top_k=5)
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Put the project root on sys.path BEFORE any project import (direct run).
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from core.vectorstore import Hit
except ModuleNotFoundError:                       # pragma: no cover
    from core.vectorstore import Hit

# --------------------------------------------------------------------------
# Paths and settings
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CACHE_DIR = PROJECT / ".cache" / "hf"
MODEL_CACHE_DIR = PROJECT / "models"

MODEL_NAME = os.getenv(
    "INCIDENTLENS_RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)
# The cross-encoder input is (question, passage). The model truncates above
# 512 word pieces, so a very long chunk would be cut. The chunks are about
# 120 tokens, so this limit is never reached in practice. It is set as a guard.
MAX_PAIR_TOKENS = 512
DEFAULT_CANDIDATES = 50
DEFAULT_BATCH = 32


def _set_cache_environment() -> None:
    """Point every model cache at the project directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE_DIR))


_MODEL = None


class RerankError(RuntimeError):
    """The reranker could not be loaded, or the scoring failed."""


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model():
    """Load and cache the cross-encoder model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    _set_cache_environment()
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:                        # pragma: no cover
        raise RerankError(
            "sentence-transformers is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    try:
        _MODEL = CrossEncoder(
            MODEL_NAME,
            max_length=MAX_PAIR_TOKENS,
            cache_folder=str(MODEL_CACHE_DIR),
        )
    except Exception as exc:                          # noqa: BLE001
        raise RerankError(
            f"cannot load the cross-encoder {MODEL_NAME}: {exc}. "
            f"The first run needs a network connection, about 90 MB."
        ) from exc
    return _MODEL


def describe_model() -> Dict[str, object]:
    """Return the facts about the reranker. Report these before a long run."""
    model = load_model()
    return {
        "model": MODEL_NAME,
        "max_pair_tokens": MAX_PAIR_TOKENS,
        "cache_dir": str(MODEL_CACHE_DIR),
        "device": str(getattr(model, "device", "unknown")),
        "num_labels": int(getattr(model.model.config, "num_labels", 1)),
    }


# --------------------------------------------------------------------------
# The reranker
# --------------------------------------------------------------------------
@dataclass
class CrossEncoderReranker:
    """Put candidate chunks in a better order with a cross-encoder."""

    name: str = "hybrid-rrf + rerank"
    candidate_k: int = DEFAULT_CANDIDATES
    batch_size: int = DEFAULT_BATCH
    last_latency_ms: float = 0.0
    last_score_seconds: float = 0.0
    last_candidates: int = 0

    def score(self, question: str, texts: Sequence[str]) -> List[float]:
        """
        Return one relevance score for each text.

        A larger score means a better match. The score is an unbounded real
        number from the model. It is NOT a probability, so it must not be
        compared with a probability threshold.
        """
        if not texts:
            return []
        model = load_model()
        pairs = [[question, text] for text in texts]
        started = time.perf_counter()
        try:
            raw = model.predict(pairs, batch_size=self.batch_size,
                                show_progress_bar=False)
        except Exception as exc:                      # noqa: BLE001
            raise RerankError(f"the rerank scoring failed: {exc}") from exc
        self.last_score_seconds = time.perf_counter() - started
        return [float(value) for value in raw]

    def search(
        self,
        question: str,
        base,
        top_k: int = 5,
        candidate_k: Optional[int] = None,
        combine: str = "replace",
        rrf_weight: float = 1.0,
        ce_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> List[Hit]:
        """
        Fetch candidates from the base retriever, then rerank them.

        Parameters
        ----------
        base
            A retriever with a search(question, top_k) method. Stage 6 (RRF), or
            stage 5 (dense) and stage 5b (BM25) also work.
        candidate_k
            The number of candidates to fetch before the rerank. A larger value
            gives the cross-encoder more choice and costs more time. The default
            is 50. This is the parameter that the ablation table tests.
        combine
            "replace"  the cross-encoder score alone decides the new order.
            "hybrid"   the cross-encoder RANK and the first-stage RANK are
                       merged with RRF. Use this when the first stage is
                       trustworthy and the cross-encoder must only adjust it.

            Why the choice matters: the cross-encoder is more accurate on a
            pair, but it has no knowledge of the whole result list. On a corpus
            with many similar documents it can push a good document down. A
            "replace" order is therefore not always better, and the ablation
            table measures which one wins.
        """
        fetch = candidate_k or self.candidate_k
        started = time.perf_counter()

        candidates = base.search(question, top_k=fetch)
        self.last_candidates = len(candidates)
        if not candidates:
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            return []

        scores = self.score(question, [hit.text for hit in candidates])

        if combine == "replace":
            # A stable sort keeps the base order when two scores are equal, so
            # the result is repeatable.
            order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        elif combine == "hybrid":
            # The first-stage rank is the position in `candidates`.
            by_ce_rank = sorted(range(len(candidates)), key=lambda i: -scores[i])
            ce_position = {index: position + 1 for position, index in enumerate(by_ce_rank)}

            merged: Dict[int, float] = {}
            for index in range(len(candidates)):
                base_rank = index + 1
                merged[index] = (
                    rrf_weight * (1.0 / (rrf_k + base_rank))
                    + ce_weight * (1.0 / (rrf_k + ce_position[index]))
                )
            order = sorted(merged, key=lambda i: -merged[i])
        else:
            raise RerankError(f"unknown combine mode: {combine!r}")

        reranked: List[Hit] = []
        for position in order:
            hit = candidates[position]
            reranked.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    text=hit.text,
                    distance=hit.distance,
                    similarity=scores[position],   # the new score
                    metadata=dict(hit.metadata),
                    rank=len(reranked),
                )
            )

        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return reranked[:top_k]

    def search_documents(
        self,
        question: str,
        base,
        top_k: int = 10,
        candidate_k: Optional[int] = None,
        combine: str = "replace",
    ) -> List[Hit]:
        """Return the best chunk for each distinct document, in rank order."""
        fetch = candidate_k or self.candidate_k
        hits = self.search(question, base=base, top_k=fetch, candidate_k=fetch,
                           combine=combine)
        best: Dict[str, Hit] = {}
        for hit in hits:
            if hit.document_id and hit.document_id not in best:
                best[hit.document_id] = hit
        return list(best.values())[:top_k]

    def as_retriever(self, base, candidate_k: Optional[int] = None,
                     combine: str = "replace"):
        """Return a function for the evaluation harness: (question, k) -> ids."""
        def retriever(question: str, top_k: int = 10) -> List[str]:
            documents = self.search_documents(
                question, base=base, top_k=top_k,
                candidate_k=candidate_k, combine=combine)
            return [hit.document_id for hit in documents]
        return retriever


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    import json

    from core.vectorstore import VectorIndex
    from core.retrieval import DenseRetriever
    from core.bm25 import BM25Retriever
    from core.fusion import RRFFusion
    from core.evaluate import (
        load_questions, check_labels, evaluate_retriever,
        compare_reports, compare_by_variant, variant_report,
    )
    from core.ingest import load_corpus

    print("=" * 78)
    print(" STAGE 7 — CROSS-ENCODER RERANKING")
    print("=" * 78)

    info = describe_model()
    print(f" model            : {info['model']}")
    print(f" device           : {info['device']}")
    print(f" max pair tokens  : {info['max_pair_tokens']}")
    print(f" model cache      : {info['cache_dir']}")

    index = VectorIndex()
    if not index.exists():
        print("\n the vector index is absent. run: python core/vectorstore.py --rebuild")
        return 1

    documents = load_corpus()
    document_ids = [d.document_id for d in documents]
    questions = load_questions()
    problems = check_labels(questions, document_ids)
    if problems:
        for problem in problems:
            print(f"   {problem}")
        return 1
    print(f"\n questions        : {len(questions)}")
    print(f" index vectors    : {index.count():,}")

    dense = DenseRetriever(index)
    bm25 = BM25Retriever.from_index(index)
    fusion = RRFFusion(dense=dense, bm25=bm25)
    reranker = CrossEncoderReranker()

    # A direct demonstration on one question, before the full evaluation.
    print("\n DEMONSTRATION")
    demo = ("Cloudflare customers received failure answers instead of addresses "
            "for websites they knew existed during an autumn 2023 resolver event.")
    print(f" Q: {demo}")
    before = fusion.search_documents(demo, top_k=5)
    print(f" before rerank: {[h.document_id for h in before]}")
    after = reranker.search_documents(demo, base=fusion, top_k=5)
    print(f" after  rerank: {[h.document_id for h in after]}")
    print(f" rerank cost  : {reranker.last_latency_ms:.0f} ms "
          f"({reranker.last_candidates} candidates, "
          f"{reranker.last_score_seconds * 1000:.0f} ms in the model)")

    # The full measurement.
    print("\n EVALUATION — this takes a few minutes on the CPU")
    reports = []

    reports.append(evaluate_retriever(
        questions, fusion.as_retriever(), name="hybrid RRF (stage 6)"))
    print_report_line(reports[-1])

    # Sweep the candidate count and the combine mode. The result is not
    # monotonic, so the shape must be measured, not assumed.
    for candidate_k in (10, 20, 30, 50):
        for combine in ("replace", "hybrid"):
            reranker.candidate_k = candidate_k
            label = f"rerank {candidate_k} {combine}"
            reports.append(evaluate_retriever(
                questions,
                reranker.as_retriever(fusion, candidate_k=candidate_k,
                                      combine=combine),
                name=label))
            print_report_line(reports[-1])

    print()
    compare_reports(reports, k=5)
    compare_by_variant(reports, questions, k=5)

    if "--json" in sys.argv:
        out = PROJECT / "data" / "rerank_results.json"
        out.write_text(json.dumps({
            "model": info,
            "by_variant": {
                report.name: {
                    variant: variant_report(report, questions, variant, k=5)
                    for variant in ("literal", "paraphrase")
                }
                for report in reports
            },
            "reports": [r.to_dict() for r in reports],
        }, indent=2), encoding="utf-8")
        print(f"\n results written to {out}")

    return 0


def print_report_line(report) -> None:
    """Print one compact line while the long evaluation runs."""
    recall = report.metrics.get("recall", {}).get(5)
    ndcg = report.metrics.get("ndcg", {}).get(5)
    print(f"   {report.name:<24} Recall@5="
          f"{'n/a' if recall is None else f'{recall:.4f}'}  "
          f"nDCG@5={'n/a' if ndcg is None else f'{ndcg:.4f}'}  "
          f"MRR={'n/a' if report.mrr is None else f'{report.mrr:.4f}'}  "
          f"latency={'n/a' if report.mean_latency_ms is None else f'{report.mean_latency_ms:,.0f} ms'}")


if __name__ == "__main__":
    sys.exit(main())
