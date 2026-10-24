"""
Stage 6 — Fusion with Reciprocal Rank Fusion (RRF).

This module merges two result lists into one. It is the sixth stage.

What it does
------------
1. It takes the dense result list and the BM25 result list.
2. It scores each chunk by its RANK in each list.
3. It merges the scores and returns one list.

The formula
-----------
    RRF_score(chunk) = sum over each list of  1 / (k + rank(chunk, list))

with k = 60. The value 60 comes from the original RRF paper by Cormack, Clarke
and Buettcher (2009). A larger k gives the top positions less weight.

Why RRF and not a weighted sum of scores
----------------------------------------
This is the important design decision, and the reason is arithmetic.

    A dense result holds a cosine distance, for example 0.31.
    A BM25 result holds a BM25 score, for example 12.4.

These two numbers are NOT comparable. The cosine distance is bounded between 0
and 2. The BM25 score is unbounded and it depends on the corpus size, the mean
chunk length and the term frequencies.

A weighted sum needs a normalisation for each corpus. A new document changes the
BM25 range, so the weight must be calibrated again. RRF uses the POSITION only.
The position is a whole number between 1 and the list length. Therefore RRF
needs no calibration, and it cannot be broken by a new document.

The tradeoff, stated honestly
-----------------------------
RRF throws away the score. Two chunks at position 1 and position 2 are treated
almost the same, even when one is a much better match. A normalised weighted sum
can use that information. Therefore RRF is robust but it is not always the most
accurate method. The ablation table measures the difference; the decision is
recorded here and tested there.

Usage
-----
    from core.fusion import reciprocal_rank_fusion, RRFFusion

    merged = reciprocal_rank_fusion([dense_ids, bm25_ids], k=60)
"""

from __future__ import annotations

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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.vectorstore import Hit

DEFAULT_RRF_K = 60


class FusionError(RuntimeError):
    """The merge step failed."""


# --------------------------------------------------------------------------
# The core function
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Merge ranked lists of ids.

    Parameters
    ----------
    ranked_lists
        One list for each retriever. Each list is in rank order, best first.
        An id may appear in more than one list; that is the point of the merge.
    k
        The RRF constant. The standard value is 60.
    weights
        An optional weight for each list. The default is 1.0 for every list.
        Use a weight when one retriever is known to be stronger.

    Returns
    -------
    A list of (id, score) pairs, sorted by score, best first.

    A note on the rank
    ------------------
    The rank starts at 1, not at 0. Therefore the first item of a list
    contributes 1 / (k + 1), not 1 / k. This is the published formula.
    """
    if k <= 0:
        raise FusionError("k must be greater than 0")
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise FusionError("the weight list has a different length than the result lists")

    scores: Dict[str, float] = {}
    for list_index, ranked in enumerate(ranked_lists):
        weight = float(weights[list_index])
        for position, item in enumerate(ranked):
            if not item:
                continue
            rank = position + 1
            scores[item] = scores.get(item, 0.0) + weight * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda pair: -pair[1])


def reciprocal_rank_fusion_documents(
    ranked_lists: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Merge ranked lists that may hold DUPLICATE documents.

    This function is for a document-level merge. A retriever returns chunks, so
    one document can hold several positions in one list. Two rules are used:

        1. Only the BEST position of a document counts inside one list.
           A document must not gain score for appearing five times in one list.
        2. A document gains score from EACH list that holds it.

    Rule 1 is important. Without it, a document with many similar chunks would
    win by volume instead of by relevance.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = {}
    for list_index, ranked in enumerate(ranked_lists):
        weight = float(weights[list_index])
        seen: set = set()
        for position, item in enumerate(ranked):
            if not item or item in seen:
                continue
            seen.add(item)
            rank = position + 1
            scores[item] = scores.get(item, 0.0) + weight * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda pair: -pair[1])


# --------------------------------------------------------------------------
# The combined retriever
# --------------------------------------------------------------------------
@dataclass
class RRFFusion:
    """A retriever that merges a dense retriever and a sparse retriever."""

    dense: object
    bm25: object
    k: int = DEFAULT_RRF_K
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    candidate_k: int = 50       # how many chunks to take from each retriever
    name: str = "hybrid-rrf"
    last_latency_ms: float = 0.0
    last_contributions: Dict[str, int] = field(default_factory=dict)

    def search(self, question: str, top_k: int = 10) -> List[Hit]:
        """
        Return the merged result list.

        The Hit object comes from the dense result when the chunk is present in
        both lists, because the dense Hit holds a similarity value. A chunk that
        only BM25 found gets a synthetic Hit with distance 0 and similarity
        equal to the RRF score.
        """
        dense_hits = self.dense.search(question, top_k=self.candidate_k)
        bm25_hits = self.bm25.search(question, top_k=self.candidate_k)

        started = time.perf_counter()
        dense_ids = [hit.chunk_id for hit in dense_hits]
        bm25_ids = [hit.chunk_id for hit in bm25_hits]

        merged = reciprocal_rank_fusion(
            [dense_ids, bm25_ids], k=self.k,
            weights=[self.dense_weight, self.bm25_weight],
        )
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        # Count where each result came from. This shows the value of each list.
        dense_set, bm25_set = set(dense_ids), set(bm25_ids)
        contributions = {"both": 0, "dense_only": 0, "bm25_only": 0}

        by_chunk: Dict[str, Hit] = {}
        for hit in dense_hits:
            by_chunk.setdefault(hit.chunk_id, hit)
        for hit in bm25_hits:
            by_chunk.setdefault(hit.chunk_id, hit)

        results: List[Hit] = []
        for chunk_id, score in merged[:top_k]:
            hit = by_chunk.get(chunk_id)
            if hit is None:
                continue
            if chunk_id in dense_set and chunk_id in bm25_set:
                contributions["both"] += 1
            elif chunk_id in dense_set:
                contributions["dense_only"] += 1
            else:
                contributions["bm25_only"] += 1

            results.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    text=hit.text,
                    distance=hit.distance,
                    similarity=score,
                    metadata=dict(hit.metadata),
                    rank=len(results),
                )
            )

        self.last_contributions = contributions
        return results

    def search_documents(self, question: str, top_k: int = 10) -> List[Hit]:
        """Return the best chunk for each distinct document, in rank order."""
        hits = self.search(question, top_k=max(top_k * 5, 25))
        best: Dict[str, Hit] = {}
        for hit in hits:
            if hit.document_id and hit.document_id not in best:
                best[hit.document_id] = hit
        return list(best.values())[:top_k]

    def as_retriever(self):
        """Return a function for the evaluation harness: (question, k) -> ids."""
        def retriever(question: str, top_k: int = 10) -> List[str]:
            return [hit.document_id for hit in self.search_documents(question, top_k)]
        return retriever


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    import core._bootstrap  # noqa: F401  (adds the project root to sys.path)
    from core.vectorstore import VectorIndex
    from core.retrieval import DenseRetriever, print_hits
    from core.bm25 import BM25Retriever

    index = VectorIndex()
    if not index.exists():
        print("the vector index is absent. run: python core/vectorstore.py --rebuild")
        return 1

    print("=" * 78)
    print(" STAGE 6 — RECIPROCAL RANK FUSION")
    print("=" * 78)
    print(f" rrf k            : {DEFAULT_RRF_K}")
    print(f" candidates/list  : 50")

    # A small check of the formula, so the arithmetic is visible.
    print("\n FORMULA CHECK")
    demo = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    for item, score in demo:
        print(f"   {item}: {score:.6f}")
    print("   note: 'a' is first in list 1 and second in list 2, so it wins.")

    dense = DenseRetriever(index)
    bm25 = BM25Retriever.from_index(index)
    fusion = RRFFusion(dense=dense, bm25=bm25)

    questions = [
        "Which file caused the bugcheck 0x00000050?",
        "Which command line parameter was omitted when the R2 team deployed new credentials?",
        "What caused the Cloudflare outage on 18 November 2025?",
        "Which manifest bug made GitHub return NXDOMAIN for many records?",
        "Which postgres request wraparound took Sentry down?",
    ]

    for question in questions:
        print(f"\n Q: {question}")
        dense_hits = dense.search(question, top_k=5)
        print(f"   dense : {[h.document_id for h in dense_hits]}")
        bm25_hits = bm25.search(question, top_k=5)
        print(f"   bm25  : {[h.document_id for h in bm25_hits]}")
        merged = fusion.search(question, top_k=5)
        print(f"   fused : {[h.document_id for h in merged]}")
        print(f"   source of the top results: {fusion.last_contributions}")
        print_hits(merged, "merged chunks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
