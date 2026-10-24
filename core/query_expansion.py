"""
Multi-query expansion — IncidentLens.

This module rewrites one user question into multiple paraphrase variants,
runs each variant through the hybrid retriever, and merges the result sets
with RRF. The merged set covers more of the document space than a single query.

What it does
------------
1. It asks an LLM to produce N alternative phrasings of the query.
2. It retrieves for each phrasing independently.
3. It merges all result sets with Reciprocal Rank Fusion.

Why this helps
--------------
A hybrid retriever (BM25 + dense) misses documents when:
- the user's phrasing does not match the document vocabulary (BM25 miss), and
- the user's embedding is far from the document embedding (dense miss).

Rewriting the query into several paraphrases increases the probability that at
least one phrasing matches the document. RRF then gives the merged result a
single ranked list.

The cost
--------
N LLM calls (one per expansion, or one batched call) plus N retrieval calls.
The LLM calls use a fast, cheap model and are cached on disk. The retrieval
calls are local CPU operations (< 30 ms each).

Usage
-----
    from core.query_expansion import MultiQueryRetriever

    retriever = MultiQueryRetriever(base_retriever=fusion, n_variants=3)
    hits = retriever.retrieve("what caused the outage?", k=5)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

# Put the project root on sys.path (direct run support).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.vectorstore import Hit
from core.fusion import reciprocal_rank_fusion
from core.providers import complete, ProviderError


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_EXPANSION_SYSTEM = (
    "You are an expert at reformulating questions about software incidents and "
    "outage post-mortems. Produce {n} alternative phrasings of the question. "
    "Use different vocabulary and structure, but keep the technical meaning exact. "
    "Return only the phrasings, one per line, with no numbering or extra text."
)

_EXPANSION_USER = "Question: {question}\n\nAlternative phrasings:"


# ---------------------------------------------------------------------------
# LLM-based query expansion
# ---------------------------------------------------------------------------
def expand_query(question: str, n: int = 3, model: str = "openai/gpt-oss-20b") -> List[str]:
    """
    Return a list of n alternative phrasings of *question*.

    Falls back to returning [question] if the provider call fails.
    """
    try:
        result = complete(
            messages=[
                {
                    "role": "system",
                    "content": _EXPANSION_SYSTEM.format(n=n),
                },
                {
                    "role": "user",
                    "content": _EXPANSION_USER.format(question=question),
                },
            ],
            model=model,
            temperature=0.5,          # Some diversity in the rewrites
            max_tokens=256,
        )
        lines = [ln.strip() for ln in result.text.strip().splitlines()]
        # Filter empty lines and keep at most n
        variants = [ln for ln in lines if ln][:n]
        if not variants:
            return [question]
        # Always include original so we never lose exact-match recall
        if question not in variants:
            variants = [question] + variants
        return variants
    except ProviderError:
        return [question]


# ---------------------------------------------------------------------------
# Multi-query retriever
# ---------------------------------------------------------------------------
class MultiQueryRetriever:
    """
    Wraps any base retriever with multi-query expansion.

    Parameters
    ----------
    base_retriever : callable
        (question: str, k: int) -> list[Hit]   — the underlying retriever.
    n_variants : int
        Number of alternative phrasings to generate (default 3).
    candidate_k : int
        How many results to fetch per variant (default 10).
    rrf_k : int
        The RRF constant (default 60).
    model : str
        The LLM model to use for expansion.
    """

    def __init__(
        self,
        base_retriever,
        n_variants: int = 3,
        candidate_k: int = 10,
        rrf_k: int = 60,
        model: str = "openai/gpt-oss-20b",
    ) -> None:
        self.base_retriever = base_retriever
        self.n_variants = n_variants
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.model = model

    def retrieve(self, question: str, k: int = 5) -> List[str]:
        """
        Retrieve top-k document IDs using multi-query expansion + RRF merge.

        Returns a list of document ID strings (in rank order) to be compatible
        with evaluate_retriever(), which expects document IDs.
        """
        variants = expand_query(question, n=self.n_variants, model=self.model)

        # Retrieve for each variant; base_retriever may return strings or Hit objects
        all_ranked_lists: List[List[str]] = []

        for variant in variants:
            try:
                raw = self.base_retriever(variant, self.candidate_k)
            except Exception:  # noqa: BLE001
                continue

            # Normalise: extract document_id string whether we got str or Hit
            doc_ids: List[str] = []
            for item in raw:
                if isinstance(item, str):
                    doc_ids.append(item)
                elif hasattr(item, "document_id"):
                    doc_ids.append(item.document_id)
                # else skip unknown type

            if doc_ids:
                all_ranked_lists.append(doc_ids)

        if not all_ranked_lists:
            return []

        # Merge with RRF — returns list of (doc_id, score) tuples
        merged = reciprocal_rank_fusion(all_ranked_lists, k=self.rrf_k)

        # Return ordered document IDs up to k, deduplicated
        result: List[str] = []
        seen: set = set()
        for doc_id, _score in merged:
            if doc_id not in seen:
                seen.add(doc_id)
                result.append(doc_id)
            if len(result) >= k:
                break

        return result

    def as_retriever(self):
        """Return a (question, k) -> list[str] callable for evaluate_retriever."""
        def _retrieve(question: str, k: int = 5):
            return self.retrieve(question, k=k)
        return _retrieve


if __name__ == "__main__":
    # Quick smoke test
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    q = "what caused the Cloudflare outage?"
    print(f"Original: {q}")
    variants = expand_query(q, n=3)
    print("Expansions:")
    for v in variants:
        print(f"  - {v}")
