"""
Stage 5 — Dense retrieval.

This module finds chunks with the vector index. It is the fifth stage.

What it does
------------
1. It changes the question into a vector (stage 3).
2. It asks the vector index for the nearest chunks (stage 4).
3. It returns the chunks in rank order.

The known weakness of dense retrieval
-------------------------------------
A dense vector holds MEANING, not characters. Therefore it is weak at an exact
token. An incident question often holds an exact token:

    SIGSEGV        0x00000050         AS33154        CSAgent.sys
    429            vbp7ht2647l8       --env

A dense model sees a hexadecimal code as similar to another hexadecimal code.
It cannot match the characters. Stage 5b (BM25) exists for this reason.

Why dedupe is necessary
-----------------------
The index returns chunks. One document holds many chunks. If four chunks of the
same document are in the result list, the list holds only one DISTINCT document.
This module can fold the chunks into documents, so a document-level metric is
correct.

Usage
-----
    from core.vectorstore import VectorIndex
    from core.retrieval import DenseRetriever

    index = VectorIndex()
    dense = DenseRetriever(index)
    hits = dense.search("what caused the outage", top_k=10)
    ids  = dense.as_retriever()("what caused the outage", 10)
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
    from core.vectorstore import Hit, VectorIndex, VectorStoreError
    from core.embeddings import embed_query
except ModuleNotFoundError:                       # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.vectorstore import Hit, VectorIndex, VectorStoreError
    from core.embeddings import embed_query


class RetrievalError(RuntimeError):
    """The retrieval step failed."""


# --------------------------------------------------------------------------
# Dense retriever
# --------------------------------------------------------------------------
@dataclass
class DenseRetriever:
    """Find chunks by vector similarity."""

    index: VectorIndex
    name: str = "dense"
    last_latency_ms: float = 0.0

    def search(
        self,
        question: str,
        top_k: int = 10,
        where: Optional[Dict[str, object]] = None,
    ) -> List[Hit]:
        """Return the top_k nearest chunks for the question."""
        if not question or not question.strip():
            raise RetrievalError("the question is empty")
        query_vector = embed_query(question)
        started = time.perf_counter()
        hits = self.index.search(query_vector, top_k=top_k, where=where)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return hits

    def search_documents(
        self,
        question: str,
        top_k: int = 10,
        fetch_k: Optional[int] = None,
    ) -> List[Hit]:
        """
        Return the best chunk for each distinct document, in rank order.

        fetch_k is the number of chunks to fetch before the fold. Use a value
        larger than top_k, because many chunks can belong to one document. A
        small fetch_k would hide a document that sits slightly lower.
        """
        fetch = fetch_k or max(top_k * 5, 25)
        hits = self.search(question, top_k=fetch)
        best: Dict[str, Hit] = {}
        for hit in hits:
            if hit.document_id and hit.document_id not in best:
                best[hit.document_id] = hit
        return list(best.values())[:top_k]

    def as_retriever(self, fetch_k: Optional[int] = None):
        """Return a function for the evaluation harness: (question, k) -> ids."""
        def retriever(question: str, top_k: int = 10) -> List[str]:
            documents = self.search_documents(question, top_k=top_k, fetch_k=fetch_k)
            return [hit.document_id for hit in documents]
        return retriever


def hits_to_document_ids(hits: Sequence[Hit]) -> List[str]:
    """Return the document ids of the hits, with duplicates removed."""
    seen = set()
    ordered: List[str] = []
    for hit in hits:
        if hit.document_id and hit.document_id not in seen:
            seen.add(hit.document_id)
            ordered.append(hit.document_id)
    return ordered


def hits_to_chunk_ids(hits: Sequence[Hit]) -> List[str]:
    """Return the chunk ids of the hits, in rank order."""
    return [hit.chunk_id for hit in hits]


def print_hits(hits: Sequence[Hit], label: str = "hits", show_text: int = 90) -> None:
    """Print a result list in a fixed layout."""
    print(f" {label} ({len(hits)})")
    for hit in hits:
        text = " ".join(hit.text.split())[:show_text]
        print(f"   {hit.rank + 1:>2}. {hit.chunk_id:<30} sim={hit.similarity:.4f} "
              f"[{hit.metadata.get('org', '?')}] {text}")


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    index = VectorIndex()
    if not index.exists():
        print("the vector index is absent. run: python core/vectorstore.py --rebuild")
        return 1

    print("=" * 78)
    print(" STAGE 5 — DENSE RETRIEVAL")
    print("=" * 78)
    print(f" vectors : {index.count():,}")

    dense = DenseRetriever(index)

    questions = [
        "Which kernel driver file caused the CrowdStrike outage and the memory bugcheck 0x00000050?",
        "Which command line parameter was omitted when the R2 team deployed new credentials?",
        "What was the name of the user request that Postgres devoured and forced Sentry down?",
        "Which manifest bug made GitHub return NXDOMAIN for many records?",
    ]

    for question in questions:
        print(f"\n Q: {question}")
        hits = dense.search(question, top_k=5)
        print_hits(hits, "chunk level")
        documents = dense.search_documents(question, top_k=3)
        print(f" document level: {[h.document_id for h in documents]}")
        print(f" latency: {dense.last_latency_ms:.0f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
