"""
Stage 5b — Sparse retrieval with BM25.

This module finds chunks by exact keyword. It is the second half of stage 5.

What it does
------------
1. It tokenises every chunk into terms.
2. It builds a BM25 index over the terms.
3. It scores each chunk against the question terms.
4. It returns the chunks in rank order.

Why BM25 is necessary
---------------------
A dense vector holds meaning. It does not hold characters. An incident question
holds tokens that must match exactly:

    0x00000050    SIGSEGV    CSAgent.sys    AS33154    --env    SERVFAIL

A dense model treats every hexadecimal code as similar to every other
hexadecimal code. BM25 matches the characters. Therefore the two retrievers fail
on different questions, and a fusion of the two is stronger than either one.

Why BM25 needs a keyword list
-----------------------------
BM25 punishes a term that appears in many chunks. The token "the" appears in
every chunk, so it has almost no weight. That is correct behavior, and it is
also why the stopword list below is small: removing a term is not necessary when
the scoring already discounts it. The list only removes the tokens that add cost
with no gain.

The important preservation rule
-------------------------------
The tokeniser must KEEP these tokens:

    hexadecimal codes  0x00000050
    version numbers    1.1.1.1, v2, 2025.1
    flags              --env
    file names         CSAgent.sys
    error codes        429, 500, HTTP

A general tokeniser removes them. That would remove the exact reason for BM25.
The function `tokenize` below is written for this purpose, and it is tested.

Usage
-----
    from core.vectorstore import VectorIndex
    from core.bm25 import BM25Retriever

    index = VectorIndex()
    bm25 = BM25Retriever.from_index(index)
    hits = bm25.search("bugcheck 0x00000050 CSAgent.sys", top_k=10)
"""

from __future__ import annotations

import math
import re
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
    from core.vectorstore import Hit, VectorIndex, VectorStoreError, similarity_from_distance
except ModuleNotFoundError:                       # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.vectorstore import Hit, VectorIndex, VectorStoreError, similarity_from_distance

# A very short list. See the note above: BM25 already discounts a common term.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of to in on at by for
with from is are was were be been being it its as not no do does did done
we our you your they their he she his her have has had will would can could
""".split())

# A token is one of:
#   a hexadecimal code       0x00000050
#   a version or an address  1.1.1.1   v2.3   2025.1
#   a word with a flag       --env     -n
#   a word with a file part  CSAgent.sys   crates.io
#   a plain word
TOKEN_RE = re.compile(
    r"""
    (?:
        0x[0-9a-f]+                     # hexadecimal
      | \d+(?:\.\d+)+                   # version, address, decimal number
      | \d+                             # a plain number
      | -{1,2}[a-z][a-z0-9-]*           # a command line flag
      | [a-z][a-z0-9]*(?:[._-][a-z0-9]+)*   # a word, possibly with . _ or -
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


class BM25Error(RuntimeError):
    """The BM25 index could not be built or searched."""


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------
def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """
    Change text into a list of terms.

    The tokeniser is written for incident text. It keeps an exact token that a
    general tokeniser would destroy. Compare:

        general:  "bugcheck 0x00000050 in CSAgent.sys" -> bugcheck, 0x00000050,
                  in, csagent, sys          (the meaning of the file name is lost)
        this one: bugcheck, 0x00000050, in, csagent.sys
                                             (the file name stays whole)
    """
    if not text:
        return []
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    if not keep_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def document_frequency(tokens_by_chunk: Sequence[Sequence[str]]) -> Dict[str, int]:
    """Return the number of chunks that hold each term."""
    counts: Dict[str, int] = {}
    for tokens in tokens_by_chunk:
        for token in set(tokens):
            counts[token] = counts.get(token, 0) + 1
    return counts


# --------------------------------------------------------------------------
# The BM25 retriever
# --------------------------------------------------------------------------
@dataclass
class BM25Retriever:
    """Find chunks by exact keyword match."""

    chunk_ids: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    metadatas: List[Dict[str, object]] = field(default_factory=list)
    _model: object = None
    _tokens: List[List[str]] = field(default_factory=list)
    name: str = "bm25"
    build_seconds: float = 0.0
    last_latency_ms: float = 0.0

    # ---------------- build ----------------
    @classmethod
    def from_index(cls, index: VectorIndex, batch: int = 1000) -> "BM25Retriever":
        """
        Build the BM25 index from the vector store.

        The vector store holds the text and the metadata, so no second read of
        the corpus is necessary. The chunk ids stay identical, which is what
        makes the fusion in stage 6 possible.
        """
        if not index.exists():
            raise BM25Error("the vector index is empty")

        collection = index.collection()
        gotten = collection.get(include=["documents", "metadatas"])
        ids = list(gotten.get("ids") or [])
        texts = [text or "" for text in (gotten.get("documents") or [])]
        metadatas = [dict(m or {}) for m in (gotten.get("metadatas") or [])]

        if not ids or len(ids) != len(texts):
            raise BM25Error("the store returned an inconsistent result")

        return cls.from_records(ids, texts, metadatas)

    @classmethod
    def from_records(
        cls,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, object]],
    ) -> "BM25Retriever":
        """Build the BM25 index from ready records. Useful for a test."""
        if not chunk_ids:
            raise BM25Error("no chunks to index")
        if not (len(chunk_ids) == len(texts) == len(metadatas)):
            raise BM25Error("the record lists have different lengths")

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:                 # pragma: no cover
            raise BM25Error(
                "rank-bm25 is not installed. Run: pip install -r requirements.txt"
            ) from exc

        started = time.time()
        tokens = [tokenize(text) for text in texts]
        if not any(tokens):
            raise BM25Error("every chunk gave an empty token list")

        model = BM25Okapi(tokens)

        return cls(
            chunk_ids=list(chunk_ids),
            document_ids=[str(m.get("document_id", "")) for m in metadatas],
            texts=list(texts),
            metadatas=[dict(m) for m in metadatas],
            _model=model,
            _tokens=tokens,
            build_seconds=time.time() - started,
        )

    # ---------------- search ----------------
    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def search(self, question: str, top_k: int = 10) -> List[Hit]:
        """
        Return the top_k chunks for the question.

        A chunk with a score of 0 holds none of the question terms. Such a chunk
        is not a result, so it is removed. Without this step the result list
        would be filled with unrelated chunks that have equal scores of 0.
        """
        if self._model is None:
            raise BM25Error("the BM25 index is not built")
        if not question or not question.strip():
            raise BM25Error("the question is empty")

        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        started = time.perf_counter()
        scores = self._model.get_scores(query_tokens)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])

        hits: List[Hit] = []
        for index in ranked:
            if scores[index] <= 0:
                break
            hits.append(
                Hit(
                    chunk_id=self.chunk_ids[index],
                    document_id=self.document_ids[index],
                    text=self.texts[index],
                    distance=0.0,
                    similarity=float(scores[index]),
                    metadata=dict(self.metadatas[index]),
                    rank=len(hits),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def search_documents(self, question: str, top_k: int = 10) -> List[Hit]:
        """Return the best chunk for each distinct document, in rank order."""
        fetch = max(top_k * 5, 25)
        hits = self.search(question, top_k=fetch)
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

    def stats(self) -> Dict[str, object]:
        """Return the facts that stage 5b must report."""
        frequencies = document_frequency(self._tokens)
        lengths = [len(tokens) for tokens in self._tokens]
        rare = sorted(frequencies.items(), key=lambda kv: kv[1])[:10]
        common = sorted(frequencies.items(), key=lambda kv: -kv[1])[:10]
        return {
            "chunks": self.size,
            "vocabulary": len(frequencies),
            "mean_terms": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "max_terms": max(lengths) if lengths else 0,
            "build_seconds": round(self.build_seconds, 2),
            "rarest_terms": rare,
            "most_common_terms": common,
        }


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    index = VectorIndex()
    if not index.exists():
        print("the vector index is absent. run: python core/vectorstore.py --rebuild")
        return 1

    print("=" * 78)
    print(" STAGE 5b — BM25 SPARSE RETRIEVAL")
    print("=" * 78)

    bm25 = BM25Retriever.from_index(index)
    stats = bm25.stats()

    print(f" chunks           : {stats['chunks']:,}")
    print(f" vocabulary       : {stats['vocabulary']:,} terms")
    print(f" mean terms/chunk : {stats['mean_terms']}")
    print(f" max terms/chunk  : {stats['max_terms']}")
    print(f" build time       : {stats['build_seconds']} s")

    print("\n most common terms (low weight)")
    print("   " + ", ".join(f"{t}({n})" for t, n in stats["most_common_terms"]))
    print("\n rarest terms (high weight)")
    print("   " + ", ".join(f"{t}({n})" for t, n in stats["rarest_terms"]))

    print("\n TOKENISER CHECK — these tokens must survive")
    probe = "bugcheck 0x00000050 in CSAgent.sys at 1.1.1.1 with --env and HTTP 429"
    print(f"   input : {probe}")
    print(f"   tokens: {tokenize(probe)}")

    questions = [
        "Which file caused the bugcheck 0x00000050?",
        "Which parameter was omitted, --env or --host?",
        "What happened to 1.1.1.1 on 4 October 2023?",
        "Which manifest bug caused NXDOMAIN answers?",
    ]

    for question in questions:
        print(f"\n Q: {question}")
        hits = bm25.search(question, top_k=3)
        if not hits:
            print("   no chunk holds a question term")
        for hit in hits:
            text = " ".join(hit.text.split())[:80]
            print(f"   {hit.rank + 1:>2}. {hit.chunk_id:<30} score={hit.similarity:>7.3f} "
                  f"[{hit.metadata.get('org', '?')}] {text}")
        print(f"   latency: {bm25.last_latency_ms:.1f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
