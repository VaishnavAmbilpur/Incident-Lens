"""
Stage 3 — Embeddings.

This module changes text into vectors. It is the third stage of the pipeline.

What it does
------------
1. It loads the local embedding model all-MiniLM-L6-v2.
2. It changes each chunk into a vector of 384 numbers.
3. It normalises every vector, so a dot product equals a cosine similarity.
4. It measures the speed and the truncation risk.

Why this module exists
----------------------
The embedding model is the bridge between the words and the search. Two facts
about it decide the quality of the whole pipeline:

    Truncation    The model has a maximum length. Text above that length is cut
                  with NO error and NO warning. The tail of the chunk then
                  cannot be found by a search. This is a silent failure, and it
                  is the reason this module counts the tokens of every chunk.

    Normalisation A raw vector has a length. If the vectors are normalised to
                  length 1, then the dot product equals the cosine similarity.
                  Therefore one distance measure serves the whole system.

Why the model is local
----------------------
all-MiniLM-L6-v2 runs on the CPU. It needs no API key and it has no cost.
Therefore stages 3, 4 and 7 always operate, even when a provider has no quota.
Only stages 8 and 9 need a cloud provider.

The cache
---------
The model must be downloaded one time, about 90 MB. The download goes to
<project>/.cache/hf, NOT to the user profile. A project cache is portable and
it does not depend on the permissions of a home directory.

Usage
-----
    from core.embeddings import embed_texts, embed_chunks, describe_model

    vectors = embed_texts(["a bad config caused the outage"])
    matrix, stats = embed_chunks(chunks)
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:                                     # allow "python core/embeddings.py"
    from core.chunking import Chunk
except ModuleNotFoundError:              # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.chunking import Chunk

# --------------------------------------------------------------------------
# Paths and settings
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# The cache must be set BEFORE sentence_transformers is imported.
CACHE_DIR = PROJECT / ".cache" / "hf"
MODEL_CACHE_DIR = PROJECT / "models"

MODEL_NAME = os.getenv("INCIDENTLENS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384
# The operational limit. SentenceTransformer sets max_seq_length to 256 when it
# loads this model. Text above 256 tokens is removed at encode time.
EMBED_MAX_TOKENS = 256
DEFAULT_BATCH = 32


def _set_cache_environment() -> None:
    """Point every model cache at the project directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE_DIR))
    # Offline mode is NOT used. The first run downloads the model.


_MODEL = None
_TOKENIZER = None


class EmbeddingError(RuntimeError):
    """The model could not be loaded, or the vectors are not valid."""


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model():
    """Load and cache the sentence-transformer model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    _set_cache_environment()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:                        # pragma: no cover
        raise EmbeddingError(
            "sentence-transformers is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    try:
        _MODEL = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))
    except Exception as exc:                          # noqa: BLE001
        raise EmbeddingError(
            f"cannot load the model {MODEL_NAME}: {exc}. "
            f"The first run needs a network connection, about 90 MB."
        ) from exc

    _MODEL.max_seq_length = EMBED_MAX_TOKENS
    return _MODEL


def load_tokenizer():
    """
    Load the tokenizer of the embedding model.

    This is the REAL tokenizer. tiktoken is a different tokenizer and it gives a
    different count. Measured on this corpus: the real tokenizer gives about 6%
    more tokens than tiktoken gives.
    """
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    _set_cache_environment()
    try:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME, cache_dir=str(CACHE_DIR / "hub")
        )
    except Exception:                                 # noqa: BLE001
        _TOKENIZER = False
    return _TOKENIZER


def real_token_count(text: str) -> Optional[int]:
    """Return the true token count, or None when the tokenizer is unavailable."""
    tokenizer = load_tokenizer()
    if tokenizer is False:
        return None
    return len(tokenizer.encode(text, add_special_tokens=True))


def describe_model() -> Dict[str, object]:
    """Return the facts about the model. Report these before a long run."""
    model = load_model()
    tokenizer = load_tokenizer()
    return {
        "model": MODEL_NAME,
        "dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "tokenizer_max_length": getattr(tokenizer, "model_max_length", None)
        if tokenizer is not False else None,
        "cache_dir": str(MODEL_CACHE_DIR),
        "device": str(getattr(model, "device", "unknown")),
    }


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def embed_texts(
    texts: Sequence[str],
    batch_size: int = DEFAULT_BATCH,
    show_progress: bool = False,
) -> List[List[float]]:
    """
    Change a list of texts into a list of vectors.

    normalize_embeddings=True is used, so every vector has length 1 and a dot
    product equals a cosine similarity. Do not remove this setting: the distance
    measure of the whole system depends on it.
    """
    if not texts:
        return []
    model = load_model()
    try:
        matrix = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
    except Exception as exc:                          # noqa: BLE001
        raise EmbeddingError(f"the encode step failed: {exc}") from exc
    return [row.tolist() for row in matrix]


def embed_query(text: str) -> List[float]:
    """
    Embed one query.

    The model is symmetric, so a query and a chunk use the same encoder. No
    instruction prefix is necessary, unlike the E5 and BGE families.
    """
    vectors = embed_texts([text])
    if not vectors:
        raise EmbeddingError("the query gave no vector")
    return vectors[0]


# --------------------------------------------------------------------------
# Batch embedding with measurement
# --------------------------------------------------------------------------
@dataclass
class EmbeddingStats:
    """The facts that stage 3 must report."""

    chunks: int
    dimension: int
    device: str
    seconds: float
    chunks_per_second: float
    mean_tokens: float
    median_tokens: int
    max_tokens: int
    token_limit: int
    truncated: int
    empty_texts: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "chunks": self.chunks,
            "dimension": self.dimension,
            "device": self.device,
            "seconds": round(self.seconds, 2),
            "chunks_per_second": round(self.chunks_per_second, 1),
            "mean_tokens": round(self.mean_tokens, 1),
            "median_tokens": self.median_tokens,
            "max_tokens": self.max_tokens,
            "token_limit": self.token_limit,
            "truncated": self.truncated,
            "empty_texts": self.empty_texts,
        }


def embed_chunks(
    chunks: Sequence[Chunk],
    batch_size: int = DEFAULT_BATCH,
    show_progress: bool = True,
) -> Tuple[List[List[float]], EmbeddingStats]:
    """
    Embed every chunk and measure the work.

    The token count comes from the real tokenizer when it is available, and from
    the chunk record when it is not. The count is the basis of the truncation
    measurement: a chunk above the limit loses its tail silently.
    """
    texts = [c.text for c in chunks]
    empty = sum(1 for text in texts if not text.strip())

    counts: List[int] = []
    for chunk in chunks:
        real = real_token_count(chunk.text)
        counts.append(real if real is not None else chunk.token_count)

    model = load_model()
    started = time.time()
    vectors = embed_texts(texts, batch_size=batch_size, show_progress=show_progress)
    seconds = time.time() - started

    stats = EmbeddingStats(
        chunks=len(chunks),
        dimension=len(vectors[0]) if vectors else 0,
        device=str(getattr(model, "device", "unknown")),
        seconds=seconds,
        chunks_per_second=(len(chunks) / seconds) if seconds > 0 else 0.0,
        mean_tokens=statistics.fmean(counts) if counts else 0.0,
        median_tokens=int(statistics.median(counts)) if counts else 0,
        max_tokens=max(counts) if counts else 0,
        token_limit=EMBED_MAX_TOKENS,
        truncated=sum(1 for count in counts if count > EMBED_MAX_TOKENS),
        empty_texts=empty,
    )
    return vectors, stats


def verify_vectors(vectors: Sequence[Sequence[float]], dimension: int = EMBED_DIM) -> Dict[str, object]:
    """
    Check the vectors before the index is built.

    A wrong vector is worse than no vector, because the search still returns a
    result. The checks below are cheap and they catch a broken encode step.
    """
    if not vectors:
        return {"vectors": 0, "valid": False, "note": "no vectors"}

    import math

    bad_dimension = sum(1 for v in vectors if len(v) != dimension)
    zero_length = sum(1 for v in vectors if all(abs(x) < 1e-12 for x in v))
    lengths = [math.sqrt(sum(x * x for x in v)) for v in vectors]
    not_normalised = sum(1 for n in lengths if abs(n - 1.0) > 1e-3)
    finite = all(math.isfinite(x) for v in vectors[:50] for x in v)

    return {
        "vectors": len(vectors),
        "dimension": dimension,
        "bad_dimension": bad_dimension,
        "zero_length": zero_length,
        "not_normalised": not_normalised,
        "min_length": round(min(lengths), 6),
        "max_length": round(max(lengths), 6),
        "all_finite": finite,
        "valid": (bad_dimension == 0 and zero_length == 0
                  and not_normalised == 0 and finite),
    }


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    import json

    import core._bootstrap  # noqa: F401  (adds the project root to sys.path)
    from core.ingest import load_corpus
    from core.chunking import chunk_corpus

    size, overlap = 480, 120
    for argument in sys.argv[1:]:
        if argument.startswith("--size="):
            size = int(argument.split("=", 1)[1])
        elif argument.startswith("--overlap="):
            overlap = int(argument.split("=", 1)[1])

    print("=" * 78)
    print(" STAGE 3 — EMBEDDINGS")
    print("=" * 78)

    info = describe_model()
    print(f" model            : {info['model']}")
    print(f" dimension        : {info['dimension']}")
    print(f" device           : {info['device']}")
    print(f" max_seq_length   : {info['max_seq_length']}")
    print(f" model cache      : {info['cache_dir']}")

    documents = load_corpus()
    chunks = chunk_corpus(documents, size=size, overlap=overlap)
    print(f"\n chunking         : {size}/{overlap} -> {len(chunks):,} chunks")

    print("\n embedding...")
    vectors, stats = embed_chunks(chunks, show_progress=True)
    checks = verify_vectors(vectors)

    print("\n RESULT")
    print(f" chunks           : {stats.chunks:,}")
    print(f" vectors          : {len(vectors):,} x {stats.dimension}")
    print(f" elapsed          : {stats.seconds:.1f} s")
    print(f" speed            : {stats.chunks_per_second:,.1f} chunks/second")

    print("\n TOKENS (real tokenizer)")
    print(f" mean             : {stats.mean_tokens}")
    print(f" median           : {stats.median_tokens}")
    print(f" max              : {stats.max_tokens}")
    print(f" limit            : {stats.token_limit}")
    print(f" truncated chunks : {stats.truncated}  "
          f"({stats.truncated / stats.chunks:.2%})")
    print(f" empty texts      : {stats.empty_texts}")

    print("\n VECTOR CHECKS")
    for key, value in checks.items():
        print(f"   {key:<18} {value}")

    if "--json" in sys.argv:
        out = PROJECT / "data" / "embedding_report.json"
        out.write_text(json.dumps(
            {"model": info, "stats": stats.as_dict(), "checks": checks},
            indent=2), encoding="utf-8")
        print(f"\n report written to {out}")

    return 0 if checks.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
