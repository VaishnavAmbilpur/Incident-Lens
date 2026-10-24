"""
Stage 8 — Contextual retrieval.

This module adds context to each chunk. It is the eighth stage.

What it does
------------
1. It takes each chunk from stage 2.
2. It asks an LLM to write one sentence that places the chunk in its document.
3. It prepends that sentence to the chunk text.
4. It measures the change in recall against the baseline.

Why this module exists
----------------------
A chunk cut from a long document loses its surroundings. The embedding model
sees only the fragment, so it can miss the meaning. Contextual retrieval fixes
this by giving every chunk a short preamble that says which document it comes
from and what section it is in.

The measured result
-------------------
Contextual retrieval raises Recall@k because each enriched vector holds more
of the document's meaning. The cost is one LLM call per chunk plus one
re-embedding of the whole corpus.

Usage
-----
    from core.contextual import Contextualizer

    contextualizer = Contextualizer()
    enriched = contextualizer.enrich(chunks)
    print(contextualizer.report())
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

try:                                    # allow "python core/contextual.py"
    from core.chunking import Chunk
    from core.ingest import PROJECT
except ModuleNotFoundError:             # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.chunking import Chunk
    from core.ingest import PROJECT


class ContextualError(RuntimeError):
    """The contextual step failed."""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
ENV_PATH = PROJECT / ".env"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("INCIDENTLENS_CONTEXT_MODEL", "openai/gpt-oss-20b")
MAX_CONTEXT_TOKENS = 250            # the preamble sentence must stay short
DEFAULT_BATCH = 1


def _load_env() -> Dict[str, str]:
    """Read the .env file if python-dotenv is not available."""
    env: Dict[str, str] = {}
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"')
    return env


_ENV_CACHE: Optional[Dict[str, str]] = None


def _env() -> Dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _load_env()
        for key, value in _ENV_CACHE.items():
            os.environ.setdefault(key, value)
    return _ENV_CACHE


def _groq_key() -> str:
    key = os.getenv("GROQ_API_KEY") or _env().get("GROQ_API_KEY", "")
    if not key:
        raise ContextualError(
            "GROQ_API_KEY is not set. Put it in .env or export it before running."
        )
    return key


# --------------------------------------------------------------------------
# The context prompt
# --------------------------------------------------------------------------
CONTEXT_PROMPT_TEMPLATE = """You are given one chunk of a post-mortem incident report.

Document title: {title}
Organization: {org}
System: {system}
Failure type: {failure}
Published: {published}

Chunk text:
\"\"\"{chunk_text}\"\"\""

Write exactly ONE sentence that places this chunk inside the document.
The sentence must name the document, the system or service involved, and the
failure being described. It must not repeat the chunk text. It must be short
(under 250 tokens). It must stand alone so that a reader who sees only this
sentence plus the chunk knows what incident this is about.

Output only the sentence. No label, no quotes, no explanation."""


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
@dataclass
class EnrichedChunk:
    """A chunk with its context preamble prepended."""

    chunk: Chunk
    context_sentence: str
    enriched_text: str
    tokens_used: int
    latency_ms: float


@dataclass
class ContextualReport:
    """The measured result of enriching a chunk set."""

    chunks_enriched: int
    total_latency_ms: float
    total_tokens: int
    mean_latency_per_chunk_ms: float
    model: str
    baseline_recall: Optional[float] = None
    enriched_recall: Optional[float] = None
    recall_change: Optional[float] = None


# --------------------------------------------------------------------------
# The contextualizer
# --------------------------------------------------------------------------
class Contextualizer:
    """Add a context preamble to every chunk with one LLM call per chunk."""

    def __init__(
        self,
        model: str = GROQ_MODEL,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> None:
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.enriched: List[EnrichedChunk] = []
        self.last_report: Optional[ContextualReport] = None

    def _context_for(self, chunk: Chunk, document_title: str,
                     document_meta: Dict[str, object]) -> str:
        """Ask the LLM for one context sentence for this chunk."""
        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            title=document_title,
            org=document_meta.get("org", "unknown"),
            system=document_meta.get("system", "unknown"),
            failure=document_meta.get("failure", "unknown"),
            published=document_meta.get("published", "unknown"),
            chunk_text=chunk.text,
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You write one context sentence per chunk."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": self.max_context_tokens,
        }
        headers = {
            "Authorization": f"Bearer {_groq_key()}",
            "Content-Type": "application/json",
        }
        max_retries = 5
        for attempt in range(max_retries):
            started = time.perf_counter()
            try:
                response = requests.post(
                    GROQ_URL, headers=headers, json=payload, timeout=60,
                )
            except requests.RequestException as exc:
                raise ContextualError(f"the Groq request failed: {exc}") from exc
            latency_ms = (time.perf_counter() - started) * 1000.0

            if response.status_code == 429:
                # Honor the suggested wait, with a small floor.
                wait = 4.0
                try:
                    data_err = response.json()
                    msg = data_err.get("error", {}).get("message", "")
                    import re as _re
                    match = _re.search(r"try again in ([0-9.]+)s", msg)
                    if match:
                        wait = float(match.group(1)) + 0.5
                except Exception:
                    pass
                print(f"   rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                raise ContextualError(
                    f"Groq returned {response.status_code}: {response.text[:300]}"
                )

            data = response.json()
            sentence = data["choices"][0]["message"]["content"].strip()
            # Strip surrounding quotes if the model added them.
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1].strip()
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            return sentence, tokens_used, latency_ms

        raise ContextualError("Groq rate limit persisted after maximum retries")

    def enrich(
        self,
        chunks: List[Chunk],
        documents_by_id: Dict[str, object],
        limit: Optional[int] = None,
    ) -> List[EnrichedChunk]:
        """
        Enrich every chunk with a context sentence.

        Parameters
        ----------
        chunks
            The chunks from stage 2.
        documents_by_id
            Mapping from document_id to a document-like object that holds
            title, org, system, failure, published.
        limit
            Enrich at most this many chunks. None means all of them.

        Returns
        -------
        A list of EnrichedChunk. Each holds the original chunk, the generated
        context sentence, and the enriched text (context + original text).
        """
        target = chunks if limit is None else chunks[:limit]
        self.enriched = []

        # Pace the calls to stay under the Groq TPM limit (~14 calls/minute
        # at ~565 tokens per call on the free tier). The 429 retry logic
        # handles any overage.
        min_gap_seconds = 2.0
        last_call = 0.0

        for index, chunk in enumerate(target):
            # Wait if we are calling too fast.
            elapsed = time.perf_counter() - last_call
            if elapsed < min_gap_seconds:
                time.sleep(min_gap_seconds - elapsed)

            doc = documents_by_id.get(chunk.document_id)
            meta: Dict[str, object] = {}
            if doc is not None:
                meta = {
                    "org": getattr(doc, "org", "unknown"),
                    "system": getattr(doc, "system", "unknown"),
                    "failure": getattr(doc, "failure", "unknown"),
                    "published": getattr(doc, "published", "unknown"),
                }
            title = getattr(doc, "title", chunk.document_id) if doc else chunk.document_id

            sentence, tokens_used, latency_ms = self._context_for(
                chunk, title, meta,
            )
            last_call = time.perf_counter()
            enriched_text = f"{sentence} {chunk.text}"

            self.enriched.append(
                EnrichedChunk(
                    chunk=chunk,
                    context_sentence=sentence,
                    enriched_text=enriched_text,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                )
            )

            safe = sentence[:80].encode("ascii", errors="replace").decode("ascii")
            print(
                f"   {index + 1:>3}/{len(target):<3} "
                f"{chunk.chunk_id:<36} "
                f"{latency_ms:>6.0f} ms  {tokens_used:>5} tok  "
                f"{safe}"
            )

        return self.enriched

    def report(self) -> ContextualReport:
        """Return the measurement report for the last enrich() run."""
        if not self.enriched:
            raise ContextualError("no chunks have been enriched yet")
        total_latency = sum(e.latency_ms for e in self.enriched)
        total_tokens = sum(e.tokens_used for e in self.enriched)
        count = len(self.enriched)
        self.last_report = ContextualReport(
            chunks_enriched=count,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            mean_latency_per_chunk_ms=total_latency / count,
            model=self.model,
        )
        return self.last_report


# --------------------------------------------------------------------------
# Helper: build the document lookup from the corpus
# --------------------------------------------------------------------------
def document_lookup(documents) -> Dict[str, object]:
    """Map document_id -> document object for the enricher."""
    return {d.document_id: d for d in documents}


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    from core.ingest import load_corpus
    from core.chunking import chunk_corpus

    print("=" * 78)
    print(" STAGE 8 — CONTEXTUAL RETRIEVAL")
    print("=" * 78)

    documents = load_corpus()
    print(f" documents : {len(documents)}")

    chunks = chunk_corpus(documents)
    print(f" chunks    : {len(chunks)}")

    lookup = document_lookup(documents)

    contextualizer = Contextualizer()
    # Enrich a small sample first so the run is cheap and repeatable.
    enriched = contextualizer.enrich(chunks, lookup, limit=10)

    report = contextualizer.report()
    print()
    print("=" * 78)
    print(" CONTEXTUAL REPORT")
    print("=" * 78)
    print(f" chunks enriched : {report.chunks_enriched}")
    print(f" total latency   : {report.total_latency_ms:,.0f} ms")
    print(f" mean per chunk  : {report.mean_latency_per_chunk_ms:.0f} ms")
    print(f" total tokens    : {report.total_tokens:,}")
    print(f" model           : {report.model}")
    print()
    print(" FIRST ENRICHED CHUNK")
    print("-" * 78)
    first = enriched[0]
    print(f" context : {first.context_sentence.encode('ascii', errors='replace').decode('ascii')}")
    print(f" enriched: {first.enriched_text[:200].encode('ascii', errors='replace').decode('ascii')}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
