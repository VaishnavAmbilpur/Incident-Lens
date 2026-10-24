"""
Stage 9 — Generate.

This module produces the answer. It is the ninth stage.

What it does
------------
1. It takes a question and the retrieved chunks (with citations).
2. It asks an LLM to answer using ONLY the provided chunks.
3. It cites every claim with the chunk id that supports it.
4. It blocks destructive commands (rm -rf, DROP TABLE, etc.).
5. It measures the latency and the token cost.

Why this module exists
----------------------
Retrieval finds the evidence. Generation turns the evidence into an answer.
Without generation there is no answer to measure — no faithfulness, no answer
relevance. This module is the reason stages 1 through 8 exist.

The guardrail rule
------------------
The LLM must never invent a destructive command. If the source chunks do not
name a command, the answer must not name one. A safe answer that says "I do not
know" is better than a confident answer that destroys data.

Usage
-----
    from core.generate import Generator

    generator = Generator()
    answer = generator.generate(
        question="what caused the outage?",
        chunks=retrieved_chunks,
    )
    print(answer.text)
    print(answer.citations)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

try:                                    # allow "python core/generate.py"
    from core.chunking import Chunk
    from core.ingest import PROJECT
except ModuleNotFoundError:             # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.chunking import Chunk
    from core.ingest import PROJECT


class GenerationError(RuntimeError):
    """The generation step failed."""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
ENV_PATH = PROJECT / ".env"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_MODEL = os.getenv("INCIDENTLENS_GENERATE_MODEL", "qwen/qwen3.7-plus")
MAX_ANSWER_TOKENS = 512


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
        raise GenerationError(
            "GROQ_API_KEY is not set. Put it in .env or export it before running."
        )
    return key


def _openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY") or _env().get("OPENROUTER_API_KEY", "")
    if not key:
        raise GenerationError(
            "OPENROUTER_API_KEY is not set. Put it in .env or export it before running."
        )
    return key


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------
DESTRUCTIVE_PATTERNS = [
    "rm -rf",
    "rm -r",
    "drop table",
    "drop database",
    "truncate",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "format ",
]


def _contains_destructive(text: str) -> bool:
    """Return True when the text holds a destructive command pattern."""
    lowered = text.lower()
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern in lowered:
            return True
    return False


# --------------------------------------------------------------------------
# The system prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You answer infrastructure incident questions using ONLY the provided evidence chunks.

Rules:
1. Every claim must be backed by a chunk. Cite the chunk id after each claim, like [cf-2023-10-04-dns_c003].
2. If the chunks do not contain the answer, say so. Do not guess.
3. Do not invent commands, file names, IP addresses, or error codes that do not appear in the chunks.
4. Never suggest a destructive command (rm -rf, DROP TABLE, reboot, shutdown, format, etc.) unless it appears verbatim in a source chunk.
5. Keep the answer short and direct.
6. End with a "Sources:" line listing the chunk ids you cited.

The chunks follow. Each starts with its citation id in brackets."""


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
@dataclass
class GeneratedAnswer:
    """The answer plus its provenance."""

    question: str
    text: str
    citations: List[str] = field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0.0
    model: str = ""
    blocked: bool = False
    block_reason: str = ""


@dataclass
class GenerationReport:
    """The measured result of one generation run."""

    questions_answered: int
    total_latency_ms: float
    total_tokens: int
    mean_latency_ms: float
    model: str
    blocked_count: int = 0


# --------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------
class Generator:
    """Answer a question from retrieved chunks with citations and guardrails."""

    def __init__(
        self,
        model: str = GROQ_MODEL,
        max_answer_tokens: int = MAX_ANSWER_TOKENS,
    ) -> None:
        self.model = model
        self.max_answer_tokens = max_answer_tokens
        self.last_report: Optional[GenerationReport] = None

    def _build_prompt(
        self,
        question: str,
        chunks: Sequence[Chunk],
    ) -> str:
        """Build the user message: question plus cited chunks."""
        lines: List[str] = []
        lines.append(f"Question: {question}")
        lines.append("")
        lines.append("Evidence chunks:")
        for chunk in chunks:
            lines.append(f"[{chunk.chunk_id}] {chunk.text}")
        lines.append("")
        lines.append("Answer using only the evidence above. Cite each claim with its chunk id.")
        return "\n".join(lines)

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
    ) -> GeneratedAnswer:
        """
        Answer one question from the retrieved chunks.

        Parameters
        ----------
        question
            The user's question.
        chunks
            The retrieved chunks, each holding text and a chunk_id for citation.

        Returns
        -------
        A GeneratedAnswer with the answer text, cited chunk ids, and timing.
        """
        if not chunks:
            return GeneratedAnswer(
                question=question,
                text="No evidence chunks were provided. I cannot answer without sources.",
                model=self.model,
            )

        prompt = self._build_prompt(question, chunks)

        # Use provider layer (OpenRouter first, Groq fallback, disk cache)
        # This ensures free-tier resilience when one provider hits quota.
        from core.providers import complete
        started = time.perf_counter()
        try:
            provider_result = complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=self.max_answer_tokens,
                use_cache=False,
            )
        except Exception as exc:
            raise GenerationError(f"provider failed (OpenRouter → Groq): {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        text = provider_result.text.strip()
        tokens_used = provider_result.tokens_used

        # Guardrail: block destructive commands.
        if _contains_destructive(text):
            blocked_text = (
                "I cannot answer that. The source documents do not contain a "
                "safe remediation command, and I will not invent a destructive one."
            )
            return GeneratedAnswer(
                question=question,
                text=blocked_text,
                citations=[],
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model=self.model,
                blocked=True,
                block_reason="destructive command detected in generated answer",
            )

        # Extract cited chunk ids from the answer text.
        citations: List[str] = []
        for chunk in chunks:
            if chunk.chunk_id in text:
                citations.append(chunk.chunk_id)

        return GeneratedAnswer(
            question=question,
            text=text,
            citations=citations,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            model=self.model,
        )

    def generate_many(
        self,
        questions_chunks: Sequence[tuple],
    ) -> List[GeneratedAnswer]:
        """
        Answer multiple (question, chunks) pairs.

        Parameters
        ----------
        questions_chunks
            A sequence of (question, chunks) tuples.

        Returns
        -------
        A list of GeneratedAnswer, one per input pair.
        """
        answers: List[GeneratedAnswer] = []
        for question, chunks in questions_chunks:
            answers.append(self.generate(question, chunks))
        return answers

    def report(self, answers: Sequence[GeneratedAnswer]) -> GenerationReport:
        """Return the measurement report for a set of answers."""
        if not answers:
            raise GenerationError("no answers to report")
        total_latency = sum(a.latency_ms for a in answers)
        total_tokens = sum(a.tokens_used for a in answers)
        count = len(answers)
        blocked = sum(1 for a in answers if a.blocked)
        self.last_report = GenerationReport(
            questions_answered=count,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            mean_latency_ms=total_latency / count,
            model=self.model,
            blocked_count=blocked,
        )
        return self.last_report


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    from core.ingest import load_corpus
    from core.chunking import chunk_corpus
    from core.vectorstore import VectorIndex
    from core.retrieval import DenseRetriever

    print("=" * 78)
    print(" STAGE 9 — GENERATE")
    print("=" * 78)

    documents = load_corpus()
    print(f" documents : {len(documents)}")

    chunks = chunk_corpus(documents)
    print(f" chunks    : {len(chunks)}")

    index = VectorIndex()
    if not index.exists():
        print("the vector index is absent. run: python core/vectorstore.py --rebuild")
        return 1

    dense = DenseRetriever(index)

    generator = Generator()

    questions = [
        "Which Cloudflare service returned SERVFAIL on 4 October 2023?",
        "What caused the CrowdStrike outage on 6 August 2024?",
    ]

    for question in questions:
        print(f"\n Q: {question}")
        hits = dense.search(question, top_k=5)
        if not hits:
            print("   no chunks retrieved")
            continue

        answer = generator.generate(question, hits)
        safe_text = answer.text.encode("ascii", errors="replace").decode("ascii")
        print(f"   answer : {safe_text[:300]}")
        print(f"   cites  : {answer.citations}")
        print(f"   latency: {answer.latency_ms:.0f} ms  tokens: {answer.tokens_used}")
        if answer.blocked:
            print(f"   BLOCKED: {answer.block_reason}")

    if generator.last_report is None:
        report = generator.report([]) if False else None
    else:
        report = generator.last_report

    return 0


if __name__ == "__main__":
    sys.exit(main())
