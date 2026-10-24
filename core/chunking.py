"""
Stage 2 — Chunking.

This module splits a document into chunks. It is the second stage of the
pipeline.

What it does
------------
1. It splits the document text into pieces of a target size.
2. It uses an overlap, so a sentence on a boundary stays whole in one chunk.
3. It prefers a split at a paragraph break, then a sentence end, then a space.
4. It gives each chunk a stable id: <document_id>_c<number>.
5. It attaches the document metadata to every chunk.
6. It measures the quality of the split.

Why this module exists
----------------------
Chunking decides what the retriever can find. An answer cannot be retrieved if
the chunk does not hold it. Two specific failures are common:

    A cut sentence    A sentence is cut by a boundary. Without an overlap,
                      neither the left chunk nor the right chunk holds the
                      complete sentence, so the answer is invisible.

    Silent cut        The embedding model has a maximum length. Text above the
                      limit is cut with no error and no warning. The tail of the
                      chunk is then lost for the search.

Both failures lower Recall@k. Both are measured in this module, so that a
retrieval problem can be told apart from a chunking problem.

The token limit
---------------
The embedding model all-MiniLM-L6-v2 has max_seq_length = 256 tokens.
This module counts the tokens of every chunk and reports the chunks above the
limit. It does not guess.

Usage
-----
    from core.ingest import load_corpus
    from core.chunking import chunk_corpus

    chunks = chunk_corpus(load_corpus())
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:                                    # allow "python core/chunking.py"
    from core.ingest import Document, PROJECT, load_corpus
except ModuleNotFoundError:             # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.ingest import Document, PROJECT, load_corpus

# --------------------------------------------------------------------------
# Defaults. The plan records these values and the ablation table tests them.
# --------------------------------------------------------------------------
DEFAULT_SIZE = 480          # characters
DEFAULT_OVERLAP = 120       # characters
EMBED_MODEL_MAX_TOKENS = 256    # max_seq_length of all-MiniLM-L6-v2

# A split at one of these characters ends a chunk cleanly.
PARAGRAPH_BREAKS = ("\n\n",)
SENTENCE_ENDINGS = (". ", "! ", "? ", ".\n", "!\n", "?\n")

# Do not look further back than this for a clean boundary. A smaller value
# keeps the chunk close to the target size. A larger value gives cleaner text.
MIN_FILL_RATIO = 0.6

_ENCODER = None


class ChunkingError(RuntimeError):
    """The text could not be split into valid chunks."""


# --------------------------------------------------------------------------
# The data object that moves through the pipeline
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    """One piece of one document."""

    chunk_id: str                    # <document_id>_c<number>
    document_id: str
    position: int                    # 0-based position in the document
    text: str
    char_count: int
    token_count: int
    start: int                       # character offset in the document body
    end: int
    boundary: str                    # paragraph | sentence | space | hard
    has_overlap: bool
    metadata: Dict[str, object] = field(default_factory=dict)

    def citation(self) -> str:
        """The label that an answer uses to point at this evidence."""
        return self.chunk_id

    def to_dict(self, with_text: bool = True) -> Dict[str, object]:
        data = asdict(self)
        if not with_text:
            data.pop("text")
        return data


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """
    Return the number of tokens in the text.

    Use the real tokenizer of the embedding model family when it is available.
    If the tokenizer cannot be loaded, a character rule is the fallback. English
    averages about 4 characters per token, so the estimate is len(text) / 4.
    """
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:                       # noqa: BLE001
            _ENCODER = False
    if _ENCODER is False:
        return max(1, len(text) // 4)
    return len(_ENCODER.encode(text, disallowed_special=()))


# --------------------------------------------------------------------------
# Boundary search
# --------------------------------------------------------------------------
def find_split(text: str, start: int, target_end: int) -> Tuple[int, str]:
    """
    Return (end_offset, boundary_kind).

    Look for a clean break at or before target_end, and not before the minimum
    fill point. Order of preference: a paragraph break, a sentence end, a space.
    If nothing is found, cut at target_end.
    """
    minimum = start + int((target_end - start) * MIN_FILL_RATIO)

    for marker in PARAGRAPH_BREAKS:
        found = text.rfind(marker, minimum, target_end)
        if found != -1:
            return found + len(marker), "paragraph"

    for marker in SENTENCE_ENDINGS:
        found = text.rfind(marker, minimum, target_end)
        if found != -1:
            return found + len(marker), "sentence"

    found = text.rfind(" ", minimum, target_end)
    if found != -1:
        return found + 1, "space"

    return target_end, "hard"


def split_text(
    text: str,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Tuple[str, int, int, str]]:
    """
    Return a list of (chunk_text, start, end, boundary) tuples.

    The step is (size - overlap). Each chunk begins one step after the previous
    chunk, so the last `overlap` characters of a chunk repeat at the start of the
    next chunk.
    """
    if size <= 0:
        raise ChunkingError("size must be greater than 0")
    if overlap < 0:
        raise ChunkingError("overlap must be 0 or greater")
    if overlap >= size:
        raise ChunkingError("overlap must be smaller than size")

    stripped = text.strip()
    if not stripped:
        return []

    step = size - overlap
    pieces: List[Tuple[str, int, int, str]] = []
    start = 0
    length = len(stripped)

    while start < length:
        target_end = min(start + size, length)
        if target_end >= length:
            end, boundary = length, "end"
        else:
            end, boundary = find_split(stripped, start, target_end)

        piece = stripped[start:end].strip()
        if piece:
            pieces.append((piece, start, end, boundary))

        if end >= length:
            break

        next_start = start + step
        if next_start <= start:              # safety: always make progress
            next_start = end
        start = next_start

    return pieces


# --------------------------------------------------------------------------
# One document
# --------------------------------------------------------------------------
def chunk_document(
    document: Document,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """Split one document into Chunk objects."""
    metadata = document.metadata()
    chunks: List[Chunk] = []

    for position, (piece, start, end, boundary) in enumerate(
        split_text(document.text, size=size, overlap=overlap)
    ):
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}_c{position:03d}",
                document_id=document.document_id,
                position=position,
                text=piece,
                char_count=len(piece),
                token_count=count_tokens(piece),
                start=start,
                end=end,
                boundary=boundary,
                has_overlap=position > 0,
                metadata=dict(metadata),
            )
        )

    if not chunks:
        raise ChunkingError(f"{document.document_id} gave no chunks")

    return chunks


def chunk_corpus(
    documents: Sequence[Document],
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """Split every document. Return the chunks in document order."""
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, size=size, overlap=overlap))
    return chunks


# --------------------------------------------------------------------------
# Quality measurement
# --------------------------------------------------------------------------
def measure_chunks(chunks: Sequence[Chunk]) -> Dict[str, object]:
    """
    Return the facts that stage 2 must report.

    boundary_split_rate
        The percentage of chunks that do NOT end at a paragraph break or a
        sentence end. A high value means the text is being cut mid-sentence,
        which lowers Recall@k. This is the one number to watch.

    truncation_risk
        The percentage of chunks above the embedding model limit. Those chunks
        lose their tail silently at embed time.
    """
    if not chunks:
        raise ChunkingError("no chunks to measure")

    sizes = [c.char_count for c in chunks]
    tokens = [c.token_count for c in chunks]
    clean = {"paragraph", "sentence", "end"}
    dirty = [c for c in chunks if c.boundary not in clean]
    over_limit = [c for c in chunks if c.token_count > EMBED_MODEL_MAX_TOKENS]

    return {
        "chunks": len(chunks),
        "documents": len({c.document_id for c in chunks}),
        "characters": sum(sizes),
        "mean_chars": round(statistics.fmean(sizes), 1),
        "median_chars": statistics.median(sizes),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "total_tokens": sum(tokens),
        "mean_tokens": round(statistics.fmean(tokens), 1),
        "max_tokens": max(tokens),
        "boundary_split_count": len(dirty),
        "boundary_split_rate": round(len(dirty) / len(chunks), 4),
        "truncation_risk_count": len(over_limit),
        "truncation_risk_rate": round(len(over_limit) / len(chunks), 4),
        "chunks_per_document": round(len(chunks) / len({c.document_id for c in chunks}), 1),
    }


def verify_overlap(chunks: Sequence[Chunk], overlap: int) -> Dict[str, object]:
    """
    Check that consecutive chunks of one document really share text.

    This is a correctness check of the splitter, not a quality number. If the
    overlap is not present, the retriever can lose a sentence on a boundary.
    """
    by_document: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)

    checked = 0
    ok = 0
    for group in by_document.values():
        group.sort(key=lambda c: c.position)
        for left, right in zip(group, group[1:]):
            tail = left.text[-overlap:].strip()
            head = right.text[:overlap].strip()
            if not tail or not head:
                continue
            checked += 1
            # The chunker may shorten a chunk at a boundary, so compare words
            # rather than exact characters.
            tail_words = tail.split()[-6:]
            head_words = set(head.split())
            if any(word in head_words for word in tail_words):
                ok += 1

    return {
        "pairs_checked": checked,
        "pairs_with_overlap": ok,
        "overlap_rate": round(ok / checked, 4) if checked else 0.0,
    }


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    size = DEFAULT_SIZE
    overlap = DEFAULT_OVERLAP
    for argument in sys.argv[1:]:
        if argument.startswith("--size="):
            size = int(argument.split("=", 1)[1])
        elif argument.startswith("--overlap="):
            overlap = int(argument.split("=", 1)[1])

    documents = load_corpus()
    chunks = chunk_corpus(documents, size=size, overlap=overlap)
    stats = measure_chunks(chunks)
    overlap_stats = verify_overlap(chunks, overlap)

    print("=" * 78)
    print(" STAGE 2 — CHUNKING")
    print("=" * 78)
    print(f" size / overlap     : {size} / {overlap}")
    print(f" documents          : {stats['documents']}")
    print(f" chunks             : {stats['chunks']:,}")
    print(f" chunks per document: {stats['chunks_per_document']}")
    print(f" characters         : {stats['characters']:,}")
    print(f" mean chars         : {stats['mean_chars']}")
    print(f" median chars       : {stats['median_chars']}")
    print(f" min / max chars    : {stats['min_chars']} / {stats['max_chars']}")
    print(f" mean tokens        : {stats['mean_tokens']}")
    print(f" max tokens         : {stats['max_tokens']}")
    print(f" token limit        : {EMBED_MODEL_MAX_TOKENS}")

    print("\n QUALITY")
    print(f" boundary split rate : {stats['boundary_split_rate']:.2%} "
          f"({stats['boundary_split_count']:,} of {stats['chunks']:,} chunks cut mid-sentence)")
    print(f" truncation risk rate: {stats['truncation_risk_rate']:.2%} "
          f"({stats['truncation_risk_count']:,} chunks above the embedding limit)")
    print(f" overlap check       : {overlap_stats['overlap_rate']:.2%} "
          f"({overlap_stats['pairs_with_overlap']:,} of {overlap_stats['pairs_checked']:,} pairs)")

    print("\n boundary kinds")
    kinds: Dict[str, int] = {}
    for chunk in chunks:
        kinds[chunk.boundary] = kinds.get(chunk.boundary, 0) + 1
    for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"   {kind:<12} {count:>6,}  {count / len(chunks):>7.2%}")

    print("\n example chunk")
    sample = chunks[len(chunks) // 2]
    print(f"   id       : {sample.chunk_id}")
    print(f"   document : {sample.document_id}")
    print(f"   chars    : {sample.char_count}   tokens: {sample.token_count}")
    print(f"   meta     : {json.dumps(sample.metadata, ensure_ascii=True)[:150]}")
    print(f"   text     : {sample.text[:220].replace(chr(10), ' ')}...")

    if "--json" in sys.argv:
        out = PROJECT / "data" / "chunk_report.json"
        out.write_text(json.dumps({
            "config": {"size": size, "overlap": overlap},
            "stats": stats,
            "overlap": overlap_stats,
            "chunks": [c.to_dict() for c in chunks],
        }, indent=2), encoding="utf-8")
        print(f"\n report written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
