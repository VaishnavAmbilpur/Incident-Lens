"""
Stage 1 — Ingest.

This module reads the corpus. It is the input of the whole pipeline.

What it does
------------
1. It finds every document in data/postmortems/.
2. It routes each file by type. A .md file is read directly. A .pdf file is
   read with pypdf.
3. It parses the metadata from the document header.
4. It removes duplicate documents.
5. It returns a list of Document objects.

Why this module exists
----------------------
Ingestion and chunking are different questions.

    Ingestion asks:  "Which documents do I have, and what do I know about them?"
    Chunking asks:   "How do I cut each document?"

A bad answer can come from either one. If they live in one function, you cannot
tell which one failed. Therefore this module does ingestion only.

The failure rule
----------------
A document that cannot be read is a FAILURE, not a skip. A silently skipped
document makes the corpus smaller than you think, and every later measurement
becomes wrong. This module raises IngestError instead of continuing quietly.

Usage
-----
    from core.ingest import load_corpus
    documents = load_corpus()
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../core
PROJECT = HERE.parent                           # project root
POSTMORTEM_DIR = PROJECT / "data" / "postmortems"

# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
SUPPORTED_SUFFIXES = {".md", ".pdf"}
MIN_LEXICAL_WORDS = 120     # below this value a document is probably a stub

# --------------------------------------------------------------------------
# Language rule
# --------------------------------------------------------------------------
# A web download can return a localized page. The AWS site serves Japanese and
# Chinese versions of the same post-mortem. That happened in Phase 1 and put two
# translated documents into the corpus. A translated document is a DUPLICATE of
# an English one: it repeats the same facts in another language. It inflates the
# document count and it corrupts every retrieval metric, because the retrieval
# gold answer is defined in English.
#
# The check below is a stopword ratio. English text in this corpus scores about
# 0.25 to 0.37. Localized text scores below 0.10. The threshold of 0.15 sits
# between the two groups with a wide margin.
ENGLISH_STOPWORDS = frozenset("""
the and of to in a was that for on with as is at by this we be from it
not are were had has have but which an or their they there when what
""".split())
MIN_STOPWORD_RATIO = 0.15
MAX_NON_LATIN_RATIO = 0.05  # above this value the text holds a non-Latin script
# The code point 0x2500 separates the Latin ranges from the scripts of East
# Asia. Latin letters with accents and the usual typographic punctuation stay
# below it. Hangul, Kana and CJK ideographs stay above it.
#
# Measured values on this corpus:
#   English documents ............ 0.00%
#   the Korean AWS document ...... 19.9%
#   the Chinese AWS document ..... 42.1%
# A CJK-only range is NOT sufficient. The Korean document holds Hangul, which
# sits outside the CJK ideograph range, so a CJK-only test passed it.

# The metadata table that fetch_corpus.py writes into each new document.
FIELD_ALIASES: Dict[str, str] = {
    "organization": "org",
    "org": "org",
    "date": "published",
    "published": "published",
    "system": "system",
    "failure category": "failure",
    "failure": "failure",
    "source": "url",
    "url": "url",
    "words": "words",
    "title": "title",
    "author": "author",
}

# --------------------------------------------------------------------------
# A small scaffold for the 3 legacy documents
# --------------------------------------------------------------------------
# The 3 documents below were downloaded before this project had a manifest.
# Their headers hold a title, a date and a source, but no organization, system
# or failure category. They cannot be inferred with confidence, so they are
# declared here. New documents must NOT use this table: fetch_corpus.py writes
# the metadata into the file itself.
LEGACY_METADATA: Dict[str, Dict[str, str]] = {
    "cloudflare_bgp_leak": {
        "org": "Cloudflare", "published": "2019-06-24",
        "system": "BGP routing", "failure": "config",
    },
    "aws_kinesis_outage": {
        "org": "Amazon Web Services", "published": "2020-11-28",
        "system": "Kinesis", "failure": "capacity",
    },
    "github_mysql_failover": {
        "org": "GitHub", "published": "2018-10-30",
        "system": "MySQL", "failure": "network",
    },
    "crowdstrike_outage_2024": {
        "org": "CrowdStrike", "published": "2024-08-06",
        "system": "Falcon sensor", "failure": "software",
    },
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


class IngestError(RuntimeError):
    """A document could not be read. Do not continue quietly."""


# --------------------------------------------------------------------------
# The data object that moves through the pipeline
# --------------------------------------------------------------------------
@dataclass
class Document:
    """One post-mortem document."""

    document_id: str                 # stable short id, for example "cf-2025-11-18"
    path: str                        # the file it came from, relative to the project
    title: str
    org: str
    published: str                   # ISO date, for example "2025-11-18"
    system: str
    failure: str
    text: str                        # the body text, without the header
    word_count: int
    char_count: int
    url: str = ""
    source_type: str = "markdown"    # "markdown" or "pdf"
    page_count: int = 0              # PDF only
    warnings: List[str] = field(default_factory=list)

    def metadata(self) -> Dict[str, object]:
        """The metadata that every chunk inherits. No text, so it stays small."""
        return {
            "document_id": self.document_id,
            "org": self.org,
            "published": self.published,
            "system": self.system,
            "failure": self.failure,
            "title": self.title,
            "source_path": self.path,
        }

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data.pop("text")             # the corpus record must stay readable
        return data


# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------
def normalise_date(value: str) -> str:
    """
    Return an ISO date string.

    Accept three shapes:
        2025-11-18              -> 2025-11-18
        November 28, 2020       -> 2020-11-28
        June 24, 2019           -> 2019-06-24
    An unknown shape is returned unchanged, so the value is never lost.
    """
    text = value.strip().strip("*_ ").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    match = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", text)
    if match:
        month_name, day, year = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return f"{int(year):04d}-{month:02d}-{int(day):02d}"
    return text


def parse_title(text: str) -> str:
    """Return the first level-1 heading, else an empty string."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def parse_metadata_table(text: str) -> Dict[str, str]:
    """
    Read the Markdown table at the top of a generated document.

    The shape is:

        | Field | Value |
        |---|---|
        | Organization | Cloudflare |
        | Date | 2025-11-18 |

    Return a dict of the field names in lower case.
    """
    found: Dict[str, str] = {}
    for line in text.splitlines()[:60]:
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 3:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = cells[0].strip().lower()
        value = cells[1].strip()
        if key in {"field", "---"} or set(key) <= {"-", ":"}:
            continue
        if key in FIELD_ALIASES and value:
            found[FIELD_ALIASES[key]] = value
    return found


def parse_bold_fields(text: str) -> Dict[str, str]:
    """
    Read the "**Field:** value" lines that the legacy documents use.

    Example:  **Published:** June 24, 2019
    """
    found: Dict[str, str] = {}
    for line in text.splitlines()[:40]:
        match = re.match(r"\*\*([A-Za-z ]+):\*\*\s*(.+?)\s*$", line.strip())
        if not match:
            continue
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key in FIELD_ALIASES and value:
            found[FIELD_ALIASES[key]] = value
    return found


def strip_header(text: str) -> str:
    """
    Remove the title line, the metadata table and the leading divider.

    The metadata is carried on the chunk object, not inside the chunk text.
    This keeps the two separate. Stage 8 of the plan adds a contextual header
    on purpose, and that change must be measurable. If the header were already
    inside the text, the measurement would be confounded.
    """
    lines = text.splitlines()
    index = 0

    # Skip the title.
    while index < len(lines) and not lines[index].startswith("# "):
        index += 1
    if index < len(lines):
        index += 1

    # Skip the metadata table and the bold fields.
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|") or set(line) <= {"-", ":"}:
            index += 1
            continue
        if re.match(r"\*\*[A-Za-z ]+:\*\*", line):
            index += 1
            continue
        if set(line) <= {"-", "*", "_"} and len(line) >= 3:
            index += 1
            continue
        break

    return "\n".join(lines[index:]).strip()


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------
def read_markdown(path: Path) -> Tuple[str, int]:
    """Return (text, page_count)."""
    try:
        return path.read_text(encoding="utf-8"), 0
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace"), 0


def read_pdf(path: Path) -> Tuple[str, int]:
    """
    Return (text, page_count).

    A PDF gives one string for each page. The pages are joined in order, so a
    sentence that crosses a page boundary stays continuous for the chunker.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:                       # pragma: no cover
        raise IngestError(f"pypdf is not installed, cannot read {path.name}") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:                          # noqa: BLE001
        raise IngestError(f"cannot open PDF {path.name}: {exc}") from exc

    pages: List[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:                      # noqa: BLE001
            raise IngestError(f"cannot read page {number} of {path.name}: {exc}") from exc

    text = "\n\n".join(p.strip() for p in pages if p.strip())
    return text, len(reader.pages)


# --------------------------------------------------------------------------
# One document
# --------------------------------------------------------------------------
def load_document(path: Path) -> Document:
    """Read one file and return a Document. Raise IngestError on a problem."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise IngestError(f"unsupported file type: {path.name}")

    document_id = path.stem
    warnings: List[str] = []

    if path.suffix.lower() == ".pdf":
        raw, page_count = read_pdf(path)
        source_type = "pdf"
    else:
        raw, page_count = read_markdown(path)
        source_type = "markdown"

    if not raw.strip():
        raise IngestError(f"{path.name} gave no text")

    # Metadata: the file header first, then the legacy table as a fallback.
    parsed = {**parse_bold_fields(raw), **parse_metadata_table(raw)}
    legacy = LEGACY_METADATA.get(document_id, {})

    def pick(key: str, default: str = "unknown") -> str:
        return parsed.get(key) or legacy.get(key) or default

    title = parsed.get("title") or parse_title(raw) or document_id
    published = normalise_date(pick("published", ""))
    if not published:
        warnings.append("no publication date found")
        published = "unknown"

    body = strip_header(raw) if source_type == "markdown" else raw.strip()
    if len(body) < 200:
        warnings.append("body is very short after the header removal")
        body = raw                       # do not lose the document

    words = body.split()
    if len(words) < MIN_LEXICAL_WORDS:
        warnings.append(f"only {len(words)} words")

    return Document(
        document_id=document_id,
        path=str(path.relative_to(PROJECT)).replace("\\", "/"),
        title=title,
        org=pick("org"),
        published=published,
        system=pick("system"),
        failure=pick("failure"),
        text=body,
        word_count=len(words),
        char_count=len(body),
        url=pick("url", ""),
        source_type=source_type,
        page_count=page_count,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------
def discover(directory: Path = POSTMORTEM_DIR) -> List[Path]:
    """Return the readable files in the corpus directory, in a stable order."""
    if not directory.is_dir():
        raise IngestError(f"corpus directory does not exist: {directory}")
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda p: p.name.lower())


def content_fingerprint(text: str) -> str:
    """
    Return a fingerprint of the words in the text.

    The fingerprint ignores case, punctuation and whitespace. Therefore two
    files that hold the same incident with different formatting compare equal.
    This is the check that caught the 3 duplicate documents in Phase 1.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(words[:2000])


def language_check(text: str) -> Tuple[bool, str]:
    """
    Return (is_english, reason).

    Reason is empty when the text passes. The two tests are a stopword ratio and
    a CJK character ratio. See the note above ENGLISH_STOPWORDS for the reason.
    """
    letters = re.findall(r"[A-Za-z']+", text.lower())
    if not letters:
        return False, "no Latin letters found"

    ratio = sum(1 for word in letters if word in ENGLISH_STOPWORDS) / len(letters)
    if ratio < MIN_STOPWORD_RATIO:
        return False, f"stopword ratio {ratio:.1%} is below {MIN_STOPWORD_RATIO:.0%}"

    non_latin = sum(1 for char in text if ord(char) > 0x2500)
    non_latin_ratio = non_latin / len(text)
    if non_latin_ratio > MAX_NON_LATIN_RATIO:
        return False, (
            f"non-Latin script ratio {non_latin_ratio:.1%} is above "
            f"{MAX_NON_LATIN_RATIO:.0%}"
        )

    return True, ""


def load_corpus(
    directory: Path = POSTMORTEM_DIR,
    strict: bool = True,
    drop_duplicates: bool = True,
    require_english: bool = True,
) -> List[Document]:
    """
    Read every document in the corpus directory.

    Parameters
    ----------
    directory
        The corpus directory. The default is data/postmortems/.
    strict
        True  -> raise IngestError on the first unreadable file.
        False -> collect the message in the report and continue.
    drop_duplicates
        True -> keep the first document of a duplicate group and report the rest.
    require_english
        True -> drop a document that fails the language check.

    Returns
    -------
    A list of Document objects, in file name order.
    """
    documents: List[Document] = []
    problems: List[str] = []

    for path in discover(directory):
        try:
            document = load_document(path)
        except IngestError as exc:
            if strict:
                raise
            problems.append(str(exc))
            continue

        if require_english:
            is_english, reason = language_check(document.text)
            if not is_english:
                problems.append(
                    f"language check dropped {document.path}: {reason}. "
                    f"A localized page is a duplicate of an English one."
                )
                continue

        documents.append(document)

    if drop_duplicates:
        documents, duplicate_messages = remove_duplicates(documents)
        problems.extend(duplicate_messages)

    for message in problems:
        print(f"  [ingest] {message}", file=sys.stderr)

    return documents


def remove_duplicates(documents: Iterable[Document]) -> Tuple[List[Document], List[str]]:
    """Keep the first document of each fingerprint group."""
    kept: List[Document] = []
    seen: Dict[str, str] = {}
    messages: List[str] = []

    for document in documents:
        fingerprint = content_fingerprint(document.text)
        first = seen.get(fingerprint)
        if first is not None:
            messages.append(
                f"duplicate dropped: {document.path} (same content as {first})"
            )
            continue
        seen[fingerprint] = document.path
        kept.append(document)

    return kept, messages


def corpus_stats(documents: List[Document]) -> Dict[str, object]:
    """Return the facts that stage 1 must report."""
    total_chars = sum(d.char_count for d in documents)
    total_words = sum(d.word_count for d in documents)

    by_org: Dict[str, int] = {}
    by_failure: Dict[str, int] = {}
    for document in documents:
        by_org[document.org] = by_org.get(document.org, 0) + 1
        by_failure[document.failure] = by_failure.get(document.failure, 0) + 1

    return {
        "documents": len(documents),
        "characters": total_chars,
        "words": total_words,
        "mean_words": round(total_words / len(documents), 1) if documents else 0,
        "pdf_documents": sum(1 for d in documents if d.source_type == "pdf"),
        "with_warnings": sum(1 for d in documents if d.warnings),
        "by_org": dict(sorted(by_org.items(), key=lambda kv: -kv[1])),
        "by_failure": dict(sorted(by_failure.items(), key=lambda kv: -kv[1])),
    }


def estimated_chunks(documents: List[Document], size: int, overlap: int) -> int:
    """
    Estimate the chunk count before chunking runs.

    The step is (size - overlap). The estimate assumes that every boundary is a
    clean cut. The real count is slightly higher, because the chunker prefers a
    split at a sentence boundary and that decision can shorten a chunk.
    """
    step = max(1, size - overlap)
    return sum(max(1, -(-d.char_count // step)) for d in documents)


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    documents = load_corpus()
    stats = corpus_stats(documents)

    print("=" * 78)
    print(" STAGE 1 — INGEST")
    print("=" * 78)
    print(f" directory        : {POSTMORTEM_DIR}")
    print(f" documents        : {stats['documents']}")
    print(f" markdown         : {stats['documents'] - stats['pdf_documents']}")
    print(f" pdf              : {stats['pdf_documents']}")
    print(f" characters       : {stats['characters']:,}")
    print(f" words            : {stats['words']:,}")
    print(f" mean words/doc   : {stats['mean_words']}")
    print(f" documents with warnings : {stats['with_warnings']}")

    print("\n by organization")
    for org, count in stats["by_org"].items():
        print(f"   {org:<24} {count:>3}")

    print("\n by failure category")
    for failure, count in stats["by_failure"].items():
        print(f"   {failure:<24} {count:>3}")

    print("\n chunk estimate")
    for size, overlap in ((480, 120), (600, 120), (800, 150)):
        estimate = estimated_chunks(documents, size, overlap)
        print(f"   size {size:>4} / overlap {overlap:>4}  ->  {estimate:>5} chunks")

    warned = [d for d in documents if d.warnings]
    if warned:
        print("\n warnings")
        for document in warned:
            print(f"   {document.document_id}: {'; '.join(document.warnings)}")

    print("\n document table")
    print(f"   {'id':<26} {'date':<12} {'org':<22} {'words':>7}")
    for document in documents:
        print(f"   {document.document_id:<26} {document.published:<12} "
              f"{document.org[:21]:<22} {document.word_count:>7,}")

    if "--json" in sys.argv:
        out = PROJECT / "data" / "ingest_report.json"
        out.write_text(json.dumps(
            {"stats": stats, "documents": [d.to_dict() for d in documents]},
            indent=2), encoding="utf-8")
        print(f"\n report written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
