"""
03_text_splitters.py
====================
Phase 1: Text Splitters & Chunking Strategies
IncidentLens Chunking Engine:
1. Ingests official documents using PostmortemDocumentLoader from Module 02.
2. Implements RecursiveCharacterTextSplitter with smart separator hierarchy:
   ["\\n\\n", "\\n", ". ", " ", ""]
3. Performs overlap math analysis and boundary inspection.
4. Ensures metadata inheritance (source, incident_name, page_number) and adds chunk_id.
5. Compares chunking configurations (Fine, Standard, Coarse) to determine optimal retrieval size.
6. Exports `get_production_chunks()` for downstream vector indexing (Modules 04, 05, 06).
"""

import sys
import os
import warnings
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 output encoding across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Dynamic import because module filename starts with digits ('02_document_loaders')
import importlib
import importlib.util

def _get_loader_class():
    current_dir = Path(__file__).resolve().parent
    loader_path = current_dir / "02_document_loaders.py"
    spec = importlib.util.spec_from_file_location("module_02_loaders", str(loader_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "PostmortemDocumentLoader")

PostmortemDocumentLoader = _get_loader_class()


# =====================================================================
# Production Chunking Configuration
# =====================================================================
# Why 800 chars / 150 overlap?
# - 800 characters ≈ 150-200 tokens: Optimal dense retrieval granularity
#   for technical post-mortems (captures a complete root-cause paragraph
#   or timeline event without diluting the semantic embedding vector).
# - 150 chars overlap (~18%): Prevents breaking critical error codes
#   and stack traces across chunk boundaries.
# =====================================================================
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def create_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Creates a RecursiveCharacterTextSplitter with separator hierarchy
    tailored for technical incident reports and documentation.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # Double newline (Markdown paragraph / section boundary)
            "\n",    # Single newline (Log lines, list items)
            ". ",    # Sentence boundary
            ", ",    # Clause boundary
            " ",     # Word boundary
            ""       # Fallback: character boundary
        ]
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Splits documents and enriches every resulting chunk with:
    - parent metadata (source, incident_name, page_number)
    - unique chunk_id (e.g. 'crowdstrike_outage_2024_p3_c4')
    - chunk_index and char_count
    """
    splitter = create_splitter(chunk_size, chunk_overlap)
    raw_chunks = splitter.split_documents(documents)

    enriched_chunks: List[Document] = []
    chunk_counters: Dict[str, int] = {}

    for chunk in raw_chunks:
        slug = chunk.metadata.get("incident_slug", "doc")
        page = chunk.metadata.get("page_number", 1)

        counter_key = f"{slug}_p{page}"
        chunk_counters[counter_key] = chunk_counters.get(counter_key, 0) + 1
        chunk_idx = chunk_counters[counter_key]

        # Attach unique chunk identifier for traceability & citations
        chunk.metadata["chunk_id"] = f"{slug}_p{page}_c{chunk_idx}"
        chunk.metadata["chunk_index"] = chunk_idx
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)

        enriched_chunks.append(chunk)

    return enriched_chunks


def get_production_chunks() -> List[Document]:
    """
    Production entry point: Loads all official post-mortems
    and returns standardized, enriched production chunks.
    """
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data" / "postmortems"

    loader = PostmortemDocumentLoader(data_dir=data_dir)
    raw_docs = loader.load_all()
    return split_documents(raw_docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)


def analyze_chunking_strategies(raw_docs: List[Document]):
    """
    Compares 3 different chunking strategies to demonstrate chunk math trade-offs:
    1. Fine-Grained   (400 chars, 80 overlap)
    2. Standard / Prod (800 chars, 150 overlap)
    3. Coarse-Grained (1500 chars, 250 overlap)
    """
    print("=" * 80)
    print(" 📐 IncidentLens — Chunking Strategy Comparison Analysis")
    print("=" * 80)

    configs = [
        ("Fine-Grained", 400, 80),
        ("Standard (Production)", 800, 150),
        ("Coarse-Grained", 1500, 250),
    ]

    print(f"\n{'Configuration':<22} | {'Chunk Size':<10} | {'Overlap':<8} | {'Total Chunks':<12} | {'Avg Chars':<10} | {'Stride (Step)'}")
    print("-" * 80)

    for name, c_size, c_overlap in configs:
        chunks = split_documents(raw_docs, chunk_size=c_size, chunk_overlap=c_overlap)
        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
        stride = c_size - c_overlap  # New non-overlapping content per chunk
        print(f"{name:<22} | {c_size:<10} | {c_overlap:<8} | {len(chunks):<12} | {avg_len:<10.1f} | {stride} chars")

    print("-" * 80)
    print("💡 Mathematical Insight:")
    print("   - Stride = Chunk Size - Overlap.")
    print("   - For Standard (800 / 150), each new chunk advances by 650 chars.")
    print("   - Overlap guarantees context continuity across sentence boundaries.")


def inspect_consecutive_chunks(chunks: List[Document]):
    """
    Demonstrates boundary overlap between two consecutive chunks
    to verify that context is preserved.
    """
    print("\n" + "=" * 80)
    print(" 🔍 Boundary & Overlap Inspection (Consecutive Chunks)")
    print("=" * 80)

    # Pick two consecutive chunks from CrowdStrike or GitHub
    target_chunks = [c for c in chunks if c.metadata.get("incident_slug") == "crowdstrike_outage_2024"]
    if len(target_chunks) < 2:
        target_chunks = chunks[:2]

    c1 = target_chunks[0]
    c2 = target_chunks[1]

    print(f"\n[Chunk 1] ID: {c1.metadata['chunk_id']} | Length: {len(c1.page_content)} chars | Page: {c1.metadata.get('page_number')}")
    print("-" * 80)
    print(c1.page_content[-220:].strip())
    print("\n... [CHUNK BOUNDARY] ...\n")
    print(f"[Chunk 2] ID: {c2.metadata['chunk_id']} | Length: {len(c2.page_content)} chars | Page: {c2.metadata.get('page_number')}")
    print("-" * 80)
    print(c2.page_content[:220].strip())
    print("-" * 80)
    print("✅ Notice how the end of Chunk 1 and the beginning of Chunk 2 share overlapping words.")
    print("   This prevents the embedding model from missing concepts that cross boundaries.")
    print("=" * 80)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data" / "postmortems"

    print("=" * 80)
    print(" ✂️ IncidentLens — Module 03: Text Splitters & Chunking")
    print("=" * 80)

    # 1. Load documents using Module 02
    loader = PostmortemDocumentLoader(data_dir=data_dir)
    raw_docs = loader.load_all()
    print(f"\n[1] Loaded {len(raw_docs)} parent document segments from data/postmortems/.")

    # 2. Compare Chunking Strategies
    analyze_chunking_strategies(raw_docs)

    # 3. Produce Production Chunks
    prod_chunks = split_documents(raw_docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    print(f"\n[2] Production Chunking Applied (Size={DEFAULT_CHUNK_SIZE}, Overlap={DEFAULT_CHUNK_OVERLAP}):")
    print(f"    - Generated {len(prod_chunks)} production-ready chunks.")

    # 4. Inspect Consecutive Chunks
    inspect_consecutive_chunks(prod_chunks)

    # 5. Breakdown by Incident
    print("\n📊 Chunks Breakdown by Incident:")
    incident_counts: Dict[str, int] = {}
    for c in prod_chunks:
        name = c.metadata.get("incident_name", "Unknown")
        incident_counts[name] = incident_counts.get(name, 0) + 1

    for inc, count in incident_counts.items():
        print(f"    - {inc:<30}: {count} chunks")

    print("\n" + "=" * 80)
    print(" ✅ Module 03 Complete: Text Splitters & Chunks Ready for Embeddings (Module 04)")
    print("=" * 80)


if __name__ == "__main__":
    main()
