"""
02_document_loaders.py
======================
Phase 1: Ingestion & Document Loaders
IncidentLens Post-Mortem Loader:
1. Discovers and loads both PDF and Markdown files from data/postmortems/.
2. Uses PyPDFLoader for multi-page PDF documents.
3. Uses TextLoader for Markdown post-mortems.
4. Normalizes and enriches metadata (incident_name, file_type, source, char_count).
5. Provides an inspection summary of loaded documents for downstream chunking.
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
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class PostmortemDocumentLoader:
    """
    Unified loader for technical disaster post-mortems supporting
    both multi-page PDFs and structured Markdown documents.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load_all(self) -> List[Document]:
        """
        Scans data/postmortems/ and loads all supported documents.
        Returns a list of standardized LangChain Document objects.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Directory '{self.data_dir}' not found. "
                "Ensure data/postmortems/ exists and contains post-mortem files."
            )

        all_documents: List[Document] = []
        files = sorted(list(self.data_dir.glob("*.*")))
        supported_files = [f for f in files if f.suffix.lower() in [".pdf", ".md"]]

        if not supported_files:
            print(f"⚠️ No .pdf or .md files found in {self.data_dir}")
            return []

        for file_path in supported_files:
            ext = file_path.suffix.lower()
            incident_slug = file_path.stem
            incident_name = incident_slug.replace("_", " ").title()

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    for idx, doc in enumerate(docs):
                        # Standardize metadata across loaders
                        doc.metadata["source_file"] = file_path.name
                        doc.metadata["incident_slug"] = incident_slug
                        doc.metadata["incident_name"] = incident_name
                        doc.metadata["file_type"] = "pdf"
                        doc.metadata["page_number"] = doc.metadata.get("page", idx) + 1
                        doc.metadata["char_count"] = len(doc.page_content)
                    all_documents.extend(docs)

                elif ext == ".md":
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_file"] = file_path.name
                        doc.metadata["incident_slug"] = incident_slug
                        doc.metadata["incident_name"] = incident_name
                        doc.metadata["file_type"] = "markdown"
                        doc.metadata["page_number"] = 1
                        doc.metadata["char_count"] = len(doc.page_content)
                    all_documents.extend(docs)

            except Exception as e:
                print(f"❌ Failed to load {file_path.name}: {e}")

        return all_documents


def print_ingestion_summary(documents: List[Document]):
    """Prints a structured summary table of all loaded post-mortem documents."""
    print("=" * 80)
    print(" 📄 IncidentLens — Phase 1: Document Ingestion Summary")
    print("=" * 80)

    if not documents:
        print("❌ No documents loaded. Please verify your dataset files.")
        return

    # Aggregate stats per file
    file_stats: Dict[str, Dict[str, Any]] = {}
    total_characters = 0

    for doc in documents:
        fname = doc.metadata.get("source_file", "unknown")
        ftype = doc.metadata.get("file_type", "unknown")
        chars = doc.metadata.get("char_count", len(doc.page_content))
        total_characters += chars

        if fname not in file_stats:
            file_stats[fname] = {
                "type": ftype,
                "pages": 0,
                "total_chars": 0,
                "incident": doc.metadata.get("incident_name", fname)
            }
        file_stats[fname]["pages"] += 1
        file_stats[fname]["total_chars"] += chars

    print(f"\n{'File Name':<35} | {'Type':<8} | {'Pages':<6} | {'Total Chars':<12} | {'Incident'}")
    print("-" * 80)
    for fname, stats in file_stats.items():
        print(f"{fname:<35} | {stats['type']:<8} | {stats['pages']:<6} | {stats['total_chars']:<12} | {stats['incident']}")

    print("-" * 80)
    print(f"Total Unique Files: {len(file_stats)}")
    print(f"Total Ingested Document Segments: {len(documents)}")
    print(f"Total Ingested Characters: {total_characters:,}")

    # Inspect the first document segment metadata
    print("\n🔍 Sample Metadata Inspection (First Loaded Segment):")
    sample_doc = documents[0]
    for key, val in sample_doc.metadata.items():
        print(f"    - {key:<15}: {val}")
    print(f"\n    Preview:\n    \"{sample_doc.page_content.strip()[:200]}...\"")
    print("=" * 80)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data" / "postmortems"
    
    loader = PostmortemDocumentLoader(data_dir=data_dir)
    documents = loader.load_all()
    print_ingestion_summary(documents)


if __name__ == "__main__":
    main()
