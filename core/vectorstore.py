"""
Stage 4 — Vector index.

This module stores the vectors and searches them. It is the fourth stage.

What it does
------------
1. It stores each chunk: its text, its vector and its metadata.
2. It searches the store with a query vector.
3. It filters by metadata before the search.
4. It measures the build time and the search time.

Why this module exists
----------------------
The pipeline must be able to test the retrieval design. Two designs are common:

    No filter      One collection holds the whole corpus. A query searches
                   everything. Simple, and correct when the corpus is small.

    With a filter  Metadata removes most of the corpus first, then the vector
                   search runs on the rest. For example, a question about a
                   network incident can exclude every other failure category.

This module supports both, so the ablation table can measure the difference.

The interface rule
------------------
This module receives ready vectors. It does NOT load the embedding model.
Therefore the store can be tested with fixed vectors, and the index can be
rebuilt without paying the embedding cost again.

Normalisation
-------------
Every vector has length 1 (see stage 3). Therefore the squared Euclidean
distance has a direct relation to the cosine similarity:

    cosine_similarity = 1 - (distance ** 2) / 2

The method `similarity_from_distance` below uses this relation.

Usage
-----
    from core.vectorstore import VectorIndex

    index = VectorIndex()
    index.build(chunks, vectors)
    hits = index.search(query_vector, top_k=10)
"""

from __future__ import annotations

import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:                                     # allow "python core/vectorstore.py"
    from core.chunking import Chunk
except ModuleNotFoundError:              # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.chunking import Chunk

# --------------------------------------------------------------------------
# Paths and settings
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DB_DIR = PROJECT / "data" / "chroma_db"
DEFAULT_COLLECTION = "incidentlens"
DEFAULT_SPACE = "l2"        # valid vectors, so l2 and cosine rank the same


class VectorStoreError(RuntimeError):
    """The store could not be opened, built or searched."""


# --------------------------------------------------------------------------
# A search result
# --------------------------------------------------------------------------
@dataclass
class Hit:
    """One search result."""

    chunk_id: str
    document_id: str
    text: str
    distance: float
    similarity: float
    metadata: Dict[str, object] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self, with_text: bool = True) -> Dict[str, object]:
        data = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "distance": round(self.distance, 6),
            "similarity": round(self.similarity, 6),
            "rank": self.rank,
            "metadata": self.metadata,
        }
        if with_text:
            data["text"] = self.text
        return data


def similarity_from_distance(distance: float) -> float:
    """
    Change a squared L2 distance into a cosine similarity.

    Valid because every vector has length 1:
        distance ** 2 = 2 - 2 * cosine
    """
    value = 1.0 - (distance / 2.0)
    return max(-1.0, min(1.0, value))


def clean_metadata(metadata: Dict[str, object]) -> Dict[str, object]:
    """
    Remove the values that the store cannot hold.

    The store accepts a string, an int, a float or a bool. It rejects None and
    a nested object. A rejected value stops the whole insert, so the values are
    cleaned here instead.
    """
    clean: Dict[str, object] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = ", ".join(str(v) for v in value)
        else:
            clean[key] = str(value)
    return clean


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------
class VectorIndex:
    """A persistent vector store for chunks."""

    def __init__(
        self,
        collection: str = DEFAULT_COLLECTION,
        db_dir: Path = DB_DIR,
        space: str = DEFAULT_SPACE,
    ) -> None:
        self.collection_name = collection
        self.db_dir = Path(db_dir)
        self.space = space
        self._client = None
        self._collection = None
        self.build_seconds: float = 0.0
        self.build_count: int = 0

    # ---------------- connection ----------------
    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:                    # pragma: no cover
            raise VectorStoreError(
                "chromadb is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.db_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(
                path=str(self.db_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        except Exception as exc:                      # noqa: BLE001
            raise VectorStoreError(f"cannot open the store at {self.db_dir}: {exc}") from exc
        return self._client

    def collection(self):
        """Return the collection. Raise if it does not exist."""
        if self._collection is not None:
            return self._collection
        client = self._connect()
        try:
            self._collection = client.get_collection(
                name=self.collection_name, embedding_function=None
            )
        except Exception as exc:                      # noqa: BLE001
            raise VectorStoreError(
                f"collection {self.collection_name!r} does not exist. "
                f"Run build() first."
            ) from exc
        return self._collection

    def exists(self) -> bool:
        """True when the collection is present and holds at least one vector."""
        try:
            client = self._connect()
            names = [c.name for c in client.list_collections()]
            if self.collection_name not in names:
                return False
            return self.count() > 0
        except Exception:                             # noqa: BLE001
            return False

    # ---------------- write ----------------
    def build(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        reset: bool = True,
        batch_size: int = 256,
    ) -> Dict[str, object]:
        """
        Store every chunk with its vector.

        reset=True removes an old collection first. A partial overwrite would
        mix two corpora in one index, and every metric would then be wrong.
        """
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"{len(chunks)} chunks but {len(vectors)} vectors"
            )
        if not chunks:
            raise VectorStoreError("nothing to store")

        client = self._connect()

        existing = [c.name for c in client.list_collections()]
        if self.collection_name in existing:
            if not reset:
                raise VectorStoreError(
                    f"collection {self.collection_name!r} already exists. "
                    f"Use reset=True to replace it."
                )
            client.delete_collection(self.collection_name)

        try:
            self._collection = client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": self.space},
            )
        except Exception as exc:                      # noqa: BLE001
            raise VectorStoreError(f"cannot create the collection: {exc}") from exc

        started = time.time()
        for start in range(0, len(chunks), batch_size):
            group = chunks[start:start + batch_size]
            group_vectors = vectors[start:start + batch_size]
            try:
                self._collection.add(
                    ids=[c.chunk_id for c in group],
                    documents=[c.text for c in group],
                    embeddings=[list(v) for v in group_vectors],
                    metadatas=[clean_metadata(c.metadata) for c in group],
                )
            except Exception as exc:                  # noqa: BLE001
                raise VectorStoreError(
                    f"cannot add the batch at position {start}: {exc}"
                ) from exc

        self.build_seconds = time.time() - started
        self.build_count = len(chunks)

        return {
            "collection": self.collection_name,
            "vectors": self.build_count,
            "seconds": round(self.build_seconds, 2),
            "vectors_per_second": round(self.build_count / self.build_seconds, 1)
            if self.build_seconds > 0 else 0.0,
            "space": self.space,
            "db_dir": str(self.db_dir),
        }

    # ---------------- read ----------------
    def count(self) -> int:
        """Return the number of vectors in the collection."""
        return int(self.collection().count())

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        where: Optional[Dict[str, object]] = None,
    ) -> List[Hit]:
        """
        Search the store.

        where is a metadata filter. Example: {"org": "Cloudflare"}.
        The filter runs inside the store, so the search cost stays low.
        """
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than 0")

        try:
            result = self.collection().query(
                query_embeddings=[list(query_vector)],
                n_results=top_k,
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:                      # noqa: BLE001
            raise VectorStoreError(f"the query failed: {exc}") from exc

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: List[Hit] = []
        for rank, chunk_id in enumerate(ids):
            distance = float(distances[rank]) if rank < len(distances) else 0.0
            metadata = dict(metas[rank]) if rank < len(metas) and metas[rank] else {}
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    text=docs[rank] if rank < len(docs) else "",
                    distance=distance,
                    similarity=similarity_from_distance(distance),
                    metadata=metadata,
                    rank=rank,
                )
            )
        return hits

    # ---------------- facts ----------------
    def stats(self) -> Dict[str, object]:
        """Return the facts that stage 4 must report."""
        count = self.count()
        sample = self.collection().get(limit=1, include=["metadatas"])
        metadata_keys: List[str] = []
        if sample.get("metadatas"):
            metadata_keys = sorted(sample["metadatas"][0].keys())

        organisations: Dict[str, int] = {}
        all_meta = self.collection().get(include=["metadatas"])
        for metadata in all_meta.get("metadatas") or []:
            org = str(metadata.get("org", "unknown"))
            organisations[org] = organisations.get(org, 0) + 1

        return {
            "collection": self.collection_name,
            "vectors": count,
            "space": self.space,
            "db_dir": str(self.db_dir),
            "metadata_keys": metadata_keys,
            "by_org": dict(sorted(organisations.items(), key=lambda kv: -kv[1])),
        }

    def reset(self) -> None:
        """Remove the collection and the database directory."""
        try:
            client = self._connect()
            if self.collection_name in [c.name for c in client.list_collections()]:
                client.delete_collection(self.collection_name)
        finally:
            self._collection = None
            if self.db_dir.exists():
                shutil.rmtree(self.db_dir, ignore_errors=True)

    def hit_rate_at_k(self, hits: Sequence[Hit], expected_document: str, k: int) -> int:
        """
        Return 1 when the expected document is in the first k hits, else 0.

        This is the small helper that stage 5 uses to compute Recall@k. It lives
        here because it is a property of a result list, not of a retriever.
        """
        for hit in hits[:k]:
            if hit.document_id == expected_document:
                return 1
        return 0


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    import json

    import core._bootstrap  # noqa: F401  (adds the project root to sys.path)
    from core.ingest import load_corpus
    from core.chunking import chunk_corpus
    from core.embeddings import embed_chunks, describe_model

    size, overlap = 480, 120
    for argument in sys.argv[1:]:
        if argument.startswith("--size="):
            size = int(argument.split("=", 1)[1])
        elif argument.startswith("--overlap="):
            overlap = int(argument.split("=", 1)[1])

    print("=" * 78)
    print(" STAGE 4 — VECTOR INDEX")
    print("=" * 78)

    index = VectorIndex()

    # A rebuild is expensive: the embedding of 1,000+ chunks takes time.
    # Use the existing index when it is present, unless --rebuild is given.
    if index.exists() and "--rebuild" not in sys.argv:
        stats = index.stats()
        print(f" collection       : {stats['collection']} (already built)")
        print(f" vectors          : {stats['vectors']:,}")
        print(f" space            : {stats['space']}")
        print(f" db dir           : {stats['db_dir']}")
        print("\n by organization")
        for org, count in stats["by_org"].items():
            print(f"   {org:<24} {count:>5}")
        print("\n Use --rebuild to build the index again.")
        return 0

    info = describe_model()
    print(f" model            : {info['model']} ({info['dimension']}-dim)")

    documents = load_corpus()
    chunks = chunk_corpus(documents, size=size, overlap=overlap)
    print(f" chunks           : {len(chunks):,}   ({size}/{overlap})")

    print("\n embedding...")
    vectors, embed_stats = embed_chunks(chunks, show_progress=True)

    print("\n building the index...")
    build = index.build(chunks, vectors, reset=True)

    stats = index.stats()

    print("\n RESULT")
    print(f" collection       : {build['collection']}")
    print(f" vectors stored   : {build['vectors']:,}")
    print(f" build time       : {build['seconds']} s")
    print(f" insert speed     : {build['vectors_per_second']:,} vectors/second")
    print(f" embedding time   : {embed_stats.seconds:.1f} s")
    print(f" distance space   : {build['space']}")
    print(f" metadata keys    : {stats['metadata_keys']}")
    print(f" db dir           : {build['db_dir']}")

    print("\n by organization (in the store)")
    for org, count in stats["by_org"].items():
        print(f"   {org:<24} {count:>5}")

    # A sanity search, to prove the store answers.
    from core.embeddings import embed_query

    question = "What caused the Cloudflare outage on 18 November 2025?"
    hits = index.search(embed_query(question), top_k=5)
    print(f"\n SANITY SEARCH: {question!r}")
    for hit in hits:
        print(f"   {hit.rank + 1}. {hit.chunk_id:<28} sim={hit.similarity:.4f} "
              f"doc={hit.document_id}")

    correct = sum(1 for h in hits if h.document_id == "cf-2025-11-18")
    print(f"\n the expected document is in the first 5 hits: "
          f"{'YES' if correct else 'NO'}")

    if "--json" in sys.argv:
        out = PROJECT / "data" / "index_report.json"
        out.write_text(json.dumps(
            {"model": info, "build": build, "stats": stats,
             "embedding": embed_stats.as_dict(),
             "sanity_search": [h.to_dict(with_text=False) for h in hits]},
            indent=2), encoding="utf-8")
        print(f" report written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
