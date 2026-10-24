"""
IncidentLens core package.

One module for each stage of the pipeline:

    1  ingest        read the documents
    2  chunking      split the documents
    3  embeddings    make the vectors
    4  vectorstore   store the vectors
    5  retrieval     find the chunks
    6  fusion        merge two result lists
    7  rerank        order the chunks better
    8  contextual    add a header to each chunk
    9  generate      make the answer
   10  evaluate      measure the quality

The interface rule: the pipeline calls a stage. A stage does not call the
pipeline. Therefore each stage can operate alone and be measured alone.
"""

__all__ = [
    "ingest", "chunking", "embeddings", "vectorstore",
    "retrieval", "bm25", "fusion", "rerank",
    "contextual", "generate", "evaluate", "providers",
    "query_expansion",
]
__version__ = "2.1.0"

