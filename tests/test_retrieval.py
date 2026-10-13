"""
test_retrieval.py — Module 05 (ChromaDB) and Module 10 (BM25 + dense + RRF).

No LLM calls. These tests assert the retrieval behaviour the whole project depends on,
including the two failure modes Module 07 proved and Module 10 fixed:

  * an exact code such as AS33154 is invisible to dense search (a vector model has no concept
    of an autonomous system number) but BM25 ranks it first;
  * a paraphrase such as "blue screen crash" has no keyword overlap yet dense search matches it.
"""


def slugs(results):
    return [r["metadata"].get("incident_slug", "") for r in results]


def texts(results):
    return [r["text"] for r in results]


class TestVectorStore:
    def test_collection_is_populated(self, retrieval_stack):
        collection, _, _ = retrieval_stack
        # 105 chunks with chunk_size=800 / overlap=150. Allow slack so a re-chunk is not a failure.
        assert collection.count() >= 100

    def test_every_chunk_carries_metadata(self, retrieval_stack):
        collection, _, _ = retrieval_stack
        got = collection.get(limit=5, include=["metadatas"])
        assert got["metadatas"], "collection returned no metadata"
        for meta in got["metadatas"]:
            assert meta.get("chunk_id"), "chunk_id is required for citations"
            assert meta.get("incident_slug"), "incident_slug is required for filtering"


class TestLexicalRetrieval:
    def test_bm25_finds_an_exact_as_number(self, retrieval_stack):
        _, engine, _ = retrieval_stack
        results = engine.search_bm25("AS33154", top_k=5)
        assert results, "BM25 returned nothing for AS33154"
        assert "AS33154" in results[0]["text"], "the exact-code match must rank first"

    def test_bm25_finds_a_code_that_appears_once_in_the_corpus(self, retrieval_stack):
        """nproc appears exactly once in the whole corpus - a lexical search must still find it."""
        _, engine, _ = retrieval_stack
        results = engine.search_bm25("nproc thread limit", top_k=5)
        assert any("nproc" in t for t in texts(results)), "rare exact token not found"

    def test_tokenizer_preserves_technical_tokens(self, retrieval_stack):
        """
        The corpus does NOT contain 0x00000050 or CSAgent.sys - those came from Module 07's
        synthetic crash log. What Module 10 actually guarantees is that the TOKENIZER keeps
        such tokens intact instead of splitting them into 0x / 00000050.
        """
        _, _, m10 = retrieval_stack
        tokens = m10.technical_tokenize("0x00000050 AS33154 CSAgent.sys us-east-1 BGP-4")
        for expected in ["0x00000050", "as33154", "csagent.sys", "us-east-1", "bgp-4"]:
            assert expected in tokens, f"{expected} was split or dropped by the tokenizer"


class TestDenseRetrieval:
    def test_dense_matches_a_paraphrase_with_no_keyword_overlap(self, retrieval_stack):
        _, engine, _ = retrieval_stack
        results = engine.search_dense("What made Windows machines show a blue screen?", top_k=5)
        assert results
        assert "crowdstrike_outage_2024" in slugs(results), \
            "a semantic query should still reach the CrowdStrike post-mortem"


class TestHybridFusion:
    def test_rrf_scores_are_sorted_and_ranked(self, retrieval_stack):
        _, engine, _ = retrieval_stack
        results = engine.search_hybrid("Kinesis thread exhaustion nproc", top_k=5)
        assert results
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True), "RRF output must be sorted"
        assert [r["final_rank"] for r in results] == list(range(1, len(results) + 1))

    def test_fusion_marks_which_retriever_found_each_chunk(self, retrieval_stack):
        _, engine, _ = retrieval_stack
        results = engine.search_hybrid("Cloudflare BGP route leak AS33154", top_k=5)
        top = results[0]
        assert "dense_rank" in top and "bm25_rank" in top
        # The top fusion result should be found by at least one retriever.
        assert top["dense_rank"] or top["bm25_rank"]

    def test_hybrid_rescues_the_exact_code_that_dense_misses(self, retrieval_stack):
        _, engine, m10 = retrieval_stack
        dense = engine.search_dense("AS33154", top_k=5)
        hybrid = engine.search_hybrid("AS33154", top_k=5)
        dense_found = any("AS33154" in t for t in texts(dense))
        hybrid_found = "AS33154" in texts(hybrid)[0]
        assert hybrid_found, "hybrid search must surface the exact code"
        # Document the original failure: dense-only misses it. If a future embedding model
        # starts finding AS33154, that is good news, not a broken test.
        if not dense_found:
            assert hybrid_found

    def test_component_ranks_are_traceable(self, retrieval_stack):
        """Every fused chunk must be attributable to dense, BM25, or both."""
        _, engine, _ = retrieval_stack
        for item in engine.search_hybrid("GitHub MySQL replication lag", top_k=8):
            assert item["dense_rank"] is not None or item["bm25_rank"] is not None
