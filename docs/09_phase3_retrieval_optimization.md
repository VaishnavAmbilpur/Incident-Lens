# 🔹 09 — Phase 3: Retrieval Optimization & Quality Engineering

> **Status:** ⬜ Not Started  
> **Focus:** Upgrading naive search into an enterprise-grade hybrid retrieval engine.  
> **Prerequisite:** Phase 2 complete — all 5 failure modes understood and documented.

---

## 3.1 Engineering Concepts

### Hybrid Search (Dense + BM25)
- **Dense (BGE-M3):** Captures semantic meaning — "database failure" ≈ "DB starvation"
- **Sparse (BM25):** Captures exact lexical matches — `SIGSEGV`, `0x00000050`, `CSAgent.sys`
- **Reciprocal Rank Fusion (RRF):** Merges both ranked lists with the formula:
  ```
  RRF_score(doc) = Σ 1 / (k + rank_i(doc))   where k = 60
  ```
- **Why Both:** Dense alone misses error codes. BM25 alone misses semantic relationships.

### Token Budgeting
- **Problem:** LLMs have finite context windows. Stuffing too many chunks wastes tokens and triggers "lost in the middle."
- **Solution:** Dynamically calculate:
  ```
  available_context = model_max_tokens - system_prompt_tokens - output_reserve
  chunks_to_include = fit_chunks_within_budget(ranked_chunks, available_context)
  ```

### Query Expansion & Multi-Query
- **Technique:** Generate 3 alternate formulations of the user query to maximize recall:
  ```
  Original: "Why did Kafka stall?"
  Variant 1: "What caused Kafka message queue to stop processing?"
  Variant 2: "Kafka consumer lag and queue backup root cause"
  Variant 3: "Kafka stall memory pressure thread exhaustion"
  ```
- **Why:** Different formulations match different chunks in the vector space.

### Cross-Encoder Reranking
- **Problem:** Bi-encoder (embedding) similarity is a rough approximation.
- **Solution:** Pass the top 20 candidates through BGE-Reranker-Large (cross-encoder) to get precise relevance scores.
- **Result:** Top-K after reranking has 80%+ fewer false positives.

### Citation Grounding
- **Rule:** Every claim in the output MUST cite `[Source: document_name, page X]`.
- **Implementation:** Post-processing step that verifies each sentence can be traced back to a retrieved chunk.

---

## 3.2 Deliverable Files

| File | Description | Verification |
|---|---|---|
| `modules/09_token_budgeting.py` | Dynamic context window optimizer | Prints budget calculation for varying chunk counts |
| `modules/10_hybrid_search_bm25.py` | Dense + Sparse RRF hybrid retriever | Query `"bugcheck 0x50"` returns exact match AND semantic matches |
| `modules/11_query_transformations.py` | Multi-query generator and query rewrites | Generates 3 variants, retrieves with each, merges results |
| `modules/12_reranking_pipeline.py` | BGE cross-encoder reranking | Shows before/after reranking scores — noise eliminated |

---

## 3.3 Key Technical Details

### RRF Fusion Implementation
```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """Merge dense and sparse ranked lists using RRF."""
    fused_scores = {}
    for rank, doc in enumerate(dense_results):
        fused_scores[doc.id] = fused_scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        fused_scores[doc.id] = fused_scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

### Token Budget Formula
```
Model: Llama 3.3 70B → max_tokens = 8192
System prompt: ~500 tokens
Output reserve: ~1024 tokens
Available for context: 8192 - 500 - 1024 = 6668 tokens
Average chunk: ~300 tokens
Max chunks: floor(6668 / 300) = 22 chunks
```

---

## 3.4 Success Criteria (Gate to Phase 4)

- [ ] Hybrid search returns exact error codes that dense-only missed
- [ ] RRF fusion ranking is measurably better than either engine alone
- [ ] Token budgeting prevents context overflow for all test queries
- [ ] Multi-query improves recall by ≥15% over single-query
- [ ] Reranker eliminates ≥70% of noise from top-20 results
- [ ] All outputs include source citations

---

## 🔗 Next

→ [10_phase4_observability.md](./10_phase4_observability.md) — Measuring quality with Ragas & LangSmith.
