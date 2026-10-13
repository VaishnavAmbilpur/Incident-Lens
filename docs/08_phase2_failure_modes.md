# 🔹 08 — Phase 2: Diagnosing & Debugging the 5 Failure Modes

> **Status:** ⬜ Not Started  
> **Focus:** Reproducing and diagnosing the 5 critical failure modes that break naive RAG in production.  
> **Prerequisite:** Phase 1 complete — naive RAG baseline working.

---

## 2.1 The 5 Failure Modes

### Failure 1: Bad Chunking
**Problem:** Code snippets and error stack traces get split mid-line by the chunker.  
**Symptom:** Retrieved chunk contains half a stack trace — the LLM can't interpret it.  
**Example:**
```
# Chunk boundary falls HERE ↓
File "csagent.sys", line 291, in process_channel_
---chunk-break---
file: content_validator raised WHEA_UNCORRECTABLE_ERROR
```

### Failure 2: Semantic Mismatch
**Problem:** User searches plain-English terms (`"blue screen crash"`), but documents use formal terms (`"system bugcheck 0x50 memory offset"`).  
**Symptom:** Dense embeddings return 0 relevant results because the semantic gap is too wide.  
**Why It Matters:** On-call engineers at 2 AM don't use academic terminology.

### Failure 3: Retrieval Noise
**Problem:** Vector search returns 10 chunks, but 8 are irrelevant fluff.  
**Symptom:** The LLM gets confused by noise and produces a generic, unhelpful answer.  
**Root Cause:** No reranking step — raw cosine similarity isn't selective enough.

### Failure 4: Context Overflow & "Lost in the Middle"
**Problem:** When you stuff 15+ chunks into a prompt, the LLM forgets information in the middle.  
**Symptom:** The answer only references the first and last chunks, missing the critical root cause buried in chunk #7.  
**Research:** Documented in "Lost in the Middle" (Liu et al., 2023).

### Failure 5: Hallucination
**Problem:** The LLM fabricates a fix command when the source documents don't specify one.  
**Symptom:** The model outputs `sudo rm -rf /var/kafka/` as a "recovery step" — a destructive command it invented.  
**Danger Level:** 🔴 **Critical** — executing hallucinated commands in production can cause data loss.

---

## 2.2 Deliverable Files

| File | Description | Verification |
|---|---|---|
| `modules/07_rag_failure_modes_demo.py` | Stress tests deliberately triggering all 5 failures | Each failure is reproducible and logged |
| `modules/08_debugging_rag.py` | Diagnostic toolkit: similarity score distribution, token inspector, chunk boundary visualizer | Prints diagnostic reports for each failure |

---

## 2.3 What Each File Does

### `07_rag_failure_modes_demo.py`
1. **Test 1 (Bad Chunking):** Chunk a stack trace with small chunk size, show the break.
2. **Test 2 (Semantic Mismatch):** Query `"blue screen crash"` against docs containing `"bugcheck 0x50"` — show 0 relevant hits.
3. **Test 3 (Retrieval Noise):** Retrieve 10 chunks, manually score relevance, show 80% are noise.
4. **Test 4 (Lost in Middle):** Stuff 15 chunks into prompt, verify the answer ignores middle chunks.
5. **Test 5 (Hallucination):** Ask for a fix when docs have none, catch the fabricated command.

### `08_debugging_rag.py`
- **Similarity Score Histogram:** Distribution of cosine scores for a given query.
- **Token Count Inspector:** Count tokens per chunk and total prompt tokens.
- **Chunk Boundary Viewer:** Visual display of where chunks begin/end in the source document.
- **Relevance Scoring Matrix:** Manual relevance labels vs. retriever ranking.

---

## 2.4 Success Criteria (Gate to Phase 3)

- [ ] All 5 failures are reproducible and documented with examples
- [ ] Diagnostic toolkit identifies each failure type programmatically
- [ ] Clear understanding of *why* each failure occurs and *what* Phase 3 technique fixes it

### Failure → Fix Mapping (Preview)

| Failure | Phase 3 Fix |
|---|---|
| Bad Chunking | Contextual retrieval headers + late chunking |
| Semantic Mismatch | BM25 hybrid search catches exact codes |
| Retrieval Noise | Cross-encoder reranker filters false positives |
| Context Overflow | Token budgeting + smart truncation |
| Hallucination | Agentic hallucination check + citation grounding |

---

## 🔗 Next

→ [09_phase3_retrieval_optimization.md](./09_phase3_retrieval_optimization.md) — Fixing all 5 failures with enterprise retrieval.
