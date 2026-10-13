# 📊 13 — Quality Benchmarks & Execution Protocol

> **Context:** Target metrics, measurement tools, and the 4-step execution workflow for every module.  
> **Prerequisite:** Read all phase docs (07–12) for implementation context.

---

## 13.1 Quality Benchmark Matrix

| Dimension | Naive RAG (Phase 1) | IncidentLens Target (Phase 6) | Measurement Tool |
|---|:---:|:---:|:---:|
| **Faithfulness (No Hallucination)** | ~64% | **≥ 92%** | Ragas `faithfulness` |
| **Context Precision** | ~58% | **≥ 88%** | Ragas `context_precision` |
| **Answer Relevance** | ~71% | **≥ 90%** | Ragas `answer_relevance` |
| **Context Recall** | ~55% | **≥ 85%** | Ragas `context_recall` |
| **Exact Code / Flag Recall** | ~40% (fails on exact strings) | **≥ 95%** | BM25 Hybrid + RRF |
| **Prompt Injection Defense** | 0% (vulnerable) | **100% blocked** | Security Guardrail Test Suite |
| **P95 Retrieval Latency** | > 2.5s | **< 600ms** | LangSmith / Langfuse Traces |
| **P95 End-to-End Latency** | > 8s | **< 2s** | LangSmith / Langfuse Traces |
| **Cache Hit Latency** | N/A | **< 50ms** | Semantic Cache Benchmark |

---

## 13.1.1 Corrected Scorecard Result (Module 15)

**Artifact:** `docs/module15_scorecard_corrected.txt` (24,752 bytes).

This table records what the corrected Module 15 scorecard MEASURED against the targets in 13.1.

| Metric | Recorded result | Target (13.1) | Result |
|---|:---:|:---:|---|
| **Context Precision** | **0.920** | >= 88% | **PASS** |
| **Context Recall** | **0.770** | >= 85% | **FAIL** |
| **Faithfulness** | NOT MEASURED | >= 92% | n/a |
| **Answer Relevance** | NOT MEASURED | >= 90% | n/a |

Notes:

- The previously recorded Context Precision of 0.600 was a MEASUREMENT ARTIFACT, not a retrieval verdict. Three measurement defects caused it: the judge saw only the first 200 characters of each 800-character chunk, a judge exception silently became a 0.5 score, and content passed as LangChain template text let braces crash the judge. With the measurement corrected, Context Precision is 0.920 and it PASSES the >= 88% target.
- Decomposition of the old recorded number: 0.600 = qwen judge + 200-character truncation + the silent-0.5 masking defect. 0.780 = allam-2-7b judge + 200-character truncation, so +0.18 is judge calibration. 0.920 = allam-2-7b judge + whole chunks, so +0.14 is the truncation fix alone.
- Context Recall's context was never truncated, so its small change from 0.750 to 0.770 is judge calibration only. The shortfall is REAL retrieval quality, confirmed under two different judges, and it remains below target.
- Faithfulness (pre-fix 0.950) and Answer Relevancy (pre-fix 0.950) have no corrected measurement. The provider was out of quota (HTTP 429), so 10 of 10 answers failed to generate. The two answer-dependent metrics are reported as NOT MEASURED, and the overall verdict is NOT FULLY MEASURED. A full re-run is required when the provider has quota.

---

## 13.2 Benchmark Measurement Points

| After Phase | What to Measure | Expected Improvement |
|---|---|---|
| **Phase 1** | Baseline Ragas scores (naive RAG) | Establish starting metrics |
| **Phase 2** | Document failure modes (qualitative) | Understand gaps |
| **Phase 3** | Re-run Ragas with hybrid search + reranking | +15-25% on precision & recall |
| **Phase 4** | Full golden dataset eval (20 questions) | Quantified baseline with traces |
| **Phase 5** | Security test pass rate + latency benchmarks | 100% injection block, < 2s P95 |
| **Phase 6** | Final Ragas scorecard with all advanced techniques | Hit target benchmarks. Corrected result: Context Precision 0.920 (PASS); Context Recall 0.770 (FAIL); Faithfulness and Answer Relevancy NOT MEASURED (generation outage). See 13.1.1. |

---

## 13.3 Execution Workflow Protocol

> **Every module follows this exact 4-step protocol:**

### Step 1: Module Architecture & Theory
- Concise explanation of **what** the module does
- **Why** it's required in production
- Which failure mode it addresses (from Phase 2)

### Step 2: Terminal Instructions
- The exact commands to run in PowerShell
- Any environment variables or prerequisites
- Example:
  ```powershell
  cd Production-Level-RAG
  .\venv\Scripts\Activate.ps1
  python modules/10_hybrid_search_bm25.py
  ```

### Step 3: File Creation & Code
- Clean, commented, production-grade Python
- Follows project style guide (type hints, docstrings, error handling)
- Imports from `src/` package where applicable

### Step 4: Execution & Metric Verification
- Run the script
- Validate output against expected results
- **Certify the milestone** before proceeding to the next file
- Log metrics to LangSmith/Langfuse where applicable

---

## 13.4 Phase Completion Checklist

Use this as a master tracking sheet:

### Phase 1: Foundations
- [ ] `01_environment_check.py` — runs clean
- [ ] `02_document_loaders.py` — all 4 docs loaded
- [ ] `03_text_splitters.py` — chunks validated
- [ ] `04_embeddings_deep_dive.py` — similarities computed
- [ ] `05_vector_stores_chroma.py` — indexed and queryable
- [ ] `06_basic_rag_pipeline.py` — naive baseline working

### Phase 2: Failure Modes
- [ ] `07_rag_failure_modes_demo.py` — all 5 failures reproduced
- [ ] `08_debugging_rag.py` — diagnostics toolkit operational

### Phase 3: Retrieval Optimization
- [ ] `09_token_budgeting.py` — budget math validated
- [ ] `10_hybrid_search_bm25.py` — RRF fusion working
- [ ] `11_query_transformations.py` — multi-query generating variants
- [ ] `12_reranking_pipeline.py` — reranker filtering noise

### Phase 4: Observability & Evaluation
- [ ] `13_langsmith_setup.py` — traces in dashboard
- [ ] `14_ragas_evaluation.py` — 4-metric scorecard printed
- [ ] `15_langfuse_observability.py` — Langfuse traces working
- [ ] `16_chroma_production.py` — persistent storage verified
- [ ] `17_cost_and_caching.py` — cache hits under 50ms
- [ ] `eval_golden_dataset.json` — 20 Q&A pairs validated

### Phase 5: Security & API
- [ ] `18_security_guardrails.py` — injections blocked
- [ ] `19_langgraph_agent.py` — stateful agent with retries
- [ ] `20_api_server.py` — FastAPI serving requests
- [ ] `test_security.py` — all tests pass
- [ ] `test_retrieval.py` — all tests pass
- [ ] `test_incidentlens.py` — all tests pass

### Phase 6: Advanced RAG
- [ ] `21_long_context_vs_rag.py` — benchmark table printed
- [ ] `22_contextual_retrieval.py` — situation headers working
- [ ] `23_late_chunking.py` — attention preserved
- [ ] `24_agentic_rag.py` — self-correction loop working
- [ ] `25_graphrag_reasoning.py` — multi-hop queries answered
- [ ] `26_multimodal_colpali.py` — charts extracted and indexed

### Final Verification
- [ ] Ragas Faithfulness ≥ 92% — NOT MEASURED in the corrected run (pre-fix 0.950); pending a quota window
- [x] Ragas Context Precision ≥ 88% — PASS at 0.920 (corrected measurement; the earlier 0.600 was a measurement artifact)
- [ ] Ragas Answer Relevance ≥ 90% — NOT MEASURED in the corrected run (pre-fix 0.950); pending a quota window
- [ ] Ragas Context Recall ≥ 85% — FAIL at 0.770 (real retrieval quality, confirmed under two judges)
- [ ] P95 Latency < 2s
- [ ] 100% Injection Defense
- [ ] All pytest tests pass

---

## 🔗 Back to Index

→ [00_ROADMAP_INDEX.md](./00_ROADMAP_INDEX.md) — Master navigation.
