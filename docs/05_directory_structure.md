# 📁 05 — Directory Structure

> **Context:** The repository layout.
> **Prerequisite:** [02_architecture_overview.md](./02_architecture_overview.md), [14_pipeline_plan.md](./14_pipeline_plan.md)
> **Revision:** 2. The module numbers in revision 1 were wrong. Read section 5.5.

---

## 5.1 The Tree Today

```text
Production-Level-RAG/
│
├── .env                              # API keys and endpoints. Do not commit this file.
├── .env.example                      # The key names, with no values
├── .gitignore                        # venv, __pycache__, .chroma, *.log
├── requirements.txt                  # The Python dependencies
├── pytest.ini                        # The pytest configuration
├── STATUS_CHECKPOINT.md              # The verified state of the project
│
├── docs/                             # The documentation
│   ├── 00_ROADMAP_INDEX.md           # The entry point
│   ├── 01_problem_and_solution.md
│   ├── 02_architecture_overview.md
│   ├── 03_tech_stack.md
│   ├── 04_system_diagrams.md
│   ├── 05_directory_structure.md     # This file
│   ├── 06_datasets.md
│   ├── 07_phase1_foundations.md
│   ├── 08_phase2_failure_modes.md
│   ├── 09_phase3_retrieval_optimization.md
│   ├── 10_phase4_observability.md
│   ├── 11_phase5_security_api.md
│   ├── 12_phase6_advanced_rag.md
│   ├── 13_quality_benchmarks.md
│   ├── 14_pipeline_plan.md           # The plan for the change
│   ├── 15_evals.md                   # The ablation table and the numbers
│   ├── 16_observability.md           # LangSmith and Langfuse
│   ├── 17_interview_defense.md       # One answer for each stage
│   ├── module15_scorecard_corrected.txt
│   └── module15_scorecard_verified.txt
│
├── data/                             # The datasets and the eval set
│   ├── postmortems/                  # The incident documents
│   │   ├── crowdstrike_outage_2024.pdf
│   │   ├── cloudflare_bgp_leak.md
│   │   ├── aws_kinesis_outage.md
│   │   └── github_mysql_failover.md
│   ├── download_official_documents.py
│   ├── eval_golden_dataset.json      # 10 curated questions with true citations
│   ├── chroma_db/                    # The vector index. Do not commit this directory.
│   └── agent_memory.sqlite           # The agent memory. Do not commit this file.
│
├── modules/                          # 26 standalone scripts. Phases 1 to 6.
│   ├── 01_environment_check.py
│   ├── 02_document_loaders.py        # The real function: load the documents
│   ├── 03_text_splitters.py          # The real function: split the text
│   ├── 04_embeddings_deep_dive.py
│   ├── 05_vector_stores_chroma.py
│   ├── 06_basic_rag_pipeline.py
│   ├── 07_rag_failure_modes_demo.py
│   ├── 08_debugging_rag.py
│   ├── 09_token_budgeting.py
│   ├── 10_hybrid_search_bm25.py
│   ├── 11_query_transformations.py
│   ├── 12_reranking_pipeline.py
│   ├── 13_optimized_rag_pipeline.py
│   ├── 14_langsmith_setup.py
│   ├── 15_ragas_evaluation.py
│   ├── 16_langfuse_observability.py
│   ├── 17_semantic_cache.py
│   ├── 18_security_guardrails.py
│   ├── 19_langgraph_agent.py
│   ├── 20_api_server.py
│   ├── 21_long_context_vs_rag.py
│   ├── 22_contextual_retrieval.py
│   ├── 23_late_chunking.py
│   ├── 24_agentic_rag.py
│   ├── 25_graphrag_reasoning.py
│   └── 26_multimodal_colpali.py
│
├── frontend/                         # The React interface
│   ├── src/App.jsx
│   ├── src/components/               # 20 components
│   └── src/lib/
│
├── learning/                         # The session logs
│   ├── 01_six_core_rag_architectures.md
│   ├── 02_session_log_phase0_to_phase4.md
│   ├── 03_session_log_phase5.md
│   └── 04_session_log_phase6.md
│
└── tests/                            # The test suite. 36 tests.
    ├── conftest.py
    ├── test_incidentlens.py          # The API tests
    ├── test_retrieval.py
    └── test_security.py
```

---

## 5.2 The Tree After the Change

The plan of [14_pipeline_plan.md](./14_pipeline_plan.md) makes one library, `core/`. The 26 scripts become 10 stage modules.

```text
Production-Level-RAG/
│
├── core/                             # THE LIBRARY. One module for each stage.
│   ├── __init__.py
│   ├── ingest.py                     # Stage 1
│   ├── chunking.py                   # Stage 2
│   ├── embeddings.py                 # Stage 3
│   ├── vectorstore.py                # Stage 4
│   ├── retrieval.py                  # Stage 5
│   ├── fusion.py                     # Stage 6
│   ├── rerank.py                     # Stage 7
│   ├── contextual.py                 # Stage 8
│   ├── generate.py                   # Stage 9
│   ├── evaluate.py                   # Stage 10
│   ├── providers.py                  # The fallback chain and the cache
│   ├── tracing.py                    # One interface for the two tracing tools
│   ├── tracing_langsmith.py
│   ├── tracing_langfuse.py
│   └── guardrails.py                 # The injection gate and the PII filter
│
├── api/
│   └── server.py                     # The FastAPI service. It calls core/.
│
├── ablation.py                       # The 6 configurations. It makes Table 15-3.
├── diagnose.py                       # The per-stage report for one query
├── cli.py                            # The command line interface
│
├── modules/                          # The old scripts. Keep for reference only.
├── docs/                             # The documentation
├── data/                             # The datasets
├── frontend/src/                     # One page, 3 components
├── tests/                            # About 120 tests
└── requirements.txt
```

---

## 5.3 The Purpose of Each Directory

**Table 5-1 — The Directories**

| Directory | Purpose | Status |
|---|---|---|
| `core/` | One module for each stage. The library. | Make new |
| `api/` | The HTTP service. It calls `core/`. | Change |
| `modules/` | The 26 learning scripts. Reference only. | Freeze |
| `data/postmortems/` | The incident documents. | Expand to 25 to 40 |
| `data/chroma_db/` | The vector index. | Do not commit |
| `frontend/` | The pipeline view. One page. | Reduce |
| `tests/` | The test suite. | Expand |
| `docs/` | The documentation. | Changed |

---

## 5.4 The Import Rule

The directory `modules/` has names such as `02_document_loaders.py`. A name that starts with a digit is not a legal Python module name. Therefore, module 20 uses a helper function to load these files by file name.

**Table 5-2 — The Import Problem**

| Item | Today | After the change |
|---|---|---|
| Stage code | In a file with a digit prefix | In `core/`, with a legal name |
| Import method | A helper function that loads by file name | A normal `import` |
| Shared helpers | The same function in every file | One function in one module |

> **NOTE:** A normal `import` statement is a sign of a library. A load-by-file-name helper is a sign of a script collection. This change is the main reason for `core/`.

---

## 5.5 Correction to the First Version of This File

The first version of this file contained these errors. Table 5-3 gives the corrections.

**Table 5-3 — The Corrections**

| Item | Revision 1 said | The truth |
|---|---|---|
| Module 04 | `04_embeddings_deep_dive.py` uses BGE-M3 | It uses `all-MiniLM-L6-v2`, 384 dimensions |
| Module 10 | "BGE-M3 + BM25 with RRF" | MiniLM dense search + BM25 with RRF |
| Module 12 | "BGE cross-encoder reranking" | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Module 13 | `13_langsmith_setup.py` | The file is `13_optimized_rag_pipeline.py`. LangSmith is module 14. |
| Module 14 | `14_ragas_evaluation.py` | The evaluation file is `15_ragas_evaluation.py` |
| Module 15 | `15_langfuse_observability.py` | The Langfuse file is `16_langfuse_observability.py` |
| Module 16 | `16_chroma_production.py` | No such file. The cache file is `17_semantic_cache.py`. |
| Module 17 | `17_cost_and_caching.py` | The file is `17_semantic_cache.py` |
| `src/` | "A reusable enterprise package" | The directory does not exist. The plan makes `core/`. |
| `docs/roadmap/` | A subdirectory | The documents are directly in `docs/` |
| Golden dataset | "20 curated Q&A pairs" | 10 questions |
| AWS post-mortem | `aws_kinesis_outage.pdf` | `aws_kinesis_outage.md` |
| `.gitignore` | ".chroma" | It lists `chroma_db/` and `*.sqlite3` |

> **WARNING:** The `.gitignore` pattern `*.sqlite3` does not match `agent_memory.sqlite`. Therefore, the Commit contains a runtime file. Add `*.sqlite` and `*.sqlite-*` to `.gitignore`, and remove the file from the repository.

---

## 🔗 Next

→ [06_datasets.md](./06_datasets.md) — The public post-mortem datasets.
