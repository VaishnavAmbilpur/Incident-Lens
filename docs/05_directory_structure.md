# 📁 05 — Directory Structure

> **Context:** Exact repository layout organized by enterprise software engineering best practices.  
> **Prerequisite:** [02_architecture_overview.md](./02_architecture_overview.md)

---

## 5.1 Full Tree

```text
Production-Level-RAG/
│
├── .env                              # API keys & endpoints (Groq, OpenRouter, Langfuse, LangSmith)
├── .gitignore                        # Standard ignores (venv, __pycache__, .chroma, *.log)
├── requirements.txt                  # Frozen open-source dependencies
├── README.md                         # Project documentation & benchmark scorecards
│
├── docs/                             # Project documentation
│   └── roadmap/                      # This roadmap (you are here)
│       ├── 00_ROADMAP_INDEX.md
│       ├── 01_problem_and_solution.md
│       ├── ... (all spec files)
│       └── 13_quality_benchmarks.md
│
├── data/                             # Raw incident datasets & golden eval sets
│   ├── postmortems/                  # Real disaster reports (PDF & Markdown)
│   │   ├── crowdstrike_outage_2024.pdf
│   │   ├── cloudflare_bgp_leak.md
│   │   ├── aws_kinesis_outage.pdf
│   │   └── github_mysql_failover.md
│   └── eval_golden_dataset.json      # 20 curated Q&A pairs with ground-truth citations
│
├── modules/                          # Standalone tutorial & verification scripts (Phases 1-6)
│   ├── 01_environment_check.py       # Phase 1: Dependency & device verification
│   ├── 02_document_loaders.py        # Phase 1: PDF & Markdown loader pipeline
│   ├── 03_text_splitters.py          # Phase 1: Chunking & overlap math
│   ├── 04_embeddings_deep_dive.py    # Phase 1: BGE-M3 geometric vector space
│   ├── 05_vector_stores_chroma.py    # Phase 1: ChromaDB indexing & metadata
│   ├── 06_basic_rag_pipeline.py      # Phase 1: Naive RAG baseline (LCEL)
│   ├── 07_rag_failure_modes_demo.py  # Phase 2: Deliberate 5-failure stress tests
│   ├── 08_debugging_rag.py           # Phase 2: Diagnostic & inspection toolkit
│   ├── 09_token_budgeting.py         # Phase 3: Dynamic context window manager
│   ├── 10_hybrid_search_bm25.py      # Phase 3: BGE-M3 + BM25 with RRF
│   ├── 11_query_transformations.py   # Phase 3: Multi-query & query rewriting
│   ├── 12_reranking_pipeline.py      # Phase 3: BGE cross-encoder reranking
│   ├── 13_langsmith_setup.py         # Phase 4: LangSmith tracing & telemetry
│   ├── 14_ragas_evaluation.py        # Phase 4: 4-metric Ragas test harness
│   ├── 15_langfuse_observability.py  # Phase 4: Langfuse open-source tracing
│   ├── 16_chroma_production.py       # Phase 4: ChromaDB disk persistence & collection mgmt
│   ├── 17_cost_and_caching.py        # Phase 4: Semantic cache implementation
│   ├── 18_security_guardrails.py     # Phase 5: Prompt injection & PII maskers
│   ├── 19_langgraph_agent.py         # Phase 5: Stateful graph with SQLite memory
│   ├── 20_api_server.py              # Phase 5: FastAPI production REST server
│   ├── 21_long_context_vs_rag.py     # Phase 6: 1M token context window benchmark
│   ├── 22_contextual_retrieval.py    # Phase 6: Anthropic situation headers
│   ├── 23_late_chunking.py           # Phase 6: Full timeline token pooling
│   ├── 24_agentic_rag.py             # Phase 6: Self-correcting LangGraph agent
│   ├── 25_graphrag_reasoning.py      # Phase 6: Cascading failure knowledge graph
│   └── 26_multimodal_colpali.py      # Phase 6: Vision-based chart & diagram RAG
│
├── src/                              # Reusable enterprise package
│   ├── __init__.py
│   ├── config.py                     # Central configuration & settings
│   ├── ingestion/                    # Clean ingestion & chunking modules
│   ├── retrieval/                    # Hybrid retrieval, rerankers & vector stores
│   ├── security/                     # Injection sanitizers & PII filters
│   ├── agent/                        # LangGraph workflow, nodes & state schemas
│   └── api/                          # FastAPI route handlers & Pydantic schemas
│
└── tests/                            # Automated regression & safety test suite
    ├── __init__.py
    ├── test_security.py              # Prompt injection penetration tests
    ├── test_retrieval.py             # Retrieval accuracy & precision tests
    └── test_incidentlens.py          # End-to-end integration test suite
```

---

## 5.2 Key Directories Explained

| Directory | Purpose | When It's Built |
|---|---|---|
| `data/postmortems/` | Raw incident documents to index | Phase 1 |
| `data/eval_golden_dataset.json` | Ground-truth Q&A pairs for Ragas evaluation | Phase 4 |
| `modules/` | Standalone scripts — each file is one learning unit | Phases 1–6 |
| `src/` | Production-ready reusable package (imported by API & agent) | Phases 3–5 |
| `tests/` | Automated pytest suite | Phase 5 |
| `docs/roadmap/` | This specification (you are reading it now) | Pre-build |

---

## 🔗 Next

→ [06_datasets.md](./06_datasets.md) — The 4 public post-mortem datasets we index.
