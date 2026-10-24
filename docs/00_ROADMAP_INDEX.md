# 🛡️ IncidentLens — Roadmap Master Index

> **System:** Autonomous Production Outage & Root-Cause Intelligence Copilot  
> **Architecture:** 100% Open-Source Hybrid RAG + Agentic Self-Correction + LLMOps  
> **Quality Bar:** ≥92% Faithfulness, ≥88% Context Precision (Ragas Evaluated)  
> **Measured today:** Context Precision 0.920 (PASS) · Context Recall 0.770 (FAIL) · Faithfulness and Answer Relevance NOT MEASURED in the last run (provider quota). The true values are in [15_evals.md](./15_evals.md).

---

## 📚 Documentation Map

Use this file as the **single entry point** to navigate the entire project spec.  
Each linked file is self-contained and can be read independently by any AI or human.

| # | File | What It Covers | Status |
|---|------|---------------|--------|
| 1 | [01_problem_and_solution.md](./01_problem_and_solution.md) | Why IncidentLens exists — the business & technical problem | 📄 Defined |
| 2 | [02_architecture_overview.md](./02_architecture_overview.md) | 6-layer unified architecture & how techniques connect | 📄 Defined |
| 3 | [03_tech_stack.md](./03_tech_stack.md) | All free cloud models, embeddings, rerankers, and tools | 📄 Defined |
| 4 | [04_system_diagrams.md](./04_system_diagrams.md) | End-to-end architecture diagram & user flow sequence | 📄 Defined |
| 5 | [05_directory_structure.md](./05_directory_structure.md) | Production file/folder layout with descriptions | 📄 Defined |
| 6 | [06_datasets.md](./06_datasets.md) | The public post-mortem corpus and the golden evaluation set | 📄 Defined |
| 7 | [07_phase1_foundations.md](./07_phase1_foundations.md) | Environment, loaders, chunking, embeddings, naive RAG | ✅ Complete (Modules 01–06) |
| 8 | [08_phase2_failure_modes.md](./08_phase2_failure_modes.md) | 5 critical RAG failure modes & debugging toolkit | ✅ Complete (Modules 07–08) |
| 9 | [09_phase3_retrieval_optimization.md](./09_phase3_retrieval_optimization.md) | Hybrid search, reranking, token budgeting, query expansion | ✅ Complete (Modules 09–13) |
| 10 | [10_phase4_observability.md](./10_phase4_observability.md) | LangSmith, Ragas evaluation, Langfuse, caching | ✅ Complete (Modules 14–17) |
| 11 | [11_phase5_security_api.md](./11_phase5_security_api.md) | Guardrails, LangGraph agent, FastAPI server, tests | ✅ Complete (Modules 18–20) |
| 12 | [12_phase6_advanced_rag.md](./12_phase6_advanced_rag.md) | Contextual retrieval, late chunking, GraphRAG, multimodal | ✅ Complete (Modules 21–26) |
| 13 | [13_quality_benchmarks.md](./13_quality_benchmarks.md) | Target metrics, measurement tools, execution protocol | 📄 Defined |
| 14 | [14_pipeline_plan.md](./14_pipeline_plan.md) | The plan: 10 stages, 6 ablation rows, 8 phases | 📄 Defined |
| 15 | [15_evals.md](./15_evals.md) | The ablation table, the metric definitions, the honest verdicts | 📄 Defined |
| 16 | [16_observability.md](./16_observability.md) | LangSmith and Langfuse: settings, division of work, switch | 📄 Defined |
| 17 | [17_interview_defense.md](./17_interview_defense.md) | One answer for each stage. The 90-second defense. | 📄 Defined |

### The new `core/` library — build state

| Stage | File | Status |
|---|---|---|
| 1 Ingest | `core/ingest.py` | ✅ Built and verified (31 documents, 2 dropped by the language check) |
| 2 Chunk | `core/chunking.py` | ✅ Built and verified (1,169 chunks, boundary split 5.90%) |
| 3 Embed | `core/embeddings.py` | ✅ Built and verified (1,169 × 384, 7 truncated chunks) |
| 4 Index | `core/vectorstore.py` | ✅ Built and verified (1,169 vectors, 1.19 s) |
| 5 Retrieve | `core/retrieval.py` | ✅ Built and measured (paraphrase MRR 0.6121) |
| 5b BM25 | `core/bm25.py` | ✅ Built and measured (literal MRR 1.0000, 3 ms) |
| 6 Fuse | `core/fusion.py` | ✅ Built and measured (paraphrase MRR 0.6920) |
| 7 Rerank | `core/rerank.py` | ✅ Built and measured (paraphrase MRR 0.7869, best result) |
| 10 Evaluate | `core/evaluate.py` | ✅ Built. Retrieval metrics done. LLM metrics (faithfulness, answer relevance) added via --llm flag. |
| 8 Contextualize | `core/contextual.py` | ✅ Complete (verified end-to-end, 10-chunk sample) |
| 9 Generate | `core/generate.py` | ✅ Complete (verified end-to-end, 2-question sample) |
| — Provider | `core/providers.py` | ✅ Built. Fallback chain (Groq→OpenRouter), 429 retry, disk cache. |
| — Query Expand | `core/query_expansion.py` | ✅ Built (v2.1.0). Multi-query LLM expansion + RRF merge. Implements ablation Row D. |
| — Ablation | `ablation.py` | ✅ Built. Runs all 6 Table 14-2 configurations. Rows A–E measured. Row D (multi-query) and Row F (contextual) now implemented. |
| — Diagnose | `diagnose.py` | ✅ Built. Reports all 10 stages for one query, identifies failure stage. |
| — API server | `api/server.py` | ✅ Built. FastAPI: GET / (UI), GET /health, GET /metrics, POST /query. Serves the built React UI from the same origin — one server, no CORS issues. |
| — Thin UI | `frontend/src/App.jsx` | ✅ Built. Single-page React app showing all 10 stages. Built to `frontend/dist/`, served by the API. |
| — Runner | `run.py` | ✅ Built. Workaround for _ctypes DLL issue — use `C:\Python313\python.exe run.py <cmd>` to launch any component. |

---

## 🔄 Progress Tracker

> **Instructions for AI agents:** Update the status column above as phases are completed.  
> Use these status markers:

| Marker | Meaning |
|--------|---------|
| ⬜ Not Started | Phase has not begun |
| 🔧 In Progress | Currently being built |
| ✅ Complete | All deliverables verified |
| 📄 Defined | Specification written, not yet implemented |

> **Note:** All pipeline work is complete. Stages 1–10 are built and verified (ingest → chunk → embed → index → retrieve → fuse → rerank → contextualize → generate → evaluate). The provider fallback layer, ablation table, diagnose command, FastAPI API server, and React thin UI are all built. The full pipeline can be run end-to-end via `api/server.py` + `frontend/`.

---

## 🧭 How to Use This Roadmap

1. **Start here.** Read this index to understand scope.
2. **Read the problem & architecture** (files 01–04) for full context.
3. **Execute phases sequentially** (files 07–12). Each phase lists:
   - Engineering concepts to understand
   - Exact deliverable files to create
   - Terminal commands to run
   - Verification criteria before moving on
4. **Check benchmarks** (file 13) after each phase to measure progress.

---

## ⚡ Quick Reference

- **Primary LLMs:** `llama-3.3-70b-versatile` (Groq) · `deepseek-r1-distill-llama-70b` (Groq) · Free SOTA Models (OpenRouter)
- **Judge Models:** `qwen/qwen3-8b` · `allam-2-7b` · `openai/gpt-oss-20b` (the three-model judge chain)
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim, local, via ChromaDB ONNX)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, CPU)
- **Vector Store:** ChromaDB (Persistent)
- **Graph Library:** NetworkX
- **Orchestrator:** LangGraph
- **Evaluation:** 4-metric harness (Ragas 0.4.3 installed)
- **Observability:** LangSmith + Langfuse 4.x
- **API:** FastAPI + Uvicorn

> **Note:** The versions and the model names in this list are the versions that are installed. See [03_tech_stack.md](./03_tech_stack.md), section 3.6, for the corrections to the first version of the documentation.
