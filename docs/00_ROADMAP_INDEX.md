# 🛡️ IncidentLens — Roadmap Master Index

> **System:** Autonomous Production Outage & Root-Cause Intelligence Copilot  
> **Architecture:** 100% Open-Source Hybrid RAG + Agentic Self-Correction + LLMOps  
> **Quality Bar:** ≥92% Faithfulness, ≥88% Context Precision (Ragas Evaluated)

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
| 6 | [06_datasets.md](./06_datasets.md) | 4 verified public post-mortem datasets | 📄 Defined |
| 7 | [07_phase1_foundations.md](./07_phase1_foundations.md) | Environment, loaders, chunking, embeddings, naive RAG | ⬜ Not Started |
| 8 | [08_phase2_failure_modes.md](./08_phase2_failure_modes.md) | 5 critical RAG failure modes & debugging toolkit | ⬜ Not Started |
| 9 | [09_phase3_retrieval_optimization.md](./09_phase3_retrieval_optimization.md) | Hybrid search, reranking, token budgeting, query expansion | ⬜ Not Started |
| 10 | [10_phase4_observability.md](./10_phase4_observability.md) | LangSmith, Ragas evaluation, Langfuse, caching | ⬜ Not Started |
| 11 | [11_phase5_security_api.md](./11_phase5_security_api.md) | Guardrails, LangGraph agent, FastAPI server, tests | ⬜ Not Started |
| 12 | [12_phase6_advanced_rag.md](./12_phase6_advanced_rag.md) | Contextual retrieval, late chunking, GraphRAG, multimodal | ⬜ Not Started |
| 13 | [13_quality_benchmarks.md](./13_quality_benchmarks.md) | Target metrics, measurement tools, execution protocol | 📄 Defined |

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

- **Primary LLMs:** Llama 3.3 70B (Groq) · DeepSeek-R1 (Groq) · Free SOTA Models (OpenRouter: `meta-llama/llama-3.3-70b-instruct:free` / `deepseek/deepseek-r1:free` / `openrouter/free`)
- **Embeddings:** BGE-M3 (FastEmbed / HuggingFace)
- **Vector Store:** ChromaDB (Persistent)
- **Graph Store:** NetworkX / Neo4j
- **Orchestrator:** LangGraph
- **Evaluation:** Ragas 4-metric harness
- **Observability:** LangSmith + Langfuse
- **API:** FastAPI + Uvicorn
