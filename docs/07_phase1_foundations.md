# 🔹 07 — Phase 1: Environment Setup & Core Foundations

> **Status:** ⬜ Not Started  
> **Focus:** Python 3.11+ workspace, LangChain LCEL pipelines, ChromaDB, and chunking fundamentals.  
> **Prerequisite:** Read [01–06](./00_ROADMAP_INDEX.md) for full context.

---

## 1.1 Engineering Concepts to Master

| Concept | What You'll Learn |
|---|---|
| **Virtual Environment** | Clean Python 3.11+ isolation with `venv` |
| **Document Loading** | PyPDF and Markdown loaders with metadata preservation |
| **Chunking Strategies** | Recursive Character vs Token chunking — overlap math, context boundary preservation |
| **Vector Embeddings** | BGE-M3 via FastEmbed — dimension trade-offs (1536 vs 512), cosine similarity |
| **ChromaDB** | Local vector store setup, indexing, metadata filters |
| **LCEL Pipelines** | `RunnablePassthrough`, `ChatPromptTemplate`, `StrOutputParser` composition |
| **Naive RAG Baseline** | End-to-end retrieve → augment → generate pipeline |

---

## 1.2 Deliverable Files

| File | Description | Verification |
|---|---|---|
| `requirements.txt` | All frozen dependencies | `pip install -r requirements.txt` completes with 0 errors |
| `.env` | API keys for Groq, OpenRouter, LangSmith, Langfuse | Keys populated and loadable |
| `.gitignore` | Standard Python + Chroma ignores | Committed |
| `modules/01_environment_check.py` | Dependency verification, GPU/CPU detection | Prints all versions and device info |
| `modules/02_document_loaders.py` | PDF + Markdown loader with metadata | Loads all 4 post-mortems, prints page counts |
| `modules/03_text_splitters.py` | Chunking strategy explorer | Prints chunk counts, overlap analysis, boundary inspection |
| `modules/04_embeddings_deep_dive.py` | BGE-M3 embedding & cosine similarity | Computes pairwise similarities for test queries |
| `modules/05_vector_stores_chroma.py` | ChromaDB indexing with metadata | Indexes all chunks, runs test query, prints top-K results |
| `modules/06_basic_rag_pipeline.py` | Naive RAG baseline with Llama 3.3 | Answers a test question with source citations |

---

## 1.3 Step-by-Step Execution

### Step 1: Create Virtual Environment
```powershell
cd Production-Level-RAG
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment
```powershell
# Copy .env and fill in your API keys
# Test: python -c "from dotenv import load_dotenv; load_dotenv(); print('OK')"
```

### Step 4: Run Each Module Sequentially
```powershell
python modules/01_environment_check.py
python modules/02_document_loaders.py
python modules/03_text_splitters.py
python modules/04_embeddings_deep_dive.py
python modules/05_vector_stores_chroma.py
python modules/06_basic_rag_pipeline.py
```

---

## 1.4 Success Criteria (Gate to Phase 2)

- [ ] All 4 post-mortem documents load without errors
- [ ] Chunks have correct overlap and no mid-sentence splits on code blocks
- [ ] BGE-M3 embeddings produce meaningful cosine similarity scores
- [ ] ChromaDB stores and retrieves chunks with metadata
- [ ] Naive RAG pipeline answers "What caused the CrowdStrike outage?" with a relevant response
- [ ] All outputs print cleanly to terminal

---

## 🔗 Next

→ [08_phase2_failure_modes.md](./08_phase2_failure_modes.md) — Diagnosing the 5 failure modes that break naive RAG.
