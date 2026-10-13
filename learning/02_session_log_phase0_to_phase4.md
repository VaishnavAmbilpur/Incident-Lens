# IncidentLens Production RAG — Complete Session Log
## ASD STE 100 Technical Learning Document

**Document ID:** IL-STE-001  
**Project:** IncidentLens — Production-Level RAG Pipeline  
**Standard:** ASD Simplified Technical English (STE 100)  
**Coverage:** Phase 0 (Setup) through Phase 4 (Observability — In Progress)  
**Last Updated:** 2026-10-13  

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Phase 0 — Project Setup](#2-phase-0--project-setup)
3. [Phase 1 — Foundations (Modules 01–06)](#3-phase-1--foundations-modules-0106)
4. [Phase 2 — Failure Mode Analysis (Modules 07–08)](#4-phase-2--failure-mode-analysis-modules-0708)
5. [Phase 3 — Retrieval Optimization (Modules 09–13)](#5-phase-3--retrieval-optimization-modules-0913)
6. [Phase 4 — Observability (Modules 14–15)](#6-phase-4--observability-modules-1415)
7. [Key Findings Summary](#7-key-findings-summary)
8. [Architecture Diagram](#8-architecture-diagram)
9. [Glossary](#9-glossary)

---

## 1. Project Purpose

### 1.1 What Is This Project

IncidentLens is a production-level Retrieval-Augmented Generation (RAG) pipeline. The pipeline answers technical questions about real system outage post-mortem reports.

### 1.2 Why We Built It

A large language model (LLM) does not know about private incident data. The LLM cannot answer questions about internal outages without context. RAG solves this problem. RAG retrieves relevant document chunks and puts them in the LLM prompt. The LLM then generates answers grounded in the retrieved facts.

### 1.3 What Problems We Solved

| Problem | Solution |
|---|---|
| LLM does not know private data | RAG injects retrieved context into the prompt |
| Exact error codes not found by vectors | BM25 lexical search added |
| Vague user queries miss relevant chunks | Multi-query expansion generates 3 technical variants |
| Noisy retrieved chunks confuse the LLM | Cross-encoder reranking prunes noise |
| Context window overflow | Dynamic token budgeting prevents overflow |
| No quality measurement | Ragas 4-metric automated scorecard |

### 1.4 Documents Used (Authentic Official Sources Only)

| File | Source | Size |
|---|---|---|
| `crowdstrike_outage_2024.pdf` | Official CrowdStrike External RCA PDF | 12 pages, 29,380 chars |
| `cloudflare_bgp_leak.md` | Verbatim Cloudflare blog post (Tom Strickx) | 8,099 chars |
| `github_mysql_failover.md` | Verbatim GitHub post-incident analysis | 15,484 chars |
| `aws_kinesis_outage.md` | Verbatim AWS Message 11201 | 5,502 chars |

> **Rule:** No synthetic or fabricated data. All documents are authentic official post-mortems.

---

## 2. Phase 0 — Project Setup

### 2.1 What We Did

We created the project directory structure and installed all required Python packages.

### 2.2 Directory Structure Created

```
Production-Level-RAG/
├── .env                    ← API keys (Groq, LangSmith, Langfuse)
├── requirements.txt        ← All Python dependencies
├── data/
│   ├── postmortems/        ← 4 authentic incident documents
│   ├── chroma_db/          ← Persistent vector database (105 chunks)
│   └── eval_golden_dataset.json  ← 10 Q&A pairs for Ragas
├── modules/                ← One Python script per concept
├── docs/                   ← Phase roadmap documentation
└── learning/               ← This file and architecture notes
```

### 2.3 Why We Used These Tools

| Tool | Why |
|---|---|
| **Python 3.11+** | Standard for ML/AI workloads |
| **LangChain** | Orchestration framework for RAG pipelines |
| **ChromaDB** | Local persistent vector database, zero cost |
| **rank-bm25** | BM25 lexical keyword search |
| **sentence-transformers** | Cross-encoder reranking model |
| **Groq Cloud** | Free LLM API (Llama, Qwen models) |
| **tiktoken** | Exact token counting for budget management |
| **Ragas** | Automated RAG quality evaluation |
| **LangSmith** | LLM call tracing and observability |

### 2.4 Environment Keys Required

```
GROQ_API_KEY          → Free at console.groq.com
OPENROUTER_API_KEY    → Free at openrouter.ai
LANGCHAIN_API_KEY     → Free at smith.langchain.com
LANGCHAIN_TRACING_V2  → "true"
LANGCHAIN_PROJECT     → "IncidentLens"
LANGFUSE_SECRET_KEY   → Free at cloud.langfuse.com
LANGFUSE_PUBLIC_KEY   → Free at cloud.langfuse.com
LANGFUSE_BASE_URL     → "https://us.cloud.langfuse.com"
```

---

## 3. Phase 1 — Foundations (Modules 01–06)

### 3.1 Module 01 — Environment Check

**File:** `modules/01_environment_check.py`  
**Command:** `.\venv\Scripts\python.exe modules/01_environment_check.py`

#### What We Did
We verified that all 10 required packages were installed. We verified all API keys were present. We made a live test call to Groq Cloud.

#### Why
Before writing code, you must confirm that the environment is correct. A missing package causes a later failure that is hard to debug.

#### Key Result
All 10 packages passed. Groq API returned a live response. LangSmith and Langfuse keys were present.

#### Windows-Specific Fix Applied
Windows uses `cp1252` encoding by default. Unicode emoji characters (checkmarks, crosses, bars) cause `UnicodeEncodeError`. We added this fix to every module:

```python
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

---

### 3.2 Module 02 — Document Loaders

**File:** `modules/02_document_loaders.py`  
**Command:** `.\venv\Scripts\python.exe modules/02_document_loaders.py`

#### What We Did
We loaded all 4 post-mortem documents. We used `PyPDFLoader` for the CrowdStrike PDF. We used `TextLoader` for the three Markdown files.

#### Why
Different document formats require different loaders. The PDF has 12 pages and embedded text. The Markdown files are plain text. LangChain provides loaders for both.

#### Key Result

| File | Type | Pages | Characters |
|---|---|---|---|
| crowdstrike_outage_2024.pdf | PDF | 12 | 29,380 |
| cloudflare_bgp_leak.md | Markdown | 1 | 8,099 |
| github_mysql_failover.md | Markdown | 1 | 15,484 |
| aws_kinesis_outage.md | Markdown | 1 | 5,502 |
| **Total** | | **15 segments** | **58,465 chars** |

#### What Dynamic Import Is and Why We Used It
Python does not allow `import 02_document_loaders` because module names cannot start with numbers. We used `importlib.util.spec_from_file_location()` to load modules by file path instead of by name. Every later module uses this pattern.

```python
def _import_helper(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

---

### 3.3 Module 03 — Text Splitters and Chunking

**File:** `modules/03_text_splitters.py`  
**Command:** `.\venv\Scripts\python.exe modules/03_text_splitters.py`

#### What We Did
We split the 15 document segments into smaller chunks. We used `RecursiveCharacterTextSplitter` with chunk size 800 and overlap 150.

#### Why Chunking Is Necessary
LLMs have limited context windows. You cannot put the entire post-mortem into a prompt. Chunking divides the document into pieces. Each chunk is embedded separately and retrieved individually.

#### Why These Parameters

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | 800 characters | ~135 tokens each. Fits well inside 8k context window |
| `chunk_overlap` | 150 characters | Prevents sentences from being cut at boundaries |
| Splitter type | Recursive | Tries `\n\n`, then `\n`, then `.`, then space. Preserves structure |

#### Chunking Strategy Comparison We Ran

| Strategy | Chunk Size | Total Chunks | Avg Chars |
|---|---|---|---|
| Fine-Grained | 400 | 209 | 287 |
| **Production (chosen)** | **800** | **105** | **599** |
| Coarse-Grained | 1500 | 51 | 1212 |

#### Key Result
105 production chunks. Each chunk has a unique `chunk_id` in metadata (e.g., `crowdstrike_outage_2024_p3_c2`).

---

### 3.4 Module 04 — Embeddings Deep Dive

**File:** `modules/04_embeddings_deep_dive.py`  
**Command:** `.\venv\Scripts\python.exe modules/04_embeddings_deep_dive.py`

#### What We Did
We demonstrated how text is converted to dense vectors. We computed cosine similarity, dot product, and Euclidean distance manually in pure Python. We used ChromaDB's built-in ONNX `all-MiniLM-L6-v2` model.

#### What an Embedding Is
An embedding is a list of 384 floating-point numbers. Each number represents a dimension of semantic meaning. Similar sentences have vectors that point in similar directions.

#### Why Local Embeddings (No API)
ChromaDB's ONNX embedding model runs entirely on the local CPU. There is zero cost per embedding. There are no API rate limits. Speed is fast.

#### Key Experiment Result
The Semantic Bridge Test showed that the query "What caused Windows computers to show a blue screen crash?" matched the CrowdStrike chunk with cosine similarity 0.369. It did NOT match a GitHub chunk (similarity 0.149). The model understood that "blue screen crash" semantically relates to Windows kernel failures even with no exact keyword overlap.

---

### 3.5 Module 05 — Vector Stores with ChromaDB

**File:** `modules/05_vector_stores_chroma.py`  
**Command:** `.\venv\Scripts\python.exe modules/05_vector_stores_chroma.py`

#### What We Did
We stored all 105 chunks in a persistent ChromaDB collection. The collection uses HNSW (Hierarchical Navigable Small World) cosine distance space.

#### Why Persistent Storage
Without persistence, the vector database is rebuilt from scratch each run. With `PersistentClient`, the index is saved to `data/chroma_db/` on disk. Subsequent runs load the index instantly.

#### Metadata Stored Per Chunk

```python
{
    "chunk_id":      "crowdstrike_outage_2024_p3_c2",
    "source_file":   "crowdstrike_outage_2024.pdf",
    "incident_name": "Crowdstrike Outage 2024",
    "incident_slug": "crowdstrike_outage_2024",
    "page_number":   3
}
```

#### Key Result
105 chunks indexed. Collection `incidentlens_postmortems` persists across restarts. Metadata filtering confirmed working (e.g., filter by `incident_slug`).

---

### 3.6 Module 06 — Basic Naive RAG Pipeline

**File:** `modules/06_basic_rag_pipeline.py`  
**Command:** `.\venv\Scripts\python.exe modules/06_basic_rag_pipeline.py`

#### What We Did
We built the first end-to-end RAG pipeline using LangChain LCEL (LangChain Expression Language). The pipeline uses pipe operators (`|`) to chain steps.

#### What LCEL Means
LCEL is LangChain's way to chain components with the `|` operator:

```python
chain = retriever | prompt | llm | StrOutputParser()
answer = chain.invoke({"question": query})
```

Each component passes its output to the next component automatically.

#### Naive RAG Architecture

```
User Query
    -> ChromaDB Dense Retrieval (top 4 chunks)
    -> Prompt Template fills in {context} and {question}
    -> Groq Cloud LLM generates answer
    -> StrOutputParser extracts text string
```

#### Key Result
The pipeline answered CrowdStrike questions correctly with zero hallucination. Every claim was cited to a specific chunk. This was the baseline we later improved.

---

## 4. Phase 2 — Failure Mode Analysis (Modules 07–08)

### 4.1 Why Phase 2 Exists

Naive RAG has known failure modes. Before optimizing, you must understand what breaks and why. Phase 2 deliberately triggers each failure to prove it exists.

---

### 4.2 Module 07 — RAG Failure Modes Demo

**File:** `modules/07_rag_failure_modes_demo.py`  
**Command:** `.\venv\Scripts\python.exe modules/07_rag_failure_modes_demo.py`

#### The 5 Failure Modes We Proved

**Failure Mode 1: Bad Chunking (Severed Stack Traces)**

A chunk size of 74 characters with 0 overlap cut a kernel crash trace mid-identifier:
```
[Chunk 1] "CRASH LOG: Kernel Driver CSAgent.sys triggered PAGE_FAULT_IN_NONPAGED_AREA"
[Chunk 2] "(0x00000050). Memory Address: 0xfffff8020ebc14ed referenced unmapped"
[Chunk 3] "page. Failure occurred in Content Interpreter evaluating Channel File 291."
```
The LLM received only Chunk 2 and could not reconstruct the error.  
**Fix:** Use overlap of 150 characters and chunk size of 800.

**Failure Mode 2: Semantic Mismatch (Exact Code Not Found)**

Query: `"AS33154"`  
Dense vector search returned 0% relevant results in Top-5. The embedding model has no semantic concept of autonomous system numbers. `AS33154` is just random characters to a dense model.  
**Fix:** BM25 lexical search.

**Failure Mode 3: Retrieval Noise (90% Noise in Top-10)**

Query: `"How did Cloudflare resolve the BGP route leak?"`  
Top-10 dense retrieval included 9 generic background chunks about BGP protocols. Only 1 chunk was about the actual resolution.  
**Fix:** Cross-encoder reranking.

**Failure Mode 4: Lost in the Middle**

When 20 chunks were passed to the LLM, the most relevant chunk at position #12 was ignored. LLMs pay less attention to content in the middle of a long context.  
**Fix:** Token budgeting limits context to the top 5-8 chunks only.

**Failure Mode 5: Hallucination**

When asked about an event not in the corpus, the LLM invented plausible-sounding but false technical details.  
**Fix:** Strict system prompt instructs the LLM to answer ONLY from provided context.

---

### 4.3 Module 08 — RAG Debugging Toolkit

**File:** `modules/08_debugging_rag.py`  
**Command:** `.\venv\Scripts\python.exe modules/08_debugging_rag.py`

#### What We Built

| Tool | What It Shows |
|---|---|
| Similarity Score Histogram | ASCII bar chart of cosine similarity decay across 15 retrieved chunks |
| Token Count Inspector | tiktoken counts for each chunk — prevents context overflow |
| Chunk Boundary Viewer | Exact character offsets — verifies no sentence was cut mid-word |
| Precision@5 Matrix | How many of Top-5 chunks belong to the expected source document |

#### Key Numbers Discovered

| Measurement | Value | Implication |
|---|---|---|
| Max cosine similarity | 0.634 | Signal chunks score above 0.50 |
| Min cosine similarity | 0.331 | Below 0.45 = noise |
| Score drop from Rank 1 to Rank 15 | 0.303 points | Sharp cliff after Rank 6 |
| Avg tokens per chunk | 135 tokens | Top-4 chunks = ~600 tokens = 7.3% of 8k window |

---

## 5. Phase 3 — Retrieval Optimization (Modules 09–13)

### 5.1 Why Phase 3 Exists

Phase 2 proved that naive dense retrieval fails on exact codes, has high noise, and suffers from context overflow. Phase 3 solves all five failure modes systematically.

---

### 5.2 Module 09 — Token Budgeting

**File:** `modules/09_token_budgeting.py`  
**Command:** `.\venv\Scripts\python.exe modules/09_token_budgeting.py`

#### The Problem
If you pass too many chunks to an LLM, the context window overflows. The LLM silently truncates content. You do not know which content was cut.

#### The Formula

```
Available Context Tokens = Model Max Tokens
                         - System Prompt Tokens
                         - User Query Tokens
                         - Output Reserve Tokens
                         - Safety Margin (50 tokens)
```

#### The Three Deployment Profiles We Tested

| Profile | Model Max | Output Reserve | Available Context | Chunks Admitted |
|---|---|---|---|---|
| A: Standard Production | 8,192 | 2,048 | 6,049 | 15 / 15 |
| B: Edge / Low-Cost | 2,048 | 512 | 1,451 | 10 / 15 |
| C: Ultra-Fast SLA | 1,200 | 500 | 625 | 4 / 15 |

#### Key Result
Profile C automatically pruned 11 noisy chunks. Profile A admitted all 15 candidates at 37.74% utilization. The same code works on any model by changing one parameter.

---

### 5.3 Module 10 — Hybrid Search (BM25 + Dense + RRF)

**File:** `modules/10_hybrid_search_bm25.py`  
**Command:** `.\venv\Scripts\python.exe modules/10_hybrid_search_bm25.py`

#### Why Hybrid Search

| Search Type | Finds | Misses |
|---|---|---|
| Dense Vector Only | Semantic concepts ("blue screen") | Exact codes (`AS33154`, `0x00000050`) |
| BM25 Keyword Only | Exact tokens | Paraphrased/synonymous content |
| **Hybrid (Both)** | **Both** | **Very little** |

#### What BM25 Is
BM25 (Best Match 25) is a ranking function. It scores each document by how often the query terms appear in it, normalized by document length and corpus frequency. It is the algorithm behind early search engines.

#### What RRF Is
Reciprocal Rank Fusion merges two ranked lists without knowing each list's score scale:

```
RRF_Score(document) = 1/(60 + dense_rank) + 1/(60 + bm25_rank)
```

The constant 60 prevents high-ranked items from dominating. Any document that appears in the top ranks of both lists gets the highest combined score.

#### Technical Tokenizer We Built
Standard tokenizers break `0x00000050` into `0x`, `00000050`. We built a regex tokenizer that preserves technical tokens:

```python
pattern = r"[a-zA-Z0-9_\.\-#:]+"
tokens  = re.findall(pattern, text.lower())
```

This preserves: `0x00000050`, `AS33154`, `CSAgent.sys`, `us-east-1`.

#### Key Results

| Benchmark | Dense Rank | BM25 Rank | Hybrid Rank | Verdict |
|---|---|---|---|---|
| Query: `AS33154` | #4 (Sim: 0.233) | **#1 (Score: 4.0)** | **#1** | BM25 saved the query |
| Query: "blue screen crash" | **#1 (Sim: 0.493)** | #6 (Score: 7.6) | **#1** | Dense saved the query |
| Query: "Kinesis thread limits" | **#1 (Sim: 0.772)** | **#1 (Score: 22.1)** | **#1 (RRF: 0.033)** | Both agreed = maximum confidence |

---

### 5.4 Module 11 — Query Transformations

**File:** `modules/11_query_transformations.py`  
**Command:** `.\venv\Scripts\python.exe modules/11_query_transformations.py`

#### The Problem: Query Myopia
A user types: "Tell me about the database split-brain and replication problems."  
This is a vague query. Dense and BM25 search both interpret it from one angle. They miss chunks that discuss the same event from different technical angles.

#### The Solution: Multi-Query Expansion
We use the Groq LLM to generate 3 specialized variants of the query:

```
Original: "Tell me about the database split-brain and replication problems."

Variant A (Code/Mechanism):
  "database split-brain race condition memory corruption pointer error code"
Variant B (Infrastructure):
  "replication lag traffic routing thread pool exhaustion distributed architecture"
Variant C (Mitigation):
  "split-brain incident timeline failover alarms mitigation telemetry logs"
```

We run all 4 queries (original + 3 variants) through the Hybrid Search Engine. We merge results using Cross-Query RRF.

#### Important Fix Applied
Groq free tier has an OTPM (Output Tokens Per Minute) limit of 1,000 tokens. The default `max_tokens` in LangChain is 2,048. This caused a 429 error. Fix:

```python
self.llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.2,
    max_tokens=300,   # Only need ~100 tokens for 3 JSON strings
    api_key=self.api_key,
)
```

#### Key Result
3 newly unlocked critical chunks per query:
- `aws_kinesis_outage_p1_c2` — Found only via Variants #1 and #3
- `github_mysql_failover_p1_c6` — Found only via Variants #2 and #3
- `github_mysql_failover_p1_c22` — Found only via Variants #2 and #3

---

### 5.5 Module 12 — Cross-Encoder Reranking

**File:** `modules/12_reranking_pipeline.py`  
**Command:** `.\venv\Scripts\python.exe modules/12_reranking_pipeline.py`  
**Install First:** `.\venv\Scripts\pip.exe install sentence-transformers`

#### Why Bi-Encoders (Embeddings) Are Imprecise
A bi-encoder encodes the query and the document into separate vectors. It computes similarity by comparing those two vectors. The query and document are never compared word-by-word.

#### Why Cross-Encoders Are More Precise
A cross-encoder receives the query and the document **together** as one input: `[query] [SEP] [document]`. The transformer's attention mechanism can see every word in the query next to every word in the document. This is much more accurate but slower.

#### The Two-Stage Strategy

```
Stage 1 (Fast -- milliseconds):
  Hybrid Search retrieves top 20 candidates from 105 chunks

Stage 2 (Precise -- ~1 second):
  Cross-Encoder scores each of the 20 (query, chunk) pairs
  Results re-sorted by cross-encoder score
  Bottom candidates discarded
  Top 5-8 passed to LLM
```

#### Model Used
`cross-encoder/ms-marco-MiniLM-L-6-v2`
- 22 million parameters (tiny)
- Runs on CPU (no GPU required)
- Trained on MS MARCO passage ranking dataset
- First run: downloads ~90 MB (cached permanently after)
- Zero API cost

#### Key Results

| Benchmark | Pre-Rerank Precision@5 | Post-Rerank Precision@5 |
|---|---|---|
| CrowdStrike crash query | 100% | 60% (cross-encoder penalized keyword mismatch) |
| GitHub MySQL split-brain | 100% | 100% (reranker promoted deeper chunks) |
| AWS Kinesis thread limits | 100% | 100% (perfect alignment) |

**Finding:** The CrowdStrike result shows cross-encoders are sensitive to query phrasing. This is why multi-query expansion (Stage 0) must precede reranking. Better query phrasing leads to better cross-encoder scores.

---

### 5.6 Module 13 — Optimized RAG Pipeline (Phase 3 Capstone)

**File:** `modules/13_optimized_rag_pipeline.py`  
**Command:** `.\venv\Scripts\python.exe modules/13_optimized_rag_pipeline.py`

#### The Full 4-Stage Pipeline

```
Stage 0: Multi-Query Expansion
  Input:  1 vague user query
  Output: 3 technical specialist variants (Groq LLM, max_tokens=300)

Stage 1: Hybrid Recall
  Input:  4 queries (original + 3 variants)
  Output: Top-20 candidates (BM25 + Dense + Cross-Query RRF)

Stage 2: Cross-Encoder Reranking
  Input:  20 candidates
  Output: Top-8 reranked (ms-marco-MiniLM-L-6-v2, local CPU)

Stage 3: Token Budget Packing
  Input:  8 ranked chunks
  Output: Chunks admitted within model context limit (tiktoken counted)

Stage 4: Grounded LLM Generation
  Input:  Budgeted context + user query
  Output: Answer with mandatory chunk-ID citations (Groq Cloud)
```

#### Head-to-Head vs Naive RAG Results

**Q1: "What is AS33154 and why did it cause an outage?"**

| Dimension | Naive RAG | Optimized RAG |
|---|---|---|
| Source chunks | 4 dense-only | 8 hybrid + reranked |
| Citations | 1 vague | 7 specific chunk IDs |
| Causal chain | 2 sentences | 6-step BGP propagation |
| Key fact found? | Partial | Yes — Verizon AS701 failure to filter |

**Q2: "MySQL primary diverge from replicas — Orchestrator detection?"**

The Optimized pipeline surfaced "954 writes not replicated" — a specific data point the Naive pipeline missed entirely. This changes the incident severity classification.

#### Pipeline Latency Breakdown (Q1)

| Stage | Latency |
|---|---|
| Stage 0 Multi-Query | 552 ms |
| Stage 1 Hybrid Recall | 1,363 ms |
| Stage 2 Cross-Encoder | 931 ms |
| Stage 3 Token Budget | 1 ms |
| Stage 4 LLM Generation | 1,340 ms |
| **Total** | **4,186 ms** |

---

## 6. Phase 4 — Observability (Modules 14–15)

### 6.1 Why Phase 4 Exists

A RAG pipeline in production is a black box without observability. You cannot debug failures, measure quality regressions, or estimate costs without instrumentation.

---

### 6.2 Module 14 — LangSmith Tracing

**File:** `modules/14_langsmith_setup.py`  
**Command:** `.\venv\Scripts\python.exe modules/14_langsmith_setup.py`

#### What LangSmith Does
LangSmith is LangChain's built-in tracing platform. It captures every LLM call automatically with zero code changes. When `LANGCHAIN_TRACING_V2=true` is set, every LCEL pipe chain emits a trace to the LangSmith dashboard.

#### What Each Trace Contains
- Full prompt text sent to the LLM
- LLM response text
- Input token count and output token count
- Latency in milliseconds per step
- Which retriever chunks were passed as context

#### Cost Estimation Formula We Added

```python
input_cost  = input_tokens  * 0.59 / 1_000_000   # $0.59 per million input tokens
output_cost = output_tokens * 0.79 / 1_000_000   # $0.79 per million output tokens
total       = input_cost + output_cost
```

At 1,000 queries/day, the pipeline costs approximately **$0.82-$0.86 per day**.

#### LangSmith Key Issue Encountered
The LangSmith key in `.env` returned HTTP 403. Cause: key may have been truncated or invalid. Fix: regenerate key from https://smith.langchain.com -> Settings -> API Keys, remove trailing whitespace in `.env`. Pipeline still runs in local mode without a valid key.

#### Pipeline Telemetry (Q1 — CrowdStrike)

| Stage | Latency |
|---|---|
| Stage 0 Multi-Query | 811 ms |
| Stage 1 Hybrid Recall | 1,484 ms |
| Stage 2 Cross-Encoder | 1,010 ms |
| Stage 3 Token Budget | 1 ms |
| Stage 4 LLM Generation | 1,044 ms |
| **Total** | **4,350 ms** |

#### Anti-Hallucination Verification (Q2 — Kinesis)
The pipeline was asked about DynamoDB being affected by the Kinesis outage. The LLM correctly refused: "The provided context chunks do not contain any information regarding Amazon DynamoDB." This confirms the Faithfulness guardrail is working — the LLM does not invent facts.

---

### 6.3 Module 15 — Ragas Automated Evaluation (In Progress)

**File:** `modules/15_ragas_evaluation.py`  
**Install:** `.\venv\Scripts\pip.exe install ragas datasets fastembed`  
**Command:** `.\venv\Scripts\python.exe modules/15_ragas_evaluation.py`

#### What Ragas Is
Ragas is an open-source framework for automated RAG evaluation. It uses an LLM as a judge to score the quality of answers.

#### The 4 Metrics

| Metric | What It Catches | Target |
|---|---|---|
| **Faithfulness** | Hallucinated claims (LLM invented facts not in context) | >= 92% |
| **Answer Relevancy** | Off-topic or incomplete answers | >= 90% |
| **Context Precision** | Noisy retrieved chunks in Top-K | >= 88% |
| **Context Recall** | Missing ground-truth facts from retrieved context | >= 85% |

#### Golden Evaluation Dataset Created

**File:** `data/eval_golden_dataset.json`  
**Total Questions:** 10  
**Coverage:** All 4 post-mortem documents

| ID | Question Category | Document | Difficulty |
|---|---|---|---|
| TC001 | Root Cause | CrowdStrike | Easy |
| TC002 | Root Cause (Detail) | CrowdStrike | Medium |
| TC003 | Mitigation | CrowdStrike | Medium |
| TC004 | Root Cause (Technical) | Cloudflare BGP | Hard |
| TC005 | Impact | Cloudflare BGP | Easy |
| TC006 | Root Cause (Mechanism) | Cloudflare BGP | Medium |
| TC007 | Root Cause (Sequence) | GitHub MySQL | Hard |
| TC008 | Mitigation | GitHub MySQL | Medium |
| TC009 | Root Cause (OS) | AWS Kinesis | Medium |
| TC010 | Impact (Cascade) | AWS Kinesis | Medium |

#### Status
**Pending.** Run the pip install command above, then run Module 15. Paste the scorecard output to complete this section.

---

## 7. Key Findings Summary

### 7.1 Critical Engineering Decisions

| Decision | Rationale | Alternative Considered |
|---|---|---|
| Chunk size 800 chars | ~135 tokens/chunk. Top-4 = 7.3% of 8k window | 400 chars (too fragmented), 1500 chars (too coarse) |
| BM25 with technical tokenizer | Preserves `AS33154`, `0x50`, `CSAgent.sys` | Standard word tokenizer (breaks hex codes) |
| RRF k=60 | Industry standard. Prevents single high-ranker from dominating | Direct score combination (fails due to incompatible score scales) |
| max_tokens=300 in multi-query | Stays under Groq free-tier OTPM limit of 1,000 | Default 2,048 (causes HTTP 429 rate limit error) |
| Cross-encoder before token packing | Noise removal first, then budget allocation | Reverse order wastes budget on noisy chunks |
| Dense-only retrieval for Ragas | Avoids rate limits during batch 10-question eval | Full pipeline (too many LLM calls per question) |

### 7.2 Verified Performance Numbers

| Measurement | Value |
|---|---|
| Total corpus tokens (105 chunks) | 14,258 tokens |
| Average tokens per chunk | 135.8 tokens |
| BM25 vocabulary | 2,275 unique technical terms |
| Pipeline latency (4-stage) | 2.8-4.4 seconds per query |
| Token budget utilization (Profile A) | ~14% (well within 8k window) |
| Cost per query (Groq production) | ~$0.00082 |
| Cost at 1,000 queries/day | ~$0.82/day |

---

## 8. Architecture Diagram

```
                     +-----------------------------------------+
                     |     USER QUERY (Natural Language)        |
                     +--------------------+--------------------+
                                          |
                     +--------------------v--------------------+
                     |  Stage 0: Multi-Query Expansion         |
                     |  (Groq LLM -- 3 technical variants)     |
                     +--------------------+--------------------+
                                          | 4 queries
                     +--------------------v--------------------+
                     |  Stage 1: Hybrid Recall                 |
                     |  +-------------+  +-----------------+   |
                     |  | BM25 Lexical|  | ChromaDB Dense  |   |
                     |  | (rank-bm25) |  | (all-MiniLM-L6) |   |
                     |  +------+------+  +--------+--------+   |
                     |         |    Reciprocal     |            |
                     |         +---Rank Fusion-----+            |
                     |              (RRF, k=60)                 |
                     |         Top-20 candidates                |
                     +--------------------+--------------------+
                                          |
                     +--------------------v--------------------+
                     |  Stage 2: Cross-Encoder Reranking       |
                     |  (ms-marco-MiniLM-L-6-v2, local CPU)    |
                     |  20 candidates -> Top 8                  |
                     +--------------------+--------------------+
                                          |
                     +--------------------v--------------------+
                     |  Stage 3: Token Budget Packing          |
                     |  (tiktoken -- fits chunks in context)   |
                     |  Available = ModelMax - Overhead         |
                     +--------------------+--------------------+
                                          |
                     +--------------------v--------------------+
                     |  Stage 4: Grounded LLM Generation       |
                     |  (Groq Cloud -- qwen/qwen3.8-27b)       |
                     |  Mandatory chunk-ID citations            |
                     +--------------------+--------------------+
                                          |
                     +--------------------v--------------------+
                     |      ANSWER WITH CITATIONS              |
                     |  [crowdstrike_outage_2024_p2_c4]        |
                     +-----------------------------------------+

               OBSERVABILITY LAYER (Phase 4)
               +----------------+  +----------------------+
               |  LangSmith     |  |  Ragas Evaluation    |
               |  (Trace every  |  |  Faithfulness        |
               |   LLM call)    |  |  Answer Relevancy    |
               |  Latency, cost |  |  Context Precision   |
               |  Token counts  |  |  Context Recall      |
               +----------------+  +----------------------+
```

---

## 9. Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation. A technique that retrieves relevant documents and injects them into an LLM prompt before generation. |
| **Chunk** | A fixed-size segment of a document. In this project, each chunk is 800 characters with 150-character overlap. |
| **Embedding** | A list of 384 floating-point numbers that represents the semantic meaning of a text. |
| **Cosine Similarity** | A number between 0 and 1 measuring how similar two vectors are. 1 = identical direction, 0 = no relation. |
| **BM25** | Best Match 25. A lexical search algorithm that scores documents by keyword frequency, normalized by document length. |
| **RRF** | Reciprocal Rank Fusion. A rank merging formula: `1/(k + rank)`. Combines two ranked lists without needing comparable scores. |
| **Cross-Encoder** | A transformer model that takes query + document as one input and produces a joint relevance score. More accurate than bi-encoders but slower. |
| **Bi-Encoder** | A model that encodes query and document independently then compares vectors. Used in ChromaDB. Fast but less precise. |
| **LCEL** | LangChain Expression Language. The pipe operator pattern for chaining LangChain components. |
| **OTPM** | Output Tokens Per Minute. Groq free-tier rate limit: 1,000 output tokens per minute per model. |
| **Faithfulness** | A Ragas metric. Measures what fraction of LLM answer claims are supported by the retrieved context. |
| **Precision@K** | The fraction of the Top-K retrieved chunks that are relevant to the query. |
| **HNSW** | Hierarchical Navigable Small World. The graph-based index algorithm ChromaDB uses for fast approximate nearest-neighbor search. |
| **Postmortem** | A technical document written after a system outage. It describes what happened, why it happened, and how to prevent it. |
| **Split-Brain** | A database condition where two nodes both believe they are the primary and accept conflicting writes. |
| **BGP Route Leak** | A network event where incorrect routing information is propagated through the internet, misdirecting traffic. |
| **tiktoken** | OpenAI's tokenizer library. We use it to count exact tokens in any text string. |
| **nproc** | The Linux operating system limit on the maximum number of user processes (threads) per user. |

---

*Document generated by IncidentLens session — ASD STE 100 style*  
*Project: `c:\Users\Vaishnav Ambilpur\Desktop\Applied Ai\Production-Level-RAG`*
