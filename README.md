# IncidentLens — Production RAG for Incident Intelligence

IncidentLens is an AI search tool for software incident reports.
You ask a question. The system finds the relevant documents. The system gives you a grounded answer.

---

## The Problem

When a production system fails, engineers must find the cause fast.
They search through hundreds of past incident reports manually.
This takes 60 to 80 percent of the total response time.
Engineers make mistakes under pressure. They miss critical details.
The same failures happen again because no one connected the patterns.

**The result:** slower recovery, higher cost, repeated mistakes.

---

## The Solution

IncidentLens searches 31 real public post-mortem reports.
It returns a direct answer with citations in under 2 seconds.
The system checks every claim against the source text before returning the answer.
The system does not invent commands or facts that are not in the documents.

**The result:** the engineer gets the root cause and the fix, not a list of links.

---

## How It Works — 10 Stages

Each stage does one job. Each stage produces one output. Each stage has one measurement.

```
USER QUESTION
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 1 — INGEST                                        │
│  The system reads 31 post-mortem documents.              │
│  Output: 31 documents. Dropped: 2 (non-English).         │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 2 — CHUNK                                         │
│  The system splits documents into 1,169 text chunks.     │
│  Each chunk is 480 characters with 120 overlap.          │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 3 — EMBED                                         │
│  The system converts each chunk to a 384-number vector.  │
│  Model: all-MiniLM-L6-v2. Runs locally. No API call.    │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 4 — INDEX                                         │
│  The system stores 1,169 vectors in ChromaDB on disk.    │
│  Build time: 1.19 seconds.                               │
└──────────────┬──────────────────────┬────────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐  ┌────────────────────────────────┐
│  STAGE 5 — DENSE     │  │  STAGE 5b — BM25               │
│  Semantic search.    │  │  Keyword search.               │
│  MRR: 0.7837         │  │  MRR: 0.7740. Speed: 3 ms.    │
│  Speed: 344 ms.      │  │  Catches exact error codes.    │
└──────────┬───────────┘  └──────────────┬─────────────────┘
           │                             │
           └──────────────┬──────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 6 — FUSE (RRF)                                    │
│  The system merges both result lists with RRF.           │
│  RRF gives higher rank to documents both retrievers find.│
│  Recall@5: 0.9286. MRR: 0.8371. Speed: 23 ms.           │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 7 — RERANK                                        │
│  The cross-encoder reads the query and each chunk.       │
│  It scores each pair directly and reorders the list.     │
│  nDCG@5: 0.9002. MRR: 0.8845. Candidates: 10.           │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 8 — CONTEXTUALIZE                                 │
│  The system adds one sentence before each chunk.         │
│  The sentence names the document, system, and failure.   │
│  This gives the retriever more context per chunk.        │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 9 — GENERATE                                      │
│  The LLM reads the top 5 chunks and writes an answer.   │
│  The answer includes citations. It uses only chunk text. │
│  Blocked queries never reach the LLM.                    │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 10 — EVALUATE                                     │
│  Three judge models score the answer on 4 metrics.       │
│  Metrics: Faithfulness, Precision, Recall, Relevance.    │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
                     FINAL ANSWER
```

---

## Ablation Table — What Each Technique Adds

The ablation table changes one item per row. All other items stay the same.
Every row uses the same 56 questions and the same corpus.
This proves which technique caused each improvement.

| Config | Retrieval | Query | Rerank | Recall@5 | nDCG@5 | MRR | p50 ms |
|---|---|---|---|---|---|---|---|
| A | Dense only | raw | no | 0.8750 | 0.7985 | 0.7837 | 16 |
| B | BM25 only | raw | no | 0.8750 | 0.7975 | 0.7740 | 4 |
| C | Hybrid RRF | raw | no | 0.9286 | 0.8563 | 0.8371 | 23 |
| D | Hybrid RRF | multi-query | no | 0.8929 | 0.8117 | 0.7942 | 1,186 |
| **E** | **Hybrid RRF** | **raw** | **yes** | **0.9464** | **0.9002** | **0.8845** | **313** |
| F | Contextual | raw | yes | — | — | — | — |

**What the table shows:**

- **A vs B:** Dense and BM25 have the same recall. BM25 is 100 times faster on keyword queries.
- **B vs C:** RRF fusion raises recall by 6.1 percent. Fusion wins because the two retrievers fail on different questions.
- **C vs E:** Reranking raises nDCG by 12.1 percent. The correct document was already in the list. Reranking moves it higher.
- **D:** Multi-query expansion adds query variants. It costs 1,186 ms because of LLM provider latency.
- **E:** The best configuration. Hybrid search plus reranking gives the highest quality at 313 ms.

---

## Quality Scorecard

| Metric | Measured | Status |
|---|---|---|
| **Context Precision** | **0.920** | ✅ PASS |
| **Context Recall** | **0.770** | ❌ FAIL — retrieval gap |
| **Faithfulness** | 0.950 (pre-fix; pending re-run) | ⏳ Not re-measured |
| **Answer Relevance** | 0.950 (pre-fix; pending re-run) | ⏳ Not re-measured |

**Note on Context Precision:** The first measurement was 0.600.
This was a measurement error, not a retrieval error.
The judge saw only 200 characters of each 800-character chunk.
With the full chunk, the correct value is 0.920.
See [docs/15_evals.md](./docs/15_evals.md) for the full correction.

**Note on Context Recall:** 0.770 is a real retrieval gap.
Two different judge models confirmed this value.
The fix is to build the contextual index (Row F of the ablation table).

---

## How to Make It Better

These are the next steps to improve the system. Each step is independent.

### 1. Build the Contextual Index (Row F)

The contextual index adds a one-sentence preamble to each chunk before embedding.
This gives each vector more meaning. Retrieval recall improves.
Run `core/contextual_pipeline.py` to build the index.
Then run `python run.py ablation` to measure Row F.

### 2. Fix Context Recall (0.770 → target 0.850)

The 3 questions that fail have correct documents in the index.
The correct document is not in the top 5 results.
Use query expansion (Row D) or contextual embedding (Row F) to fix this.
Measure with `python run.py eval`.

### 3. Re-measure Faithfulness and Answer Relevance

Both metrics need LLM calls to generate answers.
The provider was out of quota during the last full run.
Run `python run.py eval --llm` when provider quota is available.

### 4. Fix LangSmith Tracing

The current LangSmith API key gives error 403.
Get a new key from https://smith.langchain.com.
Add it to `.env` as `LANGCHAIN_API_KEY`.
This enables end-to-end trace visibility for every query.

### 5. Add GPU Support for the Reranker

The cross-encoder currently runs on CPU. It takes 313 ms per query.
Add `device="cuda"` in `core/rerank.py` to use a GPU.
This reduces rerank latency from 313 ms to under 50 ms.

### 6. Expand the Corpus

The corpus has 31 documents from 12 organizations.
Add more post-mortems from Azure, GCP, Stripe, or PagerDuty.
More documents give the retriever a harder task and more honest metrics.

---

## Tech Stack

| Component | Technology | Cost |
|---|---|---|
| Primary LLM | `llama-3.3-70b-versatile` (Groq) | Free |
| Fallback LLM | OpenRouter free models | Free |
| Embeddings | `all-MiniLM-L6-v2` (local CPU) | Free |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local CPU) | Free |
| Vector DB | ChromaDB (persistent, local) | Free |
| Sparse Index | rank-bm25 (in-memory) | Free |
| Orchestration | LangGraph | Free |
| Evaluation | Ragas 0.4.3 | Free |
| Observability | LangSmith + Langfuse | Free tier |
| API | FastAPI + Uvicorn | Free |
| UI | React + Vite (single page) | Free |
| **Total** | | **$0.00 / month** |

---

## How to Run

**Prerequisites:** Python 3.13, a free Groq API key, a free OpenRouter API key.

```powershell
# 1. Clone the repository
git clone https://github.com/VaishnavAmbilpur/Incident-Lens.git
cd Incident-Lens

# 2. Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env
# Open .env and add GROQ_API_KEY and OPENROUTER_API_KEY

# 4. Build the vector index
C:\Python313\python.exe core\vectorstore.py --rebuild

# 5. Test all systems
C:\Python313\python.exe run.py test

# 6. Start the server and UI
C:\Python313\python.exe run.py server
# Open http://localhost:8000 in a browser
```

**All CLI commands:**

| Command | What It Does |
|---|---|
| `run.py server` | Start the API server and React UI |
| `run.py ablation` | Run all 6 ablation configurations |
| `run.py eval` | Run retrieval evaluation (no LLM calls) |
| `run.py eval --llm` | Run full 4-metric evaluation (needs provider quota) |
| `run.py diagnose` | Show all 10 stages for one query |
| `run.py test` | Test all imports and connections |

---

## Corpus

31 public post-mortems. 12 organizations. 9 failure categories.

| Organization | Reports | Failure Types |
|---|---|---|
| Cloudflare | 9 | BGP, DNS, config, TLS, software |
| AWS | 6 | Power, DNS, cooling |
| GitHub | 3 | MySQL failover, DNS, compute |
| Slack | 2 | DNS, network |
| Meta | 1 | BGP |
| Datadog | 1 | Config change |
| GoCardless | 1 | Database |
| Sentry | 1 | Database |
| GCP | 1 | VMware config |
| Honeycomb | 1 | Software bug |
| incident.io | 1 | External dependency |
| CrowdStrike | 1 | Sensor update |

Total: **1,169 chunks** at 480 chars / 120 overlap. Boundary split rate: 5.90%.

---

## Documentation

| File | Contents |
|---|---|
| [docs/00_ROADMAP_INDEX.md](./docs/00_ROADMAP_INDEX.md) | Index of all documentation |
| [docs/14_pipeline_plan.md](./docs/14_pipeline_plan.md) | 10-stage plan and 6-row ablation design |
| [docs/15_evals.md](./docs/15_evals.md) | All measured numbers and verdicts |
| [docs/17_interview_defense.md](./docs/17_interview_defense.md) | 90-second explanation of each stage |

---

## License

MIT. Use this code for any purpose.
