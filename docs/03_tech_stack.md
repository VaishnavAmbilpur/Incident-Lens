# 🧰 03 — Tech Stack

> **Context:** Every model, database, and tool used in IncidentLens.
> **Prerequisite:** [02_architecture_overview.md](./02_architecture_overview.md)
> **Revision:** 2. The versions in this file are the versions that are installed. Read section 3.6.

---

## 3.1 Models

**Table 3-1 — The Models**

| Role | Model | Platform | Note | Cost |
|---|---|---|---|---|
| Reasoning and synthesis | `llama-3.3-70b-versatile` | Groq | The main answer model | Free tier |
| Reasoning and synthesis | `deepseek-r1-distill-llama-70b` | Groq | The second answer model | Free tier |
| Judge | `qwen/qwen3-8b` | Groq | The primary judge for the 4 metrics | Free tier |
| Judge | `allam-2-7b` | Groq | The second judge | Free tier |
| Judge | `openai/gpt-oss-20b` | Groq | The third judge in the chain | Free tier |
| Fallback | A free model | OpenRouter | Used when Groq stops | Free tier |
| Dense embeddings | `all-MiniLM-L6-v2` | ChromaDB ONNX, local | 384 dimensions. Runs on the CPU. | Free, no key |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | sentence-transformers, local | 22M parameters. Runs on the CPU. | Free, no key |
| Vision (optional) | A vision model | OpenRouter | Module 26 only | Free tier |

> **NOTE:** The limit on Groq applies to each model, but not to the account. The judge chain uses three models for this reason.

---

## 3.2 Infrastructure and Databases

**Table 3-2 — The Infrastructure**

| Component | Technology | Purpose |
|---|---|---|
| Vector database | ChromaDB PersistentClient | Dense vector storage with metadata filters |
| Sparse index | rank-bm25 | BM25 inverted index for exact keyword matches |
| Graph library | NetworkX | Cascading failure chains. Module 25. |
| Agent orchestrator | LangGraph | Stateful graph workflows with conditional routing |
| API server | FastAPI and Uvicorn | Async REST endpoints with streaming |
| Agent memory | SQLite checkpointer | Thread-persistent conversation state |
| Response cache | A disk cache | A second query makes no provider call |

---

## 3.3 Python Dependencies

The versions below are the versions that are installed in `venv`. Pin these versions. A reviewer must get the same result as you.

```text
# Core LangChain and LangGraph
langchain==1.4.4
langchain-core==1.6.9
langchain-community==0.4.2
langgraph==1.2.14

# Cloud model providers
langchain-groq==1.1.3
langchain-openai==1.1.9
openai==1.109.1

# Local models
sentence-transformers==6.1.0
fastembed==0.9.0

# Vector database and retrieval
langchain-chroma==1.1.0
chromadb==1.5.9
rank-bm25==0.2.2
networkx==3.7

# Ingestion and document processing
pypdf==6.20.0
python-dotenv
tiktoken==0.14.0

# LLMOps, observability and evaluation
ragas==0.4.3
langfuse==4.17.0
langsmith==0.14.7

# API and testing
fastapi==0.143.0
uvicorn==0.54.0
pytest==9.1.1
```

> **WARNING:** Do not write a version range such as `>=` in this file. Write the exact version. A version range gives a different result on a different day.

---

## 3.4 Environment Variables

```bash
# Cloud models (free tier, no credit card)
GROQ_API_KEY=""            # https://console.groq.com/keys
OPENROUTER_API_KEY=""      # https://openrouter.ai/keys

# LangSmith tracing
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY=""       # https://smith.langchain.com
LANGCHAIN_PROJECT="incidentlens"

# Langfuse tracing and cost
LANGFUSE_PUBLIC_KEY=""     # https://cloud.langfuse.com
LANGFUSE_SECRET_KEY=""
LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"

# Tracing switches. Read section 3.5.
TRACE_LANGSMITH="1"
TRACE_LANGFUSE="1"
```

> **NOTE:** `LANGFUSE_BASE_URL` is the correct name for Langfuse version 4. The old name is `LANGFUSE_HOST`. The old name causes a silent failure.

---

## 3.5 The Tracing Tools

IncidentLens uses two tracing tools. The two tools are not the same.

**Table 3-3 — LangSmith compared to Langfuse**

| Item | LangSmith | Langfuse |
|---|---|---|
| Owner | LangChain Incorporated | Langfuse GmbH, independent |
| Source code | Closed | Open source |
| Hosting | Cloud only | Cloud, or your own host |
| Main purpose | Develop and test | Monitor in production |
| Evaluation | Strong. Datasets, experiments, version comparison. | Good. Scores and datasets. |
| Cost control | Basic | Strong. Cost of each model and each user. |
| LangGraph integration | Native. One setting. | Needs an adapter. |
| Best for | A team that uses LangChain | A team that needs cost control or self-hosting |

**The division of work:**

- **LangSmith finds the reason for a bad answer.** It is the primary tool.
- **Langfuse shows the cost and the speed.** It is the second tool.

**The switches:**

| Switch | Effect |
|---|---|
| `TRACE_LANGSMITH=1` | LangSmith operates |
| `TRACE_LANGFUSE=1` | Langfuse operates |
| Both are `1` | Both tools operate |
| Both are `0` | No trace. The pipeline operates. |

The file `core/tracing.py` holds one interface. The two adapter files do the work. The pipeline calls `core/tracing.py` only. Therefore, a change to one tool does not change the pipeline.

---

## 3.6 Correction to the First Version of This File

The first version of this file contained five errors. Table 3-4 gives the corrections. Use Table 3-4, not the first version.

**Table 3-4 — The Corrections**

| Item | First version said | The truth |
|---|---|---|
| Embedding model | BGE-M3 | `all-MiniLM-L6-v2`, 384 dimensions. No BGE model is installed. |
| Reranker model | BGE-Reranker-Large | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LangChain version | `langchain>=1.2.0` | `langchain==1.4.4` |
| Langfuse version | `langfuse>=3.0.0` | `langfuse==4.17.0`. The API is the version 4 API. |
| Langfuse host key | `LANGFUSE_HOST` | `LANGFUSE_BASE_URL` |
| Quality bar | "100% free, matches GPT-4o" | The measured values are in [15_evals.md](./15_evals.md). Some metrics do not meet the target. |

> **WARNING:** A document that does not agree with the code is worse than no document. A reviewer finds the difference in one minute. Keep this file and `requirements.txt` equal.

---

## 3.7 API Key Signup Links

**Table 3-5 — The Signup Links**

| Service | URL | What you get |
|---|---|---|
| Groq Cloud | https://console.groq.com/keys | Llama 3.3 70B, DeepSeek-R1, and judge models |
| OpenRouter | https://openrouter.ai/keys | Free models and the auto-router |
| LangSmith | https://smith.langchain.com | Tracing. Free developer tier. |
| Langfuse | https://cloud.langfuse.com | Tracing and cost. Free cloud tier. |

> **NOTE:** The LangSmith key in the project returned error 403. Get a new key. Remove the space characters at the end of the key.

---

## 🔗 Next

→ [04_system_diagrams.md](./04_system_diagrams.md) — Full end-to-end architecture and sequence diagrams.
