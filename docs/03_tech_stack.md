# 🧰 03 — Tech Stack

> **Context:** Every model, database, and tool used in IncidentLens — all 100% free.  
> **Prerequisite:** [02_architecture_overview.md](./02_architecture_overview.md)

---

## 3.1 Cloud-Hosted AI Models (Matching Claude 3.5 & GPT-4o Quality)

| Role | Model | Platform | Why It Matches Top-Tier Models | Cost |
|---|---|---|---|---|
| **Reasoning & Synthesis** | DeepSeek-R1 (Distill 70B) | Groq Cloud | SOTA multi-step deductive reasoning, log troubleshooting, and code analysis | **100% Free** (No credit card) |
| **Instruction & Coding** | Llama 3.3 (70B Versatile) | Groq Cloud | Meta's flagship; benchmark parity with GPT-4o at 500+ tokens/sec | **100% Free** (No credit card) |
| **Multi-Model Hub & Vision** | Llama 3.3 70B / DeepSeek-R1 / Free Router | OpenRouter (`openrouter/free`, `meta-llama/llama-3.3-70b-instruct:free`, `deepseek/deepseek-r1:free`) | OpenAI-compatible access to top open models with high availability and no vendor lock-in | **100% Free** (No credit card) |
| **Dense Embeddings** | BGE-M3 | FastEmbed / HuggingFace | High-dimensional semantic vectors; run locally or via API with zero rate limits | **100% Free** |
| **Cross-Encoder Reranker** | BGE-Reranker-Large | FastEmbed / SentenceTransformers | Lightweight cross-encoder filtering 80% of retrieval noise | **100% Free** |
| **Cloud Observability** | Langfuse Cloud & LangSmith | Langfuse / LangChain | Production-grade tracing, dashboards, latency monitoring | **100% Free Tier** |

---

## 3.2 Infrastructure & Databases

| Component | Technology | Purpose |
|---|---|---|
| **Vector Database** | ChromaDB (PersistentClient) | Dense vector storage with metadata filtering |
| **Sparse Index** | rank-bm25 | BM25 inverted index for exact keyword matching |
| **Graph Database** | NetworkX (in-memory) / Neo4j (optional) | Cascading failure chain modeling |
| **Agent Orchestrator** | LangGraph | Stateful graph workflows with conditional routing |
| **API Server** | FastAPI + Uvicorn | Async REST endpoints with streaming |
| **Agent Memory** | SQLite Checkpointer | Thread-persistent conversation state |

---

## 3.3 Python Dependencies

```text
# Core LangChain & LangGraph
langchain>=1.2.0
langchain-core>=1.3.0
langchain-community>=0.4.0
langgraph>=1.1.0

# 100% Free Cloud-Hosted AI Models (Groq + OpenRouter)
langchain-groq>=1.0.0
langchain-openai>=0.2.0
openai>=1.50.0
fastembed>=0.5.0
sentence-transformers>=3.4.0

# Vector Database & Retrieval
langchain-chroma>=1.1.0
chromadb>=1.5.0
rank-bm25>=0.2.2

# Ingestion & Document Processing
pypdf>=6.6.0
python-dotenv>=1.2.0
tiktoken>=0.12.0

# LLMOps, Observability & Evaluation
ragas>=0.4.0
langfuse>=3.0.0
langsmith>=0.4.0

# Production API & Testing
fastapi>=0.129.0
uvicorn>=0.40.0
pytest>=9.0.0
```

---

## 3.4 Environment Variables

```bash
# 100% Free Cloud LLMs (No credit cards needed)
GROQ_API_KEY=""         # Free SOTA (Llama 3.3 70B & DeepSeek-R1): https://console.groq.com/keys
OPENROUTER_API_KEY=""   # Free Models & Router (:free models): https://openrouter.ai/keys

# Cloud Observability (100% Free Cloud Tiers)
LANGFUSE_PUBLIC_KEY=""  # Free Cloud: https://cloud.langfuse.com
LANGFUSE_SECRET_KEY=""
LANGFUSE_HOST="https://cloud.langfuse.com"

LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY=""    # Free Cloud: https://smith.langchain.com
LANGCHAIN_PROJECT="incidentlens-production-rag"
```

---

## 3.5 API Key Signup Links

| Service | Signup URL | What You Get |
|---|---|---|
| Groq Cloud | https://console.groq.com/keys | Llama 3.3 70B + DeepSeek-R1 — unlimited free tier |
| OpenRouter | https://openrouter.ai/keys | Free models (`:free`) & auto-router (`openrouter/free`) |
| LangSmith | https://smith.langchain.com | Full tracing — free developer tier |
| Langfuse | https://cloud.langfuse.com | Open-source observability — free cloud tier |

---

## 🔗 Next

→ [04_system_diagrams.md](./04_system_diagrams.md) — Full end-to-end architecture & sequence diagrams.
