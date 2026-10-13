# 📐 04 — System Diagrams

> **Context:** Visual architecture and user-flow diagrams for the complete IncidentLens pipeline.  
> **Prerequisite:** [02_architecture_overview.md](./02_architecture_overview.md)

---

## 4.1 End-to-End Architectural Diagram

```mermaid
graph TD
    subgraph Data_Ingestion["Data Ingestion & Preprocessing (100% Open-Source)"]
        A1["Post-Mortem PDFs (CrowdStrike, Cloudflare, AWS)"] --> B1["Doc Loaders (PyPDF / Markdown)"]
        B1 --> C1["Contextual Retrieval (Anthropic Prepending)"]
        C1 --> D1["Late Chunking / Recursive Splitter"]
    end

    subgraph Storage_Indexing["Storage & Indexing"]
        D1 --> E1[("ChromaDB Persistent Store (BGE-M3 Dense)")]
        D1 --> E2["BM25 Inverted Index (Sparse)"]
        D1 --> E3[("NetworkX / GraphRAG (Failure Chains)")]
    end

    subgraph Hybrid_Retrieval["Hybrid Retrieval & Reranking"]
        Q["User Query"] --> F1["Security Guardrail (Injection & PII Sanitizer)"]
        F1 --> F2["Query Expansion & Multi-Query"]
        F2 --> G1["Dense Vector Search (ChromaDB)"]
        F2 --> G2["Sparse Keyword Search (BM25)"]
        G1 --> H1["Reciprocal Rank Fusion (RRF)"]
        G2 --> H1
        H1 --> H2["BGE Cross-Encoder Reranker"]
        H2 --> H3["Token Budgeting & Smart Truncation"]
    end

    subgraph Agentic_Core["Agentic LangGraph Core"]
        H3 --> I1["Relevance Grading Node"]
        I1 -->|Low Relevance| I2["Query Rewrite Node & Fallback"]
        I2 --> F2
        I1 -->|High Relevance| I3["Synthesis Node (Llama 3.3 / DeepSeek-R1)"]
        I3 --> I4["Hallucination & Citation Check Node"]
        I4 -->|Hallucinated| I3
        I4 -->|Grounded| J1["Sanitized Production Response"]
    end

    subgraph Observability_LLMOps["Observability, LLMOps & Evaluation"]
        J1 --> K1["LangSmith Real-Time Tracing"]
        J1 --> K2["Langfuse Self-Hosted Tracing"]
        K1 --> L1["Ragas Golden Eval Harness (20 Test Cases)"]
        K2 --> L1
        L1 --> L2["Dashboard: Faithfulness, Precision, Recall"]
    end
```

---

## 4.2 End-to-End User Flow (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    actor Engineer as On-Call SRE Engineer
    participant Guard as Security Guardrail
    participant Agent as LangGraph Agentic Core
    participant Search as Hybrid Engine (BGE-M3 + BM25)
    participant LLM as Llama 3.3 / DeepSeek-R1
    participant Observability as Langfuse / LangSmith

    Engineer->>Guard: "Why did Kafka stall and what caused memory bugcheck 0x50?"
    Guard->>Guard: Sanitize input & check for prompt injection
    Guard->>Agent: Clean Query
    Agent->>Search: Dense vector query + BM25 keyword query
    Search-->>Agent: Top candidate chunks (CrowdStrike & AWS Post-Mortems)
    Agent->>Agent: Relevance Grader evaluates chunks
    alt Low relevance score
        Agent->>Agent: Query rewriter broadens technical terms
    end
    Agent->>LLM: Augmented prompt with verified context & token budget
    LLM-->>Agent: Synthesized root cause & mitigation steps
    Agent->>Agent: Hallucination & Citation Check against source text
    Agent->>Observability: Log trace, latency (ms), and token counts
    Agent-->>Engineer: Verified root cause, exact component failure, safe rollback steps, and page citations
```

---

## 4.3 Data Flow Summary

```
PDF/MD Documents
    ↓
[Loaders] → [Contextual Prepending] → [Late Chunking]
    ↓                                       ↓
[ChromaDB Dense Index]              [BM25 Sparse Index]    [NetworkX Graph]
    ↓                                       ↓                     ↓
    └──────────── Reciprocal Rank Fusion ──────────────────────────┘
                            ↓
                  [Cross-Encoder Reranker]
                            ↓
                    [Token Budgeter]
                            ↓
                  [LangGraph Agent Loop]
                    ├── Relevance Grade
                    ├── Query Rewrite (if low)
                    ├── LLM Synthesis
                    └── Hallucination Check
                            ↓
                  [Grounded Response + Citations]
                            ↓
                  [LangSmith / Langfuse Trace]
```

---

## 🔗 Next

→ [05_directory_structure.md](./05_directory_structure.md) — Complete file and folder layout.
