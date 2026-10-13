# 🔹 10 — Phase 4: Production Observability, LLMOps & Evaluation

> **Status:** ⬜ Not Started  
> **Focus:** Scientific evaluation harnesses, production tracing, and caching.  
> **Prerequisite:** Phase 3 complete — hybrid retrieval engine working.

---

## 4.1 Engineering Concepts

### LangSmith Tracing
Full-trace capture of every query through the pipeline:
- Prompt templates sent to the LLM
- Latency breakdown per step (retrieval, reranking, synthesis)
- Token counts (input, output, total cost)
- Retriever output inspection (which chunks were used)

### Ragas 4-Metric Evaluation Suite

| Metric | What It Measures | Target |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by retrieved context? (No hallucinations) | **≥ 92%** |
| **Answer Relevance** | Does the answer semantically match the user's question? | **≥ 90%** |
| **Context Precision** | Are the top-ranked retrieved chunks actually relevant? | **≥ 88%** |
| **Context Recall** | Does the retrieved context cover all ground-truth facts? | **≥ 85%** |

### Golden Test Dataset
- **20 curated Q&A pairs** with ground-truth answers and source citations
- **Coverage:** At least 5 questions per post-mortem document
- **Categories:** Root cause identification, timeline reconstruction, impact assessment, remediation steps, multi-hop reasoning

### Langfuse (Open-Source Alternative)
- Self-hosted or cloud tracing
- Compatible with LangChain callbacks
- Dashboard for latency, cost, and quality trends

### ChromaDB Production Architecture
- `PersistentClient` for disk-backed storage (survives restarts)
- Collection isolation (one collection per document type)
- Metadata filtering (by source, date, severity)

### Semantic Caching
- Hash embeddings of incoming queries
- If a near-duplicate query arrives (cosine > 0.95), return cached response
- Eliminates redundant LLM calls, reduces latency to ~10ms for cache hits

---

## 4.2 Deliverable Files

| File | Description | Verification |
|---|---|---|
| `modules/13_langsmith_setup.py` | LangSmith project config and tracing instrumentation | Traces visible in LangSmith dashboard |
| `modules/14_ragas_evaluation.py` | 20-question golden eval runner with benchmark scoring | Prints 4-metric scorecard |
| `modules/15_langfuse_observability.py` | Langfuse tracing adapter | Traces visible in Langfuse dashboard |
| `modules/16_chroma_production.py` | ChromaDB persistent disk storage + collection manager | Data survives script restart |
| `modules/17_cost_and_caching.py` | Semantic query cache | Cache hit returns in < 50ms |
| `data/eval_golden_dataset.json` | 20 curated Q&A pairs with ground truth | Valid JSON, all 4 post-mortems covered |

---

## 4.3 Golden Dataset Structure

```json
{
  "version": "1.0",
  "created": "2024-XX-XX",
  "test_cases": [
    {
      "id": "TC001",
      "question": "What Windows kernel driver caused the CrowdStrike global outage?",
      "ground_truth": "CSAgent.sys — a kernel-mode driver in the CrowdStrike Falcon sensor...",
      "source_document": "crowdstrike_outage_2024.pdf",
      "source_pages": [3, 7, 12],
      "category": "root_cause",
      "difficulty": "easy"
    },
    {
      "id": "TC002",
      "question": "Trace the cascading failure path from the AWS Kinesis outage to DynamoDB.",
      "ground_truth": "OS thread exhaustion in Kinesis front-end servers → ...",
      "source_document": "aws_kinesis_outage.pdf",
      "source_pages": [5, 8, 11],
      "category": "multi_hop",
      "difficulty": "hard"
    }
  ]
}
```

---

## 4.4 Evaluation Workflow

```
1. Load golden dataset (20 Q&A pairs)
2. For each question:
   a. Run through full RAG pipeline
   b. Capture: question, answer, retrieved_contexts, ground_truth
3. Pass all results to Ragas evaluator
4. Compute: faithfulness, answer_relevance, context_precision, context_recall
5. Print scorecard table
6. Log all traces to LangSmith + Langfuse
```

---

## 4.5 Success Criteria (Gate to Phase 5)

- [ ] LangSmith traces appear with full step-by-step latency breakdown
- [ ] Ragas scorecard shows baseline metrics across all 20 test cases
- [ ] Langfuse dashboard displays cost and latency trends
- [ ] ChromaDB persists data across script restarts
- [ ] Semantic cache achieves < 50ms on duplicate queries
- [ ] Golden dataset JSON validated and version-controlled

---

## 🔗 Next

→ [11_phase5_security_api.md](./11_phase5_security_api.md) — Security guardrails, LangGraph agent, and REST API.
