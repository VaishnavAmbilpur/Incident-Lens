# 📋 01 — Problem & Solution

> **Context:** This file explains *why* IncidentLens exists. Read this first before diving into architecture.

---

## 1.1 The Business Problem (Plain English)

When major apps like payment gateways, ride-sharing platforms, or cloud services crash, companies lose massive amounts of money and customer trust **every minute**. The biggest delay isn't fixing the problem — it's that engineers spend hours reading through hundreds of pages of past failure documents and complex charts just to figure out **what went wrong**.

### Real-World Impact
- A major cloud outage can cost **$100K–$1M+ per hour** in lost revenue.
- On-call SREs spend **60–80% of incident response time** on diagnosis, not remediation.
- Institutional knowledge is trapped in unstructured PDFs, Confluence pages, and Slack threads.

---

## 1.2 The Solution (What IncidentLens Does)

**IncidentLens acts like an instant AI emergency doctor for computer systems.**

Instead of engineers frantically searching through hundreds of old incident reports at 2 AM, the AI:
1. **Reads the error** (natural language query or raw error code).
2. **Finds the exact root cause** from indexed post-mortem history.
3. **Gives safe, step-by-step remediation instructions** with exact source citations.
4. **Prevents hallucination** — never invents a destructive command.

### Key Value Proposition
| Without IncidentLens | With IncidentLens |
|---|---|
| Engineer manually searches 200+ pages of PDFs | AI retrieves the 3 most relevant passages in < 600ms |
| Misses exact error codes buried in dense reports | BM25 keyword search catches `SIGSEGV`, `0x00000050` precisely |
| No visual chart comprehension | OpenRouter Vision / ColPali reads Grafana CPU/latency spike screenshots |
| Risk of acting on hallucinated advice | Self-correcting agent verifies every claim against source docs |

---

## 1.3 The Technical Problem

Standard RAG (Retrieval-Augmented Generation) systems **fail on infrastructure incidents** because they:

1. **Miss exact error codes** — Dense embeddings can't match `SIGSEGV` or `0x00000050` literally.
2. **Ignore visual evidence** — Grafana charts and architecture diagrams are invisible to text-only RAG.
3. **Can't trace cascading failures** — `DNS Route Leak → Kafka Queue Full → DB Starvation` requires multi-hop graph reasoning.
4. **Hallucinate destructive commands** — A naive LLM might invent a `rm -rf` or `DROP TABLE` command when the source docs don't specify one.
5. **Lose context in long documents** — The "lost in the middle" problem causes LLMs to forget critical details in long prompts.

---

## 1.4 The Technical Solution (One Sentence)

> **IncidentLens combines exact-code keyword search (BM25), cloud vision for metric charts, GraphRAG for cascading failure chains, and self-correcting agentic guardrails — all powered by free SOTA cloud models (Groq Llama 3.3 / DeepSeek-R1 and OpenRouter Free Models).**

---

## 🔗 Next

→ [02_architecture_overview.md](./02_architecture_overview.md) — How the 6 SOTA techniques fit together.
