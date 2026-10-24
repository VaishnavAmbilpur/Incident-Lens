# 16 — Observability

> **Context:** This file gives the settings for LangSmith and Langfuse.
> **Prerequisite:** [03_tech_stack.md](./03_tech_stack.md), section 3.5
> **Status:** 📄 Defined. The LangSmith key gives error 403 today.

---

## 16.1 The Two Tools

IncidentLens uses two tracing tools. The two tools are not the same.

**Table 16-1 — The Division of Work**

| Item | LangSmith | Langfuse |
|---|---|---|
| Tracing | Yes. The primary tool. | Yes. The second tool. |
| Cost of each call | Basic | Yes. The primary tool. |
| LangGraph integration | Native. One setting. | Needs an adapter. |
| Development and evaluation | Yes. The primary tool. | Good |
| Hosting | Cloud | Cloud, or your own host |
| Source code | Closed | Open source |

**One sentence:** LangSmith finds the reason for a bad answer. Langfuse shows the cost and the speed.

---

## 16.2 The Settings

**Table 16-2 — The Environment Keys**

| Key | Tool | Value |
|---|---|---|
| `LANGCHAIN_TRACING_V2` | LangSmith | `true` |
| `LANGCHAIN_API_KEY` | LangSmith | Your key |
| `LANGCHAIN_PROJECT` | LangSmith | `incidentlens` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse | Your public key |
| `LANGFUSE_SECRET_KEY` | Langfuse | Your secret key |
| `LANGFUSE_BASE_URL` | Langfuse | `https://us.cloud.langfuse.com` |
| `TRACE_LANGSMITH` | Switch | `1` or `0` |
| `TRACE_LANGFUSE` | Switch | `1` or `0` |

> **WARNING:** Use `LANGFUSE_BASE_URL`. The name `LANGFUSE_HOST` is the old name. Langfuse version 4 ignores the old name and the trace fails without a message.

---

## 16.3 The Switch Table

**Table 16-3 — The Switch**

| `TRACE_LANGSMITH` | `TRACE_LANGFUSE` | Result |
|---|---|---|
| `1` | `1` | Both tools operate |
| `1` | `0` | LangSmith only |
| `0` | `1` | Langfuse only |
| `0` | `0` | No trace. The pipeline operates. |

**The rule:** The pipeline does not know which tool operates. The pipeline calls `core/tracing.py` only. The adapter files do the work.

---

## 16.4 The Files

**Table 16-4 — The Tracing Files**

| Path | Function |
|---|---|
| `core/tracing.py` | One interface. The pipeline calls this file. |
| `core/tracing_langsmith.py` | The LangSmith adapter |
| `core/tracing_langfuse.py` | The Langfuse adapter |

**Procedure to add a third tool:**

1. Make one adapter file.
2. Add one branch in `core/tracing.py`.
3. Do not change the pipeline.

---

## 16.5 What the Trace Shows

**Table 16-5 — The Traces for Each Stage**

| Stage | What the trace shows |
|---|---|
| 1 Ingest | The document count and the duplicate count |
| 2 Chunk | The chunk count and the chunk size |
| 3 Embed | The vector size and the speed |
| 4 Index | The vector count and the build time |
| 5 Retrieve | The query, the chunks, and the scores |
| 6 Fuse | The two result lists and the fused list |
| 7 Rerank | The order before and the order after |
| 8 Contextualize | The header that the model made |
| 9 Generate | The prompt, the answer, and the token count |
| 10 Evaluate | The judge score for each metric |

> **NOTE:** This table is the reason to keep the tracing tools. The trace makes each stage of the pipeline visible. A visible pipeline shows that you understand pipelines.

---

## 16.6 The Procedure for LangSmith

1. Get a key at `smith.langchain.com`.
2. Put the key in the file `.env`.
3. Set `LANGCHAIN_TRACING_V2` to `true`.
4. Set `TRACE_LANGSMITH` to `1`.
5. Run one query.
6. Open the LangSmith dashboard.
7. **Check:** The trace shows each stage as a separate step.

> **NOTE:** The present key returns error 403. The cause is a cut key or an invalid key. Get a new key. Remove the space characters at the end of the key.

---

## 16.7 The Procedure for Langfuse

1. Get a public key and a secret key at `cloud.langfuse.com`.
2. Put the two keys in the file `.env`.
3. Set `LANGFUSE_BASE_URL` to `https://us.cloud.langfuse.com`.
4. Set `TRACE_LANGFUSE` to `1`.
5. Run one query.
6. Open the Langfuse dashboard.
7. **Check:** The trace shows the cost of the query.

The venv contains langfuse 4.17.0. Write the code for the version 4 API.

---

## 16.8 Warnings

1. **Do not trace the whole ablation table with both tools.** A trace for each cell makes a large number of records. Trace one configuration at a time.
2. **Do not keep a second tracing library.** Two adapters are sufficient.
3. **Do not let a tracing failure stop the pipeline.** If the trace fails, write a warning and continue. The answer is more important than the trace.
4. **Do not commit the file `.env`.** The keys are secret.

---

## 16.9 Checks

**Table 16-6 — The Checks**

| Check | Criterion |
|---|---|
| C1 | The LangSmith dashboard shows all 10 stages in one trace. |
| C2 | The Langfuse dashboard shows the cost of one query. |
| C3 | The switch in Table 16-3 turns each tool on and off. |
| C4 | The pipeline operates when both switches are `0`. |
| C5 | A tracing failure does not stop the answer. |

---

## 🔗 Next

→ [17_interview_defense.md](./17_interview_defense.md) — One answer for each stage.
