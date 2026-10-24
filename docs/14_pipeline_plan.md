# 14 — Pipeline Plan (IL-PLAN-001)

> **Context:** This file gives the plan for the change of IncidentLens.
> **Prerequisite:** [01_problem_and_solution.md](./01_problem_and_solution.md)
> **Status:** 📄 Defined. The stages are not complete.

---

## 14.1 Purpose

This plan tells you how to change IncidentLens.

IncidentLens shows that you know AI. IncidentLens shows that you can build pipelines. Flow-Q shows your full-stack work. IncidentLens does not show full-stack work.

---

## 14.2 Approved Words

Use these words in this project. Use one word for one meaning.

| Word | Meaning |
|---|---|
| pipeline | A sequence of stages. Data goes in at stage 1. The answer comes out at stage 10. |
| stage | One step in the pipeline. |
| chunk | A part of a document. |
| corpus | All your documents together. |
| embed | To change text into numbers. |
| retrieve | To find chunks. |
| rerank | To put chunks in a better order. |
| evaluate | To measure quality. |
| ablation | A test. Change one item. Keep all other items the same. |

---

## 14.3 Condition of the Project Today

IncidentLens has 16,057 lines of code in 26 files. Each file is a separate demonstration. The files do not share code. Each file contains the same helper functions. Each file prints a report. No file gives the full answer to a question.

You cannot keep 16,057 lines in your head. Therefore, you cannot answer questions about them.

The corpus has 4 documents and 105 chunks. The retrieval score is 1.000. A score of 1.000 is perfect. A perfect score has no space for improvement. You cannot show that a new technique is better. You cannot measure the change.

---

## 14.4 Target

Build one library. The library has 10 modules. One module is one stage. Each stage has one input. Each stage has one output. Each stage has one measurement.

**Table 14-1 — The Ten Stages**

| No. | Stage | Module | Measurement |
|---|---|---|---|
| 1 | Ingest | `core/ingest.py` | Count documents. Count duplicates. |
| 2 | Chunk | `core/chunking.py` | Count chunks. Measure the split quality. |
| 3 | Embed | `core/embeddings.py` | Measure the vector size. Measure the speed. |
| 4 | Index | `core/vectorstore.py` | Count the vectors. Measure the build time. |
| 5 | Retrieve | `core/retrieval.py` | Measure Recall@k. Measure MRR. |
| 6 | Fuse | `core/fusion.py` | Compare RRF and the weighted sum. |
| 7 | Rerank | `core/rerank.py` | Measure the change in nDCG. |
| 8 | Contextualize | `core/contextual.py` | Compare with the baseline. |
| 9 | Generate | `core/generate.py` | Measure faithfulness. |
| 10 | Evaluate | `core/evaluate.py` | Measure 4 metrics. |

**The important property:** You can operate each stage alone. You can measure each stage alone. This property shows that you understand pipelines.

---

## 14.5 The Main Artifact — The Ablation Table

The ablation table is more important than the code.

**Table 14-2 — The Ablation Table**

| Configuration | Chunk | Retrieval | Query | Rerank | Recall@5 | nDCG@5 | Faithfulness | p50 |
|---|---|---|---|---|---|---|---|---|
| A | fixed-800 | dense | raw | no | — | — | — | — |
| B | recursive | dense | raw | no | — | — | — | — |
| C | recursive | hybrid | raw | no | — | — | — | — |
| D | recursive | hybrid | multi-query | no | — | — | — | — |
| E | recursive | hybrid | multi-query | yes | — | — | — | — |
| F | late-chunk | hybrid | multi-query | yes | — | — | — | — |

One command makes this table:

```bash
incidentlens ablate
```

Table 14-2 tells you three things:

1. That you can isolate one item.
2. That you know which item moves which metric.
3. That your numbers are true and testable.

---

## 14.6 Work Phases

Do the phases in this order. Do not start a phase before you complete the phase above it.

### Phase 1 — Expand the Corpus

1. Get 25 to 40 real post-mortems from public sources.
2. Remove the duplicate documents.
3. Add metadata to each document. Use these metadata items: source, date, system, failure type.
4. Build the vector index again.
5. Measure the recall.
6. **Check:** The value of Precision@4 must go below 1.000. If the value stays at 1.000, the corpus is too small. Get more documents.

> **WARNING:** Do not start Phase 2 before you complete this check. All later measurements are not possible on a saturated metric.

### Phase 2 — Extract `core/`

1. Make the directory `core/`.
2. Make 10 modules. Use Table 14-1.
3. Move the code from the 26 old files into the 10 modules.
4. Remove all `print()` statements from `core/`. Let the CLI do the reporting.
5. **Check:** Each stage operates alone. Each stage gives its measurement.

### Phase 3 — Provider Layer

1. Make `core/providers.py`.
2. Add a fallback chain: Groq, then OpenRouter.
3. Add a retry function for error 429. Add a delay between the retries.
4. Add a disk cache. The cache stores the true provider answers.
5. **Check:** A second run of the same query makes no provider call.

> **NOTE:** There is no offline mode. There is no fake provider. The pipeline always uses a real model.

### Phase 4 — Make `evaluate.py`

1. Keep the 4 metrics: context precision, context recall, faithfulness, answer relevance.
2. Keep the judge chain.
3. If a metric has no data, write `NOT MEASURED`. Do not write a false score.
4. **Check:** The evaluator reports an outage as an outage. It does not report an outage as a quality loss.

### Phase 5 — Make the Ablation Table

1. Make `ablation.py`.
2. Run all 6 configurations. Use Table 14-2.
3. Write the results in [15_evals.md](./15_evals.md).
4. **Check:** The table has no empty cells.

### Phase 6 — Add the Diagnose Command

1. Make `diagnose.py`.
2. For one query, report each stage. Show the data from each stage.
3. Report the stage that caused a bad answer.
4. **Check:** You can tell a retrieval failure from a generation failure.

### Phase 7 — Make the Thin UI

1. Keep one page only.
2. The page shows the stages in sequence.
3. The page shows the retrieved chunks before the rerank and after the rerank.
4. Delete the other 17 components.
5. **Check:** Record a video of 15 seconds. The video shows the pipeline.

### Phase 8 — Learn the Project

1. Close the old files.
2. Write each `core/` module again from memory.
3. For each stage, answer this question: which metric moves, and why?
4. Rehearse a defense of 90 seconds for each stage.
5. **Check:** You can write `core/chunking.py` again with no reference.

> **NOTE:** Phase 8 is the purpose of the project. If you cannot write the module again, you do not know the module.

---

## 14.7 File Plan

| Path | Action | Quantity |
|---|---|---|
| `core/` | Make new | 10 modules |
| `cli.py` | Make new | 1 file |
| `ablation.py` | Make new | 1 file |
| `diagnose.py` | Make new | 1 file |
| `api/server.py` | Keep and connect to `core/` | 1 file |
| `frontend/src/App.jsx` | Keep and change | 1 file |
| `frontend/src/components/` | Delete 17 components | 3 files remain |
| `tests/` | Expand | About 120 tests |
| `docs/` | Make 4 documents | README, ARCHITECTURE, EVALS, INTERVIEW |

**Size target:** 16,000 lines become about 3,000 lines of Python and about 300 lines of JSX.

---

## 14.8 Items To Remove

| Item | Reason |
|---|---|
| Router, auth, rate limit, Docker, deploy | Flow-Q does this work |
| 17 UI components | Not a part of the pipeline |
| `26_multimodal_colpali.py` | The corpus has no charts. The module proves nothing. |
| `21_long_context_vs_rag.py` | Becomes one row of Table 14-2 |
| `09_token_budgeting.py` | Becomes a policy in `generate.py` |
| `14` and `16` (LangSmith, Langfuse) | Become one adapter, `core/tracing.py` |
| All `print()` statements | Moves to the CLI |
| `data/agent_memory.sqlite` | A run artifact. Do not commit it. |

---

## 14.9 Warnings

1. **Do not add full-stack work.** Flow-Q shows full-stack work. IncidentLens shows AI work.
2. **Do not add offline mode.** The pipeline always uses a real provider.
3. **Do not start Phase 5 before Phase 3.** The free tier has a daily limit. The cache protects the table.
4. **Do not hide a bad number.** A reported failure shows engineering skill. A false pass does not.
5. **The score of 1.000 is the primary problem.** A saturated metric cannot show an improvement.

---

## 14.10 Completion Checks

| Check | Criterion |
|---|---|
| C1 | Precision@4 is below 1.000. |
| C2 | Each of the 10 stages operates alone and gives a measurement. |
| C3 | A second query run makes no provider call. |
| C4 | The LangSmith dashboard shows all 10 stages in one trace. |
| C5 | `incidentlens ablate` fills all 6 rows of Table 14-2. |
| C6 | `incidentlens diagnose` names the failing stage. |
| C7 | The UI shows the pipeline in one page. |
| C8 | You can write each `core/` module again from memory. |
| C9 | You can defend each stage for 90 seconds with no notes. |

---

## 🔗 Next

→ [15_evals.md](./15_evals.md) — The measurement method and the ablation table.
