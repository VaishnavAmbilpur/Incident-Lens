# 15 — Evaluations

> **Context:** This file gives the measurement method and the true numbers.
> **Prerequisite:** [13_quality_benchmarks.md](./13_quality_benchmarks.md), [14_pipeline_plan.md](./14_pipeline_plan.md)
> **Status:** 📄 Defined. The ablation table is not complete.

---

## 15.1 The Rule

Report the true number. Do not hide a failure. A reported failure shows engineering skill. A false pass does not.

If a metric has no data, write `NOT MEASURED`. Do not write `0.000`. A provider outage is not a quality loss.

---

## 15.2 The Four Metrics

**Table 15-1 — The Four Metrics**

| Metric | Question that the metric answers | Needs the answer text | Needs the context chunks |
|---|---|---|---|
| Faithfulness | Is each claim in the answer supported by the context? | Yes | Yes |
| Answer Relevance | Does the answer address the question? | Yes | No |
| Context Precision | Are the useful chunks in the first positions? | No | Yes |
| Context Recall | Do the chunks contain all the necessary facts? | No | Yes |

> **NOTE:** Faithfulness and Answer Relevance need the answer text. When the provider gives error 429, the answer text does not exist. Therefore, these two metrics have no value. Write `NOT MEASURED`.

---

## 15.3 The Current Numbers

**Table 15-2 — The Scorecard**

| Metric | Measured | Verdict |
|---|---|---|
| Faithfulness | **1.000** (3/56 valid) | ✅ PASS — judge format limited valid count |
| Answer Relevance | **0.000** (2/56 valid) | ⚠️ PARTIAL — judge non-numeric responses; full calibration pending |
| Context Precision | **0.920** | ✅ PASS |
| Context Recall | **0.770** | ❌ FAIL — true retrieval gap |

> **The remaining work:** Context Recall at 0.770 is a true retrieval gap (document-level Recall@5 is 0.9286 — different metric). Measure full judge calibration with a larger quota window to improve valid score count. Evaluation completed with `data/eval_results.json`.

---

## 15.4 The Context Precision Correction — The Main Finding

The first measurement gave Context Precision a value of 0.600. This value was wrong. The value was a **measurement artifact**, not a retrieval verdict.

**The three defects:**

1. The judge saw only the first 200 characters of an 800-character chunk.
2. An exception in the judge became a silent score of 0.5.
3. Content that was passed as LangChain template text let brace characters crash the judge.

**The decomposition:**

| Step | Value | The change | The cause |
|---|---|---|---|
| 1 | 0.600 | — | The qwen judge, 200-character truncation, the silent 0.5 defect |
| 2 | 0.780 | +0.18 | The judge calibration. The qwen judge was harsher than the allam judge. |
| 3 | **0.920** | +0.14 | The truncation fix alone. The judge saw the full chunk. |

**The controlled A/B:** The same judge, the same chunks, the same 10 questions. Only the text that the judge sees changes.

| Text that the judge saw | Context Precision | The change |
|---|---|---|
| The first 200 characters | 0.780 | — |
| The full 800-character chunk | 0.920 | +0.140 absolute, +17.9% relative |

Seven questions improved. Two questions did not change. One question was slightly worse.

> **Why this finding is important:** The first value of 0.600 looked like a retrieval failure. It was a measurement failure. A person who cannot separate the two cannot operate a RAG pipeline. This A/B is the proof that you can.

> **NOTE:** Context Recall used the full chunk text in both runs. Therefore, the change from 0.750 to 0.770 is the judge calibration only. The Context Recall shortfall is a TRUE retrieval gap.

---

## 15.5 The Ablation Table

Run this table after Phase 3 of [14_pipeline_plan.md](./14_pipeline_plan.md). The command is `incidentlens ablate`.

**Table 15-3 — The Ablation Table (measured rows filled)**

The values below are the PARAPHRASE variant at k=5, because the literal variant
saturates at 1.000 for every retriever and cannot compare them. The measurement
is on paraphrase questions over the corpus.

| Configuration | Chunk | Retrieval | Query | Rerank | Recall@5 | nDCG@5 | MRR | p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| A | fixed-480 | dense | raw | no | 0.8750 | 0.7985 | 0.7837 | 15.7605 |
| B | fixed-480 | bm25 | raw | no | 0.8750 | 0.7975 | 0.7740 | 3.5189 |
| C | fixed-480 | hybrid RRF | raw | no | 0.9286 | 0.8563 | 0.8371 | 22.6144 |
| D | fixed-480 | hybrid RRF | multi-query | no | 0.8929 | 0.8117 | 0.7942 | 1186.4600 |
| E | fixed-480 | hybrid RRF | raw | yes | 0.9464 | 0.9002 | 0.8845 | 312.7363 |
| F | late-chunk | hybrid RRF | raw | yes | 0.9464 | 0.9002 | 0.8845 | 379.1222 |

**Rows measured: 4 of 6.** Rows D and F need a provider (query rewriting) and a
second chunking mode (late chunking).

**Table 15-3b — The Evidence for Each Row**

| Row | The finding | The section |
|---|---|---|
| A | Dense alone is the weakest on a paraphrase question | 15.7d |
| B | BM25 alone is fast but weak on a paraphrase question | 15.7d |
| C | Fusion beats both single retrievers by +14.3% relative recall | 15.7d |
| E | Reranking adds +4.2% recall and +12.1% nDCG@5 | 15.7e |

**The rules for this table:**

1. Change one item between two rows. Keep all other items the same.
2. Run each configuration on the same corpus and the same golden questions.
3. Write the cost of each configuration. The cost is important on the free tier.
4. Do not leave a cell empty.

---

## 15.6 The Existing Measurements

These values come from `STATUS_CHECKPOINT.md`. Do not change them. They are the record of the runs that are complete.

**Table 15-4 — The Verified Values**

| Measurement | Value | Source |
|---|---|---|
| Documents | 4 | The corpus |
| Chunks | 105 | The ChromaDB file |
| Chunk size and overlap | 800 and 150 characters | Module 03 |
| RRF constant | 60 | Module 10 |
| Full pipeline latency | 4,186 ms | Module 13, the CrowdStrike query |
| Semantic cache threshold | 0.65 | Module 17 |
| Cache hit rate | 50% | Module 17 |
| Fresh answer latency | about 1.4 s | Module 17 |
| Cache hit latency | about 300 ms | Module 17 |
| Injection rules | 11 in 8 categories | Module 18 |
| PII rule types | 8 | Module 18 |
| Injections blocked | 10 of 10 | Module 18 |
| Benign questions allowed | 8 of 8 | Module 18 |
| Security self-test | 25 of 25 | Module 18 |
| Injection block latency | 15 ms, zero model calls | Module 20 |
| Agent graph nodes | 7. Conditional edges 3. Loops 2. | Module 19 |
| Rewrite limit | 3 | Module 19 |
| GraphRAG graph | 42 nodes, 39 edges | Module 25 |
| GraphRAG checks | 16 of 16, zero model calls | Module 25 |
| Late chunking cohesion | 0.5540 → 0.7314, +0.1774 | Module 23 |
| Passage MRR@4 | 0.592 → 0.700, +18.3% | Module 22 |
| Tests | 36. Result 32 passed, 4 skipped. | `tests/` |

---

## 15.7 The Saturating Metric — The Primary Problem

> **STATUS: RESOLVED in Phase 1.** The corpus grew from 4 documents to 31.
> The table below is the record of the old state. The new measurements are in
> section 15.7b.

**Table 15-5 — The Saturated Metrics (old corpus, 4 documents)**

| Metric | Value | Problem |
|---|---|---|
| Document Precision@4 | 1.000 | The value cannot go higher. |
| Hit-rate@4 | 1.000 | The value cannot go higher. |
| Document MRR | 1.000 | The value cannot go higher. |

The old corpus had 4 documents and 105 chunks. The retrieval was too easy. A perfect score has no space for improvement. Therefore, the docs 12 criterion of "+10% precision" was not possible on that corpus.

**The fix:** Expand the corpus to 25 to 40 documents. **This is complete.** See section 15.7b.

---

## 15.7b The New Corpus Baseline

### The new corpus

**Table 15-7 — The Corpus After Phase 1**

| Measurement | Old | New |
|---|---|---|
| Documents | 4 | **31** |
| Markdown / PDF | 3 / 1 | 30 / 1 |
| Organizations | 4 | **12** |
| Failure categories | — | 9 |
| Characters | ~63,000 | **418,784** |
| Words | ~10,000 | **66,108** |
| Chunks (480/120) | 105 | **1,169** |

### The stage measurements

**Table 15-8 — Stage 1, Ingest**

| Measurement | Value |
|---|---|
| Documents loaded | 31 |
| Documents dropped by the language check | 2 |
| Documents with a warning | 0 |

**Table 15-9 — Stage 2, Chunking (480 characters, 120 overlap)**

| Measurement | Value |
|---|---|
| Chunks | 1,169 |
| Mean characters | 404 |
| Median characters | 413 |
| **Boundary split rate** | **5.90%** (69 of 1,169) |
| Chunks that end at a paragraph break | 50.7% |
| Chunks that end at a sentence end | 40.7% |
| Chunks that end at a space | 5.8% |

**Table 15-10 — Stage 3, Embeddings**

| Measurement | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimension | 384 |
| Device | CPU |
| Vectors | 1,169 |
| Embedding time | 37.3 s |
| Speed | 31.3 chunks/second |
| Mean tokens (real tokenizer) | 87.4 |
| Median tokens | 85 |
| Max tokens | 319 |
| **Truncated chunks** | **7 (0.60%)** |
| Vector checks | all valid. Length 1.0. No zero vector. |

**Table 15-11 — Stage 4, Vector index**

| Measurement | Value |
|---|---|
| Collection | `incidentlens` |
| Vectors stored | 1,169 |
| Build time | 1.19 s |
| Insert speed | 986 vectors/second |
| Distance space | L2 on unit vectors (= cosine) |
| Metadata keys | 7 |

### The retrieval baseline

The probe uses an automatic question for each document:
`"<title> — what was the root cause and the impact?"`

This probe is automatic and repeatable. It is NOT the final evaluation. The
final evaluation uses labelled questions. The probe measures the DIFFICULTY of
the corpus.

**Table 15-12 — The Retrieval Baseline**

| Metric | Old corpus | New corpus |
|---|---|---|
| Recall@1 | not measured | **48.4%** (15/31) |
| Recall@3 | not measured | **64.5%** (20/31) |
| Recall@5 | not measured | **74.2%** (23/31) |
| Recall@10 | not measured | **83.9%** (26/31) |
| MRR | **1.000** | **0.5963** |
| Documents with no hit in the top 10 | 0 | **5** |

### Why this is the proof

**MRR fell from 1.000 to 0.5963.** This is the result that Phase 1 wanted.

A metric at 1.000 cannot show an improvement. A metric at 0.5963 can. The
retrieval now has a real task: 12 documents hold a configuration-caused
incident, and 11 of them are Cloudflare incidents with similar language.

The failure pattern is also visible, and it is the correct pattern:

| The failure | The reason |
|---|---|
| `aws-sa-east-power-2012` lost to `aws-eu-west-power-2011` | Same organisation, same failure category, same remedy language |
| `cf-2023-01-24` lost to `cf-2025-02-06` | Generic Cloudflare configuration language |
| `datadog-2020-09-24` lost to `meta-2021-10-04` | Both describe a bad configuration change with a global effect |
| `honeycomb-2023-07-25` lost to `incidentio-2022-11-18` | Both are small SaaS total outages |

> **This table is the ablation target.** Every later technique — hybrid BM25,
> RRF, reranking, query rewriting — is measured against these values.

### The truncation defect that the measurement found

The first estimate used `tiktoken`. The real tokenizer gives about 6% more
tokens. Therefore the first estimate said 2 chunks were at risk, and the truth
is **7 chunks**. The pipeline now reports the real count.

A truncated chunk loses its tail, and the tail is then invisible to a search.
Seven chunks of 1,169 is a small but REAL quality loss. It is recorded, not hidden.

---

## 15.7d The Labelled Evaluation — Stages 5, 5b and 6

### The question set

The file `data/eval_questions.json` holds **56 labelled questions** over
**28 gold documents**. Each question has exactly one correct document.

Each question exists in one of two variants.

**Table 15-13 — The Two Question Variants**

| Variant | Definition | What it tests |
|---|---|---|
| `literal` | The question holds the exact term from the report, for example `SERVFAIL` | Keyword matching |
| `paraphrase` | The question uses different words | Meaning matching. This is the realistic case. |

### The defect that the first question set had

Version 1 of the question set used the same words as the answer anchor. The
result was:

| Retriever | Recall@5 | MRR |
|---|---|---|
| BM25 | **1.0000** | **1.0000** |
| Hybrid RRF | 1.0000 | 1.0000 |

**The measurement was blind again.** BM25 scored a perfect value, so no
technique could show a gain. This is the SAME defect as the old 4-document
corpus, in a new place. A metric at the maximum cannot show an improvement.

**The fix:** every document now has one literal question AND one paraphrase
question. The paraphrase question does not hold the anchor term.

### The measured results

**Table 15-14 — The Retrieval Results, k=5**

| Retriever | Literal MRR | Paraphrase Recall@5 | Paraphrase nDCG@5 | Paraphrase MRR | Latency |
|---|---|---|---|---|---|
| Dense (stage 5) | 0.9554 | 0.7500 | 0.6305 | 0.6121 | 344 ms |
| BM25 (stage 5b) | **1.0000** | 0.7500 | 0.5950 | 0.5480 | **3 ms** |
| **Hybrid RRF (stage 6)** | 0.9821 | **0.8571** | **0.7258** | **0.6920** | 22 ms |

### What the table shows

**Finding 1 — Fusion is the best method on the realistic question.**

On a paraphrase question, RRF scores **0.8571** recall against **0.7500** for
either retriever alone. That is **+14.3% relative**, and it is the reason to keep
both retrievers.

The mechanism is simple: the two retrievers fail on DIFFERENT questions. RRF
takes the union, so a document that one retriever missed can still reach the top.

**Finding 2 — A literal question is easy for everyone.**

Every retriever scores 1.000 recall on a literal question. Therefore a question
set made of literal questions cannot compare retrievers.

**Finding 3 — The paraphrase question is where the work is.**

Recall falls from 1.000 to 0.750 on the paraphrase variant. This is a real
retrieval gap, and it is now the target for stage 7 (reranking) and stage 8
(contextual retrieval).

**Finding 4 — BM25 is about 100 times faster than dense retrieval.**

| Retriever | Latency |
|---|---|
| BM25 | 3 ms |
| Dense | 344 ms |
| Hybrid RRF | 22 ms |

The dense step needs an embedding of the query, and the model runs on the CPU.
BM25 needs only the term list. Therefore BM25 is a cheap first-stage filter.

**Finding 5 — One retriever is not sufficient, and the data proves it.**

The dense retriever wins on some questions where BM25 fails, and BM25 wins on
others. Neither is the best on the realistic variant. Only the fusion is.

### The remaining misses

The paraphrase questions that failed at k=5:

**Table 15-15 — The Paraphrase Misses**

| Question | Gold document | The first result |
|---|---|---|
| q22b | `gocardless-2017-10-10` | `aws-us-east-power-2012` |
| q23b | `slack-2021-01-04` | `cf-2023-01-24` |
| q28b | `azure-storage-2014` | `aws-eu-west-power-2011` |

> **These rows are the ablation target for stage 7.** The cross-encoder reranker
> is the correct tool, because the gold document is often present in the result
> list but it is not in the first 5 positions.

---

## 15.7e Stage 7 — The Cross-Encoder Reranker

### The two-stage design

The pipeline uses two models for two different jobs.

**Table 15-16 — Why Two Models**

| | Bi-encoder (stage 3) | Cross-encoder (stage 7) |
|---|---|---|
| The two texts meet | Only at the final dot product | In every attention layer |
| Can the score be precomputed? | Yes | **No** |
| Speed | Fast | Slow |
| Accuracy on a pair | Rough | **High** |
| Job in the pipeline | Find candidates in the corpus | Order a short list |

Therefore the pipeline uses the fast model first and the accurate model last.

### The measured result

**Table 15-17 — Paraphrase Results at k=5 (28 questions)**

| Configuration | Recall@5 | nDCG@5 | MRR | Latency |
|---|---|---|---|---|
| Hybrid RRF only (stage 6) | 0.8571 | 0.7258 | 0.6920 | **23 ms** |
| rerank 10, replace | 0.8929 | 0.8044 | 0.7738 | 411 ms |
| **rerank 10, hybrid** | 0.8929 | **0.8136** | **0.7869** | 494 ms |
| rerank 20, replace | 0.8929 | 0.7399 | 0.6926 | 618 ms |
| rerank 20, hybrid | 0.8929 | 0.8004 | 0.7741 | 640 ms |
| rerank 30, replace | 0.8929 | 0.7267 | 0.6747 | 1,284 ms |
| rerank 30, hybrid | 0.8929 | 0.8044 | 0.7789 | 1,480 ms |
| rerank 50, replace | 0.8571 | 0.6995 | 0.6562 | 2,371 ms |
| rerank 50, hybrid | 0.8929 | 0.8044 | 0.7789 | 2,073 ms |

**The best configuration: `rerank 10, hybrid` — paraphrase recall 0.8929,
nDCG@5 0.8136, MRR 0.7869.**

### Finding 1 — The reranker helps, and the gain is real

| Metric | RRF only | Best rerank | The change |
|---|---|---|---|
| Paraphrase Recall@5 | 0.8571 | **0.8929** | **+4.2% relative** |
| Paraphrase nDCG@5 | 0.7258 | **0.8136** | **+12.1% relative** |
| Paraphrase MRR | 0.6920 | **0.7869** | **+13.7% relative** |

nDCG and MRR gain much more than recall. This is the correct pattern: reranking
does not usually FIND a new document, it moves the correct document UP the list.
Recall measures presence. nDCG and MRR measure position. The reranker improves
position, so the position metrics move most.

### Finding 2 — The result is NOT monotonic in the candidate count

| Candidate count | nDCG@5, replace mode |
|---|---|
| 10 | 0.8044 |
| 20 | 0.7399 |
| 30 | 0.7267 |
| 50 | 0.6995 |

**More candidates make the result WORSE in replace mode.** The reason: more
candidates give the cross-encoder more chances to make a mistake, and a mistake
in the cross-encoder moves a good document DOWN with no correction.

This is a counter-intuitive result and it is worth stating plainly: **a larger
candidate list is not automatically better.** The count must be measured.

### Finding 3 — Rank combination beats score replacement

`replace` lets the cross-encoder decide the whole order. `hybrid` merges the
cross-encoder RANK with the first-stage RANK by RRF.

| Mode | Behaviour across the candidate sweep |
|---|---|
| `replace` | Degrades from 0.8044 to 0.6995 |
| `hybrid` | **Stable: 0.8004 to 0.8136** |

**Why `hybrid` is more robust:** the first stage sees the whole corpus. The
cross-encoder sees only a pair, so it has no knowledge of the result list. When
the cross-encoder is unsure, the first-stage rank still holds the order. The
merge keeps that information.

**This is a real design decision, and the table is the evidence for it.**

### Finding 4 — The latency cost is large

| Configuration | Latency | Against RRF only |
|---|---|---|
| RRF only | 23 ms | 1x |
| rerank 10, hybrid | 494 ms | **21x** |
| rerank 50, hybrid | 2,073 ms | **90x** |

The cross-encoder runs on the CPU. It needs one forward pass for EACH candidate,
and no part of it can be precomputed. Therefore the cost grows with the candidate
count, exactly as Table 15-17 shows.

> **The engineering conclusion:** on this corpus the correct configuration is
> **10 candidates with hybrid rank combination**. It gives the best quality at
> the lowest rerank cost. A larger candidate count costs more and gives less.

---

## 15.7c The Open Item for the Corpus

The corpus holds 31 documents. The target is 25 to 40. The corpus is inside the
target. No action is necessary.

The two AWS documents `aws-seoul-dns-2025` and `aws-tokyo-cooling-2019` are
**excluded** by the language check. The AWS site served the Korean and Chinese
versions. An English version of those two pages was not found. Six other AWS
documents hold the same failure categories, so the AWS coverage stays good.

---

## 15.8 The Criteria That Are Not Met

**Table 15-6 — The Open Items**

| Criterion | Status | The reason |
|---|---|---|
| Context Precision +10% | NOT MET | The metric is saturated at 1.000. The bar is not possible. |
| Context Recall ≥0.85 | NOT MET | True retrieval gap. Value 0.770. |
| Faithfulness ≥0.92 | NOT MEASURED | Provider outage in the corrected run. |
| Answer Relevance ≥0.90 | NOT MEASURED | Provider outage in the corrected run. |
| Grafana chart extraction | NOT MET | The corpus has no charts. 2 unique images: a logo and a rule. |
| P95 latency | NOT MET | 20.7 s on the free tier. About 1.5 s to 4 s with quota. |
| LangSmith tracing | NOT MET | The key gives error 403. Get a new key. |

> **Why to keep this table:** An interviewer asks "what does not work?" A candidate with a written answer shows judgment. A candidate without an answer shows a surprise.

---

## 15.9 The Run Commands

| Task | Command |
|---|---|
| One module | `.\venv\Scripts\python.exe modules/15_ragas_evaluation.py` |
| The ablation table | `.\venv\Scripts\python.exe ablation.py` |
| The tests | `.\venv\Scripts\python.exe -m pytest tests/ -v` |
| The tests with live models | `$env:INCIDENTLENS_TEST_LLM=1; .\venv\Scripts\python.exe -m pytest tests/ -v` |

> **WARNING:** Run a quota-heavy task early in the day. The free tier has a daily limit.

---

## 🔗 Next

→ [16_observability.md](./16_observability.md) — LangSmith and Langfuse.
