# IncidentLens Production RAG - Phase 6 Session Log (Advanced 2026 Architectures)
## ASD STE 100 Technical Learning Document

**Document ID:** IL-STE-003
**Project:** IncidentLens - Production-Level RAG Pipeline
**Standard:** ASD Simplified Technical English (STE 100)
**Coverage:** Phase 6 (Modules 21, 22, 23, 24, 25 and 26)
**Last Updated:** 2026-10-13

---

## Table of Contents

1. [Purpose of This Document](#1-purpose-of-this-document)
2. [Phase 6 Overview](#2-phase-6-overview)
3. [Module 21 - Long Context vs. RAG Router](#3-module-21---long-context-vs-rag-router)
4. [Module 22 - Contextual Retrieval](#4-module-22---contextual-retrieval)
5. [Module 23 - Late Chunking](#5-module-23---late-chunking)
6. [Module 24 - Self-Correcting Agentic RAG](#6-module-24---self-correcting-agentic-rag)
7. [Module 25 - GraphRAG Multi-Hop Failure Chains](#7-module-25---graphrag-multi-hop-failure-chains)
8. [Module 26 - Multimodal Vision RAG](#8-module-26---multimodal-vision-rag)
9. [The Honest Limitations Register](#9-the-honest-limitations-register)
10. [Phase 6 Gate Results](#10-phase-6-gate-results)
11. [Module 15 Scorecard Correction (Closure)](#module-15-scorecard-correction-closure)
12. [Key Findings Summary](#11-key-findings-summary)
13. [Phase 6 Architecture Diagram](#12-phase-6-architecture-diagram)
14. [Run Commands](#13-run-commands)
15. [Glossary](#14-glossary)

---

## 1. Purpose of This Document

### 1.1 What This Document Covers

This document continues `03_session_log_phase5.md`.
That document stops at Module 20 and the pytest suite.
This document covers Phase 6: Modules 21 through 26, the six advanced 2026 RAG architectures.

### 1.2 The Rule of This Document

The six Phase 6 modules are unusual. Each module measures its own technique and prints the result.
Some results meet the `docs/12` criterion. Some do not.
This document reports both kinds with the same care.

> **Rule:** This document does not invent a number. Every number is either read from a module file or recorded from a real run.

### 1.3 What "Verified" Means in This Document

| Label | Meaning |
|---|---|
| **Verified (file)** | The fact is read directly from a file in the project. |
| **Verified (run)** | The fact is a recorded output of a real run. The module source defines the logic. The run record holds the number. |
| **Met** | The measured value reaches the stated criterion. |
| **Partially met** | Part of the criterion is proven. The remaining part is not measured, or not met. |
| **Not met** | The measured value does not reach the criterion. The value is reported as measured. |

### 1.4 The Central Lesson of Phase 6

A technique that sounds advanced is not automatically a technique that helps.
Phase 6 tested five such techniques on a small, real corpus.
Two of them moved a fine-grained metric and failed to move the coarse one.
One of them failed its own criterion because the criterion does not describe this corpus.
The modules print those outcomes. They do not round a number up to make a bar.

---

## 2. Phase 6 Overview

### 2.1 Why Phase 6 Exists

Phase 5 made the pipeline secure, stateful and callable over HTTP.
Phase 6 asks a different question: is the RETRIEVAL STRATEGY itself correct?

| Production question | Phase 6 module that answers it |
|---|---|
| Is retrieval always the right strategy? | Module 21 - Long Context vs. RAG Router |
| Does a chunk know which incident it belongs to? | Module 22 - Contextual Retrieval |
| Does a chunk know what the rest of the document says? | Module 23 - Late Chunking |
| What happens when the corpus cannot answer the question? | Module 24 - Agentic RAG |
| How does a failure travel from cause to impact? | Module 25 - GraphRAG |
| What does the pipeline do with an image? | Module 26 - Multimodal Vision RAG |

### 2.2 Phase 6 Deliverables

| File | Technique | Verification result | API requirement |
|---|---|---|---|
| `modules/21_long_context_vs_rag.py` | Long context vs. RAG router | Router 6/6, benchmark measured, 2 honest gaps | A provider (OpenRouter preferred, Groq fallback) |
| `modules/22_contextual_retrieval.py` | Contextual retrieval | Passage MRR +18.3%; the docs/12 10% bar NOT MET (saturated metric) | A provider (Groq free tier) |
| `modules/23_late_chunking.py` | Late chunking | Cohesion +0.1774, evidence rank +5.43 positions; all integrity checks pass | Fully local (CPU only) |
| `modules/24_agentic_rag.py` | Self-correcting agentic RAG | 3/3 scenarios PASS with machine-checked assertions | A provider (Groq free tier) plus optional Langfuse |
| `modules/25_graphrag_reasoning.py` | GraphRAG multi-hop chains | 16/16 checks PASS with ZERO LLM calls | Fully local (NetworkX) |
| `modules/26_multimodal_colpali.py` | Multimodal vision RAG | Pipeline VERIFIED on a synthetic chart; the Grafana criterion NOT MET on this corpus | A vision provider (OpenRouter free tier) |

### 2.3 The Phase 6 Chain

```
   [21] Router             narrow query -> RAG.  whole-document query -> LONG CONTEXT.
        |
        +--> [22] Contextual Retrieval    prepend a situation header before embedding
        +--> [23] Late Chunking           encode the parent document, then pool per chunk
        |
        v
   [24] Agentic RAG          grade -> rewrite -> synthesize -> ground check -> external fallback
        |
        v
   [25] GraphRAG             walk the proven causal chain, hop by hop, with a citation per hop
        |
        v
   [26] Vision RAG           describe the extracted images, index the descriptions, retrieve them
```

### 2.4 Verification Summary

| Module | Verification | Result |
|---|---|---|
| 21 | Router decision table | 6 of 6 queries routed to the expected strategy |
| 22 | Deterministic index comparison | Passage MRR@4 0.592 -> 0.700 (+18.3%); document metric saturated at 1.000 |
| 23 | Integrity and cohesion measurement | 105/105 spans resolved; macro adjacency +0.1774 |
| 24 | Three scenario assertion blocks | 3 of 3 PASS, 11 of 15 budgeted LLM calls |
| 25 | Sixteen deterministic checks | 16 of 16 PASS, zero LLM calls |
| 26 | Description, probe and retrieval checks | Chart reading proven on a synthetic chart; 0 real charts in the corpus |

---

## 3. Module 21 - Long Context vs. RAG Router

**File:** `modules/21_long_context_vs_rag.py`
**Command:** `.\venv\Scripts\python.exe modules\21_long_context_vs_rag.py`
**Optional command:** `.\venv\Scripts\python.exe modules\21_long_context_vs_rag.py --llm`

### 3.1 What We Did

We built a dynamic decision engine. The engine chooses one of two strategies for each query.

| Strategy | What it sees | Cost shape |
|---|---|---|
| **RAG** | The top-k chunks from hybrid BM25 + dense + RRF retrieval | About 1 to 2 thousand input tokens |
| **LONG CONTEXT** | The whole post-mortem text | About 40 to 50 thousand input tokens |

The default classifier is deterministic and explainable. It contains 12 weighted regular-expression signals.

| Signal group | Count | Signals |
|---|---|---|
| Favouring RAG | 6 | `exact_error_code` (2.5), `autonomous_system_number` (3.0), `specific_identifier` (2.0), `root_cause_ask` (2.5), `narrow_scope` (2.0), `latency_priority` (1.5) |
| Favouring LONG CONTEXT | 6 | `narrative_summary` (3.0), `timeline_reconstruction` (3.0), `whole_incident_scope` (2.5), `unfold_over_time` (2.0), `cross_section_compare` (2.5), `lessons_overall` (2.0) |

The router prints every fired signal with its weight, the matched span and a confidence.
A repeated signal counts for more: `weight * (1.0 + 0.25 * (hits - 1))`.
A margin below 1.0 falls back to RAG and is marked **contested**, because targeted retrieval is the cheap default.

### 3.2 Why

The two strategies fail in opposite directions.

| Failure | RAG | LONG CONTEXT |
|---|---|---|
| Loses the connective tissue of the incident | Yes. It only sees the retrieved chunks. | No. It sees every section. |
| Pays for text it does not need | No. | Yes. A lookup that lives in one paragraph pays for the whole document. |

A single fixed strategy must therefore be wrong for one half of the query population.
The router makes the choice per query, and the benchmark measures what the choice cost.

### 3.3 Router Decision Table

6 of 6 queries routed to the expected strategy.

| Query (short form) | Chosen | Expected | Fired signals | Confidence |
|---|---|---|---|---|
| Root cause of the Cloudflare BGP route leak | RAG | RAG | `root_cause_ask` w=2.50 | 0.754 |
| Reconstruct the full timeline across the whole incident | LONG_CONTEXT | LONG_CONTEXT | `narrative_summary` w=3.00 + `timeline_reconstruction` w=3.75 + `whole_incident_scope` w=2.50, score 9.25 vs 0.00 | 0.996 |
| Exact bugcheck code `0x00000050` | RAG | RAG | `exact_error_code` w=2.50 | printed with the run |

> **Note:** The router gives confidence as a blend of signal mass and margin.
> The rule is verified in the file. The three confidence values above are recorded run outputs.

### 3.4 Benchmark (Final Run)

| Measurement | RAG | LONG CONTEXT | Ratio |
|---|---|---|---|
| Input tokens | 887 | 1,830 | 2.1x |
| Latency | 19,483 ms | 19,347 ms | about equal |
| Quality (judge rubric) | 1.00 | 0.75 | RAG higher |

The quality rubric has two axes. `groundedness` says every claim is traceable to the supplied source. `relevance` says the answer addresses the question completely. `quality` is the mean of the two.

### 3.5 Model Discovery

The module probed and rejected two OpenRouter models before it cached a working one.

| Model | Result |
|---|---|
| `thinkingmachines/inkling*:free` | HTTP 403, "agentic-harness-only" |
| `nvidia/nemotron-3.5-lightning:free` | Empty completion |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | Accepted. 1,000,000-token window. |

The module names the provider that actually served each request. It does not mislabel a fallback as the preferred provider.

### 3.6 Key Result

The router chose the expected strategy for all 6 queries in the decision table.
The benchmark produced real latency, token, cost and quality numbers for both strategies.

### 3.7 Honest Limitations

All four limitations are stated by the module itself.

| # | Limitation | The measured fact |
|---|---|---|
| 1 | The 25 to 30x whole-document token multiple was NOT measured | The largest post-mortem is only 29,402 characters (about 7,432 tokens). Long context cost only 2.1x here. The 25 to 30x figure is the ratio for the 200-page reports in `docs/12`, not for this corpus. |
| 2 | The 0.75 long-context quality is an artefact of a truncation | The answer was cut off by the 900-token output budget in the middle of the timeline. The judge therefore scored relevance as partial. |
| 3 | The reference-cost column is an estimate | It uses a documented rate of 2.50 USD / 1M input tokens and 10.00 USD / 1M output tokens. It is not an invoice. |
| 4 | The final run was served by the fallback provider | OpenRouter's free tier hit its DAILY quota mid-session (HTTP 429 `free-models-per-day`). The final run was served by Groq. The module printed the provider it actually used. |

**Failure policy:** if no provider answers, the module prints `LONG CONTEXT UNAVAILABLE` and fails those checks. It never fabricates a number.

---

## 4. Module 22 - Contextual Retrieval

**File:** `modules/22_contextual_retrieval.py`
**Command:** `.\venv\Scripts\python.exe modules\22_contextual_retrieval.py`
**Optional commands:** add `--full`, `--llm` or `--build-only`

### 4.1 What We Did

We prepended a short situation header to each chunk BEFORE embedding.
The header is about 50 words. It makes orphaned pronouns self-contained.

```
BEFORE: "The thread count exceeded the OS limit, causing front-end
         servers to reject new connections."

AFTER : "[CONTEXT: AWS US-East-1 Kinesis Outage ... | <segment summary>]
         The thread count exceeded the OS limit, causing front-end
         servers to reject new connections."
```

The header goes into the EMBEDDED TEXT only.
Chroma keeps the untouched chunk text in metadata as `raw_text`, so the evidence pipeline still sees the original words.

### 4.2 Why the Default Is Hierarchical, Not Per-Chunk

The textbook version writes a bespoke header for all 105 chunks. That is 105 document-sized LLM calls.
The Groq free tier is rate-limited per model and per minute, so that is a rate-limit wall, not a runnable default.

| Mode | Generation targets | Provider calls on a cold cache |
|---|---|---|
| Default (hierarchical) | 4 document-level + 15 segment-level = 19 | 53 provider calls, 18 of them HTTP 429 |
| `--full` (per-chunk) | 105 per-chunk plus 19 hierarchy = 124 | 357 provider calls, 118 of them HTTP 429 |

The default is an APPROXIMATION, and the module labels it as one.
A segment blurb is shared by every chunk on that page, so it cannot capture what is unique to one chunk.

### 4.3 The Cache Is the Point of the Design

Every header is cached to `data/contextual_headers_cache.json`, keyed by a content hash.
The cache file keeps its own cross-run history.

| Invocation | LLM calls |
|---|---|
| 1 | 53 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 0 |
| 6 | 0 |
| 7 | 0 |

The first invocation spent 53 calls. Every invocation after it spent ZERO.
The module asserts a zero-call second run, and the run printed "asserted zero".

> **Note:** The history row records provider CALLS, not generation targets.
> The 53 calls cover 19 targets, because a 429 on one model fails over to the next model in the chain and each attempt is a real provider call.

### 4.4 The Two Indexes

| Property | Baseline | Contextual |
|---|---|---|
| Collection name | `incidentlens_baseline` | `incidentlens_contextual` |
| Chunks | 105 | 105 |
| Embedded text | 62,927 characters | 128,435 characters (+104.1%) |

Both indexes live in a PRIVATE Chroma directory, `data/chroma_db_contextual`.
Module 05's collection is therefore untouched.

### 4.5 Deterministic Verification

The check needs no judge model. Each golden question carries a `source_document` field.
A retrieved chunk is relevant when its `incident_slug` matches that document.

| Metric | Baseline | Contextual | Change |
|---|---|---|---|
| Document Precision@4 | 1.000 | 1.000 | none |
| Hit-rate@4 | 1.000 | 1.000 | none |
| Document MRR | 1.000 | 1.000 | none |
| Passage-level MRR@4 | 0.592 | 0.700 | **+18.3% relative** |

### 4.6 The Target Is Not Met, and the Module Proves Why

`docs/12` asks for a relative improvement of 10% or more in context precision.
The target is **NOT MET**, and the reason is arithmetic, not effort.

| Fact | Consequence |
|---|---|
| The baseline already scores Precision@4 = 1.000 and hit-rate@4 = 1.000 on ALL 10 golden questions | The document-level metric is SATURATED. A +10% gain is arithmetically unreachable on this dataset. |
| Only a finer metric can move | The finer metric did move: passage-level MRR@4 improved by 18.3%. |

The module reports the likely reasons from measurement, not from a guess.

| Reason | Detail |
|---|---|
| The corpus is small | Only 4 incidents cover 105 chunks. A plain embedder can already put 4 of 4 correct documents on top. |
| The vocabulary is disjoint | Each post-mortem is its own vocabulary domain. |
| The relevance definition is coarse | Document-level relevance only needs the right incident. It does not need the right passage. |
| The approximation is weakest where it matters | Two chunks on one page receive identical header text, because the default headers are shared per segment. |

### 4.7 The Orphaned-Pronoun Probes Regressed on the Labelled Block

The module reports two separate blocks and does not merge them.

| Block | Hit-rate@4 | MRR | Direction |
|---|---|---|---|
| Labelled probes (examiner-supplied labels) | 0.667 -> 0.333 | 0.444 -> 0.333 | REGRESSION |

The module states plainly that one probe is genuinely ambiguous, and that the labels were the examiner's own reading of sentences written for this test.
Block 1, the label-free block, is therefore the stronger evidence.

### 4.8 Key Result

The technique produced a real, measurable gain on the finest deterministic metric available on this corpus: passage-level MRR@4 rose 0.592 -> 0.700 (+18.3% relative).
The `docs/12` document-level target is NOT MET, and the module shows why the target cannot be met here.
The cache made the cost of the technique zero after the first invocation.

### 4.9 Honest Limitations

| # | Limitation | Detail |
|---|---|---|
| 1 | The 10% target is unreachable on this golden set | The baseline metric is saturated at 1.000. The module prints NOT MET with the arithmetic. |
| 2 | The default headers are an approximation | Per-segment sharing means two chunks on one page get the same header text. |
| 3 | The optional LLM judge puts the contextual variant BELOW the baseline | Measured 0.790 for the contextual variant against 0.920 for the baseline over the same 10 questions. The module reports both from the same run. |
| 4 | The passage proxy moved in both directions | +18.3% under the hierarchical headers, but -14.1% under the per-chunk `--full` headers. |
| 5 | The labelled probe block regressed | Hit-rate@4 0.667 -> 0.333. The label-free block is the stronger evidence. |
| 6 | A header is not free quality | The header is about 40 to 60 words prepended to a chunk of about 150 words. On a short chunk the header can dominate the pooled embedding, and all-MiniLM has a short effective window. |

---

## 5. Module 23 - Late Chunking

**File:** `modules/23_late_chunking.py`
**Command:** `.\venv\Scripts\python.exe modules\23_late_chunking.py`
**API requirement:** none. This module is 100% local and runs on the CPU.

### 5.1 What We Did

We inverted the order of operations.

```
STANDARD (early) chunking:
   document -> split into chunks -> embed each chunk ON ITS OWN

LATE chunking:
   document -> encode the FULL document (tokens attend to each other)
            -> pool the token vectors that belong to each chunk -> per-chunk vectors
```

The chunk boundaries are still decided before embedding.
The attention is computed over a much larger span, so a token inside a chunk has already seen the rest of its window.

### 5.2 Scale

| Property | Value |
|---|---|
| Chunks | 105 |
| Parent documents | 15 |
| Characters reconstructed | 63,017 |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| `max_seq_length` | 256 tokens |
| Embedding dimension | 384 |
| Device | CPU |
| Forward-pass windows | 66 across the 15 documents |
| Windows for the GitHub post-mortem alone | 13 |

### 5.3 Integrity Checks All Pass

The module refuses to trust its own plumbing. It checks the pooling against the chunk text and the window chain against the document.

| Check | Result |
|---|---|
| Chunk spans resolved to token ranges | 105 / 105 |
| Chunks that pooled text outside their own span | 0 |
| Windows with unusable token/offset alignment | 0 |
| Character-coverage gaps across the 15 documents | 0 |

### 5.4 Cross-Boundary Cohesion: The Fingerprint of the Technique

| Measurement | Standard | Late | Change |
|---|---|---|---|
| Macro adjacent-chunk cosine | 0.5540 | 0.7314 | **+0.1774** |
| Document groups that improved | - | 14 of 15 | - |
| Chunk-to-document-centroid affinity | 0.7346 | 0.8005 | **+0.0658** |

Adjacent chunks from the same document became more similar to each other.
That is the expected signature of pooling tokens that have attended to a shared window.

### 5.5 Temporal-Link Probes

Each probe is an early timestamped statement whose consequence sits in a much later chunk.

| Measurement | Value |
|---|---|
| Probes | 14 |
| Mean evidence rank | 30.50 -> 25.07 (**+5.43 positions**) |
| Probes that improved | 8 |
| Probes that worsened | 6 |
| Probes tied | 0 |
| Best single gain | +22 positions |
| Linked pairs that came closer | 12 of 14 |

A smaller evidence rank is better, because the linked evidence chunk is found sooner.

### 5.6 Key Result

Late chunking passed every integrity check and improved both cohesion measurements.
The retrieval benefit is real but mixed, and the module reports both directions.

### 5.7 Honest Limitations

| # | Limitation | Detail |
|---|---|---|
| 1 | "Long-range attention" is attention INSIDE a window | The 256-token window means a full post-mortem cannot be encoded in one forward pass. Distant pairs such as chunk 1 and chunk 30 still do not attend to each other. No token ever attends across a window seam. |
| 2 | Standard chunking silently truncated 7 chunks | 7 chunks exceed 256 tokens and were embedded from a truncated text. The window chain covers the whole document, so late chunking has an advantage here. |
| 3 | MiniLM is used off-label | The model was pre-trained for sentence-level mean pooling. Late chunking pools a token sub-span out of a larger attention window. |
| 4 | The retrieval effect is mixed | Rank improved on 8 of 14 probes and worsened on 6. Both directions are reported. |

### 5.8 Engineering Fix Worth Recording

`HF_HOME` must be set BEFORE `sentence-transformers` is imported.
`huggingface_hub` resolves its cache directories at import time, so changing the variable afterwards does not move the cache.

The module also verifies the cache by an actual WRITE. It creates a file and deletes it.
A directory-existence test is not enough, because a cache directory can exist and still be unwritable.
On such a host the model download dies with `PermissionError [WinError 5]`. The module then retries once against `<project>/.hf_cache`.

---

## 6. Module 24 - Self-Correcting Agentic RAG

**File:** `modules/24_agentic_rag.py`
**Command:** `.\venv\Scripts\python.exe modules\24_agentic_rag.py`

### 6.1 What We Did

We extended Module 19's graph. Module 19 already had two bounded self-correction loops.
That left one hole: the corpus is wrong for this question, and the agent has no move left.

| Addition | Kind | Purpose |
|---|---|---|
| `external_lookup` | Node | Leave the corpus through a pluggable search interface |
| `finalize_end` | Node | Stamp honest provenance on every delivery path |
| `route_after_grading_extended` | Router | Relevance LOW after `MAX_REWRITES = 3` -> external |
| `route_after_hallucination_extended` | Router | Still ungrounded after `MAX_SYNTHESIS = 2` -> external |

### 6.2 The Loop Bounds

| Bound | Value | Effect |
|---|---|---|
| `MAX_REWRITES` | 3 | Query rewrites before the external path |
| `MAX_SYNTHESIS` | 2 | Synthesis attempts before the external path |
| `MAX_EXTERNAL_TRIES` | 1 | The external node runs once |
| `MAX_TOTAL_HOPS` | 12 | Hard ceiling on node visits. Both routers enforce it. |

`external_lookup` flows unconditionally to `sanitize`.
The fallback is therefore a SINK. It cannot cycle, by construction.
The hop count is printed for every turn, so an unbounded run would be visible.

### 6.3 What `verified == True` Means

The system gives `verified == True` exactly one meaning.

```
verified == True   <=>   answer_source is "internal"
                    AND  the grounding check reached 0.80
                    AND  the answer carries at least one [chunk_id] citation
```

Every other delivery path receives a visible label.

| Path | Label |
|---|---|
| External answer | `[EXTERNAL / UNVERIFIED]` |
| Internal answer that is not checked or not grounded | `[INTERNAL / NOT VERIFIED]` plus the reason |

An ungrounded answer can therefore never be mistaken for a corpus-backed answer.

### 6.4 Three Scenarios, Machine-Checked

An LLM call counter wraps `llm_text`. The phrase "zero LLM calls" is therefore MEASURED, not claimed.

| Scenario | Input | Observed path | Result |
|---|---|---|---|
| A, in-corpus | A question the corpus answers | `grade_relevance` 0.80 -> grounded 0.80 -> `finalize_end(verified:internal)` | PASS |
| B, out-of-corpus | "Who is the current CEO of CrowdStrike?" | See 6.5 | PASS |
| C, injection | A prompt-injection payload | Blocked by the security gate with ZERO LLM calls. `retrieve` and `external_lookup` never ran. | PASS |

Total spend: 11 of 15 budgeted LLM calls.

### 6.5 Scenario B in Detail

The corpus contains the word "CrowdStrike". It does not contain the answer.

| Step | Observation |
|---|---|
| Hybrid retrieval | RRF score 0.0328. The retrieved chunks contained no answer. |
| Relevance grading | The grader refused to be fooled by the high lexical overlap. Relevance stayed LOW. |
| Rewrites | Three rewrites, scores 0.20, 0.00, 0.20, 0.30. All stayed below threshold. |
| External path | `external_lookup` ran and called `web_search()`. No endpoint was configured. |
| Delivered answer | Headed `[EXTERNAL / UNVERIFIED]` with `verified=FALSE`. |

The delivered text was:

```
The current CEO of CrowdStrike is George Kurtz. This information is
based on my model knowledge and is unverified.
```

### 6.6 The Web-Search Seam Is Pluggable

The node calls one function:

```python
web_search(query) -> Optional[List[str]]
```

| Condition | Behaviour |
|---|---|
| `INCIDENTLENS_SEARCH_ENDPOINT` is set | One HTTP POST is made to that endpoint |
| The variable is absent | The function returns `None` |
| `None` returned | The node falls back to a labelled, context-free model answer |

The docstring shows where Tavily, Bing and DuckDuckGo adapters plug in.
The module never fakes a search.

### 6.7 Langfuse Tracing Verified

| Check | Result |
|---|---|
| `auth_check` | Passed |
| Session scope | A FRESH `propagate_attributes` session per query |
| Client flush | Called |
| Root observations | Read back through the API |

The `propagate_attributes` context manager is SINGLE-USE. Reusing one object across queries raises an `AttributeError`. The module builds a new scope for every query.

### 6.8 Three Self-Inflicted Bugs Found and Fixed

These bugs were introduced during testing and caught by the checks.

| # | Bug | Consequence | Fix |
|---|---|---|---|
| 1 | A self-recursive default search provider | `RecursionError` | The default provider is registered through a lambda |
| 2 | A placeholder reason string leaked into the user-visible fallback | The user read an internal note | The reason is built from the state, not from the placeholder |
| 3 | Corpus text leaked into the external prompt | The unverified label would have lost its meaning, because the answer would have been corpus-backed | The external prompt carries no internal context |

Bug 3 is the important one. A label is only honest when the labelled thing is what the label says.

### 6.9 The Grounding Judge Had to Be Strengthened

| Problem | Measured detail |
|---|---|
| The cheap judge under-scores a good answer | It scores a correctly cited answer 0.5 |
| The cheap judge returns an empty body | At a 12-token cap the reasoning tokens eat the budget and the body is empty |
| Consequence | Scenario A would have failed, and good internal answers would have been pushed down the external path |

Module 24 applies three fixes.

| Fix | Detail |
|---|---|
| Larger judge budget | 300 tokens, the empirically verified floor |
| Honest refusal is accepted | An explicit "the context does not contain the answer" is treated as an honest refusal, not as a grounding failure |
| Second opinion | When the cheap judge scores below threshold, a stronger model is asked |

This finding was back-ported to Module 19 as `llm_score()`.
That function escalates the token budget only when the cheap attempt yields no parseable number.

### 6.10 Key Result

All three scenarios passed with machine-checked assertions.
The agent self-corrected on every low-relevance case that it was tested on: three rewrites, then an honest external fallback with a visible label.
The security gate blocked the injection with zero LLM calls.

---

## 7. Module 25 - GraphRAG Multi-Hop Failure Chains

**File:** `modules/25_graphrag_reasoning.py`
**Command:** `.\venv\Scripts\python.exe modules\25_graphrag_reasoning.py`
**Optional command:** `.\venv\Scripts\python.exe modules\25_graphrag_reasoning.py --llm`
**API requirement:** none by default. See 7.6.

### 7.1 What We Did

Vector RAG is a one-hop machine. Similarity is not causality.
We extracted the CAUSAL SKELETON of each incident into an explicitly directed graph, then answered multi-hop questions by TRAVERSING the graph.

| Property | Value |
|---|---|
| Nodes | 42 |
| Edges | 39 |
| Weakly connected components | 4, one per incident |
| Component sizes | Kinesis 9, Cloudflare 7, GitHub 10, CrowdStrike 13 |
| Graph type | Verified as a DAG (directed acyclic graph) |
| Library | NetworkX, in memory |

### 7.2 The Grounding Rule

An extracted edge is a CLAIM, and every claim must be provable.
Each edge is authored with an `evidence_chunk_id` and an `evidence_text`.
At build time the module proves, in code and with no LLM, two things.

1. The cited chunk id really exists in the corpus.
2. The `evidence_text` really is a substring of that chunk's text.

The gate is live, not decorative. A deliberately poisoned claim list was fed to the builder.

| Poisoned claim | Rejection reason |
|---|---|
| A fabricated quote | Substring mismatch |
| A real chunk id with the wrong content | Content mismatch |
| An unknown chunk id | Unknown chunk id |
| A PDF-line-wrapped quote | Matched only after whitespace folding |

**4 of 43 edges were rejected**, and nothing leaked into the graph.
The run prints each rejected edge with its exact reason.

### 7.3 Honest Drops, Not Invented Hops

| Case | Decision | Why |
|---|---|---|
| The "AWS IoT impact" node | Dropped | The only supporting sentence is a list that already names CloudWatch, Cognito and Lambda |
| One direct edge | Replaced by the two-hop chain the corpus actually proves | `out-of-bounds read` -> `TemplateGetString supplies an invalid pointer` -> `csagent.sys dereferences` |

### 7.4 Six Multi-Hop Queries

6 queries, all PASS, 20 hops total. Each hop is printed with an exact-substring citation.

The 3-hop Cloudflare case:

| Hop | Claim | Citation |
|---|---|---|
| 1 | Noction BGP Optimizer splits prefixes into more-specifics | `cloudflare_bgp_leak_p1_c3` |
| 2 | Verizon (AS701) accepted and re-announced the leaked routes | `cloudflare_bgp_leak_p1_c7` |
| 3 | Internet-wide service degradation (Cloudflare, Amazon, Linode) | `cloudflare_bgp_leak_p1_c8` |
| - | Cloudflare traffic loss (about 15% of global traffic) | printed with the chain |

A 5-deep Kinesis chain reaches Lambda, the Console, CloudWatch and Cognito.

The negative case returns `NOT ANSWERABLE FROM THE GRAPH` for a DNS TTL question.
A corpus-wide scan confirms 0 hits for that phrasing, so the graph cannot guess.

### 7.5 Single-Point-of-Failure Ranking

The ranking uses the count of downstream nodes that STOP BEING REACHABLE when a node is removed.
Actual node removal is measured, not estimated.

| Measurement | Value |
|---|---|
| Worst-case disconnection | 10 downstream nodes |
| Worst-case nodes | `100G optical maintenance`; the 21-input IPC template definition |

The module prints its own caveat.
A large downstream count with parallel paths is a FAN-OUT point, not a CHOKE point, because the traffic of causation still gets through.
Only the disconnection column measures irreplaceability.

### 7.6 Sixteen Checks, Zero LLM Calls

**16 of 16 checks PASS** with ZERO LLM calls by default.

`--llm` alone does not spend tokens. It additionally requires `INCIDENTLENS_ALLOW_LLM=1` and it rejects dummy keys.
That second opt-in exists because an earlier `--llm` test DID reach the provider through a `.env` copy.

### 7.7 Two Self-Inflicted Bugs Found and Fixed

| # | Bug | Consequence | Fix |
|---|---|---|---|
| 1 | BFS visit order was treated as a path | A chain citation could point at an edge that no path used | Chain citations now come from the BFS spanning tree |
| 2 | A branching chain reported node count instead of hop depth | It had said "9 hops" for a 5-deep tree | The module reports the deepest hop depth |

### 7.8 Key Result

Every hop in every answer carries a citation that the module proved against the corpus text.
The graph refuses to answer a question it cannot ground, and the negative case is a first-class test.

---

## 8. Module 26 - Multimodal Vision RAG

**File:** `modules/26_multimodal_colpali.py`
**Command:** `.\venv\Scripts\python.exe modules\26_multimodal_colpali.py`
**API requirement:** a vision provider (OpenRouter free tier). The module does not use `GROQ_API_KEY`, because Groq hosts no vision model on the free tier.

### 8.1 What We Did

Every earlier module reads a post-mortem as TEXT. A real incident report also carries pictures.
We built the "description" flavour of multimodal RAG.

```
   1. EXTRACT    pull every embedded image out of the PDF, hash it, deduplicate by hash
   2. RENDER     optional. Full-page rendering. NOT INSTALLED HERE.
   3. DESCRIBE   a vision model describes each unique real image
   4. CHART PROBE   draw a synthetic chart, send it, check the reply against what was drawn
   5. INDEX      descriptions go into the Chroma collection incidentlens_vision
   6. RETRIEVE   a chart-shaped query must return the chart entry
   7. CHECKS     PASS/FAIL summary
```

### 8.2 The Reality of This Corpus

This fact must be stated plainly, because the criterion depends on it.

| Property | Value |
|---|---|
| Embedded-image slots in the CrowdStrike PDF | 24 (2 per page across 12 pages) |
| UNIQUE images | **2** |
| Image 1 | A 446x80 CrowdStrike wordmark LOGO |
| Image 2 | A 949x4 decorative rule |
| Grafana charts | 0 |
| Topology diagrams | 0 |
| Flamegraphs | 0 |
| Heatmaps | 0 |

The 24 slots are 2 images repeated 12 times.
The vision model independently classified the logo as `KIND: logo / TEXT: CROWDSTRIKE / CHART_READING: NONE`. It is not data-bearing.

### 8.3 Page Rendering Is Unavailable

The module needs `pymupdf` (`fitz`) or `pdf2image` to render full pages, so that VECTOR graphics become visible.
Neither package is installed. The module prints that fact and installs nothing.

### 8.4 The Synthetic Chart Probe

Because the real images carry no data, the module drew its OWN synthetic bar chart with PIL.
It sent the chart to the vision model as a probe of whether visual reading really works.

| Drawn label | Drawn value | Read back |
|---|---|---|
| api-gateway | 42 | exact match |
| checkout | 78 | exact match |
| payments | 95 | exact match |
| search | 23 | exact match |

The reply matched all eight drawn strings exactly.
That proves real visual reading, not a plausible-sounding guess.
Six vision calls in total.

### 8.5 Retrieval Probe

| Query shape | Top result | Distance |
|---|---|---|
| Chart-shaped query 1 | The synthetic chart entry | 0.3301 |
| Chart-shaped query 2 | The synthetic chart entry | 0.5033 |

The logo and the rule ranked second and third.
Visual evidence is therefore reachable by search.

### 8.6 The Criterion Is Not Met on This Corpus

The `docs/12` criterion is: "Multimodal RAG extracts actionable information from Grafana chart images".
The criterion is **NOT MET** on this corpus, and the module says so in its own docstring and its run summary.

A synthetic probe proves the plumbing.
It does not prove that this corpus carries visual incident data.

### 8.7 Model Discovery and an API Quirk Worth Recording

| Observation | Detail |
|---|---|
| Preferred model | `google/gemma-4-26b-a4b-it:free` returned HTTP 429 three times |
| Failover model | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` |
| API quirk 1 | The model intermittently returns an HTTP 200 body with NO `choices` field. The SDK then raises `'NoneType' object is not subscriptable`. |
| API quirk 2 | The model sometimes places the answer only in a `reasoning` field, while `content` is empty |

The module retries transient or empty bodies and falls back to the reasoning text.
Without that handling the model looks broken on first contact.

### 8.8 Key Result

The vision pipeline was proven end to end on a synthetic chart: extraction, description, indexing and retrieval all worked.
The `docs/12` Grafana criterion remains NOT MET, because the corpus contains no charts.

---

## 9. The Honest Limitations Register

Phase 6 produced more negative findings than any earlier phase. They are collected here so that no reader meets them by surprise.

| # | Module | Limitation | Kind |
|---|---|---|---|
| 1 | 21 | The 25 to 30x token multiple was not measured. This corpus gave 2.1x. | Criterion measured on the wrong scale |
| 2 | 21 | The long-context quality of 0.75 is a truncation artefact at the 900-token output cap | Measurement artefact |
| 3 | 21 | The reference-cost column is an estimate at a published rate, not an invoice | Estimate |
| 4 | 21 | OpenRouter's free tier hit its daily quota, so Groq served the final run | External limit |
| 5 | 22 | The `docs/12` 10% context-precision bar is NOT MET, because the baseline is saturated at 1.000 | Criterion unreachable on this dataset |
| 6 | 22 | The default headers are a per-segment approximation | Approximation |
| 7 | 22 | The optional LLM judge scored the contextual variant BELOW the baseline | Negative result |
| 8 | 23 | "Long-range attention" is attention inside a 256-token window | Model limit |
| 9 | 23 | MiniLM is used off-label for sub-span pooling | Off-label use |
| 10 | 23 | Evidence rank improved on 8 of 14 probes and worsened on 6 | Mixed result |
| 11 | 24 | The 80% self-correction figure was not measured on a query set large enough to support a percentage | Criterion not measured at scale |
| 12 | 25 | A downstream count with parallel paths is a fan-out point, not a choke point | Definitional care |
| 13 | 26 | The Grafana chart criterion is NOT MET, because the corpus contains no charts | Criterion not applicable to this corpus |
| 14 | 26 | The demonstrated chart reading is on a SYNTHETIC chart | Synthetic evidence |
| 15 | All | The Groq and OpenRouter free tiers rate-limit by model, by minute and by day | External limit |
| 16 | 15 | The recorded Context Precision of 0.600 was a MEASUREMENT ARTIFACT: a 200-character chunk preview, a silent 0.5 score on a judge exception, and brace-unsafe judge prompts. The corrected measurement is 0.920 and it PASSES. | Measurement defect, now fixed |
| 17 | 15 | Context Recall is still below target at 0.770 (0.750 pre-fix). Its context was never truncated, so the shortfall is REAL retrieval quality, confirmed under two judges. | Criterion not met |
| 18 | 15 | Faithfulness and Answer Relevancy have NO corrected measurement (0.950 each pre-fix). The provider was out of quota, so 10 of 10 answers failed to generate. | Measurement pending a quota window |
| 19 | 15 | The run substituted `[LLM Error: 429 ...]` as the answer, and the judges scored those error strings 0.00. That 0.00 was the CORRECT verdict on an error message, but printing it as a quality collapse (F=0.000 / R=0.000 / FAIL) mislabelled a provider outage as a quality regression. | Reporting defect, now fixed |

---

## 10. Phase 6 Gate Results

**Source of the criteria:** `docs/12_phase6_advanced_rag.md`, section 6.3.

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Long context router correctly selects strategy based on query type | **MET** | 6 of 6 queries routed to the expected strategy |
| 2 | Contextual retrieval improves context precision by >= 10% over the Phase 3 baseline | **NOT MET** | Baseline Precision@4 and hit-rate@4 are both 1.000 on all 10 questions. The metric is saturated, so +10% is arithmetically unreachable. Passage-level MRR@4 did improve: 0.592 -> 0.700 (+18.3%). |
| 3 | Late chunking preserves temporal relationships across chunk boundaries | **MET** | Cohesion +0.1774; mean evidence rank +5.43 positions. Caveat: the 256-token window bounds the effect. |
| 4 | Agentic RAG self-corrects and improves answer quality on >= 80% of low-relevance queries | **PARTIALLY MET** | The agent self-corrected on every tested low-relevance case: three rewrites, then an honest external fallback. The 80% figure was NOT measured on a query set large enough to support a percentage, so it must not be claimed. |
| 5 | GraphRAG answers 3-hop cascading failure questions correctly | **MET** | The Cloudflare 3-hop chain, plus a 5-deep Kinesis chain, all with per-hop citations |
| 6 | Multimodal RAG extracts actionable information from Grafana chart images | **NOT MET** | No charts exist in the corpus. The vision pipeline was proven on a synthetic chart instead. |
| 7 | Final Ragas scorecard meets target benchmarks | **PARTIALLY MET** | See the scorecard subsections 10.1 and 10.3. Context Precision PASSES at 0.920 after the measurement fix. Context Recall FAILS at 0.770. Faithfulness and Answer Relevancy are NOT MEASURED. |

### 10.1 The Scorecard Criterion in Detail

| Metric | Measured | Target | Result |
|---|---|---|---|
| Faithfulness | NOT MEASURED (pre-fix 0.950) | >= 0.92 | n/a |
| Answer Relevancy | NOT MEASURED (pre-fix 0.950) | >= 0.90 | n/a |
| Context Precision | 0.920 | >= 0.88 | PASS |
| Context Recall | 0.770 | >= 0.85 | FAIL |

The corrected Module 15 judge HAS been re-run. Artifact: `docs/module15_scorecard_corrected.txt` (24,752 bytes).
Context Precision MEASURES 0.920 and PASSES. Context Recall measures 0.770 and FAILS.
The two answer-dependent metrics have no corrected measurement, because the provider was out of quota.
The run generated 0 of 10 answers (HTTP 429), so Faithfulness and Answer Relevancy were averaged over 0 of 10 questions.
The module prints `n/a` and NOT MEASURED for them, and the overall verdict is NOT FULLY MEASURED.
The later subsection "Module 15 Scorecard Correction (Closure)" records the defects that were fixed and the decomposition of the old number.

### 10.2 Why Criterion 4 Must Not Be Rounded Up

Three low-relevance cases were tested. All three self-corrected.
That is a 100% result on three cases, and a 100% result on three cases is not a percentage worth printing.
A percentage needs a query set. Phase 6 did not build one.
The gate therefore records the criterion as PARTIALLY MET.

### Module 15 Scorecard Correction (Closure)

Module 15 produced the final Ragas scorecard. The first recorded Context Precision was 0.600, and it FAILED the 0.88 target.
That number was largely a MEASUREMENT ARTIFACT, not a retrieval verdict.
The scorecard was re-run after the defects below were fixed.

**The three measurement defects.**

| # | Defect | Effect on the number |
|---|---|---|
| 1 | The judge saw only the first 200 characters of each 800-character chunk | The judge scored section headings instead of chunk content. Context Precision was understated. |
| 2 | A judge exception silently became a 0.5 score | A rate-limit error became a fake "partial" score. No failure was visible. |
| 3 | Content was passed as LangChain template text | A brace in the chunk text or the answer raised `ValueError: Invalid format specifier`, and the judge crashed. |

**The generation-outage reporting defect.**

The corrected run met a provider quota wall. All 10 answers FAILED TO GENERATE with HTTP 429.
Module 15 substituted `[LLM Error: 429 ...]` as the "answer".
The judges scored those error strings 0.00 for Faithfulness and Answer Relevancy.
That 0.00 was the CORRECT verdict on an error message.
Module 15 first printed it as a quality collapse (F=0.000 / R=0.000 / FAIL), which mislabelled a provider outage as a quality regression.
That reporting defect is now fixed. The module now does five things:

| # | Fixed behaviour |
|---|---|
| 1 | Failed generations are detected per question. |
| 2 | The two answer-dependent metrics are reported as NOT MEASURED. |
| 3 | The metric table prints `n/a` instead of `0.000`. |
| 4 | The overall verdict says NOT FULLY MEASURED. |
| 5 | The closing banner no longer claims verification. |

**The judge failover chain.**

The Groq free-tier token limits are PER MODEL. A saturated judge model therefore must not fail the metric.
The judge chain is `qwen/qwen3.8-27b, allam-2-7b, openai/gpt-oss-20b`.
On the corrected run the served judge calls were qwen = 1 and allam-2-7b = 19.
A mixed-judge scorecard is comparable only within itself.

**The A/B that isolates the truncation effect.**

The A/B holds the judge fixed and uses the same chunks and the same 10 questions.
ONLY the chunk text shown to the judge varies.

| Arm | Mean Context Precision | vs 0.88 target |
|---|---|---|
| 200-character preview (old behaviour) | **0.780** | FAIL |
| 800-character whole chunk (new behaviour) | **0.920** | PASS |

| Question | 200-character preview | Whole chunk | Change |
|---|---|---|---|
| TC001 | 1.0 | 1.0 | 0.00 |
| TC002 | 1.0 | 0.8 | -0.20 |
| TC003 | 0.5 | 0.8 | +0.30 |
| TC004 | 0.5 | 1.0 | +0.50 |
| TC005 | 0.5 | 0.8 | +0.30 |
| TC006 | 1.0 | 1.0 | 0.00 |
| TC007 | 0.5 | 0.8 | +0.30 |
| TC008 | 0.8 | 1.0 | +0.20 |
| TC009 | 1.0 | 1.0 | 0.00 |
| TC010 | 1.0 | 1.0 | 0.00 |

7 questions improved. 2 were unchanged. 1 was slightly worse.
The mean effect is +0.140 absolute, which is +17.9% relative.
The corrected full run independently scored 0.920, which cross-validates the 800-character arm.

**The decomposition of the old recorded number.**

- 0.600 = qwen judge + 200-character truncation + the silent-0.5 masking defect (the pre-fix baseline, artifact `docs/module15_scorecard_verified.txt`)
- 0.780 = allam-2-7b judge + 200-character truncation, so **+0.18 is judge calibration** (qwen was the harsher judge)
- 0.920 = allam-2-7b judge + whole chunks, so **+0.14 is the truncation fix alone**

The two effects are independent and additive: 0.18 plus 0.14 is 0.32, which is the full move from 0.600 to 0.920.

**The conclusion, stated plainly.**

The recorded Context Precision of 0.600 was largely a MEASUREMENT ARTIFACT, not a retrieval verdict.
With the measurement corrected it is 0.920, and it PASSES the target.

**Context Recall.**

| Measurement | Value |
|---|---|
| Pre-fix (qwen judge) | 0.750 |
| Corrected (allam-2-7b judge) | 0.770 |

Recall's context was NEVER truncated. The small change is therefore judge calibration only.
The recall shortfall is REAL retrieval quality. It is confirmed under two different judges, and it remains below the 0.85 target.

**Still NOT measured.**

Faithfulness (pre-fix 0.950) and Answer Relevancy (pre-fix 0.950) have no corrected measurement.
The generation outage removed every answer.
A full re-run is required when the provider has quota.
The module prints `n/a` and NOT MEASURED instead of failing those two metrics.

---

## 11. Key Findings Summary

### 11.1 Critical Engineering Decisions

| Decision | Rationale | Alternative Considered |
|---|---|---|
| A deterministic, explainable router instead of an LLM classifier | Every decision can be audited. The default run costs zero tokens. | An LLM classifier by default (costs tokens for a decision that regexes make correctly) |
| A margin below 1.0 falls back to RAG and is marked contested | Targeted retrieval is the cheap default, so ambiguity resolves to the cheap side | Resolve a tie toward long context (pays for the whole document on an unclear query) |
| Hierarchical headers instead of per-chunk headers | 19 generation targets instead of 105 on a per-model, per-minute free tier | 105 document-sized calls (a rate-limit wall) |
| Every header cached by content hash | The same work is never paid for twice: 53 calls once, then zero forever | Regenerate headers on each build |
| A private Chroma directory for the two new collections | Module 05's collection is left untouched | Reuse the production collection (destroys the baseline comparison) |
| Late chunking with `stride = 0` windows | Overlapping windows encode a boundary token twice and make pooled spans ambiguous | Overlapping windows (rejected after a real trial) |
| Late chunking as a fully local module | It still runs when every API quota is exhausted | Depend on an embedding API |
| `external_lookup` flows unconditionally to `sanitize` | The fallback becomes a sink and cannot cycle | A conditional edge back into the graph (permits a cycle) |
| `verified == True` has exactly one meaning | An ungrounded answer can never be mistaken for a corpus-backed one | Trust the model to describe its own provenance |
| A visible label on every non-verified path | Provenance is a property of the system, not a claim in the prose | A silent fallback |
| A 300-token judge budget plus a second opinion | A 12-token cap returns an empty body and scores a correct answer 0.5, which would push good internal answers external | One cheap judge call (fails Scenario A) |
| Every graph edge cites a real chunk and a real substring | An ungrounded graph invents causality and then cites itself | Extract edges and trust them |
| Both opt-ins for the GraphRAG narrative (`--llm` and `INCIDENTLENS_ALLOW_LLM=1`) | An earlier `--llm` test reached the provider through a `.env` copy | A single flag |
| A synthetic chart probe for vision | It is the only honest way to prove visual reading on a corpus with no charts | Report the logo as if it were a chart |
| `HF_HOME` set before the `sentence-transformers` import | `huggingface_hub` resolves its cache directories at import time | Set it after the import (has no effect) |
| The cache verified by a real file write | A cache directory can exist and still be unwritable | Test that the directory exists |

### 11.2 Verified Numbers

| Measurement | Value |
|---|---|
| Router signals | 12 (6 favouring RAG, 6 favouring long context) |
| Router decision table | 6 of 6 correct |
| RAG benchmark input tokens / latency / quality | 887 / 19,483 ms / 1.00 |
| Long-context benchmark input tokens / latency / quality | 1,830 (2.1x) / 19,347 ms / 0.75 |
| Largest post-mortem | 29,402 characters (about 7,432 tokens) |
| Contextual headers, generation targets | 19 hierarchical (4 + 15); 124 in `--full` |
| Contextual headers, provider calls on a cold cache | 53 hierarchical; 357 in `--full` |
| Contextual cache, calls per invocation | [53, 0, 0, 0, 0, 0, 0] |
| Contextual embedded text | 62,927 -> 128,435 characters (+104.1%) |
| Contextual passage MRR@4 | 0.592 -> 0.700 (+18.3% relative) |
| Contextual orphan-probe hit-rate@4 (labelled) | 0.667 -> 0.333 |
| Late chunking windows | 66 across 15 documents |
| Late chunking characters reconstructed | 63,017 |
| Late chunking integrity | 105/105 spans, 0 overflow, 0 alignment failures, 0 coverage gaps |
| Late chunking macro adjacent cosine | 0.5540 -> 0.7314 (+0.1774) |
| Late chunking centroid affinity | 0.7346 -> 0.8005 (+0.0658) |
| Late chunking mean evidence rank | 30.50 -> 25.07 (+5.43) |
| Late chunking truncated chunks (standard) | 7 |
| Agentic RAG new nodes / new routers | 2 / 2 |
| Agentic RAG loop bounds | rewrites 3, synthesis 2, external 1, hops 12 |
| Agentic RAG scenarios | 3 of 3 PASS |
| Agentic RAG LLM calls | 11 of 15 budgeted |
| Scenario B rewrite scores | 0.20, 0.00, 0.20, 0.30 |
| GraphRAG nodes / edges / components | 42 / 39 / 4 |
| GraphRAG component sizes | Kinesis 9, Cloudflare 7, GitHub 10, CrowdStrike 13 |
| GraphRAG edges rejected from the poisoned list | 4 of 43 |
| GraphRAG multi-hop queries | 6 PASS, 20 hops total |
| GraphRAG checks | 16 of 16 PASS, zero LLM calls |
| GraphRAG worst-case disconnection | 10 downstream nodes |
| Vision image slots / unique images | 24 / 2 |
| Vision synthetic chart strings matched | 8 of 8 |
| Vision calls used | 6 |
| Vision retrieval distances | 0.3301 and 0.5033 |
| Scorecard: Faithfulness / Answer Relevancy (pre-fix) | 0.950 / 0.950 (PASS pre-fix; NOT MEASURED in the corrected run) |
| Scorecard: Context Precision (pre-fix -> corrected) | 0.600 -> **0.920** (PASS after the measurement fix) |
| Scorecard: Context Recall (pre-fix -> corrected) | 0.750 -> **0.770** (FAIL, real retrieval quality) |
| Scorecard corrected run: answers generated | 0 of 10 (HTTP 429, provider out of quota) |
| Scorecard corrected run: judge calls by model | qwen = 1, allam-2-7b = 19 |
| Scorecard A/B: Context Precision (200-character preview -> whole chunk) | 0.780 -> 0.920 (+0.140 absolute, +17.9% relative) |

---

## 12. Phase 6 Architecture Diagram

```
   MODULE 21 - ROUTER
   +-----------------------------------------------------------------+
   |  12 weighted regex signals -> score_rag vs score_long            |
   |  margin < 1.0  -> RAG, marked CONTESTED                          |
   |  every fired signal printed with weight, matched span, confidence |
   +----------------+---------------------------+--------------------+
                    |                           |
                    v                           v
            STRATEGY A: RAG              STRATEGY B: LONG CONTEXT
            hybrid top-k chunks          the whole document text
            887 input tokens             1,830 input tokens
            19,483 ms                    19,347 ms
            quality 1.00                 quality 0.75 (truncated)
```

```
   MODULE 22 - CONTEXTUAL RETRIEVAL
   +-----------------------------------------------------------------+
   |  chunk -> "[CONTEXT: <doc blurb> | <segment blurb>]" + chunk     |
   |  19 generation targets (4 doc + 15 seg), cached by content hash  |
   |  cold cache 53 calls -> warm cache ZERO calls, asserted          |
   |                                                                  |
   |  incidentlens_baseline    62,927 chars   Precision@4 1.000       |
   |  incidentlens_contextual 128,435 chars   Precision@4 1.000       |
   |  passage MRR@4  0.592 -> 0.700  (+18.3%)                         |
   |  docs/12 +10% bar: NOT MET (metric saturated)                    |
   +-----------------------------------------------------------------+
```

```
   MODULE 23 - LATE CHUNKING
   +-----------------------------------------------------------------+
   |  document -> 256-token windows -> token vectors                  |
   |           -> pool the tokens of each chunk span -> chunk vector  |
   |                                                                  |
   |  105 chunks, 15 parents, 66 windows, CPU, dim 384                |
   |  integrity: 105/105 spans, 0 overflow, 0 gaps                    |
   |  adjacent cosine  0.5540 -> 0.7314  (+0.1774)                    |
   |  evidence rank    30.50  -> 25.07   (+5.43 positions)            |
   +-----------------------------------------------------------------+
```

```
   MODULE 24 - AGENTIC RAG (module 19 graph + 2 nodes + 2 routers)
                        START
                          |
                          v
                   [security_gate] --- blocked ---> [sanitize]
                          |
                          v
                   [retrieve] -> [grade_relevance]
                                     |  >= 0.60 -> [synthesize]
                                     |  LOW, budget left -> [rewrite_query] -+
                                     |  LOW, budget spent -> [external_lookup]
                                     v
                              [synthesize] -> [check_hallucination]
                                                   |  grounded -> [sanitize]
                                                   |  ungrounded, tries left -> [synthesize]
                                                   |  ungrounded, tries spent -> [external_lookup]
                                                                                      |
                                     [sanitize] <-------------------------------------+
                                          |
                                          v
                                   [finalize_end] -> END
                                          |
                       verified == True ONLY for internal + grounded + cited
                       every other path carries a visible label
```

```
   MODULE 25 - GRAPHRAG
   +-----------------------------------------------------------------+
   |  42 nodes, 39 edges, 4 components, verified DAG                  |
   |  every edge cites a real chunk id AND a real substring           |
   |  poisoned list: 4 of 43 edges rejected, with reasons             |
   |  6 multi-hop queries PASS, 20 hops, each hop cited               |
   |  negative case: NOT ANSWERABLE FROM THE GRAPH                    |
   |  16/16 checks PASS with ZERO LLM calls                           |
   +-----------------------------------------------------------------+
```

```
   MODULE 26 - MULTIMODAL VISION RAG
   +-----------------------------------------------------------------+
   |  PDF -> 24 image slots -> SHA-256 dedup -> 2 unique images       |
   |  446x80 logo (KIND: logo) + 949x4 rule                           |
   |  page rendering: UNAVAILABLE (no fitz, no pdf2image)             |
   |  synthetic PIL bar chart -> vision model -> 8/8 strings matched   |
   |  incidentlens_vision collection -> chart entry ranks first        |
   |  docs/12 Grafana criterion: NOT MET on this corpus                |
   +-----------------------------------------------------------------+
```

---

## 13. Run Commands

| Task | Command |
|---|---|
| Module 21 (router and benchmark) | `.\venv\Scripts\python.exe modules\21_long_context_vs_rag.py` |
| Module 21 with the optional LLM classifier | `.\venv\Scripts\python.exe modules\21_long_context_vs_rag.py --llm` |
| Module 22 (contextual retrieval, hierarchical) | `.\venv\Scripts\python.exe modules\22_contextual_retrieval.py` |
| Module 22 with true per-chunk headers | `.\venv\Scripts\python.exe modules\22_contextual_retrieval.py --full` |
| Module 22 with the optional LLM judge | `.\venv\Scripts\python.exe modules\22_contextual_retrieval.py --llm` |
| Module 22 index build only | `.\venv\Scripts\python.exe modules\22_contextual_retrieval.py --build-only` |
| Module 23 (late chunking, fully local) | `.\venv\Scripts\python.exe modules\23_late_chunking.py` |
| Module 24 (agentic RAG scenarios) | `.\venv\Scripts\python.exe modules\24_agentic_rag.py` |
| Module 25 (GraphRAG, deterministic) | `.\venv\Scripts\python.exe modules\25_graphrag_reasoning.py` |
| Module 25 with the optional narrative | `$env:INCIDENTLENS_ALLOW_LLM=1; .\venv\Scripts\python.exe modules\25_graphrag_reasoning.py --llm` |
| Module 26 (multimodal vision RAG) | `.\venv\Scripts\python.exe modules\26_multimodal_colpali.py` |

---

## 14. Glossary

| Term | Definition |
|---|---|
| **Router** | The Module 21 decision engine that chooses RAG or long context for one query. |
| **Signal** | One weighted regular-expression rule that argues for one strategy. |
| **Contested decision** | A routing decision whose score margin is below 1.0. It falls back to RAG. |
| **Long context** | Passing the whole document to a model with a large token window. |
| **Contextual retrieval** | Prepending a short situation header to a chunk before embedding it. |
| **Situation header** | The prepended text. It states which incident the chunk belongs to and what the chunk covers. |
| **Generation target** | One text the module asks a model to write. A provider call is different, because a retry or a failover is another call. |
| **Cache hit** | A header that already exists in the cache file. It costs zero provider calls. |
| **Saturated metric** | A metric that already scores its maximum, so no improvement is arithmetically possible. |
| **Late chunking** | Encoding the parent document first, then pooling the token vectors of each chunk span. |
| **Token window** | The maximum number of tokens one forward pass of the embedding model can hold. Here 256. |
| **Pooling** | Combining several token vectors into one vector, usually by taking the mean. |
| **Adjacent-chunk cosine** | The similarity between two neighbouring chunks of the same document. Late chunking raises it. |
| **Evidence rank** | The rank position of the chunk that holds the consequence of an earlier statement. A lower rank is better. |
| **Agentic RAG** | A RAG pipeline that can inspect its own work and take another action. |
| **Self-correction loop** | A bounded cycle that rewrites a query or re-synthesizes an answer. |
| **External fallback** | The path that leaves the corpus when internal knowledge is provably insufficient. |
| **Provenance label** | The visible text that states where an answer came from, such as `[EXTERNAL / UNVERIFIED]`. |
| **Sink node** | A graph node with no path back into the graph. It cannot take part in a cycle. |
| **Hop** | One step along a causal chain in the graph. |
| **Grounding rule** | The rule that every graph edge must cite a real chunk id and a real substring of that chunk. |
| **DAG** | A directed acyclic graph. A graph whose edges have a direction and contain no cycle. |
| **Weakly connected component** | A set of graph nodes that are connected when edge direction is ignored. |
| **Fan-out point** | A node with many downstream dependents, where parallel paths still carry the effect. |
| **Choke point** | A node whose removal actually disconnects downstream nodes. |
| **Multimodal RAG** | RAG that indexes and retrieves visual evidence as well as text. |
| **Vision model** | A model that accepts an image and returns a text description. |
| **SHA-256 deduplication** | Removing repeated images by comparing their content hash. |
| **Synthetic chart** | A chart the module draws itself, used to prove that visual reading works when the corpus has no real chart. |
| **`verified`** | The Module 24 flag. It is True only for an internal, grounded, cited answer. |
| **`HF_HOME`** | The environment variable that tells `huggingface_hub` where to store downloaded models. It must be set before the import. |
| **HTTP 429** | "Too Many Requests". The provider rate limit. Module 21 met a daily quota limit, not only a per-minute limit. |
| **RRF** | Reciprocal Rank Fusion. The hybrid retrieval fusion used by Module 10. |

---

*Document generated by the IncidentLens session - ASD STE 100 style*
*Project: `c:\Users\Vaishnav Ambilpur\Desktop\Applied Ai\Production-Level-RAG`*
