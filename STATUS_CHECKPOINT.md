# 📌 IncidentLens - Project Checkpoint

> **Date:** 2026-10-13
> **Status:** 🏆 Phases 0 through 6 complete. All 26 modules deployed and run.
> **Next Step:** Documentation follow-up (section 4). The module work is finished; the remaining task is to record it.

---

## 📊 1. Phase Status

| Phase | Scope | Status | Evidence |
|---|---|---|---|
| **Phase 0** | Project setup | ✅ Complete | Directory structure, `.env`, `requirements.txt` |
| **Phase 1** | Foundations (Modules 01-06) | ✅ Complete | 4 documents loaded, 105 chunks indexed, naive RAG answers with citations |
| **Phase 2** | Failure modes (Modules 07-08) | ✅ Complete | 5 failure modes reproduced, debugging toolkit operational |
| **Phase 3** | Retrieval optimization (Modules 09-13) | ✅ Complete | Hybrid BM25 + dense + RRF, reranking, token budgeting, 4-stage pipeline |
| **Phase 4** | Observability (Modules 14-17) | ✅ Complete | LangSmith, Ragas-style judge, Langfuse 4.x, semantic cache |
| **Phase 5** | Security, agent, API (Modules 18-20 + tests) | ✅ Complete | 25/25 security checks, agent rewrite loop, FastAPI service, 36 tests |
| **Phase 6** | Advanced RAG (Modules 21-26) | ✅ Complete | Router 6/6, late-chunking cohesion +0.1774, 3/3 agent scenarios, GraphRAG 16/16, vision pipeline verified. See 2.8 to 2.13. |

---

## 📈 2. Verified Numbers

> **Reading note:** Chunk counts, rule counts and threshold values come from the project files.
> Scorecard values, cache timings, security self-test results, agent run values, API run values and the pytest pass counts are recorded run outputs.
> No number here is estimated.

### 2.1 Corpus and Retrieval

| Measurement | Value |
|---|---|
| Documents | 4 authentic post-mortems |
| Production chunks | 105 (verified in the ChromaDB file) |
| Chunk size / overlap | 800 / 150 characters |
| RRF constant | `rrf_k=60` |
| Full 4-stage pipeline latency (Module 13) | 4,186 ms (CrowdStrike query) |

### 2.2 Evaluation Scorecard (Module 15)

Three scorecards were produced: two PRE-FIX runs that were identical in every value, and one CORRECTED run after the three measurement defects were fixed.

**Pre-fix scorecards (2 runs, identical values):**

| Metric | Score | Module 15 target | Docs 13 target | Result |
|---|---|---|---|---|
| Faithfulness | 0.950 | 0.85 | 0.92 | ✅ PASS |
| Answer Relevancy | 0.950 | 0.85 | 0.90 | ✅ PASS |
| Context Precision | 0.600 | 0.80 | 0.88 | ❌ FAIL |
| Context Recall | 0.750 | 0.80 | 0.85 | ❌ FAIL |

The 40 per-question scores repeated exactly across the two pre-fix runs. The measurement was stable.

**Corrected scorecard (artifact `docs/module15_scorecard_corrected.txt`, 24,752 bytes):**

| Metric | Score | Module 15 target | Docs 13 target | Result |
|---|---|---|---|---|
| Faithfulness | NOT MEASURED | 0.85 | 0.92 | n/a |
| Answer Relevancy | NOT MEASURED | 0.85 | 0.90 | n/a |
| Context Precision | **0.920** | 0.80 | 0.88 | ✅ PASS |
| Context Recall | **0.770** | 0.80 | 0.85 | ❌ FAIL |

Zero of 10 answers were generated, because the provider was out of quota (HTTP 429). Faithfulness and Answer Relevancy judge the ANSWER, so they were averaged over 0 of 10 questions and are reported as NOT MEASURED.
Judge calls served by model: qwen = 1, allam-2-7b = 19.
The recorded Context Precision of 0.600 was largely a MEASUREMENT ARTIFACT, not a retrieval verdict. See section 3 for the decomposition.

### 2.3 Semantic Cache (Module 17)

| Measurement | Value |
|---|---|
| Threshold | 0.65 (was 0.92, which could never hit) |
| Paraphrase similarity floor | 0.7244 |
| Highest different-topic similarity | 0.3391 |
| Safe band | 0.34 to 0.72 |
| Cache hit rate | 50% (3 fresh answers, then 3 hits) |
| Fresh answer latency | about 1.4 s |
| Cache hit latency | about 300 ms |
| Latency saved | 3,497 ms |

### 2.4 Security (Module 18)

| Measurement | Value |
|---|---|
| Injection rules | 11, in 8 categories |
| PII rule types | 8 |
| Block threshold | 0.60 (highest weight + 0.10 per extra match) |
| Injections blocked | 10 / 10 |
| Benign questions allowed | 8 / 8 |
| PII redaction and token survival | 7 / 7 |
| **Self-test total** | **25 / 25** |
| Injection block latency (API path) | 15 ms, zero LLM calls |
| Injection block latency (agent path) | 22 ms, zero LLM calls |

### 2.5 Agent (Module 19)

| Measurement | Value |
|---|---|
| Graph nodes / conditional edges / loops | 7 / 3 / 2 |
| Rewrite limit | 3 |
| Relevance improvement (recorded run) | 0.50 -> 0.80 across 3 rewrites |
| Injection blocked | 22 ms, zero LLM calls |
| Thread checkpoints (recorded run) | 29 |
| Accumulated thread steps (recorded run) | 23 |
| Memory store | `data/agent_memory.sqlite` |
| Persistence proof | Read back through a fresh saver connection |

### 2.6 API (Module 20)

| Measurement | Value |
|---|---|
| `/health` status | `ok`, 6 dependency checks |
| Vector store | 105 chunks |
| Security rules reported | 11 injection, 8 PII |
| Agent graph | `memory=sqlite`, 9 graph nodes (7 plus START and END) |
| Memory database size (recorded run) | 532 KB |
| Live question | Correct cited answer, 20,688 ms |
| `/metrics` (recorded run) | 9 requests, 3 queries, 2 blocked, 0 failed |

### 2.7 Test Suite

| Measurement | Value |
|---|---|
| Tests in `tests/` | 36 |
| Default result | 32 passed, 4 skipped, 0 failures |
| Live-LLM tests | 4, marked `@pytest.mark.llm` |
| Live-LLM result on demand | All 4 verified passing |

### 2.8 Phase 6 - Long Context vs. RAG Router (Module 21)

| Measurement | Value |
|---|---|
| Router signals | 12 (6 favouring RAG, 6 favouring long context) |
| Contested fallback | Margin below 1.0 falls back to RAG and is marked contested |
| Router decision table | 6 of 6 queries routed to the expected strategy |
| RAG benchmark | 887 input tokens, 19,483 ms, quality 1.00 |
| Long-context benchmark | 1,830 input tokens (2.1x), 19,347 ms, quality 0.75 |
| Largest post-mortem | 29,402 characters (about 7,432 tokens) |
| Final-run provider | Groq (OpenRouter free tier hit its DAILY quota, HTTP 429 `free-models-per-day`) |
| Cached long-context model | `nvidia/nemotron-3-ultra-550b-a55b:free`, 1,000,000-token window |

### 2.9 Phase 6 - Contextual Retrieval (Module 22)

| Measurement | Value |
|---|---|
| Header mode by default | Hierarchical: 4 document + 15 segment = 19 generation targets (124 in `--full`) |
| Provider calls on a cold cache | 53 hierarchical (18 were HTTP 429); 357 in `--full` |
| Cross-run cache history (calls per invocation) | [53, 0, 0, 0, 0, 0, 0] |
| Cache proof | The module asserts a zero-call second run. Every invocation after the first spent ZERO. |
| Collections | `incidentlens_baseline` and `incidentlens_contextual`, in `data/chroma_db_contextual` |
| Embedded text | 62,927 -> 128,435 characters (+104.1%) |
| Document Precision@4 | 1.000 -> 1.000 (SATURATED) |
| Hit-rate@4 | 1.000 -> 1.000 (SATURATED) |
| Document MRR | 1.000 -> 1.000 |
| Passage-level MRR@4 | 0.592 -> 0.700 (+18.3% relative) |
| Orphan-probe hit-rate@4 (labelled block) | 0.667 -> 0.333 (regression, reported) |
| docs/12 +10% precision bar | **NOT MET** (the metric is saturated, so the bar is arithmetically unreachable) |

### 2.10 Phase 6 - Late Chunking (Module 23)

| Measurement | Value |
|---|---|
| Scale | 105 chunks, 15 parent documents, 63,017 characters reconstructed |
| Model | `sentence-transformers/all-MiniLM-L6-v2`, `max_seq_length` 256, dim 384, CPU |
| Forward-pass windows | 66 across the 15 documents (13 for the GitHub post-mortem alone) |
| Integrity | 105/105 spans resolved; 0 chunks pooled outside their span; 0 alignment failures; 0 coverage gaps |
| Macro adjacent-chunk cosine | 0.5540 -> 0.7314 (+0.1774); 14 of 15 document groups improved |
| Chunk-to-document-centroid affinity | 0.7346 -> 0.8005 (+0.0658) |
| Temporal-link probes | 14 probes; mean evidence rank 30.50 -> 25.07 (+5.43 positions) |
| Probe directions | 8 improved, 6 worsened, 0 tied; best single gain +22 positions; linked pairs closer in 12 of 14 cases |
| Standard-chunking truncation | 7 chunks silently truncated at 256 tokens |
| API requirement | None. Fully local, CPU only. |

### 2.11 Phase 6 - Self-Correcting Agentic RAG (Module 24)

| Measurement | Value |
|---|---|
| New nodes / new routers | 2 (`external_lookup`, `finalize_end`) / 2 |
| Loop bounds | `MAX_REWRITES` 3, `MAX_SYNTHESIS` 2, `MAX_EXTERNAL_TRIES` 1, `MAX_TOTAL_HOPS` 12 |
| `verified == True` | Internal source AND grounding 0.80 AND at least one `[chunk_id]` citation |
| Scenario A (in-corpus) | PASS: `grade_relevance` 0.80 -> grounded 0.80 -> `finalize_end(verified:internal)` |
| Scenario B (out-of-corpus) | PASS: RRF 0.0328; rewrites 0.20, 0.00, 0.20, 0.30; external fallback labelled `[EXTERNAL / UNVERIFIED]`, `verified=FALSE` |
| Scenario C (injection) | PASS: blocked with ZERO LLM calls; `retrieve` and `external_lookup` never ran |
| LLM calls | 11 of 15 budgeted |
| Langfuse | `auth_check` passed, fresh `propagate_attributes` scope per query, client flush, observations read back |
| Judge fix | 300-token judge budget; honest refusal accepted; second opinion from a stronger model |
| Back-port | Module 19 gained `llm_score()`, which escalates the token budget only when the cheap attempt yields no parseable number |

### 2.12 Phase 6 - GraphRAG (Module 25)

| Measurement | Value |
|---|---|
| Graph | 42 nodes, 39 edges, 4 weakly connected components, verified as a DAG |
| Component sizes | Kinesis 9, Cloudflare 7, GitHub 10, CrowdStrike 13 |
| Grounding gate | 4 of 43 edges rejected from a deliberately poisoned claim list, each with an exact reason |
| Multi-hop queries | 6 PASS, 20 hops total, every hop printed with an exact-substring citation |
| 3-hop case | Noction BGP Optimizer -> Verizon (AS701) -> internet-wide degradation; cites `cloudflare_bgp_leak_p1_c3`, `_c7`, `_c8` |
| 5-deep case | Kinesis chain reaches Lambda, Console, CloudWatch and Cognito |
| Negative case | `NOT ANSWERABLE FROM THE GRAPH` for a DNS TTL question; corpus scan confirms 0 hits |
| Worst-case disconnection | 10 downstream nodes (`100G optical maintenance`; the 21-input IPC template definition) |
| Checks | 16 of 16 PASS with ZERO LLM calls by default |
| API requirement | None. `--llm` additionally requires `INCIDENTLENS_ALLOW_LLM=1` and rejects dummy keys. |

### 2.13 Phase 6 - Multimodal Vision RAG (Module 26)

| Measurement | Value |
|---|---|
| Embedded-image slots (CrowdStrike PDF) | 24 (2 per page across 12 pages) |
| UNIQUE images | 2: a 446x80 wordmark LOGO and a 949x4 decorative rule |
| Vision model classification of the logo | `KIND: logo / TEXT: CROWDSTRIKE / CHART_READING: NONE` |
| Real charts in the corpus | 0. No Grafana chart, topology diagram, flamegraph or heatmap. |
| Page rendering | UNAVAILABLE (no `pymupdf`/`fitz`, no `pdf2image`). The module installs nothing. |
| Synthetic chart probe | All 8 drawn strings read back exactly (api-gateway 42, checkout 78, payments 95, search 23) |
| Vision calls | 6 |
| Retrieval probe | Both chart-shaped queries returned the chart entry first (distances 0.3301 and 0.5033) |
| Failure-mode handling | `google/gemma-4-26b-a4b-it:free` returned HTTP 429 three times, then failed over to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` |
| docs/12 Grafana criterion | **NOT MET** on this corpus (stated in the module docstring and run summary) |

---

## 🚧 3. Open Items

- **Phase 6 criterion NOT MET: contextual retrieval did not improve context precision by >=10%.** The baseline already scores document Precision@4 = 1.000 and hit-rate@4 = 1.000 on ALL 10 golden questions, so the document-level metric is SATURATED and the +10% bar is arithmetically unreachable on this dataset.
  The finer, unsaturated metric did improve (passage-level MRR@4 0.592 -> 0.700, +18.3% relative).
  The module also reports that the optional LLM judge put the context-aware variant BELOW the baseline (0.790 vs 0.920). The target stays recorded as NOT MET.
- **Phase 6 criterion NOT MET: multimodal RAG did not extract actionable information from Grafana chart images.**
  The corpus contains no charts: the CrowdStrike PDF has 24 embedded-image slots but only 2 unique images (a 446x80 wordmark logo and a 949x4 decorative rule).
  The vision pipeline is proven on a SYNTHETIC chart drawn by the module (8 of 8 drawn strings read back). Plumbing verified, corpus evidence not available.
- **Context Precision: RESOLVED and PASSING at 0.920. Context Recall: still below target at 0.770.**
  The corrected Module 15 run (artifact `docs/module15_scorecard_corrected.txt`, 24,752 bytes) measured Context Precision at **0.920**, against the Module 15 threshold of 0.80 (PASS) and the docs/13 target of 0.88 (PASS).
  The same run measured Context Recall at **0.770**, against the Module 15 threshold of 0.80 (FAIL) and the docs/13 target of 0.85 (FAIL).
  The recorded Context Precision of 0.600 was largely a MEASUREMENT ARTIFACT: three measurement defects (the judge saw only the first 200 characters of each 800-character chunk, a judge exception silently became a 0.5 score, and content passed as LangChain template text let braces crash the judge).
  - 0.600 = qwen judge + 200-character truncation + the silent-0.5 masking defect (pre-fix baseline, `docs/module15_scorecard_verified.txt`)
  - 0.780 = allam-2-7b judge + 200-character truncation, so **+0.18 is judge calibration** (qwen was the harsher judge)
  - 0.920 = allam-2-7b judge + whole chunks, so **+0.14 is the truncation fix alone**
  The A/B that isolates the truncation effect (fixed judge, same chunks, same 10 questions, only the chunk text shown to the judge varies) measured 0.780 for the 200-character preview against 0.920 for the whole 800-character chunk: 7 questions improved, 2 were unchanged, 1 was slightly worse, mean effect +0.140 absolute (+17.9% relative).
  Context Recall's context was NEVER truncated, so its small change (0.750 -> 0.770) is judge calibration only. The recall shortfall is therefore REAL retrieval quality, confirmed under two different judges, and it remains below the 0.85 target.
- **Faithfulness and Answer Relevancy have no corrected measurement, and a quota window is PENDING.** The corrected run generated 0 of 10 answers, because the provider was out of quota (HTTP 429), so the two answer-dependent metrics were averaged over 0 of 10 questions. The pre-fix values were 0.950 each. A full re-run is required when the provider has quota.
- **Module 15 now has a judge failover chain and generation-outage reporting.** The judge chain is `qwen/qwen3.8-27b, allam-2-7b, openai/gpt-oss-20b`, because Groq free-tier token limits are PER MODEL. The corrected scoring path detects failed generations, reports the two answer-dependent metrics as NOT MEASURED, prints `n/a` instead of `0.000` in the metric table, states the overall verdict as NOT FULLY MEASURED, and no longer claims verification in the closing banner. That answers the earlier reporting defect that mislabelled a provider outage as a quality collapse (F=0.000 / R=0.000 / FAIL). On the corrected run the judges scored the substituted `[LLM Error: 429 ...]` strings 0.00 for Faithfulness and Answer Relevancy, which was the CORRECT verdict on an error message.
- **The P95 end-to-end latency criterion is NOT MET.** Measured 20.7 s on the free tier (Phase 5). Module 21 measured 19,483 ms for RAG and 19,347 ms for long context in Phase 6, on the same free tier.
  When quota is available the same paths run in about 1.5 s to 4 s. This is a free-tier throughput limit, not an architectural limit, but the criterion is recorded as not met.
- **The OpenRouter free tier is DAILY-limited, not only per-minute.** Module 21 hit HTTP 429 `free-models-per-day` mid-session, so the final benchmark run was served by Groq. Module 26 also met HTTP 429 three times on its preferred vision model.
  Consequence: Modules 21 and 26 must be run early in a quota day, and both must keep a fallback chain. Module 22's cache is the model answer to this problem: 53 provider calls once, then ZERO forever.
- **The LangSmith key was never regenerated.** The key in `.env` returned HTTP 403 historically. The cause was a possibly truncated or invalid key.
  The pipeline still runs in local mode without a valid key. The fix is to regenerate the key at `https://smith.langchain.com` -> Settings -> API Keys, and to remove trailing whitespace in `.env`.
- **Phase 6 known limitations that are recorded, not fixed:** Module 21 did not measure the 25-30x whole-document token multiple (this corpus gave 2.1x, because the largest post-mortem is only 29,402 characters). Module 23's "long-range attention" is bounded by a 256-token window, and MiniLM is used off-label for sub-span pooling. Module 25 reports that a large downstream count with parallel paths is a fan-out point, not a choke point.
- **Documentation status:**
  - `learning/02_session_log_phase0_to_phase4.md` still records Module 15 as "Pending" and does not contain the Module 16 and 17 defects. The corrections are written in the Phase 5 log.
  - The Phase 5 log exists as `03_session_log_phase5.md` and covers Modules 18, 19, 20 and the test suite.
  - The Phase 6 log exists as `04_session_log_phase6.md` and covers Modules 21 through 26.
  - The Phase 6 document copy `12_phase6_advanced_rag.md` exists outside the project folder, with the Status line completed and section 6.3 annotated with the true verdict for all seven criteria.
  - `docs/00_ROADMAP_INDEX.md` still shows every phase as "⬜ Not Started" in its status column.
  - The project copy of `docs/11_phase5_security_api.md` still shows "⬜ Not Started". An updated copy with the completed status and the ticked checklist exists outside the project folder.
- **Module 15 re-run cost.** Each golden question costs 5 LLM calls (1 answer plus 4 judges). A 10-question re-measurement costs 50 calls and must be paced for the free tier. Only Faithfulness and Answer Relevancy still need that re-run.

---

## ▶️ 4. Where to Resume

### Immediate Step

Re-measure Faithfulness and Answer Relevancy with the corrected judge when the provider has quota, then update the scorecard record:

```powershell
.\venv\Scripts\python.exe modules/15_ragas_evaluation.py
```

Run it when the Groq quota is fresh. Context Precision (0.920, PASS) and Context Recall (0.770, FAIL) are already measured by the corrected run.
The corrected module retries with backoff, fails over to the next judge model in the chain, and prints `n/a` and NOT MEASURED for a metric that has no data instead of a fake score.

### Then - the documentation follow-up

1. Add the Phase 4 corrections to `learning/02_session_log_phase0_to_phase4.md`, or leave the pointer to the Phase 5 log.
2. Update the status column in `docs/00_ROADMAP_INDEX.md` for Phases 1 through 6.
3. Regenerate the LangSmith key and confirm traces in the dashboard, or record LangSmith as permanently deferred.
4. Record the Phase 6 outcome in the roadmap: three criteria MET, three NOT MET or PARTIALLY MET, and the reason for each.

---

## ✅ 5. Phase 6 (docs/12) - Complete

**Source:** `docs/12_phase6_advanced_rag.md`

| Module | File | Technique | Verification result |
|---|---|---|---|
| 21 | `modules/21_long_context_vs_rag.py` | Long context vs. RAG router | Router 6 of 6; benchmark measured; 25-30x multiple not measured on this corpus (2.1x) |
| 22 | `modules/22_contextual_retrieval.py` | Contextual retrieval | Passage MRR +18.3%; docs/12 +10% precision bar NOT MET (saturated metric) |
| 23 | `modules/23_late_chunking.py` | Late chunking | Cohesion +0.1774; evidence rank +5.43 positions; all integrity checks pass |
| 24 | `modules/24_agentic_rag.py` | Agentic RAG | 3 of 3 scenarios PASS with machine-checked assertions; 11 of 15 LLM calls |
| 25 | `modules/25_graphrag_reasoning.py` | GraphRAG | 16 of 16 checks PASS with ZERO LLM calls; 6 multi-hop queries, 20 hops, all cited |
| 26 | `modules/26_multimodal_colpali.py` | Multimodal RAG | Pipeline verified on a synthetic chart; Grafana criterion NOT MET on this corpus |

**Final quality bar for Phase 6:**

| Metric | Target | Measured | Result |
|---|---|---|---|
| Faithfulness | >= 92% | NOT MEASURED (pre-fix 0.950) | n/a (generation outage) |
| Context Precision | >= 88% | 0.920 | ✅ PASS (corrected measurement) |
| Answer Relevance | >= 90% | NOT MEASURED (pre-fix 0.950) | n/a (generation outage) |
| Context Recall | >= 85% | 0.770 | ❌ FAIL (measured under two judges) |

> The corrected Module 15 scorecard is `docs/module15_scorecard_corrected.txt` (24,752 bytes). The previously recorded Context Precision of 0.600 was a measurement artifact.

---

## 🧪 6. Run Commands

| Task | Command |
|---|---|
| Single module (example: security) | `.\venv\Scripts\python.exe modules/18_security_guardrails.py` |
| Single module with the optional LLM layer | `.\venv\Scripts\python.exe modules/18_security_guardrails.py --llm` |
| The API server | `.\venv\Scripts\python.exe modules/20_api_server.py` |
| The test suite | `.\venv\Scripts\python.exe -m pytest tests/ -v` |
| The test suite with live LLM tests | `$env:INCIDENTLENS_TEST_LLM=1; .\venv\Scripts\python.exe -m pytest tests/ -v` |

---

## 🧭 7. Environment Keys Required

```
GROQ_API_KEY          -> free at console.groq.com
OPENROUTER_API_KEY    -> free at openrouter.ai
LANGCHAIN_API_KEY     -> free at smith.langchain.com (currently HTTP 403, see open items)
LANGCHAIN_TRACING_V2  -> "true"
LANGCHAIN_PROJECT     -> "IncidentLens"
LANGFUSE_SECRET_KEY   -> free at cloud.langfuse.com
LANGFUSE_PUBLIC_KEY   -> free at cloud.langfuse.com
LANGFUSE_BASE_URL     -> "https://us.cloud.langfuse.com"
```

> The venv contains langfuse 4.17.0. Module 16 is written for the v4 API with fallbacks for v2 and v3.

---

*Checkpoint updated from the verified state of the project. ASD STE 100 style.*
