# 17 — Interview Defense

> **Context:** One answer for each stage. Use these answers aloud. The target is 90 seconds for each answer.
> **Prerequisite:** [14_pipeline_plan.md](./14_pipeline_plan.md), [15_evals.md](./15_evals.md)
> **Status:** 📄 Defined

---

## 17.1 The Rule

Every RAG question is one of two questions:

1. **Which stage?**
2. **How do you know?**

If you can put a question into a stage and point at your own number, you have the answer. This file is the map.

---

## 17.2 The Fifteen-Second Answer

Use this answer first. Then give the detail that the interviewer asks for.

> "IncidentLens is a retrieval pipeline for incident diagnosis. It has 10 stages: ingest, chunk, embed, index, hybrid retrieve, fuse, rerank, contextualize, generate, and evaluate. I can operate each stage alone and measure each stage alone. The hard part is not the model call. The hard part is retrieval quality and the measurement of retrieval quality."

---

## 17.3 The Stage Answers

### Stage 1 — Ingest

**Question:** "How do you get the documents?"

**Answer:** "I read PDF files and Markdown files. I remove the duplicates. I attach four metadata items: source, date, system, and failure type. The metadata lets me filter the corpus. For example, I can search only the network incidents."

**The number:** Document count. Duplicate count. Metadata coverage.

---

### Stage 2 — Chunk

**Question:** "How do you split the documents?"

**Answer:** "I use 800 characters with 150 characters of overlap. I chose that size by measurement, not by feeling. A small chunk loses the context. A large chunk dilutes the signal and costs more tokens. I compared fixed-size, recursive, semantic, and late chunking in the ablation table."

**The number:** Chunk count. Mean token count. **The boundary-split rate** — the percentage of chunks that cut a sentence or a table in the middle.

**The advanced answer:** "Late chunking is different. I embed the full document first, then I pool the token vectors over each chunk span. That keeps the document context inside the chunk vector. It improved the adjacent-chunk cosine from 0.5540 to 0.7314."

---

### Stage 3 — Embed

**Question:** "Which embedding model? Why?"

**Answer:** "`all-MiniLM-L6-v2`. It gives 384 dimensions. It operates on the CPU. It needs no key and no money. It has a limit: the maximum sequence is 256 tokens. Seven chunks were silently cut at that limit. I know this because I measured the truncation."

**The tradeoff:** "A larger model such as BGE-M3 gives better vectors and multilingual support. It also gives more latency and more memory. I chose the small model because the corpus is English and the latency limit is tight."

> **NOTE:** The first version of `03_tech_stack.md` said BGE-M3. The code uses MiniLM. Say MiniLM.

---

### Stage 4 — Index

**Question:** "Which vector store? Why ChromaDB?"

**Answer:** "ChromaDB in the persistent local mode. It writes to a directory. It needs no server and no key. The corpus is small, so a local store is sufficient. A hosted store would add a key and a network delay for no gain."

**The number:** Vector count. Index build time.

---

### Stage 5 — Retrieve

**Question:** "Why hybrid retrieval? Why not dense only?"

**Answer:** "Dense embeddings cannot match a literal string. An incident query contains exact codes: `SIGSEGV`, `0x00000050`, `AS33154`. A dense vector sees these as similar to other codes. BM25 sees the exact characters. Therefore, I use both: BM25 for the exact terms and dense search for the meaning. This is the reason for the hybrid design."

**The number:** Recall@k, MRR, nDCG@k.

---

### Stage 6 — Fuse

**Question:** "How do you combine the two result lists?"

**Answer:** "I use Reciprocal Rank Fusion with k=60. RRF uses the rank position, not the score. This property is important: a BM25 score of 12.4 and a cosine distance of 0.31 are not comparable numbers. RRF does not compare them. The alternative is a weighted sum, and the weight needs a calibration for each corpus. RRF needs no calibration."

**The number:** The rank correlation between the two lists.

**The hard question:** "Why k=60?" — Answer: "k=60 comes from the original RRF paper. A larger k gives the top ranks less weight. I kept the standard value and I note that I did not tune it. That is an honest limitation."

---

### Stage 7 — Rerank

**Question:** "Why rerank if retrieval already works?"

**Answer:** "Retrieval uses a bi-encoder. The query and the chunk are embedded separately, so they never see each other. A cross-encoder reads the query and the chunk together. It is more accurate and much slower. Therefore, I use the fast retriever to get 20 candidates and the slow reranker to order the first 4 or 5."

**The model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, 22M parameters, local, no key.

**The number:** The change in nDCG@5, and the latency cost for each query.

---

### Stage 8 — Contextualize

**Question:** "What is contextual retrieval?"

**Answer:** "A chunk can lose its context. The chunk says 'the service restarted' but it does not say which service. Contextual retrieval puts a short header on each chunk before the embedding. The header names the document and the section. This made the passage-level MRR go from 0.592 to 0.700, an 18.3% relative improvement."

**The honest part:** "The document-level precision did not improve. The reason is saturation: the baseline was already 1.000. Also, one judge put the contextual version below the baseline, 0.790 against 0.920. I report both values. I do not report only the good one."

---

### Stage 9 — Generate

**Question:** "How do you stop a hallucination?"

**Answer:** "Three controls. First, the prompt requires a citation for each claim. A claim without a citation is a failure. Second, the answer must use only the supplied chunks. Third, the agent checks the grounding score. If the score is below 0.80, the agent rewrites the query and tries again, up to three times. If the score stays low, the answer is marked unverified."

**The number:** Faithfulness. Answer Relevance. The number of model calls for each query.

---

### Stage 10 — Evaluate

**Question:** "How do you know the system is good?"

**Answer:** "Four metrics: faithfulness, answer relevance, context precision, and context recall. The first two measure the answer. The last two measure the retrieval. I also keep a golden dataset of 10 questions with known answers."

**The best answer in this project:**

> "My first context precision was 0.600. That looked like a retrieval failure. It was not. Three defects in my measurement caused it. The worst one: the judge saw only the first 200 characters of each 800-character chunk. I isolated the effect with a controlled A/B — same judge, same chunks, same questions, only the text that the judge sees changed. The full chunk gave 0.920. That is +0.14 from the truncation fix alone. A separate +0.18 was judge calibration.
>
> I report context recall as 0.770 against a 0.85 target. That is a real gap. Two different judges confirmed it. I did not tune the question set to make it pass."

**Why this answer wins:** It shows that you separate a measurement failure from a system failure. Very few candidates can do this.

---

## 17.4 The Cross-Stage Questions

**"A user says the answer is wrong. What do you do?"**

Answer: "I run `incidentlens diagnose`. It reports each stage. I ask four questions in order:
1. Did stage 5 retrieve the correct document? If no, the failure is retrieval.
2. Was the correct passage inside the context window? If no, the failure is the ranking or the chunk size.
3. Did the model use the passage? If no, the failure is the prompt.
4. Is the answer supported by the passage? If no, the failure is the generation.
Each question has a metric. Therefore, I can name the failing stage. I do not guess."

**"How do you make the pipeline faster?"**

Answer: "Four methods, in this order:
1. A semantic cache. I calibrated the threshold to 0.65, because 0.92 could never hit. A repeat query goes from about 1.4 s to about 300 ms.
2. A smaller candidate count before the rerank.
3. A local embedding model. It removes the network delay.
4. Streaming. The user sees the first token early."

**"How do you make it cheaper?"**

Answer: "The local models do stages 3, 4, and 7. They have no cost. The provider does stages 8 and 9 only. The cache removes the repeat cost. The judge chain uses three models, because the Groq limit applies to each model and not to the account."

**"What is the weakest part of this project?"**

Answer: "The corpus. Four documents and 105 chunks. The retrieval metric is saturated at 1.000, so I cannot measure an improvement. The fix is in progress: 25 to 40 documents and 800 to 1,500 chunks. Until then, all precision comparisons are weak. That is the first phase of my plan."

---

## 17.5 The Failure-Mode Flashcards

For each failure: the symptom, the metric that moves, the cause, the fix.

| Failure | The metric that moves | The cause | The fix |
|---|---|---|---|
| Lost in the middle | Faithfulness falls | Long context | Put the best chunk first. Reduce the context. |
| A chunk cuts a sentence | Context Recall falls | The chunk size | Increase the overlap. Use recursive split. |
| The model does not know the term | Recall falls | The embedding model | Add BM25. Use a domain model. |
| The query and the document do not match | Recall falls | Query-document asymmetry | Rewrite the query. Use HyDE. |
| Precision is high, recall is low | Only Recall falls | The candidate count is too small | Increase k. |
| The answer is correct but not grounded | Faithfulness falls | A weak prompt | Require a citation for each claim. |
| The score drops after a change | All metrics fall | A measurement defect | Run a controlled A/B. Check the judge input. |
| Repeat questions cost money | Cost rises | No cache | Add a semantic cache. Calibrate the threshold. |
| The provider gives error 429 | `NOT MEASURED` | A quota limit | A fallback chain. A delay. A cache. |
| An injection changes the answer | Security | No input gate | A rule gate before the model. Zero model calls. |

---

## 17.6 The Ninety-Second Defense

Prepare these five answers. Each answer must be 90 seconds. Do not read from a file.

1. The 10 stages, in order, with one sentence for each.
2. The hybrid retrieval design and the reason for RRF.
3. The context precision correction and the A/B that isolated it.
4. The diagnose procedure and the four questions.
5. The weakest part of the project and the fix that is in progress.

> **NOTE:** Answer 3 is the differentiator. Rehearse answer 3 the most.

---

## 17.7 The Resume Line

> **IncidentLens — Retrieval Pipeline for Incident Diagnosis** · Python, LangChain, LangGraph, ChromaDB, FastAPI
> - Built a 10-stage RAG pipeline (ingest, chunk, embed, index, hybrid retrieve, RRF fuse, rerank, generate, evaluate) with a metric for each stage.
> - Built an ablation harness for chunking, retrieval, query rewriting, and reranking.
> - Found a +0.14 context-precision regression was a measurement artifact (the judge saw 200 characters of an 800-character chunk) with a controlled A/B; the evaluator reports `NOT MEASURED` instead of a false score during a provider outage.

---

## 🔗 Next

→ [14_pipeline_plan.md](./14_pipeline_plan.md) — Start Phase 1.
