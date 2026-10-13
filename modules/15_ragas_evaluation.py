"""
15_ragas_evaluation.py
======================
Phase 4: Production Observability — Automated RAG Quality Evaluation
IncidentLens 4-Metric RAG Scorecard:

EVALUATION APPROACH:
  Instead of using the Ragas library (which has known import compatibility issues with
  langchain-community VertexAI), we implement the 4 core metrics directly using Groq
  as a judge LLM. This gives us full control, zero dependency conflicts, and identical
  scoring semantics:

  1. Faithfulness:       Is every claim in the answer grounded in context? (Anti-hallucination)
  2. Answer Relevancy:   Does the answer address the user's question? (Anti-off-topic)
  3. Context Precision:  Are the retrieved chunks actually relevant to the question? (Anti-noise)
  4. Context Recall:     Does the context cover all ground-truth facts? (Anti-incompleteness)

MEASUREMENT FIXES:
  A. Context Precision showed the judge only the first 200 characters of each chunk. The
     chunks are 800 characters, so the judge was scoring section headings, not content.
     The judge now receives each chunk in full.
  B. The judge used to return 0.5 on ANY exception. A Groq rate limit (HTTP 429) therefore
     became a silent "partial" score, and a failed run looked like a mediocre one. The judge
     now retries with backoff and returns None on failure; failed judgments are excluded from
     the averages and counted, so a rate-limited run can no longer masquerade as a result.

WORKFLOW:
  Golden Dataset (10 Q&A pairs)
       |
  RAG Pipeline (ChromaDB dense retrieval + Groq LLM generation)
       |
  LLM-as-Judge (Groq evaluates each answer against ground truth)
       |
  4-Metric Scorecard printed to terminal
"""

import sys
import os
import json
import warnings
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output encoding across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import importlib.util

def _import_helper(filename: str, module_name: str):
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_chroma = _import_helper("05_vector_stores_chroma.py", "module_05_chroma")
get_incident_collection = mod_chroma.get_incident_collection


# =====================================================================
# 1. Load Golden Dataset
# =====================================================================
def load_golden_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# 2. RAG Runner (Dense-only for evaluation speed)
# =====================================================================
def run_rag_for_evaluation(
    questions: List[str],
    collection: Any,
    groq_api_key: str,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Runs each question through dense-only ChromaDB retrieval + Groq LLM.
    Returns a list of {question, answer, contexts} dicts for evaluation.
    """
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.0,
        max_tokens=512,
        api_key=groq_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are IncidentLens, an expert SRE post-mortem analyst. "
            "Answer the question using ONLY the provided context. "
            "Be precise and factual. Cite chunk sources where possible.\n\n"
            "CONTEXT:\n{context}"
        ),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    results = []
    for idx, question in enumerate(questions, start=1):
        print(f"  [{idx:02d}/{len(questions):02d}] Evaluating: \"{question[:65]}...\"")

        raw = collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = raw["documents"][0] if raw["documents"] else []
        context_str = "\n\n".join(docs)

        generation_failed = False
        try:
            answer = chain.invoke({"context": context_str, "question": question})
        except Exception as e:
            answer = f"[LLM Error: {e}]"
            generation_failed = True

        results.append({
            "question": question,
            "answer": answer,
            "contexts": docs,
            "generation_failed": generation_failed,
        })

        # Respect Groq free-tier TPM limits
        time.sleep(2.0)

    return results


# =====================================================================
# 3. LLM-as-Judge Evaluator (Direct Groq — No Ragas Dependency)
# =====================================================================
def _score_text(value: Optional[float]) -> str:
    """Format a judge score, or mark it clearly when the judge call failed."""
    return "n/a " if value is None else f"{value:.2f}"

class LLMJudgeEvaluator:
    """
    Implements the 4 core Ragas metrics using Groq as a judge LLM directly.
    Avoids the ragas library's broken import chain (vertexai dependency).
    Each metric is evaluated via a structured prompt that returns a score 0.0-1.0.
    """

    def __init__(self, groq_api_key: str):
        from langchain_groq import ChatGroq
        from langchain_core.output_parsers import StrOutputParser

        # Groq free-tier token limits are PER MODEL, so one saturated judge model must not
        # turn the whole scorecard into n/a. The historical judge stays FIRST for
        # comparability with earlier runs; the rest are failover targets.
        self.models: List[str] = [
            m.strip() for m in os.getenv(
                "INCIDENTLENS_JUDGE_MODELS",
                "qwen/qwen3.8-27b,allam-2-7b,openai/gpt-oss-20b",
            ).split(",") if m.strip()
        ]
        self.clients = {
            name: ChatGroq(model=name, temperature=0.0, max_tokens=200, api_key=groq_api_key)
            for name in self.models
        }
        self.parser = StrOutputParser()
        self.max_retries = 2
        self.failures: List[str] = []
        self.model_usage: Dict[str, int] = {name: 0 for name in self.models}

    def _ask_judge(self, system_prompt: str, user_prompt: str) -> Optional[float]:
        """
        Sends a judge prompt to Groq and extracts a 0.0-1.0 score, with retry AND failover.

        Returns None when EVERY model in the chain failed to answer with a number. The caller
        MUST exclude None from averages. The original version returned 0.5 on any exception,
        which silently converted rate-limit errors into fake "partial" scores.

        Failover exists because the free-tier token budget is per model: a saturated judge
        model hands the question to the next model in the chain instead of failing the metric.
        model_usage records who answered, and the scorecard prints that mix.
        """
        import re
        from langchain_core.prompts import ChatPromptTemplate

        # Content is injected as a VARIABLE, never as template text. Chunk text and LLM answers
        # can contain braces (JSON, code, stack traces), and LangChain parses these strings as
        # f-string templates, so a stray brace raises
        #   ValueError: Invalid format specifier in f-string template.
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_content}"),
            ("human", "{user_content}"),
        ])

        last_error = "unknown error"
        for model_index, model_name in enumerate(self.models):
            chain = prompt | self.clients[model_name] | self.parser
            for attempt in range(1, self.max_retries + 1):
                try:
                    raw = chain.invoke({"system_content": system_prompt,
                                        "user_content": user_prompt})
                    numbers = re.findall(r"(\d+\.\d+|\d+)", raw.strip())
                    if numbers:
                        self.model_usage[model_name] += 1
                        return min(max(float(numbers[0]), 0.0), 1.0)  # Clamp to 0-1
                    last_error = "judge reply contained no number"
                except Exception as e:
                    last_error = f"{type(e).__name__}: {str(e)[:110]}"

                if attempt < self.max_retries:
                    backoff = 4 * attempt  # 4s: enough for a per-minute window to refill
                    print(f"      [judge retry {attempt}/{self.max_retries - 1}] {model_name}: "
                          f"{last_error} -> waiting {backoff}s")
                    time.sleep(backoff)

            if model_index < len(self.models) - 1:
                print(f"      [judge failover] {model_name} exhausted -> "
                      f"{self.models[model_index + 1]}")

        self.failures.append(last_error)
        print(f"      [judge FAILED] all {len(self.models)} models: {last_error}")
        return None

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> Optional[float]:
        """Score 0-1: Is every claim in the answer supported by context?"""
        context_str = "\n---\n".join(contexts[:4])
        system = (
            "You are a strict factual accuracy judge. Your task is to evaluate if an answer "
            "is faithful to the provided context — meaning every claim in the answer can be "
            "traced back to a statement in the context.\n"
            "Score 1.0 if ALL claims are supported. Score 0.0 if the answer contains fabricated facts. "
            "Score 0.5 if some claims are supported but others are not.\n"
            "Respond with ONLY a single decimal number between 0.0 and 1.0."
        )
        user = f"CONTEXT:\n{context_str}\n\nANSWER:\n{answer}\n\nFaithfulness score:"
        time.sleep(2.5)  # pace the free tier: 4 judge calls per question
        return self._ask_judge(system, user)

    def evaluate_answer_relevancy(self, question: str, answer: str) -> Optional[float]:
        """Score 0-1: Does the answer address the question?"""
        system = (
            "You are a relevancy judge. Evaluate if the answer directly and completely "
            "addresses the question asked.\n"
            "Score 1.0 if the answer fully addresses the question. "
            "Score 0.5 if it partially addresses it. Score 0.0 if it is off-topic.\n"
            "Respond with ONLY a single decimal number between 0.0 and 1.0."
        )
        user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nRelevancy score:"
        time.sleep(2.5)  # pace the free tier: 4 judge calls per question
        return self._ask_judge(system, user)

    def evaluate_context_precision(self, question: str, contexts: List[str]) -> Optional[float]:
        """Score 0-1: How many retrieved chunks are relevant to the question?"""
        # Show each chunk in FULL to the judge. The earlier version passed only the first 200
        # characters, so the judge scored headings instead of content (Context Precision was
        # understated). Chunks are 800 characters, so 800 shows the whole chunk.
        chunks_display = "\n".join([
            f"[Chunk {i+1}]: {c[:800]}" for i, c in enumerate(contexts[:4])
        ])
        system = (
            "You are a retrieval quality judge. Evaluate what fraction of the retrieved chunks "
            "are actually relevant to answering the question.\n"
            "Score 1.0 if all chunks are relevant. Score 0.5 if half are relevant. "
            "Score 0.0 if none are relevant.\n"
            "Respond with ONLY a single decimal number between 0.0 and 1.0."
        )
        user = f"QUESTION:\n{question}\n\nRETRIEVED CHUNKS:\n{chunks_display}\n\nPrecision score:"
        time.sleep(2.5)  # pace the free tier: 4 judge calls per question
        return self._ask_judge(system, user)

    def evaluate_context_recall(self, ground_truth: str, contexts: List[str]) -> Optional[float]:
        """Score 0-1: Does the context cover all facts in the ground truth?"""
        context_str = "\n---\n".join(contexts[:4])
        system = (
            "You are a recall judge. Evaluate what fraction of the facts in the ground truth "
            "answer are also present in the retrieved context chunks.\n"
            "Score 1.0 if all ground truth facts are covered. Score 0.5 if about half are. "
            "Score 0.0 if the context misses most ground truth facts.\n"
            "Respond with ONLY a single decimal number between 0.0 and 1.0."
        )
        user = f"GROUND TRUTH:\n{ground_truth}\n\nRETRIEVED CONTEXT:\n{context_str}\n\nRecall score:"
        time.sleep(2.5)  # pace the free tier: 4 judge calls per question
        return self._ask_judge(system, user)

    def evaluate_all(
        self,
        rag_results: List[Dict[str, Any]],
        ground_truths: List[str],
    ) -> Dict[str, Any]:
        """Run all 4 metrics across all questions."""
        per_question = []

        for idx, (result, gt) in enumerate(zip(rag_results, ground_truths), 1):
            q = result["question"]
            a = result["answer"]
            c = result["contexts"]
            q_short = q[:55]

            print(f"\n  Judging [{idx:02d}/{len(rag_results):02d}]: \"{q_short}...\"")

            if result.get("generation_failed"):
                # Do not score an error string as if it were an answer. Judging it would report
                # a quality collapse that did not happen.
                faith = None
                rel = None
                print("    [generation FAILED for this question] Faithfulness and Answer Relevancy")
                print("    are NOT MEASURED here. Scoring an error message would be meaningless.")
            else:
                faith = self.evaluate_faithfulness(a, c)
                print(f"    Faithfulness:       {_score_text(faith)}")

                rel   = self.evaluate_answer_relevancy(q, a)
                print(f"    Answer Relevancy:   {_score_text(rel)}")

            prec  = self.evaluate_context_precision(q, c)
            print(f"    Context Precision:  {_score_text(prec)}")

            rec   = self.evaluate_context_recall(gt, c)
            print(f"    Context Recall:     {_score_text(rec)}")

            per_question.append({
                "question": q,
                "faithfulness": faith,
                "answer_relevancy": rel,
                "context_precision": prec,
                "context_recall": rec,
            })

        # Compute averages. Failed judge calls (None) are EXCLUDED, never counted as 0.5.
        def _average(metric: str) -> float:
            values = [r[metric] for r in per_question if r[metric] is not None]
            return sum(values) / len(values) if values else 0.0

        return {
            "faithfulness":      _average("faithfulness"),
            "answer_relevancy":  _average("answer_relevancy"),
            "context_precision": _average("context_precision"),
            "context_recall":    _average("context_recall"),
            "per_question":      per_question,
            "judge_failures":    len(self.failures),
            "judge_model_usage": dict(self.model_usage),
            "answer_metrics_samples": sum(
                1 for r in per_question if r["faithfulness"] is not None),
        }


# =====================================================================
# 4. Print Scorecard
# =====================================================================
def print_scorecard(scores: Dict[str, Any], target_thresholds: Dict[str, float]):
    print("\n" + "=" * 80)
    print(" 📊 IncidentLens — RAG 4-Metric Evaluation Scorecard")
    print("=" * 80)

    header = f"{'Metric':<22} | {'Score':>8} | {'Target':>8} | {'Status':<8} | Bar"
    print(header)
    print("-" * 78)

    metric_display = {
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
    }

    # A metric with zero samples was NOT measured. Printing 0.000 for it would read as a
    # catastrophic failure instead of a missing measurement.
    samples = scores.get("answer_metrics_samples")
    answer_metrics = {"faithfulness", "answer_relevancy"}

    for key, display in metric_display.items():
        target = target_thresholds.get(key, 0.85)
        if key in answer_metrics and samples == 0:
            print(f"{display:<22} | {'n/a':>7} | {target:>7.2f} | {'NOT MEASURED':<12} | "
                  f"(no generated answer to judge)")
            continue
        score     = scores.get(key, 0.0)
        passed    = score >= target
        status    = "PASS" if passed else "FAIL"
        bar_len   = int(score * 30)
        bar       = "#" * bar_len + "." * (30 - bar_len)
        print(f"{display:<22} | {score:>7.3f} | {target:>7.2f} | {status:<8} | {bar}")

    print("-" * 78)
    all_pass = all(
        scores.get(k, 0.0) >= v for k, v in target_thresholds.items()
    )
    if samples == 0:
        overall = ("NOT FULLY MEASURED - the provider produced no answers, so Faithfulness and "
                   "Answer Relevancy have no data. Precision and Recall remain valid.")
    elif all_pass:
        overall = "ALL METRICS PASS - Ready for Phase 5"
    else:
        overall = "Some metrics below target - review retrieval quality"
    print(f"\n  Overall: {overall}")

    failures = scores.get("judge_failures", 0)
    if failures:
        print(f"\n  *** {failures} judge call(s) FAILED and were EXCLUDED from the averages. ***")
        print("  *** This scorecard is INCOMPLETE. Re-run it before trusting the numbers. ***")

    samples = scores.get("answer_metrics_samples")
    if samples is not None and samples < len(scores.get("per_question", [])):
        print(f"\n  NOTE: Faithfulness and Answer Relevancy were averaged over {samples} of "
              f"{len(scores['per_question'])} questions. The rest had no generated answer to judge.")

    usage = {k: v for k, v in (scores.get("judge_model_usage") or {}).items() if v}
    if usage:
        print("\n  Judge calls served by model: "
              + ", ".join(f"{name} = {count}" for name, count in usage.items()))
        if len(usage) > 1:
            print("  NOTE: more than one judge model answered, because the free-tier limits are")
            print("        PER MODEL. A mixed-judge scorecard is comparable only within itself.")

    # Per-question breakdown
    if "per_question" in scores:
        print("\n" + "-" * 80)
        print(" Per-Question Score Breakdown:")
        print("-" * 80)
        for row in scores["per_question"]:
            q_short = row["question"][:55]
            print(
                f"  Q: {q_short:<55} "
                f"F={_score_text(row.get('faithfulness'))} "
                f"R={_score_text(row.get('answer_relevancy'))} "
                f"P={_score_text(row.get('context_precision'))} "
                f"C={_score_text(row.get('context_recall'))}"
            )

    print("\n" + "=" * 80)
    if samples == 0:
        print(" Module 15 ran: PARTIAL SCORECARD (2 of 4 metrics had no data).")
        print(" Re-run when the provider has quota to measure Faithfulness and Answer Relevancy.")
    else:
        print(" Module 15 Complete: Automated RAG Evaluation Verified!")
    print(" Next: Module 16 - Langfuse Open-Source Observability Tracing.")
    print("=" * 80)


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 80)
    print(" IncidentLens — Module 15: Automated RAG Evaluation (4-Metric Scorecard)")
    print("=" * 80)

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        print("GROQ_API_KEY not set. Cannot run evaluation.")
        return

    # Load golden dataset
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "eval_golden_dataset.json"
    print(f"\n[1] Loading Golden Evaluation Dataset from: {dataset_path.name}")
    test_cases = load_golden_dataset(str(dataset_path))
    print(f"    Loaded {len(test_cases)} curated Q&A pairs")

    categories = {}
    for tc in test_cases:
        cat = tc.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"    Coverage: {', '.join(f'{v} {k}' for k, v in categories.items())} questions")

    questions     = [tc["question"]     for tc in test_cases]
    ground_truths = [tc["ground_truth"] for tc in test_cases]

    # Run RAG for each question
    collection = get_incident_collection()
    print(f"\n[2] Running RAG Pipeline on {len(questions)} Questions...")
    rag_results = run_rag_for_evaluation(
        questions=questions,
        collection=collection,
        groq_api_key=groq_api_key,
        top_k=4,
    )
    failed_answers = sum(1 for r in rag_results if r.get("generation_failed"))
    print(f"    Generated {len(rag_results) - failed_answers}/{len(rag_results)} answers "
          f"from ChromaDB + Groq LLM")
    if failed_answers:
        print(f"    *** {failed_answers} generation failure(s) - the provider was out of quota. ***")
        print("    *** Faithfulness and Answer Relevancy judge the ANSWER, so they cannot be ***")
        print("    *** measured this run and are reported as not measured. Context Precision ***")
        print("    *** and Context Recall are independent of the answer and REMAIN VALID.    ***")

    # Run LLM-as-Judge evaluation
    print(f"\n[3] Running LLM-as-Judge Evaluation (Groq as judge)...")
    print(f"    (Each question requires 4 judge calls — this takes ~3-5 minutes on free tier)")
    evaluator = LLMJudgeEvaluator(groq_api_key)
    scores = evaluator.evaluate_all(rag_results, ground_truths)

    # Print scorecard
    thresholds = {
        "faithfulness":      0.85,
        "answer_relevancy":  0.85,
        "context_precision": 0.80,
        "context_recall":    0.80,
    }
    print_scorecard(scores, thresholds)


if __name__ == "__main__":
    main()
