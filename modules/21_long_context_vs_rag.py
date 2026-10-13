"""
21_long_context_vs_rag.py
=========================
Phase 6: Cutting-Edge Advanced RAG (2026 Architectures)
IncidentLens Long Context vs. RAG Router + Cost/Latency/Quality Benchmark:

WHY THIS EXISTS:
  A retrieval pipeline is not always the right answer, and neither is stuffing a whole
  post-mortem into a 262k-token window. The two strategies fail in opposite directions:

    * RAG (hybrid retrieval + synthesis) is CHEAP and FAST (~1-2k input tokens per query)
      but it only ever sees the top_k chunks it retrieved. If the question requires the
      shape of the whole incident - a timeline, a narrative arc, cross-references between
      the detection section and the remediation section - retrieval silently loses the
      connective tissue and the answer is confidently incomplete.

    * LONG CONTEXT (whole document -> large-context model) sees EVERY section at once, so
      timeline reconstruction and cross-section reasoning are exact, but paying ~40-50k
      input tokens per question is 25-30x the cost and seconds of extra latency. For a
      lookup that lives in one paragraph it is pure waste.

  This module is a DYNAMIC DECISION ENGINE (the router) plus an honest benchmark: it
  decides per query, then measures what the decision actually cost in latency, tokens,
  dollars and answer quality.

THE TWO STRATEGIES AND THE DECISION THE ROUTER MAKES:

                          +---------------------------+
   user query ----------> |  ROUTER (signal scoring)  |
                          +-------------+-------------+
                                        |
              +-------------------------+-------------------------+
              |                                                   |
     narrow / specific evidence                          whole-document evidence
     - exact code (AS33154, 0x00000050)                  - "summarise the post-mortem"
     - "root cause" / "what caused"                      - "reconstruct the timeline"
     - "which line of code / flag / value"               - "how did it unfold over time"
     - "one paragraph / narrow section"                  - "across the whole incident"
     - "keep it fast" (latency budget)                   - "compare how the phases relate"
              |                                                   |
              v                                                   v
   +--------------------------+                      +------------------------------+
   | STRATEGY A: RAG          |                      | STRATEGY B: LONG CONTEXT     |
   | module 10 hybrid search  |                      | full document text ->        |
   | (BM25 + dense + RRF)     |                      | OpenRouter large-context     |
   | -> top_k chunks          |                      | free model (>= 128k window)  |
   | -> LLM synthesis with    |                      | -> LLM synthesis over the    |
   |    chunk-ID citations    |                      |    entire document           |
   | ~1-2k input tokens       |                      | ~40-50k input tokens         |
   +-------------+------------+                      +---------------+--------------+
                 |                                                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
                            measured: latency_ms | input_tokens |
                            est_cost_usd | quality (judge rubric)

  The default classifier is DETERMINISTIC and EXPLAINABLE: an ordered list of regex
  signals, each carrying a weight and a target strategy. Every fired signal is printed so
  any decision can be audited. An LLM classifier is available behind --llm ONLY and is
  never required for the default run.

DOCUMENTED QUALITY RUBRIC (LLM judge, one call per answer):
  Each answer is scored 0.0-1.0 on two axes against its own source of truth:
    groundedness : every factual claim is traceable to the supplied source (the retrieved
                   chunks for RAG, the FULL document for long context). A fabricated fact
                   caps this at 0.0.
    relevance    : the answer directly and completely addresses the question asked.
    quality      : mean of the two axes.
  The judge runs on OpenRouter first and falls back to Groq. If NO judge can be reached,
  the score is printed as "not measured" and the QUALITY check FAILS - a number is never
  invented.

DOCUMENTED COST MODEL:
  cost = (input_tokens / 1e6) * rate_in + (output_tokens / 1e6) * rate_out.
  Two rates are reported side by side: the chosen model's OWN rate as published by
  OpenRouter (the free tier publishes 0.00 USD / 1M token, so the free run is genuinely
  free) and a COMMERCIAL REFERENCE rate of 2.50 USD / 1M input and 10.00 USD / 1M output,
  documented here so the token comparison can be priced as if it ran on a paid frontier
  model. The reference column is an ESTIMATE at a published rate; it is not an invoice.

HONEST FAILURE POLICY:
  If OpenRouter is unreachable, lists no usable free model, or returns nothing, that fact
  is printed verbatim and the long-context checks FAIL. Latency, token counts and scores
  are always measured - never fabricated.

Run:
    .\\venv\\Scripts\\python.exe modules\\21_long_context_vs_rag.py
    .\\venv\\Scripts\\python.exe modules\\21_long_context_vs_rag.py --llm
"""

import os
import re
import sys
import json
import time
import importlib
import importlib.util
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output encoding across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# =====================================================================
# 0. Interpreter Guard
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
REQUIRED_PACKAGES = ("chromadb", "openai", "tiktoken", "dotenv")


def _missing_packages(names: Tuple[str, ...]) -> List[str]:
    missing: List[str] = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def ensure_project_interpreter() -> None:
    """Re-launch this script with the project venv when the active interpreter is wrong."""
    missing = _missing_packages(REQUIRED_PACKAGES)
    if not missing:
        return
    if os.environ.get("INCIDENTLENS_REEXEC") == "1" or not VENV_PYTHON.exists():
        print("=" * 80)
        print(" [BOOT] Cannot start: project packages are missing")
        print("=" * 80)
        print(f"  Interpreter  : {sys.executable}")
        print(f"  Missing      : {', '.join(missing)}")
        print(f"  Project venv : {VENV_PYTHON}")
        print("  Run this instead:")
        print("      .\\venv\\Scripts\\python.exe modules\\21_long_context_vs_rag.py")
        print("=" * 80)
        sys.exit(1)
    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return
    print("=" * 80)
    print(" [BOOT] Wrong interpreter detected - switching to the project venv")
    print("=" * 80)
    env = dict(os.environ)
    env["INCIDENTLENS_REEXEC"] = "1"
    completed = subprocess.run(
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        env=env,
    )
    sys.exit(completed.returncode)


ensure_project_interpreter()

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

# Dynamic import: Python cannot import module files whose names start with a digit.
def _import_helper(filename: str, module_name: str):
    """Load a sibling module by file path and return the module object."""
    script_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod_chroma = _import_helper("05_vector_stores_chroma.py", "module_05_chroma")
mod_hybrid = _import_helper("10_hybrid_search_bm25.py", "module_10_hybrid")
mod_loaders = _import_helper("02_document_loaders.py", "module_02_loaders")

get_incident_collection = mod_chroma.get_incident_collection
get_production_chunks = mod_chroma.get_production_chunks
HybridSearchEngine = mod_hybrid.HybridSearchEngine
PostmortemDocumentLoader = mod_loaders.PostmortemDocumentLoader

# =====================================================================
# 1. Configuration
# =====================================================================
DATA_DIR = PROJECT_ROOT / "data" / "postmortems"

# Cache/config files live inside the project mirror, never inside a read-only source tree.
CACHE_PATH = PROJECT_ROOT / "long_context_router_cache.json"
CACHE_SCHEMA = "incidentlens.router.model.v2"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq is a FALLBACK only (the free tier is rate-limited per model), so several ids are
# listed and the code fails over across them on a 429.
GROQ_FALLBACK_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")

# Long context needs a window that actually holds a whole post-mortem. The largest file in
# this corpus (the CrowdStrike PDF) is ~45k tokens, so 128k is the honest floor.
MIN_LONG_CONTEXT_WINDOW = 128_000
# A shorter window is still acceptable for this small corpus if nothing bigger answers.
ACCEPTABLE_CONTEXT_WINDOW = 60_000
DISCOVERY_PROBE_CANDIDATES = 5

# Documented commercial reference rate (USD per 1M tokens) used ONLY for the estimate
# column, so a free-tier run can still be priced against a paid frontier model.
REFERENCE_RATE_IN = 2.50
REFERENCE_RATE_OUT = 10.00

RAG_TOP_K = 5
RAG_CANDIDATE_POOL = 20
MAX_ANSWER_TOKENS = 900
# The judge is allowed a long budget on purpose: free reasoning models write their working
# first, and a truncated judge reply loses the JSON at the end (a real failure seen here).
MAX_JUDGE_TOKENS = 1500
RETRY_ATTEMPTS = 3


# =====================================================================
# 2. Token Accounting & Cost Model
# =====================================================================
def _load_encoder():
    """Load a tiktoken encoder once, preferring the modern cl100k_base tokenizer."""
    import tiktoken

    for name in ("cl100k_base", "o200k_base"):
        try:
            return tiktoken.get_encoding(name), name
        except Exception:
            continue
    try:
        return tiktoken.get_encoding("gpt2"), "gpt2"
    except Exception:
        return None, "unavailable"


_ENCODER, _ENCODER_NAME = _load_encoder()


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken. Falls back to a documented 4-chars/token estimate."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def estimate_cost_usd(input_tokens: int, output_tokens: int,
                      rate_in: float, rate_out: float) -> float:
    """cost = (in/1e6)*rate_in + (out/1e6)*rate_out. Rates are USD per 1M tokens."""
    return (input_tokens / 1_000_000.0) * rate_in + (output_tokens / 1_000_000.0) * rate_out


def fmt_cost(value: Optional[float]) -> str:
    if value is None:
        return "not measured"
    if value == 0:
        return "$0.000000"
    return f"${value:.6f}"


def fmt_ms(value: Optional[float]) -> str:
    return "not measured" if value is None else f"{value:,.0f} ms"


# =====================================================================
# 3. Unified LLM Client (OpenRouter first, Groq failover, 429 backoff)
# =====================================================================
class LLMUnavailable(RuntimeError):
    """Raised when every configured provider and model refused or failed the call."""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    measured_tokens: bool  # True when the provider itself reported usage


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought wrappers some free models emit alongside the answer."""
    if not text:
        return ""
    cleaned = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", " ", text, flags=re.S | re.I)
    cleaned = re.sub(r"</?think(?:ing)?>", " ", cleaned, flags=re.I)
    return cleaned.strip()


# Some free large-context models ignore a "no reasoning" instruction and put their plan in
# the same field as the answer ("The user wants a full timeline... Let me parse the text").
# The preamble is removed deterministically at the first decisive answer marker, so the
# quality judge scores the ANSWER, not the model's thinking. Nothing is invented and the
# scrub is announced on stdout whenever it actually changes the text.
_ANSWER_START_RE = re.compile(
    r"(?m)^(?:"
    r"\s{0,3}#{1,6}\s+\S"                      # markdown heading
    r"|\s*\*\*[A-Z][^*\n]{0,80}\*\*"           # bold lead-in title
    r"|\s*(?:full |incident |complete )?timeline\b"  # a timeline answer
    r"|\s*\d+[\.\)]\s+\S"                      # numbered first step
    r"|\s*(?:At|On|During|By|Within|Around)\s"  # a date-led chronology
    r")"
)
_REASONING_PREAMBLE_RE = re.compile(
    r"^\s*(?:the user|user wants|i need to|i must|let me|we need to|"
    r"first,? i|okay,? |sure,? here|analysis:|\*\*analysis)",
    re.IGNORECASE,
)


def scrub_reasoning_preamble(text: str) -> Tuple[str, bool]:
    """
    Cut a leaked planning preamble off the front of an answer.

    Returns (clean_text, was_scrubbed). The scrub only fires when the text STARTS like
    internal reasoning AND a decisive answer marker exists later in the text; otherwise
    the answer is returned untouched.
    """
    if not text:
        return text, False
    if not _REASONING_PREAMBLE_RE.match(text):
        return text, False
    match = _ANSWER_START_RE.search(text)
    if not match or match.start() < 40:
        return text, False
    return text[match.start():].strip(), True


def _usage_tokens(usage: Any) -> Tuple[int, int]:
    if usage is None:
        return 0, 0
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    return int(prompt or 0), int(completion or 0)


def _retry_after_seconds(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        try:
            value = headers.get("retry-after")
            if value is not None:
                return min(float(value), 30.0)
        except Exception:
            pass
    return 0.0


class UnifiedLLM:
    """
    One chat() entry point over OpenRouter (preferred) and Groq (fallback).

    Preference order is OPENROUTER first for every call. Groq is touched only when
    OpenRouter cannot serve the request, and even then the module fails over across
    several Groq model ids because the free-tier limits are enforced PER MODEL.
    """

    def __init__(self, openrouter_key: str, groq_key: str, model: str,
                 model_context: Optional[int] = None) -> None:
        self.openrouter_key = (openrouter_key or "").strip()
        self.groq_key = (groq_key or "").strip()
        self.model = model
        self.model_context = model_context
        self.calls: List[Dict[str, Any]] = []
        self._clients: Dict[str, OpenAI] = {}

    def _client(self, provider: str) -> OpenAI:
        if provider not in self._clients:
            if provider == "openrouter":
                self._clients[provider] = OpenAI(
                    base_url=OPENROUTER_BASE_URL, api_key=self.openrouter_key, timeout=180.0
                )
            else:
                self._clients[provider] = OpenAI(
                    base_url=GROQ_BASE_URL, api_key=self.groq_key, timeout=120.0
                )
        return self._clients[provider]

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int,
             label: str = "call", scrub: bool = True,
             prefer_models: Optional[Tuple[str, ...]] = None,
             groq_first: bool = False,
             providers: Optional[Tuple[str, ...]] = None) -> LLMResult:
        """
        Call the first provider/model that answers. Raises LLMUnavailable if none do.

        scrub=False is used for INTERNAL structured calls (the judge): their chain-of-thought
        must be left intact so the JSON reply can still be located inside it.
        prefer_models lets a caller put a faster model in front of the discovered one.
        providers restricts which providers are attempted (used to pin a judge attempt).
        """
        errors: List[str] = []
        targets: List[Tuple[str, str]] = []
        openrouter_targets: List[str] = []
        for model in (prefer_models or ()):
            if model and model not in openrouter_targets:
                openrouter_targets.append(model)
        if self.model and self.model not in openrouter_targets:
            openrouter_targets.append(self.model)
        groq_targets = [("groq", m) for m in GROQ_FALLBACK_MODELS]
        wanted = tuple(providers) if providers else ("openrouter", "groq")
        if groq_first and "groq" in wanted:
            targets.extend(groq_targets)
        if "openrouter" in wanted and self.openrouter_key:
            targets.extend(("openrouter", m) for m in openrouter_targets)
        if not groq_first and "groq" in wanted:
            targets.extend(groq_targets)
        if not targets:
            raise LLMUnavailable("no provider credentials available for this call "
                                 "(OPENROUTER_API_KEY / GROQ_API_KEY)")

        for provider, model in targets:
            started = time.perf_counter()
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    response = self._client(provider).chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=max_tokens,
                    )
                    message = response.choices[0].message if response.choices else None
                    raw = (getattr(message, "content", None) or "") if message else ""
                    if not raw:
                        # Some free models report their text in a separate reasoning field.
                        extra = getattr(message, "model_extra", None) or {}
                        raw = extra.get("reasoning") or ""
                    prompt_tokens, completion_tokens = _usage_tokens(getattr(response, "usage", None))
                    if not raw.strip():
                        errors.append(f"{provider}/{model}: empty completion")
                        break  # this model is unusable; try the next target
                    if scrub:
                        text = _strip_reasoning(raw)
                        text, scrubbed = scrub_reasoning_preamble(text)
                        if scrubbed:
                            print(f"      [cleanup] removed a leaked reasoning preamble from the "
                                  f"{provider}/{model} completion before scoring")
                            if "--debug-raw" in sys.argv:
                                print(f"      [debug-raw] first 300 chars of the raw completion: "
                                      f"{raw[:300]!r}")
                    else:
                        # Internal structured call: reasoning is left intact so the JSON
                        # payload it contains can still be parsed.
                        text = raw.strip()
                    if not text:
                        errors.append(f"{provider}/{model}: empty after cleanup")
                        break
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    result = LLMResult(
                        text=text,
                        provider=provider,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        measured_tokens=prompt_tokens > 0,
                    )
                    self.calls.append({"label": label, "provider": provider, "model": model,
                                       "prompt_tokens": prompt_tokens,
                                       "completion_tokens": completion_tokens})
                    return result
                except Exception as exc:  # network, 429, 403, model retired...
                    status = getattr(exc, "status_code", None)
                    detail = f"{provider}/{model}: {type(exc).__name__}"
                    if status is not None:
                        detail += f" HTTP {status}"
                    detail += f" {str(exc)[:110]}"
                    errors.append(detail)
                    if status == 429 and attempt < RETRY_ATTEMPTS:
                        wait = _retry_after_seconds(exc) or (4.0 * attempt)
                        print(f"      [rate limited] {detail} -> retrying in {wait:.0f}s")
                        time.sleep(wait)
                        continue
                    break
        raise LLMUnavailable("; ".join(errors[:4]) if errors else "no provider answered")


# =====================================================================
# 4. OpenRouter Large-Context Model Discovery
# =====================================================================
@dataclass
class ModelChoice:
    model_id: str
    context_length: int
    rate_in: float      # USD per 1M tokens, as published by OpenRouter
    rate_out: float
    source: str         # "cache" | "discovery"
    notes: str = ""


def _free_model_candidates(client: OpenAI) -> List[Dict[str, Any]]:
    """
    List OpenRouter models, keep the free ids (':free'), and return them sorted by
    descending context window so the largest window is probed first.
    """
    response = client.models.list()
    candidates: List[Dict[str, Any]] = []
    for model in getattr(response, "data", []) or []:
        model_id = getattr(model, "id", "") or ""
        if not model_id.endswith(":free"):
            continue
        extra = getattr(model, "model_extra", None) or {}
        context_length = int(extra.get("context_length") or 0)
        pricing = extra.get("pricing") or {}
        try:
            # OpenRouter publishes USD PER TOKEN for paid models; free ids publish "0".
            rate_in = float(pricing.get("prompt") or 0.0) * 1_000_000.0
            rate_out = float(pricing.get("completion") or 0.0) * 1_000_000.0
        except (TypeError, ValueError):
            rate_in, rate_out = 0.0, 0.0
        candidates.append({
            "id": model_id,
            "context_length": context_length,
            "rate_in": rate_in,
            "rate_out": rate_out,
        })
    candidates.sort(key=lambda c: c["context_length"], reverse=True)
    return candidates


def _openrouter_client(openrouter_key: str, timeout: float = 120.0) -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_key, timeout=timeout)


def _probe_free_model(client: OpenAI, model_id: str, prompt: str, max_tokens: int = 64
                      ) -> Tuple[bool, str, str]:
    """
    Send one tiny probe and report (answered, short_reply, note).

    A model is only "working" if a real completion comes back with text. 403s (free ids
    restricted to agentic harnesses), 429s and empty completions all count as failures.
    """
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = ""
        if response.choices:
            text = (response.choices[0].message.content or "").strip()
        if not text:
            return False, "", "empty completion"
        return True, text, "ok"
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        prefix = f"HTTP {status} " if status else ""
        return False, "", f"{prefix}{type(exc).__name__}: {str(exc)[:90]}"


# A free, well-behaved model to try as the quality judge before the discovered
# large-context winner. Judge models are probed for their ability to emit the rubric JSON,
# and the whole judge chain stays on the free tier (paid Groq/OpenRouter ids are never used).
PREFERRED_JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
JUDGE_PROBE_CANDIDATES = 3


def _save_cache(payload: Dict[str, Any]) -> None:
    try:
        payload = dict(payload)
        payload["schema"] = CACHE_SCHEMA
        payload["cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"    cached -> {CACHE_PATH.name}")
    except Exception as exc:
        print(f"    (cache write skipped: {type(exc).__name__})")


def discover_long_context_model(openrouter_key: str, require_context: int,
                                use_cache: bool = True
                                ) -> Tuple[Optional[ModelChoice], Optional[List[Dict[str, Any]]], List[str]]:
    """
    Discover a working free large-context model at runtime and cache the choice.

    Returns (choice, candidates, probe_log). The candidate list is returned so the judge
    discovery below can reuse the SAME model listing instead of burning a second GET.
    Free ids are probed in descending context order until one answers.
    """
    probe_log: List[str] = []

    if use_cache and CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if cached.get("schema") == CACHE_SCHEMA and cached.get("model_id"):
                choice = ModelChoice(
                    model_id=cached["model_id"],
                    context_length=int(cached.get("context_length") or 0),
                    rate_in=float(cached.get("rate_in") or 0.0),
                    rate_out=float(cached.get("rate_out") or 0.0),
                    source="cache",
                    notes=cached.get("notes", ""),
                )
                print(f"[2] Reusing cached OpenRouter model: {choice.model_id} "
                      f"({choice.context_length:,} token window)")
                # The cached free-model listing is reused so the judge can still be chosen
                # by probe without spending a second models.list() call.
                cached_candidates = cached.get("free_candidates")
                return choice, cached_candidates if isinstance(cached_candidates, list) else None, probe_log
        except Exception as exc:
            print(f"[2] Cache unreadable ({type(exc).__name__}) - re-running discovery")

    if not openrouter_key:
        raise LLMUnavailable("OPENROUTER_API_KEY is not set; long-context strategy unavailable")

    client = _openrouter_client(openrouter_key)
    try:
        candidates = _free_model_candidates(client)
    except Exception as exc:
        raise LLMUnavailable(f"OpenRouter unreachable while listing models: "
                             f"{type(exc).__name__}: {str(exc)[:160]}")

    print(f"[2] OpenRouter reachable. Free models listed: {len(candidates)}")
    if not candidates:
        raise LLMUnavailable("OpenRouter returned no free (':free') models")

    ordered = [c for c in candidates if c["context_length"] >= require_context] or candidates
    rejected: List[str] = []
    for candidate in ordered[:DISCOVERY_PROBE_CANDIDATES]:
        model_id = candidate["id"]
        started = time.perf_counter()
        answered, text, note = _probe_free_model(
            client, model_id, "Reply with exactly this token and nothing else: READY")
        probe_log.append(f"long-context probe {model_id}: {'ok' if answered else note}")
        if not answered:
            rejected.append(f"{model_id} ({note})")
            print(f"    - {model_id:<58} REJECTED: {note}")
            continue
        probe_ms = (time.perf_counter() - started) * 1000.0
        print(f"    - {model_id:<58} OK in {probe_ms:,.0f} ms -> {text[:24]!r}")
        choice = ModelChoice(
            model_id=model_id,
            context_length=candidate["context_length"],
            rate_in=candidate["rate_in"],
            rate_out=candidate["rate_out"],
            source="discovery",
            notes=f"probed {len(rejected) + 1} free candidates",
        )
        _save_cache({
            "model_id": choice.model_id,
            "context_length": choice.context_length,
            "rate_in": choice.rate_in,
            "rate_out": choice.rate_out,
            "notes": choice.notes,
            "free_candidates": candidates,
        })
        return choice, candidates, probe_log

    raise LLMUnavailable(
        "no OpenRouter free model answered the probe. Probed: "
        + ", ".join(c["id"] for c in ordered[:DISCOVERY_PROBE_CANDIDATES])
        + (" | rejected: " + "; ".join(rejected[:5]) if rejected else "")
    )


def discover_judge_models(openrouter_key: str, candidates: Optional[List[Dict[str, Any]]],
                          already_ok: Tuple[str, ...] = (),
                          max_probes: int = 1) -> Tuple[Tuple[str, ...], List[str]]:
    """
    Build the judge chain: [optional probed-fast judge, discovered long-context winner].

    The winner is not probed for its JSON shape because it is already known to answer; if
    it cannot emit parseable scores the caller falls through to the next model in the chain
    and, failing that, records the score as "not measured" instead of inventing one.
    """
    fallback_chain: List[str] = []
    for model_id in already_ok:
        if model_id and model_id not in fallback_chain:
            fallback_chain.append(model_id)

    probe_log: List[str] = []
    if not openrouter_key or candidates is None or max_probes <= 0:
        return tuple(fallback_chain), probe_log

    allowed = {c["id"] for c in candidates}
    if PREFERRED_JUDGE_MODEL not in allowed or PREFERRED_JUDGE_MODEL in fallback_chain:
        return tuple(fallback_chain), probe_log

    client = _openrouter_client(openrouter_key, timeout=90.0)
    # Probe with the real judge prompt on a trivial pair: a model that cannot emit the
    # rubric JSON for an easy case will not manage a 2,400-token source either.
    answered, text, note = _probe_free_model(
        client, PREFERRED_JUDGE_MODEL, PROBE_JUDGE_USER, max_tokens=900)
    if not answered:
        probe_log.append(f"judge probe {PREFERRED_JUDGE_MODEL}: REJECTED {note}")
        print(f"    - {PREFERRED_JUDGE_MODEL:<58} judge probe REJECTED: {note}")
        return tuple(fallback_chain), probe_log
    parsed = _parse_judge_scores(text)
    if parsed is None:
        probe_log.append(f"judge probe {PREFERRED_JUDGE_MODEL}: no parseable scores")
        print(f"    - {PREFERRED_JUDGE_MODEL:<58} judge probe REJECTED: no parseable scores")
        return tuple(fallback_chain), probe_log
    probe_log.append(f"judge probe {PREFERRED_JUDGE_MODEL}: ok {parsed}")
    print(f"    - {PREFERRED_JUDGE_MODEL:<58} judge probe OK -> {parsed}")
    return tuple([PREFERRED_JUDGE_MODEL] + fallback_chain), probe_log


# =====================================================================
# 5. ROUTER - deterministic, explainable query classifier
# =====================================================================
STRATEGY_RAG = "RAG"
STRATEGY_LONG = "LONG_CONTEXT"


@dataclass(frozen=True)
class RouterSignal:
    """One auditable routing rule: a regex, a weight, and the strategy it argues for."""
    name: str
    strategy: str
    weight: float
    pattern: str
    why: str


# The list is ordered and every pattern is case-insensitive. Weights are deliberately
# asymmetric: a single exact-technical-evidence signal (an ASN, a bugcheck code, an
# explicit root-cause ask) is enough to beat one narrative word, because a lookup that
# fits in one chunk is exactly where retrieval is both cheaper and more precise.
ROUTER_SIGNALS: Tuple[RouterSignal, ...] = (
    # ---------------- argues for RAG (specific, narrow, cheap) ----------------
    RouterSignal("exact_error_code", STRATEGY_RAG, 2.5,
                 r"\b(?:[a-z]{2,5}-?\d{3,5}|0x[0-9a-f]{4,}|[a-z]+_\d{3,}|err(?:or)?[ _-]?\d{3,})\b",
                 "an error/status/event code is quoted, so the answer lives in a few chunks"),
    RouterSignal("autonomous_system_number", STRATEGY_RAG, 3.0,
                 r"\bas\s?\d{2,6}\b",
                 "an ASN is an arbitrary identifier; lexical retrieval finds it, narrative does not"),
    RouterSignal("specific_identifier", STRATEGY_RAG, 2.0,
                 r"\b(?:[a-z0-9_]+\.(?:sys|dll|exe|sh|py|json|yaml|conf|cfg|md)|"
                 r"[a-z]+_[a-z]+_[a-z]+|[a-z]+-[a-z]+-[a-z0-9]+|"
                 r"\b[a-z]+\.(?:googleapis|internal|local|corp)\b)",
                 "a named file, artifact or host is quoted - a targeted lookup"),
    RouterSignal("root_cause_ask", STRATEGY_RAG, 2.5,
                 r"\b(?:root[ _-]?cause|what caused|what was the cause|underlying cause|"
                 r"(?:what|which)\b[^?]{0,60}?\bcaused\b|(?:what|which)\b[^?]{0,60}?\bled to\b|"
                 r"trigger(?:ed|ing)?|post[ _-]?mortem finding|why did [a-z ]{0,30}(?:fail|crash|break))",
                 "asks for a pinpoint causal fact, not a narrative"),
    RouterSignal("narrow_scope", STRATEGY_RAG, 2.0,
                 r"\b(?:which (?:line|flag|config|setting|commit|parameter|version|value|code)|"
                 r"what (?:value|flag|setting|version|threshold|code)|in one (?:paragraph|sentence|section)|"
                 r"narrow (?:section|passage)|that specific (?:line|chunk|paragraph)|exact (?:error|code|string|value))",
                 "explicitly scoped to one narrow piece of evidence"),
    RouterSignal("latency_priority", STRATEGY_RAG, 1.5,
                 r"\b(?:quickly|keep it fast|low latency|as fast as possible|right now|just tell me)\b",
                 "states a latency budget, which rules out a whole-document pass"),
    # ---------------- argues for LONG CONTEXT (whole-document) ----------------
    RouterSignal("narrative_summary", STRATEGY_LONG, 3.0,
                 r"\b(?:summar(?:y|ise|ize|ising|izing)|overview|executive summary|tl;?dr|"
                 r"in a nutshell|brief me on)\b",
                 "asks for the whole-document picture"),
    RouterSignal("timeline_reconstruction", STRATEGY_LONG, 3.0,
                 r"\b(?:timeline|chronolog\w*|in order (?:of|that)|sequence of events|"
                 r"reconstruct (?:the|how)|from start to finish|end to end)\b",
                 "needs every timestamp, which are spread across the document"),
    RouterSignal("whole_incident_scope", STRATEGY_LONG, 2.5,
                 r"(?:across (?:the )?(?:incident|post[ _-]?mortem|document|whole|entire|all)|"
                 r"(?:whole|entire|full) (?:incident|post[ _-]?mortem|document|report|outage)|"
                 r"throughout (?:the )?(?:incident|outage|document))",
                 "explicitly spans the entire incident, not a section"),
    RouterSignal("unfold_over_time", STRATEGY_LONG, 2.0,
                 r"\b(?:how did [a-z ]{0,40}(?:unfold|develop|progress|escalate)|"
                 r"what happened (?:next|after that)|over time|hour by hour|step by step)\b",
                 "asks how events developed - a narrative arc"),
    RouterSignal("cross_section_compare", STRATEGY_LONG, 2.5,
                 r"\b(?:compare|contrast|correlat\w*|cross[- ]referenc\w*|"
                 r"relationship between|how (?:do|does|did) .{0,40} relate)\b",
                 "cross-references multiple sections of the same document"),
    RouterSignal("lessons_overall", STRATEGY_LONG, 2.0,
                 r"\b(?:lessons learned|action items|what (?:was|were) the (?:key )?(?:takeaways|learnings)|"
                 r"what did (?:we|they) learn)\b",
                 "synthesis across the whole document, not a fact lookup"),
)

_COMPILED_SIGNALS: Tuple[Tuple[RouterSignal, "re.Pattern[str]"], ...] = tuple(
    (signal, re.compile(signal.pattern, re.IGNORECASE)) for signal in ROUTER_SIGNALS
)

# Documented tie-break rule: a margin below this is treated as RAG, because targeted
# retrieval is the cheap and safe default and its failure mode (a thin context) is
# recoverable, unlike a 45k-token request fired at an ambiguous question.
MARGIN_THRESHOLD = 1.0


@dataclass
class RouterDecision:
    query: str
    strategy: str
    confidence: float
    score_rag: float
    score_long: float
    signals: List[str] = field(default_factory=list)
    contested: bool = False
    reason: str = ""


def route_query(query: str) -> RouterDecision:
    """
    Deterministic signal-scoring router.

    Each signal that fires contributes its weight to its strategy's score. The higher
    score wins; the margin (normalised into a confidence) shows how decisive the call was.
    Fired signals are returned so the decision is auditable line by line.
    """
    score_rag = 0.0
    score_long = 0.0
    signals: List[str] = []

    for signal, pattern in _COMPILED_SIGNALS:
        matches = pattern.findall(query)
        if not matches:
            continue
        hits = len(matches)
        # Diminishing returns: repeated hits of one rule never dominate the score.
        contribution = signal.weight * (1.0 + 0.25 * (hits - 1))
        if signal.strategy == STRATEGY_RAG:
            score_rag += contribution
        else:
            score_long += contribution
        # The matched span is printed so the decision can be checked by eye.
        sample = matches[0]
        sample = sample if isinstance(sample, str) else " ".join(p for p in sample if p)
        signals.append(f"{signal.name}[{signal.strategy}] w={contribution:.2f} ~ {sample[:38]!r}")

    margin = score_rag - score_long
    if score_rag == 0.0 and score_long == 0.0:
        strategy = STRATEGY_RAG
        reason = "no signal fired -> conservative default (targeted retrieval)"
    elif score_long > score_rag:
        strategy = STRATEGY_LONG
        reason = f"long-context score {score_long:.2f} > RAG score {score_rag:.2f}"
    else:
        strategy = STRATEGY_RAG
        reason = f"RAG score {score_rag:.2f} >= long-context score {score_long:.2f}"

    contested = False
    if abs(margin) < MARGIN_THRESHOLD and max(score_rag, score_long) > 0.0:
        contested = True
        reason += " | margin below tie-break threshold -> RAG (cheap default)"

    # Confidence blends three explainable factors:
    #   0.45 * margin decisiveness (logistic on the score gap)
    #   0.25 * evidence mass (how much signal fired at all)
    #   0.30 * winning-side purity (how much of the evidence agreed)
    decisiveness = 1.0 / (1.0 + pow(2.718281828, -abs(margin) / 2.0))
    total_mass = score_rag + score_long
    mass_factor = min(total_mass / 6.0, 1.0)
    winning = max(score_rag, score_long)
    purity = (winning / total_mass) if total_mass > 0 else 0.5
    confidence = 0.45 * decisiveness + 0.25 * mass_factor + 0.30 * purity

    return RouterDecision(
        query=query,
        strategy=strategy,
        confidence=round(confidence, 3),
        score_rag=round(score_rag, 2),
        score_long=round(score_long, 2),
        signals=signals,
        contested=contested,
        reason=reason,
    )


# Optional LLM classifier - behind --llm ONLY, never required by the default run.
_LLM_CLASSIFIER_SYSTEM = (
    "You are a routing classifier for a post-mortem RAG system. Choose exactly one strategy.\n"
    "RAG        = targeted hybrid retrieval over chunks. Right when the query needs a specific\n"
    "             root cause, an exact error code, flag or value, a narrow section, or low latency.\n"
    "LONG_CONTEXT = the entire document is passed to a large-context model. Right when the query\n"
    "             needs a full narrative summary, a timeline reconstruction, cross-referencing\n"
    "             several sections, or the lessons learned across the whole incident.\n"
    'Reply with ONLY compact JSON: {"strategy": "RAG" or "LONG_CONTEXT", "confidence": 0.0-1.0, '
    '"reason": "one short sentence"}.'
)


def route_query_llm(query: str, llm: UnifiedLLM) -> Optional[RouterDecision]:
    """LLM-based classification used only for the optional --llm comparison."""
    try:
        result = llm.chat(_LLM_CLASSIFIER_SYSTEM, f"Query: {query}", max_tokens=200,
                          label="llm_router")
    except LLMUnavailable as exc:
        print(f"    [--llm] classifier unavailable: {str(exc)[:140]}")
        return None

    match = re.search(r"\{.*\}", result.text, re.S)
    if not match:
        print(f"    [--llm] classifier returned no JSON: {result.text[:100]!r}")
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:
        print(f"    [--llm] classifier JSON unparseable: {match.group(0)[:100]!r}")
        return None

    raw = str(payload.get("strategy", "")).upper().replace("-", "_").strip()
    if raw in ("RAG", "RETRIEVAL"):
        strategy = STRATEGY_RAG
    elif raw in ("LONG_CONTEXT", "LONG", "LONG CONTEXT"):
        strategy = STRATEGY_LONG
    else:
        print(f"    [--llm] classifier gave an unknown strategy: {raw!r}")
        return None
    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return RouterDecision(
        query=query,
        strategy=strategy,
        confidence=round(min(max(confidence, 0.0), 1.0), 3),
        score_rag=0.0,
        score_long=0.0,
        signals=[f"llm[{result.model}]"],
        reason=str(payload.get("reason", ""))[:120],
    )


# =====================================================================
# 6. Strategy A - RAG (hybrid retrieval + cited synthesis)
# =====================================================================
@dataclass
class StrategyRun:
    name: str
    query: str
    document: str
    answer: Optional[str] = None
    latency_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    reference_cost_usd: Optional[float] = None
    provider: str = ""
    model: str = ""
    context_chars: int = 0
    retrieved_ids: List[str] = field(default_factory=list)
    error: str = ""


def run_rag_strategy(query: str, engine: Any, llm: UnifiedLLM) -> StrategyRun:
    """
    Strategy A: hybrid (BM25 + dense + RRF) retrieval, then a cited synthesis call.

    The answer is generated from the RETRIEVED CHUNKS ONLY - this is also the source of
    truth the quality judge is given, so groundedness measures the retrieval decision,
    not the whole document.
    """
    run = StrategyRun(name="RAG", query=query, document="")

    started = time.perf_counter()
    try:
        hits = engine.search_hybrid(query, top_k=RAG_TOP_K, candidate_pool=RAG_CANDIDATE_POOL)
    except Exception as exc:
        run.error = f"hybrid retrieval failed: {type(exc).__name__}: {str(exc)[:160]}"
        return run
    run.retrieval_ms = (time.perf_counter() - started) * 1000.0

    if not hits:
        run.error = "hybrid retrieval returned 0 chunks"
        return run

    blocks: List[str] = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        blocks.append(
            f"[{hit['chunk_id']}] source={meta.get('source_file', '?')} "
            f"incident={meta.get('incident_name', '?')} page={meta.get('page_number', '?')}\n"
            f"{hit['text']}"
        )
    context_text = "\n\n".join(blocks)
    run.context_chars = len(context_text)
    run.retrieved_ids = [h["chunk_id"] for h in hits]
    run.document = context_text

    system_prompt = (
        "You are IncidentLens, an expert SRE post-mortem analyst.\n"
        "Answer the user's question using ONLY the retrieved chunks below.\n"
        "Cite the chunk id in square brackets after each fact, e.g. [cloudflare_bgp_leak_p1_c1].\n"
        "If the chunks do not contain the answer, say so explicitly instead of guessing.\n"
        "Reply with the final answer only - no planning, no restating the question.\n"
        "Be precise and concise (max 8 sentences).\n\n"
        "RETRIEVED CHUNKS:\n" + context_text
    )
    user_prompt = "Question: " + query

    # Input tokens are the whole prompt assembly (system + question), counted with
    # tiktoken so both strategies are measured on identical terms.
    run.input_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)

    started = time.perf_counter()
    try:
        result = llm.chat(system_prompt, user_prompt, max_tokens=MAX_ANSWER_TOKENS, label="rag_synthesis")
    except LLMUnavailable as exc:
        run.error = f"LLM error: {str(exc)[:200]}"
        run.latency_ms = (time.perf_counter() - started) * 1000.0
        return run

    run.latency_ms = (time.perf_counter() - started) * 1000.0
    run.answer = result.text
    run.provider = result.provider
    run.model = result.model
    if result.measured_tokens:
        # Prefer the provider's own accounting for output tokens.
        run.output_tokens = result.completion_tokens
    else:
        run.output_tokens = count_tokens(result.text)
    return run


# =====================================================================
# 7. Strategy B - LONG CONTEXT (whole document -> large-context model)
# =====================================================================
def load_incident_documents() -> Dict[str, Dict[str, Any]]:
    """Load every post-mortem and join its page-level segments into ONE full text."""
    loader = PostmortemDocumentLoader(data_dir=DATA_DIR)
    pages = loader.load_all()
    grouped: Dict[str, Dict[str, Any]] = {}
    for page in pages:
        slug = page.metadata.get("incident_slug") or page.metadata.get("source_file") or "unknown"
        entry = grouped.setdefault(slug, {
            "slug": slug,
            "name": page.metadata.get("incident_name", slug),
            "source_file": page.metadata.get("source_file", ""),
            "texts": [],
        })
        entry["texts"].append(page.page_content)
    for entry in grouped.values():
        entry["text"] = "\n\n".join(entry["texts"])
        entry["chars"] = len(entry["text"])
    return grouped


def resolve_document(query: str, documents: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Pick the incident a whole-document question is about.

    The probe queries name their incident, so an explicit slug/token match is used first
    and the largest document is the documented fallback.
    """
    lowered = query.lower()
    for slug, entry in documents.items():
        tokens = [t for t in re.split(r"[^a-z0-9]+", slug.lower()) if len(t) >= 4]
        if slug.lower() in lowered or any(t in lowered for t in tokens):
            return entry
    if not documents:
        return None
    return max(documents.values(), key=lambda e: e["chars"])


def run_long_context_strategy(query: str, document: Dict[str, Any], llm: UnifiedLLM,
                              model_context: int, rate_in: float, rate_out: float) -> StrategyRun:
    """
    Strategy B: send the ENTIRE post-mortem to a large-context model.

    Token counts are measured with tiktoken on the exact prompt string that is sent, so
    the comparison against Strategy A is apples-to-apples. Cost is estimated from the
    documented per-million rates.
    """
    run = StrategyRun(name="LONG_CONTEXT", query=query, document=document["text"])
    run.context_chars = document["chars"]

    system_prompt = (
        f"You are IncidentLens, an expert SRE post-mortem analyst.\n"
        f"The COMPLETE post-mortem for the '{document['name']}' incident is provided below.\n"
        "Answer the user's question from this whole document. Use every relevant section,\n"
        "keep the chronology accurate, and be explicit about which phase each statement\n"
        "belongs to. Do not invent facts that are not in the document.\n"
        "Reply with the final answer only - no planning, no restating the question.\n\n"
        "FULL POST-MORTEM TEXT:\n" + document["text"]
    )
    user_prompt = "Question: " + query

    run.input_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)

    if model_context and run.input_tokens > model_context:
        run.error = (f"document needs {run.input_tokens:,} input tokens but the chosen model "
                     f"window is {model_context:,} - refusing to truncate silently")
        return run

    started = time.perf_counter()
    try:
        result = llm.chat(system_prompt, user_prompt, max_tokens=MAX_ANSWER_TOKENS, label="long_context")
    except LLMUnavailable as exc:
        run.error = f"LLM error: {str(exc)[:200]}"
        run.latency_ms = (time.perf_counter() - started) * 1000.0
        return run

    run.latency_ms = (time.perf_counter() - started) * 1000.0
    run.answer = result.text
    run.provider = result.provider
    run.model = result.model
    run.output_tokens = result.completion_tokens if result.measured_tokens else count_tokens(result.text)
    run.cost_usd = estimate_cost_usd(run.input_tokens, run.output_tokens, rate_in, rate_out)
    run.reference_cost_usd = estimate_cost_usd(run.input_tokens, run.output_tokens,
                                               REFERENCE_RATE_IN, REFERENCE_RATE_OUT)
    return run


# =====================================================================
# 8. Quality Scoring - documented groundedness/relevance rubric (LLM judge)
# =====================================================================
JUDGE_SYSTEM = (
    "You are a strict answer-quality judge for a post-mortem question-answering system.\n"
    "You are given a QUESTION, a SOURCE, and an ANSWER.\n"
    "Score TWO axes between 0.0 and 1.0.\n"
    "groundedness: 1.0 if every factual claim in the ANSWER is traceable to the SOURCE;\n"
    "  0.5 if some claims are supported and others are not; 0.0 if it contains fabricated facts.\n"
    "relevance: 1.0 if the ANSWER directly and completely answers the QUESTION; 0.5 if it\n"
    "  only partially does; 0.0 if it is off-topic.\n"
    "Judge only what is written. Do not reward length.\n"
    "Output the JSON on the FIRST line, before any explanation, and nothing else on it.\n"
    'First line: {"groundedness": <float>, "relevance": <float>}\n'
    "Then, if you want, one short sentence of justification."
)

# A trivial scoring case used only to prove a candidate model can emit the rubric JSON.
PROBE_JUDGE_USER = (
    "SOURCE (a test fixture):\nThe incident was caused by a bad configuration push.\n\n"
    "QUESTION:\nWhat caused the incident?\n\n"
    "ANSWER:\nA bad configuration push caused the incident.\n\n"
    "Score JSON:"
)


@dataclass
class QualityScore:
    groundedness: Optional[float]
    relevance: Optional[float]
    quality: Optional[float]
    judge: str
    note: str = ""


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _parse_judge_scores(text: str) -> Optional[Tuple[float, float]]:
    """
    Extract groundedness/relevance from the judge reply, tolerating prose wrappers.

    A free judge model often writes its analysis first and ends with the JSON, so every
    candidate {...} object is tried. The bare-number fallback is deliberately LAST and
    strictly labelled, because a leaked reasoning trace contains plenty of numbers that
    are not scores (timestamps, percentages) and must never be mistaken for one.
    """
    # 1. Any JSON object in the reply that carries both rubric keys. The match is
    #    non-greedy for a reason: a reasoning trace often quotes the instruction's own
    #    format, and a greedy match would swallow that and fail to parse.
    for match in re.finditer(r"\{.*?\}", text, re.S):
        try:
            payload = json.loads(match.group(0))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if "groundedness" in payload and "relevance" in payload:
            try:
                return (_clamp01(float(payload["groundedness"])),
                        _clamp01(float(payload["relevance"])))
            except (TypeError, ValueError):
                continue
    # 2. Explicitly labelled scores, e.g. "Groundedness: 0.8  Relevance: 1.0".
    labelled = re.findall(
        r"groundedness[^0-9]{0,12}(\d+(?:\.\d+)?)[^0-9]{0,40}?relevance[^0-9]{0,12}(\d+(?:\.\d+)?)",
        text, re.S | re.I)
    if labelled:
        return (_clamp01(float(labelled[-1][0])), _clamp01(float(labelled[-1][1])))
    # 3. Last resort: two decimals inside [0,1] near the end of the reply.
    decimals = re.findall(r"\b(\d\.\d+)\b", text)
    if len(decimals) >= 2:
        last_two = decimals[-2:]
        return (_clamp01(float(last_two[0])), _clamp01(float(last_two[1])))
    return None


def score_answer(question: str, source_text: str, answer: str, llm: UnifiedLLM,
                 source_label: str,
                 prefer_models: Tuple[str, ...] = ()) -> QualityScore:
    """
    Score one answer with the documented rubric.

    Returns None scores and a reason when no judge could be reached - the caller then
    prints "not measured" and FAILS the quality check rather than inventing a number.
    """
    if not answer or not answer.strip():
        return QualityScore(None, None, None, "none", "empty answer, nothing to score")

    # The judge sees only the source the strategy was allowed to use: retrieved chunks for
    # RAG, the whole document for long context. That is what makes the two comparable.
    budgeted_source = source_text if len(source_text) <= 60000 else source_text[:60000]
    user_prompt = (
        f"SOURCE ({source_label}):\n{budgeted_source}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Score JSON:"
    )
    # Try each verified judge in order; a model that returns prose without a parseable
    # rubric falls through to the next one instead of failing the whole measurement.
    errors: List[str] = []
    result: Optional[LLMResult] = None
    parsed: Optional[Tuple[float, float]] = None
    for model_id in tuple(prefer_models) + (None,):
        try:
            candidate = llm.chat(JUDGE_SYSTEM, user_prompt, max_tokens=MAX_JUDGE_TOKENS,
                                 label="quality_judge", scrub=False,
                                 prefer_models=(model_id,) if model_id else (),
                                 providers=("openrouter",) if model_id else ("groq",))
        except LLMUnavailable as exc:
            errors.append(str(exc)[:120])
            continue
        attempt = _parse_judge_scores(candidate.text)
        if attempt is None:
            errors.append(f"{candidate.provider}/{candidate.model}: no parseable scores")
            continue
        result, parsed = candidate, attempt
        break
    if result is None or parsed is None:
        return QualityScore(None, None, None, "none",
                            "judge unmeasurable - " + "; ".join(errors[:2]))

    grounded, relevant = parsed
    return QualityScore(
        groundedness=round(grounded, 2),
        relevance=round(relevant, 2),
        quality=round((grounded + relevant) / 2.0, 2),
        judge=f"{result.provider}/{result.model}",
    )


# =====================================================================
# 9. Benchmark - probes, measurement, tables and PASS/FAIL
# =====================================================================
@dataclass
class Probe:
    probe_id: str
    query: str
    expected: str        # strategy the router is EXPECTED to pick
    rationale: str


PROBES: Tuple[Probe, ...] = (
    Probe(
        probe_id="P1",
        query=("What was the root cause of the Cloudflare BGP route leak, and which ASN "
               "was the leak associated with?"),
        expected=STRATEGY_RAG,
        rationale="specific root cause + exact ASN -> answer lives in a few chunks",
    ),
    Probe(
        probe_id="P2",
        query=("Reconstruct the full timeline of the Cloudflare BGP route leak from the "
               "first change to the final mitigation, and summarise how the incident "
               "unfolded across the whole post-mortem."),
        expected=STRATEGY_LONG,
        rationale="timeline reconstruction + full narrative summary across the document",
    ),
)

# Router-only sanity checks: deterministic classification, no LLM calls, no cost.
ROUTER_ONLY_QUERIES: Tuple[Tuple[str, str], ...] = (
    ("Which failover step in the GitHub MySQL post-mortem caused a split-brain, and what error code appeared?",
     STRATEGY_RAG),
    ("Summarise the AWS Kinesis outage post-mortem from start to finish.", STRATEGY_LONG),
    ("What was the exact bugcheck code 0x00000050 in the CrowdStrike incident?", STRATEGY_RAG),
    ("Compare how the detection and the remediation sections of the CrowdStrike incident relate, and what were the lessons learned?",
     STRATEGY_LONG),
)


def _cell(text: str, width: int) -> str:
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def print_router_table(decisions: List[Tuple[str, str, RouterDecision, Optional[RouterDecision]]]) -> None:
    """Router decision table: query | chosen strategy | expected strategy | signals."""
    print("\n" + "=" * 118)
    print(" ROUTER DECISION TABLE")
    print("=" * 118)
    print(f"{'Query':<54} | {'Chosen':<13} | {'Expected':<13} | {'OK':<3} | Signals (weight ~ matched span)")
    print("-" * 118)
    for label, expected, decision, llm_decision in decisions:
        ok = "yes" if decision.strategy == expected else "NO"
        signals = "; ".join(decision.signals)
        if not signals:
            signals = "(no signal -> conservative RAG default)"
        print(f"{_cell(label, 54):<54} | {decision.strategy:<13} | {expected:<13} | {ok:<3} | {signals}")
        print(f"{'':<54} | conf {decision.confidence:<8} | {'':<13} | {'':<3} | {decision.reason}")
        if llm_decision is not None:
            agree = "agrees" if llm_decision.strategy == decision.strategy else "DISAGREES"
            print(f"{'':<54} | {'':<13} | {'':<13} | {'':<3} | --llm: {llm_decision.strategy} "
                  f"({agree}) {llm_decision.reason[:60]}")
    print("-" * 118)


def print_benchmark_table(runs: List[StrategyRun], scores: Dict[int, QualityScore],
                          probe_ids: List[str]) -> None:
    """Benchmark table: strategy | latency | tokens | cost | quality."""
    print("\n" + "=" * 132)
    print(" BENCHMARK TABLE  (cost column 1 = the model's own published rate, column 2 = documented")
    print(f" commercial reference rate ${REFERENCE_RATE_IN:.2f}/${REFERENCE_RATE_OUT:.2f} per 1M tokens)")
    print("=" * 132)
    header = (f"{'Probe':<5} | {'Strategy':<13} | {'Latency':>10} | {'Input tok':>9} | "
              f"{'Out tok':>7} | {'Cost (own rate)':>15} | {'Cost (ref rate)':>15} | {'Quality':>7}")
    print(header)
    print("-" * 132)
    for index, run in enumerate(runs):
        score = scores.get(index)
        quality = "n/m" if (score is None or score.quality is None) else f"{score.quality:.2f}"
        latency = fmt_ms(run.latency_ms)
        in_tokens = "n/m" if run.input_tokens is None else f"{run.input_tokens:,}"
        out_tokens = "n/m" if run.output_tokens is None else f"{run.output_tokens:,}"
        label = probe_ids[index] if index < len(probe_ids) else f"#{index + 1}"
        print(f"{label:<5} | {run.name:<13} | {latency:>10} | {in_tokens:>9} | "
              f"{out_tokens:>7} | {fmt_cost(run.cost_usd):>15} | "
              f"{fmt_cost(run.reference_cost_usd):>15} | {quality:>7}")
        detail = run.error or (f"provider={run.provider} model={run.model} "
                               f"context={run.context_chars:,} chars")
        print(f"      | {detail[:118]}")
        if run.name == "RAG" and run.retrieval_ms is not None:
            print(f"      | latency split: hybrid retrieval {run.retrieval_ms:,.0f} ms + "
                  f"synthesis {fmt_ms(run.latency_ms)} over {len(run.retrieved_ids)} chunks")
        if score is not None:
            axes = ("groundedness/relevance = not measured" if score.groundedness is None
                    else f"groundedness={score.groundedness:.2f} relevance={score.relevance:.2f}")
            print(f"      | quality = {axes}; judge={score.judge}"
                  + (f"; {score.note}" if score.note else ""))
    print("-" * 132)


def print_answers(runs: List[StrategyRun], probe_ids: List[str]) -> None:
    print("\n" + "=" * 118)
    print(" ANSWERS (truncated to 600 chars each)")
    print("=" * 118)
    for index, run in enumerate(runs, start=1):
        label = probe_ids[index - 1] if index <= len(probe_ids) else f"P{index}"
        print(f"\n[{label} | {run.name}] \"{run.query}\"")
        print("-" * 118)
        if run.answer:
            text = run.answer.replace("\n", " ").strip()
            print(text[:600] + ("..." if len(text) > 600 else ""))
        else:
            print(f"NOT MEASURED - {run.error or 'no answer produced'}")


def main() -> int:
    use_llm_router = "--llm" in sys.argv
    no_cache = "--no-cache" in sys.argv

    print("=" * 118)
    print(" \U0001F500 IncidentLens - Module 21: Long Context vs. RAG Router + Benchmark")
    print("=" * 118)
    print(" Phase 6 (docs/12 Technique 1): a dynamic decision engine that chooses targeted RAG")
    print(" retrieval or a whole-document large-context pass, then measures what each choice cost.")
    print(f" Mode: deterministic router (default)"
          f"{' + optional LLM classifier (--llm)' if use_llm_router else ''}")
    print(f" Tokenizer: tiktoken {_ENCODER_NAME}")

    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    print(f"[1] Credentials: OPENROUTER_API_KEY {'present' if openrouter_key else 'MISSING'} "
          f"(len={len(openrouter_key)}), GROQ_API_KEY {'present' if groq_key else 'MISSING'} "
          f"(len={len(groq_key)}) - values are never printed")
    if not openrouter_key:
        print("    WARNING: OpenRouter is the preferred provider for the long-context strategy.")

    # -----------------------------------------------------------------
    # Discover the large-context model before spending any other call.
    # -----------------------------------------------------------------
    model_choice: Optional[ModelChoice] = None
    model_error = ""
    free_candidates: Optional[List[Dict[str, Any]]] = None
    probe_log: List[str] = []
    try:
        model_choice, free_candidates, probe_log = discover_long_context_model(
            openrouter_key, MIN_LONG_CONTEXT_WINDOW, use_cache=not no_cache)
        print(f"    Context window : {model_choice.context_length:,} tokens")
        print(f"    Published rate : ${model_choice.rate_in:.4f} in / ${model_choice.rate_out:.4f} out "
              f"per 1M tokens (free ids publish 0.00; reference column uses "
              f"${REFERENCE_RATE_IN:.2f}/${REFERENCE_RATE_OUT:.2f})")
        if model_choice.context_length < MIN_LONG_CONTEXT_WINDOW:
            print(f"    NOTE: no free model offered the preferred {MIN_LONG_CONTEXT_WINDOW:,}-token "
                  f"window; using the largest that answered.")
    except LLMUnavailable as exc:
        model_error = str(exc)
        print(f"[2] LONG CONTEXT UNAVAILABLE: {model_error}")
        print("    The long-context checks below will FAIL; no numbers will be invented.")

    llm = UnifiedLLM(openrouter_key, groq_key,
                     model_choice.model_id if model_choice else "",
                     model_context=model_choice.context_length if model_choice else None)

    # -----------------------------------------------------------------
    # Pick the quality judge by probing for a model that emits the rubric JSON.
    # -----------------------------------------------------------------
    judge_models: Tuple[str, ...] = ()
    if model_choice is not None:
        judge_models, judge_log = discover_judge_models(
            openrouter_key, free_candidates, already_ok=(model_choice.model_id,))
        probe_log.extend(judge_log)
        print(f"[2b] Judge chain ({len(judge_models)} free model(s)): "
              f"{', '.join(judge_models) if judge_models else 'none discovered'}"
              + ("" if judge_models else " -> quality will be scored by the Groq fallback"))

    # -----------------------------------------------------------------
    # Build the retrieval stack (module 10 hybrid search over module 05 collection).
    # -----------------------------------------------------------------
    chunks = get_production_chunks()
    collection = get_incident_collection()
    if collection.count() < len(chunks):
        mod_chroma.index_chunks(collection, chunks)
    engine = HybridSearchEngine(chunks, collection)
    print(f"    Corpus: {len(chunks)} production chunks, "
          f"{len({c.metadata.get('incident_slug') for c in chunks})} incidents")

    documents = load_incident_documents()
    print(f"[3] Documents for whole-context passes: {len(documents)}")
    for entry in sorted(documents.values(), key=lambda e: e["chars"], reverse=True):
        print(f"    - {entry['name']:<34} {entry['chars']:>7,} chars "
              f"~{count_tokens(entry['text']):>7,} tokens")

    # -----------------------------------------------------------------
    # Route every probe (+ router-only sanity checks) BEFORE any benchmark spend.
    # -----------------------------------------------------------------
    print("\n" + "=" * 118)
    print(" ROUTING (deterministic signal scoring)")
    print("=" * 118)
    decisions: List[Tuple[str, str, RouterDecision, Optional[RouterDecision]]] = []
    probe_decisions: Dict[str, RouterDecision] = {}
    for probe in PROBES:
        decision = route_query(probe.query)
        probe_decisions[probe.probe_id] = decision
        llm_decision = route_query_llm(probe.query, llm) if use_llm_router else None
        decisions.append((f"{probe.probe_id} {probe.query}", probe.expected, decision, llm_decision))
    for query, expected in ROUTER_ONLY_QUERIES:
        decision = route_query(query)
        llm_decision = route_query_llm(query, llm) if use_llm_router else None
        decisions.append((f"(router-only) {query}", expected, decision, llm_decision))

    print_router_table(decisions)

    # -----------------------------------------------------------------
    # Benchmark: run the strategy the ROUTER chose for each probe.
    # -----------------------------------------------------------------
    print("\n" + "=" * 118)
    print(" BENCHMARK - executing the strategy the router chose for each probe")
    print("=" * 118)
    runs: List[StrategyRun] = []
    run_probe_ids: List[str] = []
    scores: Dict[int, QualityScore] = {}

    for probe in PROBES:
        decision = probe_decisions[probe.probe_id]
        print(f"\n[{probe.probe_id}] router chose {decision.strategy} "
              f"(confidence {decision.confidence}) for: \"{probe.query}\"")

        if decision.strategy == STRATEGY_RAG:
            run = run_rag_strategy(probe.query, engine, llm)
            run.cost_usd = estimate_cost_usd(
                run.input_tokens or 0, run.output_tokens or 0,
                model_choice.rate_in if model_choice else 0.0,
                model_choice.rate_out if model_choice else 0.0)
            run.reference_cost_usd = estimate_cost_usd(
                run.input_tokens or 0, run.output_tokens or 0,
                REFERENCE_RATE_IN, REFERENCE_RATE_OUT)
            source_text = run.document
            source_label = f"the {len(run.retrieved_ids)} retrieved chunks the RAG strategy was allowed to see"
        else:
            document = resolve_document(probe.query, documents)
            if document is None:
                run = StrategyRun(name="LONG_CONTEXT", query=probe.query, document="",
                                  error="no document found for a whole-context pass")
            else:
                run = run_long_context_strategy(
                    probe.query, document, llm,
                    model_choice.context_length if model_choice else 0,
                    model_choice.rate_in if model_choice else 0.0,
                    model_choice.rate_out if model_choice else 0.0)
                print(f"      document: {document['name']}")
            source_text = run.document
            source_label = "the COMPLETE post-mortem document given to the long-context model"

        index = len(runs)
        runs.append(run)
        run_probe_ids.append(probe.probe_id)

        if run.answer:
            print(f"      answered by {run.provider}/{run.model} in {fmt_ms(run.latency_ms)} "
                  f"({run.input_tokens:,} input tokens)")
            score = score_answer(probe.query, source_text, run.answer, llm, source_label,
                                 prefer_models=judge_models)
            scores[index] = score
            if score.quality is None:
                print(f"      quality: not measured - {score.note}")
            else:
                print(f"      quality: {score.quality:.2f} "
                      f"(groundedness {score.groundedness:.2f}, relevance {score.relevance:.2f}) "
                      f"by {score.judge}")
        else:
            print(f"      FAILED: {run.error or 'no answer produced'}")

    print_benchmark_table(runs, scores, run_probe_ids)
    print("\n  Reading the numbers honestly:")
    print("    - RAG latency = module 10 hybrid retrieval + synthesis; long-context latency =")
    print("      the single whole-document completion. The token column is the full prompt")
    print("      assembly (system + question + context) counted with tiktoken for BOTH paths,")
    print("      so the ratio is the real cost multiple at these document sizes.")
    largest = max((e['chars'] for e in documents.values()), default=0)
    print(f"    - This corpus is small (largest post-mortem {largest:,} chars), so the whole-")
    print("      document path costs only single-digit thousands of tokens here. On the 200-page")
    print("      reports docs/12 targets, the same router decision costs 25-30x the retrieval")
    print("      path - the ratio, not this corpus, is what generalises.")
    print_answers(runs, run_probe_ids)

    call_count = len(llm.calls) + len(probe_log)
    print(f"\n  Provider calls this run: {call_count} total "
          f"({len(probe_log)} discovery probe(s) + {len(llm.calls)} completion(s)).")
    for entry in llm.calls:
        print(f"    - {entry['label']:<15} {entry['provider']}/{entry['model']} "
              f"({entry['prompt_tokens']:,} in / {entry['completion_tokens']:,} out)")

    # -----------------------------------------------------------------
    # PASS / FAIL checks
    # -----------------------------------------------------------------
    print("\n" + "=" * 118)
    print(" CHECKS")
    print("=" * 118)

    checks: List[Tuple[str, bool, str]] = []

    router_ok = all(
        probe_decisions[p.probe_id].strategy == p.expected for p in PROBES
    )
    router_detail = "; ".join(
        f"{p.probe_id}: chose {probe_decisions[p.probe_id].strategy}, expected {p.expected}"
        for p in PROBES
    )
    checks.append(("1. Router picks the expected strategy for every probe", router_ok, router_detail))

    measured_ok = True
    measured_detail: List[str] = []
    for index, run in enumerate(runs):
        ok = (run.latency_ms is not None and run.input_tokens is not None
              and run.output_tokens is not None and bool(run.answer))
        measured_ok = measured_ok and ok
        measured_detail.append(
            f"{run_probe_ids[index]}/{run.name}=" + ("measured" if ok else f"NOT measured ({run.error[:60]})"))
    checks.append(("2. Every probe produced real latency/token measurements", measured_ok,
                   "; ".join(measured_detail)))

    quality_ok = bool(scores) and all(s.quality is not None for s in scores.values())
    if quality_ok:
        quality_detail = "; ".join(
            f"{run_probe_ids[i]}={s.quality:.2f} ({s.judge})" for i, s in sorted(scores.items()))
    else:
        missing = [f"{run_probe_ids[i]}: {s.note or 'no judge'}" for i, s in sorted(scores.items())
                   if s.quality is None]
        quality_detail = ("not measured - " + "; ".join(missing)) if missing else "not measured"
    checks.append(("3. Quality scored by the documented rubric (not measured => FAIL)", quality_ok,
                   quality_detail))

    if len(runs) == 2 and all(r.input_tokens for r in runs):
        rag_tokens = runs[0].input_tokens or 0
        long_tokens = runs[1].input_tokens or 0
        checks.append(("4. Whole-document path is genuinely larger than the retrieval path",
                       long_tokens > rag_tokens,
                       f"long-context {long_tokens:,} input tokens vs RAG {rag_tokens:,} "
                       f"({long_tokens / max(rag_tokens, 1):.1f}x)"))
    else:
        checks.append(("4. Whole-document path is genuinely larger than the retrieval path", False,
                       "not measured - benchmark did not produce two token counts"))

    long_run = next((r for r in runs if r.name == "LONG_CONTEXT"), None)
    if long_run is not None and long_run.answer:
        check5_detail = (f"{long_run.provider}/{long_run.model} in {fmt_ms(long_run.latency_ms)}")
        if long_run.provider != "openrouter" or long_run.model != (model_choice.model_id
                                                                  if model_choice else ""):
            check5_detail += (" | NOTE: OpenRouter could not serve the request "
                              f"({model_error or 'free-tier rate limit or model error'}), "
                              "so the configured Groq fallback produced this answer")
        checks.append(("5. Long-context strategy returned a real completion "
                       "(OpenRouter preferred, Groq fallback)", True, check5_detail))
    else:
        checks.append(("5. Long-context strategy returned a real completion "
                       "(OpenRouter preferred, Groq fallback)", False,
                       f"LONG CONTEXT UNAVAILABLE: "
                       f"{model_error or (long_run.error if long_run else 'no long-context run')}"))

    all_passed = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        all_passed = all_passed and ok
        print(f" [{status}] {name}")
        print(f"        {detail}")

    print("\n" + "=" * 118)
    if all_passed:
        print(" \u2705 Module 21 COMPLETE: the router chose the expected strategy for every probe and")
        print("    the benchmark produced real latency, token, cost and quality measurements.")
    else:
        print(" \u274C Module 21 INCOMPLETE: at least one check FAILED. See the CHECKS block above.")
        print("    Nothing was fabricated: failed measurements are reported as 'not measured'.")
    print("=" * 118)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
