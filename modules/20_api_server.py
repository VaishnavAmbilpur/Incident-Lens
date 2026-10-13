"""
20_api_server.py
================
Phase 5: Production REST API (FastAPI)
IncidentLens HTTP microservice:

ENDPOINTS:
  POST /query          Grounded answer for one question. Security gate -> agent -> sanitized answer.
  POST /query/stream   Same, but streams agent progress as Server-Sent Events.
  GET  /health         Dependency status. ?deep=true adds a live LLM reachability probe.
  POST /evaluate       Runs the 4-metric judge over N golden questions (default 2, max 5).
  GET  /metrics        Prometheus-compatible counters.
  GET  /               Service card (name, version, endpoint list).

HOW IT COMPOSES THE PROJECT:
  /query does NOT reimplement anything. It calls the pieces that were already verified:
    module 18  SecurityGate   - blocks prompt injection, redacts PII (no LLM calls)
    module 19  compiled graph - retrieve -> grade -> (rewrite) -> synthesize -> ground check
    module 19  get_engine()   - the hybrid BM25 + dense + RRF retrieval engine
    module 15  LLMJudgeEvaluator - the 4-metric scorecard used by /evaluate
  The API layer only adds what a server owes its callers: async handlers, structured errors,
  timeouts, CORS, request ids, timing headers, health checks and metrics.

THREE PRODUCTION DETAILS THAT MATTER HERE:
  1. The graph is SYNCHRONOUS. Calling it directly inside an `async def` handler would block
     the event loop and stall every other request. It is run with `asyncio.to_thread`.
  2. SQLite does not like concurrent writers. Graph invocations are serialized with an
     asyncio.Lock and the reason is documented in `_run_graph`. A real deployment would use
     the Postgres checkpointer.
  3. A blocked query costs ZERO LLM calls, because the security gate runs first.
"""

import os
import re
import sys
import json
import time
import uuid
import asyncio
import sqlite3
import threading
import subprocess
import importlib.util
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
REQUIRED_PACKAGES = ("fastapi", "uvicorn", "langgraph", "langchain_groq", "chromadb", "dotenv")

SERVICE_NAME = "IncidentLens RAG API"
SERVICE_VERSION = "1.0.0"
DEFAULT_PORT = int(os.getenv("INCIDENTLENS_PORT", "8000"))


def _missing_packages(names) -> List[str]:
    missing: List[str] = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def ensure_project_interpreter() -> None:
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
        print("      .\\venv\\Scripts\\python.exe modules\\20_api_server.py")
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

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

# =====================================================================
# 1. Compose the existing modules
# =====================================================================
def _import_helper(filename: str, module_name: str):
    script_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("[INIT] Loading modules 15 (judge), 18 (security), 19 (agent)...")
mod15 = _import_helper("15_ragas_evaluation.py", "module_15_judge")
mod18 = _import_helper("18_security_guardrails.py", "module_18_security")
mod19 = _import_helper("19_langgraph_agent.py", "module_19_agent")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("INCIDENTLENS_REQUEST_TIMEOUT", "180"))

# =====================================================================
# 2. Metrics (Prometheus text format, no extra dependency)
# =====================================================================
_METRICS_LOCK = threading.Lock()
METRICS: Dict[str, float] = {
    "query_total": 0.0,
    "query_blocked_total": 0.0,
    "query_failed_total": 0.0,
    "evaluate_total": 0.0,
    "http_requests_total": 0.0,
    "http_errors_total": 0.0,
    "query_latency_ms_sum": 0.0,
    "query_latency_ms_count": 0.0,
    "query_latency_ms_max": 0.0,
    "started_at": time.time(),
}


def metric_add(name: str, value: float = 1.0) -> None:
    with _METRICS_LOCK:
        METRICS[name] = METRICS.get(name, 0.0) + value
        if name == "query_latency_ms_sum":
            METRICS["query_latency_ms_count"] += 1.0


def metric_max(name: str, value: float) -> None:
    with _METRICS_LOCK:
        METRICS[name] = max(METRICS.get(name, 0.0), value)


def render_metrics() -> str:
    with _METRICS_LOCK:
        snapshot = dict(METRICS)
    count = snapshot.get("query_latency_ms_count", 0.0)
    avg = (snapshot.get("query_latency_ms_sum", 0.0) / count) if count else 0.0
    uptime = time.time() - snapshot.get("started_at", time.time())

    lines = [
        "# HELP incidentlens_uptime_seconds Seconds since the service started.",
        "# TYPE incidentlens_uptime_seconds gauge",
        f"incidentlens_uptime_seconds {uptime:.1f}",
        "# HELP incidentlens_http_requests_total Total HTTP requests received.",
        "# TYPE incidentlens_http_requests_total counter",
        f"incidentlens_http_requests_total {int(snapshot.get('http_requests_total', 0))}",
        "# HELP incidentlens_http_errors_total Total failed HTTP requests.",
        "# TYPE incidentlens_http_errors_total counter",
        f"incidentlens_http_errors_total {int(snapshot.get('http_errors_total', 0))}",
        "# HELP incidentlens_query_total Total /query calls.",
        "# TYPE incidentlens_query_total counter",
        f"incidentlens_query_total {int(snapshot.get('query_total', 0))}",
        "# HELP incidentlens_query_blocked_total Queries refused by the security gate.",
        "# TYPE incidentlens_query_blocked_total counter",
        f"incidentlens_query_blocked_total {int(snapshot.get('query_blocked_total', 0))}",
        "# HELP incidentlens_query_failed_total Queries that ended in an error.",
        "# TYPE incidentlens_query_failed_total counter",
        f"incidentlens_query_failed_total {int(snapshot.get('query_failed_total', 0))}",
        "# HELP incidentlens_evaluate_total Total /evaluate calls.",
        "# TYPE incidentlens_evaluate_total counter",
        f"incidentlens_evaluate_total {int(snapshot.get('evaluate_total', 0))}",
        "# HELP incidentlens_query_latency_ms Query latency in milliseconds.",
        "# TYPE incidentlens_query_latency_ms summary",
        f"incidentlens_query_latency_ms_sum {snapshot.get('query_latency_ms_sum', 0.0):.1f}",
        f"incidentlens_query_latency_ms_count {int(count)}",
        f"incidentlens_query_latency_ms_avg {avg:.1f}",
        f"incidentlens_query_latency_ms_max {snapshot.get('query_latency_ms_max', 0.0):.1f}",
    ]
    return "\n".join(lines) + "\n"


# =====================================================================
# 3. Request / response models (pydantic v2)
# =====================================================================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="Question about an incident post-mortem.")
    thread_id: Optional[str] = Field(None, max_length=100,
                                     description="Conversation id. Reuse it to keep agent memory.")


class QueryResponse(BaseModel):
    thread_id: str
    answer: str
    blocked: bool = False
    citations: List[str] = []
    relevance_score: Optional[float] = None
    relevance_source: Optional[str] = None
    grounding_score: Optional[float] = None
    grounding_checked: bool = False
    security_note: Optional[str] = None
    latency_ms: int = 0
    path: List[str] = []


class EvaluateRequest(BaseModel):
    limit: int = Field(2, ge=1, le=5,
                       description="How many golden questions to evaluate (each costs 5 LLM calls).")


class HealthCheck(BaseModel):
    name: str
    status: str
    detail: str


# =====================================================================
# 4. Lifespan: build the heavy objects once
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 80)
    print(f" {SERVICE_NAME} v{SERVICE_VERSION} starting")
    print("=" * 80)

    started = time.time()
    # Building the engine chunks the whole corpus and fits BM25: do it once, off the loop.
    engine = await asyncio.to_thread(mod19.get_engine)
    checkpointer, memory_kind = await asyncio.to_thread(mod19.build_checkpointer)
    graph = mod19.build_graph(checkpointer)

    app.state.engine = engine
    app.state.graph = graph
    app.state.memory_kind = memory_kind
    app.state.graph_lock = asyncio.Lock()
    print(f"[INIT] Graph compiled, memory={memory_kind}, startup={round((time.time()-started)*1000)} ms")
    print(f"[INIT] Listening on http://127.0.0.1:{DEFAULT_PORT}")
    yield
    print("[SHUTDOWN] IncidentLens API stopped.")


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="Production RAG API over IT incident post-mortems.",
    lifespan=lifespan,
)

# CORS: the browser must be allowed to call this from another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("INCIDENTLENS_CORS_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =====================================================================
# 5. Middleware: request id + timing
# =====================================================================
@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    metric_add("http_requests_total")

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metric_add("http_errors_total")
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    if response.status_code >= 400:
        metric_add("http_errors_total")
    return response


# =====================================================================
# 6. Structured error responses
# =====================================================================
def _error_payload(request: Request, kind: str, message: str,
                   details: Optional[List[Any]] = None) -> Dict[str, Any]:
    return {
        "error": {
            "type": kind,
            "message": message,
            "details": details or [],
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    }


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=_error_payload(request, "validation_error", "Invalid request body.",
                              jsonable_encoder(exc.errors())[:3]),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=_error_payload(request, "internal_error", f"{type(exc).__name__}: {exc}"[:200]),
    )


# =====================================================================
# 7. Helpers
# =====================================================================
CITATION_PATTERN = re.compile(r"[\[\u3010]([^\]\u3011]{2,60})[\]\u3011]")


def extract_citations(answer: str, known_ids: List[str]) -> List[str]:
    """
    Collect citation markers from an answer.

    Models format citations differently ([chunk_id], 【c20】), so known chunk ids are matched
    first and bracket contents are used as a fallback. Returned values are deduplicated and
    capped so a chatty answer cannot flood the response.
    """
    found: List[str] = []
    for cid in known_ids:
        if cid in answer and cid not in found:
            found.append(cid)
    for match in CITATION_PATTERN.findall(answer):
        marker = match.strip()
        if marker and marker not in found:
            found.append(marker)
    return found[:12]


def initial_agent_state(question: str) -> Dict[str, Any]:
    """The full state the graph expects. Every key must be present before the first node."""
    return {
        "question": question,
        "original_question": question,
        "context": [],
        "context_ids": [],
        "retrieval_scores": [],
        "retrieval_strength": 0.0,
        "relevance_score": 0.0,
        "relevance_source": "",
        "retry_count": 0,
        "synthesis_attempts": 0,
        "synthesis_failed": False,
        "hallucination_score": 0.0,
        "hallucination_checked": False,
        "blocked": False,
        "security_note": "",
        "answer": "",
        "steps": [],
        "rewrite_history": [],
    }


async def _run_graph(app: FastAPI, question: str, thread_id: str) -> Dict[str, Any]:
    """
    Invoke the agent graph and return its final state.

    Two production concerns are handled here:

    1. The compiled graph is SYNC. `asyncio.to_thread` keeps the event loop free, so one slow
       query cannot stall every other request.
    2. SQLite allows one writer at a time. The lock serializes graph runs; without it,
       concurrent requests raise "database is locked" from the checkpointer. A real
       deployment would use the Postgres checkpointer and drop the lock.
    """
    state = initial_agent_state(question)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    async with app.state.graph_lock:
        return await asyncio.wait_for(
            asyncio.to_thread(app.state.graph.invoke, state, config),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )


# =====================================================================
# 8. Endpoints
# =====================================================================
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "endpoints": ["POST /query", "POST /query/stream", "GET /health",
                      "POST /evaluate", "GET /metrics"],
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    """Answer one question through the full pipeline: gate -> agent -> sanitized answer."""
    thread_id = req.thread_id or f"api-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    metric_add("query_total")

    print(f"\n[/query] id={getattr(request.state, 'request_id', '-')} thread={thread_id}")
    print(f"         q=\"{req.question[:90]}\"")

    try:
        final = await _run_graph(request.app, req.question, thread_id)
    except asyncio.TimeoutError:
        metric_add("query_failed_total")
        return JSONResponse(
            status_code=504,
            content=_error_payload(request, "timeout",
                                   f"The agent exceeded {REQUEST_TIMEOUT_SECONDS:.0f} seconds."),
        )
    except Exception as exc:
        metric_add("query_failed_total")
        return JSONResponse(
            status_code=500,
            content=_error_payload(request, "pipeline_error", f"{type(exc).__name__}: {exc}"[:200]),
        )

    latency_ms = round((time.perf_counter() - started) * 1000)
    metric_add("query_latency_ms_sum", latency_ms)
    metric_max("query_latency_ms_max", latency_ms)
    if final.get("blocked"):
        metric_add("query_blocked_total")

    answer = final.get("answer", "")
    response = QueryResponse(
        thread_id=thread_id,
        answer=answer,
        blocked=bool(final.get("blocked")),
        citations=extract_citations(answer, final.get("context_ids", []) or []),
        relevance_score=final.get("relevance_score"),
        relevance_source=final.get("relevance_source") or None,
        grounding_score=final.get("hallucination_score"),
        grounding_checked=bool(final.get("hallucination_checked")),
        security_note=final.get("security_note") or None,
        latency_ms=latency_ms,
        path=final.get("steps", []) or [],
    )
    print(f"         blocked={response.blocked} latency={latency_ms} ms "
          f"citations={len(response.citations)}")
    return response


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    """
    Same pipeline, streamed as Server-Sent Events.

    These are PROGRESS events (one per graph node), not token-by-token text: the nodes call
    the model synchronously, so streaming tokens would need an async generation path through
    the graph. Node-level progress is what an operator actually watches, and it is honest
    about what it is.

    Why not `graph.astream()`? Because the compiled graph uses the SYNC SqliteSaver, and
    langgraph refuses to drive a sync checkpointer asynchronously - it raises and points at
    `AsyncSqliteSaver`, which needs the extra `aiosqlite` package. The node stream is produced
    by `graph.stream()` in a worker thread and bridged to the event loop through a queue.
    """
    thread_id = req.thread_id or f"api-{uuid.uuid4().hex[:8]}"
    state = initial_agent_state(req.question)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    graph = request.app.state.graph

    def sse(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(jsonable_encoder(payload))}\n\n"

    async def event_stream():
        metric_add("query_total")
        started = time.perf_counter()
        yield sse({"event": "start", "thread_id": thread_id, "question": req.question})

        # Worker thread -> event loop bridge. Each finished node is pushed onto the queue and
        # the generator awaits it, so the response stays responsive without blocking on SQLite.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def produce():
            try:
                for update in graph.stream(state, config, stream_mode="updates"):
                    for node_name, node_update in (update or {}).items():
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("node", node_name, (node_update or {}).get("steps", [])),
                        )
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None, None))
            except Exception as exc:                     # surfaced as an SSE error event
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", type(exc).__name__, str(exc)[:200]))

        try:
            async with request.app.state.graph_lock:
                producer_task = asyncio.create_task(asyncio.to_thread(produce))
                while True:
                    kind, value, extra = await queue.get()
                    if kind == "node":
                        yield sse({"event": "node", "node": value, "steps": extra})
                    elif kind == "error":
                        metric_add("query_failed_total")
                        yield sse({"event": "error", "type": value, "message": extra})
                        await producer_task
                        return
                    else:
                        break
                await producer_task
                snapshot = await asyncio.to_thread(graph.get_state, config)
                final = snapshot.values or {}
        except Exception as exc:
            metric_add("query_failed_total")
            yield sse({"event": "error", "type": type(exc).__name__, "message": str(exc)[:200]})
            return

        latency_ms = round((time.perf_counter() - started) * 1000)
        metric_add("query_latency_ms_sum", latency_ms)
        metric_max("query_latency_ms_max", latency_ms)
        if final.get("blocked"):
            metric_add("query_blocked_total")

        yield sse({
            "event": "answer",
            "thread_id": thread_id,
            "answer": final.get("answer", ""),
            "blocked": bool(final.get("blocked")),
            "citations": extract_citations(final.get("answer", ""),
                                           final.get("context_ids", []) or []),
            "latency_ms": latency_ms,
        })
        yield sse({"event": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
async def health(request: Request, deep: bool = False):
    """
    Report every dependency. `?deep=true` adds a live LLM call, which costs a token or two,
    so it is off by default: a health probe that hammers a rate-limited free tier is worse
    than useless.
    """
    checks: List[HealthCheck] = []

    # vector store
    try:
        count = await asyncio.to_thread(request.app.state.engine.collection.count)
        checks.append(HealthCheck(name="vector_store", status="ok",
                                  detail=f"{count} chunks in collection"))
    except Exception as exc:
        checks.append(HealthCheck(name="vector_store", status="error", detail=str(exc)[:120]))

    # retrieval engine (hybrid)
    try:
        engine = request.app.state.engine
        checks.append(HealthCheck(name="hybrid_retrieval", status="ok",
                                  detail=f"rrf_k={getattr(engine, 'rrf_k', '?')}"))
    except Exception as exc:
        checks.append(HealthCheck(name="hybrid_retrieval", status="error", detail=str(exc)[:120]))

    # security gate
    try:
        rules = len(mod19.GATE.detector.rules)
        pii = len(mod19.GATE.redactor.rules)
        checks.append(HealthCheck(name="security_guardrails", status="ok",
                                  detail=f"{rules} injection rules, {pii} PII rules"))
    except Exception as exc:
        checks.append(HealthCheck(name="security_guardrails", status="error", detail=str(exc)[:120]))

    # agent graph
    try:
        graph = request.app.state.graph
        checks.append(HealthCheck(name="agent_graph", status="ok",
                                  detail=f"memory={request.app.state.memory_kind}, "
                                         f"nodes={len(graph.get_graph().nodes)}"))
    except Exception as exc:
        checks.append(HealthCheck(name="agent_graph", status="error", detail=str(exc)[:120]))

    # thread memory on disk
    try:
        db = mod19.MEMORY_DB
        if db.exists():
            checks.append(HealthCheck(name="thread_memory", status="ok",
                                      detail=f"{db.name}, {db.stat().st_size} bytes"))
        else:
            checks.append(HealthCheck(name="thread_memory", status="degraded",
                                      detail="no database yet (nothing has run)"))
    except Exception as exc:
        checks.append(HealthCheck(name="thread_memory", status="error", detail=str(exc)[:120]))

    # credentials
    checks.append(HealthCheck(name="llm_credentials", status="ok" if GROQ_API_KEY else "error",
                              detail=f"GROQ_API_KEY {'present' if GROQ_API_KEY else 'MISSING'}, "
                                     f"synthesis={GROQ_MODEL}"))

    # optional live probe
    if deep and GROQ_API_KEY:
        try:
            reply = await asyncio.to_thread(mod19.llm_text, "Reply with the single word OK.",
                                            "ping", 8)
            checks.append(HealthCheck(name="llm_live", status="ok" if reply else "degraded",
                                      detail=f"reply={reply!r}" if reply else "no reply (rate limited?)"))
        except Exception as exc:
            checks.append(HealthCheck(name="llm_live", status="error", detail=str(exc)[:120]))

    status = "ok" if all(c.status == "ok" for c in checks) else "degraded"
    return {
        "status": status,
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checks": [c.model_dump() for c in checks],
    }


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest, request: Request):
    """
    Run the 4-metric judge over the first N golden questions.

    Retrieval here is dense-only and top_k=4 (the module 15 baseline) so the numbers are
    comparable with the scorecard already recorded. Each question costs 5 LLM calls
    (1 answer + 4 judges), which is why N is capped at 5.
    """
    metric_add("evaluate_total")
    dataset_path = PROJECT_ROOT / "data" / "eval_golden_dataset.json"
    try:
        test_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return JSONResponse(status_code=500, content=_error_payload(
            request, "dataset_error", f"Cannot read {dataset_path.name}: {exc}"[:180]))

    chosen = test_cases[: req.limit]
    questions = [tc["question"] for tc in chosen]
    ground_truths = [tc["ground_truth"] for tc in chosen]

    def _run() -> Dict[str, Any]:
        collection = mod19.mod05.get_incident_collection()
        rag_results = mod15.run_rag_for_evaluation(
            questions=questions, collection=collection, groq_api_key=GROQ_API_KEY, top_k=4)
        evaluator = mod15.LLMJudgeEvaluator(GROQ_API_KEY)
        scores = evaluator.evaluate_all(rag_results, ground_truths)
        return {"metrics": {k: round(v, 3) for k, v in scores.items()
                            if k in ("faithfulness", "answer_relevancy",
                                     "context_precision", "context_recall")},
                "judge_failures": scores.get("judge_failures", 0),
                "per_question": [{"question": r["question"][:80],
                                  "faithfulness": r["faithfulness"],
                                  "answer_relevancy": r["answer_relevancy"],
                                  "context_precision": r["context_precision"],
                                  "context_recall": r["context_recall"]}
                                 for r in scores.get("per_question", [])]}

    try:
        result = await asyncio.wait_for(asyncio.to_thread(_run),
                                       timeout=max(REQUEST_TIMEOUT_SECONDS, 600.0))
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content=_error_payload(
            request, "timeout", "Evaluation exceeded the time budget."))
    except Exception as exc:
        return JSONResponse(status_code=500, content=_error_payload(
            request, "evaluation_error", f"{type(exc).__name__}: {exc}"[:200]))

    thresholds = {"faithfulness": 0.92, "answer_relevancy": 0.90,
                  "context_precision": 0.88, "context_recall": 0.85}
    return {
        "questions_evaluated": len(questions),
        "metrics": result["metrics"],
        "targets": thresholds,
        "passed": {k: result["metrics"].get(k, 0.0) >= v for k, v in thresholds.items()},
        "judge_failures": result["judge_failures"],
        "complete": result["judge_failures"] == 0,
        "per_question": result["per_question"],
    }


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(render_metrics(), media_type="text/plain; version=0.0.4")


# =====================================================================
# 9. Entry point
# =====================================================================
def main() -> int:
    import uvicorn

    host = os.getenv("INCIDENTLENS_HOST", "127.0.0.1")
    port = DEFAULT_PORT
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
        elif arg.isdigit():
            port = int(arg)

    print("=" * 80)
    print(f" {SERVICE_NAME} v{SERVICE_VERSION}")
    print("=" * 80)
    print(f"  Docs    : http://{host}:{port}/docs")
    print(f"  Health  : http://{host}:{port}/health")
    print(f"  Query   : POST http://{host}:{port}/query")
    print(f"  Metrics : http://{host}:{port}/metrics")
    print("=" * 80)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
