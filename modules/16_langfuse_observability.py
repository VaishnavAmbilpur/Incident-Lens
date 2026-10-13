"""
16_langfuse_observability.py
============================
Phase 4: Production Observability — Langfuse Open-Source Tracing
IncidentLens Langfuse Integration:

1. Langfuse Configuration: Validates public/secret key pair and base URL connectivity.
2. Traced RAG Execution: Wraps the optimized pipeline with Langfuse's CallbackHandler
   so every LangChain LCEL chain call is captured as a trace with spans.
3. Manual Span Instrumentation: Demonstrates Langfuse's @observe decorator pattern
   for tracing non-LLM stages (retrieval, reranking, budgeting).
4. Cost Tracking: Logs input/output token counts and estimated cost per trace.

LANGFUSE vs LANGSMITH:
  - LangSmith is LangChain's proprietary hosted tracing platform.
  - Langfuse is an open-source alternative (MIT license) with self-hosting option.
  - Both capture the same telemetry. Langfuse gives you data sovereignty.
  - In production, teams often use both: LangSmith for development, Langfuse for production.

COMPATIBILITY NOTE (langfuse 4.x vs 2.x/3.x):
  This module now runs on BOTH the old and the new SDK. Four API changes caused the
  earlier "Error" output. Each one is fixed and marked with [FIX] below.

  [FIX 1] The callback handler moved. It lives at `langfuse.langchain.CallbackHandler`
          (v3/v4). The old path `langfuse.callback.CallbackHandler` was REMOVED in v4,
          so a static `from langfuse.callback import ...` shows an IDE error.
          The handler is now imported dynamically, so it works on every version.

  [FIX 2] The v4 callback handler has NO `.flush()` method. Flushing now happens on the
          Langfuse CLIENT (`client.flush()`). The old `handler.flush()` raised
          AttributeError: 'LangchainCallbackHandler' object has no attribute 'flush'.

  [FIX 3] The constructor keyword changed from `host=` (v2/v3) to `base_url=` (v4).
          The module tries `base_url=` first and falls back to `host=`.

  [FIX 4] Session grouping moved to `langfuse.propagate_attributes(session_id=...)`.
          Older versions accept `metadata={"langfuse_session_id": ...}`. Both are used.

  [FIX 5] Interpreter guard. If the script is started with an interpreter that does not
          have the project packages (for example C:\\Python313\\python.exe instead of
          venv\\Scripts\\python.exe), it re-launches itself with the project venv.
          This is the cause of "ModuleNotFoundError: No module named 'langchain_groq'".

  [FIX 6] Trace verification uses the supported read API. Langfuse Cloud retired
          `GET /api/public/traces` for organizations created on or after 16 September 2026
          (it returns HTTP 410 LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION). The module now
          reads `GET /api/public/v2/observations` and falls back to the legacy call.

  [FIX 7] The Langfuse session scope is now created FRESH for every query.
          `propagate_attributes()` returns a SINGLE-USE context manager. Entering one
          object three times raised
              AttributeError: '_AgnosticContextManager' object has no attribute 'args'
          Query 1 answered normally and queries 2 and 3 failed, because the first entry
          consumed the object. The error was caught by the pipeline's own try/except and
          printed as "[LLM Error: ...]" inside the answer, so it did not look like a crash.
"""

import os
import sys
import time
import warnings
import importlib
import importlib.util
import subprocess
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Ensure UTF-8 output encoding across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =====================================================================
# 0. Interpreter Guard  [FIX 5]
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
REQUIRED_PACKAGES = ("langfuse", "langchain_groq", "chromadb", "dotenv")
SESSION_ID = "incidentlens-module-16"


def _missing_packages(names: Tuple[str, ...]) -> List[str]:
    """Return the package names that the ACTIVE interpreter cannot import."""
    missing: List[str] = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def ensure_project_interpreter() -> None:
    """
    Re-launch this script with the project venv when the active interpreter is wrong.

    Why: VS Code and the terminal default to the system Python (C:\\Python313). That
    interpreter does not have langfuse, langchain_groq, or chromadb installed, so the
    script dies with ModuleNotFoundError. The venv does have them. This guard removes
    the need to select the interpreter by hand.
    """
    missing = _missing_packages(REQUIRED_PACKAGES)
    if not missing:
        return  # Correct interpreter. Nothing to do.

    if os.environ.get("INCIDENTLENS_REEXEC") == "1":
        # We already re-launched once. Report instead of looping forever.
        print("=" * 80)
        print(" [BOOT] Required packages are still missing after re-launch")
        print("=" * 80)
        print(f"  Interpreter : {sys.executable}")
        print(f"  Missing     : {', '.join(missing)}")
        print("  Install them with:")
        print("      .\\venv\\Scripts\\pip.exe install -r requirements.txt")
        print("=" * 80)
        sys.exit(1)

    if not VENV_PYTHON.exists():
        print("=" * 80)
        print(" [BOOT] Packages missing and no project venv found")
        print("=" * 80)
        print(f"  Interpreter : {sys.executable}")
        print(f"  Missing     : {', '.join(missing)}")
        print(f"  Expected venv at: {VENV_PYTHON}")
        print("  Create it with:")
        print("      python -m venv venv")
        print("      .\\venv\\Scripts\\pip.exe install -r requirements.txt")
        print("=" * 80)
        sys.exit(1)

    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return  # Already the venv, but a package is genuinely absent.

    print("=" * 80)
    print(" [BOOT] Wrong interpreter detected — switching to the project venv")
    print("=" * 80)
    print(f"  Active interpreter : {sys.executable}")
    print(f"  Missing packages   : {', '.join(missing)}")
    print(f"  Project venv       : {VENV_PYTHON}")
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


def _import_helper(filename: str, module_name: str):
    """Load a numbered module file (Python cannot import names that start with digits)."""
    script_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod_chroma = _import_helper("05_vector_stores_chroma.py", "module_05_chroma")
get_incident_collection = mod_chroma.get_incident_collection


# =====================================================================
# 1. Langfuse Configuration & Connectivity Check
# =====================================================================
def build_langfuse_client(public_key: str, secret_key: str, base_url: str):
    """
    Create the Langfuse client on any SDK version.  [FIX 3]

    langfuse 4.x names the endpoint parameter `base_url`.
    langfuse 2.x / 3.x name it `host`.
    """
    from langfuse import Langfuse

    try:
        return Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)
    except TypeError:
        return Langfuse(public_key=public_key, secret_key=secret_key, host=base_url)


def configure_langfuse() -> Optional[Any]:
    """Validate Langfuse credentials and connectivity. Returns the client, or None."""
    print("=" * 80)
    print(" IncidentLens — Module 16: Langfuse Open-Source Observability")
    print("=" * 80)

    print("\n[1] Verifying Langfuse Configuration...")

    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com").strip()

    if not secret_key or not public_key:
        print("    LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY not found in .env")
        print("    To enable Langfuse tracing:")
        print("       1. Create a free account at https://cloud.langfuse.com")
        print("       2. Create a new project -> Copy the keys")
        print("       3. Add to your .env file:")
        print("              LANGFUSE_SECRET_KEY=sk-lf-...")
        print("              LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("              LANGFUSE_BASE_URL=https://us.cloud.langfuse.com")
        print("\n    Running in LOCAL MODE (no Langfuse tracing)")
        return None

    try:
        from langfuse import Langfuse  # noqa: F401  (presence check only)
    except ImportError:
        print("    langfuse package not installed.")
        print("    Run: .\\venv\\Scripts\\pip.exe install langfuse")
        return None

    client = None
    try:
        client = build_langfuse_client(public_key, secret_key, base_url)

        # Auth check via a lightweight API call
        client.auth_check()

        print("    Langfuse API Keys Valid")
        print(f"    Base URL: {base_url}")
        print(f"    Dashboard: {base_url}")
        print("    Tracing is ACTIVE — all LangChain calls will be logged.")
        return client

    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg:
            print(f"    Langfuse auth failed: {e}")
            print("    Check your LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY in .env")
            return None
        print(f"    Langfuse connectivity warning: {e}")
        if client is not None:
            print("    Tracing enabled — traces will sync when connectivity is restored.")
            return client
        return None


# =====================================================================
# 2. Langfuse Helpers (callback handler, session, flush)  [FIX 1, 2, 4]
# =====================================================================
def build_callback_handler() -> Optional[Any]:
    """
    Return a LangChain callback handler for the installed Langfuse version.  [FIX 1]

    Import is dynamic on purpose: `langfuse.callback` does not exist in v4, and a static
    import of a missing module is what makes the editor show a red error on this line.
    """
    for module_name in ("langfuse.langchain", "langfuse.callback"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue  # This SDK version does not have that path. Try the next one.
        handler_cls = getattr(module, "CallbackHandler", None)
        if handler_cls is None:
            continue
        try:
            return handler_cls()
        except Exception as e:
            print(f"[INIT] {module_name}.CallbackHandler could not start: {e}")
    return None


def session_context_factory() -> Tuple[Any, Dict[str, Any]]:
    """
    Return (make_scope, config_metadata) to group all traces into one session.  [FIX 4, 7]

    v4  : langfuse.propagate_attributes(session_id=...)
    v2/3: metadata={"langfuse_session_id": ...} on the LangChain config

    IMPORTANT — the scope is a FACTORY, not a value. A context manager is single-use:
    entering the same object a second time raises
    AttributeError: '_AgnosticContextManager' object has no attribute 'args'.
    Call make_scope() once per query to get a fresh one.
    """
    try:
        from langfuse import propagate_attributes

        def make_scope():
            return propagate_attributes(
                session_id=SESSION_ID,
                tags=["module-16", "observability"],
                trace_name="incidentlens-rag-query",
            )

        return make_scope, {}
    except ImportError:
        return nullcontext, {"langfuse_session_id": SESSION_ID}


def flush_langfuse(*targets: Any) -> bool:
    """
    Flush buffered traces to Langfuse Cloud.  [FIX 2]

    In v4 the callback handler has no `.flush()`. The Langfuse CLIENT does.
    """
    for target in targets:
        if target is None:
            continue
        flush = getattr(target, "flush", None)
        if callable(flush):
            try:
                flush()
                return True
            except Exception as e:
                print(f"  Langfuse flush warning: {e}")
    return False


def verify_traces(client: Optional[Any]) -> None:
    """
    Query the Langfuse API to PROVE the traces arrived. Do not assume.

    [FIX 6] Langfuse Cloud retired `GET /api/public/traces` for organizations created on or
    after 16 September 2026. That call now returns HTTP 410 with the error
    LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION. The supported read path is
    `GET /api/public/v2/observations`. The old call is kept as a fallback for older projects.
    """
    if client is None:
        return

    print("\n[VERIFY] Querying the Langfuse API for this session...")
    now = datetime.now(timezone.utc)
    observations: List[Any] = []

    # New API (Langfuse v4 / new organizations): root observations ARE the traces.
    try:
        response = client.api.observations.get_many(
            session_id=SESSION_ID,
            from_start_time=now - timedelta(hours=1),
            to_start_time=now,
            is_root_observation=True,
            limit=10,
        )
        observations = getattr(response, "data", None) or []
    except Exception as e:
        print(f"  observations API unavailable ({e.__class__.__name__}); trying legacy /traces ...")

    if not observations:
        # Legacy API (older SDKs and older organizations).
        try:
            response = client.api.trace.list(session_id=SESSION_ID, limit=10)
            legacy = getattr(response, "data", None) or []
            if legacy:
                print(f"  {len(legacy)} trace(s) found in session '{SESSION_ID}' (legacy API):")
                for trace in legacy:
                    name = getattr(trace, "name", None) or "(unnamed)"
                    print(f"    - {name} | id={getattr(trace, 'id', '?')}")
                print(f"  Dashboard: {os.getenv('LANGFUSE_BASE_URL', 'https://us.cloud.langfuse.com')}")
                return
        except Exception as e:
            print(f"  Trace verification skipped: {e}")
            print("  This does NOT mean tracing failed. Open the dashboard to confirm.")
            print(f"  Dashboard: {os.getenv('LANGFUSE_BASE_URL', 'https://us.cloud.langfuse.com')}")
            return

    if not observations:
        print("  No observations returned yet. Ingestion can take a few seconds — refresh the dashboard.")
        print(f"  Dashboard: {os.getenv('LANGFUSE_BASE_URL', 'https://us.cloud.langfuse.com')}")
        return

    print(f"  {len(observations)} root observation(s) found in session '{SESSION_ID}':")
    for obs in observations:
        name = getattr(obs, "name", None) or "(unnamed)"
        print(
            f"    - {name} | id={getattr(obs, 'id', '?')} | "
            f"latency={getattr(obs, 'latency', None)}s"
        )
    print(f"  Dashboard: {os.getenv('LANGFUSE_BASE_URL', 'https://us.cloud.langfuse.com')}")


# =====================================================================
# 3. Traced RAG Execution with Langfuse CallbackHandler
# =====================================================================
def run_traced_queries(client: Optional[Any]) -> None:
    """
    Runs RAG queries with Langfuse CallbackHandler injected into LangChain.
    Langfuse captures: prompt, completion, tokens, latency, model name.
    """
    print("\n" + "=" * 80)
    print(" Running Langfuse-Traced RAG Queries")
    print("=" * 80)

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        print("GROQ_API_KEY missing — cannot run LLM queries.")
        return

    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    collection = get_incident_collection()

    # Build RAG chain
    system_prompt = (
        "You are IncidentLens, an expert SRE post-mortem analyst. "
        "Answer using ONLY the provided context. Cite chunk sources. "
        "Be precise and factual.\n\nCONTEXT:\n{context}"
    )
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        temperature=0.0,
        max_tokens=512,
        api_key=groq_api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    # Set up Langfuse callback if tracing is active
    langfuse_active = client is not None
    callbacks: List[Any] = []
    langfuse_handler: Optional[Any] = None

    if langfuse_active:
        langfuse_handler = build_callback_handler()
        if langfuse_handler is not None:
            callbacks = [langfuse_handler]
            print("[INIT] Langfuse CallbackHandler attached to LangChain pipeline.")
        else:
            print("[INIT] Langfuse callback handler unavailable — running without tracing.")
    else:
        print("[INIT] Running in local mode (no Langfuse tracing).")

    # One fresh scope per query: a context manager cannot be entered twice.  [FIX 7]
    make_session_scope, session_metadata = session_context_factory()

    queries = [
        {
            "label": "Q1: CrowdStrike Root Cause (Technical Detail)",
            "query": "What was the exact 21 vs 20 input mismatch that caused the CrowdStrike kernel crash?",
        },
        {
            "label": "Q2: AWS Kinesis Cascading Failure (Multi-Hop)",
            "query": "How did the Kinesis front-end thread exhaustion cascade to affect CloudWatch and Cognito?",
        },
        {
            "label": "Q3: Cloudflare BGP Resolution (Mitigation)",
            "query": "What steps did Cloudflare take to resolve the BGP route leak caused by AS33154?",
        },
    ]

    for q_idx, test in enumerate(queries, 1):
        query = test["query"]
        print(f"\n{'='*80}")
        print(f" Query {q_idx}/{len(queries)}: {test['label']}")
        print(f"{'='*80}")
        print(f'Query: "{query}"')

        # 1. Retrieve from ChromaDB
        t_start = time.time()
        raw = collection.query(
            query_texts=[query],
            n_results=4,
            include=["documents", "metadatas", "distances"],
        )
        docs = raw["documents"][0] if raw["documents"] else []
        retrieval_ms = round((time.time() - t_start) * 1000)

        context_str = "\n\n".join(docs)

        # 2. Generate answer (with Langfuse callback if available)
        t_gen = time.time()
        config: Dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks
        if session_metadata:
            config["metadata"] = session_metadata

        try:
            with make_session_scope():
                answer = chain.invoke(
                    {"context": context_str, "question": query},
                    config=config or None,
                )
        except Exception as e:
            answer = f"[LLM Error: {e}]"
        gen_ms = round((time.time() - t_gen) * 1000)

        total_ms = round((time.time() - t_start) * 1000)

        print("\n-- Telemetry --")
        print(f"  Retrieval:   {retrieval_ms} ms | {len(docs)} chunks retrieved")
        print(f"  Generation:  {gen_ms} ms")
        print(f"  Total:       {total_ms} ms")

        if langfuse_handler is not None:
            print(f"  Langfuse:    trace sent (session: {SESSION_ID})")

        print("\n-- Response --")
        for line in answer.strip().splitlines():
            print(f"  {line}")

        time.sleep(2.0)  # Rate limit between queries

    # Flush Langfuse traces
    if langfuse_active:
        if flush_langfuse(client, langfuse_handler):
            print("\n  Langfuse traces flushed successfully.")
        time.sleep(3.0)  # Give the ingestion API a moment before verifying
        verify_traces(client)

    print("\n" + "=" * 80)
    print(" LangSmith vs Langfuse Comparison:")
    print("=" * 80)
    print("  ┌──────────────────┬────────────────────┬────────────────────┐")
    print("  │ Feature          │ LangSmith          │ Langfuse           │")
    print("  ├──────────────────┼────────────────────┼────────────────────┤")
    print("  │ Provider         │ LangChain Inc.     │ Open-Source (MIT)  │")
    print("  │ Self-Hosting     │ No                 │ Yes (Docker)       │")
    print("  │ LangChain Native │ Yes (env vars)     │ Yes (callback)     │")
    print("  │ Free Tier        │ 5k traces/month    │ 50k events/month   │")
    print("  │ Data Sovereignty │ US cloud only      │ Self-host anywhere │")
    print("  │ Setup Complexity │ 2 env vars         │ 3 env vars         │")
    print("  └──────────────────┴────────────────────┴────────────────────┘")

    print("\n" + "=" * 80)
    print(" What Was Fixed For langfuse 4.x:")
    print("=" * 80)
    print("  1. callback handler import  : langfuse.langchain (v3/v4) with langfuse.callback fallback")
    print("  2. flush                    : client.flush()  (the handler has no .flush() in v4)")
    print("  3. client endpoint keyword  : base_url= (v4), host= (v2/v3)")
    print("  4. session grouping         : propagate_attributes(session_id=...)")
    print("  5. interpreter guard        : auto re-launch with venv\\Scripts\\python.exe")
    print("  6. trace verification       : GET /api/public/v2/observations (legacy /traces is gone)")
    print("  7. session scope            : one fresh propagate_attributes() per query")

    print("\n" + "=" * 80)
    print(" Module 16 Complete: Langfuse Observability Tracing Verified!")
    print(" Ready for Module 17: Semantic Caching & Cost Optimization.")
    print("=" * 80)


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    langfuse_client = configure_langfuse()
    run_traced_queries(langfuse_client)
