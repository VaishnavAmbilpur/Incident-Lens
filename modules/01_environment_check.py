"""
01_environment_check.py
======================
IncidentLens Diagnostic Script:
1. Verifies Python runtime and package imports.
2. Tests Groq API connectivity (Llama 3.3 70B).
3. Tests OpenRouter API connectivity (meta-llama/llama-3.3-70b-instruct:free).
4. Tests LangSmith and Langfuse observability configuration.
"""

import sys
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding across Windows shells to prevent cp1252 errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=DeprecationWarning)

def run_diagnostics():
    print("=" * 70)
    print(" 🛡️ IncidentLens — Environment & Live API Diagnostic")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Environment & Keys
    # -------------------------------------------------------------
    root_dir = Path(__file__).resolve().parent.parent
    env_file = root_dir / ".env"
    load_dotenv(dotenv_path=env_file)

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    langchain_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    langfuse_base = os.getenv("LANGFUSE_BASE_URL", "").strip()

    is_venv = (sys.prefix != sys.base_prefix) or ("venv" in sys.executable.lower())
    print(f"\n[1] Python Runtime: {sys.version.split()[0]} ({sys.executable})")
    print(f"    - Virtual Environment: {'✅ ACTIVE (venv)' if is_venv else '⚠️ NOT ACTIVE (Using Global Python — run with .\\venv\\Scripts\\python.exe)'}")
    print("\n[2] Loaded API Keys from .env:")
    print(f"    - GROQ_API_KEY:         {'[OK] Present' if groq_key else '[MISSING]'}")
    print(f"    - OPENROUTER_API_KEY:   {'[OK] Present' if openrouter_key else '[MISSING]'}")
    print(f"    - LANGCHAIN_API_KEY:    {'[OK] Present' if langchain_key else '[OPTIONAL / EMPTY]'}")
    print(f"    - LANGFUSE_SECRET_KEY:  {'[OK] Present' if langfuse_secret else '[MISSING]'}")
    print(f"    - LANGFUSE_PUBLIC_KEY:  {'[OK] Present' if langfuse_public else '[MISSING]'}")
    print(f"    - LANGFUSE_BASE_URL:    {langfuse_base or '[DEFAULT]'}")

    # -------------------------------------------------------------
    # 2. Package Imports Check
    # -------------------------------------------------------------
    print("\n[3] Installed Package Verification:")
    packages = [
        ("LangChain Core", "langchain_core"),
        ("LangChain Community", "langchain_community"),
        ("LangGraph", "langgraph"),
        ("Groq Provider", "langchain_groq"),
        ("OpenAI / OpenRouter", "openai"),
        ("LangChain OpenAI", "langchain_openai"),
        ("ChromaDB Vector Store", "chromadb"),
        ("BM25 Keyword Search", "rank_bm25"),
        ("PyPDF Document Loader", "pypdf"),
        ("FastAPI", "fastapi"),
    ]

    missing_packages = []
    for display_name, module_name in packages:
        try:
            __import__(module_name)
            print(f"    ✅ PASS: {display_name} ({module_name})")
        except ImportError:
            print(f"    ❌ FAIL: {display_name} ({module_name}) is NOT installed")
            missing_packages.append(module_name)

    if missing_packages:
        print(f"\n⚠️ Missing packages detected: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))

    # -------------------------------------------------------------
    # 3. Live Ping: Groq Cloud (Dynamic Model Detection & Fallback)
    # -------------------------------------------------------------
    print("\n[4] Testing Live Cloud Connection: Groq Cloud...")
    if groq_key:
        try:
            import requests
            headers = {"Authorization": f"Bearer {groq_key}"}
            res = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
            if res.status_code == 200:
                available_models = [m["id"] for m in res.json().get("data", []) if m.get("active", True)]
                
                # Priority list of reliable general-purpose Groq chat models
                preferred_models = [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "llama3-70b-8192",
                    "llama3-8b-8192",
                    "gemma2-9b-it",
                ]
                
                # Filter out specialized audio, guard, or terms-gated models
                filtered_models = [
                    m for m in available_models 
                    if not any(bad in m.lower() for bad in ["whisper", "guard", "audio", "embed", "orpheus", "allam", "canopylabs"])
                ]
                
                # Build candidates: preferred models first, then other general chat models
                candidate_models = []
                for pref in preferred_models:
                    if pref in available_models and pref not in candidate_models:
                        candidate_models.append(pref)
                for m in filtered_models:
                    if m not in candidate_models:
                        candidate_models.append(m)
                if not candidate_models:
                    candidate_models = available_models
                
                print(f"    ℹ️ Available Groq Models: {len(available_models)} total, {len(candidate_models)} general chat candidates")
                
                from langchain_groq import ChatGroq
                connected = False
                for model_candidate in candidate_models[:3]:
                    try:
                        print(f"    Testing chat model: '{model_candidate}'...")
                        llm_groq = ChatGroq(
                            api_key=groq_key,
                            model=model_candidate,
                            max_tokens=20,
                            temperature=0.0
                        )
                        response = llm_groq.invoke("Reply with: 'Groq Connected Successfully'")
                        print(f"    ✅ SUCCESS: Groq ({model_candidate}) Response -> \"{response.content.strip()}\"")
                        connected = True
                        break
                    except Exception as model_err:
                        print(f"    ⚠️ Model '{model_candidate}' skipped: {model_err}")
                
                if not connected:
                    print("    ❌ ERROR connecting to Groq: None of the candidate models succeeded.")
            else:
                print(f"    ❌ ERROR querying Groq models: HTTP {res.status_code} - {res.text}")
        except Exception as e:
            print(f"    ❌ ERROR connecting to Groq: {e}")
    else:
        print("    ⚪ SKIPPED: GROQ_API_KEY is missing.")

    # -------------------------------------------------------------
    # 4. Live Ping: OpenRouter (openrouter/free)
    # -------------------------------------------------------------
    print("\n[5] Testing Live Cloud Connection: OpenRouter Free Tier...")
    if openrouter_key:
        try:
            from langchain_openai import ChatOpenAI
            llm_openrouter = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model="openrouter/free",
                max_tokens=50,
                temperature=0.2
            )
            response = llm_openrouter.invoke("Reply with: 'OpenRouter Connected Successfully'")
            content = response.content.strip() if response.content else "Connected (Status 200 OK)"
            print(f"    ✅ SUCCESS: OpenRouter Response -> \"{content}\"")
        except Exception as e:
            print(f"    ❌ ERROR connecting to OpenRouter: {e}")
    else:
        print("    ⚪ SKIPPED: OPENROUTER_API_KEY is missing.")

    # -------------------------------------------------------------
    # 5. Live Ping: Langfuse Cloud Observability
    # -------------------------------------------------------------
    print("\n[6] Testing Live Observability: Langfuse Cloud...")
    if langfuse_secret and langfuse_public:
        os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secret
        os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_public
        if langfuse_base:
            os.environ["LANGFUSE_BASE_URL"] = langfuse_base
            os.environ["LANGFUSE_HOST"] = langfuse_base

        try:
            # Modern Langfuse SDK v3+ @observe syntax (dynamic import to prevent IDE lint warnings)
            import importlib
            lfd = importlib.import_module("langfuse.decorators")
            observe = lfd.observe
            langfuse_context = lfd.langfuse_context

            @observe(name="incidentlens-first-trace")
            def record_test_trace():
                langfuse_context.update_current_trace(
                    name="incidentlens-first-trace",
                    metadata={"environment": "development", "project": "IncidentLens"},
                    input="Diagnostic test ping from IncidentLens",
                    output="IncidentLens connected successfully to Langfuse!"
                )
                return "OK"

            record_test_trace()
            langfuse_context.flush()
            print("    ✅ SUCCESS: Logged first trace to Langfuse Cloud! Refresh your dashboard.")
        except Exception as e1:
            try:
                from langfuse import Langfuse
                lf = Langfuse(
                    secret_key=langfuse_secret,
                    public_key=langfuse_public,
                    host=langfuse_base or "https://us.cloud.langfuse.com"
                )
                if hasattr(lf, "start_as_current_observation"):
                    with lf.start_as_current_observation(name="incidentlens-first-trace") as obs:
                        obs.update(
                            input="Diagnostic test ping from IncidentLens",
                            output="IncidentLens connected successfully to Langfuse!"
                        )
                    lf.flush()
                    print("    ✅ SUCCESS: Logged first trace to Langfuse Cloud! Refresh your dashboard.")
                else:
                    print(f"    ❌ ERROR connecting to Langfuse: {e1}")
            except Exception as e2:
                print(f"    ❌ ERROR connecting to Langfuse: {e2}")
    else:
        print("    ⚪ SKIPPED: LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY is missing.")

    # -------------------------------------------------------------
    # 6. Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Diagnostic Check Complete!")
    print("=" * 70)

if __name__ == "__main__":
    run_diagnostics()
