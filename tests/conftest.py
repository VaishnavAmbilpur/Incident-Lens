"""
Shared pytest fixtures for the IncidentLens test suite.

The project stores one concept per NUMBERED module (01_..., 02_...). Python cannot import a
name that starts with a digit, so every module is loaded by file path instead.

Live-LLM tests are marked `@pytest.mark.llm` and are SKIPPED unless INCIDENTLENS_TEST_LLM=1.
That keeps `pytest tests/` green on a free tier that rate-limits, while still allowing the
full end-to-end check on demand:

    .\\venv\\Scripts\\python.exe -m pytest tests/ -v
    $env:INCIDENTLENS_TEST_LLM=1; .\\venv\\Scripts\\python.exe -m pytest tests/ -v
"""

import importlib.util
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = PROJECT_ROOT / "modules"


def load_module(filename: str, module_name: str):
    """Load a numbered module file by path."""
    spec = importlib.util.spec_from_file_location(module_name, str(MODULES_DIR / filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {filename}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pytest_collection_modifyitems(config, items):
    """Skip live-LLM tests unless the operator explicitly opted in."""
    if os.getenv("INCIDENTLENS_TEST_LLM", "") == "1":
        return
    skip = pytest.mark.skip(reason="live LLM test; set INCIDENTLENS_TEST_LLM=1 to run it")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def security_module():
    """Module 18: the security gate. No LLM calls, so its tests always run."""
    return load_module("18_security_guardrails.py", "m18_security")


@pytest.fixture(scope="session")
def retrieval_stack():
    """
    Module 05 (ChromaDB) + module 10 (BM25/dense/RRF).

    Returns (collection, engine, hybrid_module). Building the engine chunks the corpus and
    fits BM25, so it is session-scoped: about 5-10 seconds, once.
    """
    m05 = load_module("05_vector_stores_chroma.py", "m05_chroma")
    m10 = load_module("10_hybrid_search_bm25.py", "m10_hybrid")
    collection = m05.get_incident_collection()
    chunks = m05.get_production_chunks()
    engine = m10.HybridSearchEngine(chunks, collection)
    return collection, engine, m10


@pytest.fixture(scope="session")
def agent_module():
    """Module 19: the LangGraph agent."""
    return load_module("19_langgraph_agent.py", "m19_agent")


@pytest.fixture(scope="session")
def api_client():
    """
    Module 20 behind FastAPI's TestClient.

    Using the client as a context manager runs the app's lifespan, which is what builds the
    retrieval engine and compiles the graph. No HTTP server and no port are involved.
    """
    from fastapi.testclient import TestClient

    mod20 = load_module("20_api_server.py", "m20_api")
    with TestClient(mod20.app) as client:
        yield client
