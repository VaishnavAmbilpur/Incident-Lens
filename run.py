r"""
IncidentLens runner - works around the _ctypes DLL issue in venvs.

Usage:
    C:\Python313\python.exe run.py server     - starts the API server
    C:\Python313\python.exe run.py diagnose   - runs the diagnostic tool
    C:\Python313\python.exe run.py ablation   - runs the ablation table
    C:\Python313\python.exe run.py eval       - runs the 4-metric evaluation
    C:\Python313\python.exe run.py eval --llm - runs evals including LLM-judged metrics
    C:\Python313\python.exe run.py test       - tests all imports and connections
"""

import os
import sys
from pathlib import Path

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure the venv site-packages are on the path
PROJECT = Path(__file__).resolve().parent
VENV_SP = PROJECT / "venv" / "Lib" / "site-packages"

if str(VENV_SP) not in sys.path:
    sys.path.insert(0, str(VENV_SP))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
except ImportError:
    pass


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "server":
        import uvicorn
        uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)

    elif cmd == "diagnose":
        import subprocess
        args = [sys.executable] + [str(PROJECT / "diagnose.py")] + sys.argv[2:]
        env = {**os.environ, "PYTHONPATH": str(PROJECT) + os.pathsep + str(VENV_SP)}
        sys.exit(subprocess.run(args, env=env).returncode)

    elif cmd == "ablation":
        import subprocess
        extra = ["--json"] if "--json" in sys.argv else []
        args = [sys.executable, str(PROJECT / "ablation.py")] + extra + [
            a for a in sys.argv[2:] if a not in extra
        ]
        env = {**os.environ, "PYTHONPATH": str(PROJECT) + os.pathsep + str(VENV_SP)}
        sys.exit(subprocess.run(args, env=env).returncode)

    elif cmd == "eval":
        import subprocess
        extra = []
        if "--llm" in sys.argv:
            extra.append("--llm")
        if "--json" in sys.argv:
            extra.append("--json")
        args = [sys.executable, str(PROJECT / "core" / "evaluate.py")] + extra
        env = {**os.environ, "PYTHONPATH": str(PROJECT) + os.pathsep + str(VENV_SP)}
        sys.exit(subprocess.run(args, env=env).returncode)

    elif cmd == "test":
        print("Testing all core imports...")
        from core import (ingest, chunking, embeddings, vectorstore,
                          retrieval, bm25, fusion, rerank,
                          contextual, generate, evaluate, providers)
        print("[PASS] All 12 core modules imported successfully")

        print("\nTesting ChromaDB connection...")
        from core.vectorstore import VectorIndex
        idx = VectorIndex()
        col = idx.collection()
        count = col.count()
        print(f"[PASS] ChromaDB collection: {col.name}, vectors: {count}")

        print("\nTesting embedding model...")
        from core.embeddings import embed_query
        vec = embed_query("test query")
        print(f"[PASS] Embedding model works: {len(vec)}-dim vector")

        print("\nTesting BM25 retriever...")
        from core.bm25 import BM25Retriever
        print("[PASS] BM25 importable")

        print("\nTesting reranker model...")
        from core.rerank import CrossEncoderReranker
        ranker = CrossEncoderReranker()
        print(f"[PASS] CrossEncoderReranker loaded")

        print("\nTesting provider chain...")
        from core.providers import complete, Completion
        print("[PASS] Provider complete() function available")

        print("\n" + "=" * 50)
        print("ALL SYSTEMS OPERATIONAL")
        print("=" * 50)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
