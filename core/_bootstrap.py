"""
Bootstrap for a direct run of a core module.

The problem
-----------
A module is imported in two ways:

    python core/evaluate.py        the script directory goes on sys.path
    python -m core.evaluate        the project root goes on sys.path

In the first form, "from core.ingest import ..." fails, because the project root
is not on sys.path and the package "core" cannot be found.

The solution
------------
This file holds no imports from the project. Therefore it can always be
imported first. It adds the project root to sys.path. Every core module calls
it before any other project import.
"""

from __future__ import annotations

import sys
from pathlib import Path

# core/_bootstrap.py -> core/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
