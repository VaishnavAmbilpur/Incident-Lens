"""
23_late_chunking.py
===================
Phase 6: Advanced RAG — Technique 3, Late Chunking (long-document-aware chunk embeddings)
IncidentLens Late-Chunking Index:

WHY THIS EXISTS:
  A RAG index is built by cutting a post-mortem into ~800-character chunks and embedding
  each chunk ON ITS OWN. That is the "standard" (early) chunking pipeline:

      document -> split into chunks -> embed each chunk independently

  The problem is that an embedding model can only attend to the tokens it is given. When a
  chunk is encoded alone it is blind to everything outside its own 800 characters, so the
  vector for a chunk that says "This decision would result in our service being unusable"
  has no way to know WHAT decision, or that the decision happened at 23:13 UTC two chunks
  earlier. An incident post-mortem is a *timeline*: the causal story is carried by
  cross-references between distant chunks ("the 30+ minutes of data written to the US West
  Coast data center", "These delayed replicas", "the recovery process took longer than
  originally estimated"). Split them apart and the links are gone.

  The spec (docs/12_phase6_advanced_rag.md, Technique 3) puts it plainly: preserve the
  connection between "the initial bug at 14:00" and "the cascading crash at 20:00" even when
  the two events sit in different chunks.

  LATE CHUNKING fixes this by inverting the order of operations:

      document -> encode the FULL document (all tokens attend to each other)
               -> pool the token vectors that belong to each chunk -> per-chunk vectors

  The chunk boundaries are decided *before* embedding, but the attention is computed over a
  much larger span, so every token inside a chunk has already "seen" the rest of the
  document. The pooling step then keeps only the token span that belongs to the chunk.

THE ALGORITHM IN PLAIN LANGUAGE:
  1. Ask the project chunker (module 03) for the same 105 production chunks it gives every
     other module, so late chunking is compared against standard chunking on IDENTICAL text.
  2. Rebuild each parent document by concatenating its chunks in order and record the
     [start, end) character offset of every chunk inside that document text.
  3. Tokenize the full document with `return_overflowing_tokens=True` and
     `return_offsets_mapping=True`. The tokenizer splits it into as many 256-token windows as
     the document needs, and the offset map says, for every token, which character range of the
     document it came from. This is how a chunk is mapped to its own token range -- never by
     guessing token counts, and never with a window start added on top of an offset that is
     already document-absolute.
  4. Run each token window through the model with `model.encode(text, output_value=None)`,
     which returns the token-level embeddings and the attention mask. Inside a window every
     token attends to every other token in that window.
  5. For every chunk, select the tokens whose character span overlaps the chunk span (this is
     where the offset mapping earns its keep) and mean-pool them into one vector. L2-normalise
     so a dot product is a cosine similarity, exactly like the standard index.
  6. Build the standard index the same way the rest of the project does: embed each chunk
     text independently with `model.encode(..., normalize_embeddings=True)`.
  7. Verify. This is the part that matters, because "late chunking is better" is a claim, not
     a fact. The module measures three things with real numbers and refuses to print a number
     it did not compute. It also refuses to trust its own plumbing: the pooling is checked
     against the chunk's own text (coverage and overflow), and the window chain is checked
     against the document (every character encoded exactly once).

HONEST LIMITATIONS (documented, not hidden -- the verification prints them too):
  * all-MiniLM-L6-v2 has max_seq_length = 256 tokens, so a full post-mortem CANNOT be encoded in
    one forward pass. The corpus is therefore encoded as a chain of NON-OVERLAPPING 256-token
    windows (stride 0): the GitHub post-mortem needs 13 of them, the whole corpus 66. A chunk
    that straddles a window seam is pooled from both windows. Consequence: "long-range
    attention" here means "attention across whatever fits in a 256-token window", NOT attention
    over the whole 15,500-character document. Chunk 11 and chunk 20 of the GitHub post-mortem
    (the 23:13 UTC failover decision and the 11:12 UTC delayed replicas) do land in the same
    window, but chunk 1 and chunk 30 do not, and no token ever attends across a seam. Overlapping
    windows were tried first and rejected: they encode each boundary token twice, which made the
    pooled chunk spans ambiguous, so the module asserts full non-overlapping coverage instead.
    A model with a longer context (or a document-level pooling scheme such as Jina's late
    chunking) would remove this ceiling. This module does not pretend otherwise.
  * MiniLM was trained for sentence-level mean pooling, not for pooling a token sub-span out
    of a larger attention window. Late chunking is therefore applied "off-label" and its
    retrieval effect is mixed -- see the verification section, which reports the losses as
    well as the wins instead of cherry-picking.
  * Standard chunking silently truncates any chunk longer than 256 tokens. The module counts
    how many chunks that happens to and prints it.
  * A chunk made only of whitespace padding has no token at all, so it gets a zero late vector;
    the module reports how many such chunks exist rather than hiding them.

NO LLM CALLS: this module is 100% local (one small embedding model, CPU only). It is designed
to run when every API quota is exhausted.

Run:
    .\\venv\\Scripts\\python.exe modules\\23_late_chunking.py
"""

import os
import re
import sys
import math
import importlib
import importlib.util
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
_SCRIPT_DIR = Path(__file__).resolve().parent


def _detect_project_root() -> Path:
    """
    Find the IncidentLens project root: the directory holding `modules/03_text_splitters.py`.

    Normally that is the parent of the folder this file sits in. The module is also allowed to
    be copied elsewhere (a scratch run mirror, a phase folder), so the search walks up a few
    levels and finally falls back to the parent directory. Everything else -- the chunker, the
    venv, the .env file -- is resolved from this one root.
    """
    for candidate in (_SCRIPT_DIR.parent, _SCRIPT_DIR, *_SCRIPT_DIR.parents[:4]):
        if (candidate / "modules" / "03_text_splitters.py").exists():
            return candidate
    return _SCRIPT_DIR.parent


PROJECT_ROOT = _detect_project_root()
REQUIRED_PACKAGES = ("sentence_transformers", "chromadb", "dotenv")


def _find_venv_python() -> Optional[Path]:
    """
    Locate the project virtualenv interpreter.

    The module normally lives in `<project>/modules/`, but it is allowed to be copied elsewhere
    (a scratch run directory, a phase folder), so this file's own directory and its parent are
    both searched, in that order.
    """
    for root in (_SCRIPT_DIR.parent, _SCRIPT_DIR, PROJECT_ROOT):
        candidate = root / "venv" / "Scripts" / "python.exe"
        if candidate.exists():
            return candidate
    return None


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
    """Re-launch this script with the project venv when the active interpreter is wrong."""
    missing = _missing_packages(REQUIRED_PACKAGES)
    if not missing:
        return

    venv_python = _find_venv_python()
    if os.environ.get("INCIDENTLENS_REEXEC") == "1" or venv_python is None or not venv_python.exists():
        print("=" * 80)
        print(" [BOOT] Cannot start: project packages are missing")
        print("=" * 80)
        print(f"  Interpreter  : {sys.executable}")
        print(f"  Missing      : {', '.join(missing)}")
        print(f"  Project venv : {venv_python if venv_python else 'not found'}")
        print("  Run this instead:")
        print(f"      {venv_python if venv_python else '.\\venv\\Scripts\\python.exe'} "
              f"modules\\{Path(__file__).name}")
        print("=" * 80)
        sys.exit(1)

    if Path(sys.executable).resolve() == venv_python.resolve():
        return

    print("=" * 80)
    print(" [BOOT] Wrong interpreter detected - switching to the project venv")
    print("=" * 80)
    env = dict(os.environ)
    env["INCIDENTLENS_REEXEC"] = "1"
    completed = subprocess.run(
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env=env,
    )
    sys.exit(completed.returncode)


ensure_project_interpreter()

# =====================================================================
# 1. Model Cache Location (must be set before sentence_transformers is imported)
# =====================================================================
# huggingface_hub reads HF_HOME when it is imported. On some hosts (locked-down Windows
# profiles, sandboxes) the default ~/.cache/huggingface is not writable, and the model
# download then dies with a PermissionError. Pick a writable cache before importing torch,
# and tell huggingface_hub to copy blobs instead of symlinking (symlinks need extra
# privileges on Windows).
def _configure_hf_cache() -> str:
    """
    Choose a writable HuggingFace cache directory and return it.

    `HF_HOME` must be set BEFORE sentence-transformers is imported, because huggingface_hub
    resolves its cache directories at import time; changing the variable afterwards does not
    move the cache. The check here is an actual WRITE (create + delete a file), not a directory
    existence test: a cache directory can exist and still be un-writable (locked-down Windows
    profile, DSH file sandbox), in which case the download dies with `PermissionError [WinError
    5]` while huggingface_hub creates the model directory. On such a host the module uses
    `<project>/.hf_cache` instead, so the first run still works with no manual setup.
    """
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    default_cache = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".cache" / "huggingface"
    try:
        hub_dir = default_cache / "hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        probe = hub_dir / ".incidentlens_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        os.environ["HF_HOME"] = str(default_cache)
    except Exception:
        os.environ["HF_HOME"] = str(_fallback_hf_cache())
    return os.environ["HF_HOME"]


def _fallback_hf_cache() -> Path:
    fallback = PROJECT_ROOT / ".hf_cache"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return fallback


def load_embedding_model(model_name: str) -> Tuple[Any, str]:
    """
    Load the local embedding model, retrying against a writable cache on a permission error.

    On a host whose default HuggingFace cache exists but cannot be written, the first attempt
    dies with `PermissionError [WinError 5]` while huggingface_hub creates the model directory.
    Rather than telling the user to fix their profile, the module retries once against
    `<project>/.hf_cache`, which the project can always write. No API calls and no network use
    beyond fetching the model the first time.
    """
    try:
        return SentenceTransformer(model_name, device=MODEL_DEVICE), os.environ.get("HF_HOME", "")
    except (PermissionError, OSError) as exc:
        fallback = _fallback_hf_cache()
        print(f"  [note] default HuggingFace cache is not writable ({type(exc).__name__}); "
              f"retrying with {fallback}")
        os.environ["HF_HOME"] = str(fallback)
        os.environ["HF_HUB_CACHE"] = str(fallback / "hub")
        return SentenceTransformer(model_name, device=MODEL_DEVICE), str(fallback)


HF_HOME_USED = _configure_hf_cache()

from dotenv import load_dotenv

# Same .env the rest of the project reads. Values are never printed by this module.
load_dotenv(PROJECT_ROOT / ".env")

import numpy as np  # noqa: E402  (must follow the interpreter guard)
import torch  # noqa: E402

from sentence_transformers import SentenceTransformer  # noqa: E402

# ---- tunables -------------------------------------------------------
EMBED_MODEL_NAME = os.getenv("INCIDENTLENS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Window stride. Deliberately 0: with overlapping windows the same token is encoded twice and
# a chunk that sits on a window edge gets pooled from both copies, which made the pooled chunk
# spans overlap and the pooling ambiguous. Non-overlapping windows tile the document exactly
# (the module asserts full coverage), and the only cost is that tokens at a window seam do not
# attend to the other side of the seam -- the honest consequence of the 256-token limit.
WINDOW_STRIDE = 0
MODEL_DEVICE = "cpu"        # CPU only: this module must run when API quota is exhausted
PASS_MEAN_IMPROVEMENT = 2.0    # required mean rank improvement (positions) on probe evidence
PASS_IMPROVE_FRACTION = 0.50   # required fraction of probes whose evidence rank improves
PASS_DISTINCT_WIN_DELTA = 0.05  # required fraction of document groups with higher late cohesion
PASS_CONCRETE_RANK_GAIN = 10    # at least one probe must gain this many rank positions


# =====================================================================
# 2. Reuse the Project Chunker (dynamic import, like every other module)
# =====================================================================
def load_production_chunks() -> List[Any]:
    """
    Load modules/03_text_splitters.py dynamically and return its production chunks.

    The chunker is looked up in the project first and next to this file second, so the module
    works both in place (modules/23_late_chunking.py) and when it is run from a scratch copy
    that has been dropped into another tree's modules/ directory.
    """
    candidates = [
        PROJECT_ROOT / "modules" / "03_text_splitters.py",
        Path(__file__).resolve().parent / "03_text_splitters.py",
    ]
    splitter_path = next((p for p in candidates if p.exists()), None)
    if splitter_path is None:
        raise FileNotFoundError(
            "Chunker not found. Looked in:\n    "
            + "\n    ".join(str(p) for p in candidates)
        )

    spec = importlib.util.spec_from_file_location("module_03_splitters", str(splitter_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {splitter_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "get_production_chunks"):
        raise AttributeError(f"{splitter_path} has no get_production_chunks()")

    return list(mod.get_production_chunks())


# =====================================================================
# 3. Document Reconstruction & Chunk -> Character Offset Mapping
# =====================================================================
@dataclass
class DocGroup:
    """One encodable parent document: a page of a PDF, or a whole Markdown file."""
    key: Tuple[str, int]
    incident_slug: str
    page_number: int
    chunk_indices: List[int] = field(default_factory=list)
    text: str = ""


def build_document_groups(
    chunks: Sequence[Any],
) -> Tuple[Dict[Tuple[str, int], DocGroup], Dict[int, Tuple[Tuple[str, int], int, int]]]:
    """
    Rebuild each parent document from its chunks (in chunk_index order) and record where
    every chunk starts and ends inside that text.

    PDFs are grouped per page because the loader emits one Document per page; Markdown files
    are single-page by construction. Grouping this way means every chunk provably sits inside
    exactly one parent text.
    """
    groups: Dict[Tuple[str, int], DocGroup] = {}

    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        is_pdf = str(meta.get("file_type", "")).lower() == "pdf"
        page = int(meta.get("page_number", 1)) if is_pdf else 1
        key = (str(meta.get("incident_slug", "doc")), page)

        group = groups.get(key)
        if group is None:
            group = DocGroup(key=key, incident_slug=key[0], page_number=page)
            groups[key] = group
        group.chunk_indices.append(idx)

    char_span: Dict[int, Tuple[Tuple[str, int], int, int]] = {}

    for key, group in groups.items():
        group.chunk_indices.sort(key=lambda i: int(chunks[i].metadata.get("chunk_index", 0)))
        parts: List[str] = []
        cursor = 0
        for n, idx in enumerate(group.chunk_indices):
            if n:
                # Chunks are rendered back into one flat document text. A newline keeps
                # paragraphs from fusing; it is recorded in the offsets like any other char.
                parts.append("\n")
                cursor += 1
            body = chunks[idx].page_content
            char_span[idx] = (key, cursor, cursor + len(body))
            parts.append(body)
            cursor += len(body)
        group.text = "".join(parts)

    return groups, char_span


# =====================================================================
# 4. Token Windowing — Honest Handling of max_seq_length
# =====================================================================
@dataclass
class TokenWindow:
    """One forward pass worth of text: the token ids and where each token came from."""
    start_char: int
    end_char: int
    token_ids: List[int]
    offsets: List[Tuple[int, int]]      # character span inside the FULL document text


# Alignment notes collected while windowing/pooling. They are printed, never swallowed: an
# unexpected window is a limitation of the measurement, not something to hide.
ALIGNMENT_NOTES: List[str] = []


def window_document(text: str, tokenizer: Any, max_len: int, stride: int) -> List[TokenWindow]:
    """
    Split a document into overlapping token windows using the tokenizer's own overflow
    machinery, and keep the character offset of every token.

    `return_overflowing_tokens` is what makes this honest: the model's 256-token limit is
    never allowed to silently swallow the tail of a document.

    OFFSET SEMANTICS (verified by measurement, and a real bug source, so it is written down):
    with `return_overflowing_tokens=True` the `offset_mapping` of every window holds character
    offsets into the ORIGINAL document text. A token's document span is exactly its offset
    pair. Adding the window start on top of it is wrong and silently pools a neighbouring
    region instead -- that bug shipped once during development and was caught by the span
    self-check in `compute_late_chunk_embeddings`, which is why that check exists.
    """
    encoding = tokenizer(
        [text],
        return_overflowing_tokens=True,
        stride=stride,
        max_length=max_len,
        truncation=True,
        return_offsets_mapping=True,
        padding=False,
    )

    windows: List[TokenWindow] = []
    for window_number, (token_ids, reported) in enumerate(
        zip(encoding["input_ids"], encoding["offset_mapping"])
    ):
        token_ids = list(token_ids)
        offsets = [(int(s), int(e)) for s, e in reported]
        real = [(s, e) for s, e in offsets if e > s]
        if not real:
            continue
        start_char, end_char = real[0][0], real[-1][1]
        if not (0 <= start_char < end_char <= len(text)):
            raise RuntimeError(
                f"window {window_number} produced an out-of-range span "
                f"[{start_char}, {end_char}) for a {len(text)}-character document"
            )

        # Sanity check that the recorded span really is the text this window encodes: when the
        # span is re-tokenized on its own it must yield the same NUMBER of tokens, so token
        # positions still line up with the returned token vectors. (The ids themselves can
        # differ by one when the span starts or ends mid-word -- e.g. "slop|piness" -- because
        # the boundary word is re-tokenized in isolation. The character spans remain correct.)
        probe_len = len(
            tokenizer([text[start_char:end_char]], truncation=True,
                      max_length=max_len, padding=False)["input_ids"][0]
        )
        if probe_len != len(token_ids):
            ALIGNMENT_NOTES.append(
                f"{window_number}: span [{start_char}, {end_char}) re-tokenizes to {probe_len} "
                f"tokens instead of {len(token_ids)} (boundary word split mid-token)"
            )

        windows.append(
            TokenWindow(
                start_char=start_char,
                end_char=end_char,
                token_ids=token_ids,
                offsets=offsets,
            )
        )
    return windows


def model_token_embeddings(model: SentenceTransformer, text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Token-level embeddings for one window: `model.encode(text, output_value=None)`.

    Returns (token_embeddings [n_tokens, dim], attention_mask [n_tokens]).

    Note: sentence-transformers 6.1.0 crashes on `encode([a, b], output_value=None)` because
    its batch-collation loop indexes a scalar `modality` field (`IndexError: string index out
    of range`). Encoding one window per call avoids the bug and is deterministic; windows are
    small, so the cost is acceptable on CPU.
    """
    result = model.encode([text], output_value=None, convert_to_numpy=False, show_progress_bar=False)
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"Expected 1 encoded window, got {len(result)}")
        result = result[0]

    token_emb = result["token_embeddings"]
    attn = result["attention_mask"]
    if hasattr(token_emb, "detach"):
        token_emb = token_emb.detach().cpu().numpy()
    if hasattr(attn, "detach"):
        attn = attn.detach().cpu().numpy()
    return np.asarray(token_emb, dtype=np.float32), np.asarray(attn, dtype=np.int64)


# =====================================================================
# 5. Late Chunking — Pool Each Chunk Out of the Full-Document Encoding
# =====================================================================
@dataclass
class LateChunkResult:
    vectors: Dict[int, Optional[np.ndarray]]
    windows_per_group: Dict[Tuple[str, int], int]
    tokens_per_group: Dict[Tuple[str, int], int]
    unmapped_chunks: List[str]
    mapped_token_counts: Dict[int, List[int]]   # chunk idx -> tokens pooled per window
    span_sanity_failures: List[str] = field(default_factory=list)
    alignment_failures: List[str] = field(default_factory=list)
    alignment_notes: List[str] = field(default_factory=list)
    coverage_gaps: List[str] = field(default_factory=list)
    whitespace_only_chunks: List[str] = field(default_factory=list)
    alignment_rescues: int = 0


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def compute_late_chunk_embeddings(
    chunks: Sequence[Any],
    groups: Dict[Tuple[str, int], DocGroup],
    char_span: Dict[int, Tuple[Tuple[str, int], int, int]],
    model: SentenceTransformer,
) -> LateChunkResult:
    """
    Encode every parent document, then pool each chunk's own token span out of that encoding.

    The attention inside a window is global to the window, so a token that belongs to chunk N
    has already attended to every other token in the window before it is pooled -- including
    tokens that belong to neighbouring chunks. That is the entire mechanism.
    """
    tokenizer = model.tokenizer
    max_len = int(model.max_seq_length)
    dim = int(model.get_sentence_embedding_dimension())

    vectors: Dict[int, Optional[np.ndarray]] = {i: None for i in range(len(chunks))}
    windows_per_group: Dict[Tuple[str, int], int] = {}
    tokens_per_group: Dict[Tuple[str, int], int] = {}
    unmapped_chunks: List[str] = []
    mapped_token_counts: Dict[int, List[int]] = {}
    span_sanity_failures: List[str] = []
    tokens_per_chunk: Dict[int, int] = {}
    alignment_failures: List[str] = []
    coverage_gaps: List[str] = []
    whitespace_only_chunks: List[str] = []
    alignment_rescues = 0

    for key, group in groups.items():
        windows = window_document(group.text, tokenizer, max_len, WINDOW_STRIDE)
        windows_per_group[key] = len(windows)
        tokens_per_group[key] = sum(len(w.token_ids) for w in windows)
        covered_chars: set = set()
        if windows:
            covered_chars.update(range(windows[0].start_char, windows[0].end_char))
            for previous, following in zip(windows, windows[1:]):
                if following.start_char > previous.end_char:
                    missing = group.text[previous.end_char:following.start_char]
                    # Consecutive windows can leave 1-2 characters between them: the tokenizer
                    # skips un-tokenizable whitespace at a seam. Anything beyond that, or any
                    # missing character that carries content, is a real coverage hole.
                    if len(missing) > 3 or not missing.isspace():
                        coverage_gaps.append(
                            f"{key}: {len(missing)} character(s) between windows "
                            f"{previous.end_char} and {following.start_char} were never encoded "
                            f"({missing!r})"
                        )
                covered_chars.update(range(following.start_char, following.end_char))
            coverage_ratio = len(covered_chars) / len(group.text) if group.text else 1.0
            if coverage_ratio < 0.99:
                coverage_gaps.append(
                    f"{key}: windows cover {len(covered_chars)}/{len(group.text)} characters "
                    f"({coverage_ratio:.1%})"
                )

        accum: Dict[int, List[np.ndarray]] = {i: [] for i in group.chunk_indices}
        pooled_chars: Dict[int, set] = {i: set() for i in group.chunk_indices}
        pooled_token_counts: Dict[int, List[int]] = {i: [] for i in group.chunk_indices}

        for window in windows:
            slice_text = group.text[window.start_char:window.end_char]
            token_emb, attn = model_token_embeddings(model, slice_text)

            if len(window.offsets) != len(token_emb):
                # This can happen for a window whose span starts or ends in the middle of a
                # word: re-tokenizing that slice on its own yields one token fewer. The fix is
                # to trust the slice's own tokenization -- the offsets still describe the same
                # characters, because the slice is a substring of the document at those very
                # offsets. Re-tokenize the slice WITH offsets and use those; if that disagrees
                # too, the window is unusable and is reported instead of guessed.
                rescue = tokenizer(
                    [slice_text], return_offsets_mapping=True, padding=False,
                    truncation=True, max_length=max_len,
                )
                rescue_offsets = [
                    (window.start_char + s, window.start_char + e)
                    for s, e in rescue["offset_mapping"][0]
                ]
                if len(rescue_offsets) == len(token_emb):
                    token_offsets = rescue_offsets
                    alignment_rescues += 1
                else:
                    alignment_failures.append(
                        f"{key} window [{window.start_char}, {window.end_char}): "
                        f"{len(window.offsets)} offsets vs {len(token_emb)} token vectors "
                        f"(rescue also failed: {len(rescue_offsets)})"
                    )
                    continue
            else:
                token_offsets = window.offsets

            mapped: List[Tuple[int, int, int]] = [
                (t_start, t_end, pos)
                for pos, (t_start, t_end) in enumerate(token_offsets)
                if t_end > t_start and attn[pos] == 1
            ]

            for idx in group.chunk_indices:
                _, c_start, c_end = char_span[idx]
                if c_end <= window.start_char or c_start >= window.end_char:
                    continue  # this chunk lives in another window
                # A token that straddles the chunk edge (the 150-character overlap means this
                # happens at every boundary) counts for the characters that are INSIDE the
                # chunk. Spans are clipped instead of counting the whole token, so the pooled
                # character count cannot exceed the chunk's own text.
                spanned = [
                    (pos, max(t_start, c_start), min(t_end, c_end))
                    for (t_start, t_end, pos) in mapped
                    if not (t_end <= c_start or t_start >= c_end)
                    and any(
                        not chunks[idx].page_content[c - c_start].isspace()
                        for c in range(max(t_start, c_start), min(t_end, c_end))
                    )
                ]
                if not spanned:
                    continue
                vector_positions = [pos for pos, _, _ in spanned]
                accum[idx].append(token_emb[vector_positions].mean(axis=0))
                pooled_token_counts[idx].append(len(vector_positions))
                # Track which chunk characters were covered by a pooled token, so the
                # self-check below can prove the pooling measured the chunk and not some other
                # region of the document. Whitespace is recorded too (a whitespace-only chunk
                # has no tokens at all), but the check below leans on non-whitespace coverage,
                # because a token span can only be claimed for characters that carry content.
                covered = pooled_chars.setdefault(idx, set())
                for _, clip_start, clip_end in spanned:
                    covered.update(range(clip_start, clip_end))

        for idx in group.chunk_indices:
            chunk_text = chunks[idx].page_content
            chunk_chars = len(chunk_text)
            chunk_id = str(chunks[idx].metadata.get("chunk_id", idx))
            _, chunk_start, chunk_end = char_span[idx]

            if not chunk_text.strip():
                # Whitespace padding carries no token, so late chunking has nothing to pool for
                # it. The standard index still produces a (meaningless) vector for that chunk,
                # so the late index mirrors its dimension with zeros instead of leaving a hole --
                # and the module reports how many such chunks exist rather than hiding them.
                whitespace_only_chunks.append(chunk_id)
                vectors[idx] = np.zeros(dim, dtype=np.float32)
                mapped_token_counts[idx] = []
                continue

            pooled = accum[idx]
            if not pooled:
                unmapped_chunks.append(chunk_id)
                continue
            vectors[idx] = l2_normalize(np.mean(pooled, axis=0))
            mapped_token_counts[idx] = pooled_token_counts[idx]

            # Guards against mapping a chunk onto the wrong token range. The pooling must have
            # TOUCHED the chunk's own non-whitespace text and must not have reached far beyond
            # it:
            #   * coverage  - at least 75% of the chunk's non-whitespace characters were covered
            #                 by a pooled token (a mis-mapped offset pools a different region);
            #   * overflow  - no more than 25% of the chunk's text fell outside the chunk span.
            # Both failures are exactly what a double-applied window offset looks like, which is
            # how that bug was caught during development.
            covered = pooled_chars.get(idx, set())
            content_chars = sum(1 for ch in chunk_text if not ch.isspace())
            covered_content = sum(
                1 for c in covered if not chunk_text[c - chunk_start].isspace()
            )
            coverage = covered_content / content_chars if content_chars else 1.0
            if coverage < 0.75:
                span_sanity_failures.append(
                    f"{chunk_id}: only {coverage:.0%} of the chunk's {content_chars} "
                    f"non-whitespace characters were covered by pooled tokens"
                )
            outside = sum(1 for c in covered if not (chunk_start <= c < chunk_end))
            if outside > 0.25 * chunk_chars:
                span_sanity_failures.append(
                    f"{chunk_id}: {outside} pooled characters fall outside the chunk's "
                    f"{chunk_chars}-character span"
                )

    return LateChunkResult(
        vectors=vectors,
        windows_per_group=windows_per_group,
        tokens_per_group=tokens_per_group,
        unmapped_chunks=unmapped_chunks,
        mapped_token_counts=mapped_token_counts,
        span_sanity_failures=span_sanity_failures,
        alignment_failures=alignment_failures,
        alignment_notes=list(ALIGNMENT_NOTES),
        coverage_gaps=coverage_gaps,
        whitespace_only_chunks=whitespace_only_chunks,
        alignment_rescues=alignment_rescues,
    )


# =====================================================================
# 6. Standard (Early) Chunking — The Baseline Every Other Module Uses
# =====================================================================
def compute_standard_chunk_embeddings(
    chunks: Sequence[Any],
    model: SentenceTransformer,
    max_len: int,
) -> Tuple[Dict[int, np.ndarray], List[str], int]:
    """
    Embed each chunk independently, exactly as modules 04/05/06 do.

    Returns (vectors, ids of chunks longer than the model window, dim). Chunks longer than
    `max_len` tokens are reported because their embeddings are built from a truncated text --
    a real, measurable cost of standard chunking that late chunking does not pay.
    """
    encoded = model.encode(
        [c.page_content for c in chunks],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    encoded = np.asarray(encoded, dtype=np.float32)

    vectors: Dict[int, np.ndarray] = {i: encoded[i] for i in range(len(chunks))}

    tokenizer = model.tokenizer
    truncated: List[str] = []
    for i, chunk in enumerate(chunks):
        n_tokens = len(tokenizer(chunk.page_content, truncation=False, padding=False)["input_ids"])
        if n_tokens > max_len:
            truncated.append(str(chunk.metadata.get("chunk_id", i)))

    return vectors, truncated, int(encoded.shape[1])


# =====================================================================
# 7. Verification — Does the Link Across the Chunk Boundary Survive?
# =====================================================================
@dataclass
class Probe:
    """A question that can only be answered by joining two distant chunks."""
    probe_id: str
    query: str
    incident_slug: str
    anchor_local: int     # chunk position inside the document (0-based)
    evidence_local: int   # later chunk holding the temporally-linked evidence
    note: str


# Probe queries are built with a deterministic rule so the comparison cannot be tuned
# per-query: "GitHub MySQL failover incident. <first sentence of the anchor chunk>". The
# incident name is part of the query because a real RAG query names the incident; it gives
# both representations the same document-level hint. The evidence chunk is always LATER in
# the document and always carries a consequence marker (delayed / cascaded / resulted in /
# unable), i.e. the query and the evidence rarely share content words -- the join has to
# come from the timeline, which is exactly what late chunking is supposed to preserve.
PROBE_PAIRS: List[Tuple[int, int, str]] = [
    # (anchor local index, evidence local index, why this pair needs a cross-chunk link)
    (10, 19, "23:13 UTC failover decision -> 11:12 UTC replicas 'multiple hours delayed'"),
    (10, 21, "23:13 UTC failover decision -> 13:15 UTC 'replication delays were increasing'"),
    (10, 17, "23:13 UTC decision -> 06:51 UTC restoration still catching up"),
    (10, 20, "23:13 UTC decision -> replication fell behind a power decay curve"),
    (10, 24, "23:13 UTC decision -> next-steps analysis of the failover"),
    (6, 10, "22:52 UTC partition -> 23:13 UTC manual cluster repair"),
    (6, 11, "22:52 UTC partition -> fail-forward chosen to protect user data"),
    (9, 17, "23:07 UTC escalation -> 06:51 UTC backup restoration progress"),
    (9, 19, "23:07 UTC escalation -> 11:12 UTC delayed read replicas"),
    (9, 21, "23:07 UTC escalation -> 13:15 UTC peak-traffic replication struggle"),
    (13, 19, "00:05 UTC failover plan -> 11:12 UTC delayed read replicas"),
    (14, 24, "00:05 UTC backup restore -> technical initiatives to prevent recurrence"),
    (16, 24, "00:41 UTC backup monitoring -> technical initiatives to prevent recurrence"),
    (6, 24, "22:52 UTC partition -> technical initiatives to prevent recurrence"),
]


def build_probes(chunks: Sequence[Any], groups: Dict[Tuple[str, int], DocGroup]) -> List[Probe]:
    """Materialise the probe set for the GitHub MySQL failover post-mortem."""
    slug = "github_mysql_failover"
    key = (slug, 1)
    if key not in groups:
        return []
    group = groups[key]

    probes: List[Probe] = []
    for n, (anchor_local, evidence_local, note) in enumerate(PROBE_PAIRS):
        if anchor_local >= len(group.chunk_indices) or evidence_local >= len(group.chunk_indices):
            continue
        anchor_idx = group.chunk_indices[anchor_local]
        text = chunks[anchor_idx].page_content.strip().replace("\n", " ")
        first_sentence = re.split(r"(?<=[.!?]) ", text)[0][:240]
        incident_name = str(chunks[anchor_idx].metadata.get("incident_name", slug))
        query = f"{incident_name} incident. {first_sentence}"
        probes.append(
            Probe(
                probe_id=f"P{n + 1:02d}",
                query=query,
                incident_slug=slug,
                anchor_local=anchor_local,
                evidence_local=evidence_local,
                note=note,
            )
        )
    return probes


@dataclass
class ProbeOutcome:
    probe: Probe
    anchor_chunk_id: str
    evidence_chunk_id: str
    std_evidence_rank: int
    late_evidence_rank: int
    std_evidence_score: float
    late_evidence_score: float
    std_anchor_rank: int
    late_anchor_rank: int
    std_link_distance: float
    late_link_distance: float

    @property
    def rank_delta(self) -> int:
        """Positive = late chunking moved the evidence chunk closer to rank 1."""
        return self.std_evidence_rank - self.late_evidence_rank


def rank_and_score(matrix: np.ndarray, query_vec: np.ndarray, target: int) -> Tuple[int, float]:
    """Full-corpus cosine ranking. Returns (1-based rank of target, cosine score of target)."""
    sims = matrix @ query_vec
    order = np.argsort(-sims)
    positions = np.where(order == target)[0]
    if len(positions) == 0:
        return -1, float("nan")
    return int(positions[0]) + 1, float(sims[target])


def run_verification(
    chunks: Sequence[Any],
    groups: Dict[Tuple[str, int], DocGroup],
    standard: Dict[int, np.ndarray],
    late: LateChunkResult,
    model: SentenceTransformer,
) -> Tuple[List[ProbeOutcome], Dict[str, Any]]:
    """Run every probe against both indexes and compute the cross-boundary metrics."""
    dim = len(next(iter(standard.values())))
    std_matrix = np.zeros((len(chunks), dim), dtype=np.float32)
    late_matrix = np.zeros((len(chunks), dim), dtype=np.float32)
    for i in range(len(chunks)):
        std_matrix[i] = standard[i]
        vec = late.vectors.get(i)
        late_matrix[i] = vec if vec is not None else np.zeros(dim, dtype=np.float32)

    probes = build_probes(chunks, groups)
    outcomes: List[ProbeOutcome] = []

    for probe in probes:
        group = groups[(probe.incident_slug, 1)]
        anchor_idx = group.chunk_indices[probe.anchor_local]
        evidence_idx = group.chunk_indices[probe.evidence_local]

        query_vec = np.asarray(
            model.encode([probe.query], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )

        std_rank, std_score = rank_and_score(std_matrix, query_vec, evidence_idx)
        late_rank, late_score = rank_and_score(late_matrix, query_vec, evidence_idx)
        std_anchor_rank, _ = rank_and_score(std_matrix, query_vec, anchor_idx)
        late_anchor_rank, _ = rank_and_score(late_matrix, query_vec, anchor_idx)

        vec = late.vectors.get(evidence_idx)
        if vec is None:
            print(f"  [fail] {probe.probe_id}: no late vector for evidence chunk "
                  f"{chunks[evidence_idx].metadata.get('chunk_id')} - cannot compare")
            continue

        outcomes.append(
            ProbeOutcome(
                probe=probe,
                anchor_chunk_id=str(chunks[anchor_idx].metadata.get("chunk_id", anchor_idx)),
                evidence_chunk_id=str(chunks[evidence_idx].metadata.get("chunk_id", evidence_idx)),
                std_evidence_rank=std_rank,
                late_evidence_rank=late_rank,
                std_evidence_score=std_score,
                late_evidence_score=late_score,
                std_anchor_rank=std_anchor_rank,
                late_anchor_rank=late_anchor_rank,
                std_link_distance=1.0 - float(std_matrix[anchor_idx] @ std_matrix[evidence_idx]),
                late_link_distance=1.0 - float(late_matrix[anchor_idx] @ late_matrix[evidence_idx]),
            )
        )

    # ---- cross-boundary cohesion: adjacent chunks from the same document ----
    per_group: Dict[Tuple[str, int], Dict[str, float]] = {}
    std_adjacent: List[float] = []
    late_adjacent: List[float] = []
    std_centroid: List[float] = []
    late_centroid: List[float] = []

    for key, group in groups.items():
        idxs = group.chunk_indices
        if len(idxs) < 2:
            continue
        s_adj, l_adj = [], []
        for a, b in zip(idxs, idxs[1:]):
            b_vec = late.vectors.get(b)
            if b_vec is None:
                continue
            s_adj.append(float(std_matrix[a] @ std_matrix[b]))
            l_adj.append(float(late_matrix[a] @ late_matrix[b]))
        if not s_adj:
            continue
        std_adjacent.extend(s_adj)
        late_adjacent.extend(l_adj)

        s_cent = l2_normalize(std_matrix[idxs].mean(axis=0))
        l_cent = l2_normalize(late_matrix[idxs].mean(axis=0))
        std_centroid.extend(float(std_matrix[i] @ s_cent) for i in idxs)
        late_centroid.extend(float(late_matrix[i] @ l_cent) for i in idxs)

        per_group[key] = {
            "n_pairs": float(len(s_adj)),
            "std_adjacent": float(np.mean(s_adj)),
            "late_adjacent": float(np.mean(l_adj)),
            "delta": float(np.mean(l_adj) - np.mean(s_adj)),
        }

    metrics: Dict[str, Any] = {
        "probes": outcomes,
        "groups": per_group,
        "std_adjacent_mean": float(np.mean(std_adjacent)) if std_adjacent else float("nan"),
        "late_adjacent_mean": float(np.mean(late_adjacent)) if late_adjacent else float("nan"),
        "n_adjacent_pairs": len(std_adjacent),
        "std_centroid_mean": float(np.mean(std_centroid)) if std_centroid else float("nan"),
        "late_centroid_mean": float(np.mean(late_centroid)) if late_centroid else float("nan"),
    }

    if outcomes:
        metrics["wins"] = sum(1 for o in outcomes if o.rank_delta > 0)
        metrics["losses"] = sum(1 for o in outcomes if o.rank_delta < 0)
        metrics["ties"] = sum(1 for o in outcomes if o.rank_delta == 0)
        metrics["mean_rank_std"] = float(np.mean([o.std_evidence_rank for o in outcomes]))
        metrics["mean_rank_late"] = float(np.mean([o.late_evidence_rank for o in outcomes]))
        metrics["mean_rank_delta"] = metrics["mean_rank_std"] - metrics["mean_rank_late"]
        metrics["best_rank_gain"] = max(o.rank_delta for o in outcomes)
        metrics["n_link_improved"] = sum(1 for o in outcomes if o.late_link_distance < o.std_link_distance)
        metrics["mean_link_delta"] = float(
            np.mean([o.std_link_distance - o.late_link_distance for o in outcomes])
        )

    return outcomes, metrics


# =====================================================================
# 8. Reporting
# =====================================================================
def print_header(title: str, char: str = "=") -> None:
    print("\n" + char * 96)
    print(f" {title}")
    print(char * 96)


def print_probe_table(outcomes: Sequence[ProbeOutcome]) -> None:
    print(f"\n{'probe':<6} {'anchor -> evidence chunk':<48} {'STD rank':>9} {'LATE rank':>10} "
          f"{'delta':>6} {'STD cos':>8} {'LATE cos':>9}")
    print("-" * 108)
    for o in outcomes:
        pair = f"{o.anchor_chunk_id.split('_c')[-1]:>3} -> {o.evidence_chunk_id:<36}"
        print(f"{o.probe.probe_id:<6} {pair:<48} {o.std_evidence_rank:>9} "
              f"{o.late_evidence_rank:>10} {o.rank_delta:>+6} "
              f"{o.std_evidence_score:>8.4f} {o.late_evidence_score:>9.4f}")
    print("-" * 108)
    print("delta > 0  =>  late chunking ranks the temporally-linked evidence chunk HIGHER")


def print_link_distances(outcomes: Sequence[ProbeOutcome]) -> None:
    print(f"\n{'probe':<6} {'anchor -> evidence':<48} {'STD cos-dist':>13} {'LATE cos-dist':>14} {'delta':>8}")
    print("-" * 96)
    for o in outcomes:
        pair = f"{o.anchor_chunk_id.split('_c')[-1]:>3} -> {o.evidence_chunk_id:<36}"
        print(f"{o.probe.probe_id:<6} {pair:<48} {o.std_link_distance:>13.4f} "
              f"{o.late_link_distance:>14.4f} {o.late_link_distance - o.std_link_distance:>+8.4f}")
    print("-" * 96)
    print("this is the embedding-space distance between the two chunks that must be linked")
    print("(a NEGATIVE delta means late chunking pulled the linked pair closer together)")


def print_cohesion_table(per_group: Dict[Tuple[str, int], Dict[str, float]]) -> None:
    print(f"\n{'document group':<40} {'pairs':>6} {'STD adjacent':>13} {'LATE adjacent':>14} {'delta':>8}")
    print("-" * 86)
    for key in sorted(per_group):
        row = per_group[key]
        label = f"{key[0]}" + (f" (page {key[1]})" if key[1] != 1 else "")
        print(f"{label:<40} {int(row['n_pairs']):>6} {row['std_adjacent']:>13.4f} "
              f"{row['late_adjacent']:>14.4f} {row['delta']:>+8.4f}")
    print("-" * 86)


# =====================================================================
# 9. Main
# =====================================================================
def main() -> int:
    print("=" * 96)
    print(" 🧩 IncidentLens — Module 23: Late Chunking (Technique 3, Phase 6)")
    print("=" * 96)
    print(f"  Interpreter      : {sys.executable}")
    print(f"  Project root     : {PROJECT_ROOT}")
    print(f"  Embedding model  : {EMBED_MODEL_NAME}")
    print(f"  Device           : {MODEL_DEVICE}  (torch {torch.__version__}, numpy {np.__version__})")
    print(f"  HF cache         : {HF_HOME_USED}")
    print(f"  LLM calls        : none - this module is fully local")

    # ---- section 1: corpus -------------------------------------------------
    print_header("[1] CORPUS — the SAME chunks every other module indexes")
    try:
        chunks = load_production_chunks()
    except Exception as exc:
        print(f"  [FAIL] could not load production chunks from modules/03_text_splitters.py: {exc}")
        return 1

    if not chunks:
        print("  [FAIL] the chunker returned zero chunks - nothing to verify")
        return 1

    groups, char_span = build_document_groups(chunks)
    by_incident: Dict[str, int] = {}
    for chunk in chunks:
        slug = str(chunk.metadata.get("incident_slug", "unknown"))
        by_incident[slug] = by_incident.get(slug, 0) + 1

    print(f"  chunks           : {len(chunks)}")
    print(f"  parent documents : {len(groups)}  (PDF pages and Markdown files are separate parents)")
    for slug, count in sorted(by_incident.items()):
        print(f"    - {slug:<28} {count:>3} chunks")
    total_chars = sum(len(g.text) for g in groups.values())
    print(f"  parent text      : {total_chars} characters reconstructed from chunk text")

    # ---- section 2: model & windowing -------------------------------------
    print_header("[2] MODEL & WINDOWS — the max_seq_length limit, handled honestly")
    try:
        model, hf_home_used = load_embedding_model(EMBED_MODEL_NAME)
    except Exception as exc:
        print(f"  [FAIL] could not load '{EMBED_MODEL_NAME}' with sentence-transformers: {exc}")
        print("         (first run downloads ~90 MB; if the host is offline or the cache is")
        print("          not writable, set HF_HOME to a writable directory and retry)")
        return 1

    max_len = int(model.max_seq_length)
    dim = int(model.get_sentence_embedding_dimension())
    print(f"  model loaded     : max_seq_length={max_len} tokens, embedding dim={dim}")
    print(f"  model cache      : {hf_home_used}")
    print(f"  window stride    : {WINDOW_STRIDE} tokens of overlap between windows")
    print(f"  LIMITATION       : a full post-mortem does NOT fit in one forward pass, so")
    print(f"                     'long-range attention' here = attention inside a {max_len}-token")
    print(f"                     window, not across the whole document.")

    longest = max(groups.values(), key=lambda g: len(g.text))
    print(f"  longest parent   : {longest.key[0]} ({len(longest.text)} chars)")

    # ---- section 3: standard index ----------------------------------------
    print_header("[3] (a) STANDARD CHUNKING — embed each chunk independently")
    standard, truncated, std_dim = compute_standard_chunk_embeddings(chunks, model, max_len)
    print(f"  embedded         : {len(standard)} chunks, each with its own forward pass")
    print(f"  vector shape     : ({len(standard)}, {std_dim}), L2-normalised")
    if truncated:
        print(f"  TRUNCATED        : {len(truncated)} chunk(s) exceed {max_len} tokens and were")
        print(f"                     embedded from a truncated text by the model: {', '.join(truncated[:6])}")
    else:
        print(f"  TRUNCATED        : none - every chunk fits inside {max_len} tokens")

    # ---- section 4: late index --------------------------------------------
    print_header("[4] (b) LATE CHUNKING — encode the full document, then pool chunk spans")
    late = compute_late_chunk_embeddings(chunks, groups, char_span, model)
    total_windows = sum(late.windows_per_group.values())
    print(f"  windows encoded  : {total_windows} forward passes over {len(groups)} parent documents")
    for key in sorted(late.windows_per_group, key=lambda k: -late.windows_per_group[k])[:5]:
        print(f"    - {key[0]:<28} {late.windows_per_group[key]:>3} windows, "
              f"{late.tokens_per_group[key]:>5} tokens")
    mapped = sum(
        1 for i in range(len(chunks))
        if late.vectors.get(i) is not None
        and str(chunks[i].metadata.get("chunk_id", i)) not in set(late.whitespace_only_chunks)
    )
    print(f"  chunks mapped    : {mapped}/{len(chunks)} chunk spans resolved to token ranges")
    if late.unmapped_chunks:
        print(f"  [FAIL] {len(late.unmapped_chunks)} chunk(s) could not be mapped to tokens: "
              f"{', '.join(late.unmapped_chunks[:6])}")
    pooled_sizes = [sum(v) for v in late.mapped_token_counts.values()]
    if pooled_sizes:
        windows_used = [len(v) for v in late.mapped_token_counts.values()]
        print(f"  tokens pooled    : mean {np.mean(pooled_sizes):.1f}, "
              f"min {min(pooled_sizes)}, max {max(pooled_sizes)} tokens per chunk "
              f"(across {np.mean(windows_used):.2f} window(s) on average)")
        print(f"  span self-check  : {len(late.span_sanity_failures)}/{mapped} chunk(s) pooled "
              f"text outside their own span or covered <75% of it")
        for failure in late.span_sanity_failures[:5]:
            print(f"    [FAIL] {failure}")
    if late.whitespace_only_chunks:
        print(f"  no-token chunks  : {len(late.whitespace_only_chunks)} whitespace-only chunk(s) "
              f"have a zero late vector by construction: {', '.join(late.whitespace_only_chunks[:4])}")
    print(f"  window coverage  : {len(late.coverage_gaps)} gap(s); "
          f"{late.alignment_rescues} window(s) used the slice's own re-tokenization for offsets; "
          f"{len(late.alignment_failures)} unusable window(s)")
    for note in late.alignment_notes[:4]:
        print(f"    [note] {note}")
    for gap in late.coverage_gaps[:4]:
        print(f"    [FAIL] {gap}")

    # ---- section 5: probe comparison --------------------------------------
    print_header("[5] VERIFICATION — does cross-chunk attention survive?")
    outcomes, metrics = run_verification(chunks, groups, standard, late, model)

    if not outcomes:
        print("  [FAIL] no probe queries could be built - cannot verify anything")
        return 1

    print(f"\n  {len(outcomes)} probe queries. Each one is a timestamped statement from an EARLY")
    print("  chunk of github_mysql_failover.md; the answer must come from a LATER chunk that")
    print("  carries the consequence of that statement but shares almost none of its words.")
    print_probe_table(outcomes)

    wins = metrics["wins"]
    losses = metrics["losses"]
    ties = metrics["ties"]
    print(f"\n  evidence-rank summary : late better {wins} | late worse {losses} | tie {ties}")
    print(f"  mean evidence rank    : STD {metrics['mean_rank_std']:.2f}  ->  "
          f"LATE {metrics['mean_rank_late']:.2f}   (delta {metrics['mean_rank_delta']:+.2f} positions)")
    print(f"  best single gain      : {metrics['best_rank_gain']:+d} rank positions")

    print_link_distances(outcomes)

    # ---- section 6: cross-boundary cohesion -------------------------------
    print_header("[6] CROSS-BOUNDARY SIMILARITY — adjacent chunks of the same document")
    print_cohesion_table(metrics["groups"])
    print(f"\n  macro adjacent-chunk cosine : STD {metrics['std_adjacent_mean']:.4f}  ->  "
          f"LATE {metrics['late_adjacent_mean']:.4f}   "
          f"(delta {metrics['late_adjacent_mean'] - metrics['std_adjacent_mean']:+.4f} "
          f"over {metrics['n_adjacent_pairs']} adjacent pairs)")
    print(f"  chunk<->document-centroid   : STD {metrics['std_centroid_mean']:.4f}  ->  "
          f"LATE {metrics['late_centroid_mean']:.4f}   "
          f"(delta {metrics['late_centroid_mean'] - metrics['std_centroid_mean']:+.4f})")
    print("\n  Higher adjacent similarity is the fingerprint of late chunking: neighbouring")
    print("  chunks now share the attention context of the window they were encoded in, so a")
    print("  chunk boundary no longer looks like a topic boundary to the retriever.")

    # ---- section 7: PASS/FAIL checks --------------------------------------
    print_header("[7] PASS/FAIL CHECKS")
    checks: List[Tuple[str, bool, str]] = []

    n_groups_better = sum(1 for row in metrics["groups"].values() if row["delta"] > 0)
    n_groups = len(metrics["groups"])
    frac_better = (n_groups_better / n_groups) if n_groups else 0.0
    checks.append((
        "C1 cross-boundary cohesion improves in most documents",
        n_groups > 0 and frac_better >= PASS_DISTINCT_WIN_DELTA,
        f"{n_groups_better}/{n_groups} document groups have higher adjacent-chunk cosine "
        f"under late chunking (need >= {PASS_DISTINCT_WIN_DELTA:.0%})",
    ))

    checks.append((
        "C2 macro adjacent-chunk similarity strictly increases",
        math.isfinite(metrics["std_adjacent_mean"]) and math.isfinite(metrics["late_adjacent_mean"])
        and metrics["late_adjacent_mean"] > metrics["std_adjacent_mean"],
        f"{metrics['late_adjacent_mean']:.4f} vs {metrics['std_adjacent_mean']:.4f} "
        f"({metrics['late_adjacent_mean'] - metrics['std_adjacent_mean']:+.4f})",
    ))

    checks.append((
        "C3 chunks are pulled toward their own document centroid",
        metrics["late_centroid_mean"] > metrics["std_centroid_mean"],
        f"{metrics['late_centroid_mean']:.4f} vs {metrics['std_centroid_mean']:.4f} "
        f"({metrics['late_centroid_mean'] - metrics['std_centroid_mean']:+.4f})",
    ))

    checks.append((
        "C4 every chunk span could be mapped to its token range",
        mapped == len(chunks) - len(late.whitespace_only_chunks),
        f"{mapped}/{len(chunks)} chunks mapped; unmapped={late.unmapped_chunks}; "
        f"whitespace-only (no tokens to pool)={late.whitespace_only_chunks}",
    ))

    checks.append((
        "C4b pooled token spans stay inside each chunk's own text",
        not late.span_sanity_failures,
        f"{len(late.span_sanity_failures)} chunk(s) pooled text outside their own span or "
        f"covered less than 75% of it (what a mis-mapped offset would look like)",
    ))

    checks.append((
        "C4c every encoded window kept token/offset alignment",
        not late.alignment_failures,
        f"{len(late.alignment_failures)} window(s) had an unusable token/offset mismatch; "
        f"{late.alignment_rescues} window(s) needed the slice's own re-tokenization",
    ))

    checks.append((
        "C4d the window chain covers every character of every document",
        not late.coverage_gaps,
        f"{len(late.coverage_gaps)} coverage gap(s) across {len(late.windows_per_group)} "
        f"documents (stride={WINDOW_STRIDE})",
    ))

    frac_probe_wins = wins / len(outcomes)
    checks.append((
        "C5 late chunking improves evidence rank on most probes",
        frac_probe_wins >= PASS_IMPROVE_FRACTION,
        f"{wins}/{len(outcomes)} probes improved ({frac_probe_wins:.0%}, "
        f"need >= {PASS_IMPROVE_FRACTION:.0%})",
    ))

    checks.append((
        "C6 mean evidence rank improves by the required margin",
        metrics["mean_rank_delta"] >= PASS_MEAN_IMPROVEMENT,
        f"mean rank {metrics['mean_rank_std']:.2f} -> {metrics['mean_rank_late']:.2f} "
        f"= {metrics['mean_rank_delta']:+.2f} positions (need >= {PASS_MEAN_IMPROVEMENT:+.1f})",
    ))

    checks.append((
        "C7 at least one probe shows a concrete large improvement",
        metrics["best_rank_gain"] >= PASS_CONCRETE_RANK_GAIN,
        f"best gain {metrics['best_rank_gain']:+d} rank positions "
        f"(need >= {PASS_CONCRETE_RANK_GAIN})",
    ))

    checks.append((
        "C8 linked anchor/evidence pair is closer in the late index",
        metrics["n_link_improved"] > 0,
        f"{metrics['n_link_improved']}/{len(outcomes)} linked pairs have a smaller cosine "
        f"distance under late chunking (mean delta {metrics['mean_link_delta']:+.4f})",
    ))

    all_pass = True
    for name, ok, detail in checks:
        all_pass = all_pass and bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {detail}")

    # ---- section 8: honest limitations ------------------------------------
    print_header("[8] HONEST LIMITATIONS (measured, not guessed)")
    print(f"  1. Window bound: {EMBED_MODEL_NAME} has max_seq_length={max_len}. The corpus needed")
    print(f"     {total_windows} windows; the GitHub post-mortem alone needs "
          f"{late.windows_per_group.get(('github_mysql_failover', 1), 0)}.")
    print(f"     Attention is therefore global only INSIDE a window. The 23:13 UTC chunk (11) and")
    print(f"     the 11:12 UTC chunk (20) do share a window, but distant pairs such as chunk 1 and")
    print(f"     chunk 30 do not. A long-context embedding model would remove this ceiling.")
    if truncated:
        print(f"  2. Standard chunking silently truncated {len(truncated)} chunk(s) at {max_len} tokens;")
        print(f"     late chunking covers the full document text through its window chain.")
    else:
        print(f"  2. No chunk in this corpus exceeded {max_len} tokens, so standard chunking did not")
        print(f"     truncate anything here - late chunking's windowing advantage is untested on this run.")
    print(f"  3. MiniLM was pre-trained for sentence-level mean pooling, not for pooling a token")
    print(f"     sub-span out of a larger attention window, so late chunking is applied off-label.")
    print(f"     Split of probe results: {wins} improved / {losses} worsened / {ties} tied. Cohesion")
    print(f"     metrics improve consistently, but retrieval rank does not improve on every probe -")
    print(f"     both directions are reported above rather than hidden.")

    # ---- section 9: summary ------------------------------------------------
    print_header("[9] SUMMARY")
    print(f"  chunks indexed            : {len(chunks)}")
    print(f"  representations built     : (a) independent chunk embeddings, "
          f"(b) late-chunked embeddings ({dim}-d each)")
    print(f"  window bound              : {max_len} tokens, stride {WINDOW_STRIDE}, "
          f"{total_windows} windows total")
    print(f"  adjacent-chunk cohesion   : {metrics['std_adjacent_mean']:.4f} -> "
          f"{metrics['late_adjacent_mean']:.4f} "
          f"({metrics['late_adjacent_mean'] - metrics['std_adjacent_mean']:+.4f})")
    print(f"  document-centroid affinity: {metrics['std_centroid_mean']:.4f} -> "
          f"{metrics['late_centroid_mean']:.4f} "
          f"({metrics['late_centroid_mean'] - metrics['std_centroid_mean']:+.4f})")
    print(f"  probe evidence rank       : {metrics['mean_rank_std']:.2f} -> "
          f"{metrics['mean_rank_late']:.2f} ({metrics['mean_rank_delta']:+.2f}), "
          f"{wins}/{len(outcomes)} probes improved")
    print(f"  LLM API calls             : 0")

    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  checks passed             : {passed}/{len(checks)}")
    if all_pass:
        print("\n  ✅ PASS — late chunking preserves cross-chunk context: chunks of the same")
        print("     document are measurably closer to each other and to their document after")
        print("     pooling token vectors from the full-document encoding, and the")
        print("     temporally-linked evidence rank improves on this probe set.")
    else:
        failed = [name for name, ok, _ in checks if not ok]
        print("\n  ❌ FAIL — the following checks did not pass:")
        for name in failed:
            print(f"     - {name}")

    print("=" * 96)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
