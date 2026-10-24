"""
Provider layer — LLM calls with fallback, retry, and disk cache.

This module is Phase 3 of the pipeline plan. It wraps the two model providers
(Groq and OpenRouter) so the rest of the pipeline never calls a provider
directly.

What it does
------------
1. It holds a fallback chain: Groq first, then OpenRouter.
2. It retries on HTTP 429 (rate limit), honouring the suggested wait.
3. It caches every true provider answer on disk. A second run of the same
   request makes no provider call at all.
4. It never fakes an answer. When every provider fails, it raises an error.

Why this module exists
----------------------
Stages 8 (contextualize) and 9 (generate) call an LLM. Those calls go through
this layer, so a provider outage or a rate limit does not crash the pipeline
and does not burn quota twice for the same question.

The cache rule
--------------
The cache key is a hash of the full request: model, messages, temperature,
max_tokens. The cached value is the provider's answer text plus the token
count. A cache hit returns the answer with zero provider calls.

Usage
-----
    from core.providers import complete

    answer = complete(
        messages=[{"role": "user", "content": "What caused the outage?"}],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
    )
    print(answer.text)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

try:                                    # allow "python core/providers.py"
    from core._bootstrap import PROJECT_ROOT
except ModuleNotFoundError:             # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core._bootstrap import PROJECT_ROOT

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ENV_PATH = PROJECT_ROOT / ".env"
CACHE_DIR = PROJECT_ROOT / ".cache" / "providers"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 5
MIN_GAP_SECONDS = 2.0

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

GROQ_MODELS = (
    "openai/gpt-oss-20b",
    "deepseek-r1-distill-llama-70b",
)
OPENROUTER_MODELS = (
    "qwen/qwen3.7-plus",
    "allam-2-7b",
    "openai/gpt-oss-20b",
)


class ProviderError(RuntimeError):
    """Every provider in the chain failed."""


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
@dataclass
class Completion:
    """One provider answer."""

    text: str
    model: str
    provider: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    cached: bool = False


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def _load_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"')
    return env


_ENV_CACHE: Optional[Dict[str, str]] = None


def _env() -> Dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _load_env()
        for key, value in _ENV_CACHE.items():
            os.environ.setdefault(key, value)
    return _ENV_CACHE


def _api_key(provider: str) -> str:
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY") or _env().get("GROQ_API_KEY", "")
    elif provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY") or _env().get("OPENROUTER_API_KEY", "")
    else:
        key = ""
    if not key:
        raise ProviderError(f"{provider} API key is not set. Put it in .env.")
    return key


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------
def _cache_key(
    model: str,
    messages: Sequence[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _cache_get(key: str) -> Optional[Completion]:
    path = _cache_path(key)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Completion(
            text=data["text"],
            model=data["model"],
            provider=data["provider"],
            tokens_used=data.get("tokens_used", 0),
            latency_ms=data.get("latency_ms", 0.0),
            cached=True,
        )
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _cache_put(key: str, completion: Completion) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    data = {
        "text": completion.text,
        "model": completion.model,
        "provider": completion.provider,
        "tokens_used": completion.tokens_used,
        "latency_ms": completion.latency_ms,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
class _BaseProvider:
    """A single provider: one POST to one endpoint."""

    def __init__(self, name: str, url: str, key: str) -> None:
        self.name = name
        self.url = url
        self.key = key
        self.last_latency_ms: float = 0.0

    def complete(
        self,
        messages: Sequence[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            started = time.perf_counter()
            try:
                response = requests.post(
                    self.url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_exc = exc
                wait = 2.0 * (attempt + 1)
                print(f"   {self.name} request failed, retrying in {wait:.0f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            self.last_latency_ms = (time.perf_counter() - started) * 1000.0

            if response.status_code == 429:
                wait = 4.0
                try:
                    data_err = response.json()
                    msg = data_err.get("error", {}).get("message", "")
                    import re as _re
                    match = _re.search(r"try again in ([0-9.]+)s", msg)
                    if match:
                        wait = float(match.group(1)) + 0.5
                except Exception:
                    pass
                print(f"   {self.name} rate limited, waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                last_exc = ProviderError(f"{self.name} rate limited")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                last_exc = ProviderError(
                    f"{self.name} returned {response.status_code}: "
                    f"{response.text[:300]}"
                )
                time.sleep(1.0 * (attempt + 1))
                continue

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                last_exc = ProviderError(f"{self.name} returned no choices")
                time.sleep(1.0 * (attempt + 1))
                continue
            text = (choices[0].get("message", {}) or {}).get("content", "") or ""
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            return Completion(
                text=text,
                model=model,
                provider=self.name,
                tokens_used=tokens_used,
                latency_ms=self.last_latency_ms,
                cached=False,
            )

        raise last_exc or ProviderError(f"{self.name} failed after {MAX_RETRIES} retries")


def _groq_provider() -> _BaseProvider:
    return _BaseProvider("groq", GROQ_URL, _api_key("groq"))


def _openrouter_provider() -> _BaseProvider:
    return _BaseProvider("openrouter", OPENROUTER_URL, _api_key("openrouter"))


def _is_valid_model(model: str) -> bool:
    return model in GROQ_MODELS or model in OPENROUTER_MODELS


# --------------------------------------------------------------------------
# The fallback chain
# --------------------------------------------------------------------------
class FallbackChain:
    """Try providers in order until one returns an answer."""

    def __init__(self, providers: Sequence[_BaseProvider]) -> None:
        self.providers = list(providers)

    def complete(
        self,
        messages: Sequence[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        last_exc: Optional[Exception] = None
        for provider in self.providers:
            try:
                return provider.complete(
                    messages, model, temperature, max_tokens,
                )
            except ProviderError as exc:
                last_exc = exc
                print(f"   {provider.name} failed: {exc}")
                continue
        raise last_exc or ProviderError("all providers failed")


def _default_chain() -> FallbackChain:
    return FallbackChain([_openrouter_provider(), _groq_provider()])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def complete(
    messages: Sequence[dict],
    model: str = "openai/gpt-oss-20b",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    use_cache: bool = True,
) -> Completion:
    """
    Get one chat completion, with fallback, retry, and disk cache.

    Parameters
    ----------
    messages
        The chat messages, in OpenAI/Groq format.
    model
        The model name. Groq models go to Groq first; OpenRouter models go
        to OpenRouter first. The fallback chain tries the other provider too.
    temperature
        Sampling temperature.
    max_tokens
        Maximum tokens in the answer.
    use_cache
        When True, a disk-cached answer is returned with no provider call.

    Returns
    -------
    A Completion with the answer text, model, provider, and timing.
    """
    key = _cache_key(model, list(messages), temperature, max_tokens)

    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    chain = _default_chain()
    completion = chain.complete(
        messages, model, temperature, max_tokens,
    )

    if use_cache:
        _cache_put(key, completion)

    return completion


def complete_with_fallback_model(
    messages: Sequence[dict],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    use_cache: bool = True,
) -> Completion:
    """
    Like complete(), but if the preferred provider family is unavailable,
    try a model from the other family as a fallback.

    For example, when model is a Groq model and Groq is down, fall back to
    an OpenRouter model.
    """
    try:
        return complete(
            messages, model, temperature, max_tokens, use_cache=use_cache,
        )
    except ProviderError:
        fallback_models = OPENROUTER_MODELS if _is_valid_groq(model) else GROQ_MODELS
        last_exc: Optional[Exception] = None
        for fb_model in fallback_models:
            try:
                return complete(
                    messages, fb_model, temperature, max_tokens,
                    use_cache=use_cache,
                )
            except ProviderError as exc:
                last_exc = exc
                continue
        raise last_exc or ProviderError("no provider or fallback model available")


def _is_valid_groq(model: str) -> bool:
    return model in GROQ_MODELS


# --------------------------------------------------------------------------
# Cache management
# --------------------------------------------------------------------------
def cache_clear() -> int:
    """Remove every cached completion. Return the number removed."""
    if not CACHE_DIR.is_dir():
        return 0
    count = 0
    for path in CACHE_DIR.iterdir():
        if path.is_file() and path.suffix == ".json":
            path.unlink()
            count += 1
    return count


def cache_stats() -> Dict[str, object]:
    """Return facts about the cache directory."""
    if not CACHE_DIR.is_dir():
        return {"entries": 0, "bytes": 0, "dir": str(CACHE_DIR)}
    files = [p for p in CACHE_DIR.iterdir() if p.is_file() and p.suffix == ".json"]
    total_bytes = sum(p.stat().st_size for p in files)
    return {
        "entries": len(files),
        "bytes": total_bytes,
        "dir": str(CACHE_DIR),
    }


# --------------------------------------------------------------------------
# Manual run
# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print(" PROVIDER LAYER — FALLBACK, RETRY, DISK CACHE")
    print("=" * 78)

    stats = cache_stats()
    print(f" cache entries : {stats['entries']}")
    print(f" cache bytes   : {stats['bytes']:,}")

    messages = [
        {"role": "system", "content": "You answer in one short sentence."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    print("\n FIRST CALL (provider)")
    first = complete(messages, model="openai/gpt-oss-20b")
    print(f"   model   : {first.model}")
    print(f"   provider: {first.provider}")
    print(f"   cached  : {first.cached}")
    print(f"   latency : {first.latency_ms:.0f} ms")
    print(f"   tokens  : {first.tokens_used}")
    print(f"   answer  : {first.text[:120]}")

    print("\n SECOND CALL (cache hit)")
    second = complete(messages, model="openai/gpt-oss-20b")
    print(f"   cached  : {second.cached}")
    print(f"   answer  : {second.text[:120]}")
    assert second.cached, "second call should hit the cache"
    assert second.text == first.text, "cached answer must match"

    print("\n ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
