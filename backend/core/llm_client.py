"""Multi-provider LLM client with key arrays, health tracking & auto-fallback.

Supports any OpenAI-compatible endpoint. Configured via numbered env vars:

  LLM_PROVIDER_ROUTING=failover
  LLM_PROVIDER_1_NAME=gemini
  LLM_PROVIDER_1_URL=https://generativelanguage.googleapis.com/v1beta/openai/
  LLM_PROVIDER_1_MODEL=gemini-2.5-flash
  LLM_PROVIDER_1_KEYS=key1,key2,key3
  LLM_PROVIDER_1_KEY_ROUTING=round_robin

Legacy OLLAMA_* vars work as fallback when no provider block is set.
"""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


# ── Per-key health state ─────────────────────────────────────────────────────

@dataclass
class KeyState:
    api_key: str
    healthy: bool = True
    failures: int = 0
    cooldown_until: float = 0.0


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    keys: list[KeyState] = field(default_factory=list)
    key_routing: str = "failover"
    _rr_index: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


# ── Module-level state (initialised once, thread-safe) ──────────────────────

_providers: list[Provider] = []
_routing: str = "failover"
_lock = threading.Lock()
_ready = False


# ── Config parsing ───────────────────────────────────────────────────────────

def _parse() -> None:
    global _routing, _providers
    _routing = os.getenv("LLM_PROVIDER_ROUTING", "failover").lower()
    _providers = []

    i = 1
    while True:
        name = os.getenv(f"LLM_PROVIDER_{i}_NAME")
        if not name:
            break
        url = os.getenv(f"LLM_PROVIDER_{i}_URL", "").strip()
        model = os.getenv(f"LLM_PROVIDER_{i}_MODEL", "").strip()
        keys_raw = os.getenv(f"LLM_PROVIDER_{i}_KEYS", "").strip()
        key_routing = os.getenv(f"LLM_PROVIDER_{i}_KEY_ROUTING", "failover").lower()

        keys = [KeyState(api_key=k.strip()) for k in keys_raw.split(",") if k.strip()]
        if url and model and keys:
            _providers.append(Provider(name, url, model, keys, key_routing))
        i += 1

    # Backward compat: single OLLAMA_* endpoint
    if not _providers:
        url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()
        key = os.getenv("OLLAMA_API_KEY", "ollama").strip()
        _providers.append(Provider("ollama", url, model, [KeyState(api_key=key)], "failover"))


def _ensure() -> None:
    global _ready
    if not _ready:
        with _lock:
            if not _ready:
                _parse()
                _ready = True


# ── Key selection per provider ───────────────────────────────────────────────

def _expire_cooldowns(p: Provider, now: float) -> None:
    for k in p.keys:
        if not k.healthy and k.cooldown_until <= now:
            k.healthy = True
            k.failures = 0


def _pick_key(p: Provider) -> KeyState | None:
    with p.lock:
        now = time.time()
        _expire_cooldowns(p, now)
        healthy = [k for k in p.keys if k.healthy]

        if not healthy:
            return None

        strategy = p.key_routing

        if strategy == "random":
            return random.choice(healthy)

        if strategy == "round_robin":
            idx = p._rr_index % len(p.keys)
            key = p.keys[idx]
            p._rr_index = (p._rr_index + 1) % len(p.keys)
            # Advance past unhealthy keys (bounded)
            for _ in range(len(p.keys)):
                if key.healthy:
                    return key
                idx = p._rr_index % len(p.keys)
                key = p.keys[idx]
                p._rr_index = (p._rr_index + 1) % len(p.keys)
            return None

        # failover — first healthy
        return healthy[0]


def _mark_failure(k: KeyState) -> None:
    k.failures += 1
    k.healthy = False
    k.cooldown_until = time.time() + 30.0


def _mark_success(k: KeyState) -> None:
    k.failures = 0
    k.healthy = True


# ── Public API ───────────────────────────────────────────────────────────────

def generate(prompt: str, *, as_json: bool = False, temperature: float = 0.1) -> str:
    _ensure()

    if not _providers:
        raise RuntimeError("No LLM providers configured")

    summary: list[str] = []

    for p in _providers:
        detail: list[str] = []

        for _ in range(len(p.keys)):
            key = _pick_key(p)
            if key is None:
                detail.append("no healthy keys left")
                break

            try:
                client = OpenAI(base_url=p.base_url, api_key=key.api_key)
                kwargs: dict[str, Any] = {
                    "model": p.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
                if as_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                _mark_success(key)
                return resp.choices[0].message.content or ""
            except Exception as exc:
                _mark_failure(key)
                err = str(exc)
                # Truncate long errors, keep the last 120 chars
                short = err[-120:] if len(err) > 120 else err
                detail.append(f"...{key.api_key[-6:]}: {short}")

        summary.append(f"  [{p.name}] {'; '.join(detail)}")

    raise RuntimeError(
        "All LLM providers failed:\n" + "\n".join(summary)
    )


def get_model() -> str:
    """Return the primary provider's model name (for display)."""
    _ensure()
    return _providers[0].model if _providers else "unknown"
