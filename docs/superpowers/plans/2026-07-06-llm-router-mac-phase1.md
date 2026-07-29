# llm-router Mac Phase-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a self-hosted LiteLLM gateway on macOS that routes each prompt per-cost/latency across NVIDIA NIM (free workhorse), opencode Zen (per-token, conserve) and GitHub Copilot (per-request, frontier), with a live NIM latency health-gate, and drives it from opencode + caveman-ultra + superpowers + lean-ctx + agentmemory.

**Architecture:** Only LiteLLM + Postgres run in Docker (OrbStack). `priority_router.py` (a LiteLLM pre-call hook) classifies every prompt into a tier and picks the first responsive/cheapest backend, skipping NIM models a session-start audit (`nim_health.sh` → `model_health.yaml`) has flagged slow. agentmemory (already a running container) and lean-ctx (MCP) stay host-level. opencode points only at the router so routing + compression + audit apply to all traffic.

**Tech Stack:** LiteLLM v1.90.3 (Docker), Postgres 16, Python 3 (router hook + pytest), POSIX sh scripts, OrbStack docker CLI (no sudo), opencode CLI, YAML config.

**Reference source (original WSL repo, private):** `slvbr0/llm-router`. Full design: [../specs/2026-07-06-llm-router-mac-design.md](../specs/2026-07-06-llm-router-mac-design.md).

**Project root (all paths relative to it):** `<repo-root>` — referred to below as `$PROJ`.

**Secrets (never commit):** `NVIDIA_API_KEY` supplied. `ZEN_API_KEY` = opencode-go key (Task 12 pulls from opencode auth or prompts). Copilot = device-flow, no key.

---

## File structure

| File | Responsibility |
|---|---|
| `docker-compose.yml` | LiteLLM proxy + Postgres services (OrbStack) |
| `config.yaml` | LiteLLM model_list: NIM + Zen + Copilot aliases, fallbacks, hook wiring |
| `priority_router.py` | Per-prompt tier classifier + cost-aware pick + health-gate |
| `availability.yaml` | Provider on/off mask (hot-reload): nim/zen/copilot |
| `model_health.yaml` | Per-NIM live health (written by `nim_health.sh`, read by router) |
| `.env` / `.env.example` | Secrets: master key, pg pw, NVIDIA_API_KEY, ZEN_API_KEY |
| `scripts/nim_health.sh` | Probe NIM latency → write `model_health.yaml` (NEW) |
| `scripts/discover_models.sh` | List live model IDs from NIM + Zen |
| `scripts/copilot_device_flow.sh` | One-time Copilot OAuth into container |
| `scripts/route_test.sh` | Fire tier prompts at the router |
| `scripts/show_routing.sh` | Read Postgres audit: which model served recent requests |
| `scripts/export_audit.sh` | Snapshot audit trail to CSV |
| `tests/test_priority_router.py` | pytest unit tests for router logic |
| `.gitignore` | ignore `.env`, `model_health.yaml`, `__pycache__` |

---

## Task 1: Scaffold project + git

**Files:**
- Create: `$PROJ/.gitignore`, `$PROJ/.env.example`

- [ ] **Step 1: Initialise git (folder is not a repo yet)**

```bash
cd "$PROJ"
git init -q
git branch -m main
```

- [ ] **Step 2: Write `.gitignore`**

```
.env
model_health.yaml
__pycache__/
*.pyc
.pytest_cache/
logs/*.csv
```

- [ ] **Step 3: Write `.env.example`**

```
# Copy to .env and fill. Never commit .env.
LITELLM_MASTER_KEY=sk-change-this
POSTGRES_PASSWORD=change-this-too
NVIDIA_API_KEY=
ZEN_API_KEY=
# GitHub Copilot needs no key here — OAuth device flow on first use.
```

- [ ] **Step 4: Commit**

```bash
cd "$PROJ" && git add .gitignore .env.example docs/
git commit -m "chore: scaffold llm-router mac project + spec/plan"
```

---

## Task 2: docker-compose.yml

**Files:**
- Create: `$PROJ/docker-compose.yml`

- [ ] **Step 1: Write `docker-compose.yml`** (OrbStack-compatible; identical topology to original — no sudo, arm64 images are multi-arch)

```yaml
services:
  litellm:
    image: ghcr.io/berriai/litellm:v1.90.3
    container_name: litellm-proxy
    restart: unless-stopped
    ports:
      - "127.0.0.1:4040:4000"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./priority_router.py:/app/priority_router.py:ro
      - ./availability.yaml:/app/availability.yaml
      - ./model_health.yaml:/app/model_health.yaml
      - litellm-tokens:/app/.litellm
    env_file:
      - .env
    environment:
      - DATABASE_URL=postgresql://litellm:${POSTGRES_PASSWORD}@db:5432/litellm
      - STORE_MODEL_IN_DB=True
      - AVAILABILITY_CONFIG=/app/availability.yaml
      - MODEL_HEALTH_CONFIG=/app/model_health.yaml
      - PYTHONPATH=/app
      - GITHUB_COPILOT_TOKEN_DIR=/app/.litellm/github_copilot
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:4000/health/liveliness')"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s

  db:
    image: postgres:16-alpine
    container_name: litellm-db
    restart: unless-stopped
    environment:
      - POSTGRES_USER=litellm
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=litellm
    volumes:
      - litellm-db:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U litellm"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  litellm-db:
  litellm-tokens:
```

- [ ] **Step 2: Create an empty `model_health.yaml`** (so the bind-mount has a file to mount; router fail-opens on empty)

```bash
cd "$PROJ" && printf 'models: {}\n' > model_health.yaml
```

- [ ] **Step 3: Commit**

```bash
cd "$PROJ" && git add docker-compose.yml && git commit -m "feat: litellm+postgres compose for orbstack"
```

---

## Task 3: config.yaml (NIM + Zen + Copilot)

**Files:**
- Create: `$PROJ/config.yaml`

- [ ] **Step 1: Write `config.yaml`** — NIM roster (from original, verified 2026-07-05), NEW Zen backend, Copilot, `auto` entry, cost-ordered fallbacks. Z.ai removed.

```yaml
# LiteLLM proxy config — llm-router mac v1
# Aliases are STABLE; underlying model IDs re-verified at build (Task 8).
# Priority: NIM (free, when responsive) -> Zen (per-token) -> Copilot (per-request).

model_list:
  # ===== NVIDIA NIM (free-tier workhorse) =====
  - model_name: nim-glm
    litellm_params: {model: openai/z-ai/glm-5.2, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-deepseek
    litellm_params: {model: openai/deepseek-ai/deepseek-v4-pro, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-deepseek-flash
    litellm_params: {model: openai/deepseek-ai/deepseek-v4-flash, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-kimi
    litellm_params: {model: openai/moonshotai/kimi-k2.6, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-qwen-max
    litellm_params: {model: openai/qwen/qwen3.5-397b-a17b, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-qwen
    litellm_params: {model: openai/qwen/qwen3.5-122b-a10b, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-minimax
    litellm_params: {model: openai/minimaxai/minimax-m3, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-nemotron
    litellm_params: {model: openai/nvidia/nemotron-3-ultra-550b-a55b, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-nemotron-super
    litellm_params: {model: openai/nvidia/nemotron-3-super-120b-a12b, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-mistral
    litellm_params: {model: openai/mistralai/mistral-large-3-675b-instruct-2512, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}
  - model_name: nim-llama
    litellm_params: {model: openai/meta/llama-3.3-70b-instruct, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}

  # ===== opencode Zen (per-token; conserve). IDs verified at build (Task 8). =====
  - model_name: zen-gpt
    litellm_params: {model: openai/gpt-5.5, api_base: https://opencode.ai/zen/v1, api_key: os.environ/ZEN_API_KEY}
  - model_name: zen-glm
    litellm_params: {model: openai/glm-5.2, api_base: https://opencode.ai/zen/v1, api_key: os.environ/ZEN_API_KEY}
  - model_name: zen-deepseek
    litellm_params: {model: openai/deepseek-v4-pro, api_base: https://opencode.ai/zen/v1, api_key: os.environ/ZEN_API_KEY}
  # Zen FREE-tier (zero-cost fallback capacity)
  - model_name: zen-free-nemotron
    litellm_params: {model: openai/nemotron-3-ultra-free, api_base: https://opencode.ai/zen/v1, api_key: os.environ/ZEN_API_KEY}
  - model_name: zen-free-deepseek
    litellm_params: {model: openai/deepseek-v4-flash-free, api_base: https://opencode.ai/zen/v1, api_key: os.environ/ZEN_API_KEY}
  - model_name: zen-free-pickle
    litellm_params: {model: openai/big-pickle, api_base: https://opencode.ai/zen/v1, api_key: os.environ/ZEN_API_KEY}

  # ===== GitHub Copilot (per-request credit; frontier). Device-flow token. =====
  - model_name: cop-opus
    litellm_params: {model: github_copilot/claude-opus-4.8}
  - model_name: cop-sonnet
    litellm_params: {model: github_copilot/claude-sonnet-5}
  - model_name: cop-gpt
    litellm_params: {model: github_copilot/gpt-5.5}
  - model_name: cop-codex
    litellm_params: {model: github_copilot/gpt-5.3-codex}
  - model_name: cop-gemini
    litellm_params: {model: github_copilot/gemini-3.1-pro-preview}
  - model_name: cop-haiku
    litellm_params: {model: github_copilot/claude-haiku-4.5}
  - model_name: cop-mini
    litellm_params: {model: github_copilot/gpt-5.4-mini}

  # Router entry point — priority_router rewrites per request.
  - model_name: auto
    litellm_params: {model: openai/z-ai/glm-5.2, api_base: https://integrate.api.nvidia.com/v1, api_key: os.environ/NVIDIA_API_KEY}

# Native LiteLLM fallbacks (deployment error/rate-limit): cost-ordered NIM -> Zen -> Copilot.
router_settings:
  routing_strategy: simple-shuffle
  enable_pre_call_checks: true
  num_retries: 2
  timeout: 120
  fallbacks:
    - nim-glm: [nim-mistral, zen-free-nemotron, cop-sonnet]
    - nim-deepseek: [nim-kimi, nim-qwen, zen-deepseek, cop-codex]
    - nim-kimi: [nim-deepseek, nim-qwen, cop-codex]
    - nim-qwen: [nim-deepseek, nim-kimi, cop-codex]
    - nim-qwen-max: [nim-nemotron, nim-deepseek, zen-gpt, cop-opus]
    - nim-nemotron: [nim-qwen-max, nim-nemotron-super, zen-gpt, cop-opus]
    - nim-minimax: [nim-glm, nim-kimi, cop-sonnet]
    - nim-llama: [nim-deepseek-flash, zen-free-deepseek, cop-mini]
    - nim-deepseek-flash: [nim-llama, zen-free-deepseek, cop-mini]
    - zen-gpt: [cop-opus, cop-gpt]
    - cop-opus: [cop-sonnet, zen-gpt, cop-gpt]
    - cop-sonnet: [cop-opus, cop-gpt]

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  store_model_in_db: true

litellm_settings:
  drop_params: true
  set_verbose: false
  callbacks: priority_router.router_instance
```

- [ ] **Step 2: Commit**

```bash
cd "$PROJ" && git add config.yaml && git commit -m "feat: litellm config — nim+zen+copilot roster and fallbacks"
```

---

## Task 4: priority_router.py — tests first (TDD)

**Files:**
- Create: `$PROJ/tests/test_priority_router.py`

- [ ] **Step 1: Write the failing tests** — cover tag parse, tier classify, cost-ordered pick, provider mask, and the NEW health-gate.

```python
import importlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
pr = importlib.import_module("priority_router")

ALL_OK = {"nim": True, "zen": True, "copilot": True}

def route(prompt, avail=None, health=None):
    cleaned, directives = pr.parse_request(prompt)
    availability = avail or dict(ALL_OK)
    return pr.route(cleaned, directives, availability, health or {})

def test_short_prompt_is_cheap_tier_nim_llama():
    assert route("Say hi") == "nim-llama"

def test_default_general_is_glm():
    long = "Explain the history and philosophy of stoicism " * 20
    assert route(long) == "nim-glm"

def test_code_marker_routes_to_deepseek():
    assert route("debug this:\n```python\nprint(1)\n```") == "nim-deepseek"

def test_think_tag_routes_reason_tier():
    assert route("[THINK] prove the halting problem is undecidable") == "nim-qwen-max"

def test_frontier_tag_prefers_copilot():
    assert route("[FRONTIER] design a novel consensus protocol") == "cop-opus"

def test_cheap_falls_to_zen_when_nim_down():
    # nim provider masked off -> cheap tier falls to zen free, NOT copilot
    assert route("Say hi", avail={"nim": False, "zen": True, "copilot": True}) == "zen-free-deepseek"

def test_reason_falls_to_copilot_when_nim_and_zen_down():
    r = route("[THINK] hard", avail={"nim": False, "zen": False, "copilot": True})
    assert r == "cop-opus"

def test_health_gate_skips_slow_nim_model():
    # nim-llama flagged slow -> cheap tier skips it to next NIM (deepseek-flash)
    health = {"nim-llama": {"ok": False}}
    assert route("Say hi", health=health) == "nim-deepseek-flash"

def test_health_gate_all_cheap_nim_slow_falls_to_zen():
    health = {"nim-llama": {"ok": False}, "nim-deepseek-flash": {"ok": False}}
    assert route("Say hi", health=health) == "zen-free-deepseek"

def test_unavailable_tag_masks_provider():
    cleaned, d = pr.parse_request("[UNAVAILABLE: nim] hello")
    eff = pr.effective_availability(d, base={"nim": True, "zen": True, "copilot": True})
    assert eff["nim"] is False and eff["zen"] is True

def test_novel_tag_passes_through_cleaned():
    # [NOVEL] is reserved for phase-2 fusion; v1 must strip it and route normally
    cleaned, d = pr.parse_request("[NOVEL] discover something")
    assert "[NOVEL]" not in cleaned
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd "$PROJ" && python3 -m pytest tests/test_priority_router.py -v`
Expected: FAIL / ERROR (`priority_router` not present, or signature mismatch).

---

## Task 5: priority_router.py — implementation

**Files:**
- Create: `$PROJ/priority_router.py`

- [ ] **Step 1: Write `priority_router.py`** — extends the original with a Zen provider, cost-ordered tiers, `model_health.yaml` read, and `[NOVEL]` passthrough. `route()` and `effective_availability()` signatures match the tests.

```python
"""Deterministic per-prompt priority router for the LiteLLM proxy (mac v1).

Layers, first match wins:
  1. Explicit tier tags: [CHEAP][CODE][THINK]/[REASON][AGENT][FRONTIER].
     [FUSION]/[NOVEL]/[DISCOVERY] are reserved for the phase-2 fusion sidecar:
     stripped from the prompt here and otherwise ignored (normal v1 routing).
  2. Availability mask (availability.yaml) + per-request [AVAILABLE:]/[UNAVAILABLE:].
  3. NIM health mask (model_health.yaml) — slow/dead NIM aliases skipped per request.
  4. Content heuristics (code markers, logic words, agent words, prompt size).
  5. Cost-ordered fallback chain: nim(responsive) -> zen(free) -> zen(paid) -> copilot.

Cost bases: NIM free (load-variable) -> Zen per-token -> Copilot per-request credit.
So cheap/small prompts fall to Zen; big/hard prompts fall to Copilot.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_logger

# Provider priority (cost order). Dict order IS the layer-5 fallback order.
PRIORITY_CHAIN: Dict[str, List[str]] = {
    "nim": [
        "nim-glm", "nim-deepseek", "nim-kimi", "nim-qwen-max", "nim-qwen",
        "nim-minimax", "nim-nemotron", "nim-nemotron-super", "nim-mistral",
        "nim-deepseek-flash", "nim-llama",
    ],
    "zen": [
        "zen-free-deepseek", "zen-free-nemotron", "zen-free-pickle",
        "zen-glm", "zen-deepseek", "zen-gpt",
    ],
    "copilot": [
        "cop-opus", "cop-sonnet", "cop-gpt", "cop-codex", "cop-gemini",
        "cop-haiku", "cop-mini",
    ],
}

# Tiers: NIM first (health-gated) -> Zen (free before paid) -> Copilot tail.
# Copilot never appears in cheap/general — no per-request credit on routine work.
CHEAP_TIER    = ["nim-llama", "nim-deepseek-flash", "zen-free-deepseek", "zen-free-pickle", "cop-mini"]
GENERAL_TIER  = ["nim-glm", "nim-mistral", "zen-free-nemotron", "zen-glm", "cop-sonnet"]
CODE_TIER     = ["nim-deepseek", "nim-kimi", "nim-qwen", "zen-deepseek", "cop-codex"]
REASON_TIER   = ["nim-qwen-max", "nim-nemotron", "zen-free-nemotron", "zen-gpt", "cop-opus"]
AGENT_TIER    = ["nim-glm", "nim-minimax", "nim-kimi", "zen-glm", "cop-sonnet"]
FRONTIER_TIER = ["cop-opus", "zen-gpt", "cop-sonnet", "cop-gpt"]

MODEL_PROVIDER = {m: p for p, models in PRIORITY_CHAIN.items() for m in models}

TIER_MAP = {
    "cheap": CHEAP_TIER, "general": GENERAL_TIER, "code": CODE_TIER,
    "reason": REASON_TIER, "agent": AGENT_TIER, "frontier": FRONTIER_TIER,
}

AUTO_MODELS = ("", "auto", "default")

CODE_MARKERS = re.compile(
    r"```|Traceback \(most recent call last\)|(?:^|\s)(?:def |class |import |function\s*\()"
    r"|\b(?:debug|refactor|implement|compile|stack trace|unit test|regex|SQL)\b",
    re.IGNORECASE,
)
REASON_MARKERS = re.compile(
    r"\b(prove|proof|derive|theorem|complexity|optimi[sz]e|algorithm|why does|"
    r"trade-?off|analyze|reason through|step by step|logic)\b", re.IGNORECASE,
)
AGENT_MARKERS = re.compile(
    r"\b(plan|orchestrat|multi-step|agent|workflow|pipeline|then |after that|"
    r"first .* then|call the|use the tool)\b", re.IGNORECASE,
)
TAG_RE = re.compile(
    r"\[(CHEAP|THINK|REASON|CODE|AGENT|FRONTIER|FUSION|NOVEL|DISCOVERY|NOFUSION)\b[^\]]*\]",
    re.IGNORECASE)
AVAIL_RE = re.compile(r"\[(UN)?AVAILABLE:\s*([^\]]+)\]", re.IGNORECASE)

CHARS_PER_TOKEN = 4
SHORT_PROMPT_TOKENS = 300

TAG_TIER = {"CHEAP": "cheap", "THINK": "reason", "REASON": "reason",
            "CODE": "code", "AGENT": "agent", "FRONTIER": "frontier"}


def load_availability() -> Dict[str, bool]:
    path = Path(os.environ.get("AVAILABILITY_CONFIG", "availability.yaml"))
    defaults = {p: True for p in PRIORITY_CHAIN}
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for name, cfg in (data.get("providers", {}) or {}).items():
                if name in defaults and isinstance(cfg, dict):
                    defaults[name] = bool(cfg.get("available", True))
        except Exception as exc:
            verbose_logger.warning("PriorityRouter: availability load failed: %s", exc)
    return defaults


def load_health() -> Dict[str, Dict[str, Any]]:
    """Per-NIM health from model_health.yaml. Missing/broken -> {} (fail-open)."""
    path = Path(os.environ.get("MODEL_HEALTH_CONFIG", "model_health.yaml"))
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("models", {}) or {}
    except Exception as exc:
        verbose_logger.warning("PriorityRouter: health load failed: %s", exc)
        return {}


def parse_request(prompt: str) -> Tuple[str, Dict[str, Any]]:
    directives: Dict[str, Any] = {"tier": None, "allowed": None, "denied": set()}
    for m in TAG_RE.finditer(prompt):
        tag = m.group(1).upper()
        if tag in TAG_TIER and directives["tier"] is None:
            directives["tier"] = TAG_TIER[tag]
    for m in AVAIL_RE.finditer(prompt):
        provs = {x.strip().lower() for x in m.group(2).split(",")}
        if m.group(1):
            directives["denied"] |= provs
        else:
            directives["allowed"] = provs
    cleaned = AVAIL_RE.sub("", TAG_RE.sub("", prompt)).strip()
    return cleaned, directives


def effective_availability(directives: Dict[str, Any],
                           base: Optional[Dict[str, bool]] = None) -> Dict[str, bool]:
    availability = dict(base) if base is not None else load_availability()
    result: Dict[str, bool] = {}
    for provider, ok in availability.items():
        if directives.get("allowed") is not None:
            ok = ok and provider in directives["allowed"]
        if provider in directives.get("denied", set()):
            ok = False
        result[provider] = ok
    return result


def _model_ok(model: str, availability: Dict[str, bool], health: Dict[str, Any]) -> bool:
    if not availability.get(MODEL_PROVIDER.get(model, ""), False):
        return False
    h = health.get(model)
    if h is not None and h.get("ok") is False:
        return False  # NIM model flagged slow/dead by the audit
    return True


def pick_model(tier: List[str], availability: Dict[str, bool],
               health: Dict[str, Any]) -> Optional[str]:
    for model in tier:
        if _model_ok(model, availability, health):
            return model
    return None


def classify(prompt: str) -> str:
    if CODE_MARKERS.search(prompt):
        return "code"
    if REASON_MARKERS.search(prompt):
        return "reason"
    if AGENT_MARKERS.search(prompt):
        return "agent"
    if len(prompt) <= SHORT_PROMPT_TOKENS * CHARS_PER_TOKEN:
        return "cheap"
    return "general"


def route(prompt: str, directives: Dict[str, Any], availability: Dict[str, bool],
          health: Optional[Dict[str, Any]] = None) -> Optional[str]:
    health = health or {}
    tier = directives.get("tier") or classify(prompt)
    chosen = pick_model(TIER_MAP.get(tier, GENERAL_TIER), availability, health)
    if chosen:
        verbose_logger.info("PriorityRouter: tier=%s -> %s", tier, chosen)
        return chosen
    for provider, models in PRIORITY_CHAIN.items():  # layer-5 cost-ordered chain
        for m in models:
            if _model_ok(m, availability, health):
                return m
    return None


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class PriorityRouter(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        messages = data.get("messages") or []
        prompt, directives = "", {"tier": None, "allowed": None, "denied": set()}
        for msg in reversed(messages):
            if msg.get("role") == "user":
                raw = extract_text(msg.get("content", ""))
                cleaned, directives = parse_request(raw)
                if cleaned != raw and isinstance(msg.get("content"), str):
                    msg["content"] = cleaned
                prompt = cleaned or raw
                break
        availability = effective_availability(directives)
        health = load_health()
        requested = data.get("model", "") or ""
        if requested in AUTO_MODELS:
            target = route(prompt, directives, availability, health)
            if target:
                data["model"] = target
                verbose_logger.info("PriorityRouter: auto -> %s", target)
        else:
            provider = MODEL_PROVIDER.get(requested)
            unhealthy = (provider is not None and not availability.get(provider, True)) \
                or (health.get(requested, {}).get("ok") is False)
            if unhealthy:
                target = route(prompt, directives, availability, health)
                if target:
                    verbose_logger.info("PriorityRouter: %s unavailable -> %s", requested, target)
                    data["model"] = target
        return data


router_instance = PriorityRouter()
```

- [ ] **Step 2: Run tests — verify they pass**

Run: `cd "$PROJ" && pip install pyyaml pytest litellm >/dev/null 2>&1; python3 -m pytest tests/test_priority_router.py -v`
Expected: all PASS. (If `litellm` import is heavy, tests still import it via the module; installing `litellm` in the venv is enough. Alternatively stub: `pip install litellm` once.)

- [ ] **Step 3: Commit**

```bash
cd "$PROJ" && git add priority_router.py tests/test_priority_router.py
git commit -m "feat: per-prompt cost-aware router with nim health-gate + tests"
```

---

## Task 6: availability.yaml

**Files:**
- Create: `$PROJ/availability.yaml`

- [ ] **Step 1: Write `availability.yaml`**

```yaml
# Provider on/off mask — read fresh every request, no restart. Per-request
# [AVAILABLE:]/[UNAVAILABLE:] tags override. Cost order in comments.
providers:
  nim:
    available: true
    note: "NVIDIA NIM — free workhorse (load-variable latency)"
  zen:
    available: true
    note: "opencode Zen — per-token; free-tier models are $0"
  copilot:
    available: true
    note: "GitHub Copilot — per-request credit; frontier / last resort"
```

- [ ] **Step 2: Commit**

```bash
cd "$PROJ" && git add availability.yaml && git commit -m "feat: provider availability mask"
```

---

## Task 7: nim_health.sh (latency audit)

**Files:**
- Create: `$PROJ/scripts/nim_health.sh`

- [ ] **Step 1: Write `scripts/nim_health.sh`** — probes each NIM alias through the running router, times it, writes `model_health.yaml`.

```sh
#!/bin/sh
# Session-start NIM latency audit. Pings each NIM alias via the router,
# measures total latency, writes model_health.yaml (read per-request by
# priority_router.py). Models slower than NIM_LATENCY_MAX_MS or failing => ok:false.
#
# Usage: sh scripts/nim_health.sh          (run at session start, or when NIM feels laggy)
set -e
cd "$(dirname "$0")/.."
. ./.env
KEY="$LITELLM_MASTER_KEY"
URL="http://localhost:4040/v1/chat/completions"
MAX_MS="${NIM_LATENCY_MAX_MS:-8000}"
OUT="model_health.yaml"

NIM_ALIASES="nim-glm nim-deepseek nim-deepseek-flash nim-kimi nim-qwen-max nim-qwen nim-minimax nim-nemotron nim-nemotron-super nim-mistral nim-llama"

printf '# Auto-generated by nim_health.sh — do not edit by hand.\nmodels:\n' > "$OUT"
printf '%-22s %-6s %s\n' "alias" "ok" "latency"
printf '%-22s %-6s %s\n' "----------------------" "------" "--------"

for a in $NIM_ALIASES; do
  body='{"model":"'"$a"'","messages":[{"role":"user","content":"ping"}],"max_tokens":1}'
  start=$(python3 -c 'import time; print(int(time.time()*1000))')
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 "$URL" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "$body" || echo 000)
  end=$(python3 -c 'import time; print(int(time.time()*1000))')
  ms=$((end - start))
  if [ "$code" = "200" ] && [ "$ms" -le "$MAX_MS" ]; then ok=true; else ok=false; fi
  printf '  %s: {ok: %s, latency_ms: %s}\n' "$a" "$ok" "$ms" >> "$OUT"
  printf '%-22s %-6s %sms (http %s)\n' "$a" "$ok" "$ms" "$code"
done

echo
echo "wrote $OUT — router now skips ok:false NIM models per request."
```

- [ ] **Step 2: Make executable + commit** (don't run yet — needs the router up, Task 8)

```bash
cd "$PROJ" && chmod +x scripts/nim_health.sh
git add scripts/nim_health.sh && git commit -m "feat: nim latency health audit -> model_health.yaml"
```

---

## Task 8: Fill .env, launch router, verify model IDs

**Files:**
- Create: `$PROJ/.env`
- Modify: `$PROJ/config.yaml` (only if discovery shows drift)

- [ ] **Step 1: Create `.env` with generated secrets + the NIM key**

```bash
cd "$PROJ"
MK="sk-$(openssl rand -hex 16)"
PG="$(openssl rand -hex 12)"
cat > .env <<EOF
LITELLM_MASTER_KEY=$MK
POSTGRES_PASSWORD=$PG
NVIDIA_API_KEY=nvapi-REDACTED-put-your-key-in-.env
ZEN_API_KEY=
EOF
chmod 600 .env
echo "master key: $MK"
```

- [ ] **Step 2: Add the Zen key** (Task 12 may also set it). Try opencode's stored auth first:

```bash
cd "$PROJ"
ZK=$(python3 -c 'import json,os,sys;
p=os.path.expanduser("~/.local/share/opencode/auth.json");
d=json.load(open(p)) if os.path.exists(p) else {};
print((d.get("opencode",{}) or {}).get("key","") or (d.get("opencode",{}) or {}).get("apiKey",""))' 2>/dev/null)
if [ -n "$ZK" ]; then
  sed -i '' "s|^ZEN_API_KEY=.*|ZEN_API_KEY=$ZK|" .env && echo "ZEN_API_KEY pulled from opencode auth"
else
  echo "ZEN_API_KEY not found in opencode auth — paste it into .env manually before Zen routing works"
fi
```

- [ ] **Step 3: Start the stack**

```bash
cd "$PROJ" && docker compose up -d
docker compose ps
```
Expected: `litellm-proxy` and `litellm-db` both `Up (healthy)` within ~40s.

- [ ] **Step 4: Health check**

Run: `curl -s http://localhost:4040/health/liveliness`
Expected: `"I'm alive!"`

- [ ] **Step 5: Discover live model IDs (NIM + Zen), fix any drift in `config.yaml`**

Write `scripts/discover_models.sh`:
```sh
#!/bin/sh
# List live model IDs from NIM + Zen. Update config.yaml litellm_params.model on drift.
set -e
cd "$(dirname "$0")/.."
[ -f .env ] && . ./.env
echo "=== NVIDIA NIM models ==="
[ -n "$NVIDIA_API_KEY" ] && curl -s https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NVIDIA_API_KEY" | python3 -m json.tool | grep '"id"' | head -80 || echo "no NVIDIA_API_KEY"
echo "=== opencode Zen models ==="
[ -n "$ZEN_API_KEY" ] && curl -s https://opencode.ai/zen/v1/models \
  -H "Authorization: Bearer $ZEN_API_KEY" | python3 -m json.tool | grep '"id"' | head -80 || echo "no ZEN_API_KEY (or Zen has no /models endpoint yet)"
echo
echo "On drift: edit config.yaml litellm_params.model, then: docker compose restart litellm"
```
Then:
```bash
cd "$PROJ" && chmod +x scripts/discover_models.sh && sh scripts/discover_models.sh
```
Compare printed IDs to `config.yaml`. **NIM:** correct any changed alias target. **Zen:** confirm `gpt-5.5`, `glm-5.2`, `deepseek-v4-pro` and the free-tier slugs (`big-pickle`, `deepseek-v4-flash-free`, `nemotron-3-ultra-free`); Zen may lack a `/models` endpoint (known gap) — if so, verify each with a 1-shot curl to `/v1/chat/completions` and fix slugs that 404. Restart on any edit: `docker compose restart litellm`.

- [ ] **Step 6: Run the NIM health audit (now that the router is up)**

```bash
cd "$PROJ" && sh scripts/nim_health.sh
```
Expected: a table of NIM aliases with `ok` + latency; `model_health.yaml` written.

- [ ] **Step 7: Commit any config fixes**

```bash
cd "$PROJ" && git add config.yaml scripts/discover_models.sh
git commit -m "fix: verify live NIM/Zen model IDs" || echo "no drift"
```

---

## Task 9: Copilot device-flow auth

**Files:**
- Create: `$PROJ/scripts/copilot_device_flow.sh`

- [ ] **Step 1: Check if opencode already logged Copilot in** (you mentioned it's setting up):

```bash
docker exec litellm-proxy sh -c 'ls -l /app/.litellm/github_copilot/access-token 2>/dev/null' && echo "token present" || echo "no token — run device flow"
```
If "token present", test a Copilot call and skip to Step 3.

- [ ] **Step 2: Write + run `scripts/copilot_device_flow.sh`** (original, `sudo` removed for OrbStack):

```sh
#!/bin/sh
# Direct GitHub device-flow auth for Copilot; writes the token into the running
# litellm-proxy container. Uses the client_id LiteLLM's copilot integration expects.
set -e
CLIENT_ID="Iv1.b507a08c87ecfe98"
resp=$(curl -s -X POST https://github.com/login/device/code \
  -H "accept: application/json" -H "content-type: application/json" \
  -H "editor-version: vscode/1.85.1" -H "editor-plugin-version: copilot/1.155.0" \
  -H "user-agent: GithubCopilot/1.155.0" \
  -d "{\"client_id\":\"$CLIENT_ID\",\"scope\":\"read:user\"}")
device_code=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["device_code"])')
user_code=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["user_code"])')
interval=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("interval",5))')
expires_in=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("expires_in",900))')
echo "USER_CODE: $user_code"
echo "Visit https://github.com/login/device — valid ${expires_in}s"
elapsed=0
while [ "$elapsed" -lt "$expires_in" ]; do
  sleep "$interval"; elapsed=$((elapsed + interval))
  poll=$(curl -s -X POST https://github.com/login/oauth/access_token \
    -H "accept: application/json" -H "content-type: application/json" \
    -H "editor-version: vscode/1.85.1" -H "editor-plugin-version: copilot/1.155.0" \
    -H "user-agent: GithubCopilot/1.155.0" \
    -d "{\"client_id\":\"$CLIENT_ID\",\"device_code\":\"$device_code\",\"grant_type\":\"urn:ietf:params:oauth:grant-type:device_code\"}")
  token=$(echo "$poll" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("access_token",""))
except Exception: print("")')
  if [ -n "$token" ]; then
    echo "$token" | docker exec -i litellm-proxy sh -c 'mkdir -p /app/.litellm/github_copilot && cat > /app/.litellm/github_copilot/access-token'
    echo "TOKEN_WRITTEN_OK"; exit 0
  fi
  err=$(echo "$poll" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("error",""))
except Exception: print("")')
  case "$err" in slow_down) interval=$((interval+5));; authorization_pending|"") ;; *) echo "GITHUB_ERROR: $err"; exit 1;; esac
  echo "waiting... (${elapsed}s/${expires_in}s)"
done
echo "TIMED_OUT"; exit 1
```
Run:
```bash
cd "$PROJ" && chmod +x scripts/copilot_device_flow.sh && sh scripts/copilot_device_flow.sh
```
Enter the printed code at https://github.com/login/device. Expected: `TOKEN_WRITTEN_OK`.

- [ ] **Step 3: Verify a Copilot route**

```bash
cd "$PROJ" && . ./.env
curl -s -m 40 http://localhost:4040/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"cop-sonnet","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}' | head -c 300
```
Expected: a JSON completion (not an auth error).

- [ ] **Step 4: Commit**

```bash
cd "$PROJ" && git add scripts/copilot_device_flow.sh && git commit -m "feat: copilot device-flow auth (orbstack, no sudo)"
```

---

## Task 10: Audit scripts

**Files:**
- Create: `$PROJ/scripts/show_routing.sh`, `$PROJ/scripts/route_test.sh`, `$PROJ/scripts/export_audit.sh`

- [ ] **Step 1: Write `scripts/show_routing.sh`** (sudo removed):

```sh
#!/bin/sh
# Which model the router actually chose for recent requests (from Postgres audit).
# Usage: sh scripts/show_routing.sh [N|watch]
N="${1:-15}"
query() {
  docker exec litellm-db psql -U litellm -d litellm -P pager=off -c \
    "SELECT to_char(\"startTime\",'HH24:MI:SS') AS time, model_group AS routed_alias,
            model AS actual_model, \"total_tokens\" AS tokens
     FROM \"LiteLLM_SpendLogs\" ORDER BY \"startTime\" DESC LIMIT ${1:-15};"
}
if [ "$N" = "watch" ]; then
  while true; do clear; echo "Live routing (Ctrl-C) — newest first"; query 15; sleep 2; done
else query "$N"; fi
```

- [ ] **Step 2: Write `scripts/route_test.sh`** (sudo removed; Zen fallback case instead of Z.ai):

```sh
#!/bin/sh
# Fire one prompt per tier; show which model served each (via show_routing.sh after).
set -e
cd "$(dirname "$0")/.."
. ./.env
KEY="$LITELLM_MASTER_KEY"; URL="http://localhost:4040/v1/chat/completions"
req() { echo "=== $1 ==="; curl -s -m 40 "$URL" -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d "$2" | head -c 300; echo; echo; }
req "auto short (cheap -> nim-llama)" '{"model":"auto","messages":[{"role":"user","content":"Say hello"}]}'
req "auto [THINK] (reason -> nim-qwen-max)" '{"model":"auto","messages":[{"role":"user","content":"[THINK] prove sqrt2 irrational"}]}'
req "auto code (code -> nim-deepseek)" '{"model":"auto","messages":[{"role":"user","content":"debug:\n```python\nprint(1)\n```"}]}'
req "auto [FRONTIER] (-> cop-opus)" '{"model":"auto","messages":[{"role":"user","content":"[FRONTIER] hard design question"}]}'
req "auto [UNAVAILABLE: nim] short (-> zen free)" '{"model":"auto","messages":[{"role":"user","content":"[UNAVAILABLE: nim] say ok"}]}'
echo "=== PriorityRouter log lines ==="
docker compose logs --since 5m litellm 2>&1 | grep -i PriorityRouter | tail -10 || true
```

- [ ] **Step 3: Write `scripts/export_audit.sh`** (sudo removed; `--push` guarded on a remote existing):

```sh
#!/bin/sh
# Snapshot the routing/spend audit trail to logs/audit-*.csv. --push commits+pushes.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
stamp=$(date +%Y%m%d-%H%M%S); out="logs/audit-$stamp.csv"
docker exec litellm-db psql -U litellm -d litellm -P pager=off --csv -c "
  SELECT \"startTime\" AS time, model_group AS routed_alias, model AS actual_model,
         prompt_tokens, completion_tokens, total_tokens, spend, \"cache_hit\" AS cache_hit
  FROM \"LiteLLM_SpendLogs\" ORDER BY \"startTime\" ASC;" > "$out"
rows=$(($(wc -l < "$out") - 1)); echo "exported $rows requests -> $out"
ls -1t logs/audit-*.csv 2>/dev/null | tail -n +11 | xargs -r rm --
if [ "$1" = "--push" ]; then
  git add logs/
  git commit -q -m "audit: routing snapshot $stamp ($rows requests)" || echo "nothing to commit"
  git remote get-url origin >/dev/null 2>&1 && git push -q origin main && echo "pushed" || echo "no remote — local commit only"
fi
```

- [ ] **Step 4: Make executable, smoke-test routing, commit**

```bash
cd "$PROJ" && chmod +x scripts/show_routing.sh scripts/route_test.sh scripts/export_audit.sh
sh scripts/route_test.sh
sh scripts/show_routing.sh 10
```
Expected: `show_routing` lists rows with **distinct `actual_model` per tier** (nim-llama for short, nim-qwen-max for [THINK], nim-deepseek for code, cop-opus for [FRONTIER], a zen-* for the nim-masked short).
```bash
git add scripts/ && git commit -m "feat: audit + route-test scripts (orbstack)"
```

---

## Task 11: agentmemory → free NIM

**Files:**
- Modify: `~/.agentmemory/.env`

- [ ] **Step 1: Back up + append router LLM config** (agentmemory container already running on :3111):

```bash
cd "$PROJ" && . ./.env
cp ~/.agentmemory/.env ~/.agentmemory/.env.bak.$(date +%Y%m%d)
cat >> ~/.agentmemory/.env <<EOF

# llm-router wiring — memory LLM features run on free NIM via the router
OPENAI_API_KEY=$LITELLM_MASTER_KEY
OPENAI_BASE_URL=http://localhost:4040/v1
OPENAI_MODEL=nim-llama
GRAPH_EXTRACTION_ENABLED=true
AGENTMEMORY_AUTO_COMPRESS=true
AGENTMEMORY_INJECT_CONTEXT=true
CONSOLIDATION_ENABLED=true
EOF
```

- [ ] **Step 2: Restart the agentmemory container + verify**

```bash
docker restart agentmemory-iii-engine-1
sleep 5
curl -s http://localhost:3111/livez 2>/dev/null || curl -s http://localhost:3111/ | head -c 120
```
Expected: agentmemory responds. If an `agentmemory doctor` CLI is available (`~/.local/bin/agentmemory doctor`), run it and confirm it reports the OpenAI base URL as the router. (Note: `.env` here is agentmemory's own config, outside the git repo — nothing to commit.)

---

## Task 12: opencode — install + wire router + plugins

**Files:**
- Modify: `~/.config/opencode/opencode.json`

- [ ] **Step 1: Install opencode** (binary missing from PATH):

```bash
curl -fsSL https://opencode.ai/install | bash
export PATH="$HOME/.opencode/bin:$PATH"
grep -q '.opencode/bin' ~/.zshrc || echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> ~/.zshrc
opencode --version
```
Expected: prints a version.

- [ ] **Step 2: Back up current config, then write the router provider + MCP + plugins**

```bash
cp ~/.config/opencode/opencode.json ~/.config/opencode/opencode.json.bak.$(date +%Y%m%d)
```
Write `~/.config/opencode/opencode.json` (substitute `MASTER_KEY` with the value from `$PROJ/.env`):
```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "caveman@git+https://github.com/JuliusBrussee/caveman.git",
    "superpowers@git+https://github.com/obra/superpowers.git"
  ],
  "provider": {
    "llm-router": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LLM Router (LiteLLM :4040)",
      "options": {
        "baseURL": "http://localhost:4040/v1",
        "apiKey": "MASTER_KEY"
      },
      "models": {
        "auto":         { "name": "auto — per-prompt cost/latency router" },
        "nim-glm":      { "name": "GLM 5.2 (NIM)" },
        "nim-deepseek": { "name": "DeepSeek V4 Pro (NIM)" },
        "nim-qwen-max": { "name": "Qwen 3.5 397B (NIM)" },
        "nim-kimi":     { "name": "Kimi K2.6 (NIM)" },
        "nim-llama":    { "name": "Llama 3.3 70B (NIM)" },
        "zen-gpt":      { "name": "GPT-5.5 (opencode Zen)" },
        "zen-free-nemotron": { "name": "Nemotron Ultra Free (Zen)" },
        "cop-opus":     { "name": "Claude Opus 4.8 (Copilot)" },
        "cop-sonnet":   { "name": "Claude Sonnet 5 (Copilot)" }
      }
    }
  },
  "mcp": {
    "lean-ctx": {
      "type": "local",
      "command": ["<repo-root>"],
      "enabled": true,
      "environment": { "LEAN_CTX_DATA_DIR": "<repo-root>" }
    },
    "agentmemory": {
      "type": "local",
      "command": ["npx", "-y", "@agentmemory/agentmemory", "mcp"],
      "enabled": true,
      "environment": { "AGENTMEMORY_URL": "http://localhost:3111" }
    }
  }
}
```

- [ ] **Step 3: Set default model + caveman ultra in `AGENTS.md`**

Append to `~/.config/opencode/AGENTS.md`:
```
## Model / routing defaults
- Default model: llm-router/auto (resolves to GLM 5.2 for normal work).
- Caveman mode: ULTRA (drop articles/filler; code/commits normal).
- Trivial asks route cheap automatically; tag [THINK]/[CODE]/[FRONTIER] to force a tier.
- [NOVEL] (phase-2 fusion) is not active yet in v1 — ignored/stripped.
```

- [ ] **Step 4: Launch opencode, pick `auto`, confirm routing**

```bash
opencode
# inside: /models -> llm-router -> auto ; then send "say hello"
# in another shell:
cd "$PROJ" && sh scripts/show_routing.sh 5
```
Expected: opencode replies; `show_routing` shows the request served by a NIM model (e.g. `nim-llama`/`nim-glm`). caveman terseness visible in output; lean-ctx + agentmemory listed under opencode MCP.

---

## Task 13: End-to-end proof + rotate key

- [ ] **Step 1: Full verification sweep**

```bash
cd "$PROJ"
curl -s http://localhost:4040/health/liveliness            # "I'm alive!"
sh scripts/nim_health.sh                                   # live NIM latencies
sh scripts/route_test.sh                                   # tier prompts
sh scripts/show_routing.sh 15                              # distinct models per tier
sh scripts/export_audit.sh                                 # CSV snapshot written
```
Expected: health OK; health table printed; `show_routing` shows the tier→model mapping (nim-llama short / nim-qwen-max [THINK] / nim-deepseek code / cop-opus [FRONTIER] / zen-* when nim masked); a `logs/audit-*.csv` exists.

- [ ] **Step 2: Rotate the NIM key** (it was pasted in chat during design — security §13):

Regenerate at https://build.nvidia.com → update `NVIDIA_API_KEY` in `$PROJ/.env` → `docker compose restart litellm` → re-run `sh scripts/nim_health.sh` to confirm NIM still routes.

- [ ] **Step 3: Final commit**

```bash
cd "$PROJ" && git add -A
git commit -m "feat: llm-router mac phase-1 complete — verified routing + audit"
```

---

## Self-review notes (coverage vs spec)

- §3 architecture, §4 backends → Tasks 2,3 (compose, config with NIM+Zen+Copilot). ✅
- §5 per-prompt cost-aware routing → Task 5 (`priority_router.py`) + Task 4 tests. ✅
- §6 NIM health-gate → Task 5 (`load_health`/`_model_ok`) + Task 7 (`nim_health.sh`). ✅
- §7 agentmemory→NIM → Task 11. ✅
- §8 opencode + caveman-ultra + superpowers → Task 12. ✅
- §9 audit trail → Task 10 + Task 13. ✅
- §12 security (key rotation, .env gitignored, 127.0.0.1 bind) → Task 1, Task 13. ✅
- §11 fusion is **Phase 2** — a separate plan after v1 runs (out of scope here; `[NOVEL]` is only stripped/reserved in v1, covered by Task 4 passthrough test).

**Known build-time verifications (not placeholders):** exact NIM model IDs and Zen free-tier slugs are confirmed against live `/models` in Task 8 Step 5 and corrected there; Zen may lack a `/models` endpoint, handled with per-model curl checks.
