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

import hashlib as _hashlib
import json as _json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

try:  # litellm is present in the proxy container; host tools import only the pure functions
    from litellm.integrations.custom_logger import CustomLogger
    from litellm._logging import verbose_logger
except ModuleNotFoundError:  # pragma: no cover - host fallback (fusion CLI/MCP, standalone use)
    class CustomLogger:  # minimal stand-in; PriorityRouter is only instantiated in-container
        pass

    class _VerboseLogger:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    verbose_logger = _VerboseLogger()

# Provider priority (cost order). Dict order IS the layer-5 fallback order.
PRIORITY_CHAIN: Dict[str, List[str]] = {
    "nim": [
        "nim-glm", "nim-deepseek", "nim-inkling", "nim-gptoss",
        "nim-minimax", "nim-nemotron", "nim-nemotron-super",
        "nim-step", "nim-deepseek-flash", "nim-llama",
    ],
    # Mistral free tier (api.mistral.ai, LiteLLM native mistral/). Capable models only:
    # large/codestral (non-reason) + medium/magistral (reason natively, no param). Free -> class 0.
    "mistral": ["mis-large", "mis-codestral", "mis-medium", "mis-magistral"],
    # z.ai GLM Coding Plan (flat-rate). Served ONLY via the Anthropic-compat endpoint
    # (/api/anthropic) — the OpenAI /paas/v4 lane 429s "insufficient balance". Wired as LiteLLM
    # anthropic/ provider; reasoners take the Anthropic thinking block. Flat -> class 1 (before GO).
    "zai": ["zai-turbo", "zai-52", "zai-51", "zai-flash"],
    "zen": [
        "free-deepseek", "free-nemotron", "free-pickle",
        "free-ling", "free-laguna", "free-mimo", "free-north",
        "go-glm", "go-deepseek", "zen-gpt", "zen-fable", "zen-opus",
        # GO subscription (flat-rate /zen/go/v1): capable models, no per-token cost
        "go-mimo-lite", "go-hy3", "go-luna",   # added 2026-07-30, most generous GO quota first
        "go-minimax27", "go-qwen36", "go-kimi26", "go-glm51",
        "go-kimi", "go-minimax", "go-qwen-max", "go-qwen-plus", "go-mimo", "go-deepseek-flash",
        # PAID per-token overflow ($20 balance) — class 2 (not in GO_ALIASES), fallback/boost only
        "zen-minimax", "zen-qwen-plus", "zen-glm",
        "zen-luna", "zen-grok", "zen-qwen-max", "zen-kimi3",
        "zen-terra", "zen-sol",
        # LAST_RESORT_BRAINS at the very end of the chain: only if literally everything else is gone.
        "go-grok", "go-kimi-k3",
    ],
    "copilot": [
        "co-opus", "co-sonnet", "co-gpt", "co-codex", "co-gemini",
        "co-haiku", "co-mini",
    ],
    # Claude Max subscription via OAuth (flat-rate, rate-limited by Anthropic)
    "anthropic": ["ant-fable", "ant-opus", "ant-sonnet", "ant-haiku"],
    # Codex/ChatGPT subscription (flat) via providers/codex_oauth_proxy.py on :4042.
    "codex": ["cod-sol", "cod-terra", "cod-luna", "cod-mini"],
}

# Tiers: NIM first (health-gated) -> Zen (free before paid) -> ant-* -> Copilot tail.
# Copilot never appears in cheap/general — no per-request credit on routine work.
# NOTE: GPT-family Copilot models (co-gpt/co-codex/co-mini) fail via litellm's
# Copilot path (GPT needs the /responses API). Use Claude/Gemini Copilot + Zen's
# GPT (zen-gpt) for GPT needs instead. GPT-via-Copilot kept in config but unused.
# Flat/subscription members are ordered by QUOTA WEIGHT, not latency: they are never
# latency-probed (probing burns the very quota they are held in reserve for), so config order
# is what decides inside a cost class. Codex quota per OpenAI's published table — credits/1M
# in-out and messages per 5h window: luna 5/30 (250-2000 msgs) << mini 18.75/113 (60-350)
# << 5.5 (15-80) << sol 125/750 (10-100). So luna leads: cheapest on quota AND newer-gen.
CHEAP_TIER    = ["nim-llama", "nim-deepseek-flash", "free-deepseek", "free-pickle", "mis-codestral", "go-deepseek-flash", "go-mimo-lite", "zai-flash", "ant-haiku", "cod-luna", "cod-mini", "co-haiku"]  # free-ling in REASON (native reasoner), not cheap
GENERAL_TIER  = ["nim-glm", "nim-inkling", "nim-step", "free-nemotron", "free-laguna", "mis-large", "go-mimo-lite", "go-hy3", "go-minimax27", "go-qwen36", "go-luna", "go-glm", "zai-turbo", "zai-flash", "ant-sonnet", "cod-luna", "cod-mini", "co-sonnet"]  # free-laguna: new free worker (health-gated; type unconfirmed, was rate-limited at wiring)
CODE_TIER     = ["nim-deepseek", "nim-gptoss", "nim-step", "free-north", "mis-large", "mis-codestral", "go-luna", "go-deepseek", "go-kimi", "go-kimi26", "zai-turbo", "ant-sonnet", "cod-terra", "co-sonnet"]  # free-north/nim-step = free code reasoners; mis-codestral = free code specialist
# REASON leads with NIM thinking models (nim-glm/kimi/minimax get HIGH budget at this tier —
# Phase 1.6). Without them here, "NIM -> HIGH on reason" was unreachable: the tier held only
# non-thinking NIM models, so the budget table never applied. zen-gpt dropped (dead: 401).
REASON_TIER   = ["nim-glm", "nim-minimax", "nim-step", "free-ling", "free-mimo", "free-north", "mis-medium", "mis-magistral", "nim-inkling", "nim-nemotron", "go-hy3", "go-qwen36", "go-kimi26", "go-glm", "go-glm51", "go-qwen-max", "zai-52", "zai-51", "free-nemotron", "ant-sonnet", "co-opus"]  # free-*/mis-*/nim-step = free native reasoners; zai-5x = flat GLM reasoners
AGENT_TIER    = ["nim-glm", "nim-minimax", "nim-step", "mis-large", "mis-medium", "go-mimo-lite", "go-luna", "go-minimax27", "go-minimax", "go-glm", "zai-turbo", "ant-sonnet", "cod-terra", "co-sonnet"]
# FRONTIER = cost-first order, every member at HIGH thinking: free NIM (if healthy) -> GO
# flat-rate -> Anthropic Max -> Copilot. Health-gating handles "if available". NIM thinking
# models lead so a frontier task can run on free GLM 5.2 + 32k thinking before touching paid.
# zen-gpt dropped (dead: 401). Config order = intent (frontier is not latency-sorted).
FRONTIER_TIER = ["nim-glm", "nim-minimax", "mis-medium", "zai-52", "go-glm", "go-qwen-max", "ant-opus", "ant-fable", "ant-sonnet", "cod-sol", "co-opus", "co-sonnet", "co-gemini",
                 "go-grok", "go-kimi-k3"]   # LAST_RESORT_BRAINS: frontier tail only, after everything else  # mis-medium(free)/zai-52(flat) reasoners between free NIM and GO
# ORCHESTRATOR / AUDITOR = checks scout/agent output and gives a verdict — LOW token, run often.
# QUALITY-first order (NOT latency-sorted): capable, high-quota reliables (go-glm, ant-sonnet/opus
# Max flat-rate, GO reasoners, nim-glm free) carry every verdict. The near-frontier brains
# (grok/kimi-k3) are DELIBERATELY absent — they're RESTRICTED_AUTO (explicit-only), because their
# low GO quota + GO->paid-Zen overflow on saturation make auto-selection a cost trap. Every member
# runs HIGH thinking.
# Orchestrator brains (grok-4.5, kimi-k3): near-frontier but LOW GO quota, and a GO 429 when the
# plan is saturated overflows to PER-TOKEN Zen billing on the same account (both keys charge). So
# they are RESTRICTED from ALL auto-routing — never selected by any tier, boost, or fallback.
# Reachable ONLY by an explicit alias request (pick "go-kimi-k3" yourself). Enforced in _model_ok,
# which is the single gate for every auto path (pick_model + the layer-5 chain).
# The two scarcest GO models (grok-4.5 120 req/5h, kimi-k3 110 req/5h). NOT blocked outright:
# reachable by naming the alias, and present at the very END of FRONTIER as the last thing tried
# before giving up. Kept out of every other tier because a GO 429 when the plan is saturated
# overflows to PER-TOKEN Zen billing on the same account — routine auto-selection is a cost trap
# (this is what produced the surprise 63k-token kimi-k3 calls).
LAST_RESORT_BRAINS = frozenset({"go-kimi-k3", "go-grok"})
RESTRICTED_AUTO = LAST_RESORT_BRAINS   # back-compat alias

# PREMIUM: scarce quota and/or frontier price. Allowed ONLY in the frontier and orchestrator
# tiers, never on everyday work, no matter how much cheaper capacity is down. Each everyday tier
# already carries its own flat fallback, so there is never a reason to reach for one of these:
#   go-grok      120 req/5h      go-kimi-k3   110 req/5h    (the two scarcest GO models)
#   cod-sol      10-100 msgs/5h, 125/750 credits per 1M     (25x go-luna)
#   ant-opus / ant-fable          frontier-priced Claude
# Enforced in route(); a test asserts none of them appears in an everyday tier list.
PREMIUM_ONLY = LAST_RESORT_BRAINS | frozenset({"cod-sol", "ant-opus", "ant-fable"})
# Cost-safe order (explicit, NOT latency-sorted): FREE reasoners -> z.ai flat -> Anthropic Max flat
# -> zen GO LAST. Even [BOOST][ORCH] stays on free/flat and only reaches zen GO if all of those are
# down — because a GO 429 when saturated overflows to per-token Zen. zen paid is never in the tier
# (reached only via a GO model's own fallback if literally everything else is unavailable). Brains
# excluded (RESTRICTED_AUTO).
ORCHESTRATOR_TIER = ["mis-medium", "nim-glm",                         # free reasoners
                     "zai-52",                                          # z.ai flat
                     "ant-sonnet", "ant-opus",                          # Anthropic Max flat
                     "go-glm", "go-qwen-max", "go-qwen-plus", "go-deepseek", "go-minimax", "go-mimo"]  # zen GO LAST

# --- Thinking-capable model sets and budget table ---
NIM_THINKING = frozenset({"nim-glm", "nim-minimax"})
GO_THINKING  = frozenset({"go-glm", "go-glm51", "go-qwen36", "go-kimi26",
                          "go-kimi", "go-minimax", "go-grok", "go-kimi-k3",
                          "go-qwen-max", "go-qwen-plus", "go-deepseek",  # verified live: these emit CoT
                          # PAID overflow reasoners (same underlying models -> reasoning_effort;
                          # gpt-5.6 luna/terra/sol take reasoning_effort too. minimax excluded: no CoT).
                          "zen-glm", "zen-qwen-plus", "zen-qwen-max", "zen-grok",
                          "zen-kimi3", "zen-luna", "zen-terra", "zen-sol"})
# Codex reasoners: the proxy maps OpenAI `reasoning_effort` -> Responses `reasoning.effort`.
COD_THINKING = frozenset({"cod-sol"})
ANT_THINKING = frozenset({"ant-opus", "ant-fable", "ant-sonnet"})  # ant-haiku excluded
# opencode Zen FREE-tier native reasoners. Verified live: they emit a reasoning_content trace by
# DEFAULT with no param, and reasoning_effort only REDUCES that trace. Free, so never a reason to
# hold back — always HIGH, and we inject NO param so their native max-depth reasoning is preserved
# (a param here would shrink it for zero benefit). They exist only for the annotation + tiering.
FREE_REASONERS = frozenset({"free-ling", "free-mimo", "free-north"})
# Mistral reasoners (magistral / mistral-medium hybrid): reason natively on api.mistral.ai with NO
# param — verified live, they 200 with reasoning inline (no separate reasoning_content field to
# toggle). Treated like FREE_REASONERS: inject nothing, annotate high.
MIS_REASONERS = frozenset({"mis-medium", "mis-magistral"})
# NIM native reasoners: step-3.7-flash emits a reasoning_content trace by DEFAULT (verified live:
# 1505 chars with no param; enable_thinking barely changes it). So it is NOT in NIM_THINKING —
# it needs no chat_template_kwargs toggle; treat it like the other native reasoners (inject none).
NIM_NATIVE = frozenset({"nim-step"})
NATIVE_REASONERS = FREE_REASONERS | MIS_REASONERS | NIM_NATIVE
# z.ai GLM reasoners on the Anthropic-compat endpoint: verified live they accept the Anthropic
# thinking block {"type":"enabled","budget_tokens":N} and return a `thinking` content block. Same
# injection + budget table as ANT_THINKING (think-class "ant"), even though cost-wise they're flat.
ZAI_THINKING  = frozenset({"zai-52", "zai-51"})
ALL_THINKING  = NIM_THINKING | GO_THINKING | ANT_THINKING | ZAI_THINKING | COD_THINKING | NATIVE_REASONERS

# (class, level) -> budget_tokens
_THINK_TOKENS: Dict[Tuple[str, str], int] = {
    ("nim", "high"):   32768,
    ("go",  "medium"):  8192,
    ("go",  "high"):   16384,
    ("ant", "medium"):  8192,
    ("ant", "high"):   16384,
    ("free", "high"):       0,   # native reasoner — no injected budget
}

# tier -> (nim_level, go_level, ant_level); absent = no thinking
_THINK_TABLE: Dict[str, Tuple[str, str, str]] = {
    "reason":   ("high", "medium", "medium"),
    "code":     ("high", "medium", "medium"),
    "agent":    ("high", "medium", "medium"),
    "frontier": ("high", "high",   "high"),
    "orchestrator": ("high", "high", "high"),   # top-layer brain — always max reasoning
}


def _model_think_class(model: str) -> Optional[str]:
    if model in NIM_THINKING:     return "nim"
    if model in GO_THINKING:      return "go"
    if model in ANT_THINKING or model in ZAI_THINKING: return "ant"  # Anthropic thinking-block shape
    if model in COD_THINKING:     return "go"   # Codex takes the same effort knob (no token budget)
    if model in NATIVE_REASONERS: return "free"
    return None


def _think_budget(model: str, tier: str, boost: bool) -> Optional[Tuple[str, int]]:
    """Return (level, budget_tokens) or None if model/tier don't warrant thinking."""
    cls = _model_think_class(model)
    if cls is None:
        return None
    if cls == "free":
        return ("high", 0)          # free native reasoner: always HIGH, no budget, any tier
    if boost:
        level = "high"
    else:
        row = _THINK_TABLE.get(tier)
        if row is None:
            return None
        level = row[{"nim": 0, "go": 1, "ant": 2}[cls]]
    return (level, _THINK_TOKENS[(cls, level)])


_THINK_STATE = Path("/tmp/llmr-last-think.json")

# Bind-mounted to the host (docker-compose) so the health refresher can see it. When a PAID model
# answers, we touch this file; the host refresher polls it and re-audits the FREE models within
# ~30s (not the 15-min tick), so a recovered free model is picked on the NEXT prompt instead of
# lingering on the paid lane. Derived from MODEL_HEALTH_CONFIG's dir so it sits beside it (/app).
_REFRESH_TRIGGER = Path(
    os.environ.get("MODEL_HEALTH_CONFIG", "model_health.yaml")
).resolve().parent / ".llmr-refresh-trigger"


def _trigger_free_refresh() -> None:
    """Signal the host refresher to re-audit free models now. Best-effort: a write failure (e.g. a
    host-side import of this module with no bind mount) is harmless — the 15-min tick still runs."""
    try:
        _REFRESH_TRIGGER.write_text(str(int(time.time())))
    except Exception:
        pass


def _served_is_paid(model: Any) -> bool:
    # response.model / chunk.model echoes the deployment ALIAS (verified: health asserts served==alias),
    # so a paid call is exactly a zen-* alias. Underlying model names are shared with GO twins,
    # so only the alias distinguishes paid from flat — never match on the underlying id.
    return isinstance(model, str) and model.startswith("zen-")


def _write_think_state(model: str, tier: str, level: str) -> None:
    try:
        _THINK_STATE.write_text(_json.dumps({"model": model, "tier": tier, "think": level}))
    except Exception:
        pass


MODEL_PROVIDER = {m: p for p, models in PRIORITY_CHAIN.items() for m in models}

TIER_MAP = {
    "cheap": CHEAP_TIER, "general": GENERAL_TIER, "code": CODE_TIER,
    "reason": REASON_TIER, "agent": AGENT_TIER, "frontier": FRONTIER_TIER,
    "orchestrator": ORCHESTRATOR_TIER,
}

# Every zero-marginal-cost alias, steadiest host first (zen-free -> Mistral -> load-variable NIM).
# free_fallback() appends these to a tier so that when a tier's own free models are all down, a
# borrowed free model answers instead of a flat subscription. RESTRICTED_AUTO is irrelevant here
# (the brains are not free), but _model_ok still gates everything at pick time.
FREE_POOL = [
    "free-nemotron", "free-ling", "free-north", "free-mimo",
    "free-deepseek", "free-pickle", "free-laguna",
    "mis-large", "mis-medium", "mis-codestral", "mis-magistral",
    "nim-glm", "nim-inkling", "nim-step", "nim-gptoss", "nim-deepseek",
    "nim-nemotron", "nim-minimax", "nim-deepseek-flash", "nim-llama", "nim-nemotron-super",
]

# Tiers that borrow free capacity. frontier/orchestrator are EXCLUDED on purpose: those are
# explicit quality-intent lists, and a small free model must not answer a frontier prompt just
# because it costs nothing. If they run dry, the layer-5 cost chain still covers it.
FREE_FALLBACK_TIERS = ("cheap", "general", "code", "reason", "agent")

# How demanding each tier is. Used for two things: borrowing may only go DOWN this scale, and a
# very large prompt is refused a model that only ever qualified for the easy end of it.
TIER_CAPABILITY = {"cheap": 0, "general": 1, "code": 2, "agent": 2, "reason": 3,
                   "frontier": 4, "orchestrator": 4}


def _capability_rank(model: str) -> int:
    """Highest tier the model natively belongs to — its demonstrated capability level.

    A model listed in REASON is trusted for anything easier; one listed only in CHEAP is not
    trusted upward. Unlisted models (reachable only via the layer-5 chain) rank 0."""
    return max((TIER_CAPABILITY.get(name, 0)
                for name, tier in TIER_MAP.items() if model in tier), default=0)


# Beyond this the prompt is a long agentic conversation, not a question, and a model that only
# ever qualified for CHEAP/GENERAL must not answer it however fast or free it is. Observed in the
# audit trail: big-pickle served 81,802 tokens and llama-3.3-70b served 79,948 — they returned
# *something*, which is exactly why this failed silently instead of erroring.
LARGE_PROMPT_TOKENS = 25_000

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
    r"\[(CHEAP|THINK|REASON|CODE|AGENT|FRONTIER|ORCH|AUDIT|SCOUT|ANALYST|VERIFIER|AUDITOR"
    r"|BOOST|FUSION|NOVEL|DISCOVERY|NOFUSION)\b[^\]]*\]",
    re.IGNORECASE)
AVAIL_RE = re.compile(r"\[(UN)?AVAILABLE:\s*([^\]]+)\]", re.IGNORECASE)
BOOST_RE = re.compile(
    r"\b(redo|not good|wrong answer|bad answer|shallow|try again|doesn't work|"
    r"not right|incorrect|improve this|too shallow)\b",
    re.IGNORECASE,
)

CHARS_PER_TOKEN = 4
SHORT_PROMPT_TOKENS = 300

TAG_TIER = {"CHEAP": "cheap", "THINK": "reason", "REASON": "reason",
            "CODE": "code", "AGENT": "agent", "FRONTIER": "frontier",
            "ORCH": "orchestrator", "AUDIT": "orchestrator",
            # Research-pipeline role tags: workers under the [ORCH] boss.
            "SCOUT": "cheap",       # read / list / gather — fast + free
            "ANALYST": "reason",    # interpret findings
            "VERIFIER": "reason",   # check claims vs sources
            "AUDITOR": "reason"}    # claim-level check (distinct from project-level [AUDIT])


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


# A model measured at or below this is treated as "does not cache". Deliberately near zero rather
# than a quality bar: the aim is to demote models that cache NOTHING (free-north 0.0% over 50
# samples, nim-step 0.0% over 28), not to rank the ones that merely cache less well.
CACHE_MIN_HIT_PCT = 5.0
CACHE_MIN_SAMPLES = 5


def load_cache_profile() -> Dict[str, Dict[str, Any]]:
    """Per-model prompt-cache hit rate from model_cache.yaml (written by scripts/cache_audit.sh).
    Missing/broken -> {} (fail-open: nothing is penalised)."""
    path = Path(os.environ.get("MODEL_CACHE_CONFIG", "model_cache.yaml"))
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("models", {}) or {}
    except Exception as exc:
        verbose_logger.warning("PriorityRouter: cache profile load failed: %s", exc)
        return {}


def _cache_rank(model: str, profile: Dict[str, Dict[str, Any]]) -> int:
    """0 = caches (or unknown), 1 = measured at ~zero.

    Unknown ranks 0 on purpose. Penalising a model we have not measured would quietly bias routing
    towards whatever happens to have traffic already, which is how a metric becomes self-fulfilling."""
    row = profile.get(model)
    if not row:
        return 0
    try:
        # A probe is a direct measurement (send the same prefix twice, read cached_tokens), not a
        # statistical sample, so it does not need a sample floor. Observed-traffic rows do: one
        # unlucky request must not condemn a model.
        if row.get("source") != "probe" and int(row.get("samples", 0)) < CACHE_MIN_SAMPLES:
            return 0                      # too thin to act on
        return 0 if float(row.get("hit_pct", 0)) > CACHE_MIN_HIT_PCT else 1
    except (TypeError, ValueError):
        return 0


# --- session stickiness -------------------------------------------------------------------------
# Switching model mid-conversation throws away the provider's prompt cache: the next turn re-runs
# prefill over the whole payload. Measured switch rate on large requests is 15.6%, so ~84% of turns
# already stay put — this protects that majority rather than creating it.
#
# In-memory and per-container on purpose. A restart forgetting every session is harmless (the next
# turn simply re-decides), and it avoids a persistence layer for state whose worst-case loss is one
# cache miss.
_SESSION_MODELS: Dict[str, Tuple[str, str]] = {}     # key -> (tier, model)
_SESSION_MAX = 512


def session_key(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Stable identifier for a conversation: a hash of its FIRST user message.

    The first turn does not change as the conversation grows, so the key survives the whole
    session. Two conversations opening with identical text collide — harmless, since identical
    openings route identically anyway."""
    for m in messages:
        if m.get("role") == "user":
            text = extract_text(m.get("content", ""))
            if text:
                return _hashlib.sha256(text[:2000].encode("utf-8", "ignore")).hexdigest()[:16]
    return None


def _sticky_get(key: Optional[str], tier: str, availability: Dict[str, bool],
                health: Dict[str, Any]) -> Optional[str]:
    """The model this conversation is already on, if it is still a valid choice.

    Deliberately re-checks availability and health: stickiness must not pin a session to a model
    that has since died, which would turn one bad probe into a broken conversation."""
    if not key:
        return None
    entry = _SESSION_MODELS.get(key)
    if not entry:
        return None
    prev_tier, model = entry
    if prev_tier != tier:
        return None          # the work changed character; re-decide rather than force continuity
    if model in PREMIUM_ONLY and tier not in ("frontier", "orchestrator"):
        return None
    return model if _model_ok(model, availability, health) else None


def _sticky_put(key: Optional[str], tier: str, model: str) -> None:
    if not key or not model:
        return
    if len(_SESSION_MODELS) >= _SESSION_MAX:
        _SESSION_MODELS.pop(next(iter(_SESSION_MODELS)), None)   # drop oldest
    _SESSION_MODELS[key] = (tier, model)


# Anthropic is the ONE lane that does not cache on its own. Zen (free + GO), z.ai and Codex all
# cache automatically — verified live, and on z.ai sending cache_control changes nothing. Anthropic
# requires an explicit breakpoint, so without this ant-* re-reads the entire payload every turn:
# measured 1.87M prompt tokens with exactly 0 cached. With it, a 19,568-token payload came back as
# 8 input + 19,559 cache_read on the following turn.
#
# Two breakpoints: the system prompt (stable for the whole session) and the end of the history
# before the newest turn (grows, and is what makes an agentic session expensive). Anthropic allows
# up to 4; using the minimum keeps the request shape simple.
ANTHROPIC_CACHE_MIN_TOKENS = 2048       # haiku's floor; sonnet/opus cache from 1024


def _mark_cache_control(msg: Dict[str, Any]) -> bool:
    """Attach an ephemeral cache breakpoint to a message. True if it was applied.

    Anthropic only accepts cache_control on a content BLOCK, so a plain string body has to be
    promoted to a one-element block list first."""
    content = msg.get("content")
    if isinstance(content, str):
        if not content:
            return False
        msg["content"] = [{"type": "text", "text": content,
                           "cache_control": {"type": "ephemeral"}}]
        return True
    if isinstance(content, list) and content:
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") in ("text", None):
                block["cache_control"] = {"type": "ephemeral"}
                return True
    return False


def apply_anthropic_cache(data: Dict[str, Any], model: str, approx_tokens: int) -> int:
    """Add cache breakpoints for Anthropic-served models. Returns how many were set.

    No-op for every other provider: they cache automatically, and an unexpected field is exactly
    the kind of thing that 400s and then hides behind a fallback."""
    if MODEL_PROVIDER.get(model) != "anthropic" or approx_tokens < ANTHROPIC_CACHE_MIN_TOKENS:
        return 0
    messages = data.get("messages") or []
    marked = 0
    for msg in messages:                       # system prompt: stable for the session
        if msg.get("role") == "system" and _mark_cache_control(msg):
            marked += 1
            break
    # ...and the end of the history, so each turn reuses every turn before it.
    for msg in reversed(messages[:-1]):
        if msg.get("role") in ("user", "assistant") and _mark_cache_control(msg):
            marked += 1
            break
    return marked


def parse_request(prompt: str) -> Tuple[str, Dict[str, Any]]:
    directives: Dict[str, Any] = {"tier": None, "allowed": None, "denied": set(), "boost": False}
    for m in TAG_RE.finditer(prompt):
        tag = m.group(1).upper()
        if tag == "BOOST":
            directives["boost"] = True
        elif tag in TAG_TIER and directives["tier"] is None:
            directives["tier"] = TAG_TIER[tag]
    for m in AVAIL_RE.finditer(prompt):
        provs = {x.strip().lower() for x in m.group(2).split(",")}
        if m.group(1):
            directives["denied"] |= provs
        else:
            directives["allowed"] = provs
    if not directives["boost"]:
        directives["boost"] = bool(BOOST_RE.search(prompt))
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


# Cost classes for latency-aware ordering: within a class, MEASURED latency decides
# (NIM vs zen-free genuinely compete); classes never leapfrog (free before flat-rate
# before per-token before per-credit).
# GO subscription (flat, cost class 1). Ordering WITHIN this class is by quota generosity
# (requests per 5h), not latency — GO models are never latency-probed, so tier config order is
# what decides: deepseek-flash 31,650 > mimo-lite 30,100 > qwen-plus 4,300 = hy3 4,300 >
# deepseek 3,450 > mimo 3,250 > minimax 3,200 > luna 2,050 > kimi 1,350 > qwen-max 950 >
# glm 880 >> grok 120 / kimi-k3 110 (the two RESTRICTED brains).
GO_ALIASES = {"go-glm", "go-deepseek", "go-kimi", "go-minimax",
                  "go-qwen-max", "go-qwen-plus", "go-mimo", "go-deepseek-flash",
                  "go-mimo-lite", "go-hy3", "go-luna",
                  "go-minimax27", "go-qwen36", "go-kimi26", "go-glm51",
                  "go-grok", "go-kimi-k3"}


def _cost_class(model: str) -> int:
    # Marginal-cost order. Everything with a sunk/flat cost ranks before real per-token spend:
    #   0 free           NIM + Mistral free + Zen free tier
    #   1 GO flat        opencode GO subscription — the most generous allowances, spend these first
    #   2 Codex flat     ChatGPT/Codex subscription
    #   3 Anthropic flat Claude Max subscription
    #   4 z.ai flat      GLM Coding Plan
    #   5 zen paid       real per-token money — the backstop
    #   6 copilot        per-request credit — last
    # None of the flat lanes is latency-probed (probing burns the quota they are held in reserve
    # for), so ORDER INSIDE a class is the tier's config order — which encodes quota generosity /
    # price, most generous first. See the tier definitions.
    p = MODEL_PROVIDER.get(model, "")
    if p == "nim" or p == "mistral" or model.startswith("free-"):
        return 0
    if model in GO_ALIASES:
        return 1
    if p == "codex":
        return 2
    if p == "anthropic":
        return 3
    if p == "zai":
        return 4
    if p == "zen":
        return 5
    return 6


def _stability_rank(model: str) -> int:
    """Within the free class (0), prefer the STABLE hosted free providers over load-variable NIM.
    Mistral + opencode-Zen-free have steady latency; NIM free is load-variable and flaps, which
    makes a session feel unreliable. So among free models: Mistral/zen-free first, NIM after.
    (Only breaks ties inside one cost class — cost_class still dominates, so this never lifts a
    free model above a cheaper one or vice-versa.)"""
    p = MODEL_PROVIDER.get(model, "")
    if p == "mistral" or model.startswith("free-"):
        return 0        # stable free — preferred
    if p == "nim":
        return 1        # NIM free is load-variable — used after Mistral/zen-free
    return 0            # non-free providers: no intra-class stability preference


def order_tier(tier: List[str], health: Dict[str, Any],
               native: Optional[Set[str]] = None, stability: bool = True,
               cache: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """Stable sort by (cost_class, native_rank, stability_rank, measured latency). Unprobed
    aliases keep their config position within the bucket (fail-open: no data -> no reordering).

    `native` marks the tier's OWN members. Anything outside it is a borrowed free model appended
    by free_fallback() — it ranks after the tier's own picks but, because cost_class dominates,
    still ahead of every flat/subscription model. That is the whole point: a reason-tier free
    model should answer a cheap prompt before we spend a subscription on it."""
    def key(item):
        idx, model = item
        h = health.get(model) or {}
        lat = h.get("latency_ms")
        return (_cost_class(model),
                0 if (native is None or model in native) else 1,
                _cache_rank(model, cache) if cache else 0,
                _stability_rank(model) if stability else 0,
                lat if lat is not None else 10**9, idx)
    return [m for _, m in sorted(enumerate(tier), key=lambda im: key(im))]


def free_fallback(tier: List[str], min_rank: int = 0) -> List[str]:
    """tier + every other FREE alias not already in it.

    Free capacity is lumpy: NIM flaps, Mistral's key can lapse, zen-free rate-limits. Without
    this, a cheap prompt whose four designated free models are all down falls straight to a
    flat subscription while other free models sit idle and healthy. Ordered zen-free -> Mistral
    -> NIM so the steadier hosts lead (order_tier's stability rank agrees, this just makes the
    tiebreak deterministic)."""
    seen = set(tier)
    # ONE-DIRECTIONAL: only borrow a model that already qualifies for work at least this
    # demanding. A capable free model may serve an easier tier (free is free); a cheap-tier model
    # must never be pulled UP into CODE/REASON/AGENT just because it was idle and fast.
    extra = [m for m in FREE_POOL if m not in seen and _capability_rank(m) >= min_rank]
    return tier + extra


def pick_model(tier: List[str], availability: Dict[str, bool],
               health: Dict[str, Any], latency_sort: bool = True,
               native: Optional[Set[str]] = None, stability: bool = True,
               cache: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[str]:
    # latency_sort=False for explicit-intent tiers (FRONTIER lists copilot first ON PURPOSE)
    candidates = order_tier(tier, health, native, stability, cache) if latency_sort else tier
    for model in candidates:
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
          health: Optional[Dict[str, Any]] = None,
          context_chars: Optional[int] = None,
          cache: Optional[Dict[str, Dict[str, Any]]] = None,
          sess: Optional[str] = None) -> Optional[str]:
    """`prompt` is the LAST USER MESSAGE — that is what carries the tags and the content markers.
    `context_chars` is the size of the WHOLE payload (system + history + tool results), which is
    what the model actually has to read. They are wildly different: a user typing "fix this" into
    an 80k-token session sends 2 tokens of prompt and 80k of context, and sizing on the prompt
    alone is how big-pickle ended up serving 81,802 tokens."""
    health = health or {}
    tier = directives.get("tier") or classify(prompt)
    tier_models = TIER_MAP.get(tier, GENERAL_TIER)
    # The scarce brains are allowed ONLY at the frontier tail. They are listed nowhere else, but
    # strip them defensively so a future tier edit can't quietly put a 110-req/5h model on routine
    # work — the failure mode is silent and expensive (GO saturation bills per-token Zen).
    if tier not in ("frontier", "orchestrator"):
        tier_models = [m for m in tier_models if m not in PREMIUM_ONLY]
    native = set(tier_models)
    # A very long prompt is an agentic conversation, not a question. Require real capability for
    # it regardless of which tier the heuristics picked: the audit trail showed 24% of >20k-token
    # requests being answered by CHEAP-tier models (big-pickle at 81,802 tokens, llama-3.3-70b at
    # 79,948). They returned *something*, so nothing errored and the quality loss was invisible.
    approx_tokens = (context_chars if context_chars is not None else len(prompt)) // CHARS_PER_TOKEN
    min_rank = TIER_CAPABILITY.get(tier, 0)
    # Cache preference applies ONLY to large payloads. On a short prompt there is nothing to
    # cache, so a model that never caches is not worse and must not be demoted for it.
    cache_pref = cache if approx_tokens >= LARGE_PROMPT_TOKENS else None
    if approx_tokens >= LARGE_PROMPT_TOKENS:
        min_rank = max(min_rank, TIER_CAPABILITY["code"])
        tier_models = [m for m in tier_models
                       if _capability_rank(m) >= min_rank or _cost_class(m) > 0]
    if tier in FREE_FALLBACK_TIERS:
        # Borrow other free aliases as a tail, but only ones already trusted at this level.
        # cost_class still dominates the sort, so they rank behind the tier's own free models
        # and ahead of any flat/paid one.
        tier_models = free_fallback(tier_models, min_rank=min_rank)
    # (The brains grok/kimi-k3 are no longer boost-escalated here: they are RESTRICTED_AUTO —
    # explicit-only — so [BOOST] on the orchestrator tier just raises thinking depth on the
    # high-quota capables, never spends a low-quota brain that could overflow GO -> paid Zen.)
    # CHEAP drops the stability tiebreak: everything in reach there is free, so one capable free
    # model is as good as another and raw speed is the only thing worth optimising. Elsewhere the
    # tiebreak still protects a session from load-variable NIM leading a long piece of work.
    # Stay on the conversation's current model when it is still valid: switching would discard the
    # provider's prompt cache and re-run prefill over the entire payload. An explicit tier tag has
    # already been honoured above (it changes `tier`, which breaks stickiness by design).
    sticky = _sticky_get(sess, tier, availability, health)
    if sticky and sticky in tier_models:
        verbose_logger.info("PriorityRouter: sticky %s -> %s", tier, sticky)
        return sticky

    chosen = pick_model(tier_models, availability, health,
                        latency_sort=tier not in ("frontier", "orchestrator"),   # explicit-intent tiers keep config (quality) order
                        native=native, stability=(tier != "cheap"), cache=cache_pref)
    if chosen:
        verbose_logger.info("PriorityRouter: tier=%s -> %s", tier, chosen)
        _sticky_put(sess, tier, chosen)
        return chosen
    for provider, models in PRIORITY_CHAIN.items():  # layer-5 cost-ordered chain
        for m in models:
            # The desperation walk must not reach a scarce brain on an everyday prompt: that is
            # precisely how kimi-k3 used to answer routine requests at 63k tokens. Frontier may.
            if m in PREMIUM_ONLY and tier not in ("frontier", "orchestrator"):
                continue
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
        # Size of the ENTIRE payload the model must read — system prompt, full history, tool
        # results — not just the turn the user typed. Routing capability off the last message
        # alone let a 2-token "fix this" in an 80k session be treated as a trivial prompt.
        context_chars = sum(len(extract_text(m.get("content", ""))) for m in messages)
        prompt, directives = "", {"tier": None, "allowed": None, "denied": set(), "boost": False}
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
        # The health probe must reach the alias it names. Rerouting an alias *because* it is
        # marked unhealthy makes that verdict permanent: the probe would be answered by a
        # substitute, the served-model check would fail, and the model would be re-marked
        # unhealthy on every audit -- it could never recover once the provider restored it.
        health_probe = bool((data.get("metadata") or {}).get("health_probe"))
        cache_profile = load_cache_profile()
        sess = session_key(messages)

        if requested in AUTO_MODELS:
            target = route(prompt, directives, availability, health, context_chars, cache_profile, sess)
            if target:
                data["model"] = target
                verbose_logger.info("PriorityRouter: auto -> %s", target)
        elif not health_probe:
            provider = MODEL_PROVIDER.get(requested)
            unhealthy = (provider is not None and not availability.get(provider, True)) \
                or (health.get(requested, {}).get("ok") is False)
            if unhealthy:
                target = route(prompt, directives, availability, health, context_chars, cache_profile, sess)
                if target:
                    verbose_logger.info("PriorityRouter: %s unavailable -> %s", requested, target)
                    data["model"] = target

        # Anthropic is the only lane that needs an explicit cache breakpoint; every other provider
        # caches on its own. Applied AFTER routing, so it keys off the model that will actually
        # serve the request rather than the one that was asked for.
        final_model_for_cache = data.get("model", "")
        n_cache = apply_anthropic_cache(data, final_model_for_cache, context_chars // CHARS_PER_TOKEN)
        if n_cache:
            verbose_logger.info("PriorityRouter: %d anthropic cache breakpoint(s) on %s",
                                n_cache, final_model_for_cache)

        # Inject thinking budget based on final model + tier
        final_model = data.get("model", "")
        tier = directives.get("tier") or classify(prompt)
        # Fusion's orchestrator/conductor/auditor can request HIGH thinking via metadata without
        # polluting prompt content with a [FRONTIER] tag the model would read.
        force_tier = (data.get("metadata") or {}).get("force_tier")
        if force_tier:
            tier = force_tier
        boost = directives.get("boost", False)
        think = _think_budget(final_model, tier, boost)
        level = "off"
        if think:
            level, tokens = think
            if final_model in ANT_THINKING or final_model in ZAI_THINKING:
                # Anthropic (and z.ai's Anthropic-compat endpoint) accept the thinking block
                # {"type":"enabled","budget_tokens":N} — verified live for z.ai GLM-5.2/4.6, which
                # return a `thinking` content block. A wrong shape 400s and the fallback chain then
                # silently serves a different provider, so this is invisible without fallbacks off.
                data["thinking"] = {"type": "enabled", "budget_tokens": tokens}
            elif final_model in NATIVE_REASONERS:
                # Native reasoners (free-* + Mistral magistral/medium): reason at max depth with
                # NO param (reasoning_effort only shrinks it; Mistral has no toggle at all). Inject
                # nothing — annotate high, let them run.
                pass
            elif final_model in GO_THINKING or final_model in COD_THINKING:
                # opencode GO validates its request body strictly and rejects NIM's parameter
                # names: 400 "Unsupported parameter(s): `thinking_budget_tokens`,
                # `enable_thinking`". It takes an OpenAI-style effort knob instead, with no token
                # budget. Sending the NIM pair here 400s every zen-* reason/code/agent/frontier
                # call; a fallback then answers and the failure never surfaces.
                # Codex is the same shape: codex_oauth_proxy.py reads reasoning_effort off the
                # request and emits Responses-API `reasoning: {effort}`. It also REJECTS
                # max_output_tokens, so no token budget is sent on this lane either.
                data.setdefault("extra_body", {})["reasoning_effort"] = level
            else:
                # NVIDIA NIM ALSO rejects enable_thinking/thinking_budget_tokens (same 400). GLM's
                # vLLM chat template toggles reasoning via chat_template_kwargs.enable_thinking —
                # verified live: it returns a reasoning_content trace, the plain params do not.
                # Boolean on/off; NIM exposes no token-budget knob, so `tokens` is unused here.
                data.setdefault("extra_body", {}).setdefault(
                    "chat_template_kwargs", {})["enable_thinking"] = True
            verbose_logger.info("PriorityRouter: think=%s budget=%d model=%s tier=%s",
                                level, tokens, final_model, tier)
            _write_think_state(final_model, tier, level)

        # Annotation state rides WITH the request (metadata is proxy-internal, never forwarded
        # to the provider). A module global / temp file would be read by whichever request
        # happens to finish next — under concurrency that stamps one request's model onto
        # another's response.
        data.setdefault("metadata", {})["llmr_ann"] = {
            "model": final_model, "tier": tier, "think": level,
        }
        return data


    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """Prepend [model · think:level · tier] to non-streaming responses.

        Reads per-request state from `data`; no annotation if it is absent, so a request
        this hook never saw in pre_call can never inherit another request's label.
        """
        # A paid model answering (incl. via LiteLLM fallback) -> kick a free-model re-audit so the
        # next prompt can return to free. response.model is the actual served alias (post-fallback).
        if _served_is_paid(getattr(response, "model", None)):
            _trigger_free_refresh()
        ann = (data.get("metadata") or {}).get("llmr_ann")
        if not ann or not ann.get("model"):
            return response
        annotation = _annotation(ann)
        try:
            choices = getattr(response, "choices", None)
            if choices and hasattr(choices[0], "message"):
                msg = choices[0].message
                if getattr(msg, "content", None):
                    msg.content = annotation + msg.content
        except Exception:
            pass
        return response

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response,
                                                      request_data):
        """Prepend the same banner to STREAMING responses. opencode streams, so without this the
        [model · think:level · tier] label only ever appeared in the non-streaming path and the
        user never saw which model routing chose. Inject it into the first content-bearing chunk
        (skip leading reasoning-only chunks so the banner leads the visible answer, not the CoT)."""
        ann = (request_data.get("metadata") or {}).get("llmr_ann")
        banner = _annotation(ann) if ann and ann.get("model") else None
        injected = False
        paid_checked = False
        async for chunk in response:
            if not paid_checked:                       # opencode streams: catch paid via chunk.model
                try:
                    if _served_is_paid(getattr(chunk, "model", None)):
                        _trigger_free_refresh()
                except Exception:
                    pass
                paid_checked = True
            if banner and not injected:
                try:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        delta.content = banner + delta.content
                        injected = True
                except Exception:
                    pass
            yield chunk


def _annotation(ann):
    return f"[{ann['model']} · think:{ann['think']} · {ann['tier']}]\n\n"


router_instance = PriorityRouter()
