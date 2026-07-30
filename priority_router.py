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

import json as _json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "mistral": ["mist-large", "mist-codestral", "mist-medium", "mist-magistral"],
    # z.ai GLM Coding Plan (flat-rate). Served ONLY via the Anthropic-compat endpoint
    # (/api/anthropic) — the OpenAI /paas/v4 lane 429s "insufficient balance". Wired as LiteLLM
    # anthropic/ provider; reasoners take the Anthropic thinking block. Flat -> class 1 (before GO).
    "zai": ["zai-turbo", "zai-52", "zai-51", "zai-flash"],
    "zen": [
        "zen-free-deepseek", "zen-free-nemotron", "zen-free-pickle",
        "zen-free-ling", "zen-free-laguna", "zen-free-mimo", "zen-free-north",
        "zen-glm", "zen-deepseek", "zen-gpt", "zen-fable", "zen-opus",
        # GO subscription (flat-rate /zen/go/v1): capable models, no per-token cost
        "zen-kimi", "zen-minimax", "zen-qwen-max", "zen-qwen-plus", "zen-mimo", "zen-deepseek-flash",
        "zen-grok", "zen-kimi-k3",              # top-layer brains (low quota, orchestrator-only)
        # PAID per-token overflow ($20 balance) — class 2 (not in ZEN_GO_ALIASES), fallback/boost only
        "zen-paid-minimax", "zen-paid-qwen-plus", "zen-paid-glm",
        "zen-paid-luna", "zen-paid-grok", "zen-paid-qwen-max", "zen-paid-kimi3",
        "zen-paid-terra", "zen-paid-sol",
    ],
    "copilot": [
        "cop-opus", "cop-sonnet", "cop-gpt", "cop-codex", "cop-gemini",
        "cop-haiku", "cop-mini",
    ],
    # Claude Max subscription via OAuth (flat-rate, rate-limited by Anthropic)
    "anthropic": ["ant-fable", "ant-opus", "ant-sonnet", "ant-haiku"],
}

# Tiers: NIM first (health-gated) -> Zen (free before paid) -> ant-* -> Copilot tail.
# Copilot never appears in cheap/general — no per-request credit on routine work.
# NOTE: GPT-family Copilot models (cop-gpt/cop-codex/cop-mini) fail via litellm's
# Copilot path (GPT needs the /responses API). Use Claude/Gemini Copilot + Zen's
# GPT (zen-gpt) for GPT needs instead. GPT-via-Copilot kept in config but unused.
CHEAP_TIER    = ["nim-llama", "nim-deepseek-flash", "zen-free-deepseek", "zen-free-pickle", "mist-codestral", "zai-flash", "ant-haiku", "cop-haiku"]  # zen-free-ling in REASON (native reasoner), not cheap
GENERAL_TIER  = ["nim-glm", "nim-inkling", "nim-step", "zen-free-nemotron", "zen-free-laguna", "mist-large", "zai-turbo", "zai-flash", "zen-glm", "ant-sonnet", "cop-sonnet"]  # zen-free-laguna: new free worker (health-gated; type unconfirmed, was rate-limited at wiring)
CODE_TIER     = ["nim-deepseek", "nim-gptoss", "nim-step", "zen-free-north", "mist-large", "mist-codestral", "zai-turbo", "zen-kimi", "zen-deepseek", "ant-sonnet", "cop-sonnet"]  # zen-free-north/nim-step = free code reasoners; mist-codestral = free code specialist
# REASON leads with NIM thinking models (nim-glm/kimi/minimax get HIGH budget at this tier —
# Phase 1.6). Without them here, "NIM -> HIGH on reason" was unreachable: the tier held only
# non-thinking NIM models, so the budget table never applied. zen-gpt dropped (dead: 401).
REASON_TIER   = ["nim-glm", "nim-minimax", "nim-step", "zen-free-ling", "zen-free-mimo", "zen-free-north", "mist-medium", "mist-magistral", "nim-inkling", "nim-nemotron", "zai-52", "zai-51", "zen-glm", "zen-qwen-max", "zen-free-nemotron", "ant-sonnet", "cop-opus"]  # zen-free-*/mist-*/nim-step = free native reasoners; zai-5x = flat GLM reasoners
AGENT_TIER    = ["nim-glm", "nim-minimax", "nim-step", "mist-large", "mist-medium", "zai-turbo", "zen-glm", "zen-minimax", "ant-sonnet", "cop-sonnet"]
# FRONTIER = cost-first order, every member at HIGH thinking: free NIM (if healthy) -> GO
# flat-rate -> Anthropic Max -> Copilot. Health-gating handles "if available". NIM thinking
# models lead so a frontier task can run on free GLM 5.2 + 32k thinking before touching paid.
# zen-gpt dropped (dead: 401). Config order = intent (frontier is not latency-sorted).
FRONTIER_TIER = ["nim-glm", "nim-minimax", "mist-medium", "zai-52", "zen-glm", "zen-qwen-max", "ant-opus", "ant-fable", "ant-sonnet", "cop-opus", "cop-sonnet", "cop-gemini"]  # mist-medium(free)/zai-52(flat) reasoners between free NIM and GO
# ORCHESTRATOR / AUDITOR = checks scout/agent output and gives a verdict — LOW token, run often.
# QUALITY-first order (NOT latency-sorted): capable, high-quota reliables (zen-glm, ant-sonnet/opus
# Max flat-rate, GO reasoners, nim-glm free) carry every verdict. The near-frontier brains
# (grok/kimi-k3) are DELIBERATELY absent — they're RESTRICTED_AUTO (explicit-only), because their
# low GO quota + GO->paid-Zen overflow on saturation make auto-selection a cost trap. Every member
# runs HIGH thinking.
# Orchestrator brains (grok-4.5, kimi-k3): near-frontier but LOW GO quota, and a GO 429 when the
# plan is saturated overflows to PER-TOKEN Zen billing on the same account (both keys charge). So
# they are RESTRICTED from ALL auto-routing — never selected by any tier, boost, or fallback.
# Reachable ONLY by an explicit alias request (pick "zen-kimi-k3" yourself). Enforced in _model_ok,
# which is the single gate for every auto path (pick_model + the layer-5 chain).
RESTRICTED_AUTO = frozenset({"zen-kimi-k3", "zen-grok"})
# Cost-safe order (explicit, NOT latency-sorted): FREE reasoners -> z.ai flat -> Anthropic Max flat
# -> zen GO LAST. Even [BOOST][ORCH] stays on free/flat and only reaches zen GO if all of those are
# down — because a GO 429 when saturated overflows to per-token Zen. zen paid is never in the tier
# (reached only via a GO model's own fallback if literally everything else is unavailable). Brains
# excluded (RESTRICTED_AUTO).
ORCHESTRATOR_TIER = ["mist-medium", "nim-glm",                         # free reasoners
                     "zai-52",                                          # z.ai flat
                     "ant-sonnet", "ant-opus",                          # Anthropic Max flat
                     "zen-glm", "zen-qwen-max", "zen-qwen-plus", "zen-deepseek", "zen-minimax", "zen-mimo"]  # zen GO LAST

# --- Thinking-capable model sets and budget table ---
NIM_THINKING = frozenset({"nim-glm", "nim-minimax"})
GO_THINKING  = frozenset({"zen-glm", "zen-kimi", "zen-minimax", "zen-grok", "zen-kimi-k3",
                          "zen-qwen-max", "zen-qwen-plus", "zen-deepseek",  # verified live: these emit CoT
                          # PAID overflow reasoners (same underlying models -> reasoning_effort;
                          # gpt-5.6 luna/terra/sol take reasoning_effort too. minimax excluded: no CoT).
                          "zen-paid-glm", "zen-paid-qwen-plus", "zen-paid-qwen-max", "zen-paid-grok",
                          "zen-paid-kimi3", "zen-paid-luna", "zen-paid-terra", "zen-paid-sol"})
ANT_THINKING = frozenset({"ant-opus", "ant-fable", "ant-sonnet"})  # ant-haiku excluded
# opencode Zen FREE-tier native reasoners. Verified live: they emit a reasoning_content trace by
# DEFAULT with no param, and reasoning_effort only REDUCES that trace. Free, so never a reason to
# hold back — always HIGH, and we inject NO param so their native max-depth reasoning is preserved
# (a param here would shrink it for zero benefit). They exist only for the annotation + tiering.
FREE_REASONERS = frozenset({"zen-free-ling", "zen-free-mimo", "zen-free-north"})
# Mistral reasoners (magistral / mistral-medium hybrid): reason natively on api.mistral.ai with NO
# param — verified live, they 200 with reasoning inline (no separate reasoning_content field to
# toggle). Treated like FREE_REASONERS: inject nothing, annotate high.
MIST_REASONERS = frozenset({"mist-medium", "mist-magistral"})
# NIM native reasoners: step-3.7-flash emits a reasoning_content trace by DEFAULT (verified live:
# 1505 chars with no param; enable_thinking barely changes it). So it is NOT in NIM_THINKING —
# it needs no chat_template_kwargs toggle; treat it like the other native reasoners (inject none).
NIM_NATIVE = frozenset({"nim-step"})
NATIVE_REASONERS = FREE_REASONERS | MIST_REASONERS | NIM_NATIVE
# z.ai GLM reasoners on the Anthropic-compat endpoint: verified live they accept the Anthropic
# thinking block {"type":"enabled","budget_tokens":N} and return a `thinking` content block. Same
# injection + budget table as ANT_THINKING (think-class "ant"), even though cost-wise they're flat.
ZAI_THINKING  = frozenset({"zai-52", "zai-51"})
ALL_THINKING  = NIM_THINKING | GO_THINKING | ANT_THINKING | ZAI_THINKING | NATIVE_REASONERS

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
    # so a paid call is exactly a zen-paid-* alias. Underlying model names are shared with GO twins,
    # so only the alias distinguishes paid from flat — never match on the underlying id.
    return isinstance(model, str) and model.startswith("zen-paid-")


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
    if model in RESTRICTED_AUTO:
        return False  # brains (grok/kimi-k3): explicit-only, never auto-routed (quota + GO->paid overflow)
    if not availability.get(MODEL_PROVIDER.get(model, ""), False):
        return False
    h = health.get(model)
    if h is not None and h.get("ok") is False:
        return False  # NIM model flagged slow/dead by the audit
    return True


# Cost classes for latency-aware ordering: within a class, MEASURED latency decides
# (NIM vs zen-free genuinely compete); classes never leapfrog (free before flat-rate
# before per-token before per-credit).
ZEN_GO_ALIASES = {"zen-glm", "zen-deepseek", "zen-kimi", "zen-minimax",
                  "zen-qwen-max", "zen-qwen-plus", "zen-mimo", "zen-deepseek-flash",
                  "zen-grok", "zen-kimi-k3"}


def _cost_class(model: str) -> int:
    # Marginal-cost order (all flat/sunk subs rank before real-per-token spend):
    #   0 free  ->  1 z.ai flat  ->  2 GO flat  ->  3 Anthropic Max flat  ->  4 zen paid  ->  5 copilot
    # z.ai flat leads the paid-flat band (user tests its limits first); Anthropic Max (sunk) is used
    # before spending zen per-token credits; zen paid is the real-money backstop; copilot last.
    p = MODEL_PROVIDER.get(model, "")
    if p == "nim" or p == "mistral" or model.startswith("zen-free-"):
        return 0                       # free (NIM + Mistral free tier + zen free-tier)
    if p == "zai":
        return 1                       # z.ai GLM Coding Plan (flat-rate)
    if model in ZEN_GO_ALIASES:
        return 2                       # opencode GO subscription flat-rate
    if p == "anthropic":
        return 3                       # Claude Max OAuth (flat-rate; opus gated to invoke tiers)
    if p == "zen":
        return 4                       # zen per-token paid credits ($20 backstop)
    return 5                           # copilot per-request credit


def _stability_rank(model: str) -> int:
    """Within the free class (0), prefer the STABLE hosted free providers over load-variable NIM.
    Mistral + opencode-Zen-free have steady latency; NIM free is load-variable and flaps, which
    makes a session feel unreliable. So among free models: Mistral/zen-free first, NIM after.
    (Only breaks ties inside one cost class — cost_class still dominates, so this never lifts a
    free model above a cheaper one or vice-versa.)"""
    p = MODEL_PROVIDER.get(model, "")
    if p == "mistral" or model.startswith("zen-free-"):
        return 0        # stable free — preferred
    if p == "nim":
        return 1        # NIM free is load-variable — used after Mistral/zen-free
    return 0            # non-free providers: no intra-class stability preference


def order_tier(tier: List[str], health: Dict[str, Any]) -> List[str]:
    """Stable sort by (cost_class, stability_rank, measured latency). Unprobed aliases keep their
    config position within the (class, stability) bucket (fail-open: no data -> no reordering)."""
    def key(item):
        idx, model = item
        h = health.get(model) or {}
        lat = h.get("latency_ms")
        return (_cost_class(model), _stability_rank(model),
                lat if lat is not None else 10**9, idx)
    return [m for _, m in sorted(enumerate(tier), key=lambda im: key(im))]


def pick_model(tier: List[str], availability: Dict[str, bool],
               health: Dict[str, Any], latency_sort: bool = True) -> Optional[str]:
    # latency_sort=False for explicit-intent tiers (FRONTIER lists copilot first ON PURPOSE)
    candidates = order_tier(tier, health) if latency_sort else tier
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
          health: Optional[Dict[str, Any]] = None) -> Optional[str]:
    health = health or {}
    tier = directives.get("tier") or classify(prompt)
    tier_models = TIER_MAP.get(tier, GENERAL_TIER)
    # (The brains grok/kimi-k3 are no longer boost-escalated here: they are RESTRICTED_AUTO —
    # explicit-only — so [BOOST] on the orchestrator tier just raises thinking depth on the
    # high-quota capables, never spends a low-quota brain that could overflow GO -> paid Zen.)
    chosen = pick_model(tier_models, availability, health,
                        latency_sort=tier not in ("frontier", "orchestrator"))   # explicit-intent tiers keep config (quality) order
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

        if requested in AUTO_MODELS:
            target = route(prompt, directives, availability, health)
            if target:
                data["model"] = target
                verbose_logger.info("PriorityRouter: auto -> %s", target)
        elif not health_probe:
            provider = MODEL_PROVIDER.get(requested)
            unhealthy = (provider is not None and not availability.get(provider, True)) \
                or (health.get(requested, {}).get("ok") is False)
            if unhealthy:
                target = route(prompt, directives, availability, health)
                if target:
                    verbose_logger.info("PriorityRouter: %s unavailable -> %s", requested, target)
                    data["model"] = target

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
                # Native reasoners (zen-free-* + Mistral magistral/medium): reason at max depth with
                # NO param (reasoning_effort only shrinks it; Mistral has no toggle at all). Inject
                # nothing — annotate high, let them run.
                pass
            elif final_model in GO_THINKING:
                # opencode GO validates its request body strictly and rejects NIM's parameter
                # names: 400 "Unsupported parameter(s): `thinking_budget_tokens`,
                # `enable_thinking`". It takes an OpenAI-style effort knob instead, with no token
                # budget. Sending the NIM pair here 400s every zen-* reason/code/agent/frontier
                # call; a fallback then answers and the failure never surfaces.
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
