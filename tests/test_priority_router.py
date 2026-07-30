import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
pr = importlib.import_module("priority_router")

ALL_OK = {"nim": True, "zen": True, "copilot": True}


def route(prompt, avail=None, health=None):
    cleaned, directives = pr.parse_request(prompt)
    availability = avail or dict(ALL_OK)
    return pr.route(cleaned, directives, availability, health or {})


def test_short_prompt_is_cheap_tier_stable_free_first():
    # Stability-first: within the free class, zen-free/Mistral lead NIM. Cheap tier -> zen-free.
    assert route("Say hi") == "zen-free-deepseek"


def test_default_general_is_stable_free():
    # >1200 chars so it lands in general tier, not cheap (<=300 tok = 1200 chars)
    long = "Explain the history and philosophy of stoicism " * 30
    assert route(long) == "zen-free-nemotron"


def test_code_marker_routes_to_stable_free():
    assert route("debug this:\n```python\nprint(1)\n```") == "zen-free-north"


def test_think_tag_routes_reason_tier():
    # Stable free reasoners (zen-free-*) lead the reason tier over NIM.
    assert route("[THINK] prove the halting problem is undecidable") == "zen-free-ling"


def test_frontier_tag_prefers_free_nim_when_healthy():
    # Cost-first frontier: free NIM (if healthy) before GO/Anthropic/Copilot.
    assert route("[FRONTIER] design a novel consensus protocol") == "nim-glm"

def test_frontier_falls_to_anthropic_then_copilot_when_free_down():
    # NIM + Zen down -> Anthropic Max, then Copilot. (avail without 'anthropic' key -> ant gated out.)
    r = route("[FRONTIER] design a novel consensus protocol",
              avail={"nim": False, "zen": False, "copilot": True})
    assert r == "cop-opus"

def test_frontier_gives_nim_thinking():
    # FRONTIER keeps explicit (quality) order — not stability-sorted — so nim-glm still leads and
    # its reasoning gets toggled via chat_template_kwargs.enable_thinking (verified live).
    # (REASON now leads with a stable-free native reasoner, so it's covered separately.)
    r = _hook({"model": "auto", "messages": [{"role": "user", "content": "[FRONTIER] hard task"}]})
    assert r["model"] == "nim-glm"
    assert r["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert "thinking_budget_tokens" not in r["extra_body"]   # NIM has no budget knob


def test_cheap_falls_to_zen_when_nim_down():
    assert route("Say hi", avail={"nim": False, "zen": True, "copilot": True}) == "zen-free-deepseek"


def test_reason_falls_to_copilot_when_nim_and_zen_down():
    r = route("[THINK] hard", avail={"nim": False, "zen": False, "copilot": True})
    assert r == "cop-opus"


def test_health_gate_skips_slow_nim_model():
    # With zen down, cheap tier is NIM-only; a slow nim-llama is skipped to the next NIM.
    # (Stability-first would otherwise pick a zen-free model regardless of nim health.)
    health = {"nim-llama": {"ok": False}}
    assert route("Say hi", avail={"nim": True, "zen": False, "copilot": True},
                 health=health) == "nim-deepseek-flash"


def test_health_gate_all_cheap_nim_slow_falls_to_zen():
    health = {"nim-llama": {"ok": False}, "nim-deepseek-flash": {"ok": False}}
    assert route("Say hi", health=health) == "zen-free-deepseek"


def test_unavailable_tag_masks_provider():
    cleaned, d = pr.parse_request("[UNAVAILABLE: nim] hello")
    eff = pr.effective_availability(d, base={"nim": True, "zen": True, "copilot": True})
    assert eff["nim"] is False and eff["zen"] is True


def test_novel_tag_passes_through_cleaned():
    cleaned, d = pr.parse_request("[NOVEL] discover something")
    assert "[NOVEL]" not in cleaned


# --- latency-aware ordering (2026-07-08) ---

def test_order_tier_free_class_sorts_by_latency():
    # zen-free faster than nim -> it wins within the free class
    health = {"nim-glm": {"ok": True, "latency_ms": 8000},
              "zen-free-nemotron": {"ok": True, "latency_ms": 1500}}
    tier = ["nim-glm", "zen-free-nemotron", "zen-glm", "cop-sonnet"]
    out = pr.order_tier(tier, health)
    assert out.index("zen-free-nemotron") < out.index("nim-glm")


def test_served_is_paid_matches_only_paid_aliases():
    # Paid detection keys on the zen-paid-* ALIAS (underlying model names are shared with GO twins).
    assert pr._served_is_paid("zen-paid-kimi3") is True
    assert pr._served_is_paid("zen-paid-glm") is True
    assert pr._served_is_paid("zen-kimi-k3") is False      # GO brain — flat, not paid
    assert pr._served_is_paid("kimi-k3") is False          # underlying id — ambiguous, never match
    assert pr._served_is_paid("nim-glm") is False
    assert pr._served_is_paid(None) is False


def test_trigger_free_refresh_writes_when_writable(tmp_path, monkeypatch):
    # Best-effort touch of the bind-mounted trigger file. Point it at a temp path and confirm it writes.
    trig = tmp_path / ".llmr-refresh-trigger"
    monkeypatch.setattr(pr, "_REFRESH_TRIGGER", trig)
    pr._trigger_free_refresh()
    assert trig.exists() and trig.read_text().strip().isdigit()


def test_stability_mistral_and_zenfree_beat_nim_within_free():
    # Within the free class, Mistral + zen-free are preferred over load-variable NIM even when NIM
    # is faster (stability > raw speed for a good session).
    health = {"nim-glm": {"ok": True, "latency_ms": 500},        # NIM fast...
              "mist-large": {"ok": True, "latency_ms": 3000},    # ...Mistral slower...
              "zen-free-north": {"ok": True, "latency_ms": 3000}}
    out = pr.order_tier(["nim-glm", "mist-large", "zen-free-north"], health)
    assert out.index("mist-large") < out.index("nim-glm")        # stable free first...
    assert out.index("zen-free-north") < out.index("nim-glm")    # ...despite NIM being faster
    assert pr._stability_rank("mist-large") == 0 and pr._stability_rank("nim-glm") == 1


def test_order_tier_go_never_before_free():
    # GO (class 1) stays behind ALL free (class 0) even when faster
    health = {"nim-glm": {"ok": True, "latency_ms": 8000},
              "zen-glm": {"ok": True, "latency_ms": 900}}
    tier = ["nim-glm", "zen-glm", "cop-sonnet"]
    out = pr.order_tier(tier, health)
    assert out.index("nim-glm") < out.index("zen-glm")
    assert out.index("zen-glm") < out.index("cop-sonnet")


def test_order_tier_unprobed_keeps_config_order():
    out = pr.order_tier(["nim-deepseek", "nim-gptoss", "cop-sonnet"], {})
    assert out == ["nim-deepseek", "nim-gptoss", "cop-sonnet"]


def test_pick_model_uses_latency_order():
    health = {"nim-glm": {"ok": True, "latency_ms": 9000},
              "zen-free-nemotron": {"ok": True, "latency_ms": 1200}}
    avail = {"nim": True, "zen": True, "copilot": True}
    got = pr.pick_model(["nim-glm", "zen-free-nemotron", "cop-sonnet"], avail, health)
    assert got == "zen-free-nemotron"


def test_unhealthy_zen_now_filtered_too():
    health = {"zen-free-nemotron": {"ok": False, "latency_ms": 11000}}
    avail = {"nim": False, "zen": True, "copilot": True}
    got = pr.pick_model(["zen-free-nemotron", "cop-sonnet"], avail, health)
    assert got == "cop-sonnet"


# --- adaptive thinking depth ---

def test_think_budget_nim_reason_returns_high():
    assert pr._think_budget("nim-glm", "reason", False) == ("high", 32768)

def test_think_budget_nim_frontier_returns_high():
    assert pr._think_budget("nim-minimax", "frontier", False) == ("high", 32768)

def test_think_budget_nim_minimax_agent_returns_high():
    assert pr._think_budget("nim-minimax", "agent", False) == ("high", 32768)

def test_think_budget_go_reason_returns_medium():
    assert pr._think_budget("zen-glm", "reason", False) == ("medium", 8192)

def test_think_budget_go_frontier_returns_high():
    assert pr._think_budget("zen-kimi", "frontier", False) == ("high", 16384)

def test_think_budget_ant_reason_returns_medium():
    assert pr._think_budget("ant-opus", "reason", False) == ("medium", 8192)

def test_think_budget_ant_frontier_returns_high():
    assert pr._think_budget("ant-fable", "frontier", False) == ("high", 16384)

def test_think_budget_ant_haiku_excluded():
    assert pr._think_budget("ant-haiku", "frontier", False) is None

def test_think_budget_cheap_tier_returns_none():
    assert pr._think_budget("nim-glm", "cheap", False) is None

def test_think_budget_general_tier_returns_none():
    assert pr._think_budget("zen-glm", "general", False) is None

def test_think_budget_non_thinking_model_returns_none():
    assert pr._think_budget("nim-deepseek", "reason", False) is None

def test_think_budget_boost_forces_high_on_go():
    assert pr._think_budget("zen-glm", "reason", True) == ("high", 16384)

def test_think_budget_boost_forces_high_on_ant():
    assert pr._think_budget("ant-sonnet", "agent", True) == ("high", 16384)


# --- BOOST escalation ---

def test_boost_tag_sets_boost_directive():
    _, d = pr.parse_request("[BOOST] redo this answer")
    assert d["boost"] is True

def test_boost_autodetect_redo():
    _, d = pr.parse_request("redo this, the answer was wrong")
    assert d["boost"] is True

def test_boost_autodetect_shallow():
    _, d = pr.parse_request("too shallow, try again")
    assert d["boost"] is True

def test_no_boost_on_normal_prompt():
    _, d = pr.parse_request("explain recursion")
    assert d.get("boost") is False

def test_boost_tag_stripped_from_cleaned():
    cleaned, _ = pr.parse_request("[BOOST] explain this better")
    assert "[BOOST]" not in cleaned
    assert "explain this better" in cleaned

def test_boost_composes_with_tier_tag():
    _, d = pr.parse_request("[REASON][BOOST] prove this theorem again")
    assert d["tier"] == "reason"
    assert d["boost"] is True


# --- thinking injection in async_pre_call_hook ---

import asyncio

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

def _hook(data, avail=None, health=None):
    orig_avail = pr.load_availability
    orig_health = pr.load_health
    pr.load_availability = lambda: avail or {"nim": True, "zen": True, "copilot": True, "anthropic": True}
    pr.load_health = lambda: health or {}   # no health overrides unless a test asks for them
    try:
        return _run(pr.router_instance.async_pre_call_hook(None, None, data, "completion"))
    finally:
        pr.load_availability = orig_avail
        pr.load_health = orig_health

def test_hook_injects_ant_thinking_param_at_frontier():
    data = {"model": "ant-opus", "messages": [{"role": "user", "content": "[FRONTIER] design a system"}]}
    result = _hook(data)
    # Anthropic's discriminator: 'enabled' | 'disabled' | 'adaptive'. Anything else 400s.
    assert result.get("thinking", {}).get("type") == "enabled"
    assert result["thinking"]["budget_tokens"] == 16384


def test_hook_ant_thinking_type_is_an_accepted_discriminator():
    """Regression: the type tag was 'thinking'; Anthropic 400s on it and the fallback chain
    then silently served another provider, so the bug presented as a success."""
    data = {"model": "ant-fable", "messages": [{"role": "user", "content": "[FRONTIER] hard"}]}
    result = _hook(data)
    assert result["thinking"]["type"] in {"enabled", "disabled", "adaptive"}


def test_hook_annotation_rides_on_request_not_global_state():
    """Regression: the annotation came from a process-global temp file, so a request with no
    thinking inherited the label of whichever request happened to write that file last."""
    d1 = _hook({"model": "ant-opus", "messages": [{"role": "user", "content": "[FRONTIER] hard"}]})
    d2 = _hook({"model": "ant-haiku", "messages": [{"role": "user", "content": "hi"}]})
    assert d1["metadata"]["llmr_ann"] == {"model": "ant-opus", "tier": "frontier", "think": "high"}
    # ant-haiku has no extended thinking -> reports its own model and think:off, never
    # ant-opus/high leaked from the preceding request.
    assert d2["metadata"]["llmr_ann"]["model"] == "ant-haiku"
    assert d2["metadata"]["llmr_ann"]["think"] == "off"
    assert "thinking" not in d2

def test_hook_injects_nim_glm_extra_body_at_reason():
    data = {"model": "nim-glm", "messages": [{"role": "user", "content": "[REASON] prove this"}]}
    result = _hook(data)
    # NIM toggles reasoning via chat_template_kwargs.enable_thinking; the plain enable_thinking/
    # thinking_budget_tokens pair 400s ("Unsupported parameter") and silently falls back.
    assert result["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert "enable_thinking" not in result["extra_body"]        # not at top level
    assert "thinking_budget_tokens" not in result["extra_body"]

def test_hook_injects_go_medium_at_reason():
    data = {"model": "zen-glm", "messages": [{"role": "user", "content": "[REASON] explain this trade-off"}]}
    result = _hook(data)
    # GO takes an effort knob, not a token budget: `thinking_budget_tokens` 400s on this provider.
    assert result.get("extra_body", {}).get("reasoning_effort") == "medium"

def test_hook_no_thinking_on_cheap_prompt_nim_glm():
    data = {"model": "nim-glm", "messages": [{"role": "user", "content": "say hi"}]}
    result = _hook(data)
    assert "thinking" not in result
    assert "extra_body" not in result

def test_hook_boost_forces_high_on_go():
    data = {"model": "zen-glm", "messages": [{"role": "user", "content": "[BOOST][REASON] redo this"}]}
    result = _hook(data)
    assert result["extra_body"]["reasoning_effort"] == "high"

def test_hook_force_tier_lifts_orchestrator_to_high():
    """Fusion's aggregator/conductor/auditor synthesize over drafts that are themselves only
    MEDIUM-thought. The synthesizer is what catches their errors, so it must think HIGH. It
    asks for that via metadata, not by prepending [FRONTIER] to a prompt the model would read."""
    drafts = "--- DRAFT 1 (nim-glm) ---\nsome draft text"
    without = _hook({"model": "zen-glm", "messages": [{"role": "user", "content": drafts}]})
    with_ft = _hook({"model": "zen-glm", "metadata": {"force_tier": "frontier"},
                     "messages": [{"role": "user", "content": drafts}]})
    assert with_ft["extra_body"]["reasoning_effort"] == "high"            # go/frontier -> HIGH
    assert with_ft["metadata"]["llmr_ann"]["think"] == "high"
    # ...and the same prompt without the override does NOT get HIGH, or the test proves nothing.
    assert without.get("extra_body", {}).get("reasoning_effort") != "high"

def test_hook_force_tier_is_not_forwarded_to_provider():
    """force_tier is a routing directive, not a completion parameter. metadata is proxy-internal,
    so it must never surface as a top-level key the upstream API would reject."""
    result = _hook({"model": "ant-opus", "metadata": {"force_tier": "frontier"},
                    "messages": [{"role": "user", "content": "audit this answer"}]})
    assert result["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "force_tier" not in result

# --- per-provider thinking parameter shape ----------------------------------------------------
# Each provider validates its request body and 400s on parameters it does not know. A 400 here is
# invisible in normal operation: the fallback chain answers with a different model and returns 200.
# These pin the shape that each provider actually accepts (verified live, fallbacks off).

def test_hook_go_thinking_uses_reasoning_effort_not_nim_params():
    """opencode GO 400s on `enable_thinking`/`thinking_budget_tokens`. It takes reasoning_effort."""
    r = _hook({"model": "zen-glm", "messages": [{"role": "user", "content": "[REASON] prove it"}]})
    eb = r.get("extra_body", {})
    assert eb.get("reasoning_effort") == "medium"          # go/reason -> MEDIUM
    assert "enable_thinking" not in eb
    assert "thinking_budget_tokens" not in eb

def test_hook_go_frontier_is_high_effort():
    r = _hook({"model": "zen-glm", "messages": [{"role": "user", "content": "[FRONTIER] hard"}]})
    assert r["extra_body"]["reasoning_effort"] == "high"

def test_hook_nim_uses_chat_template_kwargs_not_go_or_plain_params():
    """Each provider has its own dialect: ant=thinking block, GO=reasoning_effort,
    NIM=chat_template_kwargs.enable_thinking. Guards against cross-contamination."""
    r = _hook({"model": "nim-glm", "messages": [{"role": "user", "content": "[REASON] prove it"}]})
    eb = r.get("extra_body", {})
    assert eb["chat_template_kwargs"]["enable_thinking"] is True
    assert "reasoning_effort" not in eb                     # not GO's shape
    assert "enable_thinking" not in eb                      # not the rejected top-level param

def test_hook_ant_keeps_thinking_block():
    r = _hook({"model": "ant-opus", "messages": [{"role": "user", "content": "[FRONTIER] hard"}]})
    assert r["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "extra_body" not in r

def test_hook_no_model_ever_gets_two_providers_thinking_shapes():
    """A model must never carry MORE than one thinking dialect.
    ant + z.ai -> top-level thinking block; GO -> extra_body.reasoning_effort;
    NIM -> extra_body.chat_template_kwargs.enable_thinking; NATIVE (zen-free + Mistral) -> none."""
    from priority_router import ALL_THINKING, NATIVE_REASONERS
    for m in sorted(ALL_THINKING):
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[FRONTIER] hard"}]})
        eb = r.get("extra_body", {})
        shapes = [bool(r.get("thinking")),
                  "reasoning_effort" in eb,
                  "enable_thinking" in eb.get("chat_template_kwargs", {})]
        expected = 0 if m in NATIVE_REASONERS else 1   # native reasoners inject nothing
        assert sum(shapes) == expected, f"{m} got {sum(shapes)} dialects (want {expected}): {eb}"

# --- health probe must not be rerouted ---------------------------------------------------------

def test_health_probe_reaches_the_alias_it_names_even_when_marked_unhealthy():
    """The router reroutes any alias marked ok:false. If the health probe were rerouted, its
    served-model assertion would fail, the model would be re-marked unhealthy on every audit,
    and it could never recover once the provider restored it. Once dead, dead forever."""
    dead = {"nim-minimax": {"ok": False}}
    msg = [{"role": "user", "content": "Reply with the single word: OK"}]
    probe = _hook({"model": "nim-minimax", "metadata": {"health_probe": True}, "messages": msg},
                  health=dead)
    normal = _hook({"model": "nim-minimax", "messages": msg}, health=dead)
    assert probe["model"] == "nim-minimax", "health probe was rerouted; the verdict would be permanent"
    assert normal["model"] != "nim-minimax", "a normal request to a dead model should reroute"

def test_health_probe_flag_does_not_leak_to_provider():
    r = _hook({"model": "nim-glm", "metadata": {"health_probe": True},
               "messages": [{"role": "user", "content": "hi"}]})
    assert "health_probe" not in r


# --- streaming annotation (opencode streams; the banner must survive) ------------------------

class _Delta:
    def __init__(self, content): self.content = content
class _Choice:
    def __init__(self, content): self.delta = _Delta(content)
class _Chunk:
    def __init__(self, content): self.choices = [_Choice(content)]

async def _agen(chunks):
    for c in chunks: yield c

def _stream(chunks, ann):
    req = {"metadata": {"llmr_ann": ann}} if ann else {"metadata": {}}
    async def collect():
        out = []
        async for c in pr.router_instance.async_post_call_streaming_iterator_hook(None, _agen(chunks), req):
            out.append(c.choices[0].delta.content)
        return out
    return _run(collect())

def test_streaming_banner_prepends_to_first_content_chunk():
    ann = {"model": "nim-glm", "think": "high", "tier": "frontier"}
    out = _stream([_Chunk("Raft "), _Chunk("trades ")], ann)
    assert out[0] == "[nim-glm · think:high · frontier]\n\nRaft "   # banner on first chunk only
    assert out[1] == "trades "                                     # later chunks untouched

def test_streaming_banner_skips_leading_reasoning_only_chunks():
    # reasoning models stream reasoning_content first (delta.content is None); banner must wait
    # for the first VISIBLE content chunk, not lead the chain-of-thought.
    ann = {"model": "nim-glm", "think": "high", "tier": "frontier"}
    out = _stream([_Chunk(None), _Chunk(None), _Chunk("Answer")], ann)
    assert out[0] is None and out[1] is None
    assert out[2] == "[nim-glm · think:high · frontier]\n\nAnswer"

def test_streaming_no_annotation_when_state_absent():
    out = _stream([_Chunk("hi")], None)
    assert out[0] == "hi"      # no metadata -> no banner, never crashes


# --- free-tier native reasoners (zen-free-ling/mimo/north) -------------------------------------
# They reason at max depth with NO param (reasoning_effort only shrinks it), and they're free,
# so: always HIGH, inject nothing, annotate the banner high.

def test_free_reasoner_budget_is_high_any_tier_no_boost():
    for tier in ("reason", "code", "cheap", "general", "frontier"):
        assert pr._think_budget("zen-free-mimo", tier, False) == ("high", 0)

def test_free_reasoner_injects_no_param_but_annotates_high():
    r = _hook({"model": "zen-free-ling", "messages": [{"role": "user", "content": "[REASON] prove it"}]})
    assert "extra_body" not in r                        # no reasoning_effort, no chat_template_kwargs
    assert "thinking" not in r
    assert r["metadata"]["llmr_ann"]["think"] == "high"  # banner still shows high (native)

def test_free_reasoner_carries_exactly_one_or_zero_dialects():
    # covered by the ALL_THINKING guard, but pin it: free reasoners inject ZERO param dialects.
    for m in ("zen-free-ling", "zen-free-mimo", "zen-free-north"):
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[REASON] x"}]})
        eb = r.get("extra_body", {})
        assert not r.get("thinking") and "reasoning_effort" not in eb \
            and "enable_thinking" not in eb.get("chat_template_kwargs", {})

def test_free_reasoners_are_in_reason_tier():
    for m in ("zen-free-ling", "zen-free-mimo", "zen-free-north"):
        assert m in pr.REASON_TIER


# --- orchestrator/auditor tier (top-layer brain: grok-4.5 + kimi-k3 -> capable reasoners) -----

def test_orch_default_is_free_first_not_zen():
    # Orchestrator now leads with FREE (mist-medium/nim-glm), zen GO is LAST. With only nim/zen up,
    # the free NIM reasoner leads; zen-glm is only reached if all free/flat are down.
    assert route("[ORCH] synthesize these drafts") == "nim-glm"
    assert route("[AUDIT] check this answer") == "nim-glm"

def test_orch_boost_stays_free_not_brain_not_zen():
    # Even [BOOST][ORCH] on drift stays on FREE (brains restricted; zen GO is last-resort).
    av = {"nim": True, "zen": True, "copilot": True, "anthropic": True}
    assert route("[BOOST][ORCH] this verdict is wrong", avail=av) == "nim-glm"
    assert route("[BOOST][AUDIT] redo this", avail=av) == "nim-glm"

def test_orch_free_before_zen_full_avail():
    # With everything up, the stable free reasoner (mist-medium) leads — zen never touched.
    av = {"nim": True, "mistral": True, "zai": True, "zen": True, "anthropic": True, "copilot": True}
    assert route("[BOOST][ORCH] wrong, redo", avail=av) == "mist-medium"

def test_brains_never_auto_selected_in_any_tier_or_fallback():
    # RESTRICTED_AUTO: _model_ok gates them out of pick_model AND the layer-5 chain everywhere.
    av = {"nim": True, "mistral": True, "zai": True, "zen": True, "anthropic": True, "copilot": True}
    for m in ("zen-kimi-k3", "zen-grok"):
        assert not pr._model_ok(m, av, {})
    # Even with everything else health-gated down, route() must not return a brain.
    allbad = {t: {"ok": False} for t in pr.MODEL_PROVIDER if t not in ("zen-kimi-k3", "zen-grok")}
    got = route("[ORCH] x", avail=av, health=allbad)
    assert got not in ("zen-kimi-k3", "zen-grok")

def test_brain_still_reachable_by_EXPLICIT_request():
    # Restriction is AUTO-only: if opencode explicitly names the alias, it still routes there.
    r = _hook({"model": "zen-kimi-k3", "messages": [{"role": "user", "content": "hard verdict"}]})
    assert r["model"] == "zen-kimi-k3"                       # not rerouted away

def test_orchestrator_falls_to_zen_only_when_free_and_flat_down():
    # zen GO is the LAST orchestrator resort: reached only when free (nim) + flat (z.ai/anthropic)
    # are all unavailable. Here nim is down and mistral/zai/anthropic absent -> zen-glm.
    av = {"nim": False, "zen": True, "copilot": True}
    assert route("[ORCH] x", avail=av) == "zen-glm"

def test_orchestrator_free_nim_leads_when_flat_down():
    # brains + anthropic down, but free NIM up -> nim-glm leads (free before zen GO).
    av = {"nim": True, "zen": True, "copilot": True, "anthropic": False}
    h = {"zen-kimi-k3": {"ok": False}, "zen-grok": {"ok": False}}
    assert route("[ORCH] x", avail=av, health=h) == "nim-glm"

def test_orch_brains_get_high_thinking_via_go_shape():
    r = _hook({"model": "zen-kimi-k3", "messages": [{"role": "user", "content": "[ORCH] hard"}]})
    assert r["extra_body"]["reasoning_effort"] == "high"   # GO shape, orchestrator -> high
    r2 = _hook({"model": "zen-grok", "messages": [{"role": "user", "content": "[ORCH] hard"}]})
    assert r2["extra_body"]["reasoning_effort"] == "high"

def test_orch_brains_are_flatrate_cost_class():
    # GO flat-rate leads the flat band (class 1), not per-token/anthropic
    assert pr._cost_class("zen-kimi-k3") == 1 and pr._cost_class("zen-grok") == 1

def test_orch_l2_capables_in_tier():
    for m in ("zen-qwen-max", "zen-qwen-plus", "zen-deepseek", "zen-minimax", "zen-mimo"):
        assert m in pr.ORCHESTRATOR_TIER

def test_orch_l2_reasoners_get_high_others_off():
    # qwen/deepseek emit CoT (verified live) -> reasoning_effort high; minimax/mimo don't -> off
    for m in ("zen-qwen-max", "zen-qwen-plus", "zen-deepseek"):
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[ORCH] x"}]})
        assert r["extra_body"]["reasoning_effort"] == "high"
    for m in ("zen-mimo",):     # not in GO_THINKING -> no param, banner off
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[ORCH] x"}]})
        assert "extra_body" not in r
        assert r["metadata"]["llmr_ann"]["think"] == "off"


# --- research-pipeline role tags -------------------------------------------------------------

def test_role_tags_map_to_worker_tiers():
    # Stability-first: workers land on the stable-free leader of their tier (zen-free), not NIM.
    assert route("[SCOUT] list the files in this repo") == "zen-free-deepseek"  # cheap
    assert route("[ANALYST] interpret these benchmark numbers") == "zen-free-ling"  # reason
    assert route("[VERIFIER] check this claim against the source") == "zen-free-ling"
    assert route("[AUDITOR] does the code match the spec") == "zen-free-ling"

def test_role_tag_parsed_and_stripped():
    cleaned, d = pr.parse_request("[SCOUT] gather links")
    assert d["tier"] == "cheap" and "[SCOUT]" not in cleaned
    _, d2 = pr.parse_request("[VERIFIER] cross-check")
    assert d2["tier"] == "reason"

def test_orch_boss_distinct_from_worker_roles():
    # [ORCH] = orchestrator tier (free-first: nim-glm), role tags are the reason/cheap workers.
    assert route("[ORCH] final verdict") == "nim-glm"           # orchestrator tier (free leads, zen last)
    assert route("[VERIFIER] check") == "zen-free-ling"          # reason tier (stable-free leads)


# --- paid per-token overflow ($20 balance) ---------------------------------------------------

PAID = ["zen-paid-minimax", "zen-paid-qwen-plus", "zen-paid-glm", "zen-paid-luna",
        "zen-paid-grok", "zen-paid-qwen-max", "zen-paid-kimi3", "zen-paid-terra", "zen-paid-sol"]

def test_paid_models_are_pertoken_cost_class_4():
    # per-token (class 4) sits BELOW every flat sub (z.ai 1, GO 2, Anthropic Max 3) and above only
    # copilot -> all flat-rate capacity is drained before the $20 balance is touched.
    for m in PAID:
        assert pr._cost_class(m) == 4, f"{m} cost class {pr._cost_class(m)} != 4"

def test_paid_models_never_in_a_default_tier():
    # reached only via fallback map / boost, never auto-selected -> protects the balance.
    for tier in pr.TIER_MAP.values():
        for m in PAID:
            assert m not in tier, f"{m} leaked into a default tier"

def test_paid_reasoners_use_reasoning_effort_minimax_excluded():
    r = _hook({"model": "zen-paid-glm", "messages": [{"role": "user", "content": "[REASON] x"}]})
    assert r["extra_body"]["reasoning_effort"] == "medium"      # go/reason
    r2 = _hook({"model": "zen-paid-minimax", "messages": [{"role": "user", "content": "[REASON] x"}]})
    assert "extra_body" not in r2                               # minimax: no CoT -> no param


# --- Mistral (free, class 0) + z.ai GLM Coding Plan (flat, class 1) ---

ALL_OK2 = {"nim": True, "mistral": True, "zai": True, "zen": True, "anthropic": True, "copilot": True}


def test_provider_map_knows_mistral_and_zai():
    assert pr.MODEL_PROVIDER["mist-large"] == "mistral"
    assert pr.MODEL_PROVIDER["mist-magistral"] == "mistral"
    assert pr.MODEL_PROVIDER["zai-52"] == "zai"
    assert pr.MODEL_PROVIDER["zai-flash"] == "zai"


def test_cost_class_cascade_free_go_zai_flat_paid_copilot():
    # The whole ordering intent in one assertion: strictly increasing marginal cost.
    assert pr._cost_class("mist-large") == 0        # Mistral free
    assert pr._cost_class("nim-glm") == 0           # NIM free
    assert pr._cost_class("zen-glm") == 1           # GO flat   (BEFORE z.ai)
    assert pr._cost_class("zai-52") == 2            # z.ai flat
    assert pr._cost_class("ant-sonnet") == 3        # Anthropic Max flat (BEFORE zen paid)
    assert pr._cost_class("cod-sol") == 3           # Codex flat SHARES the Anthropic band
    assert pr._cost_class("zen-paid-glm") == 4      # zen per-token
    assert pr._cost_class("cop-opus") == 5          # copilot per-request


def test_go_flat_ranks_before_zai_in_reason_tier():
    # Sorted reason tier: free (mist/nim/zen-free) then GO zen-glm(1) then z.ai(2) then ant(3).
    ordered = pr.order_tier(pr.REASON_TIER, {})
    assert ordered.index("zen-glm") < ordered.index("zai-52")
    assert ordered.index("mist-medium") < ordered.index("zen-glm")  # free before flat
    assert ordered.index("ant-sonnet") > ordered.index("zai-52")    # anth after both flats


def test_anthropic_flat_ranks_before_zen_paid():
    # Both appear in the frontier-ish fallback space; anth (sunk) must precede zen paid ($).
    tier = ["zen-paid-glm", "ant-sonnet"]
    assert pr.order_tier(tier, {}) == ["ant-sonnet", "zen-paid-glm"]


def test_mistral_reasoners_are_native_no_param():
    # magistral / mistral-medium reason inline with NO param (verified live). Inject nothing.
    for m in ("mist-medium", "mist-magistral"):
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[REASON] prove it"}]},
                  avail=ALL_OK2)
        assert r["model"] == m
        assert "thinking" not in r
        assert "extra_body" not in r
        assert r["metadata"]["llmr_ann"]["think"] == "high"   # annotated high, native depth


def test_mistral_non_reasoners_get_no_thinking():
    for m in ("mist-large", "mist-codestral"):
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[CODE] write a parser"}]},
                  avail=ALL_OK2)
        assert "thinking" not in r
        assert "extra_body" not in r


def test_zai_reasoners_take_anthropic_thinking_block():
    # z.ai's Anthropic-compat endpoint returns a `thinking` block for this shape (verified live).
    r = _hook({"model": "zai-52", "messages": [{"role": "user", "content": "[REASON] hard"}]},
              avail=ALL_OK2)
    assert r["thinking"]["type"] == "enabled"
    assert r["thinking"]["budget_tokens"] == 8192          # ant-class, reason -> medium budget
    r2 = _hook({"model": "zai-52", "messages": [{"role": "user", "content": "[FRONTIER] hard"}]},
               avail=ALL_OK2)
    assert r2["thinking"]["budget_tokens"] == 16384        # frontier -> high


def test_zai_turbo_non_reasoner_no_thinking():
    r = _hook({"model": "zai-turbo", "messages": [{"role": "user", "content": "[CODE] refactor"}]},
              avail=ALL_OK2)
    assert "thinking" not in r
    assert "extra_body" not in r


def test_nim_step_is_free_native_reasoner_no_param():
    # step-3.7-flash reasons natively on NIM (no chat_template_kwargs toggle) -> inject nothing.
    assert pr._cost_class("nim-step") == 0                    # NIM free
    assert pr.MODEL_PROVIDER["nim-step"] == "nim"
    assert "nim-step" in pr.NATIVE_REASONERS
    assert "nim-step" not in pr.NIM_THINKING                  # NOT the enable_thinking path
    r = _hook({"model": "nim-step", "messages": [{"role": "user", "content": "[REASON] prove it"}]},
              avail={"nim": True, "zen": True, "copilot": True})
    assert "thinking" not in r
    assert "extra_body" not in r                              # native — no enable_thinking, no effort
    assert r["metadata"]["llmr_ann"]["think"] == "high"       # annotated high


def test_go_flat_beats_zai_when_free_reasoners_health_gated():
    # Cost cascade at the routing level: with EVERY free model gated off, the reason tier serves
    # GO zen-glm (class 1) before z.ai (class 2). NIM/Mistral are also masked by availability.
    # All of FREE_POOL must go down, not just the reason-tier ones: since free_fallback() the
    # tier borrows any other healthy free model first — which is the point of that feature.
    health = {m: {"ok": False} for m in pr.FREE_POOL}
    r = route("[REASON] prove the halting problem",
              avail={"mistral": False, "nim": False, "zai": True, "zen": True,
                     "anthropic": True, "copilot": True},
              health=health)
    assert r == "zen-glm"       # GO flat beats z.ai


# --- Codex / ChatGPT subscription (flat, class 3 alongside Anthropic) -------------------------

def test_codex_shares_the_anthropic_flat_band():
    # Both are sunk-cost subscriptions: neither should always drain before the other.
    for m in ("cod-sol", "cod-terra", "cod-luna"):
        assert pr._cost_class(m) == pr._cost_class("ant-sonnet") == 3


def test_codex_ranks_after_free_and_both_flats_but_before_zen_paid():
    tier = ["zen-paid-glm", "cod-luna", "zai-52", "zen-glm", "nim-glm"]
    assert pr.order_tier(tier, {}) == ["nim-glm", "zen-glm", "zai-52", "cod-luna", "zen-paid-glm"]


def test_codex_reasoner_gets_effort_knob_not_thinking_block():
    # cod-sol takes reasoning_effort (proxy -> Responses `reasoning.effort`). Sending the
    # Anthropic thinking block or NIM's enable_thinking here 400s and a fallback hides it.
    r = _hook({"model": "cod-sol", "messages": [{"role": "user", "content": "[REASON] prove it"}]})
    assert r["extra_body"]["reasoning_effort"] in ("medium", "high")
    assert "thinking" not in r
    assert "chat_template_kwargs" not in r.get("extra_body", {})


def test_codex_non_reasoners_get_no_thinking_param():
    for m in ("cod-luna", "cod-terra"):
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[REASON] x"}]})
        assert "thinking" not in r
        assert "reasoning_effort" not in r.get("extra_body", {})


# --- free-capacity borrowing across tiers ------------------------------------------------------
# Free capacity is lumpy (NIM flaps, Mistral keys lapse, zen-free rate-limits). A tier whose own
# free models are all down must borrow another tier's free model before spending a subscription.

def test_tier_native_free_still_wins_when_healthy():
    # Borrowing must not disturb the normal case: the tier's own free pick still leads.
    assert route("hi") == "zen-free-deepseek"


def test_cheap_borrows_a_free_model_before_any_subscription():
    down = {m: {"ok": False} for m in pr.CHEAP_TIER if pr._cost_class(m) == 0}
    got = route("hi", health=down)
    assert pr._cost_class(got) == 0, f"{got} is not free — a subscription was spent too early"
    assert got not in pr.CHEAP_TIER          # genuinely borrowed from another tier


def test_subscription_only_once_every_free_model_is_down():
    down = {m: {"ok": False} for m in pr.FREE_POOL}
    got = route("hi", health=down)
    assert pr._cost_class(got) > 0


def test_frontier_and_orchestrator_never_borrow_free_models():
    # Explicit quality-intent tiers: a tiny free model must not answer a frontier prompt just
    # because it is free. They fall through to the layer-5 cost chain instead.
    for t in ("frontier", "orchestrator"):
        got = pr.route("hard", {"tier": t}, ALL_OK, {})
        assert got in pr.TIER_MAP[t], f"{t} leaked a borrowed model: {got}"


def test_borrowing_never_breaks_the_cost_cascade():
    extended = pr.free_fallback(pr.CHEAP_TIER)
    classes = [pr._cost_class(m) for m in pr.order_tier(extended, {}, native=set(pr.CHEAP_TIER))]
    assert classes == sorted(classes), "cost classes must never leapfrog"


def test_free_pool_contains_only_free_models():
    for m in pr.FREE_POOL:
        assert pr._cost_class(m) == 0, f"{m} is in FREE_POOL but is not cost class 0"


def test_health_probes_only_free_models():
    """Ranking flat/subscription models by quota weight works ONLY because they are never
    latency-probed: with no latency_ms they all tie at 10**9 and order_tier falls through to
    config order (= quota order). Add a flat alias to nim_health.sh and that silently reverts
    to latency ordering — no error, no failing test, just a subscription draining in the wrong
    order (and the probe burning the quota it is held in reserve for). This is that guard."""
    import re
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "scripts" / "nim_health.sh").read_text()
    assigns = dict(re.findall(r'^(\w+)="([^"]*)"', script, re.M))
    probed = []
    for tok in assigns.get("ALL_ALIASES", "").split():
        probed.extend(assigns.get(tok.lstrip("$"), "").split() if tok.startswith("$") else [tok])

    assert probed, "could not parse ALL_ALIASES out of scripts/nim_health.sh"
    for alias in probed:
        assert pr._cost_class(alias) == 0, (
            f"{alias} is latency-probed by nim_health.sh but is cost class "
            f"{pr._cost_class(alias)}, not free"
        )
