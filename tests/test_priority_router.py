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


def test_short_prompt_is_cheap_tier_and_free():
    # CHEAP keeps the stability tiebreak like every other tier, so a stable-free host leads even
    # here. With no health data the stable group ties and config order decides inside it.
    got = route("Say hi")
    assert pr._cost_class(got) == 0, f"cheap picked a non-free model: {got}"
    assert pr._stability_rank(got) == 0, f"cheap led with load-variable {got}"


def test_cheap_prefers_a_stable_free_host_even_when_nim_is_faster():
    # The reliability rule that replaced "fastest free wins" in CHEAP. Measured live: NIM served
    # 529s and 8-11s stalls while Mistral/zen-free held ~1-2.4s. A trivial prompt answered a
    # second sooner is worth nothing against one that stalls, so a fast NIM does NOT jump the
    # queue over a slower stable host.
    fast_nim = {"nim-llama": {"ok": True, "latency_ms": 600},
                "free-deepseek": {"ok": True, "latency_ms": 2200},
                "free-pickle": {"ok": True, "latency_ms": 2100}}
    got = route("Say hi", health=fast_nim)
    assert got != "nim-llama"
    assert pr._stability_rank(got) == 0

    # Latency is still the next key, so it decides WITHIN the stable group.
    fast_zen = {"nim-llama": {"ok": True, "latency_ms": 3000},
                "nim-deepseek-flash": {"ok": True, "latency_ms": 3200},
                "free-deepseek": {"ok": True, "latency_ms": 700}}
    assert route("Say hi", health=fast_zen) == "free-deepseek"


def test_reason_keeps_the_stability_tiebreak():
    # Long/expensive work still prefers steady hosts over load-variable NIM even when NIM is
    # momentarily faster — a flappy model must not lead a multi-minute task.
    h = {"nim-glm": {"ok": True, "latency_ms": 500},
         "free-ling": {"ok": True, "latency_ms": 3000}}
    assert route("[REASON] prove it", health=h) == "free-deepseek"


def test_default_general_is_stable_free():
    # >1200 chars so it lands in general tier, not cheap (<=300 tok = 1200 chars)
    long = "Explain the history and philosophy of stoicism " * 30
    assert route(long) == "free-deepseek"


def test_code_marker_routes_to_stable_free():
    assert route("debug this:\n```python\nprint(1)\n```") == "free-deepseek"


def test_think_tag_routes_reason_tier():
    # Stable free reasoners (free-*) lead the reason tier over NIM.
    assert route("[THINK] prove the halting problem is undecidable") == "free-deepseek"


def test_frontier_tag_prefers_free_nim_when_healthy():
    # Cost-first frontier: the free DeepSeek 0731 leads on published agentic scores
    # (Terminal-Bench 2.1 82.7 vs 72.1 for the previous *pro*), then free NIM, then flat/paid.
    assert route("[FRONTIER] design a novel consensus protocol") == "free-deepseek"

def test_frontier_falls_to_anthropic_then_copilot_when_free_down():
    # NIM + Zen down -> Anthropic Max, then Copilot. (avail without 'anthropic' key -> ant gated out.)
    r = route("[FRONTIER] design a novel consensus protocol",
              avail={"nim": False, "zen": False, "copilot": True})
    assert r == "co-opus"

def test_frontier_gives_nim_thinking():
    # FRONTIER keeps explicit (quality) order — not stability-sorted — so nim-glm still leads and
    # its reasoning gets toggled via chat_template_kwargs.enable_thinking (verified live).
    # (REASON now leads with a stable-free native reasoner, so it's covered separately.)
    # Pin nim-glm explicitly: the point here is the injected parameter SHAPE, not who leads the
    # tier. free-deepseek now leads frontier and is a native reasoner, which injects nothing.
    r = _hook({"model": "nim-glm", "messages": [{"role": "user", "content": "[FRONTIER] hard task"}]})
    assert r["model"] == "nim-glm"
    assert r["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert "thinking_budget_tokens" not in r["extra_body"]   # NIM has no budget knob


def test_cheap_falls_to_zen_when_nim_down():
    assert route("Say hi", avail={"nim": False, "zen": True, "copilot": True}) == "free-deepseek"


def test_reason_falls_to_copilot_when_nim_and_zen_down():
    r = route("[THINK] hard", avail={"nim": False, "zen": False, "copilot": True})
    assert r == "co-opus"


def test_health_gate_skips_slow_nim_model():
    # With zen down, cheap tier is NIM-only; a slow nim-llama is skipped to the next NIM.
    # (Stability-first would otherwise pick a zen-free model regardless of nim health.)
    health = {"nim-llama": {"ok": False}}
    assert route("Say hi", avail={"nim": True, "zen": False, "copilot": True},
                 health=health) == "nim-deepseek-flash"


def test_health_gate_all_cheap_nim_slow_falls_to_zen():
    health = {"nim-llama": {"ok": False}, "nim-deepseek-flash": {"ok": False}}
    assert route("Say hi", health=health) == "free-deepseek"


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
              "free-nemotron": {"ok": True, "latency_ms": 1500}}
    tier = ["nim-glm", "free-nemotron", "go-glm", "co-sonnet"]
    out = pr.order_tier(tier, health)
    assert out.index("free-nemotron") < out.index("nim-glm")


def test_served_is_paid_matches_only_paid_aliases():
    # Paid detection keys on the zen-* ALIAS (underlying model names are shared with GO twins).
    assert pr._served_is_paid("zen-kimi3") is True
    assert pr._served_is_paid("zen-glm") is True
    assert pr._served_is_paid("go-kimi-k3") is False      # GO brain — flat, not paid
    assert pr._served_is_paid("kimi-k3") is False          # underlying id — ambiguous, never match
    assert pr._served_is_paid("nim-glm") is False
    assert pr._served_is_paid(None) is False


def test_trigger_free_refresh_writes_when_writable(tmp_path, monkeypatch):
    # Best-effort touch of the bind-mounted trigger file. Point it at a temp path and confirm it writes.
    trig = tmp_path / ".llmr-refresh-trigger"
    monkeypatch.setattr(pr, "_REFRESH_TRIGGER", trig)
    pr._trigger_free_refresh()
    assert trig.exists() and trig.read_text().strip().isdigit()


def test_free_hosts_rank_zenfree_then_mistral_then_nim():
    # Reliability order, and it outranks measured latency: NIM fastest here and still last, and
    # zen-free leads Mistral even on identical latency (they used to tie, so whichever was quicker
    # that minute led — the flapping this ordering exists to stop).
    health = {"nim-glm": {"ok": True, "latency_ms": 500},        # NIM fast...
              "mis-large": {"ok": True, "latency_ms": 3000},    # ...Mistral slower...
              "free-north": {"ok": True, "latency_ms": 3000}}
    out = pr.order_tier(["nim-glm", "mis-large", "free-north"], health)
    assert out == ["free-north", "mis-large", "nim-glm"]
    assert (pr._stability_rank("free-north"), pr._stability_rank("mis-large"),
            pr._stability_rank("nim-glm")) == (1, 2, 3)
    # ...and the new DeepSeek outranks even the rest of zen-free.
    assert pr._stability_rank("free-deepseek") == 0


def test_new_deepseek_is_the_default_in_every_auto_tier():
    # The new DeepSeek V4 Flash (0731, DeepSeek-hosted) is free AND near-frontier, so it leads
    # every automatically-routed tier whenever it is healthy. FRONTIER/ORCHESTRATOR are excluded
    # on purpose: they keep config (quality) order rather than the cost/stability sort.
    av = {p: True for p in pr.PRIORITY_CHAIN}
    for tier in ("cheap", "general", "code", "reason", "agent"):
        got = pr.route("x", {"tier": tier}, av, {})
        assert got == "free-deepseek", f"{tier} led with {got}, not the new DeepSeek"


def test_free_ladder_is_new_deepseek_then_zenfree_then_mistral_then_nim():
    # The full ordering asked for: new DeepSeek -> rest of Zen free -> Mistral -> NIM, and only
    # then anything that costs (flat before paid, enforced by _cost_class).
    assert pr._stability_rank("free-deepseek") == 0
    assert pr._stability_rank("free-north") == 1
    assert pr._stability_rank("mis-large") == 2
    assert pr._stability_rank("nim-glm") == 3
    tier = ["co-sonnet", "zen-glm", "go-glm", "nim-glm", "mis-large", "free-north", "free-deepseek"]
    out = pr.order_tier(tier, {})
    assert out[:4] == ["free-deepseek", "free-north", "mis-large", "nim-glm"]
    # everything free first, then ascending cost — flat (go) before per-token (zen) before copilot
    assert [pr._cost_class(m) for m in out] == sorted(pr._cost_class(m) for m in out)


def test_old_deepseek_builds_are_kept_but_never_lead():
    # The old relays stay wired (they are still capacity) but must not outrank the new build.
    # NIM's same-named deepseek-v4-flash answered "1, 2" to a question the new one gets right.
    assert "nim-deepseek-flash" in pr.CHEAP_TIER          # kept, not deleted
    assert pr._stability_rank("nim-deepseek-flash") == 3  # ...but last among free
    assert "go-deepseek-flash" in pr.NEW_DEEPSEEK
    assert pr._cost_class("go-deepseek-flash") == 1       # flat: always behind the free twin


def test_models_that_reason_natively_are_all_declared():
    # Verified live with the health-probe payload and no param: each emits reasoning_content
    # (nim-nemotron-super 59 chars, free-nemotron 52, nim-nemotron 44, all finish_reason=length on
    # a 16-token cap). Being absent from this set cost twice -- the banner said think:off while the
    # model was visibly reasoning, and nim_health.sh judged their thinking time against a
    # non-reasoner's latency ceiling and benched them as unhealthy.
    for m in ("free-ling", "free-mimo", "free-north", "free-nemotron",
              "nim-step", "nim-nemotron", "nim-nemotron-super",
              "mis-medium", "mis-magistral"):
        assert m in pr.NATIVE_REASONERS, f"{m} reasons natively but is not declared"
        assert m in pr.ALL_THINKING, f"{m} would be annotated think:off"
    # Checked the same way and returns no reasoning_content -- must NOT get the reasoner ceiling.
    assert "nim-deepseek" not in pr.NATIVE_REASONERS


def test_an_unhealthy_zenfree_does_not_block_the_next_free_host():
    # Reliability order must not become a dead end: rank 0 that is down is skipped, not waited on.
    health = {"free-north": {"ok": False, "latency_ms": 900},
              "mis-large": {"ok": True, "latency_ms": 3000},
              "nim-glm": {"ok": True, "latency_ms": 500}}
    av = {p: True for p in pr.PRIORITY_CHAIN}
    assert pr.pick_model(["free-north", "mis-large", "nim-glm"], av, health) == "mis-large"


def test_order_tier_go_never_before_free():
    # GO (class 1) stays behind ALL free (class 0) even when faster
    health = {"nim-glm": {"ok": True, "latency_ms": 8000},
              "go-glm": {"ok": True, "latency_ms": 900}}
    tier = ["nim-glm", "go-glm", "co-sonnet"]
    out = pr.order_tier(tier, health)
    assert out.index("nim-glm") < out.index("go-glm")
    assert out.index("go-glm") < out.index("co-sonnet")


def test_order_tier_unprobed_keeps_config_order():
    out = pr.order_tier(["nim-deepseek", "nim-gptoss", "co-sonnet"], {})
    assert out == ["nim-deepseek", "nim-gptoss", "co-sonnet"]


def test_pick_model_uses_latency_order():
    health = {"nim-glm": {"ok": True, "latency_ms": 9000},
              "free-nemotron": {"ok": True, "latency_ms": 1200}}
    avail = {"nim": True, "zen": True, "copilot": True}
    got = pr.pick_model(["nim-glm", "free-nemotron", "co-sonnet"], avail, health)
    assert got == "free-nemotron"


def test_unhealthy_zen_now_filtered_too():
    health = {"free-nemotron": {"ok": False, "latency_ms": 11000}}
    avail = {"nim": False, "zen": True, "copilot": True}
    got = pr.pick_model(["free-nemotron", "co-sonnet"], avail, health)
    assert got == "co-sonnet"


# --- adaptive thinking depth ---

def test_think_budget_nim_reason_returns_high():
    assert pr._think_budget("nim-glm", "reason", False) == ("high", 32768)

def test_think_budget_nim_frontier_returns_high():
    assert pr._think_budget("nim-minimax", "frontier", False) == ("high", 32768)

def test_think_budget_nim_minimax_agent_returns_high():
    assert pr._think_budget("nim-minimax", "agent", False) == ("high", 32768)

def test_think_budget_go_reason_returns_medium():
    assert pr._think_budget("go-glm", "reason", False) == ("medium", 8192)

def test_think_budget_go_frontier_returns_high():
    assert pr._think_budget("go-kimi", "frontier", False) == ("high", 16384)

def test_think_budget_ant_reason_returns_medium():
    assert pr._think_budget("ant-opus", "reason", False) == ("medium", 8192)

def test_think_budget_ant_frontier_returns_high():
    assert pr._think_budget("ant-fable", "frontier", False) == ("high", 16384)

def test_think_budget_ant_haiku_excluded():
    assert pr._think_budget("ant-haiku", "frontier", False) is None

def test_think_budget_cheap_tier_returns_none():
    assert pr._think_budget("nim-glm", "cheap", False) is None

def test_think_budget_general_tier_returns_none():
    assert pr._think_budget("go-glm", "general", False) is None

def test_think_budget_non_thinking_model_returns_none():
    assert pr._think_budget("nim-deepseek", "reason", False) is None

def test_think_budget_boost_forces_high_on_go():
    assert pr._think_budget("go-glm", "reason", True) == ("high", 16384)

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
    data = {"model": "go-glm", "messages": [{"role": "user", "content": "[REASON] explain this trade-off"}]}
    result = _hook(data)
    # GO takes an effort knob, not a token budget: `thinking_budget_tokens` 400s on this provider.
    assert result.get("extra_body", {}).get("reasoning_effort") == "medium"

def test_hook_no_thinking_on_cheap_prompt_nim_glm():
    data = {"model": "nim-glm", "messages": [{"role": "user", "content": "say hi"}]}
    result = _hook(data)
    assert "thinking" not in result
    assert "extra_body" not in result

def test_hook_boost_forces_high_on_go():
    data = {"model": "go-glm", "messages": [{"role": "user", "content": "[BOOST][REASON] redo this"}]}
    result = _hook(data)
    assert result["extra_body"]["reasoning_effort"] == "high"

def test_hook_force_tier_lifts_orchestrator_to_high():
    """Fusion's aggregator/conductor/auditor synthesize over drafts that are themselves only
    MEDIUM-thought. The synthesizer is what catches their errors, so it must think HIGH. It
    asks for that via metadata, not by prepending [FRONTIER] to a prompt the model would read."""
    drafts = "--- DRAFT 1 (nim-glm) ---\nsome draft text"
    without = _hook({"model": "go-glm", "messages": [{"role": "user", "content": drafts}]})
    with_ft = _hook({"model": "go-glm", "metadata": {"force_tier": "frontier"},
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
    r = _hook({"model": "go-glm", "messages": [{"role": "user", "content": "[REASON] prove it"}]})
    eb = r.get("extra_body", {})
    assert eb.get("reasoning_effort") == "medium"          # go/reason -> MEDIUM
    assert "enable_thinking" not in eb
    assert "thinking_budget_tokens" not in eb

def test_hook_go_frontier_is_high_effort():
    r = _hook({"model": "go-glm", "messages": [{"role": "user", "content": "[FRONTIER] hard"}]})
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


# --- free-tier native reasoners (free-ling/mimo/north) -------------------------------------
# They reason at max depth with NO param (reasoning_effort only shrinks it), and they're free,
# so: always HIGH, inject nothing, annotate the banner high.

def test_free_reasoner_budget_is_high_any_tier_no_boost():
    for tier in ("reason", "code", "cheap", "general", "frontier"):
        assert pr._think_budget("free-mimo", tier, False) == ("high", 0)

def test_free_reasoner_injects_no_param_but_annotates_high():
    r = _hook({"model": "free-ling", "messages": [{"role": "user", "content": "[REASON] prove it"}]})
    assert "extra_body" not in r                        # no reasoning_effort, no chat_template_kwargs
    assert "thinking" not in r
    assert r["metadata"]["llmr_ann"]["think"] == "high"  # banner still shows high (native)

def test_free_reasoner_carries_exactly_one_or_zero_dialects():
    # covered by the ALL_THINKING guard, but pin it: free reasoners inject ZERO param dialects.
    for m in ("free-ling", "free-mimo", "free-north"):
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[REASON] x"}]})
        eb = r.get("extra_body", {})
        assert not r.get("thinking") and "reasoning_effort" not in eb \
            and "enable_thinking" not in eb.get("chat_template_kwargs", {})

def test_free_reasoners_are_in_reason_tier():
    for m in ("free-ling", "free-mimo", "free-north"):
        assert m in pr.REASON_TIER


# --- orchestrator/auditor tier (top-layer brain: grok-4.5 + kimi-k3 -> capable reasoners) -----

def test_orch_default_is_free_first_not_zen():
    # Orchestrator leads with FREE, zen GO is LAST. The free DeepSeek 0731 heads the tier on
    # published agentic scores; if it is down the other free reasoners (mis-medium/nim-glm)
    # follow, and go-* is reached only when every free and flat option is gone.
    assert route("[ORCH] synthesize these drafts") == "free-deepseek"
    assert route("[AUDIT] check this answer") == "free-deepseek"
    down = {"free-deepseek": {"ok": False}}
    nxt = route("[AUDIT] check this answer", health=down)
    assert pr._cost_class(nxt) == 0 and nxt != "free-deepseek"

def test_orch_boost_stays_free_not_brain_not_zen():
    # Even [BOOST][ORCH] on drift stays on FREE (brains restricted; zen GO is last-resort).
    av = {"nim": True, "zen": True, "copilot": True, "anthropic": True}
    for p in ("[BOOST][ORCH] this verdict is wrong", "[BOOST][AUDIT] redo this"):
        got = route(p, avail=av)
        assert pr._cost_class(got) == 0, f"orchestrator boost spent money: {got}"
        assert got not in pr.LAST_RESORT_BRAINS, f"boost reached a brain: {got}"

def test_orch_free_before_zen_full_avail():
    # With everything up, a FREE reasoner leads and zen GO is never touched. The leader is the
    # new DeepSeek; what matters is that it costs nothing and is not a GO/paid alias.
    av = {"nim": True, "mistral": True, "zai": True, "zen": True, "anthropic": True, "copilot": True}
    got = route("[BOOST][ORCH] wrong, redo", avail=av)
    assert got == "free-deepseek" and pr._cost_class(got) == 0

def test_brains_never_auto_selected_on_everyday_tiers():
    # The scarce brains (grok 120 req/5h, kimi-k3 110) are allowed ONLY at the frontier tail.
    # No everyday tier may reach them — not even with every other model health-gated down, which
    # is the path that produced the surprise 63k-token kimi-k3 calls.
    av = {"nim": True, "mistral": True, "zai": True, "zen": True, "anthropic": True, "copilot": True}
    allbad = {t: {"ok": False} for t in pr.MODEL_PROVIDER if t not in pr.PREMIUM_ONLY}
    # NB: frontier and orchestrator are excluded — those two are exactly where premium belongs.
    for tier in ("cheap", "general", "code", "reason", "agent"):
        got = pr.route("x", {"tier": tier}, av, allbad)
        assert got not in pr.PREMIUM_ONLY, f"{tier} reached a premium model: {got}"


def test_brains_are_the_frontier_tail_last_resort():
    # Frontier normally picks something else...
    av = {"nim": True, "mistral": True, "zai": True, "zen": True, "anthropic": True, "copilot": True}
    assert pr.route("x", {"tier": "frontier"}, av, {}) not in pr.LAST_RESORT_BRAINS
    # ...but when every other frontier member is down, a brain IS the last thing tried.
    down = {m: {"ok": False} for m in pr.FRONTIER_TIER if m not in pr.LAST_RESORT_BRAINS}
    assert pr.route("x", {"tier": "frontier"}, av, down) in pr.LAST_RESORT_BRAINS
    # They sit at the very end of the tier, after copilot.
    for b in pr.LAST_RESORT_BRAINS:
        assert pr.FRONTIER_TIER.index(b) > pr.FRONTIER_TIER.index("co-opus")

def test_brain_still_reachable_by_EXPLICIT_request():
    # Restriction is AUTO-only: if opencode explicitly names the alias, it still routes there.
    r = _hook({"model": "go-kimi-k3", "messages": [{"role": "user", "content": "hard verdict"}]})
    assert r["model"] == "go-kimi-k3"                       # not rerouted away

def test_orchestrator_falls_to_zen_only_when_free_and_flat_down():
    # zen GO is the LAST orchestrator resort: reached only when free (nim) + flat (z.ai/anthropic)
    # are all unavailable. Here nim is down and mistral/zai/anthropic absent -> go-glm.
    # Every FREE option must be gone, which now includes the zen-free DeepSeek, before GO is
    # allowed to answer. (Marking the provider down is not enough: go-* shares it.)
    av = {"nim": False, "zen": True, "copilot": True}
    h = {m: {"ok": False} for m in pr.MODEL_PROVIDER if pr._cost_class(m) == 0}
    got = route("[ORCH] x", avail=av, health=h)
    assert got.startswith("go-") and pr._cost_class(got) == 1, f"expected a GO alias, got {got}"

def test_orchestrator_free_nim_leads_when_flat_down():
    # brains + anthropic down, but free NIM up -> nim-glm leads (free before zen GO).
    av = {"nim": True, "zen": True, "copilot": True, "anthropic": False}
    h = {"go-kimi-k3": {"ok": False}, "go-grok": {"ok": False}}
    got = route("[ORCH] x", avail=av, health=h)
    assert pr._cost_class(got) == 0, f"flat lane answered while free was up: {got}"
    # ...and with the new DeepSeek benched, the free NIM reasoner is next.
    h2 = dict(h); h2["free-deepseek"] = {"ok": False}
    assert pr._cost_class(route("[ORCH] x", avail=av, health=h2)) == 0

def test_orch_brains_get_high_thinking_via_go_shape():
    r = _hook({"model": "go-kimi-k3", "messages": [{"role": "user", "content": "[ORCH] hard"}]})
    assert r["extra_body"]["reasoning_effort"] == "high"   # GO shape, orchestrator -> high
    r2 = _hook({"model": "go-grok", "messages": [{"role": "user", "content": "[ORCH] hard"}]})
    assert r2["extra_body"]["reasoning_effort"] == "high"

def test_orch_brains_are_flatrate_cost_class():
    # GO flat-rate leads the flat band (class 1), not per-token/anthropic
    assert pr._cost_class("go-kimi-k3") == 1 and pr._cost_class("go-grok") == 1

def test_orch_l2_capables_in_tier():
    for m in ("go-qwen-max", "go-qwen-plus", "go-deepseek", "go-minimax", "go-mimo"):
        assert m in pr.ORCHESTRATOR_TIER

def test_orch_l2_reasoners_get_high_others_off():
    # qwen/deepseek emit CoT (verified live) -> reasoning_effort high; minimax/mimo don't -> off
    for m in ("go-qwen-max", "go-qwen-plus", "go-deepseek"):
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[ORCH] x"}]})
        assert r["extra_body"]["reasoning_effort"] == "high"
    for m in ("go-mimo",):     # not in GO_THINKING -> no param, banner off
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[ORCH] x"}]})
        assert "extra_body" not in r
        assert r["metadata"]["llmr_ann"]["think"] == "off"


# --- research-pipeline role tags -------------------------------------------------------------

def test_role_tags_map_to_worker_tiers():
    # Stability-first: workers land on the stable-free leader of their tier (zen-free), not NIM.
    assert route("[SCOUT] list the files in this repo") == "free-deepseek"  # cheap
    assert route("[ANALYST] interpret these benchmark numbers") == "free-deepseek"  # reason
    assert route("[VERIFIER] check this claim against the source") == "free-deepseek"
    assert route("[AUDITOR] does the code match the spec") == "free-deepseek"

def test_role_tag_parsed_and_stripped():
    cleaned, d = pr.parse_request("[SCOUT] gather links")
    assert d["tier"] == "cheap" and "[SCOUT]" not in cleaned
    _, d2 = pr.parse_request("[VERIFIER] cross-check")
    assert d2["tier"] == "reason"

def test_orch_boss_distinct_from_worker_roles():
    # [ORCH] = orchestrator tier (free-first: nim-glm), role tags are the reason/cheap workers.
    assert route("[ORCH] final verdict") == "free-deepseek"     # orchestrator tier (free leads, zen last)
    assert route("[VERIFIER] check") == "free-deepseek"       # reason tier (new DeepSeek leads)


# --- paid per-token overflow ($20 balance) ---------------------------------------------------

PAID = ["zen-minimax", "zen-qwen-plus", "zen-glm", "zen-luna",
        "zen-grok", "zen-qwen-max", "zen-kimi3", "zen-terra", "zen-sol"]

def test_paid_models_are_pertoken_cost_class_5():
    # per-token (class 4) sits BELOW every flat sub (z.ai 1, GO 2, Anthropic Max 3) and above only
    # copilot -> all flat-rate capacity is drained before the $20 balance is touched.
    for m in PAID:
        assert pr._cost_class(m) == 5, f"{m} cost class {pr._cost_class(m)} != 4"

def test_paid_models_never_in_a_default_tier():
    # reached only via fallback map / boost, never auto-selected -> protects the balance.
    for tier in pr.TIER_MAP.values():
        for m in PAID:
            assert m not in tier, f"{m} leaked into a default tier"

def test_paid_reasoners_use_reasoning_effort_minimax_excluded():
    r = _hook({"model": "zen-glm", "messages": [{"role": "user", "content": "[REASON] x"}]})
    assert r["extra_body"]["reasoning_effort"] == "medium"      # go/reason
    r2 = _hook({"model": "zen-minimax", "messages": [{"role": "user", "content": "[REASON] x"}]})
    assert "extra_body" not in r2                               # minimax: no CoT -> no param


# --- Mistral (free, class 0) + z.ai GLM Coding Plan (flat, class 1) ---

ALL_OK2 = {"nim": True, "mistral": True, "zai": True, "zen": True, "anthropic": True, "copilot": True}


def test_provider_map_knows_mistral_and_zai():
    assert pr.MODEL_PROVIDER["mis-large"] == "mistral"
    assert pr.MODEL_PROVIDER["mis-magistral"] == "mistral"
    assert pr.MODEL_PROVIDER["zai-52"] == "zai"
    assert pr.MODEL_PROVIDER["zai-flash"] == "zai"


def test_cost_class_cascade_is_free_go_codex_ant_zai_paid_copilot():
    # The whole ordering intent in one assertion: strictly increasing marginal cost. Every
    # flat/sunk subscription ranks before any real per-token spend.
    assert pr._cost_class("mis-large") == 0        # Mistral free
    assert pr._cost_class("nim-glm") == 0          # NIM free
    assert pr._cost_class("free-ling") == 0        # Zen free tier
    assert pr._cost_class("go-glm") == 1           # opencode GO flat — most generous, spend first
    assert pr._cost_class("cod-luna") == 2         # Codex/ChatGPT flat
    assert pr._cost_class("ant-sonnet") == 3       # Claude Max flat
    assert pr._cost_class("zai-52") == 4           # z.ai flat
    assert pr._cost_class("zen-glm") == 5          # zen per-token (real money)
    assert pr._cost_class("co-opus") == 6          # copilot per-request


def test_go_flat_ranks_before_zai_in_reason_tier():
    assert pr._cost_class("go-glm") < pr._cost_class("zai-52")
    # Sorted reason tier: free (mist/nim/zen-free) then GO go-glm(1) then z.ai(2) then ant(3).
    ordered = pr.order_tier(pr.REASON_TIER, {})
    assert ordered.index("go-glm") < ordered.index("zai-52")
    assert ordered.index("mis-medium") < ordered.index("go-glm")  # free before flat
    assert ordered.index("ant-sonnet") < ordered.index("zai-52")    # Anthropic flat before z.ai flat


def test_anthropic_flat_ranks_before_zen_paid():
    # Both appear in the frontier-ish fallback space; anth (sunk) must precede zen paid ($).
    tier = ["zen-glm", "ant-sonnet"]
    assert pr.order_tier(tier, {}) == ["ant-sonnet", "zen-glm"]


def test_mistral_reasoners_are_native_no_param():
    # magistral / mistral-medium reason inline with NO param (verified live). Inject nothing.
    for m in ("mis-medium", "mis-magistral"):
        r = _hook({"model": m, "messages": [{"role": "user", "content": "[REASON] prove it"}]},
                  avail=ALL_OK2)
        assert r["model"] == m
        assert "thinking" not in r
        assert "extra_body" not in r
        assert r["metadata"]["llmr_ann"]["think"] == "high"   # annotated high, native depth


def test_mistral_non_reasoners_get_no_thinking():
    for m in ("mis-large", "mis-codestral"):
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
    # GO go-glm (class 1) before z.ai (class 2). NIM/Mistral are also masked by availability.
    # All of FREE_POOL must go down, not just the reason-tier ones: since free_fallback() the
    # tier borrows any other healthy free model first — which is the point of that feature.
    health = {m: {"ok": False} for m in pr.FREE_POOL}
    r = route("[REASON] prove the halting problem",
              avail={"mistral": False, "nim": False, "zai": True, "zen": True,
                     "anthropic": True, "copilot": True},
              health=health)
    # Assert the INTENT (GO flat class 1 beats z.ai class 2), not a specific alias — which GO
    # model wins depends on the quota ranking and changes as opencode adds models.
    assert r in pr.GO_ALIASES, f"expected a GO model, got {r}"
    assert pr._cost_class(r) == 1 < pr._cost_class("zai-52")


# --- Codex / ChatGPT subscription (flat, class 3 alongside Anthropic) -------------------------

def test_codex_flat_precedes_anthropic_flat():
    # Both are sunk-cost subscriptions, but Codex is spent before Claude Max.
    for m in ("cod-sol", "cod-terra", "cod-luna", "cod-mini"):
        assert pr._cost_class(m) == 2 < pr._cost_class("ant-sonnet")


def test_codex_ranks_after_go_but_before_anthropic_zai_and_paid():
    tier = ["zen-glm", "zai-52", "ant-sonnet", "cod-luna", "go-glm", "nim-glm"]
    assert pr.order_tier(tier, {}) == ["nim-glm", "go-glm", "cod-luna", "ant-sonnet", "zai-52", "zen-glm"]



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
    assert route("hi") in pr.CHEAP_TIER


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
# --- opencode GO quota ranking (2026-07-30) ----------------------------------------------------

def test_go_luna_beats_codex_luna():
    # Same underlying model (gpt-5.6-luna). GO is a flat subscription at cost class 1; Codex is
    # class 3. Routing must prefer the cheaper class rather than the Codex proxy.
    assert pr.order_tier(["cod-luna", "go-luna"], {},
                         native={"cod-luna", "go-luna"})[0] == "go-luna"


def test_new_go_models_are_flat_class_one():
    for m in ("go-mimo-lite", "go-hy3", "go-luna"):
        assert m in pr.GO_ALIASES
        assert pr._cost_class(m) == 1


def test_generous_go_models_lead_their_band():
    # Inside the GO band, config order encodes quota generosity (req/5h), since GO is never
    # latency-probed: mimo-lite 30,100 > hy3 4,300 > luna 2,050 > glm-5.2 880.
    go = [m for m in pr.order_tier(pr.GENERAL_TIER, {}, native=set(pr.GENERAL_TIER))
          if m in pr.GO_ALIASES]
    assert go.index("go-mimo-lite") < go.index("go-glm")
    assert go.index("go-hy3") < go.index("go-glm")


def test_low_quota_brains_stay_out_of_the_go_band_after_expansion():
    # grok 120/5h and kimi-k3 110/5h are the scarcest GO models. The roster grew to 17, but they
    # must still appear in NO everyday tier — only the frontier tail.
    for m in ("go-grok", "go-kimi-k3"):
        for name, tier in pr.TIER_MAP.items():
            if name != "frontier":
                assert m not in tier, f"{m} leaked into {name}"


# --- premium confinement + free-borrows-down / flat-does-not ----------------------------------

def test_premium_models_appear_in_no_everyday_tier_list():
    """Scarce-quota or frontier-priced models belong to frontier/orchestrator only.

    Every everyday tier already carries its own flat fallback, so reaching for one of these is
    never justified. Guarding the LISTS (not just routing) because the expensive direction of
    this mistake is silent: a 110-req/5h model quietly answering routine prompts."""
    for name, tier in pr.TIER_MAP.items():
        if name in ("frontier", "orchestrator"):
            continue
        leaked = [m for m in tier if m in pr.PREMIUM_ONLY]
        assert not leaked, f"{name} tier lists premium model(s): {leaked}"


def test_free_borrows_down_but_flat_does_not():
    """Free capacity is shared across tiers; flat capacity is not.

    A free model listed only in REASON must still be able to answer a CHEAP prompt (free is free).
    A flat model must NOT, because each tier already has its own flat fallback chosen for that
    tier's cost/capability — borrowing one would spend a subscription the tier didn't budget for."""
    borrowed = set(pr.free_fallback(pr.CHEAP_TIER)) - set(pr.CHEAP_TIER)
    assert borrowed, "nothing was borrowed — free_fallback is not doing its job"
    for m in borrowed:
        assert pr._cost_class(m) == 0, f"{m} was borrowed into CHEAP but is not free"


# --- borrowing is ONE-DIRECTIONAL + large prompts demand capability ----------------------------
# Regression: the audit trail showed 24% of >20k-token requests answered by CHEAP-tier models
# (big-pickle at 81,802 tokens, llama-3.3-70b at 79,948). Nothing errored — they returned a worse
# answer — so the failure was invisible. Two causes, one test block each.

def test_weak_models_are_never_borrowed_up_into_capable_tiers():
    for tier in ("code", "reason", "agent"):
        borrowed = [m for m in pr.free_fallback(pr.TIER_MAP[tier], pr.TIER_CAPABILITY[tier])
                    if m not in pr.TIER_MAP[tier]]
        for m in borrowed:
            assert pr._capability_rank(m) >= pr.TIER_CAPABILITY[tier], (
                f"{m} (rank {pr._capability_rank(m)}) borrowed up into {tier}")


def test_capable_models_still_borrow_down_into_cheap():
    # The direction that IS wanted: free is free, so a REASON-grade free model may answer a
    # trivial prompt when cheap's own models are down.
    borrowed = set(pr.free_fallback(pr.CHEAP_TIER, pr.TIER_CAPABILITY["cheap"])) - set(pr.CHEAP_TIER)
    assert borrowed, "cheap borrowed nothing — the down direction regressed"
    assert any(pr._capability_rank(m) >= pr.TIER_CAPABILITY["reason"] for m in borrowed)


def test_large_prompt_never_routes_to_a_low_capability_model():
    huge = "refactor the authentication module " * 6000      # ~37k tokens
    assert len(huge) // pr.CHARS_PER_TOKEN >= pr.LARGE_PROMPT_TOKENS
    av = {p: True for p in pr.PRIORITY_CHAIN}
    # knock out the code tier's own free models so it must reach for a substitute
    down = {m: {"ok": False} for m in pr.CODE_TIER if pr._cost_class(m) == 0}
    got = pr.route(huge, {}, av, down)
    assert got is not None
    assert got not in pr.CHEAP_TIER, f"huge prompt routed to cheap-tier model {got}"
    assert pr._capability_rank(got) >= pr.TIER_CAPABILITY["code"] or pr._cost_class(got) > 0


def test_small_prompt_still_allowed_on_cheap_models():
    # The size floor must not drag every short prompt up a tier.
    got = route("say hi")
    assert pr._cost_class(got) == 0
    assert pr.classify("say hi") == "cheap"


def test_size_floor_measures_the_whole_payload_not_just_the_last_message():
    """A short question inside a huge session must still get a capable model.

    This is the ACTUAL shape of the bug seen in the audit trail. The hook parses tags and content
    markers off the last user message, so sizing on that same string made a 2-token "fix this"
    inside an 80k-token conversation look trivial — which is how big-pickle came to serve 81,802
    prompt tokens. The model reads the whole payload, so the whole payload is what decides."""
    av = {p: True for p in pr.PRIORITY_CHAIN}
    huge_ctx = pr.LARGE_PROMPT_TOKENS * pr.CHARS_PER_TOKEN * 2      # ~50k tokens of history
    got = pr.route("fix this", {}, av, {}, context_chars=huge_ctx)
    # Assert CAPABILITY, not tier membership: a model can sit in CHEAP and still be capable
    # (mis-codestral is a code specialist that also serves cheap work). What must never happen is
    # a rank-0 model — one that only ever qualified for trivial prompts — taking an 80k payload.
    assert pr._capability_rank(got) >= pr.TIER_CAPABILITY["code"] or pr._cost_class(got) > 0, (
        f"short prompt in a huge session routed to low-capability {got} "
        f"(rank {pr._capability_rank(got)})")


def test_small_conversation_is_unaffected_by_the_payload_measure():
    av = {p: True for p in pr.PRIORITY_CHAIN}
    got = pr.route("say hi", {}, av, {}, context_chars=len("say hi"))
    assert pr._cost_class(got) == 0


# --- cache-aware routing + session stickiness (2026-07-31) -------------------------------------
# Measured: free-north caches 0% across 28.35M prompt tokens and is the CODE-tier default, so
# every turn re-runs prefill over the whole payload. Costs nothing on a free lane; costs latency
# on every request.

PROFILE = {"free-north": {"hit_pct": 0.0, "samples": 50},
           "nim-step": {"hit_pct": 0.0, "samples": 28},
           "free-pickle": {"hit_pct": 92.8, "samples": 17},
           "thin-model": {"hit_pct": 0.0, "samples": 2}}


def test_cache_rank_demotes_only_measured_zeros():
    assert pr._cache_rank("free-north", PROFILE) == 1
    assert pr._cache_rank("nim-step", PROFILE) == 1
    assert pr._cache_rank("free-pickle", PROFILE) == 0
    assert pr._cache_rank("mis-large", PROFILE) == 0      # unknown is never penalised
    assert pr._cache_rank("thin-model", PROFILE) == 0     # too few samples to act on
    assert pr._cache_rank("free-north", {}) == 0          # no profile at all -> fail open


def test_cache_preference_reorders_only_within_a_cost_class():
    # It must never promote a paid model over a free one — cost class still dominates.
    tier = ["free-north", "free-pickle", "go-glm", "ant-sonnet"]
    out = pr.order_tier(tier, {}, cache=PROFILE)
    assert out.index("free-pickle") < out.index("free-north")   # caching free model first
    assert [pr._cost_class(m) for m in out] == sorted(pr._cost_class(m) for m in out)


def test_small_payloads_ignore_the_cache_preference():
    # Nothing to cache on a short prompt, so a 0%-cache model is not worse and keeps its place.
    av = {p: True for p in pr.PRIORITY_CHAIN}
    small = pr.route("say hi", {}, av, {}, context_chars=40, cache=PROFILE)
    assert pr._cost_class(small) == 0


def test_mid_size_payloads_get_the_cache_preference():
    # The cache floor is NOT the capability floor. A caller re-sending a fixed ~8k prefix pays the
    # whole prefill on every request if the model cannot cache, long before the payload is "large".
    assert pr.CACHE_PREF_MIN_TOKENS < pr.LARGE_PROMPT_TOKENS
    av = {p: True for p in pr.PRIORITY_CHAIN}
    mid = pr.CACHE_PREF_MIN_TOKENS * pr.CHARS_PER_TOKEN * 2          # ~8k tokens
    assert mid // pr.CHARS_PER_TOKEN < pr.LARGE_PROMPT_TOKENS        # still under the capability floor
    picked = pr.route("refactor this module", {"tier": "code"}, av, {}, mid, PROFILE)
    # free-north heads CODE on latency but caches nothing; at this size that now costs it the slot.
    assert picked != "free-north"
    assert pr._cache_rank(picked, PROFILE) == 0


def test_session_sticks_then_releases_on_tier_change_or_death():
    av = {p: True for p in pr.PRIORITY_CHAIN}
    pr._SESSION_MODELS.clear()
    k = pr.session_key([{"role": "user", "content": "a session"}])
    first = pr.route("one", {}, av, {}, 200, None, k)
    assert pr.route("two", {}, av, {}, 200, None, k) == first      # stays put
    # A tier change must RE-DECIDE rather than inherit. Asserting "different name" would be wrong:
    # the new DeepSeek legitimately leads several tiers, so the same model can win twice on merit.
    # frontier keeps config (quality) order instead of the stability sort, so its leader really is
    # a different model — which is what proves the sticky value was dropped, not reused.
    assert pr.route("x", {"tier": "frontier"}, av, {}, 200, None, k) == pr.FRONTIER_TIER[0]
    dead = {first: {"ok": False}}
    assert pr.route("one", {}, av, dead, 200, None, k) != first    # never pin a dead model


def test_session_store_is_bounded():
    pr._SESSION_MODELS.clear()
    av = {p: True for p in pr.PRIORITY_CHAIN}
    for i in range(pr._SESSION_MAX + 50):
        pr.route("hi", {}, av, {}, 200, None, f"key{i}")
    assert len(pr._SESSION_MODELS) <= pr._SESSION_MAX
    pr._SESSION_MODELS.clear()


def test_session_key_is_stable_as_the_conversation_grows():
    first = {"role": "user", "content": "the opening turn"}
    short = [first]
    grown = [first, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "more"}]
    assert pr.session_key(short) == pr.session_key(grown)
    assert pr.session_key([{"role": "assistant", "content": "no user turn"}]) is None


# --- Anthropic prompt caching (2026-07-31) ----------------------------------------------------
# Anthropic is the ONLY lane that does not cache by itself. Zen (free + GO), z.ai and Codex all
# cache automatically — verified live, and on z.ai sending cache_control changes nothing at all.
# Without an explicit breakpoint ant-* re-read every payload: 1.87M prompt tokens, 0 cached.

def test_anthropic_gets_two_cache_breakpoints():
    data = {"messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": "old"},
                         {"role": "assistant", "content": "reply"},
                         {"role": "user", "content": "newest"}]}
    assert pr.apply_anthropic_cache(data, "ant-sonnet", 5000) == 2
    sysmsg = data["messages"][0]
    assert sysmsg["content"][0]["cache_control"] == {"type": "ephemeral"}
    # the newest turn must stay uncached — it is the part that changes
    assert data["messages"][-1]["content"] == "newest"


def test_non_anthropic_providers_are_left_alone():
    # They cache automatically; an unexpected field is what 400s and then hides behind a fallback.
    for m in ("go-glm", "free-north", "cod-luna", "zai-52", "nim-glm"):
        data = {"messages": [{"role": "system", "content": "sys"},
                             {"role": "user", "content": "hi"}]}
        assert pr.apply_anthropic_cache(data, m, 5000) == 0
        assert data["messages"][0]["content"] == "sys"      # untouched


def test_small_anthropic_payloads_skip_caching():
    # Below Anthropic's minimum cacheable prefix a breakpoint is rejected, not merely useless.
    data = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]}
    assert pr.apply_anthropic_cache(data, "ant-sonnet", 100) == 0


def test_desperation_walk_is_cost_ordered_not_provider_ordered():
    """Layer 5 must not inherit PRIORITY_CHAIN's dict order. That order groups by provider, and
    provider order is not cost order — it reaches z.ai (4) before GO (1) and Copilot (6) before
    Anthropic (3) and Codex (2). With every tier empty the walk is the only thing choosing a lane,
    so an inversion here spends the dearest lane while flat quota sits unused."""
    saved = dict(pr.TIER_MAP)
    try:
        for k in pr.TIER_MAP:
            pr.TIER_MAP[k] = []
        avail = {p: True for p in pr.PRIORITY_CHAIN}
        got = pr.route("anything", {"tier": "general"}, avail, {})
        assert got is not None, "walk returned nothing while every provider was available"
        assert pr._cost_class(got) == 0, (
            f"walk chose {got} (class {pr._cost_class(got)}) while a free lane was available"
        )
        # Free lanes gone: it must take the cheapest remaining class, not the dict's next provider
        # — GO (1) before z.ai (4), never Copilot (6) while Codex (2) is up.
        avail["nim"] = avail["mistral"] = False
        avail["zen"] = False          # drops both the free- and go- aliases in that provider
        got = pr.route("anything", {"tier": "general"}, avail, {})
        assert pr._cost_class(got) == 2, (
            f"walk chose {got} (class {pr._cost_class(got)}) instead of the Codex flat lane"
        )
    finally:
        pr.TIER_MAP.update(saved)


def test_fallback_chains_never_skip_a_cheaper_lane_to_reach_copilot():
    """LiteLLM's static fallbacks fire on a mid-call error, independently of tier selection. They
    used to run free -> GO -> Copilot, so an error escaped to per-request credit while the flat
    Codex lane (class 2) was not a target in any of the 55 chains."""
    import pathlib

    import yaml

    cfg = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    )
    chains = cfg["router_settings"]["fallbacks"]
    assert chains, "no fallback chains configured"
    for group in chains:
        for alias, members in group.items():
            classes = [pr._cost_class(m) for m in members]
            assert classes == sorted(classes), f"{alias} fallback is not cost-ascending: {members}"
            if 6 in classes:  # ends at Copilot -> a cheaper flat lane must come first
                assert any(c in (1, 2, 3) for c in classes), (
                    f"{alias} escapes to Copilot without trying a flat lane: {members}"
                )
