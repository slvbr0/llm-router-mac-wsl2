import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
fu = importlib.import_module("fusion.fusion")

ALL_OK = {"nim": True, "zen": True, "copilot": True}
CFG = fu.load_config()


def test_easy_panel_is_provider_diverse():
    p = fu.build_panel("easy", CFG, ALL_OK, {})
    provs = {fu.provider_of(a) for a in p["proposers"]}
    assert len(p["proposers"]) >= 2 and len(provs) >= 2


def test_deep_panel_is_largest():
    e = fu.build_panel("easy", CFG, ALL_OK, {})
    h = fu.build_panel("hard", CFG, ALL_OK, {})
    d = fu.build_panel("deep", CFG, ALL_OK, {})
    assert len(d["proposers"]) > len(h["proposers"])       # deep = heavyweight committee
    assert len(h["proposers"]) >= 2 and len(e["proposers"]) >= 2


def test_panel_skips_unhealthy_nim():
    health = {"nim-inkling": {"ok": False}}
    p = fu.build_panel("easy", CFG, ALL_OK, health)
    assert "nim-inkling" not in p["proposers"]


def test_free_first_when_nim_healthy():
    # easy (paid_anchor 0): all-free when NIM healthy — no paid credit spent
    p = fu.build_panel("easy", CFG, ALL_OK, {})
    assert all(fu.provider_of(a) != "copilot" for a in p["proposers"])


def test_paid_backfills_when_free_degraded():
    # hard with most free NIM down -> paid backfills to keep the panel width
    health = {"nim-inkling": {"ok": False}, "nim-step": {"ok": False}, "nim-nemotron": {"ok": False}}
    p = fu.build_panel("hard", CFG, ALL_OK, health)
    assert any(fu.provider_of(a) == "copilot" for a in p["proposers"])   # paid pulled in
    assert len(p["proposers"]) >= 2


def test_panel_skips_masked_provider():
    p = fu.build_panel("easy", CFG, {"nim": True, "zen": False, "copilot": True}, {})
    assert all(fu.provider_of(a) != "zen" for a in p["proposers"])


def test_aggregator_first_healthy():
    # deep aggregator is [go-glm, co-opus, nim-nemotron, nim-inkling]; zen masked ->
    # skip unhealthy nim-nemotron too -> co-opus
    health = {"nim-nemotron": {"ok": False}}
    p = fu.build_panel("deep", CFG, {"nim": True, "zen": False, "copilot": True}, health)
    assert p["aggregator"] == "co-opus"


def test_parse_novel_tags():
    assert fu.parse_novel("[NOVEL] solve X") == ("solve X", "fuse", None)
    assert fu.parse_novel("[NOVEL DEEP] solve X") == ("solve X", "fuse", "deep")
    assert fu.parse_novel("[NOVEL RESEARCH] what is new in Y") == ("what is new in Y", "research", None)
    assert fu.parse_novel("plain prompt") == ("plain prompt", None, None)


def test_difficulty_maps_tiers():
    assert fu.difficulty_of("say hi") == "easy"
    assert fu.difficulty_of("[THINK] prove theorem") in ("easy", "hard")
    assert fu.difficulty_of("prove this theorem about complexity") == "hard"


def _stub_call(results):
    """Return a fake call_model keyed by alias; unknown alias -> failure."""
    def fake(alias, messages, key, timeout, max_tokens=None, force_tier=None):
        r = results.get(alias)
        if r is None:
            return {"alias": alias, "provider": fu.provider_of(alias), "ok": False,
                    "content": "", "tokens": 0, "latency_ms": 1, "error": "stub"}
        return {"alias": alias, "provider": fu.provider_of(alias), "ok": True,
                "content": r, "tokens": 10, "latency_ms": 1}
    return fake


def test_fuse_happy_path_aggregates(monkeypatch):
    monkeypatch.setattr(fu, "load_availability", lambda: dict(ALL_OK))
    monkeypatch.setattr(fu, "load_health", lambda: {})
    monkeypatch.setattr(fu, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(fu, "_log", lambda r: None)
    monkeypatch.setattr(fu, "call_model", _stub_call({
        "nim-inkling": "draft A", "free-deepseek": "draft B", "nim-llama": "draft C"}))
    out = fu.fuse("[NOVEL] short task", conduct=False)
    assert out["answer"] == "draft A"           # aggregator nim-inkling stubbed -> its content
    assert out["receipt"]["degraded"] is False
    assert len(out["receipt"]["proposers"]) >= 2   # early-trigger returns at quorum, not always all
    assert out["receipt"]["est_cost"]["usd_estimate"] == 0.0


def test_fuse_degrades_when_proposers_fail(monkeypatch):
    monkeypatch.setattr(fu, "load_availability", lambda: dict(ALL_OK))
    monkeypatch.setattr(fu, "load_health", lambda: {})
    monkeypatch.setattr(fu, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(fu, "_log", lambda r: None)
    monkeypatch.setattr(fu, "call_model", _stub_call({"nim-inkling": "solo answer"}))
    out = fu.fuse("[NOVEL] short task", escalate=False, conduct=False)   # single round, no auto-escalation
    assert out["receipt"]["degraded"] is True
    assert out["answer"] == "solo answer"


def test_auto_escalates_when_degraded(monkeypatch):
    # easy panel degrades (only 1 proposer ok); hard panel adds co-haiku -> 2 ok -> not degraded
    monkeypatch.setattr(fu, "load_availability", lambda: dict(ALL_OK))
    monkeypatch.setattr(fu, "load_health", lambda: {})
    monkeypatch.setattr(fu, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(fu, "_log", lambda r: None)
    monkeypatch.setattr(fu, "call_model", _stub_call({
        "nim-inkling": "m", "co-haiku": "h", "nim-nemotron": "q",
        "go-glm": "z", "go-kimi": "k"}))   # easy: 1 ok; hard: enough ok + agg (GO models in pool)
    out = fu.fuse("[NOVEL] short task", conduct=False)
    assert out["receipt"]["escalation_path"] == ["easy", "hard"]
    assert out["receipt"]["degraded"] is False


def test_pinned_depth_skips_escalation(monkeypatch):
    monkeypatch.setattr(fu, "load_availability", lambda: dict(ALL_OK))
    monkeypatch.setattr(fu, "load_health", lambda: {})
    monkeypatch.setattr(fu, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(fu, "_log", lambda r: None)
    monkeypatch.setattr(fu, "call_model", _stub_call({"nim-inkling": "m"}))  # everything degrades
    out = fu.fuse("[NOVEL DEEP] task", conduct=False)                   # pinned -> no escalation attempts
    assert out["receipt"]["escalation_path"] == ["deep"]


def test_research_requires_confirm(monkeypatch):
    monkeypatch.setattr(fu, "_log", lambda r: None)
    out = fu.fuse("[NOVEL RESEARCH] what is new")
    assert out["receipt"]["confirmed"] is False
    assert "Pro Search" in out["answer"]


def test_copilot_cost_counted():
    cfg = fu.load_config()
    entries = [{"alias": "co-sonnet", "provider": "copilot", "ok": True, "tokens": 500},
               {"alias": "nim-glm", "provider": "nim", "ok": True, "tokens": 400},
               {"alias": "zen-gpt", "provider": "zen", "ok": True, "tokens": 1000}]
    c = fu.estimate_cost(entries, cfg)
    assert c["copilot_credits"] == 1 and c["zen_paid_tokens"] == 1000 and c["free_tokens"] == 400
    assert c["usd_estimate"] > 0


def test_parse_novel_tree_tags():
    assert fu.parse_novel("[NOVEL TREE] solve X") == ("solve X", "tree", None)
    assert fu.parse_novel("[NOVEL TREE DEEP] solve X") == ("solve X", "tree", "deep")
    assert fu.parse_novel("[NOVEL] solve X") == ("solve X", "fuse", None)   # unchanged


def test_fuse_dispatches_tree(monkeypatch):
    import fusion.abmcts as ab
    monkeypatch.setattr(ab, "conduct_abmcts",
                        lambda prompt, depth=None: {"answer": "tree!", "receipt": {"mode": "tree"}})
    out = fu.fuse("[NOVEL TREE] task")
    assert out["answer"] == "tree!"


# --- judges must not think ---------------------------------------------------------------------
# A judge that gets a thinking budget injected inherits that provider's thinking support. go-glm
# has none: it 400s on the router's thinking params. The judge call then fails, returns None, is
# excluded from the mean, and a two-judge bias control silently collapses onto the baseline's own
# family judge -- without a single visible number changing. This cost a full bench run on
# 2026-07-10. Judges emit a two-integer verdict; they gain little from thinking and must not
# depend on it, in either judging path.

def _budget(alias, tier):
    from priority_router import _think_budget
    t = _think_budget(alias, tier, False)
    return t[1] if t else 0


def test_bench_judge_tier_injects_no_thinking_for_any_judge():
    fb = importlib.import_module("fusion.frontier_bench")
    judges = [j for j in (CFG["bench"]["judge"], CFG["bench"].get("judge_b")) if j]
    for j in judges:
        assert _budget(j, fb.JUDGE_TIER) == 0, (
            f"{j} gets a thinking budget at tier {fb.JUDGE_TIER!r}; judges must not think")
    assert fb.JUDGE_MAX_TOKENS > 0


def test_bench_judge_tier_is_pinned_not_inferred_from_prompt():
    """The instrument must not be configured by the thing it measures."""
    import inspect
    fb = importlib.import_module("fusion.frontier_bench")
    src = inspect.getsource(fb.judge)
    assert "force_tier=JUDGE_TIER" in src


def test_abmcts_judge_cap_exceeds_budget_at_its_pinned_tier():
    """reward.judge_pairwise pins force_tier='cheap', which must mean no thinking for any judge
    in the fallback chain -- otherwise judge_max_tokens=300 truncates and every node takes the
    neutral 0.5 reward, degrading the tree search to random sampling."""
    cap = CFG["abmcts"]["judge_max_tokens"]
    chain = CFG["abmcts"]["judge"]
    chain = chain if isinstance(chain, list) else [chain]
    for j in chain:
        assert _budget(j, "cheap") == 0, f"{j} thinks at tier 'cheap'; cap {cap} would truncate it"
