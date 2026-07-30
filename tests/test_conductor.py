import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
cd = importlib.import_module("fusion.conductor")


def test_parse_conductor_output_from_messy_text():
    raw = ("Here is my review.\nSCORE: 6\nGAPS: missing base case, no complexity\n"
           "ANSWER:\nThe final synthesized answer here.")
    out = cd.parse_conductor(raw)
    assert out["score"] == 6
    assert out["answer"] == "The final synthesized answer here."
    assert "missing base case" in out["gaps"]


def test_parse_conductor_unparseable_returns_none_score():
    out = cd.parse_conductor("no json at all, just prose")
    assert out["score"] is None
    assert out["answer"] == "no json at all, just prose"


def test_next_action_deepen_when_improved():
    hist = [{"action": "init", "score": 5}, {"action": "deepen", "score": 7}]
    assert cd.next_action(hist, improve_epsilon=1) == "deepen"


def test_next_action_widen_when_stalled():
    hist = [{"action": "init", "score": 6}, {"action": "deepen", "score": 6}]
    assert cd.next_action(hist, improve_epsilon=1) == "widen"


def test_next_action_widen_when_regressed():
    hist = [{"action": "init", "score": 7}, {"action": "deepen", "score": 4}]
    assert cd.next_action(hist, improve_epsilon=1) == "widen"


import fusion.fusion as fu


def _stub(results):
    def fake(alias, messages, key, timeout, max_tokens=None):
        c = results.get(alias, "")
        return {"alias": alias, "provider": fu.provider_of(alias), "ok": bool(c),
                "content": c, "tokens": 10 if c else 0, "latency_ms": 1}
    return fake


def test_conductor_pass_parses_score(monkeypatch):
    monkeypatch.setattr(cd, "call_model", lambda *a, **k: {
        "alias": "nim-nemotron", "provider": "nim", "ok": True, "tokens": 20, "latency_ms": 1,
        "content": "SCORE: 9\nGAPS: none\nANSWER:\nsynth"})
    cfg = cd.load_config()
    out = cd.conductor_pass("task", [{"alias": "nim-glm", "content": "d1"}],
                            "nim-nemotron", "k", cfg, {"nim": True}, {})
    assert out["score"] == 9 and out["answer"] == "synth"


def test_committee_returns_only_good(monkeypatch):
    monkeypatch.setattr(cd, "build_panel", lambda *a, **k: {
        "proposers": ["nim-inkling", "zen-free-deepseek", "nim-llama"], "aggregator": "nim-inkling"})
    # _committee delegates to fan_out (in fusion.fusion) -> patch fu.call_model
    monkeypatch.setattr(fu, "call_model", _stub({"nim-inkling": "A", "zen-free-deepseek": "B"}))
    cfg = cd.load_config()
    good = cd._committee("t", "easy", cfg, "k", {"nim": True, "zen": True, "copilot": True}, {})
    assert len(good) >= 2 and {"nim-inkling", "zen-free-deepseek"} <= {g["alias"] for g in good}


def _seq_conductor(scores):
    it = iter(scores)

    def fake(task, drafts, model, key, cfg, availability, health):
        s = next(it)
        return {"answer": f"ans{s}", "score": s, "gaps": ["g"] if s < 8 else [],
                "_call": {"alias": model, "provider": "nim", "ok": True, "tokens": 5}}
    return fake


def _noop_committee(*a, **k):
    return [{"alias": "nim-inkling", "provider": "nim", "ok": True, "content": "d", "tokens": 5}]


def test_conduct_stops_at_threshold(monkeypatch):
    monkeypatch.setattr(cd, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(cd, "load_availability", lambda: {"nim": True, "zen": True, "copilot": True})
    monkeypatch.setattr(cd, "load_health", lambda: {})
    monkeypatch.setattr(cd, "_log", lambda r: None)
    monkeypatch.setattr(cd, "_committee", _noop_committee)
    monkeypatch.setattr(cd, "conductor_pass", _seq_conductor([9]))
    out = cd.conduct("[NOVEL] task")
    assert out["receipt"]["rounds"] == 0
    assert out["receipt"]["best_score"] == 9


def test_conduct_returns_best_not_last(monkeypatch):
    monkeypatch.setattr(cd, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(cd, "load_availability", lambda: {"nim": True, "zen": True, "copilot": True})
    monkeypatch.setattr(cd, "load_health", lambda: {})
    monkeypatch.setattr(cd, "_log", lambda r: None)
    monkeypatch.setattr(cd, "_committee", _noop_committee)
    monkeypatch.setattr(cd, "conductor_pass", _seq_conductor([6, 8, 4]))
    out = cd.conduct("[NOVEL] task")
    assert out["receipt"]["best_score"] == 8
    assert out["answer"] == "ans8"


def test_conductor_pass_falls_back_to_best_draft_when_answer_empty(monkeypatch):
    # conductor emits SCORE but no ANSWER body -> fall back to longest draft, never empty
    monkeypatch.setattr(cd, "call_model", lambda *a, **k: {
        "alias": "nim-nemotron", "provider": "nim", "ok": True, "tokens": 20, "latency_ms": 1,
        "content": "SCORE: 7\nGAPS: none\nANSWER:"})     # empty answer body
    cfg = cd.load_config()
    drafts = [{"alias": "nim-glm", "content": "short"},
              {"alias": "nim-inkling", "content": "a much longer and better draft answer"}]
    out = cd.conductor_pass("task", drafts, "nim-nemotron", "k", cfg, {"nim": True}, {})
    assert out["score"] == 7
    assert out["answer"] == "a much longer and better draft answer"
