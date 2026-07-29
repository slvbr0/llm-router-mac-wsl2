import importlib, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
rw = importlib.import_module("fusion.reward")
fu = importlib.import_module("fusion.fusion")

CFG = fu.load_config()


def test_parse_verdict_last_json_wins():
    raw = 'thinking... {"winner": "A"} more... {"winner": "B"}'
    assert rw.parse_verdict(raw) == "B"


def test_parse_verdict_unparseable_returns_none():
    assert rw.parse_verdict("no json here") is None


def test_elo_win_raises_loss_lowers_bounded():
    t = rw.Ratings(k=0.1)
    t.add("n1"); t.add("n2")
    r1, r2 = t.get("n1"), t.get("n2")
    t.record_win("n1", "n2")
    assert t.get("n1") > r1 and t.get("n2") < r2
    for _ in range(200):                              # hammer -> stays in [0,1]
        t.record_win("n1", "n2")
    assert 0.0 <= t.get("n2") <= t.get("n1") <= 1.0


def test_first_node_seeds_half():
    t = rw.Ratings(k=0.1)
    t.add("n1")
    assert t.get("n1") == 0.5


def _judge_stub(winner_json):
    def fake(alias, messages, key, timeout, max_tokens=None, force_tier=None):
        return {"alias": alias, "provider": "copilot", "ok": True,
                "content": winner_json, "tokens": 10, "latency_ms": 1}
    return fake


def test_judge_pairwise_covers_both_blind_orders(monkeypatch):
    # stub always says "A wins"; with random A/B swap, over 20 trials the challenger
    # must land on both sides -> both True and False outcomes appear
    random.seed(1)
    monkeypatch.setattr(rw, "call_model", _judge_stub('{"winner": "A"}'))
    results = {rw.judge_pairwise("task", "challenger", "incumbent", CFG, "k")
               for _ in range(20)}
    assert True in results and False in results


def test_judge_unparseable_returns_none(monkeypatch):
    monkeypatch.setattr(rw, "call_model", _judge_stub("garbage no json"))
    assert rw.judge_pairwise("task", "a", "b", CFG, "k") is None


def test_judge_falls_back_when_first_judge_down(monkeypatch):
    # first judge in the chain fails; a later one answers -> verdict still produced
    judges = CFG["abmcts"]["judge"]
    assert isinstance(judges, list) and len(judges) >= 2
    def fake(alias, messages, key, timeout, max_tokens=None, force_tier=None):
        if alias == judges[0]:                        # primary judge down
            return {"alias": alias, "provider": "copilot", "ok": False,
                    "content": "", "tokens": 0, "latency_ms": 1, "error": "throttled"}
        return {"alias": alias, "provider": "zen", "ok": True,
                "content": '{"winner": "A"}', "tokens": 10, "latency_ms": 1}
    monkeypatch.setattr(rw, "call_model", fake)
    assert rw.judge_pairwise("task", "a", "b", CFG, "k") is not None
