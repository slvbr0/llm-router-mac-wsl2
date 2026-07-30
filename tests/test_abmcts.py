import importlib, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
ab = importlib.import_module("fusion.abmcts")
fu = importlib.import_module("fusion.fusion")
rw = importlib.import_module("fusion.reward")   # binds call_model at import; patch it separately

CFG = fu.load_config()
A = CFG["abmcts"]


def test_node_posterior_updates():
    n = ab.Node(0, parent=None, answer=None, alias=None)
    n.observe(0.8); n.observe(0.6)
    assert abs(n.value_mean() - 0.7) < 1e-9
    assert n.n_obs == 2


def test_backprop_reaches_ancestors():
    root = ab.Node(0, parent=None, answer=None, alias=None)
    child = ab.Node(1, parent=root, answer="a", alias="zen-glm")
    root.children.append(child)
    leaf = ab.Node(2, parent=child, answer="b", alias="nim-glm")
    child.children.append(leaf)
    ab.backprop(leaf, 0.9)
    assert leaf.n_obs == 1 and child.n_obs == 1 and root.n_obs == 1
    assert root.value_mean() == 0.9


def test_select_descends_into_strong_child():
    random.seed(3)
    root = ab.Node(0, parent=None, answer=None, alias=None)
    strong = ab.Node(1, parent=root, answer="s", alias="zen-glm")
    root.children.append(strong)
    for _ in range(8):
        ab.backprop(strong, 0.95)                     # strong child, tight posterior
    root.gen_arm = [0.1 * A["prior_weight"], A["prior_weight"]]   # pessimistic GEN at root
    picks = {ab.select(root, A)[0].id for _ in range(50)}
    assert 1 in picks                                  # descends into the strong child


def test_select_gen_fires_when_children_weak():
    random.seed(3)
    root = ab.Node(0, parent=None, answer=None, alias=None)
    weak = ab.Node(1, parent=root, answer="w", alias="nim-gptoss")
    root.children.append(weak)
    for _ in range(8):
        ab.backprop(weak, 0.05)                       # stalled child
    node, action = None, None
    for _ in range(50):
        node, action = ab.select(root, A)
        if action == "gen" and node.id == 0:
            break
    assert action == "gen" and node.id == 0            # widen at root wins eventually


def _gen_stub(answers_by_alias):
    """fake call_model: judge alias returns challenger-wins JSON; others return canned text."""
    def fake(alias, messages, key, timeout, max_tokens=None, force_tier=None):
        if alias in A["judge"]:                       # judge is a fallback list
            return {"alias": alias, "provider": "copilot", "ok": True,
                    "content": '{"winner": "A"}', "tokens": 5, "latency_ms": 1}
        c = answers_by_alias.get(alias, "generic draft answer")
        return {"alias": alias, "provider": fu.provider_of(alias), "ok": bool(c),
                "content": c, "tokens": 50 if c else 0, "latency_ms": 1}
    return fake


def _patch_env(monkeypatch):
    monkeypatch.setattr(ab, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(ab, "load_availability",
                        lambda: {"nim": True, "zen": True, "copilot": True})
    monkeypatch.setattr(ab, "load_health", lambda: {})
    monkeypatch.setattr(ab, "_log", lambda r: None)


def test_conduct_abmcts_returns_best_and_receipt(monkeypatch):
    random.seed(11)
    _patch_env(monkeypatch)
    stub = _gen_stub({"zen-glm": "a strong answer"})
    monkeypatch.setattr(ab, "call_model", stub)
    import fusion.reward as rw
    monkeypatch.setattr(rw, "call_model", stub)
    out = ab.conduct_abmcts("[NOVEL TREE] design a thing")
    r = out["receipt"]
    assert out["answer"].strip()
    assert r["mode"] == "tree"
    assert r["generations"] >= 1 and r["generations"] <= r["budget"]
    assert r["best_rating"] >= 0.5
    assert isinstance(r["tree"], list) and isinstance(r["bandit"], dict)
    assert "est_cost" in r and "wall_ms" in r


def test_conduct_abmcts_empty_generations_degrade(monkeypatch):
    random.seed(11)
    _patch_env(monkeypatch)
    def all_fail(alias, messages, key, timeout, max_tokens=None, force_tier=None):
        return {"alias": alias, "provider": fu.provider_of(alias), "ok": False,
                "content": "", "tokens": 0, "latency_ms": 1, "error": "down"}
    monkeypatch.setattr(ab, "call_model", all_fail)
    import fusion.reward as rw
    monkeypatch.setattr(rw, "call_model", all_fail)
    out = ab.conduct_abmcts("[NOVEL TREE] task")
    assert out["receipt"]["degraded"] is True
    assert out["receipt"]["generations"] == 0
    assert "no successful generations" in out["answer"]


def test_early_stop_after_k_defenses(monkeypatch):
    random.seed(11)
    _patch_env(monkeypatch)
    calls = {"n": 0}
    def stub(alias, messages, key, timeout, max_tokens=None, force_tier=None):
        calls["n"] += 1
        return {"alias": alias, "provider": fu.provider_of(alias), "ok": True,
                "content": f"answer {calls['n']}", "tokens": 50, "latency_ms": 1}
    monkeypatch.setattr(ab, "call_model", stub)
    # incumbent always defends: challenger never wins (deterministic, no blind-swap ambiguity)
    monkeypatch.setattr(ab, "judge_pairwise", lambda *a, **k: False)
    out = ab.conduct_abmcts("[NOVEL TREE DEEP] task")
    r = out["receipt"]
    assert r["generations"] < r["budget"]              # early stop fired
    assert r["generations"] <= 1 + A["early_stop_wins"] + 1


def test_tree_cost_includes_pairwise_judge_calls(monkeypatch):
    """The search spends one pairwise judge per challenger, each carrying two full answers as
    input -- comparable to the generation spend it measures. Costing the tree from generations
    alone understated arm B by ~70% and would have made fusion look cheaper than the baseline
    it actually ties on price with."""
    random.seed(11)
    _patch_env(monkeypatch)
    stub = _gen_stub({"zen-glm": "a strong answer"})
    monkeypatch.setattr(ab, "call_model", stub)     # generate() binds call_model in abmcts
    monkeypatch.setattr(rw, "call_model", stub)     # judge_pairwise binds it in reward
    res = ab.conduct_abmcts("[NOVEL TREE] design a thing")
    r = res["receipt"]
    assert r["judge_calls"] > 0, "no judge call was recorded"
    gen_aliases = {c["alias"] for c in res["calls"]}
    judged = [c for c in res["calls"] if c["alias"] in A["judge"]]
    assert judged, f"judge calls missing from cost breakdown; saw aliases {gen_aliases}"
    assert r["total_tokens"] >= sum(c["tokens"] for c in res["calls"] if c["ok"])
