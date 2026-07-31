import importlib, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
bd = importlib.import_module("fusion.bandit")
fu = importlib.import_module("fusion.fusion")

CFG = fu.load_config()
ALL_OK = {"nim": True, "zen": True, "copilot": True}


def test_pool_gates_unhealthy_nim_but_keeps_paid():
    health = {"nim-inkling": {"ok": False}}
    pool = bd.build_pool(CFG, ALL_OK, health)
    assert "nim-inkling" not in pool
    assert "go-glm" in pool and "co-opus" in pool     # paid always in


def test_pool_respects_availability_mask():
    pool = bd.build_pool(CFG, {"nim": True, "zen": False, "copilot": True}, {})
    assert all(fu.provider_of(a) != "zen" for a in pool)


def test_prior_mean_by_rank():
    assert bd.prior_mean("co-opus", CFG) == 0.65       # frontier
    assert bd.prior_mean("go-glm", CFG) == 0.55        # strong
    assert bd.prior_mean("nim-inkling", CFG) == 0.45    # base
    assert bd.prior_mean("free-pickle", CFG) == 0.45  # unlisted -> base prior


def test_cold_start_prefers_frontier_prior():
    random.seed(7)
    b = bd.Bandit(["co-opus", "nim-inkling"], CFG)
    picks = [b.pick() for _ in range(200)]
    assert picks.count("co-opus") > picks.count("nim-inkling")


def test_bandit_learns_from_reward():
    random.seed(7)
    b = bd.Bandit(["co-opus", "nim-inkling"], CFG)
    for _ in range(6):                                   # base-ranked arm keeps winning
        b.update("nim-inkling", 1.0)
        b.update("co-opus", 0.0)
    picks = [b.pick() for _ in range(200)]
    assert picks.count("nim-inkling") > picks.count("co-opus")


def test_stats_snapshot():
    b = bd.Bandit(["go-glm"], CFG)
    b.update("go-glm", 0.8)
    s = b.stats()
    assert s["go-glm"]["pulls"] == 1
    assert 0.5 < s["go-glm"]["mean_reward"] < 0.7      # prior-blended mean
