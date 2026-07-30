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
    assert "zen-glm" in pool and "cop-opus" in pool     # paid always in


def test_pool_respects_availability_mask():
    pool = bd.build_pool(CFG, {"nim": True, "zen": False, "copilot": True}, {})
    assert all(fu.provider_of(a) != "zen" for a in pool)


def test_prior_mean_by_rank():
    assert bd.prior_mean("cop-opus", CFG) == 0.65       # frontier
    assert bd.prior_mean("zen-glm", CFG) == 0.55        # strong
    assert bd.prior_mean("nim-inkling", CFG) == 0.45    # base
    assert bd.prior_mean("zen-free-pickle", CFG) == 0.45  # unlisted -> base prior


def test_cold_start_prefers_frontier_prior():
    random.seed(7)
    b = bd.Bandit(["cop-opus", "nim-inkling"], CFG)
    picks = [b.pick() for _ in range(200)]
    assert picks.count("cop-opus") > picks.count("nim-inkling")


def test_bandit_learns_from_reward():
    random.seed(7)
    b = bd.Bandit(["cop-opus", "nim-inkling"], CFG)
    for _ in range(6):                                   # base-ranked arm keeps winning
        b.update("nim-inkling", 1.0)
        b.update("cop-opus", 0.0)
    picks = [b.pick() for _ in range(200)]
    assert picks.count("nim-inkling") > picks.count("cop-opus")


def test_stats_snapshot():
    b = bd.Bandit(["zen-glm"], CFG)
    b.update("zen-glm", 0.8)
    s = b.stats()
    assert s["zen-glm"]["pulls"] == 1
    assert 0.5 < s["zen-glm"]["mean_reward"] < 0.7      # prior-blended mean
