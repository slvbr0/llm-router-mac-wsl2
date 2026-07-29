"""Per-alias Thompson bandit for Multi-LLM AB-MCTS (v2.1.1).
Each arm keeps a Gaussian reward posterior (mean, n) seeded by a capability prior;
pick() Thompson-samples, update() folds in realized reward. Pure stdlib."""
import math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fusion.fusion import provider_of, _ok  # noqa: E402

PRIOR_MEANS = {"frontier": 0.65, "strong": 0.55, "base": 0.45}


def build_pool(cfg, availability, health):
    """Every alias in prior_rank that passes availability + (NIM-only) health gating.
    Paid providers (zen/copilot) skip the health file — they are reliable backends."""
    ranks = cfg["abmcts"]["prior_rank"]
    pool = []
    for aliases in ranks.values():
        for a in aliases:
            if not availability.get(provider_of(a), False):
                continue
            if provider_of(a) == "nim" and not _ok(a, availability, health):
                continue
            pool.append(a)
    return pool


def prior_mean(alias, cfg):
    for rank, aliases in cfg["abmcts"]["prior_rank"].items():
        if alias in aliases:
            return PRIOR_MEANS[rank]
    return PRIOR_MEANS["base"]                     # unlisted -> base prior (spec §7)


class Bandit:
    def __init__(self, pool, cfg):
        a = cfg["abmcts"]
        self.sigma0, n0 = a["sigma0"], a["prior_weight"]
        # arm -> [sum_reward, n]; prior = n0 pseudo-observations at the prior mean
        self.arms = {al: [prior_mean(al, cfg) * n0, n0] for al in pool}
        self.pulls = {al: 0 for al in pool}

    def pick(self):
        best, best_s = None, -1e9
        for al, (s, n) in self.arms.items():
            sample = s / n + random.gauss(0, 1) * self.sigma0 / math.sqrt(n)
            if sample > best_s:
                best, best_s = al, sample
        return best

    def update(self, alias, reward):
        self.arms[alias][0] += reward
        self.arms[alias][1] += 1
        self.pulls[alias] += 1

    def stats(self):
        return {al: {"pulls": self.pulls[al],
                     "mean_reward": round(s / n, 3)}
                for al, (s, n) in self.arms.items() if self.pulls[al] > 0}
