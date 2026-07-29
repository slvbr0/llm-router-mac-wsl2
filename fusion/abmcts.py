"""Multi-LLM AB-MCTS (v2.1.1) — faithful Sakana AB-MCTS-A + per-alias bandit.
Tree of solution nodes; every node has a Gaussian value posterior plus a GEN arm
("make a new child here"). Thompson sampling picks child-vs-GEN at each level:
GEN at root = fresh answer (wider), GEN at a node = refine it (deeper).
Ref: arXiv 2503.04412. Spec: docs/superpowers/specs/2026-07-06-abmcts-v2.1.1-design.md"""
import math, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fusion.fusion import (call_model, estimate_cost, load_availability,  # noqa: E402
                           load_config, load_env, load_health, difficulty_of,
                           parse_novel, _log)
from fusion.bandit import Bandit, build_pool  # noqa: E402
from fusion.reward import Ratings, judge_pairwise  # noqa: E402

GEN_PRIOR_MEAN = 0.6                                # optimistic — drives exploration (spec §4.1)


class Node:
    def __init__(self, node_id, parent, answer, alias):
        self.id, self.parent = node_id, parent
        self.answer, self.alias = answer, alias
        self.children = []
        self.critique = ""                           # judge's note on why it lost (refine fuel)
        self.sum_r, self.n_obs = 0.0, 0              # value posterior (mean, n)
        self.gen_arm = None                          # lazily seeded from cfg in _sample_gen

    def observe(self, reward):
        self.sum_r += reward
        self.n_obs += 1

    def value_mean(self):
        return self.sum_r / self.n_obs if self.n_obs else 0.5


def _sample(mean, n, sigma0):
    return mean + random.gauss(0, 1) * sigma0 / math.sqrt(max(n, 1))


def _sample_gen(node, a_cfg):
    if node.gen_arm is None:                         # optimistic prior, n0 pseudo-obs
        n0 = a_cfg["prior_weight"]
        node.gen_arm = [GEN_PRIOR_MEAN * n0, n0]
    s, n = node.gen_arm
    return _sample(s / n, n, a_cfg["sigma0"])


def select(root, a_cfg):
    """Thompson walk: at each node sample every child's value + the GEN arm; max wins.
    Returns (node, "gen") when GEN wins there, else descends into the winning child."""
    node = root
    while True:
        gen_s = _sample_gen(node, a_cfg)
        best_child, best_s = None, gen_s
        for c in node.children:
            cs = _sample(c.value_mean(), c.n_obs, a_cfg["sigma0"])
            if cs > best_s:
                best_child, best_s = c, cs
        if best_child is None:
            return node, "gen"
        node = best_child


def backprop(node, reward):
    """Fold reward into the node and every ancestor's posterior; also credit the
    GEN arm of the parent that spawned this node."""
    n = node
    while n is not None:
        n.observe(reward)
        n = n.parent
    if node.parent is not None and node.parent.gen_arm is not None:
        node.parent.gen_arm[0] += reward
        node.parent.gen_arm[1] += 1


FRESH_PROMPT = ("Answer the TASK as well as you can. Be rigorous, complete, and concrete.\n\n"
                "TASK:\n{task}")
REFINE_PROMPT = ("Improve the CURRENT ANSWER to the TASK. Fix the WEAKNESS if given; deepen "
                 "rigor and completeness. Output only the improved answer.\n\n"
                 "TASK:\n{task}\n\nCURRENT ANSWER:\n{answer}\n\nWEAKNESS:\n{critique}")

import re as _re
_TAG_RE = _re.compile(r"^\s*\[[^\]]*\]\s*")


def _strip_tag(prompt):
    return _TAG_RE.sub("", prompt).strip()


def generate(alias, node, task, key, a_cfg):
    """One model call: fresh answer at the root, refinement at a solution node."""
    if node.answer is None:                            # root GEN -> wider
        content = FRESH_PROMPT.format(task=task)
    else:                                              # node GEN -> deeper
        content = REFINE_PROMPT.format(task=task, answer=node.answer,
                                       critique=node.critique or "none given")
    r = call_model(alias, [{"role": "user", "content": content}],
                   key, a_cfg["gen_timeout_s"], a_cfg["gen_max_tokens"])
    # A healthy alias echoes itself in response.model; on a 4xx litellm silently serves a
    # different model and still returns 200. Counting that as a win for `alias` teaches the
    # bandit that a dead arm is strong, and bills its tokens to the wrong model. The arm did
    # not answer, so it does not score.
    if r["ok"] and r.get("served_model") and r["served_model"] != alias:
        r["ok"] = False
        r["error"] = f"silent fallback: served {r['served_model']!r}, expected {alias!r}"
    return r


def _slim_calls(calls):
    """Per-call token accounting for cost analysis, without dragging the generated text along."""
    return [{"alias": c["alias"], "ok": c["ok"], "tokens": c.get("tokens", 0),
             "tok_in": c.get("tok_in", 0), "tok_out": c.get("tok_out", 0)} for c in calls]


def conduct_abmcts(prompt, depth=None):
    """[NOVEL TREE] entry point. Returns {answer, receipt} (spec §4/§5)."""
    t0 = time.time()
    cleaned, _, tag_depth = parse_novel(prompt)
    if cleaned.startswith("["):                        # pre-Task-6 tags or unknown brackets
        cleaned = _strip_tag(cleaned)
    cfg = load_config(); a_cfg = cfg["abmcts"]
    difficulty = depth or tag_depth or difficulty_of(cleaned)
    budget = a_cfg["budget"][difficulty]
    env = load_env(); key = env.get("LITELLM_MASTER_KEY", "")
    availability, health = load_availability(), load_health()

    pool = build_pool(cfg, availability, health)
    bandit = Bandit(pool, cfg)
    ratings = Ratings(k=a_cfg["elo_k"])
    root = Node(0, parent=None, answer=None, alias=None)
    nodes, calls = [], []
    # Judge calls are kept apart from `calls` so the generation logic below (bandit updates,
    # failure handling) stays about generations only -- but they are real spend and are folded
    # into total_tokens, est_cost and the returned per-call breakdown.
    judge_calls = []
    best = None
    defenses = 0                                       # consecutive challenges best survived

    for _step in range(1, budget + 1):
        node, _action = select(root, a_cfg)
        alias = bandit.pick()
        g = generate(alias, node, cleaned, key, a_cfg)
        calls.append(g)
        if not g["ok"] or not g["content"].strip():
            bandit.update(alias, 0.0)                  # failed pull — the arm learns (spec §4.2)
            continue
        child = Node(len(nodes) + 1, parent=node, answer=g["content"], alias=alias)
        node.children.append(child)
        nodes.append(child)
        ratings.add(child.id)

        if best is None:                               # first node seeds the incumbent
            best = child
            backprop(child, 0.5)
            bandit.update(alias, 0.5)
            continue

        won = judge_pairwise(cleaned, child.answer, best.answer, cfg, key, sink=judge_calls)
        if won is None:                                # unjudgeable after retry — skip update
            backprop(child, 0.5)
            bandit.update(alias, 0.5)
            continue
        if won:
            ratings.record_win(child.id, best.id)
            child.critique = ""
            best, defenses = child, 0
            reward = 1.0
        else:
            ratings.record_win(best.id, child.id)
            child.critique = ("an independent judge preferred another answer "
                              "for correctness/depth")
            defenses += 1
            reward = 0.0
        backprop(child, reward)
        bandit.update(alias, reward)
        if defenses >= a_cfg["early_stop_wins"]:       # incumbent dominant — stop (spec §4.5)
            break

    if best is None:
        billed = calls + judge_calls
        receipt = {"mode": "tree", "difficulty": difficulty, "budget": budget,
                   "generations": 0, "degraded": True, "tree": [], "bandit": {},
                   "best_node": None, "best_rating": None,
                   "total_tokens": sum(c["tokens"] for c in billed),
                   "judge_calls": len(judge_calls),
                   "est_cost": estimate_cost(billed, cfg),
                   "wall_ms": int((time.time() - t0) * 1000)}
        _log(receipt)
        return {"answer": "[NOVEL TREE] degraded: no successful generations "
                          "(all arms failed — check provider availability/health).",
                "receipt": receipt, "calls": _slim_calls(billed)}

    billed = calls + judge_calls                       # the search's true spend: generate + judge
    receipt = {"mode": "tree", "difficulty": difficulty, "budget": budget,
               "generations": len(nodes), "degraded": False,
               "tree": [{"id": n.id, "parent": n.parent.id, "alias": n.alias,
                         "action": "fresh" if n.parent.id == 0 else "refine",
                         "rating": round(ratings.get(n.id), 3)} for n in nodes],
               "bandit": bandit.stats(),
               "best_node": best.id, "best_rating": round(ratings.get(best.id), 3),
               "total_tokens": sum(c["tokens"] for c in billed),
               "judge_calls": len(judge_calls),
               "est_cost": estimate_cost(billed, cfg),
               "wall_ms": int((time.time() - t0) * 1000)}
    _log(receipt)
    return {"answer": best.answer, "receipt": receipt, "calls": _slim_calls(billed)}


if __name__ == "__main__":
    res = conduct_abmcts(" ".join(sys.argv[1:]) or "[NOVEL TREE] say hello")
    print(res["answer"])
    r = res["receipt"]
    print(f"\n--- receipt: gen={r['generations']}/{r['budget']} best={r.get('best_rating')} "
          f"tokens={r['total_tokens']} cost={r['est_cost']} wall={r['wall_ms']//1000}s")
    print(f"    tree={r['tree']}\n    bandit={r['bandit']}")
