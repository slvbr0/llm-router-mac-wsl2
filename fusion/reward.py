"""Pairwise blind judge + Elo-style ratings in [0,1] for AB-MCTS node rewards (v2.1.1).
Pairwise comparison kills absolute-score drift and self-bias (bench lessons)."""
import random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fusion.fusion import call_model  # noqa: E402

JUDGE_PROMPT = ("Two answers to the same task. Which is better overall (correctness, depth, "
                'clarity)? Output ONLY a JSON object, nothing else: {{"winner": "A"}} or '
                '{{"winner": "B"}}.\n\nTASK:\n{task}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}\n\nJSON:')
VERDICT_RE = re.compile(r'\{[^{}]*"winner"\s*:\s*"([AB])"[^{}]*\}')


def parse_verdict(raw):
    m = VERDICT_RE.findall(raw or "")
    return m[-1] if m else None                        # last block — reasoning models think first


def judge_pairwise(task, challenger, incumbent, cfg, key, sink=None):
    """Blind A/B: returns True if challenger wins, False if it loses, None if unjudgeable.
    Random order so the judge can't position-bias. `judge` config may be a fallback LIST —
    a single judge is a single point of failure that blinds the whole search (live smoke:
    co-haiku throttled -> every rating stuck at 0.5). Retry each judge once (spec §4.3).

    `sink`: every judge call is appended here, including failed attempts and retries. The search
    cannot be costed from its generations alone — it spends one pairwise judge per challenger,
    each carrying two full answers as input, which is the same order of magnitude as the
    generation spend it is measuring. Leaving these out understates the tree's cost by ~70%."""
    a_cfg = cfg["abmcts"]
    judges = a_cfg["judge"] if isinstance(a_cfg["judge"], list) else [a_cfg["judge"]]
    swap = random.random() < 0.5
    a, b = (incumbent, challenger) if swap else (challenger, incumbent)
    for alias in judges:
        for _ in range(2):                             # try + one retry per judge
            # Pin the tier. This prompt happens to auto-classify as `general` today (no thinking),
            # which is the only reason judge_max_tokens=300 works. But classification reads the
            # TASK text: a reasoning-flavoured task tips the judge to REASON, buying nim-kimi a
            # 32768-token thinking budget against a 300-token cap. It would truncate mid-think on
            # every call, judge_pairwise would return None every time, every node would take the
            # neutral 0.5 reward, and the tree search would silently degrade to random sampling
            # while still emitting a full, healthy-looking receipt.
            r = call_model(alias,
                           [{"role": "user", "content": JUDGE_PROMPT.format(
                               task=task, a=a[:6000], b=b[:6000])}],
                           key, a_cfg["judge_timeout_s"], a_cfg["judge_max_tokens"],
                           force_tier="cheap")
            if sink is not None:
                sink.append(r)                         # billed whether or not it produced a verdict
            v = parse_verdict(r["content"]) if r["ok"] else None
            if v is not None:
                challenger_letter = "B" if swap else "A"
                return v == challenger_letter
    return None                                        # skip update — no reward noise


class Ratings:
    """Elo-style online ratings squashed to [0,1]. New nodes start at 0.5 (spec §4.3)."""

    def __init__(self, k=0.1):
        self.k, self.r = k, {}

    def add(self, node_id):
        self.r.setdefault(node_id, 0.5)

    def get(self, node_id):
        return self.r[node_id]

    def record_win(self, winner_id, loser_id):
        rw_, rl = self.r[winner_id], self.r[loser_id]
        expected = 1.0 / (1.0 + 10 ** ((rl - rw_) / 0.4))   # 0.4 = 400 Elo pts on [0,1] scale
        self.r[winner_id] = min(1.0, rw_ + self.k * (1.0 - expected))
        self.r[loser_id] = max(0.0, rl - self.k * (1.0 - expected))
