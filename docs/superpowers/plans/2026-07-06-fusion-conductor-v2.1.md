# Fusion Conductor v2.1 (AB-MCTS-lite) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conductor that scores each committee synthesis and runs a greedy adaptive width-vs-depth search (deepen while improving, widen when stalled, return best-so-far) — capturing Sakana AB-MCTS's key edge without full Thompson sampling.

**Architecture:** New `fusion/conductor.py` holds the conductor pass (synthesize + self-score + gaps in one structured call), a greedy action selector, deepen/widen steps that reuse v2.0's `build_panel`/fan-out, and the `conduct()` search loop. `fusion.py`'s `fuse()` calls `conduct()` when `conductor.enabled`. Everything is stdlib + the existing helpers.

**Tech Stack:** Python 3.12 (stdlib + pyyaml), Phase-1 router `:4040`, pytest. No new deps.

**Spec:** [../specs/2026-07-06-fusion-conductor-v2.1-design.md](../specs/2026-07-06-fusion-conductor-v2.1-design.md)
**Project root:** `<repo-root>` = `$PROJ`. Router up for integration steps (`llmr-start up`).

---

## File structure

| File | Responsibility |
|---|---|
| `fusion/conductor.py` | conductor pass, action selector, deepen/widen, `conduct()` loop (NEW) |
| `fusion/fusion.py` | `fuse()` routes to `conduct()` when enabled; export `_fuse_once`, `build_panel`, `call_model` |
| `fusion/fusion.yaml` | `conductor` block |
| `fusion/fusion_bench.py` | `--conduct` flag on the fusion arm |
| `tests/test_conductor.py` | conductor + search-loop unit tests (NEW) |

---

## Task 1: conductor config + parse/score tests

**Files:**
- Modify: `$PROJ/fusion/fusion.yaml`
- Create: `$PROJ/tests/test_conductor.py`

- [ ] **Step 1: Add the `conductor` block to `fusion/fusion.yaml`** (append at end)

```yaml
conductor:
  enabled: true                # [NOVEL] uses the conductor loop by default (self-limits by score)
  models: {easy: nim-mistral, hard: nim-qwen-max, deep: cop-opus}
  score_threshold: 8           # stop once best score >= this
  max_rounds: 3                # candidate evaluations beyond round 0 (cost cap)
  improve_epsilon: 1           # score gain that counts as "improved" -> keep deepening
  refine_max_tokens: 900
```

- [ ] **Step 2: Write failing tests** `tests/test_conductor.py` (conftest already stubs litellm):

```python
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
cd = importlib.import_module("fusion.conductor")


def test_parse_conductor_output_from_messy_text():
    raw = ('Here is my review.\n{"score": 6, "gaps": ["missing base case", "no complexity"], '
           '"answer": "The final synthesized answer here."}\nDone.')
    out = cd.parse_conductor(raw)
    assert out["score"] == 6
    assert out["answer"] == "The final synthesized answer here."
    assert "missing base case" in out["gaps"]


def test_parse_conductor_unparseable_returns_none_score():
    out = cd.parse_conductor("no json at all, just prose")
    assert out["score"] is None
    assert out["answer"] == "no json at all, just prose"      # fall back to raw text


def test_next_action_deepen_when_improved():
    # last candidate improved best by >= epsilon -> keep deepening
    hist = [{"action": "init", "score": 5}, {"action": "deepen", "score": 7}]
    assert cd.next_action(hist, improve_epsilon=1) == "deepen"


def test_next_action_widen_when_stalled():
    hist = [{"action": "init", "score": 6}, {"action": "deepen", "score": 6}]
    assert cd.next_action(hist, improve_epsilon=1) == "widen"


def test_next_action_widen_when_regressed():
    hist = [{"action": "init", "score": 7}, {"action": "deepen", "score": 4}]
    assert cd.next_action(hist, improve_epsilon=1) == "widen"
```

- [ ] **Step 3: Run — verify fail**

Run: `cd "$PROJ" && python3 -m pytest tests/test_conductor.py -q`
Expected: ERROR (`fusion.conductor` not found).

- [ ] **Step 4: Commit**

```bash
cd "$PROJ" && git add fusion/fusion.yaml tests/test_conductor.py
git commit -m "test: conductor config + parse/action specs (failing)"
```

---

## Task 2: conductor.py — parse + action selector

**Files:**
- Create: `$PROJ/fusion/conductor.py`

- [ ] **Step 1: Write the parse + action-selector core** of `fusion/conductor.py`

```python
"""Conductor (AB-MCTS-lite): score each committee synthesis and run a greedy adaptive
width-vs-depth search. Deepen while improving, widen when stalled, return best-so-far.
Full Thompson-sampling AB-MCTS is v2.1.1 (see spec §9)."""
import json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fusion.fusion import (build_panel, call_model, estimate_cost, provider_of,   # noqa: E402
                           load_env, load_config, load_availability, load_health,
                           difficulty_of, parse_novel, _log)
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

CONDUCTOR_PROMPT = (
    "You are the CONDUCTOR of a model committee. Given a TASK and the committee's DRAFTS, do "
    "three things and return ONLY a JSON object (no other text):\n"
    "1. Write the single best final answer (reconcile drafts, fix errors, fill gaps).\n"
    "2. Score that answer 1-10 for correctness, depth, completeness.\n"
    "3. List concrete GAPS still missing or wrong (empty list if none).\n"
    'Output: {{"answer": "<final answer>", "score": <1-10>, "gaps": ["<gap>", ...]}}\n\n'
    "TASK:\n{task}\n\nDRAFTS:\n{drafts}\n\nJSON:")


def parse_conductor(raw):
    """Extract {answer, score, gaps} from possibly-messy model text. Tolerant."""
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"answer": str(d.get("answer", "")).strip() or raw.strip(),
                    "score": int(d["score"]) if str(d.get("score", "")).strip().isdigit() else None,
                    "gaps": [str(g) for g in d.get("gaps", []) if str(g).strip()]}
        except Exception:
            pass
    return {"answer": raw.strip(), "score": None, "gaps": []}


def next_action(history, improve_epsilon):
    """Greedy width/depth: deepen while the last step improved best by >= epsilon; else widen."""
    if len(history) < 2:
        return "deepen"
    best_before = max(h["score"] or 0 for h in history[:-1])
    last = history[-1]["score"] or 0
    return "deepen" if last - best_before >= improve_epsilon else "widen"
```

- [ ] **Step 2: Run parse/action tests — pass**

Run: `cd "$PROJ" && python3 -m pytest tests/test_conductor.py -q`
Expected: 5 PASS.

- [ ] **Step 3: Commit**

```bash
cd "$PROJ" && git add fusion/conductor.py
git commit -m "feat: conductor parse + greedy width/depth action selector"
```

---

## Task 3: conductor pass + committee steps (stubbed tests)

**Files:**
- Modify: `$PROJ/fusion/conductor.py`
- Modify: `$PROJ/tests/test_conductor.py` (append)

- [ ] **Step 1: Append `conductor_pass`, `_committee`, `deepen`, `widen` to `fusion/conductor.py`**

```python
def _committee(task, difficulty, cfg, key, availability, health, exclude=None):
    """Run one committee fan-out for `task`; return list of good drafts (dicts)."""
    panel = build_panel(difficulty, cfg, availability, health)
    proposers = [a for a in panel["proposers"] if a not in (exclude or set())] or panel["proposers"]
    tmo = cfg["panels"][difficulty].get("timeout_s", cfg["proposer_timeout_s"])
    ptok = cfg["proposer_max_tokens"]
    msgs = [{"role": "user", "content": task}]
    with ThreadPoolExecutor(max_workers=len(proposers)) as ex:
        drafts = list(ex.map(lambda a: call_model(a, msgs, key, tmo, ptok), proposers))
    return [d for d in drafts if d["ok"] and d["content"].strip()]


def conductor_pass(task, drafts, model, key, cfg, availability, health):
    """One conductor call: synthesize + self-score + gaps over the given drafts."""
    tmo = cfg["proposer_timeout_s"]
    atok = cfg["conductor"]["refine_max_tokens"]
    dtxt = "\n\n".join(f"--- DRAFT {i+1} ({d['alias']}) ---\n{d['content']}"
                       for i, d in enumerate(drafts)) or "(no drafts)"
    r = call_model(model, [{"role": "user",
                            "content": CONDUCTOR_PROMPT.format(task=task, drafts=dtxt)}],
                   key, tmo, atok)
    parsed = parse_conductor(r["content"] if r["ok"] else "")
    parsed["_call"] = r
    return parsed


def deepen(task, best, difficulty, model, cfg, key, availability, health):
    """Refine: committee re-answers targeting the gaps, conductor re-synthesizes."""
    gaps = "; ".join(best.get("gaps", [])) or "improve rigor, completeness, and correctness"
    refine_task = (f"{task}\n\nA previous answer was:\n{best['answer']}\n\n"
                   f"Fix these gaps and improve it: {gaps}")
    drafts = _committee(refine_task, difficulty, cfg, key, availability, health)
    return conductor_pass(task, drafts, model, key, cfg, availability, health)


def widen(task, difficulty, model, cfg, key, availability, health, used_models):
    """Fresh independent drafts (rotate proposers away from already-used), new synthesis."""
    drafts = _committee(task, difficulty, cfg, key, availability, health, exclude=used_models)
    return conductor_pass(task, drafts, model, key, cfg, availability, health)
```

- [ ] **Step 2: Append stubbed tests** to `tests/test_conductor.py`

```python
import fusion.fusion as fu


def _stub(results):
    def fake(alias, messages, key, timeout, max_tokens=None):
        c = results.get(alias, "")
        return {"alias": alias, "provider": fu.provider_of(alias), "ok": bool(c),
                "content": c, "tokens": 10 if c else 0, "latency_ms": 1}
    return fake


def test_conductor_pass_parses_score(monkeypatch):
    monkeypatch.setattr(cd, "call_model", lambda *a, **k: {
        "alias": "nim-qwen-max", "provider": "nim", "ok": True, "tokens": 20, "latency_ms": 1,
        "content": '{"answer": "synth", "score": 9, "gaps": []}'})
    cfg = cd.load_config()
    out = cd.conductor_pass("task", [{"alias": "nim-glm", "content": "d1"}],
                            "nim-qwen-max", "k", cfg, {"nim": True}, {})
    assert out["score"] == 9 and out["answer"] == "synth"


def test_committee_returns_only_good(monkeypatch):
    monkeypatch.setattr(cd, "build_panel", lambda *a, **k: {
        "proposers": ["nim-mistral", "zen-free-deepseek", "nim-llama"], "aggregator": "nim-mistral"})
    monkeypatch.setattr(cd, "call_model", _stub({"nim-mistral": "A", "zen-free-deepseek": "B"}))
    cfg = cd.load_config()
    good = cd._committee("t", "easy", cfg, "k", {"nim": True, "zen": True, "copilot": True}, {})
    assert len(good) == 2 and {g["alias"] for g in good} == {"nim-mistral", "zen-free-deepseek"}
```

- [ ] **Step 3: Run — pass**

Run: `cd "$PROJ" && python3 -m pytest tests/test_conductor.py -q`
Expected: 7 PASS.

- [ ] **Step 4: Commit**

```bash
cd "$PROJ" && git add fusion/conductor.py tests/test_conductor.py
git commit -m "feat: conductor pass + committee/deepen/widen steps"
```

---

## Task 4: conduct() search loop

**Files:**
- Modify: `$PROJ/fusion/conductor.py`
- Modify: `$PROJ/tests/test_conductor.py` (append)

- [ ] **Step 1: Append the `conduct()` loop to `fusion/conductor.py`**

```python
def conduct(prompt, depth=None, escalate_diff=None):
    """Full conductor search over a [NOVEL] prompt. Returns {answer, receipt}."""
    t0 = time.time()
    cleaned, _mode, tag_depth = parse_novel(prompt)
    cfg = load_config()
    ccfg = cfg["conductor"]
    env = load_env(); key = env.get("LITELLM_MASTER_KEY", "")
    availability, health = load_availability(), load_health()
    difficulty = depth or tag_depth or escalate_diff or difficulty_of(cleaned)
    model = ccfg["models"].get(difficulty, "nim-qwen-max")
    thr, maxr, eps = ccfg["score_threshold"], ccfg["max_rounds"], ccfg["improve_epsilon"]

    # round 0
    drafts0 = _committee(cleaned, difficulty, cfg, key, availability, health)
    used = {d["alias"] for d in drafts0}
    c0 = conductor_pass(cleaned, drafts0, model, key, cfg, availability, health)
    history = [{"round": 0, "action": "init", "model": model, "score": c0["score"] or 0}]
    calls = [d for d in drafts0] + [c0["_call"]]
    best = {"answer": c0["answer"], "score": c0["score"] or 0, "round": 0, "action": "init"}

    rnd = 0
    while rnd < maxr and best["score"] < thr:
        rnd += 1
        act = next_action(history, eps)
        if act == "deepen":
            cand = deepen(cleaned, best, difficulty, model, cfg, key, availability, health)
        else:
            cand = widen(cleaned, difficulty, model, cfg, key, availability, health, used)
        calls.append(cand["_call"])
        cscore = cand["score"] or 0
        history.append({"round": rnd, "action": act, "model": model, "score": cscore})
        if cscore > best["score"]:
            best = {"answer": cand["answer"], "score": cscore, "round": rnd, "action": act}
        # converged: a widen that didn't improve -> stop
        if act == "widen" and cscore <= history[-2]["score"]:
            break

    entries = [c for c in calls if c.get("alias")]
    receipt = {"mode": "fuse", "conductor": True, "difficulty": difficulty,
               "rounds": rnd, "best_round": best["round"], "best_score": best["score"],
               "search_path": history,
               "total_tokens": sum(c.get("tokens", 0) for c in entries),
               "est_cost": estimate_cost(entries, cfg),
               "wall_ms": int((time.time() - t0) * 1000)}
    _log(receipt)
    return {"answer": best["answer"], "receipt": receipt}
```

- [ ] **Step 2: Append loop tests** to `tests/test_conductor.py`

```python
def _seq_conductor(scores):
    """conductor_pass stub returning a scripted sequence of scores."""
    it = iter(scores)

    def fake(task, drafts, model, key, cfg, availability, health):
        s = next(it)
        return {"answer": f"ans{s}", "score": s, "gaps": ["g"] if s < 8 else [],
                "_call": {"alias": model, "provider": "nim", "ok": True, "tokens": 5}}
    return fake


def _noop_committee(*a, **k):
    return [{"alias": "nim-mistral", "provider": "nim", "ok": True, "content": "d", "tokens": 5}]


def test_conduct_stops_at_threshold(monkeypatch):
    monkeypatch.setattr(cd, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(cd, "load_availability", lambda: {"nim": True, "zen": True, "copilot": True})
    monkeypatch.setattr(cd, "load_health", lambda: {})
    monkeypatch.setattr(cd, "_log", lambda r: None)
    monkeypatch.setattr(cd, "_committee", _noop_committee)
    monkeypatch.setattr(cd, "conductor_pass", _seq_conductor([9]))     # round 0 already >= 8
    out = cd.conduct("[NOVEL] task")
    assert out["receipt"]["rounds"] == 0
    assert out["receipt"]["best_score"] == 9


def test_conduct_returns_best_not_last(monkeypatch):
    # scores: init 6, deepen 8 (best), deepen? no -> next_action after 8>6 deepens, gets 4 (regress)
    monkeypatch.setattr(cd, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(cd, "load_availability", lambda: {"nim": True, "zen": True, "copilot": True})
    monkeypatch.setattr(cd, "load_health", lambda: {})
    monkeypatch.setattr(cd, "_log", lambda r: None)
    monkeypatch.setattr(cd, "_committee", _noop_committee)
    monkeypatch.setattr(cd, "conductor_pass", _seq_conductor([6, 8, 4]))
    # thresholds: default threshold 8 -> after reaching 8 at round1 it stops (>=8). best=8.
    out = cd.conduct("[NOVEL] task")
    assert out["receipt"]["best_score"] == 8
    assert out["answer"] == "ans8"
```

- [ ] **Step 3: Run — pass**

Run: `cd "$PROJ" && python3 -m pytest tests/test_conductor.py -q`
Expected: 9 PASS.

- [ ] **Step 4: Commit**

```bash
cd "$PROJ" && git add fusion/conductor.py tests/test_conductor.py
git commit -m "feat: conduct() adaptive width/depth search loop with best-so-far"
```

---

## Task 5: wire conduct() into fuse()

**Files:**
- Modify: `$PROJ/fusion/fusion.py`

- [ ] **Step 1: Route `fuse()` to the conductor when enabled.** In `fusion/fusion.py`, change the
`fuse` signature and add an early branch right after the `research` check.

Find:
```python
def fuse(prompt, mode=None, depth=None, confirm_research=False, escalate=True):
```
Replace with:
```python
def fuse(prompt, mode=None, depth=None, confirm_research=False, escalate=True, conduct=None):
```
Then, immediately after this block:
```python
    if mode == "research":
        return _research(cleaned, cfg, t0, confirm_research)
```
insert:
```python
    use_conduct = cfg.get("conductor", {}).get("enabled", False) if conduct is None else conduct
    if use_conduct:
        from fusion.conductor import conduct as _conduct
        return _conduct(prompt, depth=depth)
```

- [ ] **Step 2: Run the whole suite — nothing regresses**

Run: `cd "$PROJ" && python3 -m pytest tests/ -q`
Expected: all PASS (existing fusion tests still green; conductor tests green). Note: existing
`test_fusion.py` stub tests call `fuse(..., escalate=...)` without conductor — they pass
`conduct=False` implicitly only if config disables it. To keep those unit tests deterministic,
they must bypass the conductor: update the three `fuse(...)` calls in `tests/test_fusion.py`
(`test_fuse_happy_path_aggregates`, `test_fuse_degrades_when_proposers_fail`,
`test_auto_escalates_when_degraded`, `test_pinned_depth_skips_escalation`) to pass `conduct=False`.

Apply that edit:
```python
# in each of those tests, change the fuse(...) call to add conduct=False, e.g.:
out = fu.fuse("[NOVEL] short task", conduct=False)
```

- [ ] **Step 3: Re-run — all green**

Run: `cd "$PROJ" && python3 -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
cd "$PROJ" && git add fusion/fusion.py tests/test_fusion.py
git commit -m "feat: fuse() routes to conductor when conductor.enabled; unit tests pin conduct=False"
```

---

## Task 6: live integration (router up)

- [ ] **Step 1: Refresh health, run a real conductor fuse via CLI**

```bash
cd "$PROJ" && sh scripts/nim_health.sh >/dev/null 2>&1
python3 -m fusion.fusion "[NOVEL] Prove the sum of the first n odd numbers equals n squared, rigorously."
```
Expected: a synthesized answer on stdout; stderr receipt shows `mode=fuse` with a multi-round
search. Then inspect the search path:
```bash
tail -1 logs/fusion-*.jsonl | python3 -c 'import sys,json;r=json.load(sys.stdin);print("rounds",r.get("rounds"),"best_score",r.get("best_score"),"path",[(h["action"],h["score"]) for h in r.get("search_path",[])])'
```
Expected: a `search_path` like `[("init",6),("deepen",8)]` (or a `widen` if a deepen stalled),
`best_score` = the max in the path.

- [ ] **Step 2: Confirm best-so-far + cost in the receipt**

Verify the printed answer corresponds to the highest-scored round and `est_cost` reflects all
rounds. If the conductor consistently fails to score (unparseable), switch the hard conductor
model in `fusion.yaml` (`conductor.models.hard`) to `nim-mistral` (non-reasoning → cleaner JSON)
and re-run. Commit any config change:
```bash
cd "$PROJ" && git add fusion/fusion.yaml && git commit -m "fix: conductor model choice for clean scoring" || echo "no change"
```

---

## Task 7: harness --conduct + honest re-bench

**Files:**
- Modify: `$PROJ/fusion/fusion_bench.py`

- [ ] **Step 1: Add a `--conduct` flag** to `fusion/fusion_bench.py`. Change the arg parser and the
fusion call.

Find:
```python
    ap.add_argument("prompts"); ap.add_argument("--baseline"); ap.add_argument("--judge")
    args = ap.parse_args()
```
Replace with:
```python
    ap.add_argument("prompts"); ap.add_argument("--baseline"); ap.add_argument("--judge")
    ap.add_argument("--conduct", action="store_true", help="use the v2.1 conductor loop")
    args = ap.parse_args()
```
Find:
```python
        f = fuse("[NOVEL] " + p, escalate=False)            # single round — clean apples-to-apples
```
Replace with:
```python
        f = fuse("[NOVEL] " + p, escalate=False, conduct=args.conduct)
```

- [ ] **Step 2: Smoke it on ONE prompt (only when NIM health looks good)**

```bash
cd "$PROJ" && sh scripts/nim_health.sh
# proceed only if most NIM models are ok with low latency
printf 'Implement an LRU cache with TTL in Python without functools; include complexity analysis.\n' > /tmp/b1.txt
python3 fusion/fusion_bench.py /tmp/b1.txt --conduct 2>&1 | tail -8
```
Expected: a table row with fus/base scores (not `-`), and a FUSION vs BASELINE summary. If it
prints `BENCH INVALID` or skips the row, the free tier is flaky right now — stop and retry later.

- [ ] **Step 3: Commit**

```bash
cd "$PROJ" && git add fusion/fusion_bench.py
git commit -m "feat: fusion_bench --conduct (v2.1 conductor arm)"
```

---

## Task 8: document + push

- [ ] **Step 1: Update README "Status & roadmap"** — replace the "Next (v2.1 — designed)" line with:

```markdown
**Phase 2 v2.1 (conductor / AB-MCTS-lite — BUILT):** `[NOVEL]` runs a conductor that scores each
synthesis and searches adaptively — deepen while improving, **widen when stalled** (Sakana's edge),
return best-so-far. Self-limits: a strong round-0 answer costs ~1 round. `fusion/conductor.py`;
receipt shows the search path. Bench with `python3 fusion/fusion_bench.py <prompts> --conduct`
(run on a healthy NIM window).

**Next (v2.1.1 — designed):** the real AB-MCTS — Thompson sampling over width/depth + full tree +
multi-LLM bandit. Spec §9.
```

- [ ] **Step 2: Full suite green + push**

```bash
cd "$PROJ" && python3 -m pytest tests/ -q
git add -A && git commit -m "docs: v2.1 conductor built + roadmap"
git push origin main
```
Expected: all tests pass; pushed.

---

## Self-review (spec coverage)

- §3 architecture / §4.5 search loop → Task 4 (`conduct`) ✅
- §4.1 conductor pass → Task 3 ✅ · §4.2 action selector → Task 2 ✅ · §4.3 deepen/widen → Task 3 ✅
- §4.4 best-so-far → Task 4 (`best` tracked; `test_conduct_returns_best_not_last`) ✅
- §5 config → Task 1 ✅ · §6 receipt (search_path/best_round/cost) → Task 4 ✅
- §7 error handling (unparseable → None score, treated as no-improvement; committee fail → best stands) → Tasks 2,3,4 ✅
- §8 testing (parse, action, best-so-far, stop-at-threshold/K, integration, harness) → Tasks 1-4,6,7 ✅
- §9 v2.1.1 intentionally NOT built (deferred) ✅ · §10 files (conductor.py new) → all ✅
- §11 cost (max_rounds cap, free conductor where healthy) → Tasks 1,4 ✅
