# Phase 2 v2.1 — Conductor / AB-MCTS-lite · Design Spec

**Date:** 2026-07-06
**Builds on:** Phase-2 v2.0 fusion engine (this repo). Extends it; v2.0 stays intact.
**Author:** Claude Code (brainstorming session, approved by salva).

---

## 1. Goal

v2.0 fan-out (committee → one-shot synthesis) proved the base but is single-round. v2.1 adds a
**conductor** — a capable model that scores the synthesis and drives a **search** toward
Fable-class quality, spending more compute **only when the answer is actually weak**.

North star (unchanged): **Fable-5 quality at a fraction of the cost.**

**Why AB-MCTS-lite, not a linear refine loop:** Sakana's own result is that *sequential
refinement deepens a potentially misguided path*; their edge is the ability to **widen** (spin up
a fresh idea) when refinement stalls. v2.1 adopts that one idea with a **greedy** width-vs-depth
rule (not full Thompson sampling — that's v2.1.1). So v2.1 = Mixture-of-Agents + adaptive
width/depth search + best-so-far, a faithful-in-spirit, buildable slice of AB-MCTS.

## 2. Non-goals (v2.1)

Full Bayesian tree search · Thompson sampling · per-model multi-armed-bandit selection ·
learned coordination — all deferred to **v2.1.1** (§9), the true AB-MCTS.

## 3. Architecture

```
[NOVEL] → round 0: committee fan-out (v2.0) → CONDUCTOR synthesizes + self-scores
                                              {answer, score 1-10, gaps[]}
          best ← round0
          │
   loop while rounds < max_rounds and best.score < threshold:
      action = DEEPEN  if last step improved score (or was init)      # promising → refine it
             = WIDEN   if last step stalled/regressed                 # stuck → fresh idea, don't refine a loser
      candidate = DEEPEN → committee re-answers (task + best.answer + gaps) → conductor synthesize+score
                  WIDEN  → committee fresh drafts (rotated models)     → conductor synthesize+score
      if candidate.score > best.score: best ← candidate
   return best.answer + receipt(search_path, best_round, best_score)
```

Return the **highest-scored** candidate across the search, not the last (a refine can regress).
The conductor is the v2.0 aggregator **promoted**: one structured call does synthesize **and**
self-critique, so no separate judge call.

## 4. Components

New file **`fusion/conductor.py`** (keeps `fusion.py` focused on panel/fan-out primitives):

### 4.1 Conductor
`conductor_pass(task, drafts, model, key, cfg) -> {answer, score, gaps}`. One call: given the task
+ the current committee drafts, produce the best synthesis, a self-score (1–10), and a short list
of concrete gaps/errors ("drift"). Structured output parsed with a tolerant regex (like the bench
judge). Model per difficulty from config; health-gated with fallback.

### 4.2 Action selector (greedy width/depth)
`next_action(history) -> "deepen" | "widen"`: after round 0, **deepen** while the last candidate
improved the best score by ≥ `improve_epsilon`; **widen** the first time a step stalls/regresses.
If a widen also fails to improve, stop (converged). This is the greedy stand-in for Thompson
sampling.

### 4.3 Deepen / Widen steps
- **Deepen:** refine prompt = original task + current best answer + its gaps → committee
  re-answers **targeting the gaps** → conductor synthesizes+scores → new candidate.
- **Widen:** fresh committee drafts on the original task with **rotated proposers** (prefer models
  not yet used / whose drafts scored high — a lightweight multi-LLM nudge, not a bandit) →
  conductor synthesizes+scores an **independent** candidate.

### 4.4 Best-so-far tracker
Keep `{answer, score, round, action}` for the max-scored candidate; that is the returned answer.

### 4.5 Search loop
`conduct(task, difficulty, cfg, key, availability, health) -> {answer, receipt}` orchestrates
round 0 + the loop, calling v2.0's `build_panel` / fan-out for each committee step.

## 5. Config (`fusion.yaml` — new `conductor` block)

```yaml
conductor:
  enabled: true                # [NOVEL] uses the conductor loop by default (self-limits by score)
  models: {easy: nim-mistral, hard: nim-qwen-max, deep: cop-opus}
  score_threshold: 8           # stop once best score >= this
  max_rounds: 3                # candidate evaluations beyond round 0 (cost cap)
  improve_epsilon: 1           # score gain that counts as "improved" -> keep deepening
  refine_max_tokens: 900
```

`enabled: false` (or `fuse(..., conduct=False)`) falls back to pure v2.0 single-round — used by
the A/B harness's "fusion" arm can opt in, and for cheap/latency-sensitive paths.

## 6. Receipt additions

```
conductor: true, rounds: <int>, best_round: <int>, best_score: <int>,
search_path: [{round, action: init|deepen|widen, model, score}],
total_tokens, est_cost (summed across all rounds), wall_ms
```
Appended to `logs/fusion-*.jsonl`. Makes the search visible: when it widened, when it stopped, and
the true multi-round cost.

## 7. Error handling

- Conductor output unparseable → treat score as the previous best (no false improvement), return
  current best, stop the loop. Never crash.
- A committee step (deepen/widen) fully fails → that candidate is skipped; best-so-far stands.
- All v2.0 rules per step: health-gate, free+paid mixing, per-panel timeout, degrade-to-single.
- Hard cap `max_rounds` bounds worst-case cost (~1 + max_rounds committee fan-outs).

## 8. Testing

- **Unit (stubbed conductor + call_model):**
  - conductor output parse (`{answer, score, gaps}` from messy text).
  - action selection: deepen when improved, widen when stalled/regressed.
  - **best-so-far returned even when the last round regresses** (the key AB-MCTS property).
  - stop at `score_threshold`; stop at `max_rounds`.
  - receipt `search_path` records each round's action + score.
- **Integration (router up):** one `[NOVEL]` conduct run returns an answer with a multi-round
  receipt; a deliberately hard prompt shows ≥1 widen or deepen.
- **Harness:** extend `fusion_bench.py` with `--conduct` so the A/B compares conductor-fusion vs
  baseline (run when NIM free-tier is healthy).

## 9. v2.1.1 — the REAL AB-MCTS (next; WILL be needed)

Replace the greedy heuristics with the faithful algorithm:
- **Thompson Sampling** over width-vs-depth: maintain Bayesian posteriors of the payoff of
  "new solution" vs "refine this node" at each node; sample to choose — the actual AB-MCTS.
- **Full search tree** (not just best-so-far + linear candidates): nodes = solutions, expansion
  balanced by the posteriors; return the best leaf.
- **Multi-LLM bandit:** a separate probability model per LLM (nim/zen/copilot aliases), Thompson
  selection of *which model* generates the next node, updated by realized reward on the branch —
  "more promising models become increasingly likely" (Sakana Multi-LLM AB-MCTS).
- Requires a reliable **node scorer** (LLM-judge) and reward calibration; more state + tuning.
Built after v2.1-lite validates the cost/quality win on the harness. Ref: Sakana "Wider or
Deeper?" (arXiv 2503.04412) + Multi-LLM AB-MCTS.

## 10. Files

| File | Responsibility |
|---|---|
| `fusion/conductor.py` | conductor pass, action selector, deepen/widen, search loop (NEW) |
| `fusion/fusion.py` | `fuse()` calls `conduct()` when `conductor.enabled`; panel/fan-out primitives unchanged |
| `fusion/fusion.yaml` | `conductor` block |
| `fusion/fusion_bench.py` | `--conduct` flag on the fusion arm |
| `tests/test_fusion.py` | conductor + search-loop unit tests |

## 11. Security / cost discipline

Bounded by `max_rounds`; self-limits when round 0 already scores well. Every round's cost is in
the receipt. Reuses Phase-1 master key + `127.0.0.1` router; no new secrets. Conductor prefers
free capable models (`nim-qwen-max`) where healthy, paid frontier (`cop-opus`) only for `deep`.
