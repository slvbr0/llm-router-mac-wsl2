# Fusion Benchmarks — results & caveats

All A/B runs: blind pairwise-ish judging (fusion answer vs baseline answer, random order),
scores 1–10, unparseable judge outputs excluded. Logs in `logs/bench*.txt`.

## Bench 1 — greedy conductor vs Copilot Opus (2026-07-06, n=4)

`python3 fusion/fusion_bench.py fusion/prompts.bench4.txt --conduct` · judge co-haiku

| Prompt | Fusion (greedy) | co-opus |
|---|---|---|
| Monotone-convergence proof | **10** | 3 |
| Rate limiter design | **9** | 7 |
| LRU+TTL code | 7 | **9** |
| MoA explanation | **8** | 7 |

```
FUSION   avg 8.50   est cost $0.08   $/point 0.0094
BASELINE avg 6.50   est cost $0.16   $/point 0.0246   (co-opus)
LATENCY  fusion 55s vs baseline 635s
→ hypothesis SUPPORTED
```

Caveats: n=4, single judge, baseline prompt-1 likely truncated near its 6000-token cap
(inflates one row). NIM was 6/11 healthy → fusion cost pessimistic. Signal, not proof.
Gotcha discovered: co-opus hides extended thinking that eats the token budget — a
1500-token cap returned *empty* answers; the harness now uses 6000 tokens + 1200s
(wait it out, never truncate mid-think). Log: `logs/bench4-conduct-final-194009.txt`.

## Bench 2 — Task 9 3-way: TREE vs greedy vs go-glm (2026-07-07, n=12)

`fusion/prompts.bench12.txt` (code-weighted + math + design + concept) · baseline
**go-glm** (GLM 5.2 via GO subscription, the strongest flat-rate single model) ·
judge **nim-kimi** (independent provider — no self-judging).

| Engine | Avg quality | Scored | Latency/prompt |
|---|---|---|---|
| **`[NOVEL TREE]` (Multi-LLM AB-MCTS, v2.1.1)** | **9.10** | 10/12* | ~18 min |
| go-glm single-shot | 8.40 / 8.33 | — | ~2 min |
| `[NOVEL]` greedy conductor (v2.1) | 7.92 | 12/12 | ~3 min |

\* 2 skips = router restarts during the run (fallback/timeout fixes), not the engine.

Tree vs single: **6W / 3T / 1L** — and the tree **won the code category** (LRU 10–8,
queue 9–8), which was greedy's weak spot in Bench 1. Greedy's 7.92 includes one
catastrophic 1/10 (URL shortener — bad round); without it ≈8.5, i.e. ~tied with the
single model.

**Verdict: the real AB-MCTS is the champion** — beats the strongest single subscription
model by +0.7 at ~$0 marginal cost (free NIM + GO flat-rate), paying ~10× latency
(deliberate mode, not everyday).

Caveats: the harness prices any baseline as Copilot credits — the printed baseline `$`
is wrong for GO/Zen baselines; the quality columns are the comparison. Judge = one
model (nim-kimi). Log: `logs/bench12-3way-go-175729.txt`.

## Bench 3 — frontier baseline resolved: fable-5 vs fusion (+ audit) (2026-07-10, n=15)

This is the run the old "PENDING" section below was waiting on. Three arms, blind
pairwise, answer order randomised, two independent judges, both scored all 15 prompts
(full coverage — no skips). n=15 = bench4+bench12 deduped.

- **A = ant-fable single-shot, HIGH extended thinking** (the target — the literal
  frontier baseline).
- **B = `[NOVEL TREE]` fusion committee.**
- **C = B + one ant-opus audit pass.**
- Judges: **ant-haiku** and **go-glm** (cross-family, no self-judging on either
  side). Auditor for arm C: **ant-opus**.

```
A (fable)        avg 8.92   cost $0.128 (exact)          latency 35s
B (fusion)       avg 8.67   cost $0.325–0.356 (bounds)    latency 469s (13×)
C (fusion+audit) avg 8.40   cost $0.458–0.489 (bounds)    latency 524s
→ hypothesis REJECTED — quality tie-to-behind, provably 2.5–3.6× dearer, 13× slower
```

### Quality (cross-judge mean, 1–10)

| Arm | ant-haiku | go-glm | cross-judge mean |
|---|---|---|---|
| A — fable | 8.77 | 9.07 | **8.92** |
| B — fusion | 9.07 | 8.27 | 8.67 |
| C — fusion+audit | 8.73 | 8.07 | 8.40 |

Judges disagree on *direction* — ant-haiku prefers fusion, go-glm prefers fable. No
family bias as predicted (each judge doesn't favor its own family).

### Head-to-head (30 judge-prompt pairs = 15 prompts × 2 judges)

| Matchup | W | L | T |
|---|---|---|---|
| B (fusion) vs A (fable) — all | 12 | 10 | 8 |
| C (fusion+audit) vs A (fable) — all | 7 | 13 | 10 |
| B vs A — **hard** (n=8, 16 pairs) | **8** | 3 | 5 |
| B vs A — **easy** (n=7, 14 pairs) | 4 | **7** | 3 |

fusion B vs fable A is a statistical tie overall, but it's not uniform: fusion **wins
hard prompts** and **loses easy prompts**. C loses outright, overall and net.

Difficulty means: easy → A=8.86, B=8.36, C=8.36. hard → A=8.97, B=8.94, C=8.44 (fusion
reaches near-fable quality on hard prompts, falls off on easy ones).

### Cost (list-price equivalent, per prompt)

Marginal cost is ~$0 on every arm (free NIM + GO flat-rate + Max flat-rate); these are
counterfactual public-API prices, thinking billed as output — same convention as
Bench 1/2.

- A = **$0.128** (fully priced, exact).
- B = point est **$0.356**; **provable bounds lo=$0.325, hi=$0.356**.
- C = point est **$0.489**; bounds lo=$0.458, hi=$0.489.

**Provable result: arm B is NOT cheaper than fable.** Even billing every unpriced
token at $0 (a true lower bound), B's floor $0.325 is still > A's $0.128. No missing
price can flip this. → B ≈ 2.5× dearer, C ≈ 3.6× dearer.

Caveat: 164,810 fusion tokens (~33% of B) had no published price. The point estimate
bills them at GLM 5.2 rates, which is **not** a strict upper bound (go-qwen-max lists
above it) — that's why the "not cheaper" claim above cites the *lower* bound, not the
point estimate.

### Latency

A = 35s/prompt. B = 469s (**13× slower**). C = 524s.

### Committee mix (111 total generations across 15 prompts)

By provider: **Anthropic 51%** (ant-opus 27% + ant-sonnet 24%), **opencode-GO 38%**,
**NIM 11%**. NIM is low because most NIM models were degraded/timing out during the
run — only nim-deepseek/mistral/minimax/nemotron stayed alive, nim-kimi 404'd
throughout.

### Conclusions

1. **The hypothesis "match fable 5 at lower cost" fails on both axes.** Fusion is a
   quality tie-to-slightly-behind AND provably 2.5–3.6× more expensive AND 13× slower.
2. **The cost failure is structural, not tunable.** The bandit rationally converges on
   ant-opus/ant-sonnet (the strongest arms available to it), so a committee that's
   allowed to include paid frontier models becomes "run opus/sonnet many times and
   pick the best" — which cannot cost less than one fable call by construction.
3. **The ant-opus audit pass (arm C) makes things worse, not better.** C scores below
   the un-audited committee (8.40 vs 8.67) and loses to fable 7–13. Recommendation:
   drop the audit.

### When would you actually use it?

Real advantages, but narrow — do not oversell:

1. **Redundancy when fable/Anthropic is unavailable.** On hard tasks the committee
   reaches fable-class quality (8W-3L-5T, mean 8.94 vs 8.97) by drawing on three
   independent provider families (NIM + GO + Anthropic sonnet/opus). If Anthropic is
   down, rate-limited, or you have no Max plan, the committee degrades gracefully to
   NIM+GO instead of failing outright — this happened mid-benchmark when the OAuth
   token expired.
2. **Preserving scarce frontier quota.** Offloading hard tasks to the committee
   (which leans on GO + sonnet + NIM) spends Max/GO flat-rate budget instead of
   burning fable's own rate limits — useful on plans with 5h reset windows.
3. **Self-verified answers on hard problems.** The AB-MCTS explores multiple
   independent approaches and pairwise-judges them, so on hard prompts it beats
   single-shot more often than it loses. When correctness matters more than cost or
   latency and you want a second-opinion / adversarially-checked answer, that's a
   feature a single fable call doesn't have.

When NOT to use it: easy/routine tasks (it overthinks and loses, 4W-7L), anything
cost-sensitive (provably dearer), anything latency-sensitive (13× slower), and never
run the arm-C audit pass.

Log: `logs/frontier-bench16-193521.json` (+ `.err`).

Also invalidated along the way (kept for honesty): two bench12 attempts died on
baseline infrastructure — Copilot throttling after 11 straight opus calls (now the
harness paces 60s between prompts), then exhausted Copilot credits (402) and an empty
Zen credits balance (401, before the GO endpoint discovery).
