# Phase 2 v2.0 — Fusion / Mixture-of-Agents · Design Spec

**Date:** 2026-07-06
**Builds on:** Phase-1 llm-router (this repo). Does not modify it.
**Author:** Claude Code (brainstorming session, approved by salva).

---

## 1. Goal & hypothesis

**Hypothesis to test:** a committee of cheap models spread across providers, coordinated
and synthesized, reaches **Fable-5-class quality at a fraction of the cost.**

`[NOVEL]` is an opt-in fusion mode. Two engines under one tag family:
- **`[NOVEL]`** — model Mixture-of-Agents for hard reasoning (offline, ~free NIM-heavy panel).
- **`[NOVEL RESEARCH]`** — web-grounded MoA via `pwm council` (Perplexity), for discovery.

v2.0 builds the **single-round** engine + the **A/B harness that measures the hypothesis**.
Multi-round swarm (critique→refine loops) is v2.1 — built only if v2.0 shows the win.

**Success criteria:** on a sample prompt set, the harness shows fusion matching or beating a
single frontier baseline on judged quality at materially lower cost/quality-point — or it
shows it doesn't (a valid, useful negative result).

## 2. Non-goals (v2.0)

Multi-round swarm loops · sidecar proxy · re-pointing opencode · auto-firing without the
`[NOVEL]` tag · streaming fused output · persistent fusion service.

## 3. Architecture

```
opencode ([NOVEL] tag) → agent calls MCP `fuse` tool → fusion.py engine
                                                          │
                              classify difficulty (import priority_router.classify)
                                                          │
                              build provider-diverse panel (reads model_health.yaml)
                                                          │
                     ┌──────────── parallel calls to :4040 ────────────┐
                     ▼            ▼            ▼            ▼
                proposer1     proposer2    proposer3    proposer4     (diverse models/providers)
                     └──────────────── drafts ────────────────┘
                                                          │
                            aggregator model (via :4040) reconciles drafts → final answer
                                                          │
                                    answer + RECEIPT (models, tokens, est cost, latency)
```
`[NOVEL RESEARCH]` bypasses the model panel → calls `pwm council` (web-grounded MoA) and
returns its synthesis + citations, still wrapped in a receipt.

The engine is a **library** (`fusion.py`) exposed two ways: an **MCP `fuse` tool** (agent
auto-calls on the tag) and a **CLI** (used by the A/B harness and for manual runs). No proxy,
no re-pointing — consistent with Phase-1's MCP-centric design.

## 4. Components

### 4.1 Difficulty classifier
Import `classify(prompt)` from Phase-1 `priority_router.py` (single source of truth). Map its
tier to a panel profile: `cheap`/`general` → *easy*; `code`/`reason`/`agent` → *hard*.
`[NOVEL DEEP]` forces *hard/large*; explicit tier tags in the prompt still apply.

### 4.2 Panel builder (provider-diverse, difficulty-aware, health-aware)
Reads `availability.yaml` + `model_health.yaml` (reuse Phase-1 loaders). Picks proposers to
**span providers** (diversity is MoA's edge) and skip down/slow NIM models:

| Profile | Proposers (span NIM / Zen / Copilot) | Aggregator |
|---|---|---|
| easy | `nim-glm`, `zen-free-deepseek`, `nim-kimi` | `nim-glm` |
| hard | `nim-deepseek`, `nim-qwen-max`, `zen-gpt`, `cop-sonnet`, `nim-glm` | `nim-qwen-max` (or `cop-opus` if healthy) |
| research | `pwm council`: `gpt54,claude_sonnet,gemini_pro` + synthesis | pwm chairman |

Rules: prefer free/cheap providers first; include ≥1 non-NIM proposer for diversity when
available; drop any proposer whose provider is masked or whose NIM model is `ok:false`;
require ≥2 healthy proposers (else degrade — §4.6). Panel definitions live in `fusion.yaml`
so they're tunable without code changes.

### 4.3 Fan-out
Parallel POSTs to `http://localhost:4040/v1/chat/completions` (one per proposer), auth with
`LITELLM_MASTER_KEY`. Concurrency = panel size. Per-proposer timeout (default 90s). Because
aliases map to different providers, the fan-out **spreads load/cost across providers** — no
single quota/cost spike. Each returns a draft + token counts; failures/timeouts are dropped.

### 4.4 Aggregator
One healthy strong model (via `:4040`) receives the original prompt + all drafts, with an
aggregation prompt: *reconcile the drafts, resolve disagreements, and produce the single best
answer — do not merely concatenate.* Returns the final answer. Aggregator picked from the
profile, health-gated (falls to next healthy candidate).

### 4.5 Receipt
Every fuse returns `{answer, receipt}`. Receipt:
```
mode, difficulty, proposers:[{alias,provider,tokens,latency_ms,ok}], aggregator:{alias,tokens,latency_ms},
total_tokens, est_cost:{free_tokens, zen_paid_tokens, copilot_credits, pwm_searches, usd_estimate}, wall_ms
```
Cost estimate uses a small table in `fusion.yaml` (NIM=0, zen-free=0, zen-paid=$/1M from
config, copilot=1 credit/call, pwm=1 Pro Search/model). Appended to `logs/fusion-*.jsonl`
so cumulative fusion cost is auditable (and future `/usage` can read it).

### 4.6 Error handling
- **<2 proposers succeed** → fall back to a single best healthy model; return its answer with
  `receipt.degraded=true`. Never fail the user.
- **Aggregator fails** → return the longest/most-complete draft + note in receipt.
- **All NIM down** → panel builder uses Zen/Copilot equivalents (still provider-diverse).
- **Research mode** → `pwm council` costs ~4 Pro Search; the tool **confirms/limits** before
  spending (respects the Phase-1 propose-confirm discipline) and refuses if quota is near zero.

## 5. A/B harness (`fusion_bench.py`) — the hypothesis test

CLI: `python fusion/fusion_bench.py fusion/prompts.sample.txt [--baseline cop-opus]`.
For each prompt:
1. **Fusion path** — run the engine (cheap committee) → answer + receipt.
2. **Baseline path** — one frontier model (`cop-opus` default; configurable) → answer + tokens.
3. **Judge** — a neutral strong model (default `nim-qwen-max`; `--judge` to override) blind-scores
   both answers 1–10 on quality, order randomized, model identity hidden.
Output: per-prompt table + summary — avg fusion quality vs baseline, avg tokens/cost each, and
**cost-per-quality-point** for both. This is the *"Fable at a fraction?"* verdict, with numbers.
Honest reporting: if fusion loses, the harness says so.

## 6. Files

| File | Responsibility |
|---|---|
| `fusion/fusion.py` | engine: classify → panel → fan-out → aggregate → receipt (library + CLI) |
| `fusion/mcp_server.py` | stdio MCP server exposing `fuse(prompt, mode, depth)` |
| `fusion/fusion_bench.py` | A/B harness: fusion vs baseline + judge scoring |
| `fusion/fusion.yaml` | panel profiles, aggregator choices, cost table, baseline/judge defaults |
| `fusion/prompts.sample.txt` | ~8 sample eval prompts (mix of easy/hard/reasoning) |
| `tests/test_fusion.py` | unit tests (stubbed HTTP) |
| `logs/fusion-*.jsonl` | per-run receipts (gitignored) |

Reuses Phase-1 `.env`, `availability.yaml`, `model_health.yaml`, and `priority_router.classify`.

## 7. Wiring into opencode

Add the fusion MCP to `~/.config/opencode/opencode.json` (`mcp.fusion` → `python .../mcp_server.py`).
AGENTS.md rule: when the user's prompt contains `[NOVEL]` / `[NOVEL RESEARCH]` / `[NOVEL DEEP]`,
call the `fuse` tool with the prompt (strip the tag) and return its answer + a one-line receipt
summary. No auto-fire without the tag (opt-in, cost-aware).

## 8. Testing

- **Unit (stubbed HTTP):** panel builder (difficulty→profile, provider diversity, health-skip,
  min-2 rule), receipt/cost math, tag parsing (`[NOVEL]`/`RESEARCH`/`DEEP`).
- **Integration:** one real easy-panel fuse via `:4040` → answer + well-formed receipt.
- **Harness smoke:** 2-prompt run produces a comparison table with scores.

## 9. Roadmap

**v2.1 (next) — the CONDUCTOR pattern (a capable supervisor on top of cheap workers).**
The v2.0 aggregator is a one-shot synthesizer. v2.1 promotes it to a **conductor-critic**: a
strong model that (1) reviews the cheap workers' drafts, (2) detects drift / errors / gaps, and
(3) either synthesizes (drafts good) or issues **targeted feedback** → workers do **one refine
round** on just the weak parts → conductor synthesizes. Cheap-but-efficient models do the bulk
work per difficulty; one capable model keeps them on track with a **bounded token budget** (it
reviews and directs, it does not redo the work). This is the general form the user asked for —
useful whenever a hard/novel prompt makes some models drift, not only in multi-agent workflows.

Design notes:
- **Conductor model = most-capable-that-fits-the-latency-budget** (capable ⇒ reasoning ⇒ slow
  even at low output): `nim-qwen-max` (free, ~30s) for hard; `cop-opus` (frontier) for deep.
- Shaped as **AB-MCTS-lite** (Sakana AI's proven recipe: adaptive width-vs-depth — per step
  either generate a NEW draft or REFINE the most promising one, guided by the conductor's
  score), capped at K refine rounds. Sakana's AB-MCTS (o4-mini + Gemini + DeepSeek) beat any
  single model by ~30% on ARC-AGI-2; Fugu reached Fable-class quality via learned coordination
  — evidence the loop + supervisor, not just the fan-out, is where the jump lives.
- Refine only the drift (targeted), not the whole answer → keeps the token multiple bounded.
- Built only if v2.0's harness shows the single-round win is real (else the loop can't save it).

*(A full multi-agent workflow framework — CrewAI/LangGraph on top of the router for a 10-agent
"AI company" — is a separate v3 track, not this. The conductor here is a fusion-internal role.)*
**v2.2 (idea):** difficulty-adaptive round count; caching identical sub-prompts; fusion receipts
surfaced in `/usage`.

## 10. Security / cost discipline

Fusion multiplies calls (N proposers + aggregator). Kept cheap by NIM-heavy panels; every run's
true cost is in the receipt. Research mode guards the pwm quota (confirm before ~4 Pro Search).
No new secrets; reuses the Phase-1 master key and `127.0.0.1` router.
