# Status & Roadmap — full detail

**Live (Phase 1):** deterministic per-prompt router · NIM + Zen + Copilot ·
live NIM health-gate · full audit · `/usage` · agentmemory on free NIM ·
caveman-ultra + superpowers + lean-ctx in opencode · Perplexity research tool.

**Phase 1.5 (Anthropic Claude Max OAuth — BUILT):** `ant-*` aliases (ant-opus, ant-fable,
ant-sonnet, ant-haiku) route through `providers/claude_oauth_proxy.py` on host:4041, reading
the OAuth token from macOS Keychain (`Claude Code-credentials`). No third-party proxy.
FRONTIER tier puts ant-opus first (flat-rate Max subscription, no per-token cost).
Copilot remains as fallback when credits available.

**Phase 1.6 (Adaptive thinking depth — BUILT):** Auto-inject reasoning budgets based on
routing tier and provider cost-class. 87 tests pass. Details:
- `nim-glm / nim-kimi / nim-minimax` → **HIGH** (32 768 tokens) on REASON/CODE/AGENT/FRONTIER — free, no cost reason to hold back
- `go-glm / go-kimi / go-minimax` → MEDIUM (8 192) on REASON/CODE/AGENT; HIGH (16 384) on FRONTIER
- `ant-opus / ant-fable / ant-sonnet` → MEDIUM (8 192) on REASON/CODE/AGENT; **HIGH (16 384) on FRONTIER**
- `ant-haiku` excluded (no extended thinking support)
- **BOOST escalation:** `[BOOST]` tag · `/boost` command · autodetect ("redo"/"shallow"/"wrong") → forces HIGH across all thinking models
- **Inline annotation:** every non-streaming response prefixed `[model · think:level · tier]`; `/current` shows last think budget
- See spec: [docs/superpowers/specs/2026-07-10-adaptive-thinking-depth-design.md](docs/superpowers/specs/2026-07-10-adaptive-thinking-depth-design.md)

**Phase 2 v2.0 (fusion/MoA — BUILT):** `[NOVEL]` / `[NOVEL DEEP]` / `[NOVEL RESEARCH]`
via the `fuse` MCP tool + `python3 -m fusion.fusion` CLI. Health-aware free+paid committee →
synthesis + cost receipt; difficulty-aware panels, auto-escalation on degrade.

**Phase 2 v2.1 (conductor / AB-MCTS-lite — BUILT):** `[NOVEL]` now runs a **conductor** — a
capable orchestrator that scores each committee synthesis and searches adaptively: **deepen
while improving, widen when stalled** (Sakana's edge), return best-so-far. Self-limits (a strong
round-0 answer costs ~1 round). Key properties, all validated live:
- **Conductor picked dynamically** — fastest healthy capable NIM by measured latency
  (`pick_conductor`) → capable opencode-go (Zen GLM 5.2) → Copilot. Reasoning models welcome.
- **Early-trigger fan-out** — proceeds at a quorum of good drafts; the timeout is a *cap*, not a
  wait. Slow-but-capable models are welcome (cost/quality over speed — minutes to save 10–100×).
- Robust reasoning-model parsing (last-block) + draft-fallback (never returns empty).
- `fusion/conductor.py`; receipt shows the `search_path`. 36 tests pass.

**Benchmark:** SUPPORTED vs co-opus (n=4) — see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Phase 2 v2.1.1 (Multi-LLM AB-MCTS — BUILT):** `[NOVEL TREE]` / `[NOVEL TREE DEEP]` runs the
faithful Sakana algorithm (arXiv 2503.04412, AB-MCTS-A + Multi-LLM): Thompson sampling over an
adaptive search tree (a GEN arm per node — wider at the root, deeper at a node), a per-alias
Thompson bandit with capability priors choosing which model generates each node, and a pairwise
blind judge (Elo ratings in [0,1]) as the reward signal — with a **judge fallback chain** (one
throttled judge must not blind the search). Paid providers are **first-class arms** — the thesis
is frontier quality from a mix of cheap tokens, not free-only. `[NOVEL]` (greedy conductor)
unchanged. Budget: easy 6 / hard 12 / deep 20 generations. Live-validated: bandit punished
throttled paid arms and shifted to healthy NIM mid-search. 58 tests pass.
**Benched: TREE is the champion** (12 prompts) — details in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

**Pending / next:**
- Restore `copilot: available: true` in `availability.yaml` when monthly credits reset.
- **Frontier baseline bench** (ant-fable + HIGH thinking vs co-opus vs nim-glm-thinking on hard prompts) — held until explicitly triggered. Will validate whether the adaptive thinking depth feature + Anthropic Max produces frontier-grade results at flat-rate cost.
- **Phase 2 v2.2 — fusion conductor upgrade** (planned, not yet implemented):
  See [docs/fusion-conductor-v2.2-design.md](docs/fusion-conductor-v2.2-design.md) for full design.
  Summary: rework conductor selection to be thinking-aware; fable as auditor; opus as tool-capable conductor.
- **v3 — multi-agent worker swarm** (planned): orchestrator (ant-fable / ant-opus / go-glm 5.2)
  decomposes task, dispatches parallel workers (ant-sonnet / ant-haiku / healthy NIM).
  Requires CrewAI / LangGraph or bespoke async harness on top of the existing router.

---

## Context-resend cost (measured 2026-07-28, not scheduled)

Audit of 7 days of `LiteLLM_SpendLogs`, to find where input tokens actually go:

| metric | value |
|---|---|
| prompt tokens | **14.36 M** |
| completion tokens | 1.51 M |
| ratio | **9.5 input per 1 output** — ~90% of traffic is resent conversation history |
| model changed between consecutive substantial requests | **33.1 %** |
| requests carrying prompt-cache fields | 182 (negligible) |

Resending history every turn is inherent to stateless chat APIs — **no client harness avoids it**
(audited opencode, pi, jcode, qwen-code, grok-build; see below). The addressable part is ours:

1. **Session-sticky routing** — the router is fully stateless and re-decides per prompt, so a third
   of turns land on a different model. Every switch is a cold cache and a full re-tokenisation.
   Pinning a conversation to its model unless the tier genuinely changes is the single biggest
   lever. Cost caveat: most traffic is free-tier, so the win there is **latency and context
   headroom**; the money win lands on the flat/paid lanes (`ant-*`, `zai-*`, `zen-*`).
2. **Prompt caching** — no `cache_control` is sent anywhere today, so cache-read pricing (~10×
   cheaper on Anthropic/z.ai) is unused. Only useful once (1) exists, since switching invalidates.
3. **Configure DCP** — `~/.config/opencode/dcp.jsonc` contains only a schema reference, so the
   context-pruning plugin that was installed for exactly this problem is running at defaults.

Done already (2026-07-28): lean agent prompts, ~1,840 tok/request — real, but ~2 % of the problem.

### Harness alternatives — audited, all rejected

| harness | system prompt | overridable | MCP | verdict |
|---|---|---|---|---|
| **opencode (current)** | **~460 tok** (after lean override) | yes | first-party, 4 servers working | **keep** |
| `earendil-works/pi` | ~400 | yes | **none first-party** (3rd-party adapter) | MCP is load-bearing here |
| `1jehuang/jcode` | ~490 | **no — compiled in** | **stdio only** | no custom slash-commands |
| `QwenLM/qwen-code` | ~4–5 k | yes (`QWEN_SYSTEM_MD`) | full | 2× the overhead; best weak-model evidence |
| `xai-org/grok-build` | ~1.2 k (+5 k with apply_patch) | yes | full | single-vendor repo, no external PRs |
| `Alishahryar1/free-claude-code` | — | — | — | not a harness — a proxy that duplicates *this* router |

After the lean-prompt work opencode beats grok-build and qwen-code on overhead and ties jcode
(whose 490 cannot be lowered). None of them improves weak-model tool calling — all use plain
native function calling. Migrating would mean rebuilding AGENTS.md, 10 commands, 4 MCP servers and
the caveman plugin to arrive at roughly the current position.

---

## Borrow-from-optillm (evaluated 2026-07-17, not scheduled)

[optillm](https://github.com/algorithmicsuperintelligence/optillm) is a test-time-compute proxy
in the same space as our fusion. Most of its 20+ techniques we already have (MoA=fusion,
MCTS=AB-MCTS, best-of-N, prover-verifier=judge, router=priority_router, memory=agentmemory,
deep-research=[NOVEL RESEARCH], MCP). Four ideas worth adopting later, ranked by value/effort:

1. **System Prompt Learning (SPL)** — top pick. Learns which solving strategy worked on past
   problems and reuses it. Pairs directly with agentmemory (we already persist memories) → router
   and fusion improve over time, not just get cheaper. Highest novel value; substrate already here.
2. **Self-consistency** — the cheap gap: sample N at temperature, majority-vote. Sits between
   single-shot and full AB-MCTS. A light `[NOVEL SELFCON]` mode = big reasoning gain per token.
3. **Z3 / Chain-of-Code verifier** — for the `[VERIFIER]`/`[AUDITOR]` role: run a solver or execute
   code to check math/logic/code claims deterministically instead of LLM-judges-LLM.
4. **MARS temperature exploration** — cheap committee diversity: our fusion diversifies by provider;
   also vary temperature per agent (+30 on AIME in their bench).

Skip: `&`/`|` operator syntax (we use tags), CePO/long-context (lean-ctx covers), privacy/web plugins.
Recommended order when revisited: SPL → self-consistency.
