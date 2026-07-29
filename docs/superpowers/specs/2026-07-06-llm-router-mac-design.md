# llm-router — Mac-native port · Design Spec

**Date:** 2026-07-06
**Source:** port of `slvbr0/llm-router` (private; WSL2/Windows original) to macOS.
**Author:** Claude Code (brainstorming session, approved by salva).

---

## 1. Goal

Self-hosted LLM gateway on the Mac that (a) **routes each prompt to the best-value
model** and (b) **saves tokens/cost**, so the user's metered **opencode Zen ("go")
balance is preserved** — bulk load is absorbed by ~free NVIDIA NIM **when it's
performing**, compression shrinks every request, and paid backends step in only
when NIM can't deliver on latency/quality.

**Routing is a PER-PROMPT trade-off** — LiteLLM's `priority_router` classifies each
request and picks a backend from three levers. It is **not** a session-wide primary;
every prompt is decided on its own.

- **consume (each backend bills differently — this shapes the fallback):**
  - **NIM** — free, but throughput is **load-variable** (NVIDIA queue) → primary *when responsive*.
  - **Zen** — **per-1M-tokens** → cheap for *small* prompts; avoid huge-context prompts here.
  - **Copilot** — **per-request credit**, flat per call → 1 credit tiny or huge, so
    **reserve for big/hard prompts**; never spend a credit on trivia.
- **response** — a session-start audit records live NIM per-model latency;
  the router skips slow/dead NIM models *per request* (no restart, re-runnable anytime).
- **performance** — tier picks a capable-enough model; `[FRONTIER]` forces top-end.

Net effect per prompt: **small + NIM down → Zen** (per-token, cheap); **big/hard +
NIM down → Copilot** (credit amortized, frontier quality). NIM leads whenever it's fast.

## 2. Environment (audited 2026-07-06)

| Fact | Value |
|---|---|
| Machine | Apple Silicon arm64, macOS 26.4.1 |
| Container runtime | **OrbStack** (docker CLI v29.4.0, no sudo needed) |
| agentmemory | **already running** — container `agentmemory-iii-engine-1` (iiidev/iii:0.11.2) on `127.0.0.1:3111`, LLM features currently **noop** (no provider key) |
| lean-ctx | installed `~/.local/bin/lean-ctx`; used as MCP in clients |
| opencode | **binary NOT in PATH**; config dir `~/.config/opencode/` exists (lean-ctx MCP only, no backend provider) |
| Ports | `:4040` free, `:4444` free, `:3111` taken by agentmemory |
| Keys | `NVIDIA_API_KEY` supplied; `ZEN_API_KEY` (opencode-go) to supply; Copilot via device-flow |

## 3. Architecture

```
opencode (+ claude / codex)            host CLIs on Mac
   │  lean-ctx MCP   → compresses file reads / search INSIDE client
   │  caveman ultra  → compresses model OUTPUT (~65%)
   ▼ http://localhost:4040/v1
LiteLLM router  (OrbStack container)
   │  priority_router.py — 6 tiers, first-match; latency- + availability-gated
   ├─► NVIDIA NIM      free-tier dev → WORKHORSE / default (GLM 5.2)
   ├─► opencode Zen    metered $     → CONSERVE (free models = fallback; paid = frontier only)
   └─► GitHub Copilot  flat sub      → frontier last resort (device-flow auth)
   │
   └─► Postgres (container) — audit trail: every routing decision, tokens, spend

agentmemory (container, already up :3111) — shared memory MCP;
   its own LLM features (graph/compress/consolidate) routed to nim-llama (free)
```

**Design rule (from original, kept):** only the router + Postgres run in Docker.
Everything else is host-level (MCP servers spawned by clients, agentmemory
container already independent). Restarting the router never kills memory/compression.

**MCP-only compression decision:** the `:4444` lean-ctx HTTP proxy from the
original is **dropped**. lean-ctx already compresses at MCP level inside every
client. No launchd service needed for v1.

## 4. Backends & model roster

Aliases are **stable**; underlying model IDs may drift and are re-verified at build
(`discover_models.sh` / `nim_health.sh`). NIM roster mirrors original (18 aliases,
GLM/DeepSeek/Qwen/Kimi/Nemotron/Mistral/Llama…). New for Mac port:

- **Zen backend** (`api_base: https://opencode.ai/zen/v1`, `api_key: ZEN_API_KEY`):
  - `zen-gpt` → `opencode/gpt-5.5` (paid, frontier)
  - `zen-glm` → `opencode/glm-5.2` (paid)
  - `zen-deepseek` → `opencode/deepseek-v4-pro` (paid)
  - `zen-free-nemotron` → `opencode/nemotron-3-ultra-free` (free)
  - `zen-free-deepseek` → `opencode/deepseek-v4-flash-free` (free)
  - `zen-free-pickle` → `opencode/big-pickle` (free)
  - *(exact free IDs confirmed against Zen at build)*
- **Copilot backend** (unchanged): `cop-opus/sonnet/gpt/codex/gemini/haiku/mini`.
- **Z.ai** — **dropped** (Zen supersedes it).

## 5. Routing (priority_router.py)

First match wins (original layering kept):
1. Explicit tags `[CHEAP] [CODE] [THINK]/[REASON] [AGENT] [FRONTIER]`, `[AVAILABLE:]/[UNAVAILABLE:]`.
2. Global `availability.yaml` provider mask (hot-reload).
3. **NEW: `model_health.yaml` per-model mask** (hot-reload) — degraded/slow NIM models skipped.
4. Content heuristics (code markers → code; logic words → reason; agent words → agent; short prompt → cheap).
5. Default → GENERAL tier → **`nim-glm` (GLM 5.2)**.

**Tier order = `NIM(health-gated) → Zen(free → paid) → Copilot`.** Health-gate
(§6) removes slow/dead NIM models, so when NIM underperforms the request lands on
Zen automatically — Zen is the immediate fallback, not Copilot. Copilot is the tail.

| Tier | Order (each NIM entry health-gated; skipped if audit says slow/down) |
|---|---|
| cheap | `nim-llama` → `nim-deepseek-flash` → `zen-free-deepseek` → `zen-free-pickle` → `cop-mini` |
| general (default) | **`nim-glm`** → `nim-mistral` → `zen-free-nemotron` → `zen-glm`(paid) → `cop-sonnet` |
| code | `nim-deepseek` → `nim-kimi` → `nim-qwen` → `zen-deepseek`(paid) → `cop-codex` |
| reason | `nim-qwen-max` → `nim-nemotron` → `zen-free-nemotron` → `zen-gpt`(paid) → `cop-opus` |
| agent | `nim-glm` → `nim-minimax` → `nim-kimi` → `zen-glm`(paid) → `cop-sonnet` |
| frontier | `cop-opus` → `zen-gpt`(paid) → `cop-sonnet` → `cop-gpt` |

**Per-prompt invariants (cost-aware):**
- **NIM leads whenever it's responsive** (free, GLM 5.2 default). A NIM model the
  audit flags slow/down is skipped for that request → tier falls through.
- **Cheap/small tiers fall to Zen, not Copilot** — Zen bills per-token, so a tiny
  prompt is cents; a Copilot credit on trivia is waste. Zen *free* models before *paid*.
- **Reason/frontier tiers fall to Copilot** — big/hard prompt amortizes the flat
  per-request credit and buys frontier quality; `zen-gpt` (paid) sits alongside.
- **Copilot never appears in cheap/general** — no credit is ever spent on routine work.
- No session-wide flip. Each prompt is routed on its own class + live NIM health.

**Provider priority chain (layer-4 fallback):** `nim(responsive) → zen(free) → zen(paid) → copilot`.

## 6. NIM latency health-gate (NEW component)

NIM is free but **load-variable** (NVIDIA's shared queue), so its latency is the one
thing the router can't assume. The audit measures it; the router consumes it per
request. It is **latency data, not a routing lock** — the per-prompt classifier still
decides tier/backend; the audit only removes NIM models that are currently too slow.

**`scripts/nim_health.sh`** — run at session start, or anytime NIM feels laggy (~10s):
- Sends a 1-token prompt to each NIM alias; measures success + total latency.
- Threshold (default 8s, `NIM_LATENCY_MAX_MS`) → models slower/failing = `degraded`.
- Writes `model_health.yaml`: `{ models: { alias: {ok: bool, latency_ms: int} } }`.
- Prints a table (`alias | ok | latency`) so the human sees what's fast right now.

**`priority_router.py`** reads `model_health.yaml` fresh per request (same pattern as
`load_availability`): `pick_model` + fallback chain **skip any NIM alias whose `ok`
is false**, so a slow/dead NIM model drops out of that request's candidates and the
tier falls through (→ Zen for cheap, → Copilot for hard). No restart.
Missing/stale file = all healthy (**fail-open** — never blocks work).

## 7. agentmemory wiring

Already running; only needs its LLM pointed at the router so its features are free:
edit `~/.agentmemory/.env`:
```
OPENAI_API_KEY=<LITELLM_MASTER_KEY>
OPENAI_BASE_URL=http://localhost:4040/v1
OPENAI_MODEL=nim-llama
GRAPH_EXTRACTION_ENABLED=true
AGENTMEMORY_AUTO_COMPRESS=true
AGENTMEMORY_INJECT_CONTEXT=true
CONSOLIDATION_ENABLED=true
```
Restart the container. All memory LLM work now runs on free NIM.

## 8. opencode

- Install the `opencode` CLI (missing from PATH).
- `~/.config/opencode/opencode.json`: add provider `llm-router`
  (`@ai-sdk/openai-compatible`, `baseURL http://localhost:4040/v1`,
  `apiKey <LITELLM_MASTER_KEY>`), models = real aliases with **`auto` default →
  GLM 5.2**. Keep lean-ctx MCP; add agentmemory MCP.
- Plugins: **caveman** (level **ultra**) + **superpowers** (git spec).
- opencode talks ONLY to the router — never Zen/NIM/Copilot directly — so routing +
  compression + audit apply to all traffic.

## 9. Audit / token accounting

Postgres `LiteLLM_SpendLogs` (survives restart in `litellm-db` volume): timestamp,
routed alias, actual model, tokens, spend, cache_hit. Scripts copied from original:
`show_routing.sh` (live/last-N), `export_audit.sh [--push]` (CSV snapshot).
This is where the user watches **Zen burn per prompt** and per-tier token use.

## 10. Build units (→ implementation plan)

1. **Scaffold** — copy router files here; generate `.env` (master key, pg pw);
   add `NVIDIA_API_KEY` + `ZEN_API_KEY`; `.gitignore` `.env` + health/audit.
2. **Zen + tiers** — extend `config.yaml` (Zen backend + aliases), rewrite tier
   tables in `priority_router.py`, drop Z.ai.
3. **NIM health-gate** — write `nim_health.sh`; add `model_health.yaml` read to router.
4. **Launch** — `docker compose up -d`; health-check `:4040`.
5. **Verify roster** — `discover_models.sh` + `nim_health.sh` vs live NIM/Zen; fix drift.
6. **Copilot auth** — `copilot_device_flow.sh` (sudo dropped), one-time OAuth.
7. **agentmemory wiring** — edit `.env`, restart container, `doctor`.
8. **opencode** — install; provider + MCP + caveman(ultra) + superpowers.
9. **Prove** — `route_test.sh` → `show_routing.sh` shows distinct healthy models
   per tier; `export_audit.sh` snapshot.

## 11. Phase 2 — FUSION / "Novel-Discovery" mode (Mixture-of-Agents)

Ships **after** v1 is working; bolts on without touching the v1 router. Extends the
original repo's unbuilt `FUSION_ADDEND.md` with two deliberate inversions.

**What it is:** Mixture-of-Agents — fan one prompt out to a panel of diverse models
(proposers), then an aggregator model synthesizes their drafts into one answer.
Coordinated cheaper models reach frontier-tier quality (the "Fable-by-committee"
result). Opt-in for the hardest / most-open-ended prompts.

**Inversions vs original design:**
- **Opt-in, default OFF.** Triggered by an explicit key **`[NOVEL]`** (aliases
  `[FUSION]`, `[DISCOVERY]`). No tag → normal v1 single-model routing. (Original was
  default-on with `[NOFUSION]` escape — wrong for a cost-conserving stack.)
- **Free-NIM panel.** Proposers are diverse **free NIM** models so a `[NOVEL]` run is
  ~$0 unless escalated. `[NOVEL FRONTIER]` swaps the aggregator to Copilot/Zen-paid.

**Topology (sidecar, matches original):**
```
opencode → Fusion sidecar :4041 → (fans out via) LiteLLM :4040 → NIM/Zen/Copilot
             │ only [NOVEL] prompts fuse; everything else passes straight through
```
Small FastAPI/OpenAI-compatible service in a container beside litellm. opencode's
provider baseURL moves to `:4041`; non-`[NOVEL]` traffic is proxied verbatim to
`:4040`, so v1 behavior is byte-identical when the tag is absent.

**Panels are PROVIDER-DIVERSE** (spanning NIM + Zen + Copilot beats any single-provider
panel — architectural diversity is MoA's whole edge). Aliases are stable:
| Panel | Trigger | Proposers (span providers) | Aggregator |
|---|---|---|---|
| default | `[NOVEL]` | `nim-glm`, `nim-deepseek`, `zen-free-nemotron` | `nim-qwen-max` |
| deep | `[NOVEL DEEP]` | `nim-glm`, `nim-deepseek`, `nim-qwen-max`, `zen-gpt`, **`cop-sonnet`** | `nim-nemotron` |
| frontier | `[NOVEL FRONTIER]` | `nim-qwen-max`, `zen-gpt`, **`cop-opus`** | `cop-opus` |

**Fusion = the right place to spend Copilot.** Its per-request credit buys exactly one
strong diverse proposer, credit fully amortized into a high-value answer — never
wasted on routine single calls.

**Health/rate-limit-aware substitution:** panel calls go **through `:4040`**, so the
health-gate, availability masks, fallbacks, and audit all apply. If a NIM proposer is
degraded (§6) **or rate-limited (429 / req-min cap)**, the sidecar substitutes the
equivalent tier model from Zen (then Copilot) to **preserve panel width and provider
spread** — a slow NIM never shrinks the panel. Single-layer MoA for v2.0
(proposers → aggregator); multi-round debate = v2.1.

**Cost guard:** fusion multiplies calls (N proposers + 1 aggregator). Kept ~free by
the NIM-only default panel; every sub-call is logged in Postgres, so the audit shows
the true token multiple of a `[NOVEL]` run vs a single-model answer.

**A/B harness (gate before trusting it):** `scripts/compare_fusion.sh <prompts>` runs
each prompt through single-model (`:4040`) and fusion (`:4041`), records tokens /
latency / blind-judged quality (from `LiteLLM_SpendLogs` + a judge pass). Keep
`[NOVEL]` opt-in only for prompt classes where it wins on quality at acceptable cost.

**Phase-2 build units:** (a) fusion sidecar service + Dockerfile + compose entry;
(b) `[NOVEL]` tag parse + panel builder + aggregator prompt; (c) passthrough proxy
for untagged traffic; (d) point opencode at `:4041`; (e) `compare_fusion.sh` harness.

## 12. Out of scope (v3+)

lean-ctx `:4444` HTTP proxy chain · Z.ai backend · local prompt compressor
(LLMLingua needs CUDA; Apple-Silicon MLX port = v3) · launchd auto-start of
`nim_health.sh` · multi-round fusion debate · SSH/WSL layer.

## 13. Security

- `NVIDIA_API_KEY` was pasted in chat → **rotate at build.nvidia.com after
  validation**. `ZEN_API_KEY` to be supplied at build (or pulled from opencode
  auth), never committed.
- `.env` gitignored. Router bound to `127.0.0.1` only. Postgres internal to the
  compose network. Master key is the only client credential.
