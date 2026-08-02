# LLM Router (macOS + Linux/WSL2) — Full Stack Documentation

**Built & verified 2026-07-06.** Source-of-truth for what runs where, how it was
wired, and the gotchas discovered during the build.

Goal: **spend the least** (cheap-first routing + compression) and **route each
request to the best-value model** (deterministic, health-aware, audited), with an
optional future **fusion** layer for maximum quality.

---

## 1. What runs where (Docker vs host)

```
macOS (Apple Silicon) — OrbStack docker runtime
│
├─ DOCKER (compose project in this repo)
│   ├─ litellm-proxy   → 127.0.0.1:4040   (the router)
│   └─ litellm-db      → Postgres, internal only (spend logs, model store)
│
├─ DOCKER (separate, pre-existing)
│   └─ agentmemory-iii-engine-1 → 127.0.0.1:3111  (persistent memory + MCP)
│
└─ HOST (run in your shell / spawned by clients)
    ├─ opencode        (primary client; installed at ~/.opencode/bin)
    ├─ lean-ctx        (context compression, MCP — ~/.local/bin/lean-ctx)
    ├─ caveman + superpowers  (opencode plugins)
    ├─ pwm / pwm-mcp   (Perplexity research tool, MCP — pipx)
    └─ ponytail        (Claude Code plugin — ~/.claude/plugins; not loaded by opencode)
```

**Why the split:** only the thing with external deps (LiteLLM + Postgres) is
containerized. Memory and compression are host/independent so restarting the router
never disrupts them. No launchd services needed for Phase 1 — containers use
`restart: unless-stopped`, MCP servers are spawned on demand by clients.

---

## 2. Backends

| Alias prefix | Backend | Auth | Billing | Role |
|---|---|---|---|---|
| `nim-*` | NVIDIA NIM | `NVIDIA_API_KEY` | free (load-variable) | class 0 — workhorse |
| `mis-*` | Mistral | `MISTRAL_API_KEY` | free tier | class 0 |
| `free-*` | opencode Zen free tier | `ZEN_API_KEY` | $0 | class 0 |
| `go-*` (17) | opencode GO subscription | `ZEN_API_KEY` | flat, quota per 5h | class 1 — most generous, spent first |
| `cod-*` | Codex / ChatGPT | OAuth (`~/.codex/auth.json`) | flat subscription | class 2 |
| `ant-*` | Anthropic Claude Max | OAuth (proxy `:4041`) | flat subscription | class 3 |
| `zai-*` | z.ai GLM Coding Plan | `ZAI_API_KEY` | flat | class 4 |
| `zen-*` | opencode Zen per-token | `ZEN_PAID_API_KEY` | real money | class 5 — backstop |
| `co-*` | GitHub Copilot | OAuth device flow | per-request credit | class 6 — last |

The prefix names the **billing lane**, not the vendor — so the cost of a routing decision is
readable straight off an alias. Scarce or frontier-priced models (`go-grok`, `go-kimi-k3`,
`cod-sol`, `ant-opus`, `ant-fable`) are confined to the frontier and orchestrator tiers.

NIM model IDs verified live against `integrate.api.nvidia.com` 2026-07-06 (no drift).
Aliases are **stable**; only `auto` is special (rewritten per request by the hook).

**Copilot caveat:** the **GPT-family** models (`gpt-5.5`, `gpt-5.4-mini`,
`gpt-5.3-codex`) exist in Copilot's `/models` but **fail through litellm's Copilot
path** (they need Copilot's `/responses` API, which litellm's `github_copilot`
provider doesn't call). **Claude (opus/sonnet/haiku) and Gemini work.** So
`co-gpt`/`co-codex`/`co-mini` are defined but dropped from active tiers; use
`zen-gpt` (Zen's GPT-5.5) when a GPT model is wanted.

---

## 3. Router logic (`priority_router.py`)

A LiteLLM `async_pre_call_hook`. Per prompt: parse tags → apply availability +
health masks → classify tier → pick the first responsive/cheapest model → rewrite
`data["model"]`. Layers and cost-ordered tiers are described in the
[README](../README.md#how-routing-decides-model-auto). Unit-tested in
`tests/test_priority_router.py` (127 tests; `tests/conftest.py` stubs `litellm` so
tests run without the heavy dep — litellm is only present at runtime in the container).

`model_health.yaml`, `availability.yaml` and `model_cache.yaml` are read **fresh per
request** (no restart to apply changes). All three fail-open.

The cache profile is the one that fails open *invisibly*: with no file the router
simply stops preferring models that cache, and everything still answers. Verify the
mount rather than the container's health — see the WSL2 note in the README install
section.

### Three places cost order is decided, not one

Per-request selection is only the first. Each of the others fires on a different
failure and can invert the cascade on its own:

| path | when it runs | ordered by |
|---|---|---|
| tier selection (`order_tier`) | every request | `_cost_class`, then native / cache / stability / latency |
| desperation walk (layer 5) | nothing in the tier or the borrowed free tail is available | `_cost_class` — **explicitly sorted, not `PRIORITY_CHAIN` order** |
| LiteLLM `router_settings.fallbacks` | the chosen deployment errors mid-call | hand-written per-alias lists in `config.yaml` |

`PRIORITY_CHAIN` groups aliases by **provider**, and provider order is not cost
order — iterating it reaches z.ai (4) before GO (1), and Copilot (6) before
Anthropic (3) and Codex (2). The walk therefore sorts by `_cost_class` and treats
the dict as nothing more than the source of candidates; the original index breaks
ties so each provider keeps its own preference order inside a class.

The static fallback chains are the easiest to get wrong, because nothing about a
healthy router reveals them — they only run when a call fails. They must stay
cost-ascending, and a chain ending at Copilot must try a flat lane (GO, Codex,
Anthropic) first, or a transient 503 on a free model escapes straight to
per-request credit while flat subscription quota sits unused. Both properties are
asserted in `tests/test_priority_router.py`
(`test_desperation_walk_is_cost_ordered_not_provider_ordered`,
`test_fallback_chains_never_skip_a_cheaper_lane_to_reach_copilot`), so a hand edit
that breaks either fails the suite instead of surfacing as a bill.

A related trap sits on the caller's side: a request body carrying
`"fallbacks": []` overrides the config chain and makes that alias hard-fail rather
than degrade. The shape is deliberate in `scripts/nim_health.sh` — a probe answered
by a substitute would certify a dead model as healthy — but it should not be copied
into ordinary clients. `Fallbacks=[]` in an error means the request asked for that;
it does not mean the chain is missing.

---

## 4. Health-gate + latency routing (`scripts/nim_health.sh`)

NIM is free but throughput swings with NVIDIA's shared queue, so latency is the one
thing the router can't assume. The audit measures it; the router consumes it.

- Probes **FREE aliases only** (NIM + Mistral free + `free-*`, ~21) **in parallel**. Flat lanes (GO, Codex, Anthropic, z.ai) are deliberately NOT probed — a probe spends the quota they are held in reserve for, and a subscription that authenticates works. Router sorts each tier by (cost class, native, cache, stability, measured latency); `[FRONTIER]`/`[ORCH]` keep config (quality) order. 15-min session refresher: `scripts/install_health_timer.sh` (launchd blocked by TCC on ~/Documents). opencode slash-commands: /refresh /current /info-* /speed /think /performance.
- **Free hosts are ordered by reliability, not speed:** `free-*` (Zen) → `mis-*` (Mistral) → `nim-*` (NIM). This applies in *every* tier, CHEAP included, and outranks measured latency — latency only orders models *within* a band. Measured in one afternoon: NIM served 529s, 19s timeouts and an 8.2s stall while Zen held ~2s and Mistral ~1.2s. Mistral sits in the middle because its key is a single point of failure that has lapsed before. An `ok:false` host is still skipped, so the order never becomes a dead end.
- Probes every alias **in parallel** (`&` + `wait`) → wall-clock ≈ slowest single
  probe (~11s), not the sum.
- `max_tokens: 16` and a "reply OK" prompt — **not** `max_tokens: 1`, which
  false-negatives reasoning models (GLM, DeepSeek-pro emit thinking tokens and stall
  under a 1-token cap).
- **Native reasoners get their own ceiling** (`NIM_REASONER_LATENCY_MAX_MS`, default 2×).
  They think by default with no param to switch it off, so the probe's wall-clock is
  mostly reasoning time rather than lane health — and on a 16-token cap they return
  `finish_reason=length` with `content:""`. Judging that against a non-reasoner's limit
  benches a healthy model for doing the one thing it exists to do (`nim-nemotron-super`
  15788ms and `free-nemotron` 14310ms were both marked dead this way). The alias list is
  imported from `priority_router.NATIVE_REASONERS`, never copied, so it cannot drift.
- `curl -m` capped just past the widest bench threshold — no point waiting 30s for a
  model that's already benched.
- A slow bench reports as `200-slow>Nms`, not a bare `200`. "Healthy 200 but ok:false"
  with no explanation is what made this class of bug hard to see.
- Writes `model_health.yaml`; router skips `ok:false` NIM models. Re-run anytime.

---

## 5. Compression & memory

- **lean-ctx** — MCP inside the client (`ctx_read`/`ctx_search`/`ctx_shell`), compresses
  file reads / tool output before they reach the model. The `:4444` HTTP proxy from
  the WSL original is intentionally **dropped** (MCP-level compression already works).
- **caveman (ultra)** — output-token compression (~65%), opencode plugin.
- **agentmemory** — already-running container on `:3111`; its own LLM features
  (graph extraction, compression, consolidation, context injection) are pointed at the
  router (`OPENAI_BASE_URL=http://localhost:4040/v1`, `OPENAI_MODEL=auto`) so they run
  on **free NIM** with the same health-gating. Config: `~/.agentmemory/.env`.
- **Lean agent prompts** — opencode sends a provider system prompt on *every* request
  (~2,300 tok for a custom OpenAI-compatible provider), on top of `AGENTS.md` (~2,100).
  Defining `agent.build`/`agent.plan` prompts in `opencode.json` **suppresses the built-in
  one**; ours are ~460/~370 tok. What was cut is the 876-token "Tone and style" block —
  six worked examples teaching terseness that caveman + AGENTS.md already enforce. The
  **Tool usage / Following conventions / Code References** sections were deliberately KEPT:
  this router sends most traffic to weak free models, which are exactly the ones that stop
  calling tools correctly without them (verified: nim-llama, nim-step, free-deepseek,
  free-north all still emit valid `tool_calls` on the lean prompt). Detail +
  rollback: `~/.config/opencode/prompts/README.md`.

### Client: use the opencode TUI, not the desktop app

The desktop app is official but BETA and breaks the compression stack **silently**:
plugin `tool.execute.before`/`event` hooks never fire (upstream issue #38604 — plugins show
as active while doing nothing, so caveman and opencode-dcp stop working with no error), and
GUI apps don't inherit shell `PATH`, so `npx`/`uvx`-launched MCP servers (lean-ctx,
agentmemory, perplexity, fusion) fail to spawn. Custom slash-commands in the GUI were closed
NOT PLANNED upstream. Config and auth are shared, so nothing breaks by having it installed —
just don't drive from it. `tui.json` **merges** with `opencode.json` (it is the TUI's own
theme/keybind config); a duplicate plugin entry there is harmless.

---

## 6. Perplexity (`pwm`) — research/audit tool

Installed via `pipx install perplexity-web-mcp-cli --python <python3.12>` (the package
requires `>=3.10,<3.14`; Homebrew's default `python3.14` is excluded — pin 3.12).
Wired into opencode three ways by `pwm setup add opencode` + `pwm skill install
opencode`: MCP tools (`pplx_*`), the skill, and proactive rules in `AGENTS.md`.

- **Separate from the router** — `pplx_*` calls go straight to Perplexity, not `:4040`.
- **Quota:** weekly Pro-Search pool ~300 (not 33/month), Deep Research ~5-10/month.
  Every query = 1 Pro Search regardless of model or source.
- **Not agentic** — Perplexity models don't emit tool-calls (confirmed: given a tool,
  it answered directly). Use as a consultant/researcher, never in a tool-execution loop.
- **Trigger = propose-then-confirm** — explicit `research:`/`audit:`/`[R]`/`[AUDIT]`
  fires directly; auto-detected needs are proposed first (no silent quota burn).
- `pwm council` = native provider-diverse Mixture-of-Agents (candidate Phase-2 engine).

---

## 7. `/usage` accounting

`scripts/usage.sh` (also the opencode `/usage` command) aggregates: per-provider
tokens + spend from `LiteLLM_SpendLogs`, the **free-vs-paid token split** (a proxy for
"tokens kept off paid backends"), NIM health count, and `pwm usage` quotas. Copilot's
per-request credit balance is not exposed by GitHub; Zen's $ balance lives on the
opencode.ai dashboard.

---

## 8. Gotchas discovered (keep for troubleshooting)

- **litellm blocks startup on Copilot device-flow.** With `co-*` models in the config
  and no token, litellm's `github_copilot` authenticator runs its own device flow and
  stalls startup. Fix: provide a token (`scripts/copilot_device_flow.sh`) — then
  **restart litellm** so it re-initialises the `co-*` deployments it dropped on the
  first (tokenless) boot.
- **opencode's stored Copilot token ≠ litellm's.** opencode's `github-copilot` OAuth
  token fails litellm's exchange (`404 copilot_internal/v2/token`) — different OAuth
  app. Run the device flow; don't reuse opencode's token.
- **Copilot GPT models fail via litellm** — see §2. Claude/Gemini only.
- **`pwm` needs Python 3.10–3.13** — pin pipx to 3.12; Homebrew 3.14 is excluded.
- **NIM health probe** — use `max_tokens: 16`, not `1` (reasoning-model false negatives).
- **`docker exec` needs `-i`** for heredoc stdin; the litellm image has **no `curl`**
  (use `python3 urllib`); scripts use `docker` **without sudo** (OrbStack).

---

## 9. Daily commands

```bash
cd <repo>
docker compose up -d && docker compose ps        # start / check
sh scripts/nim_health.sh                          # refresh NIM health (session start)
sh scripts/route_test.sh                          # smoke test tiers
sh scripts/show_routing.sh 20                     # what served recent requests
sh scripts/usage.sh                               # tokens / savings / quota
$EDITOR availability.yaml                          # toggle a provider (no restart)
opencode                                           # drive the stack (/models → auto)
```

---

## 10. Fusion (Phase 2) — `[NOVEL]` Mixture-of-Agents

Built as an **MCP tool + CLI** (not the sidecar the original sketched): `fusion/fusion.py`
(engine) + `fusion/conductor.py` (v2.1 search) + `fusion/mcp_server.py` (`fuse` tool) +
`fusion/fusion_bench.py` (A/B harness). Opt-in via `[NOVEL]` / `[NOVEL DEEP]` /
`[NOVEL RESEARCH]`. Config: `fusion/fusion.yaml`.

- **v2.0** — health-aware free+paid committee → aggregator synthesis + cost receipt.
  Difficulty-aware panels, auto-escalation on degrade, `[NOVEL RESEARCH]` → `pwm council`.
- **v2.1 (conductor / AB-MCTS-lite)** — capable orchestrator scores each synthesis and searches
  adaptively (deepen while improving, widen when stalled, best-so-far). Conductor is picked
  **dynamically**: fastest healthy capable NIM by measured latency → Zen (GLM 5.2) → Copilot.
  **Early-trigger fan-out**: proceeds at a quorum; timeout is a cap, not a wait. Reasoning
  models welcome (generous tokens + last-block parse + draft-fallback so it never returns empty).
- **v2.1.1 (Multi-LLM AB-MCTS)** — `[NOVEL TREE]` / `[NOVEL TREE DEEP]`. Faithful Sakana
  (arXiv 2503.04412): per-node Gaussian value posteriors + optimistic GEN arm, Thompson SELECT
  (wider vs deeper), per-alias Thompson bandit (capability priors: frontier .65 / strong .55 /
  base .45, unlisted → base; paid providers first-class arms), pairwise blind judge → Elo
  ratings in [0,1] as reward, **judge fallback chain** (single judge throttled = blind search —
  learned live). Early stop when the best answer survives 4 straight challenges. Files:
  `fusion/abmcts.py`, `fusion/bandit.py`, `fusion/reward.py`; config: `abmcts:` in `fusion.yaml`.
- **Philosophy**: fusion optimizes **cost/quality, not speed** — minutes to save 10–100× is fine.
- **Bench**: `python3 fusion/fusion_bench.py fusion/prompts.bench4.txt --conduct`. **Verdict
  (n=4): SUPPORTED** — fusion 8.50 vs co-opus 6.50 quality, half the cost, 11× faster
  (`logs/bench4-conduct-final-194009.txt`). Gotcha: the `co-opus` baseline hides extended
  thinking that eats the token cap — a 1500-tok cap returned *empty* answers; needs 6000 tok +
  1200s (wait it out, never truncate mid-think). Fusion lost the pure-code prompt (7 vs 9).

Design specs: fusion v2.0 + v2.1 under `docs/superpowers/specs/`; plans under `.../plans/`.

## 11. Roadmap

- **v3 — multi-agent workflows** (the "AI company"): CrewAI/LangGraph on top of the router,
  orchestrator on a capable model, workers on free NIM. Frameworks are model-agnostic → they
  point at `:4040`.
- Out of scope: lean-ctx `:4444` proxy, Z.ai, launchd auto-start, MLX prompt compressor.
