# llm-router-mac-wsl2

One OpenAI-compatible endpoint, many free and cheap models behind it, cost discipline in the routing.

**2026-07-30** — added Codex/ChatGPT OAuth backend (`cod-luna` / `cod-mini` / `cod-terra` / `cod-sol`, flat-rate, ranked by published quota weight); refreshed the NIM roster after NVIDIA delisted the Qwen family (`nim-inkling`, `nim-gptoss` in, `nim-qwen*` / `nim-mistral` out) and the Zen free tier (`zen-free-ling`, `zen-free-laguna` in, dead `zen-free-hy3` out); flat band reordered to Zen GO before z.ai.

## What it does

It exposes a single OpenAI-compatible endpoint at `http://localhost:4040/v1`. The port is configurable in `docker-compose.yml` or with `--port`. Behind it sit a dozen-plus models across NVIDIA NIM, Mistral, opencode Zen, z.ai GLM, Claude via OAuth, Codex/ChatGPT via OAuth, and GitHub Copilot; clients ask for model `auto` and the router picks a real model per request.

## How it decides

Free tiers drain first. NIM and Mistral cost nothing, so they get tried before anything that costs anything. When they are down or exhausted, the router falls through to flat-rate subscription models — opencode Zen GO, z.ai's GLM Coding Plan, Claude through an existing Max subscription, GPT through an existing Codex/ChatGPT subscription — where the marginal request is still free. A per-token paid model is the last resort, not the default.

None of this is learned. `priority_router.py` is a readable priority list. Here is one request through it, `model: "auto"`, prompt `"debug this stack trace ..."`:

1. **Prompt arrives.** LiteLLM calls `async_pre_call_hook`. The router takes the last user message.
2. **Tag check.** `parse_request` strips `[CHEAP] [CODE] [THINK] [AGENT] [FRONTIER] [ORCH]` and `[AVAILABLE: …] / [UNAVAILABLE: …]` from the text and turns them into directives. A tag sets the tier outright. Phrases like "redo" or "wrong answer" set `boost`, which raises thinking depth without changing the tier. The stripped text is what the model sees.
3. **Availability mask.** `availability.yaml` is read on this request and merged with the per-request allow/deny lists. It is a per-provider on/off switch you edit by hand.
4. **Health mask.** `load_health()` re-reads `model_health.yaml` from disk on **every** request — no cache, no process restart needed. Any alias marked `ok: false` is dropped from the candidate list; a missing or unparseable file fails open, so nothing is dropped. The file is written by `scripts/nim_health.sh`, which probes only the free aliases (NIM, Mistral free, zen-free) in parallel with a 16-token "reply OK" call, fallbacks off. A probe passes only if it returns 200, under `NIM_LATENCY_MAX_MS` (8000 by default), echoes back the alias it asked for, and reports `completion_tokens > 0` — a 200 alone would certify a dead alias that a substitute answered. `scripts/install_health_timer.sh` runs that audit every 15 minutes, plus within ~30s of a paid model answering. Flat-rate backends are never probed; probing spends their quota for nothing.
5. **Content heuristics.** No tag, so `classify()` runs regexes in order: code markers (fences, tracebacks, `def`/`import`, "debug") → `code`; logic words → `reason`; agent words → `agent`; under ~300 tokens → `cheap`; otherwise `general`. This prompt hits `code`.
6. **Cost chain pick.** `CODE_TIER` is sorted by `(cost class, stability, measured latency)`. Cost classes never leapfrog: free → Zen GO flat → z.ai flat → (Claude Max + Codex) flat → Zen per-token → Copilot. Claude Max and Codex share one class because both are sunk-cost subscriptions; inside a class, measured latency decides, so neither is always drained first. Within the free class, steady providers rank above load-variable NIM, and measured latency breaks the remaining ties. The first candidate passing the masks wins — say `mist-codestral`. `[FRONTIER]` and `[ORCH]` skip the latency sort and keep config order, because there the order is the intent. If the whole tier is masked out, the layer-5 chain walks every provider in cost order instead.
7. **Thinking budget.** Each family takes a different parameter shape, so the router injects the right one: an Anthropic thinking block, `reasoning_effort`, or `chat_template_kwargs.enable_thinking`. Native reasoners get nothing injected.
8. **The call goes out.** A failure mid-request is not the router's fallback — LiteLLM's own chain serves a substitute. The router sees it afterwards: if the alias that actually answered was `zen-paid-*`, it touches a trigger file, and the health refresher re-audits the free models so the next prompt can go back to free.
9. **Logged.** LiteLLM writes the routed alias, the served model, tokens and spend to Postgres. The response is prefixed `[model · think:level · tier]` so the choice is visible while you work.

## Architecture

```
  client (Claude Code / Codex / opencode), model: "auto"
        |  POST /v1/chat/completions
        v
  LiteLLM proxy  :4040            <- config.yaml, docker-compose.yml
        |  callbacks: priority_router.router_instance
        v
  PriorityRouter.async_pre_call_hook            <- priority_router.py
        |  parse_request()      tags -> directives
        |  load_availability()  <-- availability.yaml   (hand-edited on/off)
        |  load_health()        <-- model_health.yaml   (re-read every request)
        |  classify()           prompt -> cheap|code|reason|agent|general
        |  order_tier() / pick_model()  cost class -> stability -> latency
        v
  provider: NIM | Mistral | Zen | z.ai GLM | Claude OAuth | Codex OAuth | Copilot
        |                          ^                ^          |
        |     providers/claude_oauth_proxy.py :4041 |          |
        |     providers/codex_oauth_proxy.py  :4042 ----------+
        v
  async_post_call_success_hook -> Postgres LiteLLM_SpendLogs  (audit)

  scripts/nim_health.sh --(every 15 min, install_health_timer.sh)--> model_health.yaml
```

`config.yaml` registers the router as a LiteLLM callback (`callbacks: priority_router.router_instance`) and holds every model block; `docker-compose.yml` binds the proxy to `127.0.0.1:4040` and mounts `priority_router.py` read-only next to the Postgres audit database. All routing logic lives in `priority_router.py` — `parse_request`, `classify`, `order_tier` and `pick_model` run inside `PriorityRouter.async_pre_call_hook`, and `async_post_call_success_hook` handles the audit write plus the paid-model refresh trigger.

Two YAML files are state, not config-you-ship: `availability.yaml` is the per-provider switch you edit by hand, and `model_health.yaml` is written by `scripts/nim_health.sh` on a 15-minute timer installed by `scripts/install_health_timer.sh`. Both are read fresh on every request, so neither needs a restart.

Providers are plain LiteLLM backends except two that reuse an existing subscription instead of an API key: Claude through `providers/claude_oauth_proxy.py` (:4041, a pass-through that injects OAuth headers), and Codex/ChatGPT through `providers/codex_oauth_proxy.py` (:4042). The Codex shim does more work — that subscription serves GPT only via `chatgpt.com/backend-api/codex/responses`, which speaks the Responses API and refuses non-streaming requests, so the proxy translates chat/completions in both directions and re-assembles the SSE stream. It reads `~/.codex/auth.json` fresh per request so a token the Codex CLI refreshes is picked up without a restart. Everything that answered is logged to the Postgres `LiteLLM_SpendLogs` table, which is what `scripts/show_routing.sh`, `scripts/usage.sh` and `scripts/export_audit.sh` read.

## Watching it route

```bash
sh scripts/show_routing.sh 15      # 09:41:02 | nim-glm | z-ai/glm-5.2 | 1840
sh scripts/usage.sh                # nim (free) | 412 reqs | 2.1M tok | $0.0000
sh scripts/export_audit.sh         # exported 1204 requests -> logs/audit-20260729-114302.csv
```

`show_routing.sh watch` refreshes the same table every 2s. All three read the Postgres audit table directly; `psql` against `LiteLLM_SpendLogs` is the interface if you want a different cut.

## Install

Both paths need at least one provider key. NVIDIA NIM (`build.nvidia.com`) and Mistral (`console.mistral.ai`) have free tiers with no card.

Docker:

```bash
git clone https://github.com/slvbr0/llm-router-mac-wsl2.git
cd llm-router-mac-wsl2
cp .env.example .env      # LITELLM_MASTER_KEY, POSTGRES_PASSWORD, one provider key
docker compose up -d      # LiteLLM proxy + Postgres audit db
curl -s http://localhost:4040/health/liveliness
```

Plain Python:

```bash
git clone https://github.com/slvbr0/llm-router-mac-wsl2.git
cd llm-router-mac-wsl2
python3 -m venv venv && source venv/bin/activate
pip install 'litellm[proxy]'
litellm --config config.yaml --port 4040
```

Set `DATABASE_URL` for the audit trail, or skip it and lose only that. This is Python. There is no npm or npx package. The proxy binds `127.0.0.1` on purpose.

## Use it with

Claude Code, Codex, and opencode have all been run against this router. API key is your `LITELLM_MASTER_KEY`, model name is `auto`.

```bash
# opencode
OPENAI_BASE_URL=http://localhost:4040/v1 OPENAI_MODEL=auto opencode
# Codex, in config
base_url = "http://localhost:4040/v1"
# Claude Code, in settings
"ANTHROPIC_BASE_URL": "http://localhost:4040"
```

## When to use / when not

Use it for daily coding assistants, batch work, anything you call over and over where a frontier model every time is waste.

Do not use it for a single high-stakes call where you want one specific model, for latency-critical serving, or for a team that needs an SLA. It is one process on your machine with no uptime guarantee.

## Adding free providers

The wired free and flat-rate backends are NVIDIA NIM ([build.nvidia.com](https://build.nvidia.com), free, no card), Mistral ([console.mistral.ai](https://console.mistral.ai), free tier), [opencode Zen](https://opencode.ai/zen) (free tier plus the GO subscription), [z.ai](https://z.ai) GLM Coding Plan, Claude via an existing Max subscription, GPT via an existing Codex/ChatGPT subscription, and GitHub Copilot. Anything with an [OpenRouter](https://openrouter.ai/models?q=free)-style free tier drops in the same way — [Groq](https://console.groq.com), [Cerebras](https://cloud.cerebras.ai), [Gemini](https://aistudio.google.com), and the free-tier indexes are listed in the guide below.

Three edits for a plain API-key provider. Add the model block to `config.yaml`:

```yaml
- model_name: groq-llama                     # your alias
  litellm_params:
    model: groq/llama-3.3-70b-versatile      # LiteLLM prefix + upstream id
    api_key: os.environ/GROQ_API_KEY
```

Then put the alias in a tier list in `priority_router.py` (`CHEAP_TIER`, `CODE_TIER`, `REASON_TIER`, …) plus `PRIORITY_CHAIN`, and give it a cost class in `_cost_class()` — free returns `0`. Then add `GROQ_API_KEY` to `.env` and `.env.example`, and `docker restart litellm-proxy`; `config.yaml` does not hot-reload.

[docs/ADDING_PROVIDERS.md](docs/ADDING_PROVIDERS.md) has the full version: the free-tier indexes, the LiteLLM prefix table, where to register a reasoning parameter, how to add a flaky free alias to the health probe, and the OAuth-style pattern for providers with no static key.

## Companion tools

Pairs with prompt and token-compression tooling on the client side: [caveman](https://github.com/JuliusBrussee/caveman), [lean-ctx](https://github.com/yvgude/lean-ctx), [rtk](https://github.com/rtk-ai/rtk).
Linked, not bundled. Install them separately if you want them.

## Foundations

- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — what the fusion and routing modes actually scored.
- [docs/ADDING_PROVIDERS.md](docs/ADDING_PROVIDERS.md) — the extension surface, end to end.
- [docs/STACK.md](docs/STACK.md) — what runs in Docker, what runs on the host, and the gotchas found during the build.

## Benchmarks

Implements routing and fusion approaches from the literature as opt-in modes: mixture-of-agents committees and adaptive tree search.
Results live in [docs/BENCHMARKS.md](docs/BENCHMARKS.md), including a negative one — multi-model fusion did not match a properly configured frontier baseline.
No performance claims here. Read the doc.

## References

- [BerriAI/litellm](https://github.com/BerriAI/litellm) — the proxy this sits on.
- Wang et al., *Mixture-of-Agents Enhances Large Language Model Capabilities* — [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
- Sakana AI, *Adaptive Branching Monte Carlo Tree Search* — [arXiv:2503.04412](https://arxiv.org/abs/2503.04412)
- Ong et al., *RouteLLM: Learning to Route LLMs with Preference Data* — [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)
- [slvbr0/llm-router-v1-archive](https://github.com/slvbr0/llm-router-v1-archive) — the earlier WSL-only design this superseded.
- [slvbr0/inverse-jobs](https://github.com/slvbr0/inverse-jobs) — scrapes job boards, scores postings against your profile, mails a daily digest of the few worth applying to by hand.
- [slvbr0/audio-to-text-hours-long](https://github.com/slvbr0/audio-to-text-hours-long) — transcribes hours-long audio locally with chunked Whisper, then compresses and summarizes it.
