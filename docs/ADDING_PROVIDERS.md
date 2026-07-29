# Adding a provider

The router is a plain [LiteLLM](https://github.com/BerriAI/litellm) proxy config plus one
Python file. That is the whole extension surface — there is no plugin system to learn.

## How providers work here

LiteLLM is the abstraction layer. Every model the router can reach is one block in
`config.yaml` under `model_list`:

```yaml
- model_name: groq-llama                     # the alias YOU use everywhere else
  litellm_params:
    model: groq/llama-3.3-70b-versatile      # provider prefix + upstream model id
    api_base: https://api.groq.com/openai/v1 # optional — only if not LiteLLM's default
    api_key: os.environ/GROQ_API_KEY         # read from .env at container start
```

`model_name` is your alias; nothing else in the repo cares what the upstream model is
called. `litellm_params.model` carries the provider prefix (`groq/`, `openrouter/`,
`gemini/`, `mistral/`, …) — LiteLLM uses it to pick the right request shape and auth
header. Confirm the exact prefix at
[docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers) before pasting:
a wrong prefix returns a 400, and with fallbacks enabled that failure hides behind a
substitute model instead of announcing itself.

`priority_router.py` then works purely in aliases: which tier an alias belongs to, what
it costs, whether it is healthy.

## Adding a plain API-key provider

Three edits, no new infrastructure.

**1. Declare it in `config.yaml`** — add the `model_list` block above. Then add the env
var to **both** `.env` (your real key) and `.env.example` (an empty placeholder, so the
next person knows it exists).

**2. Slot the alias into a tier in `priority_router.py`** — append it to whichever tier
lists fit the model: `CHEAP_TIER`, `GENERAL_TIER`, `CODE_TIER`, `REASON_TIER`,
`AGENT_TIER`, `FRONTIER_TIER`, `ORCHESTRATOR_TIER`. Those lists are the pools `auto`
routing picks from — an alias in none of them is only reachable by naming it explicitly.
Also add it to `PRIORITY_CHAIN` under its provider key, which is what maps alias →
provider.

**3. Give it a cost class in `_cost_class()`** — free providers return `0`, so cost-first
routing prefers them; flat-rate/subscription and per-token backends get higher numbers
(the current bands: `0` free · `1` z.ai flat · `2` opencode GO flat · `3` Anthropic Max
flat · `4` zen per-token · `5` Copilot per-request). A free open-weight host (Groq,
Cerebras, Gemini free tier) belongs in class 0 alongside NIM and Mistral.

Then restart the proxy — `config.yaml` does **not** hot-reload:

```bash
docker restart litellm-proxy
sh scripts/route_test.sh          # verify the alias answers
```

If it is a free, load-variable backend, also add the alias to the free probe list in
`scripts/nim_health.sh` so the latency health-gate can skip it when it is slow or down.
Models that are never probed fail open (assumed healthy, config order kept), which is the
right default for flat-rate backends but not for a flaky free one.

If the model reasons and takes a reasoning-budget parameter, add it to the matching set
near the top of `priority_router.py` (`NIM_THINKING`, `GO_THINKING`, `ANT_THINKING`,
`ZAI_THINKING`, or `NATIVE_REASONERS` if it reasons with no parameter at all). Each
provider family takes a *different* param shape, and sending the wrong one is a 400 that
a fallback will quietly hide.

## Providers with non-standard auth

The three edits above assume a static API key. Anything else — OAuth device flows, tokens
that live in a system keychain, endpoints that need injected headers — needs a small
bespoke module that terminates the auth and presents an ordinary OpenAI/Anthropic-shaped
endpoint back to LiteLLM.

The worked example is `providers/claude_oauth_proxy.py`: it reads the Claude Code OAuth
token from the macOS Keychain (or `~/.claude/.credentials.json` on Linux/WSL2), injects
the OAuth headers and system prompt, forwards to `api.anthropic.com`, and refreshes the
token before expiry. `config.yaml` then points the `ant-*` aliases at that local proxy as
if it were any other `api_base`. Copy that pattern: handle the auth in a host process,
keep LiteLLM's view boring.

## Free-model sources

Curated, actively-maintained indexes worth checking before you wire anything:

- [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) — the classic auto-updated free-tier index.
- [mnfst/awesome-free-llm-apis](https://github.com/mnfst/awesome-free-llm-apis) — permanent-free tiers only, Provider vs Inference split.
- [amardeeplakshkar/awesome-free-llm-apis](https://github.com/amardeeplakshkar/awesome-free-llm-apis) — tracks OpenAI-SDK compatibility and per-entry rate limits.

Free / generous-tier providers with first-class LiteLLM support (drop-in for cost class 0):

| Provider | LiteLLM prefix | Free tier |
|---|---|---|
| OpenRouter (`:free` models) | `openrouter/` | freemium, many free models |
| Google Gemini | `gemini/` | free tier, no card |
| Groq | `groq/` | fast open-weight, rate-limited free |
| Cerebras | `cerebras/` | fastest-token free tier |
| Together AI | `together_ai/` | trial credit |
| Fireworks AI | `fireworks_ai/` | trial credit |
| Hyperbolic | `hyperbolic/` | free credit |
| Chutes | `chutes/` | decentralized, genuinely-free models (variable reliability) |
| Cloudflare Workers AI | `cloudflare/` | generous daily free allocation |

## Caveat

We tested the providers listed in `config.yaml`. Others should work through LiteLLM, but
they are untested here — PRs and issues welcome, especially if a prefix, auth shape or
reasoning parameter turns out to differ from what this guide assumes.
