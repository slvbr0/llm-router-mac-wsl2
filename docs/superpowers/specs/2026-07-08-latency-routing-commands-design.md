# Latency-aware routing + auto-refresh + opencode /commands · Design Spec

**Date:** 2026-07-08 · **Approved by:** salva (chat)

## 1. Goal
Always pick the fastest healthy model *within the same cost class* — NIM free vs Zen free
compete on measured latency (NIM often faster, sometimes unresponsive — measurement decides).
Keep health data fresh automatically (15 min), and add opencode slash-commands for daily ops.

## 2. Components

### 2.1 Extended health probe (`scripts/nim_health.sh`, name kept for compat)
Probes ALL routed aliases in parallel (not just NIM): 11 NIM + 3 zen-free + 7 zen-GO.
Same method (max_tokens 16, "reply OK", curl -m cap). Writes `model_health.yaml` with
`{ok, latency_ms}` per alias. Copilot NOT probed (per-request credits — never burn on audit).
Zen probes cost ~16 tokens each on free/flat-rate — negligible.

### 2.2 Latency-aware tier ordering (`priority_router.py`)
Within a tier, candidates sort by `(cost_class, measured_latency)`:
- class 0 = free: `nim-*`, `zen-free-*` — **compete by latency**
- class 1 = GO flat-rate: `zen-glm, zen-deepseek, zen-kimi, zen-minimax, zen-qwen-max, zen-mimo, zen-deepseek-flash`
- class 2 = zen per-token: `zen-gpt`
- class 3 = copilot per-credit: `cop-*`
Missing latency (never probed / file absent) → keep config order within class (fail-open,
stable). Health `ok:false` still filters out (now applies to zen too, not just NIM).

### 2.3 Auto-refresh (15 min)
launchd agent `~/Library/LaunchAgents/com.llmr.health.plist` runs the probe every 900 s.
Managed by `scripts/install_health_timer.sh` (idempotent load/reload). Manual: `llmr-health`
or `/refresh` in opencode.

### 2.4 opencode slash-commands (`~/.config/opencode/opencode.json` → `command`)
| Command | Behavior |
|---|---|
| `/refresh` | agent runs the health probe, reports the latency table |
| `/current` | agent runs `show_routing.sh 1` → last prompt's alias → real model |
| `/info-go` / `/info-co` / `/info-total` | per-provider tokens+spend from audit DB (`usage.sh`); dashboards stay source-of-truth for quota % |
| `/speed` / `/think` / `/performance` | prepend `[CHEAP]` / `[THINK]` / `[FRONTIER]` to `$ARGUMENTS` — sugar; raw tags keep working |

## 3. Testing
- `classify`/tier tests unchanged (11 existing).
- New: `order_tier()` — free class sorts by latency (nim vs zen-free flip on data); GO never
  before a healthy free; unprobed aliases keep config order; ok:false zen excluded.

## 4. Out of scope
Copilot probing · GO quota-% API (not exposed) · router-side auto-probe (host launchd owns it).
