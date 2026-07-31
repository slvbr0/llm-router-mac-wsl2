# Cache-aware routing + session stickiness

**Goal:** stop re-processing the same 80k of context every turn on models that cannot cache it.

## What the audit actually found (2026-07-31, 7 days, payloads >3k tokens)

Prompt caching is **already working** — automatically, provider-side, with no `cache_control` from
us. The earlier claim that "no caching is happening" was wrong: it grepped `cache_read`, but the
field LiteLLM records is `metadata.usage_object.prompt_tokens_details.cached_tokens`.

| model | prompt tokens | cached | hit rate |
|---|---|---|---|
| **free-north** | **28.35 M** | 0 | **0 %** |
| zai-52 | 4.75 M | 3.64 M | 76.6 % |
| free-deepseek | 2.73 M | 2.63 M | 96.2 % |
| free-pickle | 1.28 M | 1.28 M | 99.6 % |
| free-laguna | 1.16 M | 0.56 M | 48.2 % |
| zai-turbo | 1.06 M | 0.43 M | 40.7 % |
| nim-step | 0.85 M | 0 | 0 % |
| nim-* (others) | — | — | 0–25 % |

`free-north` is the CODE-tier default and by far the biggest consumer — 10× the next model — and
it caches **nothing**. Verified per-row: `prompt_tokens_details` is present and `cached_tokens` is
0 on every request, including four consecutive 77,365-token calls. `free-pickle` at the same size
caches ~99%. So this is the model, not our request shape.

Cost is unaffected (these are free lanes). **Latency is not:** every turn re-runs prefill over the
whole payload. That is the most likely cause of "the router feels slow".

## Design

### 1. Measured cache profile (`model_cache.yaml`), not a hardcoded list

A hardcoded "these models cache" table goes stale the moment a provider changes. Instead mirror
what already works for health: a script measures from the audit trail, the router reads the file
fresh per request, and a missing file fails open.

- `scripts/cache_audit.sh` → reads `LiteLLM_SpendLogs`, writes `model_cache.yaml`:
  `{alias: {hit_pct: 96.2, samples: 47}}`
- Only models with enough large-payload samples get a verdict; everything else is "unknown" and is
  never penalised on the strength of thin data.

### 2. Cache-aware ordering — only where it pays

New sort component in `order_tier`, active **only** when the payload is large:

```
(cost_class, native_rank, cache_rank, stability_rank, latency, idx)
```

`cache_rank` is 0 for a model with a measured hit rate ≥ threshold (or unknown), 1 for one
measured at ~0%. It sits *after* cost class, so it can never promote a paid model over a free one —
it only reorders within a class, which is exactly where the choice is free anyway.

Small payloads ignore it entirely: with nothing to cache, a 0% model is not worse.

### 3. Session stickiness

Switching models mid-conversation throws the provider's cache away. Measured switch rate on large
requests is 15.6%, so ~84% of turns already stay put — stickiness protects that rather than
creating it.

- Key: hash of the first user message (stable as a conversation grows; two conversations opening
  with identical text share a key, which is harmless — they route the same anyway).
- Reuse the remembered model when the tier is unchanged, the model is still healthy and available,
  and no explicit alias/tier tag overrides it.
- Bounded dict (drop oldest past N) so a long-lived container cannot grow without limit.

## Non-goals

- **Injecting `cache_control`.** Anthropic-shaped lanes accept explicit breakpoints, but caching
  already happens automatically on the lanes that support it, and the measured gap is a model that
  cannot cache at all. Adding a param would not move `free-north` off 0%.
- **Fighting DCP.** Pruning history changes the prefix and costs cache hits. Both are wanted; the
  tension is real and is not resolved here.

## Verification

Offline: unit tests for ordering, stickiness, and fail-open.
Live (deferred — container in use): restart, then confirm a large CODE payload picks a caching
model over `free-north`, and that `cached_tokens` climbs for it in the audit trail.
