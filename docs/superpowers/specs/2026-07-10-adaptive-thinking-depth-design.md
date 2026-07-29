# Adaptive Thinking Depth — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically inject provider-appropriate thinking budgets (low/medium/high) based on the routing tier and cost-class of the chosen model, with user-controlled escalation and per-response visibility.

**Architecture:** Extend `priority_router.py`'s existing `async_pre_call_hook` to inject thinking params after model selection. Two param formats: Anthropic `thinking` dict and OpenAI-compat `extra_body`. Escalation via tag/slash/autodetect. Visibility via inline response annotation + `/current` extension.

**Tech Stack:** Python (priority_router.py extension) · LiteLLM custom logger hooks · opencode.json `/boost` command

---

## Thinking-capable model sets

```python
NIM_THINKING = {"nim-glm", "nim-kimi", "nim-minimax"}   # free — HIGH budget
GO_THINKING  = {"zen-glm", "zen-kimi", "zen-minimax"}   # GO flat-rate — MEDIUM default
ANT_THINKING = {"ant-opus", "ant-fable", "ant-sonnet"}   # Max OAuth — MEDIUM default (not ant-haiku)
```

`ant-haiku` excluded — Claude Haiku 4.5 does not support extended thinking.

---

## Budget table

| Tier | NIM thinking | GO thinking | ant-* thinking |
|---|---|---|---|
| CHEAP / GENERAL | none | none | none |
| REASON / CODE | HIGH | MEDIUM | MEDIUM |
| AGENT | HIGH | MEDIUM | MEDIUM |
| FRONTIER / NOVEL | HIGH | HIGH | HIGH |
| BOOST (any escalation) | HIGH | HIGH | HIGH |

"none" = thinking param omitted entirely (model uses default fast mode).

---

## Token budgets

| Level | NIM (free) | GO (flat-rate) | ant-* (Max OAuth) |
|---|---|---|---|
| MEDIUM | — | 8 192 | 8 192 |
| HIGH | 32 768 | 16 384 | 16 384 |

NIM HIGH = 32 768 — GLM 5.2 / Kimi K2 support up to ~38k; NIM is free, maximize quality.
GO + ant HIGH = 16 384 — sufficient for hard tasks without hammering rate limits.
NIM has no MEDIUM row: it always gets HIGH when thinking is active (free, no reason to hold back).

---

## Param injection format

Two wire formats, injected in `async_pre_call_hook` after model selection:

```python
# Anthropic (ant-*) — LiteLLM native param, survives drop_params: true
data["thinking"] = {"type": "thinking", "budget_tokens": N}

# NIM + GO (OpenAI-compat) — extra_body bypasses drop_params
data.setdefault("extra_body", {}).update({
    "enable_thinking": True,
    "thinking_budget_tokens": N,
})
```

`extra_body` merges rather than overwrites so other callers can also set it.

---

## Escalation signals

Three composable signals — any one triggers BOOST (HIGH for all providers):

| Signal | Example | Detection |
|---|---|---|
| Tag `[BOOST]` | `[BOOST] redo this, too shallow` | `parse_request()` — new `BOOST` entry in `TAG_RE` |
| Slash `/boost` | `/boost` in opencode prompt | opencode.json command → template `[BOOST] $ARGUMENTS` |
| Autodetect | "not good", "redo", "wrong", "shallow", "bad answer", "try again" | new `BOOST_RE` regex; sets same BOOST flag |

BOOST is stored in `directives["boost"] = True`. Budget table lookup: if boost → HIGH everywhere.

`[BOOST]` composes with tier tags: `[REASON][BOOST] explain this again` → REASON tier + HIGH budget.

---

## Visibility

### Per-response annotation (non-streaming)

`async_post_call_success_hook` prepends one line to `choices[0].message.content`:

```
[nim-glm · think:high · reason]

Actual model answer starts here.
```

Format: `[{alias} · think:{level} · {tier}]`
- `level` = `high` / `medium` / `none`
- Only shown when thinking was actually injected (level ≠ none)
- Blank line separates annotation from content

### Streaming fallback

Streaming response content modification is not reliably possible in LiteLLM hooks. For streaming responses: annotation skipped, but the budget IS still injected. User can run `/current` to see what was used.

### `/current` command extension

`show_routing.sh` already shows last routed alias + model. Extend it to also log and display thinking budget: `nim-glm (z-ai/glm-5.2) · think:high · tier:reason`.

The router stores the thinking decision in a module-level `_last_think` dict keyed by request ID (or latest), written by `async_pre_call_hook`, readable by the shell script via a small temp file (`/tmp/llmr-last-think.json`).

---

## Routing log

`verbose_logger.info` already logs model selection. Add budget to the same line:

```
PriorityRouter: tier=reason model=nim-glm think=high budget_tokens=32768
```

---

## Files changed

| File | Change |
|---|---|
| `priority_router.py` | `NIM_THINKING`, `GO_THINKING`, `ANT_THINKING` sets; `_think_budget()` function; inject in `async_pre_call_hook`; `BOOST_RE` + `directives["boost"]`; `async_post_call_success_hook` for annotation; write `/tmp/llmr-last-think.json` |
| `scripts/show_routing.sh` | Read `/tmp/llmr-last-think.json`, append to output |
| `~/.config/opencode/opencode.json` | Add `/boost` command → `[BOOST] $ARGUMENTS` |
| `tests/test_priority_router.py` | New tests: budget table, BOOST escalation, param injection format, annotation |

No new files. No new dependencies.

---

## Out of scope (this spec)

- Retry loop with quality evaluation — still manual (user signals via BOOST)
- Anthropic thinking token cost tracking — Max OAuth is flat-rate; no per-token billing to track
- LOW budget level — reserved, not implemented yet
- Streaming annotation — deferred
