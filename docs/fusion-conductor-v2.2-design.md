# Fusion Conductor v2.2 — Design Notes

> Status: **planned, not yet implemented.** Implement after frontier bench confirms
> adaptive thinking depth delivers expected quality lift.

---

## Goal

Upgrade the `[NOVEL]` conductor and committee selection to be thinking-aware and
role-differentiated: a dedicated **auditor** (Fable), a **conductor** that can use
tools when needed (Opus), and thinking-enabled NIM committee members — producing
the quality ceiling of a frontier model from a mix of free and flat-rate tokens.

---

## Roles

### Committee members (unchanged protocol)
Same parallel fan-out as today. With Phase 1.6 thinking depth in place, any
committee member that is a thinking-capable NIM model (`nim-glm`, `nim-kimi`,
`nim-minimax`) automatically gets HIGH thinking budgets (free — no reason to
hold back). This means committee drafts are already substantially higher quality
with zero code changes to fusion.

### Conductor (upgraded)
Current `pick_conductor` selects the fastest healthy capable NIM by latency,
then falls back to Zen GLM → Copilot. New priority:

```
1. Capable NIM (nim-glm, nim-kimi, nim-minimax) — fastest healthy one, HIGH thinking
2. ant-opus                                      — conductor WITH tool access (can call
                                                   tools / run code when synthesis needs it)
3. ant-fable                                     — only when ant-opus unavailable AND
                                                   task genuinely needs deep reasoning
```

**Opus as conductor rationale:** Synthesis is the expensive, high-value step.
Opus 4.8 is capable, supports tools (can verify claims, run code, fetch refs),
and is flat-rate on Max — using it as conductor costs nothing extra and lifts
synthesis quality. Fable is reserved for when Opus is unavailable, since Fable
is the deeper reasoner but does not need tool access for synthesis.

### Auditor (new role — ant-fable)
After conductor produces a synthesis, Fable reads the result and provides targeted
fixes. **Fable does NOT call any tools** — it is a read-only critic:

```
committee drafts → conductor synthesis → fable auditor review → final answer
```

Fable's audit prompt: "Read this synthesis and the original question. Identify
any errors, missing nuance, or logical gaps. Provide targeted corrections only —
do not rewrite from scratch."

Fable audit is opt-in: `[NOVEL DEEP]` and `[NOVEL TREE DEEP]` trigger it.
Standard `[NOVEL]` skips the audit step (too slow for everyday use).

---

## Conductor selection (updated `pick_conductor`)

```python
CONDUCTOR_PRIORITY = [
    # (alias, requires_thinking)
    ("nim-glm",    True),   # free, HIGH thinking budget
    ("nim-kimi",   True),   # free, HIGH thinking budget
    ("nim-minimax",True),   # free, HIGH thinking budget
    ("ant-opus",   False),  # flat-rate, tool access, MEDIUM thinking
    ("ant-fable",  False),  # flat-rate, deep reasoning, fallback only
]
```

Pick first healthy + available entry. `requires_thinking=True` means the model
should only be selected as conductor if thinking budget can be injected (Phase 1.6
already handles this via `_think_budget`).

---

## `[NOVEL DEEP]` / `[NOVEL TREE DEEP]` flow (upgraded)

```
[NOVEL DEEP] prompt
  │
  ├─ Committee fan-out (N=5 for DEEP)
  │    nim-glm  · think:HIGH  32k
  │    nim-kimi · think:HIGH  32k
  │    ant-sonnet · think:MEDIUM 8k   (if no healthy NIM for 5th slot)
  │    ...
  │
  ├─ Conductor synthesis
  │    1. pick_conductor → capable NIM (thinking:HIGH) → ant-opus (tools if needed)
  │    2. conductor scores drafts, synthesizes best answer
  │
  └─ Fable audit  (DEEP only)
       ant-fable · no tools · targeted fix pass
       → final answer
```

Estimated cost per `[NOVEL DEEP]` call (rough):
- Committee: ~0 (NIM free + flat-rate GO/Anthropic)
- Conductor: ~0 (NIM) or flat-rate (ant-opus)
- Fable audit: flat-rate Max, ~2-4k tokens per audit

---

## What needs to change in code

| File | Change |
|---|---|
| `fusion/conductor.py` | Update `pick_conductor()` to prefer ant-opus/ant-fable over Copilot fallback; inject thinking for NIM conductor |
| `fusion/fusion.py` | Add `_audit_with_fable()` step after conductor synthesis for DEEP mode |
| `fusion/abmcts.py` | Optionally: prefer thinking-capable models as high-capability prior arms |
| Tests | New tests for updated conductor selection + audit step |

**Prerequisite:** frontier bench results (ant-fable + HIGH thinking vs current champion)
must show quality at or above current TREE score (9.1) before this is worth building.

---

## When to implement

1. ✅ Phase 1.6 (adaptive thinking depth) — DONE
2. ⏳ Frontier bench — ant-fable+thinking vs nim-glm+thinking vs cop-opus (trigger explicitly)
3. ⏳ v2.2 conductor upgrade — after bench confirms quality ceiling
4. ⏳ v3 worker swarm — separate initiative
