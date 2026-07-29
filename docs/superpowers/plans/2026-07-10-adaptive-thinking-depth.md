# Adaptive Thinking Depth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inject provider-appropriate thinking budgets (medium/high) into requests based on routing tier and cost-class, with BOOST escalation and per-response visibility.

**Architecture:** Extend `priority_router.py` with model sets, `_think_budget()` lookup, injection in `async_pre_call_hook`, BOOST detection in `parse_request`, and a `async_post_call_success_hook` for response annotation. State written to `/tmp/llmr-last-think.json` for `/current` extension.

**Tech Stack:** Python stdlib + existing LiteLLM CustomLogger API · opencode.json commands

---

## File map

| File | Change |
|---|---|
| `priority_router.py` | Model sets, budget table, `_think_budget()`, BOOST detection, injection hook, annotation hook, state file |
| `tests/test_priority_router.py` | 12 new tests (budget table, BOOST, injection, annotation) |
| `scripts/show_routing.sh` | Read `/tmp/llmr-last-think.json`, append think info |
| `~/.config/opencode/opencode.json` | Add `/boost` command |

---

### Task 1: Model sets, budget table, and `_think_budget()`

**Files:**
- Modify: `priority_router.py` (after line 72, before `MODEL_PROVIDER`)
- Test: `tests/test_priority_router.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_priority_router.py`:

```python
# --- adaptive thinking depth ---

def test_think_budget_nim_reason_returns_high():
    assert pr._think_budget("nim-glm", "reason", False) == ("high", 32768)

def test_think_budget_nim_frontier_returns_high():
    assert pr._think_budget("nim-kimi", "frontier", False) == ("high", 32768)

def test_think_budget_nim_minimax_agent_returns_high():
    assert pr._think_budget("nim-minimax", "agent", False) == ("high", 32768)

def test_think_budget_go_reason_returns_medium():
    assert pr._think_budget("zen-glm", "reason", False) == ("medium", 8192)

def test_think_budget_go_frontier_returns_high():
    assert pr._think_budget("zen-kimi", "frontier", False) == ("high", 16384)

def test_think_budget_ant_reason_returns_medium():
    assert pr._think_budget("ant-opus", "reason", False) == ("medium", 8192)

def test_think_budget_ant_frontier_returns_high():
    assert pr._think_budget("ant-fable", "frontier", False) == ("high", 16384)

def test_think_budget_ant_haiku_excluded():
    # ant-haiku does not support thinking
    assert pr._think_budget("ant-haiku", "frontier", False) is None

def test_think_budget_cheap_tier_returns_none():
    assert pr._think_budget("nim-glm", "cheap", False) is None

def test_think_budget_general_tier_returns_none():
    assert pr._think_budget("zen-glm", "general", False) is None

def test_think_budget_non_thinking_model_returns_none():
    assert pr._think_budget("nim-deepseek", "reason", False) is None

def test_think_budget_boost_forces_high_on_go():
    assert pr._think_budget("zen-glm", "reason", True) == ("high", 16384)

def test_think_budget_boost_forces_high_on_ant():
    assert pr._think_budget("ant-sonnet", "agent", True) == ("high", 16384)
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
cd "<repo-root>"
python3 -m pytest tests/test_priority_router.py -k "think_budget" -v 2>&1 | tail -20
```

Expected: `AttributeError: module 'priority_router' has no attribute '_think_budget'`

- [ ] **Step 3: Implement model sets and `_think_budget()` in `priority_router.py`**

Add after line 72 (after `FRONTIER_TIER`), before `MODEL_PROVIDER`:

```python
# --- Thinking-capable model sets and budget table ---
NIM_THINKING = frozenset({"nim-glm", "nim-kimi", "nim-minimax"})
GO_THINKING  = frozenset({"zen-glm", "zen-kimi", "zen-minimax"})
ANT_THINKING = frozenset({"ant-opus", "ant-fable", "ant-sonnet"})  # haiku excluded
ALL_THINKING = NIM_THINKING | GO_THINKING | ANT_THINKING

# (class, level) -> budget_tokens
_THINK_TOKENS = {
    ("nim", "high"):    32768,
    ("go",  "medium"):   8192,
    ("go",  "high"):    16384,
    ("ant", "medium"):   8192,
    ("ant", "high"):    16384,
}

# tier -> (nim_level, go_level, ant_level);  absent = no thinking
_THINK_TABLE: Dict[str, Tuple[str, str, str]] = {
    "reason":   ("high", "medium", "medium"),
    "code":     ("high", "medium", "medium"),
    "agent":    ("high", "medium", "medium"),
    "frontier": ("high", "high",   "high"),
}

def _model_think_class(model: str) -> Optional[str]:
    if model in NIM_THINKING: return "nim"
    if model in GO_THINKING:  return "go"
    if model in ANT_THINKING: return "ant"
    return None


def _think_budget(model: str, tier: str, boost: bool) -> Optional[Tuple[str, int]]:
    """Return (level, budget_tokens) or None (model doesn't support thinking / tier is cheap)."""
    cls = _model_think_class(model)
    if cls is None:
        return None
    if boost:
        level = "high"
    else:
        row = _THINK_TABLE.get(tier)
        if row is None:
            return None
        level = row[{"nim": 0, "go": 1, "ant": 2}[cls]]
    return (level, _THINK_TOKENS[(cls, level)])
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python3 -m pytest tests/test_priority_router.py -k "think_budget" -v 2>&1 | tail -20
```

Expected: 13 passed

- [ ] **Step 5: Run full suite — no regressions**

```bash
python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: `63 passed`

- [ ] **Step 6: Commit**

```bash
cd "<repo-root>"
git add priority_router.py tests/test_priority_router.py
git commit -m "feat: thinking model sets + _think_budget() lookup table"
```

---

### Task 2: BOOST escalation signal detection

**Files:**
- Modify: `priority_router.py` (`TAG_RE`, `TAG_TIER`, `parse_request`)
- Test: `tests/test_priority_router.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_priority_router.py`:

```python
def test_boost_tag_sets_boost_directive():
    _, d = pr.parse_request("[BOOST] redo this answer")
    assert d["boost"] is True

def test_boost_autodetect_redo():
    _, d = pr.parse_request("redo this, the answer was wrong")
    assert d["boost"] is True

def test_boost_autodetect_shallow():
    _, d = pr.parse_request("too shallow, try again")
    assert d["boost"] is True

def test_no_boost_on_normal_prompt():
    _, d = pr.parse_request("explain recursion")
    assert d.get("boost") is False

def test_boost_tag_stripped_from_cleaned():
    cleaned, _ = pr.parse_request("[BOOST] explain this better")
    assert "[BOOST]" not in cleaned
    assert "explain this better" in cleaned

def test_boost_composes_with_tier_tag():
    cleaned, d = pr.parse_request("[REASON][BOOST] prove this theorem again")
    assert d["tier"] == "reason"
    assert d["boost"] is True
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python3 -m pytest tests/test_priority_router.py -k "boost" -v 2>&1 | tail -15
```

Expected: `KeyError: 'boost'` or `AssertionError`

- [ ] **Step 3: Add BOOST to TAG_RE, add BOOST_RE, update parse_request**

Replace `TAG_RE` line in `priority_router.py`:

```python
TAG_RE = re.compile(
    r"\[(CHEAP|THINK|REASON|CODE|AGENT|FRONTIER|BOOST|FUSION|NOVEL|DISCOVERY|NOFUSION)\b[^\]]*\]",
    re.IGNORECASE)
```

Add after `TAG_RE`:

```python
BOOST_RE = re.compile(
    r"\b(redo|not good|wrong answer|bad answer|shallow|try again|doesn't work|"
    r"not right|incorrect|improve this|too shallow)\b",
    re.IGNORECASE,
)
```

Replace `parse_request` function:

```python
def parse_request(prompt: str) -> Tuple[str, Dict[str, Any]]:
    directives: Dict[str, Any] = {"tier": None, "allowed": None, "denied": set(), "boost": False}
    for m in TAG_RE.finditer(prompt):
        tag = m.group(1).upper()
        if tag == "BOOST":
            directives["boost"] = True
        elif tag in TAG_TIER and directives["tier"] is None:
            directives["tier"] = TAG_TIER[tag]
    for m in AVAIL_RE.finditer(prompt):
        provs = {x.strip().lower() for x in m.group(2).split(",")}
        if m.group(1):
            directives["denied"] |= provs
        else:
            directives["allowed"] = provs
    if not directives["boost"]:
        directives["boost"] = bool(BOOST_RE.search(prompt))
    cleaned = AVAIL_RE.sub("", TAG_RE.sub("", prompt)).strip()
    return cleaned, directives
```

- [ ] **Step 4: Run BOOST tests**

```bash
python3 -m pytest tests/test_priority_router.py -k "boost" -v 2>&1 | tail -15
```

Expected: 6 passed

- [ ] **Step 5: Full suite — no regressions**

```bash
python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: `69 passed`

- [ ] **Step 6: Commit**

```bash
git add priority_router.py tests/test_priority_router.py
git commit -m "feat: BOOST escalation tag + autodetect (redo/shallow/wrong → high thinking)"
```

---

### Task 3: Inject thinking params in `async_pre_call_hook`

**Files:**
- Modify: `priority_router.py` (`PriorityRouter.async_pre_call_hook`, add `_write_think_state`)
- Test: `tests/test_priority_router.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_priority_router.py`:

```python
import asyncio

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

def test_hook_injects_ant_thinking_param():
    data = {"model": "ant-opus", "messages": [{"role": "user", "content": "[FRONTIER] design a system"}]}
    avail_all = {"nim": True, "zen": True, "copilot": True, "anthropic": True}
    # patch load_availability to return avail_all
    orig = pr.load_availability
    pr.load_availability = lambda: avail_all
    try:
        result = _run(pr.router_instance.async_pre_call_hook(None, None, data, "completion"))
    finally:
        pr.load_availability = orig
    assert "thinking" in result
    assert result["thinking"]["type"] == "thinking"
    assert result["thinking"]["budget_tokens"] == 16384  # ant frontier HIGH

def test_hook_injects_nim_extra_body():
    data = {"model": "auto", "messages": [{"role": "user", "content": "[REASON] prove the halting problem"}]}
    avail_all = {"nim": True, "zen": True, "copilot": True, "anthropic": True}
    orig = pr.load_availability
    pr.load_availability = lambda: avail_all
    try:
        result = _run(pr.router_instance.async_pre_call_hook(None, None, data, "completion"))
    finally:
        pr.load_availability = orig
    # auto routes to nim-qwen-max for reason tier (not in NIM_THINKING → no inject)
    # OR test with explicit nim-glm
    data2 = {"model": "nim-glm", "messages": [{"role": "user", "content": "[REASON] prove this"}]}
    pr.load_availability = lambda: avail_all
    try:
        result2 = _run(pr.router_instance.async_pre_call_hook(None, None, data2, "completion"))
    finally:
        pr.load_availability = orig
    assert result2.get("extra_body", {}).get("enable_thinking") is True
    assert result2["extra_body"]["thinking_budget_tokens"] == 32768

def test_hook_no_thinking_on_cheap_tier():
    data = {"model": "nim-glm", "messages": [{"role": "user", "content": "say hi"}]}
    avail_all = {"nim": True, "zen": True, "copilot": True, "anthropic": True}
    orig = pr.load_availability
    pr.load_availability = lambda: avail_all
    try:
        result = _run(pr.router_instance.async_pre_call_hook(None, None, data, "completion"))
    finally:
        pr.load_availability = orig
    assert "thinking" not in result
    assert "extra_body" not in result
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python3 -m pytest tests/test_priority_router.py -k "hook_injects or hook_no_thinking" -v 2>&1 | tail -15
```

Expected: AssertionError (thinking not in result)

- [ ] **Step 3: Add `_write_think_state` and injection to `async_pre_call_hook`**

Add after `_think_budget` function (before `MODEL_PROVIDER`):

```python
import json as _json

_THINK_STATE = Path("/tmp/llmr-last-think.json")

def _write_think_state(model: str, tier: str, level: str) -> None:
    try:
        _THINK_STATE.write_text(_json.dumps({"model": model, "tier": tier, "think": level}))
    except Exception:
        pass
```

Replace `PriorityRouter.async_pre_call_hook` — add thinking injection block after the model is determined. The final `data["model"]` is set by the existing logic; add this block just before `return data`:

```python
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        messages = data.get("messages") or []
        prompt, directives = "", {"tier": None, "allowed": None, "denied": set(), "boost": False}
        for msg in reversed(messages):
            if msg.get("role") == "user":
                raw = extract_text(msg.get("content", ""))
                cleaned, directives = parse_request(raw)
                if cleaned != raw and isinstance(msg.get("content"), str):
                    msg["content"] = cleaned
                prompt = cleaned or raw
                break
        availability = effective_availability(directives)
        health = load_health()
        requested = data.get("model", "") or ""
        if requested in AUTO_MODELS:
            target = route(prompt, directives, availability, health)
            if target:
                data["model"] = target
                verbose_logger.info("PriorityRouter: auto -> %s", target)
        else:
            provider = MODEL_PROVIDER.get(requested)
            unhealthy = (provider is not None and not availability.get(provider, True)) \
                or (health.get(requested, {}).get("ok") is False)
            if unhealthy:
                target = route(prompt, directives, availability, health)
                if target:
                    verbose_logger.info("PriorityRouter: %s unavailable -> %s", requested, target)
                    data["model"] = target

        # Inject thinking budget based on final model + tier
        final_model = data.get("model", "")
        tier = directives.get("tier") or classify(prompt)
        boost = directives.get("boost", False)
        think = _think_budget(final_model, tier, boost)
        if think:
            level, tokens = think
            if final_model in ANT_THINKING:
                data["thinking"] = {"type": "thinking", "budget_tokens": tokens}
            else:
                data.setdefault("extra_body", {}).update({
                    "enable_thinking": True,
                    "thinking_budget_tokens": tokens,
                })
            verbose_logger.info("PriorityRouter: think=%s budget=%d model=%s tier=%s",
                                level, tokens, final_model, tier)
            _write_think_state(final_model, tier, level)

        return data
```

- [ ] **Step 4: Run injection tests**

```bash
python3 -m pytest tests/test_priority_router.py -k "hook_injects or hook_no_thinking" -v 2>&1 | tail -15
```

Expected: 3 passed

- [ ] **Step 5: Full suite**

```bash
python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: `75 passed` (63 + 13 budget + 6 boost - overlap)

- [ ] **Step 6: Commit**

```bash
git add priority_router.py tests/test_priority_router.py
git commit -m "feat: inject thinking budgets in async_pre_call_hook (nim/go extra_body, ant thinking param)"
```

---

### Task 4: Response annotation via `async_post_call_success_hook`

**Files:**
- Modify: `priority_router.py` (add `async_post_call_success_hook` to `PriorityRouter`)

No tests for this task — the hook depends on LiteLLM runtime internals; tested live.

- [ ] **Step 1: Add `async_post_call_success_hook` to `PriorityRouter`**

Add after `async_pre_call_hook` (before `router_instance = PriorityRouter()`):

```python
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """Prepend [model · think:level · tier] annotation to non-streaming responses."""
        try:
            state = _json.loads(_THINK_STATE.read_text()) if _THINK_STATE.exists() else None
        except Exception:
            return response
        if not state:
            return response
        annotation = f"[{state['model']} · think:{state['think']} · {state['tier']}]\n\n"
        try:
            choices = getattr(response, "choices", None)
            if choices and hasattr(choices[0], "message"):
                msg = choices[0].message
                if getattr(msg, "content", None):
                    msg.content = annotation + msg.content
        except Exception:
            pass
        return response
```

- [ ] **Step 2: Full suite — hook must not break existing tests**

```bash
python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: same count as before, all passed

- [ ] **Step 3: Commit**

```bash
git add priority_router.py
git commit -m "feat: response annotation hook — prepend [model · think:level · tier]"
```

---

### Task 5: `/current` extension — show think state

**Files:**
- Modify: `scripts/show_routing.sh`

- [ ] **Step 1: Extend show_routing.sh to show think state**

Replace the full file:

```sh
#!/bin/sh
# Which model the router actually chose for recent requests (from Postgres audit).
# Usage: sh scripts/show_routing.sh [N|watch]
N="${1:-15}"
query() {
  docker exec litellm-db psql -U litellm -d litellm -P pager=off -c \
    "SELECT to_char(\"startTime\",'HH24:MI:SS') AS time, model_group AS routed_alias,
            model AS actual_model, \"total_tokens\" AS tokens
     FROM \"LiteLLM_SpendLogs\" ORDER BY \"startTime\" DESC LIMIT ${1:-15};"
}
show_think() {
  f="/tmp/llmr-last-think.json"
  if [ -f "$f" ]; then
    python3 -c "
import json, sys
d = json.load(open('$f'))
print(f\"  last think: {d['model']} · think:{d['think']} · tier:{d['tier']}\")
" 2>/dev/null
  fi
}
if [ "$N" = "watch" ]; then
  while true; do
    clear; echo "Live routing (Ctrl-C) — newest first"
    query 15
    show_think
    sleep 2
  done
else
  query "$N"
  show_think
fi
```

- [ ] **Step 2: Test manually**

```bash
sh "<repo-root>/scripts/show_routing.sh" 3
```

Expected: table output + `last think: nim-glm · think:high · tier:reason` (or "file not found" if no request made yet — that's fine)

- [ ] **Step 3: Commit**

```bash
git add scripts/show_routing.sh
git commit -m "feat: show_routing.sh shows last thinking budget (model · think:level · tier)"
```

---

### Task 6: opencode `/boost` command

**Files:**
- Modify: `~/.config/opencode/opencode.json`

- [ ] **Step 1: Add `/boost` command to opencode.json**

In `~/.config/opencode/opencode.json`, add to the `"command"` block (after `"performance"`):

```json
"boost": {
  "description": "Re-run with HIGH thinking budget ([BOOST] escalation for all thinking models)",
  "template": "[BOOST] $ARGUMENTS"
}
```

- [ ] **Step 2: Verify JSON is valid**

```bash
python3 -c "import json; json.load(open('<repo-root>')); print('valid')"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add "<repo-root>"
git commit -m "feat: /boost opencode command → [BOOST] $ARGUMENTS (high thinking budget escalation)"
```

---

### Task 7: Live smoke test + final commit

No code changes. Verify end-to-end in the running stack.

- [ ] **Step 1: Restart LiteLLM to pick up new priority_router.py**

```bash
cd "<repo-root>"
docker compose restart litellm
sleep 8
curl -s http://localhost:4040/health/liveliness
```

Expected: `{"status": "healthy"}`

- [ ] **Step 2: Send a REASON-tier prompt via curl and check thinking was injected**

```bash
KEY=$(python3 -c "
import re
t = open('.env').read()
print(re.search(r'LITELLM_MASTER_KEY=([^\n]+)', t).group(1).strip(\"'\\\""))
")
curl -s -X POST http://localhost:4040/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"nim-glm","messages":[{"role":"user","content":"[REASON] why is the sky blue"}],"max_tokens":200}' \
  | python3 -m json.tool 2>/dev/null | grep -E "content|think|model" | head -10
```

Expected: response content starts with `[nim-glm · think:high · reason]` annotation OR annotation absent for streaming — check docker logs for `think=high` log line:

```bash
docker compose logs litellm --tail=20 2>&1 | grep "think="
```

Expected: `PriorityRouter: think=high budget=32768 model=nim-glm tier=reason`

- [ ] **Step 3: Test BOOST escalation**

```bash
curl -s -X POST http://localhost:4040/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"zen-glm","messages":[{"role":"user","content":"[BOOST] redo the previous answer about sky color"}],"max_tokens":100}' \
  | python3 -m json.tool 2>/dev/null | grep "content" | head -5
docker compose logs litellm --tail=10 2>&1 | grep "think="
```

Expected log: `think=high budget=16384 model=zen-glm tier=general` (BOOST forces HIGH even on general)

- [ ] **Step 4: Run full test suite one final time**

```bash
python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: `75+ passed`

- [ ] **Step 5: Push to GitHub**

```bash
git push origin main
```

---

## Test count summary

| Task | New tests | Running total |
|---|---|---|
| Before | 0 | 63 |
| Task 1 — budget table | 13 | 76 |
| Task 2 — BOOST detection | 6 | 82 |
| Task 3 — injection hook | 3 | 85 |
| Task 4 — annotation | 0 (live only) | 85 |
| Tasks 5–7 | 0 | 85 |
