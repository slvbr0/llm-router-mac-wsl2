# Fusion / MoA v2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `[NOVEL]` fusion engine — a provider-diverse, difficulty-aware, single-round Mixture-of-Agents (panel → aggregator → answer + cost receipt), exposed as an MCP tool + CLI, with an A/B harness that measures "Fable-quality at a fraction of the cost" against a frontier baseline.

**Architecture:** `fusion/fusion.py` is a stdlib-only library (urllib + ThreadPoolExecutor + yaml) that classifies difficulty (reusing Phase-1 `priority_router.classify`), builds a health-aware provider-diverse panel from `fusion/fusion.yaml`, fans out in parallel through the Phase-1 router (`:4040`), aggregates with one strong model, and returns `{answer, receipt}`. A ~100-line stdio MCP server exposes it as a `fuse` tool to opencode; `fusion_bench.py` runs fusion vs a baseline model with a blind judge. `[NOVEL RESEARCH]` shells out to `pwm council`.

**Tech Stack:** Python 3.12 (stdlib + pyyaml; no new deps), Phase-1 LiteLLM router on `:4040`, pwm CLI for research mode, pytest.

**Spec:** [../specs/2026-07-06-fusion-moa-v2-design.md](../specs/2026-07-06-fusion-moa-v2-design.md)
**Project root:** `<repo-root>` — `$PROJ` below. Router must be up (`llmr-start up`).

---

## File structure

| File | Responsibility |
|---|---|
| `fusion/fusion.yaml` | panel profiles, aggregator candidates, cost table, bench defaults |
| `fusion/fusion.py` | engine: tag parse → classify → panel → fan-out → aggregate → receipt; CLI |
| `fusion/mcp_server.py` | stdio MCP server exposing `fuse(prompt, mode, depth)` |
| `fusion/fusion_bench.py` | A/B harness: fusion vs baseline, blind judge, summary table |
| `fusion/prompts.sample.txt` | 8 eval prompts |
| `tests/test_fusion.py` | unit tests (HTTP stubbed via monkeypatch) |

---

## Task 1: fusion.yaml + panel-builder tests

**Files:**
- Create: `$PROJ/fusion/fusion.yaml`
- Test: `$PROJ/tests/test_fusion.py`

- [ ] **Step 1: Write `fusion/fusion.yaml`**

```yaml
# Fusion v2.0 config — panels are provider-diverse (MoA's edge) and tunable here.
panels:
  easy:
    proposers: [nim-glm, zen-free-deepseek, nim-kimi]
    aggregator: [nim-glm, nim-mistral, cop-sonnet]        # first healthy wins
  hard:
    proposers: [nim-deepseek, nim-qwen-max, zen-gpt, cop-sonnet, nim-glm]
    aggregator: [nim-qwen-max, cop-opus, cop-sonnet]
research:
  council_models: "gpt54,claude_sonnet,gemini_pro"        # pwm council -m
  chairman: sonar
min_proposers: 2
proposer_timeout_s: 90
cost:                       # receipt estimation
  zen_paid_usd_per_1m: 3.0
  copilot_usd_per_credit: 0.04
  zen_paid_aliases: [zen-gpt, zen-glm, zen-deepseek]
bench:
  baseline: cop-opus
  judge: nim-qwen-max
```

- [ ] **Step 2: Write failing panel-builder tests** in `tests/test_fusion.py` (the existing `tests/conftest.py` already stubs `litellm`, which `priority_router` — imported by fusion — needs):

```python
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
fu = importlib.import_module("fusion.fusion")

ALL_OK = {"nim": True, "zen": True, "copilot": True}
CFG = fu.load_config()

def test_easy_panel_is_provider_diverse():
    p = fu.build_panel("easy", CFG, ALL_OK, {})
    provs = {fu.provider_of(a) for a in p["proposers"]}
    assert len(p["proposers"]) >= 2 and len(provs) >= 2

def test_hard_panel_larger_than_easy():
    e = fu.build_panel("easy", CFG, ALL_OK, {})
    h = fu.build_panel("hard", CFG, ALL_OK, {})
    assert len(h["proposers"]) > len(e["proposers"])

def test_panel_skips_unhealthy_nim():
    health = {"nim-glm": {"ok": False}}
    p = fu.build_panel("easy", CFG, ALL_OK, health)
    assert "nim-glm" not in p["proposers"]

def test_panel_skips_masked_provider():
    p = fu.build_panel("easy", CFG, {"nim": True, "zen": False, "copilot": True}, {})
    assert all(fu.provider_of(a) != "zen" for a in p["proposers"])

def test_aggregator_first_healthy():
    health = {"nim-qwen-max": {"ok": False}}
    p = fu.build_panel("hard", CFG, ALL_OK, health)
    assert p["aggregator"] == "cop-opus"

def test_parse_novel_tags():
    assert fu.parse_novel("[NOVEL] solve X") == ("solve X", "fuse", None)
    assert fu.parse_novel("[NOVEL DEEP] solve X") == ("solve X", "fuse", "hard")
    assert fu.parse_novel("[NOVEL RESEARCH] what is new in Y") == ("what is new in Y", "research", None)
    assert fu.parse_novel("plain prompt") == ("plain prompt", None, None)

def test_difficulty_maps_tiers():
    assert fu.difficulty_of("say hi") == "easy"                       # cheap tier
    assert fu.difficulty_of("[THINK] prove theorem") in ("easy","hard")  # tag stripped upstream; classify on text
    assert fu.difficulty_of("prove this theorem about complexity") == "hard"
```

- [ ] **Step 3: Run — verify fail**

Run: `cd "$PROJ" && python3 -m pytest tests/test_fusion.py -q`
Expected: ERROR (`fusion.fusion` not found).

- [ ] **Step 4: Commit**

```bash
cd "$PROJ" && git add fusion/fusion.yaml tests/test_fusion.py
git commit -m "test: fusion panel-builder + tag-parse specs (failing)"
```

---

## Task 2: fusion.py — engine core (make tests pass)

**Files:**
- Create: `$PROJ/fusion/fusion.py`, `$PROJ/fusion/__init__.py` (empty)

- [ ] **Step 1: Write `fusion/fusion.py`**

```python
"""Fusion v2.0 — provider-diverse single-round Mixture-of-Agents over the Phase-1 router.

fuse(prompt) : classify difficulty -> build health-aware panel -> parallel drafts via :4040
               -> aggregator synthesizes -> {answer, receipt}. [NOVEL RESEARCH] -> pwm council.
Stdlib-only (urllib, ThreadPoolExecutor) + pyyaml. CLI: python3 -m fusion.fusion "[NOVEL] ..."
"""
import json, os, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from priority_router import classify, load_availability, load_health, MODEL_PROVIDER  # noqa: E402

NOVEL_RE = re.compile(r"\[NOVEL(\s+(DEEP|RESEARCH))?\]", re.IGNORECASE)
ROUTER = "http://localhost:4040/v1/chat/completions"


def load_env():
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def load_config():
    return yaml.safe_load((ROOT / "fusion" / "fusion.yaml").read_text())


def provider_of(alias):
    return MODEL_PROVIDER.get(alias, "")


def parse_novel(prompt):
    m = NOVEL_RE.search(prompt)
    if not m:
        return prompt.strip(), None, None
    cleaned = NOVEL_RE.sub("", prompt).strip()
    kind = (m.group(2) or "").upper()
    if kind == "RESEARCH":
        return cleaned, "research", None
    return cleaned, "fuse", ("hard" if kind == "DEEP" else None)


def difficulty_of(prompt):
    tier = classify(prompt)
    return "easy" if tier in ("cheap", "general") else "hard"


def _ok(alias, availability, health):
    if not availability.get(provider_of(alias), False):
        return False
    h = health.get(alias)
    return not (h is not None and h.get("ok") is False)


def build_panel(difficulty, cfg, availability, health):
    prof = cfg["panels"][difficulty]
    proposers = [a for a in prof["proposers"] if _ok(a, availability, health)]
    aggregator = next((a for a in prof["aggregator"] if _ok(a, availability, health)), None)
    return {"proposers": proposers, "aggregator": aggregator}


def call_model(alias, messages, key, timeout):
    body = json.dumps({"model": alias, "messages": messages}).encode()
    req = urllib.request.Request(ROUTER, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
        u = d.get("usage", {}) or {}
        return {"alias": alias, "provider": provider_of(alias), "ok": True,
                "content": d["choices"][0]["message"]["content"] or "",
                "tokens": u.get("total_tokens", 0),
                "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"alias": alias, "provider": provider_of(alias), "ok": False,
                "content": "", "tokens": 0, "error": str(e)[:120],
                "latency_ms": int((time.time() - t0) * 1000)}


def estimate_cost(entries, cfg):
    c = cfg["cost"]
    zen_paid = set(c["zen_paid_aliases"])
    free_t = sum(e["tokens"] for e in entries if e["provider"] == "nim"
                 or (e["provider"] == "zen" and e["alias"] not in zen_paid))
    zen_t = sum(e["tokens"] for e in entries if e["alias"] in zen_paid)
    credits = sum(1 for e in entries if e["provider"] == "copilot" and e["ok"])
    usd = zen_t / 1e6 * c["zen_paid_usd_per_1m"] + credits * c["copilot_usd_per_credit"]
    return {"free_tokens": free_t, "zen_paid_tokens": zen_t,
            "copilot_credits": credits, "pwm_searches": 0, "usd_estimate": round(usd, 4)}


AGG_PROMPT = ("You are the aggregator of a model committee. Below is a user task and several "
              "independent drafts. Reconcile them: resolve disagreements, take the strongest "
              "reasoning, fix errors, and produce the single best final answer. Do NOT mention "
              "the drafts or the committee; just answer the task.\n\nTASK:\n{task}\n\n{drafts}")


def fuse(prompt, mode=None, depth=None, confirm_research=False):
    t0 = time.time()
    cleaned, tag_mode, tag_depth = parse_novel(prompt)
    mode = mode or tag_mode or "fuse"
    cfg = load_config()
    if mode == "research":
        return _research(cleaned, cfg, t0, confirm_research)
    env = load_env(); key = env.get("LITELLM_MASTER_KEY", "")
    difficulty = depth or tag_depth or difficulty_of(cleaned)
    availability, health = load_availability(), load_health()
    panel = build_panel(difficulty, cfg, availability, health)
    receipt = {"mode": "fuse", "difficulty": difficulty, "proposers": [], "degraded": False}
    msgs = [{"role": "user", "content": cleaned}]
    tmo = cfg["proposer_timeout_s"]

    if len(panel["proposers"]) >= cfg["min_proposers"]:
        with ThreadPoolExecutor(max_workers=len(panel["proposers"])) as ex:
            drafts = list(ex.map(lambda a: call_model(a, msgs, key, tmo), panel["proposers"]))
        receipt["proposers"] = drafts
        good = [d for d in drafts if d["ok"] and d["content"].strip()]
    else:
        good = []

    if len(good) < cfg["min_proposers"]:                       # degrade: single best model
        receipt["degraded"] = True
        fallback = panel["aggregator"] or "auto"
        one = call_model(fallback, msgs, key, tmo)
        receipt["aggregator"] = one
        answer = one["content"] if one["ok"] else "(fusion failed: no models reachable)"
    else:
        dtxt = "\n\n".join(f"--- DRAFT {i+1} ({d['alias']}) ---\n{d['content']}"
                           for i, d in enumerate(good))
        agg_msgs = [{"role": "user", "content": AGG_PROMPT.format(task=cleaned, drafts=dtxt)}]
        agg = call_model(panel["aggregator"] or "auto", agg_msgs, key, tmo)
        receipt["aggregator"] = agg
        answer = agg["content"] if agg["ok"] and agg["content"].strip() else \
            max(good, key=lambda d: len(d["content"]))["content"]   # aggregator failed -> best draft
        if not agg["ok"]:
            receipt["degraded"] = True

    entries = receipt["proposers"] + [receipt.get("aggregator", {"tokens": 0, "provider": "", "alias": "", "ok": False})]
    receipt["total_tokens"] = sum(e.get("tokens", 0) for e in entries)
    receipt["est_cost"] = estimate_cost([e for e in entries if e.get("alias")], cfg)
    receipt["wall_ms"] = int((time.time() - t0) * 1000)
    _log(receipt)
    return {"answer": answer, "receipt": receipt}


def _research(cleaned, cfg, t0, confirm):
    r = cfg["research"]
    n = len(r["council_models"].split(",")) + 1               # models + synthesis
    if not confirm:
        return {"answer": f"[NOVEL RESEARCH] would spend ~{n} Pro Searches (pwm council: "
                          f"{r['council_models']}). Re-call with confirm_research=true to proceed.",
                "receipt": {"mode": "research", "confirmed": False, "pwm_searches_needed": n}}
    try:
        out = subprocess.run(["pwm", "council", cleaned, "-m", r["council_models"],
                              "--chairman", r["chairman"], "--json"],
                             capture_output=True, text=True, timeout=600,
                             env={**os.environ, "PATH": os.environ.get("PATH", "") + ":" +
                                  str(Path.home() / ".local/bin")})
        answer = out.stdout.strip() or out.stderr.strip()[:500]
    except Exception as e:
        answer = f"(pwm council failed: {e})"
    receipt = {"mode": "research", "confirmed": True, "pwm_searches": n,
               "est_cost": {"free_tokens": 0, "zen_paid_tokens": 0, "copilot_credits": 0,
                            "pwm_searches": n, "usd_estimate": 0.0},
               "wall_ms": int((time.time() - t0) * 1000)}
    _log(receipt)
    return {"answer": answer, "receipt": receipt}


def _log(receipt):
    logs = ROOT / "logs"; logs.mkdir(exist_ok=True)
    with open(logs / f"fusion-{time.strftime('%Y%m%d')}.jsonl", "a") as f:
        f.write(json.dumps(receipt) + "\n")


if __name__ == "__main__":
    res = fuse(" ".join(sys.argv[1:]) or "[NOVEL] say hello",
               confirm_research="--confirm" in sys.argv)
    print(res["answer"])
    r = res["receipt"]
    print(f"\n--- receipt: mode={r.get('mode')} diff={r.get('difficulty','-')} "
          f"tokens={r.get('total_tokens','-')} cost={r.get('est_cost',{})} "
          f"wall={r.get('wall_ms','-')}ms degraded={r.get('degraded',False)}", file=sys.stderr)
```

Also: `touch "$PROJ/fusion/__init__.py"`

- [ ] **Step 2: Run tests — verify pass**

Run: `cd "$PROJ" && python3 -m pytest tests/test_fusion.py -q`
Expected: all PASS (7 tests).

- [ ] **Step 3: Commit**

```bash
cd "$PROJ" && git add fusion/ && git commit -m "feat: fusion engine — panel builder, fan-out, aggregator, receipt"
```

---

## Task 3: fuse() behavior tests (stubbed HTTP)

**Files:**
- Modify: `$PROJ/tests/test_fusion.py` (append)

- [ ] **Step 1: Append stubbed end-to-end tests**

```python
def _stub_call(results):
    """Return a fake call_model keyed by alias; unknown alias -> failure."""
    def fake(alias, messages, key, timeout):
        r = results.get(alias)
        if r is None:
            return {"alias": alias, "provider": fu.provider_of(alias), "ok": False,
                    "content": "", "tokens": 0, "latency_ms": 1, "error": "stub"}
        return {"alias": alias, "provider": fu.provider_of(alias), "ok": True,
                "content": r, "tokens": 10, "latency_ms": 1}
    return fake

def test_fuse_happy_path_aggregates(monkeypatch, tmp_path):
    monkeypatch.setattr(fu, "ROOT", fu.ROOT)  # keep config
    monkeypatch.setattr(fu, "load_availability", lambda: dict(ALL_OK))
    monkeypatch.setattr(fu, "load_health", lambda: {})
    monkeypatch.setattr(fu, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(fu, "_log", lambda r: None)
    monkeypatch.setattr(fu, "call_model", _stub_call({
        "nim-glm": "draft A", "zen-free-deepseek": "draft B", "nim-kimi": "draft C"}))
    out = fu.fuse("[NOVEL] short task")
    # aggregator (nim-glm) was stubbed too -> its content returned
    assert out["answer"] == "draft A"
    assert out["receipt"]["degraded"] is False
    assert len(out["receipt"]["proposers"]) == 3
    assert out["receipt"]["est_cost"]["usd_estimate"] == 0.0   # all free

def test_fuse_degrades_when_proposers_fail(monkeypatch):
    monkeypatch.setattr(fu, "load_availability", lambda: dict(ALL_OK))
    monkeypatch.setattr(fu, "load_health", lambda: {})
    monkeypatch.setattr(fu, "load_env", lambda: {"LITELLM_MASTER_KEY": "k"})
    monkeypatch.setattr(fu, "_log", lambda r: None)
    monkeypatch.setattr(fu, "call_model", _stub_call({"nim-glm": "solo answer"}))
    out = fu.fuse("[NOVEL] short task")
    assert out["receipt"]["degraded"] is True
    assert out["answer"] == "solo answer"

def test_research_requires_confirm(monkeypatch):
    monkeypatch.setattr(fu, "_log", lambda r: None)
    out = fu.fuse("[NOVEL RESEARCH] what is new")
    assert out["receipt"]["confirmed"] is False
    assert "Pro Search" in out["answer"]

def test_copilot_cost_counted(monkeypatch):
    cfg = fu.load_config()
    entries = [{"alias": "cop-sonnet", "provider": "copilot", "ok": True, "tokens": 500},
               {"alias": "nim-glm", "provider": "nim", "ok": True, "tokens": 400},
               {"alias": "zen-gpt", "provider": "zen", "ok": True, "tokens": 1000}]
    c = fu.estimate_cost(entries, cfg)
    assert c["copilot_credits"] == 1 and c["zen_paid_tokens"] == 1000 and c["free_tokens"] == 400
    assert c["usd_estimate"] > 0
```

- [ ] **Step 2: Run — all pass**

Run: `cd "$PROJ" && python3 -m pytest tests/test_fusion.py -q`
Expected: 11 PASS.

- [ ] **Step 3: Commit**

```bash
cd "$PROJ" && git add tests/test_fusion.py && git commit -m "test: fuse happy-path, degrade, research-confirm, cost math"
```

---

## Task 4: MCP server

**Files:**
- Create: `$PROJ/fusion/mcp_server.py`

- [ ] **Step 1: Write the stdio MCP server** (JSON-RPC over stdin/stdout; protocol subset: initialize / tools/list / tools/call):

```python
"""Minimal stdio MCP server exposing the fusion engine as one tool: fuse."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion.fusion import fuse  # noqa: E402

TOOL = {
    "name": "fuse",
    "description": ("Mixture-of-Agents fusion for [NOVEL] prompts: fans one task to a "
                    "provider-diverse model panel via the local router and synthesizes the best "
                    "answer, returning a cost receipt. mode='research' uses Perplexity council "
                    "(web-grounded; requires confirm_research=true after quota warning)."),
    "inputSchema": {"type": "object", "properties": {
        "prompt": {"type": "string", "description": "The task (with or without [NOVEL] tags)"},
        "mode": {"type": "string", "enum": ["fuse", "research"]},
        "depth": {"type": "string", "enum": ["easy", "hard"]},
        "confirm_research": {"type": "boolean"}},
        "required": ["prompt"]},
}


def reply(id_, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": id_, "result": result}) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid, method, params = msg.get("id"), msg.get("method", ""), msg.get("params", {}) or {}
        if method == "initialize":
            reply(mid, {"protocolVersion": params.get("protocolVersion", "2025-03-26"),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fusion", "version": "2.0"}})
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            reply(mid, {"tools": [TOOL]})
        elif method == "tools/call" and params.get("name") == "fuse":
            a = params.get("arguments", {}) or {}
            try:
                res = fuse(a["prompt"], mode=a.get("mode"), depth=a.get("depth"),
                           confirm_research=bool(a.get("confirm_research")))
                r = res["receipt"]
                summary = (f"[fusion receipt] mode={r.get('mode')} diff={r.get('difficulty','-')} "
                           f"tokens={r.get('total_tokens','-')} est_cost={r.get('est_cost',{})} "
                           f"wall={r.get('wall_ms','-')}ms degraded={r.get('degraded', False)}")
                reply(mid, {"content": [{"type": "text", "text": res["answer"] + "\n\n" + summary}]})
            except Exception as e:
                reply(mid, {"content": [{"type": "text", "text": f"fusion error: {e}"}],
                            "isError": True})
        elif mid is not None:
            reply(mid, {})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the protocol by hand**

```bash
cd "$PROJ" && printf '%s\n%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
 | python3 fusion/mcp_server.py
```
Expected: two JSON lines — serverInfo `fusion`, then a tools array containing `fuse`.

- [ ] **Step 3: Commit**

```bash
cd "$PROJ" && git add fusion/mcp_server.py && git commit -m "feat: stdio MCP server exposing fuse tool"
```

---

## Task 5: A/B harness

**Files:**
- Create: `$PROJ/fusion/fusion_bench.py`, `$PROJ/fusion/prompts.sample.txt`

- [ ] **Step 1: Write `fusion/prompts.sample.txt`** (one prompt per line; `#` = comment)

```
# hard reasoning
Prove or disprove: every bounded monotone sequence of real numbers converges. Explain rigorously but concisely.
Design a rate limiter for a distributed API gateway: compare token bucket vs sliding window log, pick one, justify, and sketch the data model.
A social app's feed ranking causes engagement but harms wellbeing metrics. Propose a ranking objective that trades these off, with the math and its failure modes.
Debug this: a Python asyncio service leaks memory only under high concurrency; list the 5 most likely causes ranked, with how to confirm each.
# general/creative
Explain how a Mixture-of-Agents system can outperform its strongest member, and when it cannot.
Write a 150-word executive summary arguing for (or against) self-hosting an LLM router vs using a single provider.
# code
Implement an LRU cache with TTL in Python without using functools; include complexity analysis.
Given a Postgres table of (ts, alias, tokens), write one SQL query returning each alias's share of total tokens per day for the last 7 days.
```

- [ ] **Step 2: Write `fusion/fusion_bench.py`**

```python
"""A/B harness: fusion committee vs single frontier baseline, blind-judged.
Usage: python3 fusion/fusion_bench.py fusion/prompts.sample.txt [--baseline cop-opus] [--judge nim-qwen-max]
"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion.fusion import fuse, call_model, load_env, load_config  # noqa: E402

JUDGE_PROMPT = ("You are a strict evaluator. TASK:\n{task}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}\n\n"
                "Score each answer 1-10 for correctness, depth and clarity. Respond ONLY with "
                'JSON: {{"a": <int>, "b": <int>}}')


def judge(task, ans_a, ans_b, judge_alias, key):
    r = call_model(judge_alias, [{"role": "user", "content":
                                  JUDGE_PROMPT.format(task=task, a=ans_a[:6000], b=ans_b[:6000])}],
                   key, 120)
    try:
        s = json.loads(r["content"][r["content"].find("{"):r["content"].rfind("}") + 1])
        return int(s["a"]), int(s["b"])
    except Exception:
        return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts"); ap.add_argument("--baseline"); ap.add_argument("--judge")
    args = ap.parse_args()
    cfg = load_config(); env = load_env(); key = env.get("LITELLM_MASTER_KEY", "")
    baseline = args.baseline or cfg["bench"]["baseline"]
    judge_alias = args.judge or cfg["bench"]["judge"]
    prompts = [l.strip() for l in Path(args.prompts).read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    rows, fq, bq, fcost, bcred = [], [], [], 0.0, 0
    for i, p in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] fusing…", file=sys.stderr)
        f = fuse("[NOVEL] " + p)
        b = call_model(baseline, [{"role": "user", "content": p}], key, 180)
        swap = random.random() < 0.5                       # blind: random order
        a1, a2 = (b["content"], f["answer"]) if swap else (f["answer"], b["content"])
        sa, sb = judge(p, a1, a2, judge_alias, key)
        fs, bs = (sb, sa) if swap else (sa, sb)
        fq.append(fs); bq.append(bs)
        fcost += f["receipt"]["est_cost"]["usd_estimate"]; bcred += 1
        rows.append((p[:50], fs, bs, f["receipt"]["total_tokens"], b["tokens"]))
    print(f"\n{'prompt':52} {'fus':>3} {'base':>4} {'f_tok':>6} {'b_tok':>6}")
    for r in rows:
        print(f"{r[0]:52} {r[1]:>3} {r[2]:>4} {r[3]:>6} {r[4]:>6}")
    n = max(len(prompts), 1)
    favg, bavg = sum(fq) / n, sum(bq) / n
    bcost = bcred * cfg["cost"]["copilot_usd_per_credit"]
    print(f"\nFUSION  avg quality {favg:.2f}  est cost ${fcost:.4f}"
          f"  → $/point {fcost / max(favg, .1):.4f}")
    print(f"BASELINE({baseline}) avg quality {bavg:.2f}  est cost ${bcost:.4f}"
          f"  → $/point {bcost / max(bavg, .1):.4f}")
    print("\nVERDICT:", "fusion matches/beats baseline at lower cost — hypothesis SUPPORTED"
          if favg >= bavg - 0.5 and fcost < bcost else
          "fusion does NOT beat baseline on cost/quality — hypothesis NOT supported (honest result)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
cd "$PROJ" && git add fusion/fusion_bench.py fusion/prompts.sample.txt
git commit -m "feat: A/B harness — fusion vs baseline, blind judge, verdict"
```

---

## Task 6: Live integration test (router must be up)

- [ ] **Step 1: One real easy fuse via CLI**

```bash
cd "$PROJ" && sh scripts/nim_health.sh >/dev/null 2>&1
python3 -m fusion.fusion "[NOVEL] In two sentences: why can a committee of models beat its strongest member?"
```
Expected: a coherent synthesized answer on stdout; a receipt line on stderr with `mode=fuse diff=easy`, non-zero tokens, `usd_estimate` ≈ 0.0 (free panel), `degraded=False`. Check `logs/fusion-*.jsonl` got a line.

- [ ] **Step 2: One real hard fuse (bigger panel; may spend 1 Copilot credit + small Zen)**

```bash
python3 -m fusion.fusion "[NOVEL DEEP] Design a minimal experiment to test whether a 4-model committee of cheap LLMs matches one frontier model. Define metric, judge protocol, and confounders."
```
Expected: answer + receipt `diff=hard`, ≥3 proposers, cost shows any copilot/zen spend.

- [ ] **Step 3: Verify router audit saw the fan-out**

Run: `sh scripts/show_routing.sh 10`
Expected: rows for the panel aliases within the last minutes (nim-*/zen-*/cop-* mix).

- [ ] **Step 4: Commit any fixes; else proceed**

```bash
cd "$PROJ" && git add -A && git commit -m "fix: live fuse adjustments" || echo "clean"
```

---

## Task 7: Wire into opencode + AGENTS.md

**Files:**
- Modify: `~/.config/opencode/opencode.json` (add `mcp.fusion`)
- Modify: `~/.config/opencode/AGENTS.md` (append rule)

- [ ] **Step 1: Add the fusion MCP**

```bash
python3 - <<'PY'
import json, os
p = os.path.expanduser("~/.config/opencode/opencode.json")
d = json.load(open(p))
d.setdefault("mcp", {})["fusion"] = {
  "type": "local",
  "command": ["python3", "<repo-root>/fusion/mcp_server.py"],
  "enabled": True}
json.dump(d, open(p, "w"), indent=2)
print("added mcp.fusion")
PY
```

- [ ] **Step 2: Append the trigger rule to AGENTS.md**

```bash
cat >> ~/.config/opencode/AGENTS.md <<'EOF'

## Fusion — [NOVEL] Mixture-of-Agents (opt-in, tag-gated)
When the user's prompt contains [NOVEL], [NOVEL DEEP] or [NOVEL RESEARCH], call the `fuse` MCP tool with the FULL prompt (tags included — the engine parses them). Return its answer, then the one-line receipt summary. Rules:
- NEVER fire fusion without one of these tags (it multiplies model calls).
- [NOVEL RESEARCH] costs ~4 Perplexity Pro Searches: the tool returns a quota warning first — relay it, and only re-call with confirm_research=true after the user agrees.
- If the tool errors, fall back to answering normally via llm-router/auto and say fusion was unavailable.
EOF
echo appended
```

- [ ] **Step 3: Verify opencode loads it**

```bash
export PATH="$HOME/.opencode/bin:$PATH"
opencode run -m llm-router/auto "[NOVEL] one sentence: what is 2+2 and why?" 2>&1 | tail -6
```
Expected: answer produced via the fuse tool (receipt line visible) — or at minimum opencode starts with no MCP errors and `fusion` listed. (Agent adherence can vary; the tag rule is in AGENTS.md.)

- [ ] **Step 4: Commit (project side only — configs are outside repo)**

```bash
cd "$PROJ" && git add -A && git commit -m "chore: fusion wiring artifacts" || echo clean
```

---

## Task 8: Run the hypothesis benchmark + document

- [ ] **Step 1: Full bench run (8 prompts; spends ~8 Copilot credits for baseline + judge tokens on NIM)**

```bash
cd "$PROJ" && python3 fusion/fusion_bench.py fusion/prompts.sample.txt 2>&1 | tee logs/bench-$(date +%Y%m%d).txt
```
Expected: per-prompt table + FUSION vs BASELINE summary + VERDICT line.

- [ ] **Step 2: Record the result in the README** — append under "Status & roadmap":

```markdown
**Phase 2 v2.0 (fusion/MoA):** built — `[NOVEL]`/`[NOVEL DEEP]`/`[NOVEL RESEARCH]` via the
`fuse` MCP tool + `python3 -m fusion.fusion` CLI. First benchmark (logs/bench-*.txt):
<one line: fusion avg X vs baseline Y, cost $A vs $B — verdict>.
```
(Fill the line from the actual bench output — honest numbers, win or lose.)

- [ ] **Step 3: Final commit + push**

```bash
cd "$PROJ" && git add -A
git commit -m "feat: fusion v2.0 complete — engine, MCP tool, bench + first results"
git push origin main
```

---

## Self-review (spec coverage)

- §3 architecture / §4.1-4.5 engine → Tasks 1-3 ✅ · §4.6 error handling → Task 3 (degrade tests) + engine fallbacks ✅
- §5 harness → Task 5 + run in Task 8 ✅ · §6 files → all created ✅ · §7 opencode wiring + AGENTS rule → Task 7 ✅
- §8 testing (unit stubbed / integration / harness smoke) → Tasks 1,3 / 6 / 8 ✅
- §10 cost discipline → receipts (Task 2), research confirm-gate (Tasks 2,3,7) ✅
- v2.1 swarm intentionally absent (spec §9) ✅
```
