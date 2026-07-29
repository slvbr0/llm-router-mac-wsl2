"""Fusion v2.0 — provider-diverse single-round Mixture-of-Agents over the Phase-1 router.

fuse(prompt) : classify difficulty -> build health-aware panel -> parallel drafts via :4040
               -> aggregator synthesizes -> {answer, receipt}. [NOVEL RESEARCH] -> pwm council.
Stdlib-only (urllib, ThreadPoolExecutor) + pyyaml. CLI: python3 -m fusion.fusion "[NOVEL] ..."
"""
import json, os, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from priority_router import classify, load_availability, load_health, MODEL_PROVIDER  # noqa: E402

NOVEL_RE = re.compile(r"\[NOVEL(\s+(TREE\s+DEEP|TREE|DEEP|RESEARCH))?\]", re.IGNORECASE)
ROUTER = "http://localhost:4040/v1/chat/completions"

# The router annotates every non-streaming response with "[model · think:level · tier]".
# That is for humans reading a chat. Fusion feeds model output straight back into other
# models (drafts -> conductor, candidates -> judge), so the banner must come off or the
# conductor reads it as part of the draft and the judge sees model identity, which would
# break blind pairwise scoring.
_ANN_RE = re.compile(r"^\[[^\]\n]*·[^\]\n]*·[^\]\n]*\]\s*\n+")


def strip_annotation(text: str) -> str:
    return _ANN_RE.sub("", text or "", count=1)


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
    kind = " ".join((m.group(2) or "").upper().split())      # normalize whitespace
    if kind == "RESEARCH":
        return cleaned, "research", None
    if kind == "TREE":
        return cleaned, "tree", None
    if kind == "TREE DEEP":
        return cleaned, "tree", "deep"
    return cleaned, "fuse", ("deep" if kind == "DEEP" else None)


def difficulty_of(prompt):
    tier = classify(prompt)
    return "easy" if tier in ("cheap", "general") else "hard"


def _ok(alias, availability, health):
    if not availability.get(provider_of(alias), False):
        return False
    h = health.get(alias)
    return not (h is not None and h.get("ok") is False)


def build_panel(difficulty, cfg, availability, health):
    """Health-aware free+paid mix: check availability, keep `paid_anchor` paid models
    (quality/diversity), fill the rest from healthy FREE models, backfill from paid only
    when free is short. Cheap when free NIM is healthy; more paid when it's degraded."""
    prof = cfg["panels"][difficulty]
    free_ok = [a for a in prof.get("free", []) if _ok(a, availability, health)]
    paid_ok = [a for a in prof.get("paid", []) if _ok(a, availability, health)]
    size, anchor = prof["size"], prof.get("paid_anchor", 0)

    proposers = list(paid_ok[:anchor])                      # 1) paid anchors
    for a in free_ok:                                       # 2) fill with free
        if len(proposers) >= size:
            break
        proposers.append(a)
    for a in paid_ok[anchor:]:                              # 3) backfill paid if free short
        if len(proposers) >= size:
            break
        if a not in proposers:
            proposers.append(a)

    aggregator = next((a for a in prof["aggregator"] if _ok(a, availability, health)), None)
    return {"proposers": proposers, "aggregator": aggregator}


def call_model(alias, messages, key, timeout, max_tokens=None, force_tier=None):
    payload = {"model": alias, "messages": messages}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if force_tier:
        payload.setdefault("metadata", {})["force_tier"] = force_tier
    body = json.dumps(payload).encode()
    req = urllib.request.Request(ROUTER, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
        u = d.get("usage", {}) or {}
        # in/out split is needed for list-price costing: output runs 3-5x input, and extended
        # thinking tokens bill as OUTPUT (they are already counted inside completion_tokens).
        # served_model exposes what actually answered -- a litellm fallback can quietly
        # substitute a different provider for the alias we asked for.
        return {"alias": alias, "provider": provider_of(alias), "ok": True,
                "content": strip_annotation(d["choices"][0]["message"]["content"] or ""),
                "tokens": u.get("total_tokens", 0),
                "tok_in": u.get("prompt_tokens", 0),
                "tok_out": u.get("completion_tokens", 0),
                "served_model": d.get("model", ""),
                "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"alias": alias, "provider": provider_of(alias), "ok": False,
                "content": "", "tokens": 0, "tok_in": 0, "tok_out": 0,
                "served_model": "", "error": str(e)[:120],
                "latency_ms": int((time.time() - t0) * 1000)}


def fan_out(proposers, messages, key, timeout, max_tokens, quorum):
    """Parallel proposer calls that RETURN EARLY once `quorum` good drafts arrive — the timeout
    is a cap, not a wait. Stragglers are abandoned (not waited on). Returns the drafts collected
    so far (good + any failures seen)."""
    ex = ThreadPoolExecutor(max_workers=len(proposers))
    futs = {ex.submit(call_model, a, messages, key, timeout, max_tokens): a for a in proposers}
    results, good = [], 0
    try:
        for fut in as_completed(futs, timeout=timeout + 5):
            d = fut.result()
            results.append(d)
            if d["ok"] and d["content"].strip():
                good += 1
                if good >= quorum:
                    break                       # enough responded — trigger, don't wait for the rest
    except TimeoutError:
        pass                                    # cap hit; proceed with whatever arrived
    ex.shutdown(wait=False, cancel_futures=True)  # abandon stragglers
    return results


def _quorum(n, cfg):
    """How many good drafts are 'enough' to proceed: all but the slowest one, but >= min."""
    return max(cfg["min_proposers"], n - 1)


def estimate_cost(entries, cfg):
    c = cfg["cost"]
    zen_paid = set(c["zen_paid_aliases"])
    zen_go = set(c.get("zen_go_aliases", []))          # GO subscription: flat-rate, $0 marginal
    free_t = sum(e["tokens"] for e in entries if e["provider"] == "nim"
                 or (e["provider"] == "zen" and e["alias"] not in zen_paid
                     and e["alias"] not in zen_go))
    go_t = sum(e["tokens"] for e in entries if e["alias"] in zen_go)
    zen_t = sum(e["tokens"] for e in entries if e["alias"] in zen_paid)
    credits = sum(1 for e in entries if e["provider"] == "copilot" and e["ok"])
    usd = zen_t / 1e6 * c["zen_paid_usd_per_1m"] + credits * c["copilot_usd_per_credit"]
    return {"free_tokens": free_t, "zen_go_tokens": go_t, "zen_paid_tokens": zen_t,
            "copilot_credits": credits, "pwm_searches": 0, "usd_estimate": round(usd, 4)}


AGG_PROMPT = ("You are the aggregator of a model committee. Below is a user task and several "
              "independent drafts. Reconcile them: resolve disagreements, take the strongest "
              "reasoning, fix errors, and produce the single best final answer. Do NOT mention "
              "the drafts or the committee; just answer the task.\n\nTASK:\n{task}\n\n{drafts}")


ESCALATE = {"easy": "hard", "hard": "deep", "deep": None}


def _fuse_once(cleaned, difficulty, cfg, key, availability, health):
    """One MoA round at a fixed difficulty. Returns {answer, receipt}."""
    panel = build_panel(difficulty, cfg, availability, health)
    receipt = {"mode": "fuse", "difficulty": difficulty, "proposers": [], "degraded": False}
    msgs = [{"role": "user", "content": cleaned}]
    tmo = cfg["panels"][difficulty].get("timeout_s", cfg["proposer_timeout_s"])
    ptok, atok = cfg["proposer_max_tokens"], cfg["aggregator_max_tokens"]

    if len(panel["proposers"]) >= cfg["min_proposers"]:
        drafts = fan_out(panel["proposers"], msgs, key, tmo, ptok, _quorum(len(panel["proposers"]), cfg))
        receipt["proposers"] = drafts
        good = [d for d in drafts if d["ok"] and d["content"].strip()]
    else:
        good = []

    if len(good) < cfg["min_proposers"]:                       # degrade: single best model
        receipt["degraded"] = True
        one = call_model(panel["aggregator"] or "auto", msgs, key, tmo, atok, force_tier="frontier")
        receipt["aggregator"] = one
        answer = one["content"] if one["ok"] else "(fusion failed: no models reachable)"
    else:
        dtxt = "\n\n".join(f"--- DRAFT {i+1} ({d['alias']}) ---\n{d['content']}"
                           for i, d in enumerate(good))
        agg_msgs = [{"role": "user", "content": AGG_PROMPT.format(task=cleaned, drafts=dtxt)}]
        agg = call_model(panel["aggregator"] or "auto", agg_msgs, key, tmo, atok, force_tier="frontier")
        receipt["aggregator"] = agg
        answer = agg["content"] if agg["ok"] and agg["content"].strip() else \
            max(good, key=lambda d: len(d["content"]))["content"]   # aggregator failed -> best draft
        if not agg["ok"]:
            receipt["degraded"] = True

    entries = receipt["proposers"] + [receipt.get("aggregator", {"tokens": 0, "provider": "", "alias": "", "ok": False})]
    receipt["total_tokens"] = sum(e.get("tokens", 0) for e in entries)
    receipt["est_cost"] = estimate_cost([e for e in entries if e.get("alias")], cfg)
    return {"answer": answer, "receipt": receipt}


def fuse(prompt, mode=None, depth=None, confirm_research=False, escalate=True, conduct=None):
    """Fuse a [NOVEL] prompt. depth=None auto-classifies (easy/hard); only escalate
    to a heavier panel if the round comes back DEGRADED (objective miss). Explicit
    depth (or [NOVEL DEEP]) pins the level and skips auto-escalation. When the conductor
    is enabled (config or conduct=True), route to the v2.1 adaptive search instead."""
    t0 = time.time()
    cleaned, tag_mode, tag_depth = parse_novel(prompt)
    mode = mode or tag_mode or "fuse"
    cfg = load_config()
    if mode == "research":
        return _research(cleaned, cfg, t0, confirm_research)
    if mode == "tree":                                # v2.1.1 [NOVEL TREE] -> Multi-LLM AB-MCTS
        from fusion.abmcts import conduct_abmcts
        return conduct_abmcts(prompt, depth=depth)
    use_conduct = cfg.get("conductor", {}).get("enabled", False) if conduct is None else conduct
    if use_conduct:
        from fusion.conductor import conduct as _conduct
        return _conduct(prompt, depth=depth)
    env = load_env(); key = env.get("LITELLM_MASTER_KEY", "")
    availability, health = load_availability(), load_health()
    pinned = depth or tag_depth                                  # explicit -> no auto-escalation
    difficulty = pinned or difficulty_of(cleaned)

    res = _fuse_once(cleaned, difficulty, cfg, key, availability, health)
    path = [difficulty]
    # Auto-escalate one level at a time while the round is degraded (missed) and we can go deeper.
    while escalate and not pinned and res["receipt"]["degraded"] and ESCALATE.get(difficulty):
        difficulty = ESCALATE[difficulty]
        path.append(difficulty)
        res = _fuse_once(cleaned, difficulty, cfg, key, availability, health)

    res["receipt"]["escalation_path"] = path
    res["receipt"]["wall_ms"] = int((time.time() - t0) * 1000)
    _log(res["receipt"])
    return res


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
