"""Frontier bench: can a committee of cheap thinking models match Claude Fable 5?

Three arms, blind pairwise judged, priced at public API list rates:

  A  ant-fable  single-shot, [FRONTIER] -> HIGH extended thinking   (the target)
  B  [NOVEL TREE] fusion, free NIM + GO subscription committee      (the challenger)
  C  arm B, then ONE ant-opus audit pass over the synthesis         (the cheap top-up)

Design notes that matter for believing the result:

* Baseline answers are generated once and CACHED, so B and C are judged against the very
  same fable text. Paired comparison; fable's run-to-run variance cannot explain a B-vs-C gap.
* Baseline runs with fallbacks DISABLED and asserts the served model. LiteLLM will silently
  serve a different provider on a 4xx, which would turn "baseline = fable" into
  "baseline = whatever answered".
* Judging is blind: answer order is randomised per prompt and the router's
  "[model · think:level · tier]" banner is stripped before the judge sees anything.
* Two independent judges. Disagreement is reported, not averaged away.
* Marginal cost is ~$0 on every arm (free NIM, GO subscription, Anthropic Max). "$0 vs $0"
  tells a reader nothing, so we report LIST-PRICE EQUIVALENT: what this would have cost at
  public API rates. Extended-thinking tokens bill as output and are included.

Usage:
  python3 fusion/frontier_bench.py fusion/prompts.bench4.txt
  python3 fusion/frontier_bench.py fusion/prompts.bench12.txt --cache logs/baseline-fable.json
"""
import argparse, json, random, re, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion.fusion import (load_env, load_config, call_model, strip_annotation,  # noqa: E402
                           ROUTER)
from fusion.abmcts import conduct_abmcts  # noqa: E402

BASELINE_SUFFIX = "\n\n(Aim for a focused answer, roughly 700-1200 words.)"

# Anthropic requires max_tokens > thinking budget_tokens. FRONTIER injects 16384, so a low cap
# hard-errors -- and the fallback chain then hides it. Leave room for thinking AND the answer.
BASELINE_MAX_TOKENS = 24000
BASELINE_TIMEOUT_S = 900

AUDIT_PROMPT = (
    "You are auditing a candidate answer to a task. Read the task and the answer. Identify any "
    "factual errors, logical gaps, missing nuance, or unjustified claims, and produce a corrected "
    "final answer.\n\nDo not rewrite from scratch and do not pad: keep everything the answer got "
    "right, and change only what is actually wrong or missing. Output ONLY the corrected final "
    "answer, with no preamble and no mention of this audit.\n\nTASK:\n{task}\n\nANSWER:\n{answer}"
)
AUDIT_MAX_TOKENS = 8000
AUDIT_TIMEOUT_S = 600

JUDGE_PROMPT = (
    'Score two answers to a task. Output ONLY a JSON object, nothing else, no explanation: '
    '{{"a": <1-10>, "b": <1-10>}}.\n\nTASK:\n{task}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}\n\nJSON:'
)
JUDGE_RE = re.compile(r'\{[^{}]*"a"\s*:\s*(\d+)[^{}]*"b"\s*:\s*(\d+)[^{}]*\}')

# The judge is the measuring instrument, so its config must not depend on what it is measuring.
# Left to auto-classification, a judge prompt wrapping two long technical answers lands on REASON,
# and the router then injects a thinking budget for any thinking-capable judge. That is how the
# 2026-07-10 run lost its second judge: go-glm rejects the router's thinking params outright
# (400 "Unsupported parameter(s)"), so every go-glm judge call failed and was silently dropped
# from the mean, collapsing a two-judge bias control down to the baseline's own family judge.
#
# A judge emits a two-integer JSON verdict. Extended thinking buys little there and costs a hard
# dependency on per-provider thinking support. Pin the tier to `cheap` -> no thinking is injected
# for ANY judge, on any provider, whatever the task text looks like.
JUDGE_TIER = "cheap"            # deterministic: no thinking params injected, no provider 400s
JUDGE_MAX_TOKENS = 1500         # verdict JSON only; no thinking budget to leave room for
JUDGE_TIMEOUT_S = 300


def raw_call(alias, prompt, key, max_tokens, timeout, no_fallback=True, force_tier=None):
    """Direct router call. no_fallback makes provider failures LOUD instead of silently
    substituting another model."""
    payload = {"model": alias, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens}
    if no_fallback:
        payload["num_retries"] = 0
        payload["fallbacks"] = []
    if force_tier:
        payload.setdefault("metadata", {})["force_tier"] = force_tier
    req = urllib.request.Request(ROUTER, data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
        u = d.get("usage", {}) or {}
        return {"ok": True, "content": strip_annotation(d["choices"][0]["message"]["content"] or ""),
                "served_model": d.get("model", ""), "tok_in": u.get("prompt_tokens", 0),
                "tok_out": u.get("completion_tokens", 0), "wall_s": time.time() - t0}
    except Exception as e:
        detail = ""
        if hasattr(e, "read"):
            try:
                detail = e.read().decode()[:200]
            except Exception:
                pass
        return {"ok": False, "content": "", "served_model": "", "tok_in": 0, "tok_out": 0,
                "wall_s": time.time() - t0, "error": f"{type(e).__name__}: {e} {detail}"[:260]}


def judge(task, ans_a, ans_b, judge_alias, key, log=None):
    r = call_model(judge_alias, [{"role": "user", "content":
                                  JUDGE_PROMPT.format(task=task, a=ans_a[:6000], b=ans_b[:6000])}],
                   key, JUDGE_TIMEOUT_S, JUDGE_MAX_TOKENS, force_tier=JUDGE_TIER)
    if not r["ok"]:
        if log:
            log(f"    !! judge {judge_alias} call failed: {r.get('error','')[:110]}")
        return None, None
    m = JUDGE_RE.findall(r["content"])
    if not m:
        # A dropped judge is not a neutral event: it silently reduces a two-judge design to one,
        # and the surviving judge shares a family with the baseline. Say so, loudly.
        if log:
            log(f"    !! judge {judge_alias} unparseable ({r['tok_out']} tok_out, "
                f"tail={r['content'][-70:]!r})")
        return None, None                  # unparseable -> excluded, never scored 0
    return int(m[-1][0]), int(m[-1][1])


def blind_judge(task, fusion_ans, base_ans, judge_alias, key, rng, log=None):
    """Randomise presentation order so position bias cannot favour one arm."""
    swap = rng.random() < 0.5
    a1, a2 = (base_ans, fusion_ans) if swap else (fusion_ans, base_ans)
    s1, s2 = judge(task, a1, a2, judge_alias, key, log)
    if s1 is None:
        return None, None
    return (s2, s1) if swap else (s1, s2)   # -> (fusion_score, baseline_score)


def list_cost(entries, prices, ceiling_alias="go-glm"):
    """USD at public API list rates. Returns (usd, unpriced_tokens, unpriced_aliases).

    `usd` is the POINT ESTIMATE: unpriced aliases are billed at `ceiling_alias`'s rate.

    This is NOT a bound, and an earlier version of this docstring wrongly claimed it was. The
    unpriced aliases are not uniformly cheaper than GLM 5.2: go-qwen-max lists at ~$6.40/1M
    output, above GLM's $4.40, while nim-deepseek is free and go-deepseek lists far below.
    The error runs in both directions, so use lo/hi from `cost_bounds()` for any claim about
    which arm is cheaper. On the 2026-07-10 bench, 33% of arm B's tokens were unpriced -- a
    third of the bill was an assumption, on the very number the benchmark exists to measure.
    """
    ceiling_in, ceiling = prices.get(ceiling_alias, (0.0, 0.0))
    usd, unpriced_tok, unpriced = 0.0, 0, set()
    for e in entries:
        if not e.get("ok"):
            continue
        ti, to = e.get("tok_in", 0), e.get("tok_out", 0)
        p = prices.get(e["alias"])
        if p is None:
            unpriced_tok += ti + to
            unpriced.add(e["alias"])
            usd += ti / 1e6 * ceiling_in + to / 1e6 * ceiling
        else:
            usd += ti / 1e6 * p[0] + to / 1e6 * p[1]
    return usd, unpriced_tok, sorted(unpriced)


def cost_bounds(entries, prices, ceiling_alias="go-glm"):
    """(lo, hi) USD. Priced aliases bill at their real rate in both. Unpriced aliases bill at
    $0 in `lo` and at the priciest PRICED alias's rate in `hi`.

    Invents no prices. `lo` is a TRUE lower bound: it charges every unpriced model $0, and no
    model costs less than free. That is the one that settles the headline. If arm B's `lo` still
    exceeds arm A's fully-priced cost, then B is not cheaper than A no matter what the missing
    numbers turn out to be.

    `hi` is only indicative, not a strict upper bound: it bills unpriced aliases at the priciest
    commodity rate we hold (GLM 5.2, $4.40/1M out), and go-qwen-max actually lists above that.
    Never rest a "fusion is expensive" claim on `hi`; rest a "fusion is not cheaper" claim on `lo`."""
    hi_in, hi_out = max(prices.values(), key=lambda p: p[1]) if prices else (0.0, 0.0)
    cl_in, cl_out = prices.get(ceiling_alias, (hi_in, hi_out))
    lo = hi = 0.0
    for e in entries:
        if not e.get("ok"):
            continue
        ti, to = e.get("tok_in", 0), e.get("tok_out", 0)
        p = prices.get(e["alias"])
        if p is None:
            hi += ti / 1e6 * cl_in + to / 1e6 * cl_out   # ceiling = priciest COMMODITY rate
        else:
            lo += ti / 1e6 * p[0] + to / 1e6 * p[1]
            hi += ti / 1e6 * p[0] + to / 1e6 * p[1]
    return lo, hi


def build_baseline(prompts, alias, key, cache_path, log):
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        log(f"baseline cache: {len(cache)} answers loaded from {cache_path}")
    for i, p in enumerate(prompts, 1):
        if p in cache and cache[p].get("ok"):
            continue
        log(f"[baseline {i}/{len(prompts)}] {alias} (FRONTIER, HIGH thinking)…")
        r = raw_call(alias, "[FRONTIER] " + p + BASELINE_SUFFIX, key,
                     BASELINE_MAX_TOKENS, BASELINE_TIMEOUT_S)
        if r["ok"] and r["served_model"] != alias:
            r["ok"] = False
            r["error"] = f"served {r['served_model']!r}, expected {alias!r} (silent fallback)"
        if not r["ok"]:
            log(f"    FAILED: {r.get('error', '')[:160]}")
        else:
            log(f"    ok  {r['tok_in']}in/{r['tok_out']}out  {r['wall_s']:.0f}s")
        cache[p] = r
        cache_path.write_text(json.dumps(cache, indent=1))   # checkpoint every prompt
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--cache", default="logs/baseline-ant-fable.json")
    ap.add_argument("--baseline")
    ap.add_argument("--auditor", default="ant-opus")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts")
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    key = load_env().get("LITELLM_MASTER_KEY", "")
    baseline = args.baseline or cfg["bench"]["baseline"]
    judges = [cfg["bench"]["judge"], cfg["bench"].get("judge_b")]
    judges = [j for j in judges if j]
    prices = {k: tuple(v) for k, v in cfg["cost"]["list_usd_per_1m"].items()}
    rng = random.Random(20260710)                       # deterministic A/B ordering; reproducible

    prompts = [l.strip() for l in Path(args.prompts).read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    if args.limit:
        prompts = prompts[:args.limit]

    def log(m):
        print(m, file=sys.stderr, flush=True)

    log(f"baseline={baseline}  auditor={args.auditor}  judges={judges}  n={len(prompts)}")

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    base = build_baseline(prompts, baseline, key, cache_path, log)

    rows, cost = [], {"A": 0.0, "B": 0.0, "C": 0.0}
    bounds = {a: {"lo": 0.0, "hi": 0.0} for a in ("A", "B", "C")}
    unpriced_total, unpriced_aliases = 0, set()
    wall = {"A": [], "B": [], "C": []}

    for i, p in enumerate(prompts, 1):
        b = base.get(p, {})
        if not b.get("ok"):
            log(f"[{i}/{len(prompts)}] SKIP — no valid baseline")
            continue

        log(f"[{i}/{len(prompts)}] fusion (TREE)…")
        t0 = time.time()
        f = conduct_abmcts("[NOVEL TREE] " + p)
        b_wall = time.time() - t0
        fus_ans = strip_annotation(f["answer"])
        entries = f.get("calls", [])
        if not fus_ans.strip():
            log("    fusion produced empty answer — skipping prompt")
            continue

        c_ans, c_entries, audit_wall = fus_ans, [], 0.0
        if not args.skip_audit:
            log(f"    audit ({args.auditor})…")
            a = raw_call(args.auditor, AUDIT_PROMPT.format(task=p, answer=fus_ans), key,
                         AUDIT_MAX_TOKENS, AUDIT_TIMEOUT_S, force_tier="frontier")
            audit_wall = a["wall_s"]
            if a["ok"] and a["content"].strip():
                if a["served_model"] != args.auditor:
                    log(f"    audit served {a['served_model']!r} != {args.auditor!r} — not counted")
                else:
                    c_ans = a["content"]
                    c_entries = [{"alias": args.auditor, "ok": True,
                                  "tok_in": a["tok_in"], "tok_out": a["tok_out"]}]
            else:
                log(f"    audit failed ({a.get('error','empty')[:90]}) — arm C falls back to arm B")

        a_entries = [{"alias": baseline, "ok": True, "tok_in": b["tok_in"], "tok_out": b["tok_out"]}]
        cA, uA, alA = list_cost(a_entries, prices)
        cB, uB, alB = list_cost(entries, prices)
        cC, uC, alC = list_cost(entries + c_entries, prices)
        cost["A"] += cA; cost["B"] += cB; cost["C"] += cC
        for arm, ents in (("A", a_entries), ("B", entries), ("C", entries + c_entries)):
            lo, hi = cost_bounds(ents, prices)
            bounds[arm]["lo"] += lo
            bounds[arm]["hi"] += hi
        # arm C's entries are a superset of arm B's, so uC already contains uB. Adding both
        # would report every fusion token as unpriced twice.
        unpriced_total += uC
        unpriced_aliases |= set(alB) | set(alC)
        wall["A"].append(b["wall_s"]); wall["B"].append(b_wall)
        wall["C"].append(b_wall + audit_wall)

        scores = {}
        for j in judges:
            fb, bb = blind_judge(p, fus_ans, b["content"], j, key, rng, log)
            fc, bc = blind_judge(p, c_ans, b["content"], j, key, rng, log)
            scores[j] = {"B": fb, "A_vs_B": bb, "C": fc, "A_vs_C": bc}
            log(f"    judge {j}: B={fb} C={fc} A={bb}/{bc}")

        # Persist the answers. Judging is cheap and generation is not: without these, a bad judge
        # config means regenerating every arm to re-score. Kept out of `summarise`/`report`.
        rows.append({"prompt": p, "difficulty": f["receipt"]["difficulty"], "scores": scores,
                     "cost": {"A": cA, "B": cB, "C": cC},
                     "answers": {"A": b["content"], "B": fus_ans, "C": c_ans},
                     # per-call in/out tokens: lets cost be re-derived under new prices without
                     # regenerating a single answer. A third of arm B's bill was an assumption;
                     # the fix for that must not cost another hour of tree search.
                     "entries": {"A": a_entries, "B": entries, "C": c_entries},
                     "bandit": f["receipt"].get("bandit", {}),
                     "tok": {"A": b["tok_in"] + b["tok_out"],
                             "B": sum(e.get("tokens", 0) for e in entries)}})

    summary = summarise(rows, judges)
    out = {"baseline": baseline, "auditor": args.auditor, "judges": judges, "n": len(rows),
           "rows": rows, "cost_list_usd": cost, "cost_bounds_usd": bounds, "summary": summary,
           "unpriced_tokens": unpriced_total, "unpriced_aliases": sorted(unpriced_aliases),
           "wall_s": {k: (sum(v) / len(v) if v else 0) for k, v in wall.items()}}
    report(out, log)
    print(json.dumps(out, indent=1))
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarise(rows, judges):
    """Per-judge means, plus a cross-judge mean. Judges disagree along family lines
    (an Anthropic judge favours the Anthropic baseline; a committee member favours fusion),
    so per-judge numbers are reported and never silently averaged away.

    `scored` counts the prompts each judge actually returned a number for. A judge that fails
    to parse contributes nothing to its own mean, and _mean would then quietly compute
    ALL_JUDGES from the surviving judge alone -- turning a two-judge bias control into a
    single-judge verdict without changing a single visible number. Coverage is reported so
    that collapse is impossible to miss."""
    s = {}
    for j in judges:
        s[j] = {
            "A": _mean([r["scores"][j]["A_vs_B"] for r in rows] +
                       [r["scores"][j]["A_vs_C"] for r in rows]),
            "B": _mean([r["scores"][j]["B"] for r in rows]),
            "C": _mean([r["scores"][j]["C"] for r in rows]),
            "scored": sum(1 for r in rows if r["scores"][j]["B"] is not None),
            "of": len(rows),
        }
    s["ALL_JUDGES"] = {arm: _mean([s[j][arm] for j in judges]) for arm in ("A", "B", "C")}
    s["ALL_JUDGES"]["judges_with_full_coverage"] = sum(
        1 for j in judges if s[j]["scored"] == len(rows))
    return s


def report(out, log):
    n = out["n"]
    if not n:
        log("\nBENCH INVALID: no prompt produced a scorable triple.")
        return
    s, c, w = out["summary"], out["cost_list_usd"], out["wall_s"]
    if all(s["ALL_JUDGES"][arm] is None for arm in ("A", "B", "C")):
        log(f"\n{'='*72}\nBENCH INVALID: no judge returned a single score across {n} prompts.")
        log("Cost and wall-clock below would be real, but there is no quality result to report. "
            "Do not publish. (Usual cause: the router was unreachable mid-run.)")
    log(f"\n{'='*72}\nn={n}  baseline={out['baseline']}  auditor={out['auditor']}")
    log(f"{'arm':<34} {'quality':>8} {'list $':>9} {'$/prompt':>9} {'wall s':>8}")
    names = {"A": f"A  {out['baseline']} + HIGH thinking",
             "B": "B  [NOVEL TREE] fusion",
             "C": f"C  fusion + {out['auditor']} audit"}
    for arm in ("A", "B", "C"):
        q = s["ALL_JUDGES"][arm]
        # An arm with no surviving judge has quality None. format(None, '>8') raises, which used
        # to crash the report at the exact moment it had bad news to deliver.
        qs = "n/a" if q is None else f"{round(q, 2)}"
        log(f"{names[arm]:<34} {qs:>8} "
            f"{c[arm]:>9.4f} {c[arm]/n:>9.4f} {w[arm]:>8.0f}")
    log("\nper-judge (bias runs toward each judge's own family):")
    for j in out["judges"]:
        cov = "" if s[j]["scored"] == s[j]["of"] else f"   <-- scored only {s[j]['scored']}/{s[j]['of']}"
        log(f"  {j:<12} A={s[j]['A']} B={s[j]['B']} C={s[j]['C']}{cov}")
    full = s["ALL_JUDGES"]["judges_with_full_coverage"]
    if full < len(out["judges"]):
        log(f"\n!! ONLY {full}/{len(out['judges'])} JUDGES SCORED EVERY PROMPT. The cross-judge "
            f"mean above is NOT a two-judge result — it is dominated by whichever judge survived, "
            f"and {out['baseline']}'s own family judge is one of them. Do not publish this as a "
            f"bias-controlled comparison.")
    bnd = out.get("cost_bounds_usd")
    if bnd:
        log("\ncost bounds (lo charges every unpriced alias $0 — a true lower bound):")
        for arm in ("A", "B", "C"):
            log(f"  {arm}  lo=${bnd[arm]['lo']/n:.4f}/prompt   hi=${bnd[arm]['hi']/n:.4f}/prompt")
        # The only cost claim the missing prices cannot overturn.
        if bnd["B"]["lo"] > bnd["A"]["hi"]:
            log(f"  => arm B is NOT cheaper than {out['baseline']}: even billing every unpriced "
                f"model at $0, B costs more. No price we are missing can change this.")
        elif bnd["B"]["hi"] < bnd["A"]["lo"]:
            log(f"  => arm B is strictly cheaper than {out['baseline']} under every pricing.")
        else:
            log("  => cost intervals OVERLAP. This bench has NOT established which arm is "
                "cheaper; do not claim it has.")
    if out["unpriced_tokens"]:
        log(f"\n{out['unpriced_tokens']} fusion tokens had no published price. The point estimate "
            f"above bills them at GLM 5.2 rates, which is NOT a bound (go-qwen-max lists above "
            f"it): {', '.join(out['unpriced_aliases'])}")
    log("Marginal cost was ~$0 on every arm (free NIM / GO subscription / Anthropic Max). "
        "'list $' is the counterfactual public-API price, thinking tokens billed as output.")


if __name__ == "__main__":
    main()
