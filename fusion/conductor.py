"""Conductor (AB-MCTS-lite): score each committee synthesis and run a greedy adaptive
width-vs-depth search. Deepen while improving, widen when stalled, return best-so-far.
Full Thompson-sampling AB-MCTS is v2.1.1 (see spec §9)."""
import re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fusion.fusion import (build_panel, call_model, estimate_cost, provider_of, fan_out,  # noqa: E402
                           _quorum, _ok, load_env, load_config, load_availability, load_health,
                           difficulty_of, parse_novel, _log)

CONDUCTOR_PROMPT = (
    "You are the CONDUCTOR of a model committee. Given a TASK and the committee's DRAFTS, "
    "reconcile the drafts into the single best answer, then respond in EXACTLY this format and "
    "nothing before it:\n"
    "SCORE: <integer 1-10 for how good your final answer is>\n"
    "GAPS: <short comma-separated list of what is still missing or wrong, or 'none'>\n"
    "ANSWER:\n"
    "<the single best final answer — plain text, may contain code/math>\n\n"
    "TASK:\n{task}\n\nDRAFTS:\n{drafts}")


def parse_conductor(raw):
    """Line-based parse, robust to REASONING models that think first then emit the format.
    Takes the LAST SCORE/GAPS and the text after the LAST 'ANSWER:' (the final output, not the
    thinking preamble). Robust to LaTeX/code in the answer. Missing markers -> None / raw."""
    sm = re.findall(r'SCORE:\s*(\d+)', raw, re.IGNORECASE)
    score = int(sm[-1]) if sm else None
    gm = re.findall(r'GAPS:\s*(.+)', raw, re.IGNORECASE)
    gaps = []
    if gm:
        gaps = [g.strip() for g in gm[-1].split(",")
                if g.strip() and g.strip().lower() != "none"]
    ans = list(re.finditer(r'ANSWER:\s*\n?', raw, re.IGNORECASE))
    # ANSWER: marker present -> body after it (may be "" so the caller falls back to a draft).
    # No marker at all -> use the whole text (plain-prose response, best effort).
    answer = raw[ans[-1].end():].strip() if ans else raw.strip()
    return {"answer": answer, "score": score, "gaps": gaps}


def next_action(history, improve_epsilon):
    """Greedy width/depth: deepen while the last step improved best by >= epsilon; else widen."""
    if len(history) < 2:
        return "deepen"
    best_before = max(h["score"] or 0 for h in history[:-1])
    last = history[-1]["score"] or 0
    return "deepen" if last - best_before >= improve_epsilon else "widen"


def pick_conductor(difficulty, cfg, availability, health):
    """Conductor = capable orchestrator, chosen DYNAMICALLY. Config lists capable candidates.
    Prefer a healthy NIM candidate with the SMALLEST measured response time (from the audit);
    if no NIM is healthy/decent, fall to a capable opencode-go (Zen) model, then Copilot."""
    cands = cfg["conductor"]["models"][difficulty]
    if isinstance(cands, str):
        cands = [cands]
    healthy = [m for m in cands if _ok(m, availability, health)]
    if not healthy:
        return cands[-1]                               # last resort even if masked (better than nothing)
    nim = [m for m in healthy if provider_of(m) == "nim"]
    if nim:                                            # fastest capable NIM by measured latency
        nim.sort(key=lambda m: health.get(m, {}).get("latency_ms", 10 ** 9))
        return nim[0]
    return healthy[0]                                  # else first healthy non-NIM (Zen/Copilot)


def _committee(task, difficulty, cfg, key, availability, health, exclude=None):
    """One committee fan-out for `task`; returns good drafts. Early-triggers (fan_out) once a
    quorum responds — does not wait out the timeout for stragglers."""
    panel = build_panel(difficulty, cfg, availability, health)
    proposers = [a for a in panel["proposers"] if a not in (exclude or set())] or panel["proposers"]
    tmo = cfg["panels"][difficulty].get("timeout_s", cfg["proposer_timeout_s"])
    ptok = cfg["proposer_max_tokens"]
    msgs = [{"role": "user", "content": task}]
    drafts = fan_out(proposers, msgs, key, tmo, ptok, _quorum(len(proposers), cfg))
    return [d for d in drafts if d["ok"] and d["content"].strip()]


def conductor_pass(task, drafts, model, key, cfg, availability, health):
    """One conductor call: synthesize + self-score + gaps over the given drafts."""
    tmo = cfg["conductor"].get("timeout_s", cfg["proposer_timeout_s"])
    atok = cfg["conductor"]["refine_max_tokens"]
    dtxt = "\n\n".join(f"--- DRAFT {i+1} ({d['alias']}) ---\n{d['content']}"
                       for i, d in enumerate(drafts)) or "(no drafts)"
    r = call_model(model, [{"role": "user",
                            "content": CONDUCTOR_PROMPT.format(task=task, drafts=dtxt)}],
                   key, tmo, atok, force_tier="frontier")
    parsed = parse_conductor(r["content"] if r["ok"] else "")
    # Reasoning conductors sometimes spend the token budget thinking + scoring and get cut off
    # before the ANSWER body. If so, fall back to the best proposer draft (never return empty).
    if not parsed["answer"].strip() and drafts:
        parsed["answer"] = max(drafts, key=lambda d: len(d["content"]))["content"]
    parsed["_call"] = r
    return parsed


def deepen(task, best, difficulty, model, cfg, key, availability, health):
    """Refine: committee re-answers targeting the gaps, conductor re-synthesizes."""
    gaps = "; ".join(best.get("gaps", [])) or "improve rigor, completeness, and correctness"
    refine_task = (f"{task}\n\nA previous answer was:\n{best['answer']}\n\n"
                   f"Fix these gaps and improve it: {gaps}")
    drafts = _committee(refine_task, difficulty, cfg, key, availability, health)
    return conductor_pass(task, drafts, model, key, cfg, availability, health)


def widen(task, difficulty, model, cfg, key, availability, health, used_models):
    """Fresh independent drafts (rotate proposers away from already-used), new synthesis."""
    drafts = _committee(task, difficulty, cfg, key, availability, health, exclude=used_models)
    return conductor_pass(task, drafts, model, key, cfg, availability, health)


def conduct(prompt, depth=None, escalate_diff=None):
    """Full conductor search over a [NOVEL] prompt. Returns {answer, receipt}."""
    t0 = time.time()
    cleaned, _mode, tag_depth = parse_novel(prompt)
    cfg = load_config()
    ccfg = cfg["conductor"]
    env = load_env(); key = env.get("LITELLM_MASTER_KEY", "")
    availability, health = load_availability(), load_health()
    difficulty = depth or tag_depth or escalate_diff or difficulty_of(cleaned)
    model = pick_conductor(difficulty, cfg, availability, health)
    thr, maxr, eps = ccfg["score_threshold"], ccfg["max_rounds"], ccfg["improve_epsilon"]

    drafts0 = _committee(cleaned, difficulty, cfg, key, availability, health)
    used = {d["alias"] for d in drafts0}
    c0 = conductor_pass(cleaned, drafts0, model, key, cfg, availability, health)
    history = [{"round": 0, "action": "init", "model": model, "score": c0["score"] or 0}]
    calls = [d for d in drafts0] + [c0["_call"]]
    best = {"answer": c0["answer"], "score": c0["score"] or 0, "round": 0, "action": "init"}

    rnd = 0
    while rnd < maxr and best["score"] < thr:
        rnd += 1
        act = next_action(history, eps)
        if act == "deepen":
            cand = deepen(cleaned, best, difficulty, model, cfg, key, availability, health)
        else:
            cand = widen(cleaned, difficulty, model, cfg, key, availability, health, used)
        calls.append(cand["_call"])
        cscore = cand["score"] or 0
        history.append({"round": rnd, "action": act, "model": model, "score": cscore})
        if cscore > best["score"]:
            best = {"answer": cand["answer"], "score": cscore, "round": rnd, "action": act}
        if act == "widen" and cscore <= history[-2]["score"]:
            break

    entries = [c for c in calls if c.get("alias")]
    receipt = {"mode": "fuse", "conductor": True, "difficulty": difficulty,
               "rounds": rnd, "best_round": best["round"], "best_score": best["score"],
               "search_path": history,
               "total_tokens": sum(c.get("tokens", 0) for c in entries),
               "est_cost": estimate_cost(entries, cfg),
               "wall_ms": int((time.time() - t0) * 1000)}
    _log(receipt)
    return {"answer": best["answer"], "receipt": receipt}
