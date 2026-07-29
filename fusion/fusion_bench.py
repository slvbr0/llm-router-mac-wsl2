"""A/B harness: fusion committee vs single frontier baseline, blind-judged.
Usage: python3 fusion/fusion_bench.py fusion/prompts.sample.txt [--baseline cop-opus] [--judge nim-qwen-max]
"""
import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion.fusion import fuse, call_model, load_env, load_config  # noqa: E402

import re

JUDGE_PROMPT = ("Score two answers to a task. Output ONLY a JSON object, nothing else, no "
                "explanation: {{\"a\": <1-10>, \"b\": <1-10>}}.\n\nTASK:\n{task}\n\n"
                "ANSWER A:\n{a}\n\nANSWER B:\n{b}\n\nJSON:")


def judge(task, ans_a, ans_b, judge_alias, key):
    r = call_model(judge_alias, [{"role": "user", "content":
                                  JUDGE_PROMPT.format(task=task, a=ans_a[:6000], b=ans_b[:6000])}],
                   key, 120, 300)
    m = re.findall(r'\{[^{}]*"a"\s*:\s*(\d+)[^{}]*"b"\s*:\s*(\d+)[^{}]*\}', r["content"])
    if m:
        return int(m[-1][0]), int(m[-1][1])
    return None, None                          # unparseable -> excluded from averages (not scored 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts"); ap.add_argument("--baseline"); ap.add_argument("--judge")
    ap.add_argument("--conduct", action="store_true", help="use the v2.1 conductor loop")
    ap.add_argument("--tree", action="store_true", help="use the v2.1.1 AB-MCTS tree search")
    args = ap.parse_args()
    cfg = load_config(); env = load_env(); key = env.get("LITELLM_MASTER_KEY", "")
    baseline = args.baseline or cfg["bench"]["baseline"]
    judge_alias = args.judge or cfg["bench"]["judge"]
    prompts = [l.strip() for l in Path(args.prompts).read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    rows, fq, bq, fcost, bcred = [], [], [], 0.0, 0
    fwall, bwall = [], []
    for i, p in enumerate(prompts, 1):
        if i > 1:
            time.sleep(60)      # pace the baseline — hammering cop-opus back-to-back gets
                                # throttled (bench12: 11 straight b_ok=False after prompt 1)
        print(f"[{i}/{len(prompts)}] fusing…", file=sys.stderr)
        tag = "[NOVEL TREE] " if args.tree else "[NOVEL] "
        f = fuse(tag + p, escalate=False, conduct=None if args.tree else args.conduct)
        tb = time.time()
        # Baseline budget must cover HIDDEN THINKING + the visible answer, and we must WAIT for
        # the model to finish — never cut it off mid-think. Reasoning frontier models (opus w/
        # extended thinking) burn tokens thinking, THEN emit the answer; a low cap starves the
        # answer and a short timeout truncates it to EMPTY. Copilot is per-credit, not per-token,
        # so a generous cap costs nothing. 6000 tokens (think + answer) + 900s timeout = wait it out.
        # The prompt hints the answer length so opus doesn't over-think a simple ask.
        bp = p + "\n\n(Aim for a focused answer, roughly 700-1200 words.)"
        b = call_model(baseline, [{"role": "user", "content": bp}], key, 1200, 6000)
        if not b["ok"] or not b["content"].strip():         # frontier baselines flake/think-only — retry once
            print("  baseline flaked/empty, retrying…", file=sys.stderr)
            b = call_model(baseline, [{"role": "user", "content": bp}], key, 1200, 6000)
        b_wall = time.time() - tb                           # frontier baselines are SLOW (~200-500s)
        if not f["answer"].strip() or not b["ok"] or not b["content"].strip():
            print(f"  skip: fusion or baseline empty (f_tok={f['receipt']['total_tokens']} "
                  f"b_ok={b['ok']})", file=sys.stderr)
            rows.append((p[:46], f["receipt"]["difficulty"], "-", "-",
                         f["receipt"]["total_tokens"], b["tokens"]))
            fcost += f["receipt"]["est_cost"]["usd_estimate"]; bcred += 1
            continue
        swap = random.random() < 0.5                       # blind: random order
        a1, a2 = (b["content"], f["answer"]) if swap else (f["answer"], b["content"])
        sa, sb = judge(p, a1, a2, judge_alias, key)
        fs, bs = (sb, sa) if swap else (sa, sb)
        fq.append(fs); bq.append(bs)
        fcost += f["receipt"]["est_cost"]["usd_estimate"]; bcred += 1
        fwall.append(f["receipt"]["wall_ms"] / 1000); bwall.append(b_wall)
        rows.append((p[:46], f["receipt"]["difficulty"], fs, bs,
                     f["receipt"]["total_tokens"], b["tokens"]))
    print(f"\n{'prompt':48} {'diff':>5} {'fus':>4} {'base':>4} {'f_tok':>6} {'b_tok':>6}")
    for r in rows:
        print(f"{r[0]:48} {r[1]:>5} {str(r[2]):>4} {str(r[3]):>4} {r[4]:>6} {r[5]:>6}")
    fqv = [x for x in fq if x is not None]; bqv = [x for x in bq if x is not None]
    if not fqv or not bqv:
        print("\nBENCH INVALID: judge produced no parseable scores — check judge model.")
        return
    favg, bavg = sum(fqv) / len(fqv), sum(bqv) / len(bqv)
    bcost = bcred * cfg["cost"]["copilot_usd_per_credit"]
    print(f"\n(scored {len(fqv)}/{len(prompts)} prompts; unparseable judge outputs excluded)")
    print(f"\nFUSION  avg quality {favg:.2f}  est cost ${fcost:.4f}"
          f"  → $/point {fcost / max(favg, .1):.4f}")
    print(f"BASELINE({baseline}) avg quality {bavg:.2f}  est cost ${bcost:.4f}"
          f"  → $/point {bcost / max(bavg, .1):.4f}")
    if fwall and bwall:
        print(f"\nLATENCY  fusion avg {sum(fwall)/len(fwall):.0f}s  vs  baseline avg "
              f"{sum(bwall)/len(bwall):.0f}s")
    print("\nVERDICT:", "fusion matches/beats baseline at lower cost — hypothesis SUPPORTED"
          if favg >= bavg - 0.5 and fcost < bcost else
          "fusion does NOT beat baseline on cost/quality — hypothesis NOT supported (honest result)")


if __name__ == "__main__":
    main()
