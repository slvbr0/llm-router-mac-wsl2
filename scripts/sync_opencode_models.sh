#!/bin/sh
# Make opencode's model picker match what the router actually serves.
#
# opencode enumerates models by hand in opencode.json instead of reading /v1/models, so every alias
# added to config.yaml is invisible in the picker until someone remembers to add it there too. That
# drift is silent and one-directional: `auto` keeps working (the router picks for you), so nothing
# looks broken — you simply cannot select the new model. This regenerates the list from the router.
#
# Not everything the router exposes belongs in a picker, so a few aliases are held back ON PURPOSE
# (see EXCLUDE below): offering a model that always errors is worse than not offering it.
#
# Usage:  sh scripts/sync_opencode_models.sh [--dry-run]
set -e
cd "$(dirname "$0")/.."
[ -f .env ] && . ./.env
CFG="${OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"
PORT="${LLMR_PORT:-4040}"
[ -f "$CFG" ] || { echo "no opencode config at $CFG"; exit 1; }

MODELS=$(curl -s -m 10 "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $LITELLM_MASTER_KEY")
echo "$MODELS" | grep -q '"data"' || { echo "router not answering on :$PORT — nothing to sync"; exit 1; }

MODELS="$MODELS" CFG="$CFG" DRY="${1:-}" python3 - <<'PY'
import json, os, collections, shutil, time

# Held back deliberately. Each of these is reachable in config.yaml (fallback chains may still use
# them) but must not be offered as a manual choice, because picking one fails:
EXCLUDE = {
    # GPT-family Copilot models need Copilot's /responses API, which litellm's github_copilot
    # provider does not call. They 400 every time. Use cod-*/zen-gpt for GPT instead.
    "co-gpt", "co-codex", "co-mini",
    # Upstream removed these from the Zen paid catalog — verified live: claude-fable-5 returns
    # "Model is disabled", qwen3.7-max/plus return "not supported". Kept in config only because
    # fallback chains still reference them.
    "zen-fable", "zen-qwen-max", "zen-qwen-plus",
}

cfg_path = os.environ["CFG"]
router = [m["id"] for m in json.loads(os.environ["MODELS"])["data"]]
cfg = json.load(open(cfg_path), object_pairs_hook=collections.OrderedDict)
prov = cfg["provider"]["llm-router"]
old = prov["models"]

# Keep "auto" first — it is the one people actually use — then router order, which is config.yaml
# order: cost lane by lane. Existing per-model settings are preserved, never clobbered.
new = collections.OrderedDict()
if "auto" in old:
    new["auto"] = old["auto"]
for mid in router:
    if mid in EXCLUDE or mid == "auto":
        continue
    new[mid] = old.get(mid, {})

added = [m for m in new if m not in old]
removed = [m for m in old if m not in new]

print(f"router serves {len(router)}  ->  picker will list {len(new)}  (held back: {len(EXCLUDE)})")
if added:   print("  + " + ", ".join(added))
if removed: print("  - " + ", ".join(removed) + "   (gone from the router, or excluded)")
if not added and not removed:
    print("  already in sync"); raise SystemExit

if os.environ.get("DRY") == "--dry-run":
    print("  dry run — nothing written"); raise SystemExit

bak = f"{cfg_path}.bak-{time.strftime('%H%M%S')}"
shutil.copy2(cfg_path, bak)
prov["models"] = new
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
json.load(open(cfg_path))          # fail loudly rather than leave opencode with broken JSON
print(f"  wrote {cfg_path} (backup {bak}) — restart opencode to reload")
PY
