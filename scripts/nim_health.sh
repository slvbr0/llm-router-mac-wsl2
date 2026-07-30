#!/bin/sh
# Latency audit for FREE aliases only (NIM + Mistral free + zen-free), IN PARALLEL.
# Writes model_health.yaml (read per-request by priority_router.py): the router picks the
# FASTEST healthy model within a cost class, so NIM and the free tiers compete on measured latency.
# Flat/paid backends (z.ai, opencode GO, Anthropic OAuth, Copilot) are NOT probed — probing would
# spend their quota every ~15 min for no gain (see the alias-list note below).
# Models slower than NIM_LATENCY_MAX_MS or failing => ok:false.
#
# Usage: sh scripts/nim_health.sh   (session start / llmr-health / /refresh / launchd 15-min timer)
set -e
cd "$(dirname "$0")/.."
. ./.env
KEY="$LITELLM_MASTER_KEY"
URL="http://localhost:${LLMR_PORT:-4040}/v1/chat/completions"   # LLMR_PORT lets a staged stack audit itself
MAX_MS="${NIM_LATENCY_MAX_MS:-8000}"
# No point waiting far past the bench threshold — anything over it is benched anyway.
CURL_TIMEOUT="${NIM_PROBE_TIMEOUT_S:-$(( MAX_MS / 1000 + 3 ))}"
OUT="model_health.yaml"

# FREE aliases only (NIM + Mistral free + zen-free). Latency-probing is for LOAD-VARIABLE free
# models. PAID/flat backends are NOT probed — it just burns tokens for nothing; if the key/plan
# works, the models work:
#   - z.ai Coding Plan (zai-*): flat/paid — NOT latency-probed. Its KEY is verified once at startup
#     by scripts/zai_auth.sh (a free /models GET → {creds:true}), analogous to the Claude OAuth
#     proxy creds check. That confirms auth without spending completion tokens.
#   - opencode GO (zen-*) + brains: flat/quota-metered — not probed (would drain quota).
#   - Anthropic OAuth (ant-*) / Copilot: not probed; if the OAuth/credits work, the models work.
# Unprobed models fail-open (assumed healthy, keep config order) in priority_router.py.
NIM_ALIASES="nim-glm nim-deepseek nim-deepseek-flash nim-inkling nim-gptoss nim-minimax nim-nemotron nim-nemotron-super nim-step nim-llama"
MIST_ALIASES="mist-large mist-codestral mist-medium mist-magistral"     # Mistral free tier
ZEN_FREE_ALIASES="zen-free-deepseek zen-free-nemotron zen-free-pickle zen-free-ling zen-free-laguna zen-free-mimo zen-free-north"
ALL_ALIASES="$NIM_ALIASES $MIST_ALIASES $ZEN_FREE_ALIASES"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

probe() {
  a="$1"
  # Two things this probe MUST do, or it certifies dead models as healthy:
  #
  #  1. "num_retries":0,"fallbacks":[] — with the fallback chain on, a dead alias is answered by
  #     a different model and curl sees 200. That is exactly how nim-kimi sat at ok:true while
  #     every real call to it 404'd ("Function ... not found"): the health checker, whose whole
  #     job is finding dead models, was the last place the silent fallback could hide.
  #  2. assert the served model. A healthy alias echoes itself in response.model; a fallback
  #     returns the name of whatever actually answered. A 200 alone proves nothing.
  #  3. require usage.completion_tokens > 0. An exhausted-credit backend returns 200, echoes its
  #     own name, and generates nothing. Do NOT test visible content instead: a reasoning model
  #     spends this tiny cap on thinking tokens and returns finish_reason=length with content ""
  #     while being perfectly healthy. completion_tokens separates "generated nothing" from
  #     "generated, then hit the cap".
  #  4. "metadata":{"health_probe":true} — the router reroutes any alias currently marked
  #     ok:false. Without this the probe for an unhealthy model is answered by a substitute, the
  #     served-model check fails, and the model is re-marked unhealthy on every audit: once dead,
  #     dead forever, even after the provider restores it.
  #
  # max_tokens must be generous: reasoning models (GLM, DeepSeek-pro) emit
  # thinking tokens and stall under a tiny cap, giving false "slow" readings.
  body='{"model":"'"$a"'","messages":[{"role":"user","content":"Reply with the single word: OK"}],"max_tokens":16,"num_retries":0,"fallbacks":[],"metadata":{"health_probe":true}}'
  start=$(python3 -c 'import time; print(int(time.time()*1000))')
  code=$(curl -s -o "$TMP/$a.body" -w '%{http_code}' -m "$CURL_TIMEOUT" "$URL" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "$body" || echo 000)
  end=$(python3 -c 'import time; print(int(time.time()*1000))')
  ms=$((end - start))
  read served filled <<EOF
$(python3 -c 'import json,sys
try:
    d = json.load(open(sys.argv[1]))
    gen = (d.get("usage") or {}).get("completion_tokens") or 0
    print(d.get("model",""), "1" if gen > 0 else "0")
except Exception:
    print("", "0")' "$TMP/$a.body" 2>/dev/null || echo " 0")
EOF
  if [ "$code" = "200" ] && [ "$ms" -le "$MAX_MS" ] && [ "$served" = "$a" ] && [ "$filled" = "1" ]; then
    ok=true
  else
    ok=false
    if [ "$code" = "200" ]; then
      [ "$served" != "$a" ] && code="200-served:${served:-none}"
      [ "$served" = "$a" ] && [ "$filled" = "0" ] && code="200-generated-nothing"
    fi
  fi
  echo "$ok $ms $code" > "$TMP/$a"
}

# Fire all probes concurrently. (Serialising the NIM probes was tried and changed nothing: the
# slow aliases time out at ~11s either way. NIM's latency is load-variable, and MAX_MS is the
# deliberate cut -- a model too slow to use is correctly benched, not a false negative.)
for a in $ALL_ALIASES; do probe "$a" & done
wait

# Assemble output in stable alias order.
printf '# Auto-generated by nim_health.sh (all providers) — do not edit by hand.\nmodels:\n' > "$OUT"
printf '%-22s %-6s %s\n' "alias" "ok" "latency"
printf '%-22s %-6s %s\n' "----------------------" "------" "--------"
for a in $ALL_ALIASES; do
  read ok ms code < "$TMP/$a"
  printf '  %s: {ok: %s, latency_ms: %s}\n' "$a" "$ok" "$ms" >> "$OUT"
  printf '%-22s %-6s %sms (http %s)\n' "$a" "$ok" "$ms" "$code"
done

echo
echo "wrote $OUT — router picks fastest healthy per cost class (free: NIM vs zen-free compete)."
