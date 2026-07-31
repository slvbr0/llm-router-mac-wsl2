#!/bin/sh
# On-demand usage snapshot: tokens per provider + free-vs-paid (saved) + NIM health
# + Perplexity quota. Minimalist, single command. Usage: sh scripts/usage.sh
cd "$(dirname "$0")/.."

echo "══════════ LLM ROUTER — USAGE ══════════"

# --- Per-provider tokens + spend (LiteLLM Postgres audit) ---
docker exec litellm-db psql -U litellm -d litellm -P pager=off -c "
  SELECT
    -- Aliases were renamed 2026-07-30 so the prefix names the BILLING LANE. Rows written before
    -- the cutover use the old scheme, and 'zen-glm' meant GO-flat then but per-token paid now —
    -- same string, different lane. Classify by date so historical totals stay truthful.
    CASE
      WHEN model_group LIKE 'nim-%'                                   THEN 'nim (free)'
      WHEN model_group LIKE 'free-%' OR model_group LIKE 'zen-free-%' THEN 'zen (free)'
      WHEN model_group LIKE 'mis-%' OR model_group LIKE 'mist-%'      THEN 'mistral (free)'
      WHEN model_group LIKE 'go-%'                                    THEN 'opencode GO (flat)'
      WHEN model_group LIKE 'zai-%'                                   THEN 'z.ai (flat)'
      WHEN model_group LIKE 'ant-%'                                   THEN 'anthropic (flat)'
      WHEN model_group LIKE 'cod-%'                                   THEN 'codex (flat)'
      WHEN model_group LIKE 'zen-%' AND \"startTime\" < timestamp '2026-07-30'
                                                                      THEN 'opencode GO (flat, legacy name)'
      WHEN model_group LIKE 'zen-%'                                   THEN 'zen (paid)'
      WHEN model_group LIKE 'co-%' OR model_group LIKE 'cop-%'        THEN 'copilot'
      ELSE 'other' END                                        AS provider,
    count(*)                                                  AS reqs,
    sum(prompt_tokens)                                        AS prompt_tok,
    sum(completion_tokens)                                    AS compl_tok,
    sum(total_tokens)                                         AS total_tok,
    round(sum(spend)::numeric, 4)                             AS spend_usd
  FROM \"LiteLLM_SpendLogs\"
  GROUP BY provider ORDER BY total_tok DESC NULLS LAST;" 2>/dev/null

# --- Free vs paid split (= tokens kept off paid backends) ---
docker exec litellm-db psql -U litellm -d litellm -P pager=off -t -c "
  SELECT
    -- FREE = zero marginal cost. Everything else is flat/subscription or true per-token; only
    -- the last two actually cost money per request, so they are what 'PAID' counts.
    'FREE tokens (nim + zen-free + mistral): ' ||
    coalesce(sum(total_tokens) FILTER (
      WHERE model_group LIKE 'nim-%' OR model_group LIKE 'free-%'
         OR model_group LIKE 'zen-free-%' OR model_group LIKE 'mis-%' OR model_group LIKE 'mist-%'),0) ||
    '   |   PAID tokens (zen per-token + copilot): ' ||
    (coalesce(sum(total_tokens) FILTER (
       WHERE model_group LIKE 'zen-%' AND model_group NOT LIKE 'zen-free-%'
         AND \"startTime\" >= timestamp '2026-07-30'),0)
     + coalesce(sum(total_tokens) FILTER (
       WHERE model_group LIKE 'co-%' OR model_group LIKE 'cop-%'),0))
  FROM \"LiteLLM_SpendLogs\";" 2>/dev/null | sed 's/^ *//'

# --- NIM live health (from last audit) ---
if [ -f model_health.yaml ]; then
  ok=$(grep -c 'ok: true' model_health.yaml 2>/dev/null || echo 0)
  bad=$(grep -c 'ok: false' model_health.yaml 2>/dev/null || echo 0)
  slow=$(grep 'ok: false' model_health.yaml | sed 's/:.*//;s/ *//' | tr '\n' ' ')
  echo "NIM HEALTH: ${ok}/$((ok+bad)) healthy   slow/down: ${slow:-none}   (refresh: sh scripts/nim_health.sh)"
else
  echo "NIM HEALTH: no audit yet — run sh scripts/nim_health.sh"
fi

# --- Perplexity (pwm) quota ---
if command -v pwm >/dev/null 2>&1 || [ -x "$HOME/.local/bin/pwm" ]; then
  PWM="$(command -v pwm || echo "$HOME/.local/bin/pwm")"
  echo "PERPLEXITY (pwm) — weekly Pro Search pool ~300:"
  "$PWM" usage 2>/dev/null | grep -E 'Pro Search|Deep Research|Browser Agent|Create Files' | sed 's/│//g;s/^ */  /'
else
  echo "PERPLEXITY: pwm not on PATH"
fi

echo "════════════════════════════════════════"
echo "Copilot billing = per-request credits (GitHub UI, not exposed here) · Zen = per-1M tokens (opencode.ai dashboard)"
