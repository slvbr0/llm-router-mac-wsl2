#!/bin/sh
# Snapshot the routing/spend audit trail to logs/audit-*.csv. --push commits+pushes.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
stamp=$(date +%Y%m%d-%H%M%S); out="logs/audit-$stamp.csv"
docker exec litellm-db psql -U litellm -d litellm -P pager=off --csv -c "
  SELECT \"startTime\" AS time, model_group AS routed_alias, model AS actual_model,
         prompt_tokens, completion_tokens, total_tokens, spend, \"cache_hit\" AS cache_hit
  FROM \"LiteLLM_SpendLogs\" ORDER BY \"startTime\" ASC;" > "$out"
rows=$(($(wc -l < "$out") - 1)); echo "exported $rows requests -> $out"
ls -1t logs/audit-*.csv 2>/dev/null | tail -n +11 | xargs -r rm --
if [ "$1" = "--push" ]; then
  git add logs/
  git commit -q -m "audit: routing snapshot $stamp ($rows requests)" || echo "nothing to commit"
  git remote get-url origin >/dev/null 2>&1 && git push -q origin main && echo "pushed" || echo "no remote — local commit only"
fi
