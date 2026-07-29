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
  [ -f "$f" ] && python3 -c "
import json
d = json.load(open('$f'))
print(f\"  last think: {d['model']} \xb7 think:{d['think']} \xb7 tier:{d['tier']}\")
" 2>/dev/null
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
