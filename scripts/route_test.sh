#!/bin/sh
# Fire one prompt per tier; show which model served each (via show_routing.sh after).
set -e
cd "$(dirname "$0")/.."
. ./.env
KEY="$LITELLM_MASTER_KEY"; URL="http://localhost:4040/v1/chat/completions"
req() { echo "=== $1 ==="; curl -s -m 40 "$URL" -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d "$2" | head -c 300; echo; echo; }
req "auto short (cheap -> nim-llama)" '{"model":"auto","messages":[{"role":"user","content":"Say hello"}]}'
req "auto [THINK] (reason -> nim-qwen-max)" '{"model":"auto","messages":[{"role":"user","content":"[THINK] prove sqrt2 irrational"}]}'
req "auto code (code -> nim-deepseek)" '{"model":"auto","messages":[{"role":"user","content":"debug:\n```python\nprint(1)\n```"}]}'
req "auto [FRONTIER] (-> cop-opus)" '{"model":"auto","messages":[{"role":"user","content":"[FRONTIER] hard design question"}]}'
req "auto [UNAVAILABLE: nim] short (-> zen free)" '{"model":"auto","messages":[{"role":"user","content":"[UNAVAILABLE: nim] say ok"}]}'
echo "=== PriorityRouter log lines ==="
docker compose logs --since 5m litellm 2>&1 | grep -i PriorityRouter | tail -10 || true
