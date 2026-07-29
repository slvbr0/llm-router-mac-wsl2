#!/bin/sh
# Direct GitHub device-flow auth for Copilot; writes the token into the running
# litellm-proxy container. Uses the client_id LiteLLM's copilot integration expects.
set -e
CLIENT_ID="Iv1.b507a08c87ecfe98"
resp=$(curl -s -X POST https://github.com/login/device/code \
  -H "accept: application/json" -H "content-type: application/json" \
  -H "editor-version: vscode/1.85.1" -H "editor-plugin-version: copilot/1.155.0" \
  -H "user-agent: GithubCopilot/1.155.0" \
  -d "{\"client_id\":\"$CLIENT_ID\",\"scope\":\"read:user\"}")
device_code=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["device_code"])')
user_code=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["user_code"])')
interval=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("interval",5))')
expires_in=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("expires_in",900))')
echo "USER_CODE: $user_code"
echo "Visit https://github.com/login/device — valid ${expires_in}s"
elapsed=0
while [ "$elapsed" -lt "$expires_in" ]; do
  sleep "$interval"; elapsed=$((elapsed + interval))
  poll=$(curl -s -X POST https://github.com/login/oauth/access_token \
    -H "accept: application/json" -H "content-type: application/json" \
    -H "editor-version: vscode/1.85.1" -H "editor-plugin-version: copilot/1.155.0" \
    -H "user-agent: GithubCopilot/1.155.0" \
    -d "{\"client_id\":\"$CLIENT_ID\",\"device_code\":\"$device_code\",\"grant_type\":\"urn:ietf:params:oauth:grant-type:device_code\"}")
  token=$(echo "$poll" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("access_token",""))
except Exception: print("")')
  if [ -n "$token" ]; then
    echo "$token" | docker exec -i litellm-proxy sh -c 'mkdir -p /app/.litellm/github_copilot && cat > /app/.litellm/github_copilot/access-token'
    echo "TOKEN_WRITTEN_OK"; exit 0
  fi
  err=$(echo "$poll" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("error",""))
except Exception: print("")')
  case "$err" in slow_down) interval=$((interval+5));; authorization_pending|"") ;; *) echo "GITHUB_ERROR: $err"; exit 1;; esac
  echo "waiting... (${elapsed}s/${expires_in}s)"
done
echo "TIMED_OUT"; exit 1
