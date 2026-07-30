#!/bin/sh
# Start the Codex/ChatGPT OAuth shim (idempotent, survives the shell that started it).
# Same pattern as start_oauth_proxy.sh: nohup + detached stdio so an SSH invocation returns.
cd "$(dirname "$0")/.."
PIDFILE=/tmp/codex-oauth-proxy.pid
PORT="${CODEX_OAUTH_PORT:-4042}"
P=$(cat "$PIDFILE" 2>/dev/null)
if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then
  echo "   Codex proxy already running (pid $P) — $(curl -s -m 4 http://127.0.0.1:$PORT/health)"
  exit 0
fi
mkdir -p logs
nohup python3 providers/codex_oauth_proxy.py > logs/codex_oauth_proxy.log 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 2
H=$(curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)
if [ -n "$H" ]; then echo "   Codex proxy started (pid $(cat $PIDFILE)) — $H"
else echo "   ✗ Codex proxy did not answer on :$PORT (see logs/codex_oauth_proxy.log)"; fi
exit 0
