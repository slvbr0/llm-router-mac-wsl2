#!/bin/sh
# Start the Codex/ChatGPT OAuth shim (idempotent, survives the shell that started it).
# Same pattern as start_oauth_proxy.sh: nohup + detached stdio so an SSH invocation returns.
cd "$(dirname "$0")/.."
PIDFILE=/tmp/codex-oauth-proxy.pid
PORT="${CODEX_OAUTH_PORT:-4042}"
# Bind host. macOS keeps the loopback default — Docker Desktop / OrbStack proxy
# host.docker.internal to the host's loopback, so the container reaches the proxy there.
# On Linux/WSL2 host.docker.internal resolves to the docker bridge gateway instead, and a
# loopback-bound socket is invisible to the container, so bind the bridge: containers and
# the host reach it, the LAN does not. Address is read off the interface, never hardcoded.
if [ -z "$CODEX_OAUTH_HOST" ] && [ -d /sys/class/net/docker0 ]; then
  CODEX_OAUTH_HOST=$(ip -4 -o addr show docker0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
  export CODEX_OAUTH_HOST
fi
# Health checks must probe whatever we actually bound, or a healthy bridge-bound proxy
# reports "did not answer" and start.sh looks broken when it is not.
HOST="${CODEX_OAUTH_HOST:-127.0.0.1}"
P=$(cat "$PIDFILE" 2>/dev/null)
if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then
  echo "   Codex proxy already running (pid $P) — $(curl -s -m 4 http://$HOST:$PORT/health)"
  exit 0
fi
mkdir -p logs
nohup python3 providers/codex_oauth_proxy.py > logs/codex_oauth_proxy.log 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 2
H=$(curl -s -m 5 "http://$HOST:$PORT/health" 2>/dev/null)
if [ -n "$H" ]; then echo "   Codex proxy started (pid $(cat $PIDFILE)) — $H"
else echo "   ✗ Codex proxy did not answer on $HOST:$PORT (see logs/codex_oauth_proxy.log)"; fi
exit 0
