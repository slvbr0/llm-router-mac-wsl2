#!/bin/sh
# End a session: snapshot the audit trail. Router can stay up 24/7 (idle ~0 cost).
# Usage: sh scripts/stop.sh            # snapshot only, leave router running
#        sh scripts/stop.sh --down     # also stop the containers
cd "$(dirname "$0")/.."

echo "① audit snapshot…"
sh scripts/export_audit.sh

# Stop the OAuth shims if running. Both are started by start.sh, so both stop here —
# start.sh gained the Codex proxy without stop.sh gaining the matching kill, which left
# an orphan holding :4042 across a --down.
stop_proxy() {
  pidfile="$1"; label="$2"
  [ -f "$pidfile" ] || return 0
  pid=$(cat "$pidfile" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null && echo "   $label proxy stopped."
  fi
  rm -f "$pidfile"
}
stop_proxy /tmp/claude-oauth-proxy.pid Claude
stop_proxy /tmp/codex-oauth-proxy.pid Codex

if [ "$1" = "--down" ]; then
  echo "② stopping router containers (data kept in volumes)…"
  docker compose stop
  echo "   restart later with: sh scripts/start.sh"
else
  echo "② leaving router up (restart: unless-stopped — no action needed)."
fi
