#!/bin/sh
# End a session: snapshot the audit trail. Router can stay up 24/7 (idle ~0 cost).
# Usage: sh scripts/stop.sh            # snapshot only, leave router running
#        sh scripts/stop.sh --down     # also stop the containers
cd "$(dirname "$0")/.."

echo "① audit snapshot…"
sh scripts/export_audit.sh

# Stop OAuth proxy if running
if [ -f /tmp/claude-oauth-proxy.pid ] && kill -0 "$(cat /tmp/claude-oauth-proxy.pid)" 2>/dev/null; then
  kill "$(cat /tmp/claude-oauth-proxy.pid)" 2>/dev/null && rm -f /tmp/claude-oauth-proxy.pid
  echo "   OAuth proxy stopped."
fi

if [ "$1" = "--down" ]; then
  echo "② stopping router containers (data kept in volumes)…"
  docker compose stop
  echo "   restart later with: sh scripts/start.sh"
else
  echo "② leaving router up (restart: unless-stopped — no action needed)."
fi
