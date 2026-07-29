#!/bin/sh
# Background health refresher: re-audits ALL provider latencies every 15 min while a
# session is up, so the router always picks from fresh data. Started by start.sh
# (idempotent via pidfile). NOTE: on macOS launchd/cron can't run this — TCC denies them
# ~/Documents access; a session-scoped loop inherits your permissions instead. On Linux/WSL2
# there is no such restriction (systemd --user or cron would also work); the loop is kept for
# both so one code path covers both platforms.
# Stop: kill $(cat /tmp/llmr-health-timer.pid)
set -e
cd "$(dirname "$0")/.."
PIDFILE=/tmp/llmr-health-timer.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "health refresher already running (pid $(cat "$PIDFILE"))"
  exit 0
fi
# Re-audit on TWO signals, polling every 30s:
#   1. baseline — every 15 min (900s) regardless.
#   2. paid-trigger — the router touches .llmr-refresh-trigger after a PAID model answers; we then
#      re-audit the FREE models so a recovered one is picked on the next prompt instead of lingering
#      on paid. Debounced: at most one trigger-audit per 60s (free probes are $0, but don't thrash).
# `trap '' HUP` so the refresher outlives the starting shell — over SSH/headless the loop
# otherwise dies on SIGHUP when the session closes and health data silently goes stale.
(
  trap '' HUP
  LAST_AUDIT=0
  LAST_TRIG=0
  while true; do
    NOW=$(date +%s)
    # file mtime — BSD stat (macOS) and GNU stat (Linux/WSL2) take different flags.
    TRIG=$(stat -f %m .llmr-refresh-trigger 2>/dev/null || stat -c %Y .llmr-refresh-trigger 2>/dev/null || echo 0)
    paid_fired=0
    [ "$TRIG" -gt "$LAST_TRIG" ] && [ $((NOW - LAST_AUDIT)) -ge 60 ] && paid_fired=1
    if [ "$paid_fired" = 1 ] || [ $((NOW - LAST_AUDIT)) -ge 900 ]; then
      sh scripts/nim_health.sh >> logs/health_timer.log 2>&1
      LAST_AUDIT=$(date +%s)
      LAST_TRIG=$TRIG
    fi
    sleep 30
  done
) > /dev/null 2>&1 < /dev/null &
# ^ detach the loop's stdio. Without this the subshell keeps the caller's stdout open, so a
# non-interactive invocation (ssh host 'install_health_timer.sh', cron, CI) never returns even
# though the refresher started fine. Its real output already goes to logs/health_timer.log.
echo $! > "$PIDFILE"
echo "health refresher started (pid $!) — 15-min baseline + paid-triggered free re-audit (log: logs/health_timer.log)"
