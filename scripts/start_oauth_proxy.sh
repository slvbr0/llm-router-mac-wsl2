#!/bin/sh
# Start the Claude OAuth proxy (idempotent via pidfile).
# The proxy translates LiteLLM Anthropic requests into OAuth Bearer auth.
# Stop: kill $(cat /tmp/claude-oauth-proxy.pid)
set -e
cd "$(dirname "$0")/.."

PIDFILE=/tmp/claude-oauth-proxy.pid
LOGFILE=logs/claude_oauth_proxy.log
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "   oauth proxy already running (pid $(cat "$PIDFILE"))"
    exit 0
fi

# Creds: file (all platforms) OR macOS Keychain (Claude Code 2.x on macOS only).
# On Linux/WSL2 there is no `security` binary — Claude Code stores the token in the file lane,
# which the proxy tries first anyway, so only the file check applies there.
CREDS_FILE="$HOME/.claude/.credentials.json"
CREDS_KEYCHAIN=""
if command -v security >/dev/null 2>&1; then
    CREDS_KEYCHAIN=$(security find-generic-password -s "Claude Code-credentials" -w 2>/dev/null || true)
fi
if [ ! -f "$CREDS_FILE" ] && [ -z "$CREDS_KEYCHAIN" ]; then
    echo "   ⚠  Claude OAuth: no credentials found (file or Keychain)"
    echo "   → Run: claude  then /login   (installs to ~/.claude/.credentials.json)"
    echo "   → Then: availability.yaml  anthropic: available: true"
    echo "   OAuth proxy NOT started — ant-* models unavailable."
    exit 0
fi

# nohup + </dev/null: the proxy must outlive the shell that started it. Without this a
# session-started proxy dies on SIGHUP when the terminal/SSH session closes, and every
# ant-* call then fails with "All connection attempts failed" long after startup looked fine.
nohup python3 providers/claude_oauth_proxy.py > "$LOGFILE" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 0.5

# Quick health check
STATUS=$(curl -s -m 3 http://127.0.0.1:4041/health 2>/dev/null || echo "")
if [ -n "$STATUS" ]; then
    echo "   OAuth proxy started (pid $(cat "$PIDFILE")) — $STATUS"
else
    echo "   OAuth proxy started (pid $(cat "$PIDFILE")) — log: $LOGFILE"
fi
