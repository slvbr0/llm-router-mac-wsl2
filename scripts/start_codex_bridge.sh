#!/bin/sh
# Bridge to the OTHER machine's Codex shim, so this router can reach BOTH ChatGPT subscriptions.
#
# Why a bridge and not a second creds file: the shim deliberately does not implement the OAuth
# refresh grant (the `codex` CLI owns that, and duplicating it races the CLI for the same refresh
# token). A second auth.json here would therefore go stale within the hour. Keeping each
# subscription on the machine whose CLI already maintains it sidesteps that entirely — both tokens
# stay fresh, nothing races, and no refresh code has to exist.
#
#   this box  --SSH--> peer box :4042 (peer's codex shim, peer's account)
#   localhost:4142 ----------------^        config.yaml cod2-* -> host.docker.internal:4142
#
# Idempotent: re-running replaces a stale tunnel. Safe to call from start.sh.
#
# Usage:  sh scripts/start_codex_bridge.sh [peer-ssh-target]
#   CODEX_BRIDGE_PORT   local port to expose the peer's shim on   (default 4142)
#   CODEX_BRIDGE_PEER   ssh target                                (default salva@100.83.213.106)
#   CODEX_BRIDGE_HOST   address of the shim ON the peer           (default: auto-detect)
set -e
cd "$(dirname "$0")/.."
PORT="${CODEX_BRIDGE_PORT:-4142}"
# Peer defaults to the OTHER box. On macOS that is the RTX host (ssh_config alias `rtxhost`, which
# is what binds the key — connecting by IP offers none once ssh-agent is out of the picture). On
# Linux/WSL2 it is the Mac over Tailscale.
if [ -z "$CODEX_BRIDGE_PEER" ] && [ -z "$1" ]; then
  if [ -d /sys/class/net/docker0 ]; then CODEX_BRIDGE_PEER="salva@100.118.196.115"
  else CODEX_BRIDGE_PEER="rtxhost"; fi
fi
PEER="${1:-$CODEX_BRIDGE_PEER}"

# Where the forward LISTENS locally. The router container has to reach it, and "localhost" means
# different things on the two platforms: Docker Desktop/OrbStack proxy host.docker.internal to the
# Mac's loopback, so 127.0.0.1 is right there; on Linux/WSL2 host.docker.internal resolves to the
# docker bridge gateway instead, and a loopback-bound forward is invisible to the container. Bind
# the bridge itself so ONE config.yaml (cod2-* -> host.docker.internal:4142) serves both boxes.
BIND="$CODEX_BRIDGE_BIND"
if [ -z "$BIND" ] && [ -d /sys/class/net/docker0 ]; then
  BIND=$(ip -4 -o addr show docker0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
fi
BIND="${BIND:-127.0.0.1}"
# The peer's shim listens on ITS loopback (mac) or its docker bridge (WSL2, bound 0.0.0.0).
REMOTE_SHIM="${CODEX_BRIDGE_REMOTE:-127.0.0.1}"

# Already up and answering? Leave it alone.
if curl -s -m 4 "http://$BIND:$PORT/health" 2>/dev/null | grep -q '"status": *"ok"'; then
  echo "   Codex bridge already up on $BIND:$PORT — $(curl -s -m 4 http://$BIND:$PORT/health)"
  exit 0
fi

# Clear a stale forwarder that is listening but not answering.
if command -v lsof >/dev/null 2>&1; then
  STALE=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)
  [ -n "$STALE" ] && kill $STALE 2>/dev/null && echo "   cleared stale bridge (pid $STALE)"
fi

# The peer's shim binds inside WSL2. Windows OpenSSH cannot see WSL2's docker bridge (172.17.0.1),
# so the tunnel must target WSL2's own primary IP — which CHANGES on every WSL2 reboot. Read it
# rather than hardcoding it; that is the single most common reason this bridge stops working.
HOST="$CODEX_BRIDGE_HOST"
if [ -z "$HOST" ] && [ -d /sys/class/net/docker0 ]; then
  HOST="$REMOTE_SHIM"          # peer is the Mac: its shim sits on the Mac's own loopback
fi
if [ -z "$HOST" ]; then
# NB: no 2>/dev/null inside the remote command. The peer's sshd hands it to Windows cmd,
# which does not understand that redirection and fails with 'no se puede encontrar la ruta'.
  HOST=$(ssh -o ConnectTimeout=10 -T "$PEER" 'wsl.exe hostname -I || hostname -I' 2>/dev/null | tr -d '\r' | awk '{print $1}')
fi
[ -z "$HOST" ] && { echo "   could not resolve the peer's shim address — bridge NOT started"; exit 1; }

ssh -f -N -o ConnectTimeout=15 -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L "$BIND:$PORT:$HOST:4042" "$PEER" || {
  echo "   ssh forward failed — cod2-* will be unreachable (router falls back, does not error)"; exit 1; }

sleep 2
H=$(curl -s -m 6 "http://$BIND:$PORT/health" 2>/dev/null || true)
case "$H" in
  *'"status": "ok"'*) echo "   Codex bridge up: $BIND:$PORT -> $HOST:4042 — $H" ;;
  *) echo "   bridge port is open but the peer shim did not answer (is it running there?)" ;;
esac
