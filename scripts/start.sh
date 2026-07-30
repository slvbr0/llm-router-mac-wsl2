#!/bin/sh
# Start a session: ensure the router is up, refresh NIM health, then launch opencode.
# Usage:
#   sh scripts/start.sh          → also launches the terminal opencode (TUI)
#   sh scripts/start.sh up       → router + audit only; then open the opencode APP yourself
cd "$(dirname "$0")/.."

echo "① router containers…"
if ! docker info >/dev/null 2>&1; then
  echo "   ✗ Docker not reachable — start it, then re-run."
  echo "     macOS: open the OrbStack (or Docker Desktop) app."
  echo "     WSL2:  sudo service docker start   (or start Docker Desktop on Windows)"
  exit 1
fi
# The paid-trigger file is bind-mounted into the container; it must exist as a FILE before
# `docker up` (a missing bind target makes Docker create a directory). Runtime-only, gitignored.
[ -e .llmr-refresh-trigger ] || printf '0\n' > .llmr-refresh-trigger
docker compose up -d >/dev/null 2>&1

printf "   waiting for :4040 "
i=0
while [ $i -lt 20 ]; do
  h=$(curl -s -m 4 http://localhost:4040/health/liveliness 2>/dev/null)
  [ -n "$h" ] && { echo "→ $h"; break; }
  printf "."; sleep 3; i=$((i+1))
done
[ -z "$h" ] && { echo " ✗ router not healthy — check: docker compose logs litellm"; exit 1; }

echo "② Claude OAuth proxy…"
sh scripts/start_oauth_proxy.sh

echo "②b z.ai auth (paid — creds check only, no model probing)…"
sh scripts/zai_auth.sh

echo "②c Codex/ChatGPT OAuth shim…"
sh scripts/start_codex_proxy.sh

echo "③ NIM latency audit…"
sh scripts/nim_health.sh
sh scripts/install_health_timer.sh   # 15-min background re-audit while session up

if [ "$1" = "up" ]; then
  echo "④ router ready. Open the opencode APP and pick  llm-router/auto."
  echo "   token savings anytime:  llmr-usage   ·   refresh NIM:  llmr-health"
  exit 0
fi

echo "④ launching opencode (pick llm-router/auto the first time)…"
echo "   token savings anytime:  /usage   ·   refresh NIM:  llmr-health"
exec opencode
