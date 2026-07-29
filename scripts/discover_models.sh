#!/bin/sh
# List live model IDs from NIM + Zen. Update config.yaml litellm_params.model on drift.
set -e
cd "$(dirname "$0")/.."
[ -f .env ] && . ./.env
echo "=== NVIDIA NIM models ==="
[ -n "$NVIDIA_API_KEY" ] && curl -s https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NVIDIA_API_KEY" | python3 -m json.tool | grep '"id"' | head -80 || echo "no NVIDIA_API_KEY"
echo "=== opencode Zen models ==="
[ -n "$ZEN_API_KEY" ] && curl -s https://opencode.ai/zen/v1/models \
  -H "Authorization: Bearer $ZEN_API_KEY" | python3 -m json.tool | grep '"id"' | head -80 || echo "no ZEN_API_KEY (or Zen has no /models endpoint yet)"
echo
echo "On drift: edit config.yaml litellm_params.model, then: docker compose restart litellm"
