#!/bin/sh
# z.ai creds check — the paid/flat analogue of the Claude OAuth proxy check.
#
# z.ai is a PAID/flat plan: we do NOT latency-probe its models (that just burns tokens — if the
# plan works, the models work). Instead we verify the KEY authenticates, once at startup, via a
# FREE models listing (GET /models spends no completion tokens). Prints a status line just like the
# OAuth proxy's {status, creds}. Non-fatal: on failure it reports and returns 0 so startup continues.
cd "$(dirname "$0")/.." 2>/dev/null || true
. ./.env 2>/dev/null || true
KEY="${ZAI_API_KEY:-}"

if [ -z "$KEY" ]; then
  echo '   {"status": "error", "creds": false, "reason": "ZAI_API_KEY not set in .env"}'
  exit 0
fi

code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 \
  https://api.z.ai/api/paas/v4/models -H "Authorization: Bearer $KEY" 2>/dev/null || echo 000)

if [ "$code" = "200" ]; then
  echo '   {"status": "ok", "creds": true}'
else
  # 401 = bad/expired key; 000 = network. Models still route (fail-open); this is a heads-up.
  echo "   {\"status\": \"error\", \"creds\": false, \"http\": $code}"
fi
exit 0
