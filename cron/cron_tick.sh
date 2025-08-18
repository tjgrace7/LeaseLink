#!/bin/sh
set -euo pipefail

if [[ -z "${CRON_SECRET:-}" ]]; then
  echo "CRON_SECRET is not set" >&2
  exit 1
fi

# Replace with your actual domain
API_BASE="https://leaselink.onrender.com"

echo "[cron] POST $API_BASE/internal/cron/tick"
curl -sS -X POST \
  -H "x-cron-secret: $CRON_SECRET" \
  "$API_BASE/internal/cron/tick" \
  | tee /dev/stderr
