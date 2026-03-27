#!/usr/bin/env bash
# Monthly / scheduled model retrain. Run from cron, e.g. first Sunday 01:00 UTC:
#   0 1 * * 0 [ "$(date +\%u)" -eq 7 ] && [ "$(date +\%d)" -le 07 ] && /path/to/scripts/refresh_model.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python run_retrain_model.py "$@"
