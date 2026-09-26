#!/usr/bin/env bash
# VitalGuard AI - one command to get the whole prototype up.
set -euo pipefail
cd "$(dirname "$0")"   # run from anywhere; repo root is always the script's directory

PY=${PYTHON:-python3}
PORT=${PORT:-8000}

if [ ! -f model/risk_model.joblib ]; then
  echo "▶ no model found - training (steps 6-8) ..."
  $PY train.py
fi

echo "▶ starting API + dashboard on :$PORT"
$PY -m uvicorn vitalguard.server:app --host 0.0.0.0 --port "$PORT" &
API=$!
trap 'kill $API 2>/dev/null || true' EXIT

for _ in $(seq 1 40); do
  curl -sf "http://localhost:$PORT/api/health" >/dev/null && break; sleep 0.5
done

if [ "$(curl -s "http://localhost:$PORT/api/snapshot" | $PY -c 'import json,sys;print(json.load(sys.stdin)["stats"]["patients"])')" = "0" ]; then
  echo "▶ empty database - seeding a demo ward"
  curl -s -X POST "http://localhost:$PORT/api/seed" -H 'content-type: application/json' -d '{"n":14}' >/dev/null
fi

echo
echo "  dashboard   http://localhost:$PORT/"
echo "  API docs    http://localhost:$PORT/docs"
echo "  capture     open the dashboard on a phone over HTTPS and tap 'Simulate fingertip'"
echo
[ "${STREAMLIT:-0}" = "1" ] && $PY -m streamlit run dashboard/app.py --server.port "${SPORT:-8501}" &
wait $API
