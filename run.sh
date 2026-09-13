#!/usr/bin/env bash
# Start, stop and check the VibeVoice demo.
#
#   ./run.sh            start, wait for the model, print the URLs
#   ./run.sh stop       shut down (weights, modes, vocabulary and key all persist)
#   ./run.sh status     what is running, and whether the model is loaded
#   ./run.sh logs       follow the backend log
#   ./run.sh restart    stop, then start

set -euo pipefail
cd "$(dirname "$0")"

# .env allows trailing comments, so strip them along with any stray whitespace.
env_value() {
  grep -E "^$1=" .env 2>/dev/null | tail -1 \
    | sed -E 's/^[^=]+=//; s/[[:space:]]*#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//' || true
}
WEB_PORT="$(env_value WEB_PORT)"; WEB_PORT="${WEB_PORT:-8080}"
API_PORT="$(env_value API_PORT)"; API_PORT="${API_PORT:-8001}"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
dim()  { printf '\033[2m%s\033[0m\n' "$1"; }

health() { curl -fsS "http://localhost:${API_PORT}/api/health" 2>/dev/null || true; }

# Parse the JSON rather than grepping it: FastAPI emits compact JSON, so a
# pattern like '"status": "ready"' silently never matches and the wait below
# would spin until its timeout while the model was already up.
health_status() {
  health | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true
}

wait_ready() {
  local waited=0 status
  printf 'Loading the checkpoint onto the GPU'
  while true; do
    status="$(health_status)"
    [ "$status" = "ready" ] && break
    # Surface a load failure instead of spinning until the timeout.
    if [ "$status" = "error" ]; then
      echo; echo "The backend failed to load the model:"
      health | python3 -c 'import json,sys; print(" ", json.load(sys.stdin).get("error"))' 2>/dev/null
      echo "Try: ./run.sh logs"
      return 1
    fi
    if [ "$waited" -ge 300 ]; then
      echo; echo "Timed out after ${waited}s. Try: ./run.sh logs"; return 1
    fi
    printf '.'; sleep 3; waited=$((waited + 3))
  done
  echo " ready in ${waited}s"
}

start() {
  if [ ! -f models/VibeVoice-ASR-Streaming-7B/preprocessor_config.json ]; then
    echo "The checkpoint is missing from ./models -- see the README for the hf download command." >&2
    exit 1
  fi
  docker compose up -d
  wait_ready || exit 1
  echo
  bold "  ASR demo      http://localhost:${WEB_PORT}"
  bold "  Voice editing http://localhost:${WEB_PORT}/editor.html"
  dim  "  API           http://localhost:${API_PORT}/api/health"
  # No quotes inside the f-string expressions: the whole program is already
  # wrapped in shell single quotes.
  health | python3 -c '
import json, sys
d = json.load(sys.stdin)
g = d.get("gpu") or {}
name = g.get("name") or "GPU"
used = g.get("reserved_gb") or 0
total = g.get("total_gb") or 0
secs = d.get("load_seconds") or 0
print(f"  {name} · {used:.1f}/{total:.0f} GB · loaded in {secs:.0f}s")' 2>/dev/null || true
}

stop() {
  docker compose down
  echo "Stopped. Weights, modes, learned vocabulary and your API key are all kept."
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  logs)    docker compose logs -f backend ;;
  status)
    docker compose ps
    s="$(health)"
    if [ -n "$s" ]; then
      echo "$s" | python3 -c '
import json, sys
d = json.load(sys.stdin)
g = d.get("gpu") or {}
status = d.get("status")
name = g.get("name") or ""
used = g.get("reserved_gb") or 0
total = g.get("total_gb") or 0
print(f"model: {status}  {name} {used:.1f}/{total:.0f} GB")'
    else
      echo "model: backend not reachable on :${API_PORT}"
    fi ;;
  *) echo "usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
