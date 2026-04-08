#!/bin/sh
# llama-lifecycle.sh -- lifecycle manager + monitoring sidecar for llama-server
# Runs inside the atlas-llama-lifecycle Alpine container.
#
# Controls llama-server via signal files in /signals/ volume shared with
# the WSL2 host. llama-bridge.py watches these signal files and starts/stops
# the native Windows llama-server via PowerShell interop.
# Bridge runs as systemd user service, auto-starts with WSL2.
#
# *** WARNING: DO NOT replace signal-file I/O with HTTP calls (e.g. wget) ***
# Docker Desktop's host.docker.internal resolves to the Docker Desktop VM,
# NOT to WSL2 where llama-bridge.py runs. HTTP from this container to the
# bridge WILL FAIL silently. Signal files via the shared /signals/ volume
# are the ONLY reliable communication path. (Fixed in commit e5d6da5.)

mkdir -p /signals
apk add --no-cache jq python3 >/dev/null 2>&1

LLAMA_HOST="host.docker.internal"
LLAMA_PORT="8090"
LLAMA_URL="http://${LLAMA_HOST}:${LLAMA_PORT}"
# --- Wait for llama-bridge readiness via shared volume ---
# Bridge writes PID to /signals/bridge.pid on startup. We check file
# existence only (PID is in WSL2 namespace, not Docker's).
echo "[lifecycle] Waiting for llama-bridge (/signals/bridge.pid)..."
BRIDGE_TRIES=0
while [ ! -f /signals/bridge.pid ]; do
  BRIDGE_TRIES=$((BRIDGE_TRIES + 1))
  if [ "$BRIDGE_TRIES" -ge 30 ]; then
    echo "[lifecycle] ERROR: llama-bridge not ready after 60s"
    echo "[lifecycle] Fix: systemctl --user enable --now llama-bridge"
    exit 1
  fi
  sleep 2
done
echo "[lifecycle] llama-bridge ready (pid=$(cat /signals/bridge.pid))"

# --- Signal llama-server start/stop via shared volume ---
# Timestamped commands ensure the bridge detects each new request.
request_start() {
  echo "start-$(date +%s)" > /signals/llama-command
}

request_stop() {
  echo "stop-$(date +%s)" > /signals/llama-command
}

if ! wget -qO /dev/null -T 5 "${LLAMA_URL}/health" 2>/dev/null; then
  echo "[lifecycle] Requesting llama-server start..."
  request_start
else
  echo "[lifecycle] llama-server already running"
fi

# Wait for llama-server health (up to 180s)
echo "[lifecycle] Waiting for llama-server health..."
TRIES=0
while ! wget -qO /dev/null -T 5 "${LLAMA_URL}/health" 2>/dev/null; do
  TRIES=$((TRIES + 1))
  if [ "$TRIES" -ge 90 ]; then
    echo "[lifecycle] ERROR: llama-server not healthy after 180s"
    exit 1
  fi
  sleep 2
done
echo "[lifecycle] llama-server healthy on port ${LLAMA_PORT}"

# SIGTERM -> stop llama-server (docker compose down triggers this)
cleanup() {
  echo "[lifecycle] SIGTERM received, signaling llama-server stop..."
  request_stop
  sleep 2
  exit 0
}
trap cleanup TERM INT

PREV_SLOT_STATE="idle"
SWAP_IN_PROGRESS=false
SWAP_START_TS=0

# --- HTTP endpoint for container-to-container swap requests ---
# atlas-proxy calls POST /swap {"model":"...","draft":"..."}
# lifecycle writes to /signals/llama-command for the bridge.
# This is the correct path: proxy->lifecycle(HTTP)->bridge(signal file)->WSL2
cat > /tmp/lifecycle_http.py <<'PYEOF'
import http.server, json, os, sys, time

class SwapHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/swap":
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n)) if n else {}
            m, dr = d.get("model", ""), d.get("draft", "")
            if m:
                cmd = "swap:%s:%s:%d" % (m, dr, int(time.time()))
                with open("/signals/llama-command", "w") as f:
                    f.write(cmd)
                print("[lifecycle-http] Swap forwarded: %s" % cmd, flush=True)
                self._json(200, {"status": "swap_initiated", "command": cmd})
            else:
                self._json(400, {"error": "model field required"})
        else:
            self._json(404, {"error": "not_found"})

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "service": "llama-lifecycle"})
        elif self.path == "/model":
            cm = ""
            try:
                with open("/signals/current-model") as f:
                    cm = f.read().strip()
            except FileNotFoundError:
                pass
            self._json(200, {"current_model": cm})
        else:
            self._json(404, {"error": "not_found"})

    def _json(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, fmt, *args):
        pass

class SwapServer(http.server.HTTPServer):
    allow_reuse_address = True

if __name__ == "__main__":
    server = SwapServer(("0.0.0.0", 8080), SwapHandler)
    print("[lifecycle-http] Listening on 0.0.0.0:8080", flush=True)
    server.serve_forever()
PYEOF
python3 /tmp/lifecycle_http.py &
LIFECYCLE_HTTP_PID=$!
echo "[lifecycle] HTTP swap endpoint on :8080 (PID=$LIFECYCLE_HTTP_PID)"

# Stream filtered llama-server logs to docker logs
tail -F /signals/llama-server.log 2>/dev/null \
  | awk '/log_server_r|[Ee]rror|[Ww]arn|[Ff]atal|main:|listening|loaded/{print;fflush()}' &

# Real-time inference monitor via /slots + /health
echo "[lifecycle] Monitoring llama-server (polling /slots every 3s)..."
while true; do
  sleep 3 & wait $!

  if ! wget -qO- -T 5 "${LLAMA_URL}/health" >/dev/null 2>&1; then
    if $SWAP_IN_PROGRESS; then
      NOW_TS=$(date +%s)
      if [ $((NOW_TS - SWAP_START_TS)) -lt 120 ]; then
        echo "[lifecycle] llama-server down during model swap, waiting..."
        PREV_SLOT_STATE="idle"
        continue
      else
        echo "[lifecycle] Model swap timed out after 120s"
        SWAP_IN_PROGRESS=false
      fi
    fi
    echo "[lifecycle] llama-server lost, requesting restart..."
    request_start
    PREV_SLOT_STATE="idle"
    continue
  fi

  # If swap was in progress and health restored, swap is complete
  if $SWAP_IN_PROGRESS; then
    echo "[lifecycle] Model swap complete - llama-server healthy"
    SWAP_IN_PROGRESS=false
  fi

  SLOTS_JSON=$(wget -qO- -T 5 "${LLAMA_URL}/slots" 2>/dev/null) || continue
  IS_PROC=$(echo "$SLOTS_JSON" | jq -r '.[0].is_processing // false' 2>/dev/null) || { PREV_SLOT_STATE="idle"; continue; }
  SLOT_STATE=$([ "$IS_PROC" = "true" ] && echo "active" || echo "idle")
  N_PRED=$(echo "$SLOTS_JSON" | jq -r '.[0].n_predicted // 0' 2>/dev/null) || continue
  N_CTX=$(echo "$SLOTS_JSON" | jq -r '.[0].n_past // 0' 2>/dev/null) || continue
  TASK_ID=$(echo "$SLOTS_JSON" | jq -r '.[0].id_task // "-"' 2>/dev/null) || continue

  if [ "$SLOT_STATE" = "active" ] && [ "$PREV_SLOT_STATE" = "idle" ]; then
    echo "[lifecycle] >>> INFERENCE STARTED | task=$TASK_ID prompt_tokens=$N_CTX"
    PREV_SLOT_STATE="active"
  elif [ "$SLOT_STATE" = "active" ] && [ "$PREV_SLOT_STATE" = "active" ]; then
    echo "[lifecycle]     generating | task=$TASK_ID predicted=$N_PRED ctx=$N_CTX"
  elif [ "$SLOT_STATE" = "idle" ] && [ "$PREV_SLOT_STATE" = "active" ]; then
    echo "[lifecycle] <<< INFERENCE COMPLETE | task=$TASK_ID total_predicted=$N_PRED"
    PREV_SLOT_STATE="idle"
  fi
done
