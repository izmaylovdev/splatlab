#!/usr/bin/env bash
# SplatLab — connect this Mac to the GCP relay so rented Vast boxes can reach it.
#
# Replaces deploy/ngrok_up.sh. Opens an outbound SSH session to the relay VM
# (created by deploy/gcp_relay_create.sh) with reverse port forwards, so the
# relay's public ports lead to the control plane here:
#
#     relay:17233 -> localhost:7233  (Temporal)
#     relay:16379 -> localhost:6379  (Redis)
#     relay:19000 -> localhost:9000  (MinIO / S3)
#
# Each service gets its own SSH session, so a bulk MinIO transfer can't queue in
# front of Temporal/Redis control traffic inside one shared connection (measured:
# Redis connect+ping went from ~0.5s to 2.5-8.5s during a dataset download, and
# boxes timed out). Each session runs in a reconnect loop in the background (log: deploy/relay.log)
# and the box-facing addresses are written into deploy/.env as BOX_*. The relay
# IP is static, so unlike ngrok these never change — re-running this script just
# restarts the tunnel.
#
#   bash deploy/gcp_tunnel_up.sh          # start / restart
#   bash deploy/gcp_tunnel_up.sh stop     # stop the tunnel
#
# Then (re)start the pool so it forwards the addresses to new boxes:
#   docker compose -f deploy/docker-compose.yml --profile pool up -d pool
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$DEPLOY_DIR/.env"
LOG_FILE="$DEPLOY_DIR/relay.log"           # gitignored
PID_FILE="$DEPLOY_DIR/.relay.pid"          # gitignored
KNOWN_HOSTS="$DEPLOY_DIR/.relay_known_hosts"

[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found" >&2; exit 1; }
getenv() { grep -E "^$1=" "$ENV_FILE" | tail -n1 | cut -d= -f2- ; }
HOST="$(getenv GCP_RELAY_HOST)"
RUSER="$(getenv GCP_RELAY_USER)"
KEY="$(getenv GCP_RELAY_KEY)"
REDIS_PASSWORD="$(getenv REDIS_PASSWORD)"
[ -n "$HOST" ] && [ -n "$RUSER" ] && [ -n "$KEY" ] || {
  echo "!! GCP_RELAY_* not set — run deploy/gcp_relay_create.sh first" >&2; exit 1; }

stop_tunnel() {
  if [ -f "$PID_FILE" ]; then kill $(cat "$PID_FILE") 2>/dev/null || true; rm -f "$PID_FILE"; fi
  pkill -f "ssh -N -i $KEY" 2>/dev/null || true
}
stop_tunnel
if [ "${1:-}" = "stop" ]; then echo "relay tunnel stopped"; exit 0; fi

# One reconnect loop per service. ExitOnForwardFailure makes ssh quit (and the
# loop retry) if the relay still holds the port from a dead session; the relay's
# sshd drops those within ~45s (ClientAliveInterval 15 x 3).
tunnel_loop() {  # $1 = relay port, $2 = local port
  while true; do
    ssh -N -i "$KEY" -o IdentitiesOnly=yes \
      -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN_HOSTS" \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -R "0.0.0.0:$1:localhost:$2" "$RUSER@$HOST" || true
    echo "$(date -u +%FT%TZ) tunnel $1->$2 down, reconnecting in 5s"
    sleep 5
  done
}
: > "$PID_FILE"
for fwd in 17233:7233 16379:6379 19000:9000; do   # Temporal, Redis, MinIO
  tunnel_loop "${fwd%%:*}" "${fwd##*:}" >>"$LOG_FILE" 2>&1 &
  echo $! >> "$PID_FILE"
done
echo "relay tunnels up (loop pids $(tr '\n' ' ' < "$PID_FILE")) — log: $LOG_FILE"

# Wait until the relay's public ports really lead back here (Redis answers PONG).
python3 - "$HOST" "$REDIS_PASSWORD" <<'PY'
import socket, sys, time
host, pw = sys.argv[1], sys.argv[2]
for _ in range(40):
    try:
        with socket.create_connection((host, 16379), timeout=3) as s:
            s.sendall(f"AUTH {pw}\r\nPING\r\n".encode())
            time.sleep(0.5)
            if b"PONG" in s.recv(256):
                print(f"relay up: {host}:16379 -> local Redis answered PONG")
                sys.exit(0)
    except OSError:
        pass
    time.sleep(1.5)
sys.exit("!! relay ports not reachable yet — check deploy/relay.log")
PY

# Upsert the box-facing addresses into .env (same keys ngrok_up.sh wrote).
python3 - "$ENV_FILE" "$HOST" "$REDIS_PASSWORD" <<'PY'
import re, sys
env_file, host, redis_pw = sys.argv[1:]
vals = {
    "BOX_TEMPORAL_ADDRESS": f"{host}:17233",
    "BOX_REDIS_URL": f"redis://:{redis_pw}@{host}:16379",
    "BOX_S3_ENDPOINT": f"http://{host}:19000",
}
lines = open(env_file).read().splitlines()
seen = set()
for i, line in enumerate(lines):
    m = re.match(r"^(\w+)=", line)
    if m and m.group(1) in vals:
        lines[i] = f"{m.group(1)}={vals[m.group(1)]}"
        seen.add(m.group(1))
lines += [f"{k}={v}" for k, v in vals.items() if k not in seen]
open(env_file, "w").write("\n".join(lines) + "\n")
print("updated deploy/.env:")
for k, v in vals.items():
    print(f"  {k}={re.sub(r':[^@/]*@', ':***@', v)}")
PY

echo
echo "Next: docker compose -f deploy/docker-compose.yml --profile pool up -d pool"
