#!/usr/bin/env bash
# SplatLab — bring up ngrok tunnels so rented Vast GPU boxes can reach this Mac.
#
# The Mac sits behind carrier-grade NAT (no inbound — a "static" IP that isn't
# routable to this machine), so a rented box can't dial in directly. ngrok gives
# each local control-plane port a public TCP address that the box connects OUT to.
# This script starts the three tunnels, discovers their (randomly-assigned, on the
# free tier) public addresses, and writes them into deploy/.env as BOX_* — which
# the pool forwards to each rented box.
#
#   bash deploy/ngrok_up.sh
#
# Then (re)start the pool so it picks up the new addresses:
#   docker compose -f deploy/docker-compose.yml --profile pool up -d pool
#
# Re-run this whenever ngrok restarts — free-tier addresses change each session.
# (A paid ngrok plan gives static addresses; then this only needs running once.)
#
# Prerequisites:
#   * ngrok installed            brew install ngrok
#   * NGROK_AUTHTOKEN in .env     ngrok.com -> Your Authtoken
#   * TCP enabled on the account  ngrok requires credit-card verification for TCP
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$DEPLOY_DIR/.env"
CFG_FILE="$DEPLOY_DIR/.ngrok.yml"     # gitignored — holds the authtoken
LOG_FILE="$DEPLOY_DIR/ngrok.log"      # gitignored

command -v ngrok >/dev/null 2>&1 || { echo "!! ngrok not installed — 'brew install ngrok'" >&2; exit 1; }
[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — cp deploy/.env.example deploy/.env" >&2; exit 1; }

# Read single values from .env without exporting the whole file.
getenv() { grep -E "^$1=" "$ENV_FILE" | tail -n1 | cut -d= -f2- ; }
NGROK_AUTHTOKEN="$(getenv NGROK_AUTHTOKEN)"
REDIS_PASSWORD="$(getenv REDIS_PASSWORD)"
[ -n "$NGROK_AUTHTOKEN" ] || { echo "!! set NGROK_AUTHTOKEN in deploy/.env (ngrok.com -> Your Authtoken)" >&2; exit 1; }
[ -n "$REDIS_PASSWORD" ]  || { echo "!! set REDIS_PASSWORD in deploy/.env first" >&2; exit 1; }

# Self-contained ngrok config: authtoken + the three control-plane tunnels. The
# addr is the host-published compose port (docker-compose maps 7233/6379/9000).
cat > "$CFG_FILE" <<YAML
version: "3"
agent:
  authtoken: $NGROK_AUTHTOKEN
tunnels:
  temporal: { proto: tcp, addr: 7233 }
  redis:    { proto: tcp, addr: 6379 }
  minio:    { proto: tcp, addr: 9000 }
YAML
chmod 600 "$CFG_FILE"

# (Re)start the agent so the local API reflects only our three tunnels.
pkill -f "ngrok start --all --config $CFG_FILE" 2>/dev/null || true
sleep 1
nohup ngrok start --all --config "$CFG_FILE" >"$LOG_FILE" 2>&1 &
echo "ngrok pid $! — inspector at http://127.0.0.1:4040"

# Discover the assigned addresses from ngrok's local API and upsert BOX_* into .env.
python3 - "$ENV_FILE" "$REDIS_PASSWORD" <<'PY'
import json, re, sys, time, urllib.request

env_file, redis_pw = sys.argv[1], sys.argv[2]
API = "http://127.0.0.1:4040/api/tunnels"
WANT = ("temporal", "redis", "minio")

def fetch():
    with urllib.request.urlopen(API, timeout=3) as r:
        return {t["name"]: t["public_url"] for t in json.load(r).get("tunnels", [])}

tun = {}
for _ in range(30):
    try:
        tun = fetch()
    except Exception:
        tun = {}
    if all(k in tun for k in WANT):
        break
    time.sleep(1)

missing = [k for k in WANT if k not in tun]
if missing:
    sys.exit(f"!! ngrok tunnels not up: missing {missing}. Check deploy/ngrok.log — "
             "the free tier needs credit-card verification to enable TCP endpoints.")

hostport = lambda url: url.split("://", 1)[-1]   # tcp://7.tcp.ngrok.io:12345 -> 7.tcp.ngrok.io:12345
vals = {
    "BOX_TEMPORAL_ADDRESS": hostport(tun["temporal"]),
    "BOX_REDIS_URL": f"redis://:{redis_pw}@{hostport(tun['redis'])}",
    "BOX_S3_ENDPOINT": f"http://{hostport(tun['minio'])}",
}

lines = open(env_file).read().splitlines()
seen = set()
for i, line in enumerate(lines):
    m = re.match(r"^(\w+)=", line)
    if m and m.group(1) in vals:
        lines[i] = f"{m.group(1)}={vals[m.group(1)]}"
        seen.add(m.group(1))
for k, v in vals.items():
    if k not in seen:
        lines.append(f"{k}={v}")
open(env_file, "w").write("\n".join(lines) + "\n")

print("updated deploy/.env:")
for k, v in vals.items():
    print(f"  {k}={re.sub(r':[^@/]*@', ':***@', v)}")
PY

echo
echo "Next: docker compose -f deploy/docker-compose.yml --profile pool up -d pool"
