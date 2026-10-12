#!/usr/bin/env bash
# SplatLab — create the Google Cloud relay VM that replaces ngrok.
#
# Rented Vast boxes can't reach this Mac directly (carrier-grade NAT, no inbound),
# and ngrok's free tier caps monthly bandwidth. Instead: a tiny GCP VM with a
# static IP. The Mac opens an outbound SSH session to it with reverse port
# forwards (deploy/gcp_tunnel_up.sh), and boxes dial VM_IP:<port>.
#
#   GCP_PROJECT=<project-id> bash deploy/gcp_relay_create.sh
#
# Creates (idempotently): an SSH key ~/.ssh/splatlab_relay, a static IP, a
# firewall rule (SSH + the three relay ports), and an e2-micro VM — free-tier
# eligible in us-central1/us-east1/us-west1. Writes GCP_RELAY_HOST/_USER/_KEY into
# deploy/.env for gcp_tunnel_up.sh.
#
# Exposure matches the ngrok setup it replaces: the relay ports are reachable from
# the internet (Vast box IPs vary too much to allow-list). Redis and MinIO are
# password-protected; Temporal has no auth.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$DEPLOY_DIR/.env"

PROJECT="${GCP_PROJECT:?set GCP_PROJECT=<project-id>}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-$REGION-a}"
NAME="${GCP_RELAY_NAME:-splatlab-relay}"
RELAY_USER=relay
KEY="$HOME/.ssh/splatlab_relay"
# Relay-side ports (high, to keep internet scanner noise down) -> Mac services.
PORTS="17233,16379,19000"

g() { gcloud --project "$PROJECT" --quiet "$@"; }

[ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -C "splatlab-relay" -f "$KEY" >/dev/null
echo "ssh key: $KEY"

g compute addresses describe "$NAME" --region "$REGION" >/dev/null 2>&1 \
  || g compute addresses create "$NAME" --region "$REGION"
IP=$(g compute addresses describe "$NAME" --region "$REGION" --format='value(address)')
echo "static IP: $IP"

g compute firewall-rules describe "$NAME" >/dev/null 2>&1 \
  || g compute firewall-rules create "$NAME" --network default --direction INGRESS \
       --allow "tcp:22,tcp:${PORTS//,/,tcp:}" --target-tags "$NAME" \
       --description "SplatLab relay: SSH from the Mac + box-facing relay ports"

# sshd must let the Mac's reverse forwards bind on the public interface, and
# should notice a dead Mac session quickly so a reconnect can re-bind the ports.
STARTUP=$(cat <<'SH'
#!/bin/sh
cat >/etc/ssh/sshd_config.d/splatlab-relay.conf <<'CONF'
GatewayPorts clientspecified
ClientAliveInterval 15
ClientAliveCountMax 3
CONF
systemctl reload ssh || systemctl reload sshd
SH
)

if ! g compute instances describe "$NAME" --zone "$ZONE" >/dev/null 2>&1; then
  g compute instances create "$NAME" --zone "$ZONE" \
    --machine-type e2-micro \
    --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size 30GB --boot-disk-type pd-standard \
    --address "$IP" --tags "$NAME" \
    --metadata "ssh-keys=$RELAY_USER:$(cat "$KEY.pub")" \
    --metadata-from-file startup-script=<(echo "$STARTUP")
fi

# Record the relay for gcp_tunnel_up.sh (upsert into .env).
python3 - "$ENV_FILE" "$IP" "$RELAY_USER" "$KEY" <<'PY'
import re, sys
env_file, ip, user, key = sys.argv[1:]
vals = {"GCP_RELAY_HOST": ip, "GCP_RELAY_USER": user, "GCP_RELAY_KEY": key}
lines = open(env_file).read().splitlines()
seen = set()
for i, line in enumerate(lines):
    m = re.match(r"^(\w+)=", line)
    if m and m.group(1) in vals:
        lines[i] = f"{m.group(1)}={vals[m.group(1)]}"
        seen.add(m.group(1))
lines += [f"{k}={v}" for k, v in vals.items() if k not in seen]
open(env_file, "w").write("\n".join(lines) + "\n")
PY
echo "relay recorded in deploy/.env (GCP_RELAY_HOST=$IP)"
echo "Next: bash deploy/gcp_tunnel_up.sh"
