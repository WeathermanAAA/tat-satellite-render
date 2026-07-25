#!/usr/bin/env bash
# provision_box.sh -- bring a bare Ubuntu box to "can run emit lanes" state.
#
# IDEMPOTENT: safe to re-run on an existing box; every step checks first.
# Piped in by `fleet.sh provision <box>`, so it must not depend on anything
# in the repo (the repo is one of the things it installs).
#
# What it deliberately does NOT do:
#   * write secrets -- `fleet.sh setenv` does that, so credentials never
#     travel in a script or a git object
#   * start lanes -- `fleet.sh deploy` does that, from the fleet.yml
#     assignment map, so a box never invents its own lane set
#
# Anything a new box needs belongs HERE, not in someone's shell history:
# that is the whole point -- box N+2 inherits every fix box N+1 needed.
set -euo pipefail

REPO_SSH="git@github.com:WeathermanAAA/tat-satellite-render.git"
REPO_HTTPS="https://github.com/WeathermanAAA/tat-satellite-render.git"
DIR=/root/tsr-s2

say() { echo "[provision] $*"; }

# --- 1. base packages -------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
if ! command -v docker >/dev/null; then
  say "installing docker"
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl gnupg git python3 python3-yaml >/dev/null
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin >/dev/null
  systemctl enable --now docker
else
  say "docker present: $(docker --version)"
fi
command -v git >/dev/null || apt-get install -y -qq git
python3 -c 'import yaml' 2>/dev/null || apt-get install -y -qq python3-yaml >/dev/null

# --- 2. the repo, on main, nothing else ------------------------------------
# FLEET GIT RULE: a box tracks origin/main and only origin/main. No box-only
# branches -- work that exists only on a box is invisible to every other box
# and to the next agent, and that is how a fleet silently forks.
if [ ! -d "$DIR/.git" ]; then
  say "cloning $DIR"
  if git ls-remote "$REPO_SSH" >/dev/null 2>&1; then
    git clone -q "$REPO_SSH" "$DIR"
  else
    say "no SSH deploy key yet -- cloning read-only over HTTPS"
    say "QUEUED FOR ANDREW: add this box's deploy key to the repo so it can PUSH:"
    [ -f /root/.ssh/id_ed25519.pub ] || ssh-keygen -q -t ed25519 -N '' -C "tat-box-$(hostname)" -f /root/.ssh/id_ed25519
    say "  $(cat /root/.ssh/id_ed25519.pub)"
    git clone -q "$REPO_HTTPS" "$DIR"
  fi
else
  say "repo present"
fi
cd "$DIR"
git fetch -q origin main
git checkout -q main 2>/dev/null || git checkout -q -B main origin/main
git reset -q --hard origin/main
say "repo at $(git rev-parse --short HEAD) on $(git branch --show-current)"

# --- 3. secrets file exists (values come from fleet.sh setenv) --------------
touch "$DIR/.env"; chmod 600 "$DIR/.env"
MISSING=()
for k in $(python3 - <<'PY'
import yaml
print("\n".join(yaml.safe_load(open("/root/tsr-s2/fleet.yml"))["env_keys"]))
PY
); do
  grep -q "^${k}=" "$DIR/.env" || MISSING+=("$k")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  say "MISSING SECRETS (lanes will not start): ${MISSING[*]}"
  say "  fix with:  fleet.sh setenv <box> KEY=VALUE"
else
  say "all required secrets present"
fi

# --- 4. build the emit image ------------------------------------------------
say "building tat-s2:latest"
docker compose -f docker-compose.s2.yml build -q emit-cron
say "image: $(docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' tat-s2:latest)"

# --- 5. heartbeat -----------------------------------------------------------
# A box that dies quietly is worse than one that dies loudly: its lanes just
# stop publishing and the site looks merely stale. The heartbeat makes silence
# visible -- see scripts/heartbeat.sh.
if [ ${#MISSING[@]} -eq 0 ]; then
  say "installing heartbeat timer"
  install -m 0755 "$DIR/scripts/heartbeat.sh" /usr/local/bin/tat-heartbeat.sh
  cat >/etc/systemd/system/tat-heartbeat.service <<'UNIT'
[Unit]
Description=TAT fleet heartbeat (publishes box health to R2)
[Service]
Type=oneshot
EnvironmentFile=/root/tsr-s2/.env
ExecStart=/usr/local/bin/tat-heartbeat.sh
UNIT
  cat >/etc/systemd/system/tat-heartbeat.timer <<'UNIT'
[Unit]
Description=Run the TAT fleet heartbeat every minute
[Timer]
OnBootSec=60
OnUnitActiveSec=60
AccuracySec=5s
[Install]
WantedBy=timers.target
UNIT
  systemctl daemon-reload
  systemctl enable --now tat-heartbeat.timer
  say "heartbeat timer active"
else
  say "heartbeat NOT installed (needs R2 secrets)"
fi

say "done. next:  fleet.sh deploy <box>"
