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
# host-side python deps: yaml for fleet.yml, boto3 for the heartbeat's R2 PUT.
# The heartbeat runs on the HOST (not in a lane container) on purpose -- it has
# to keep reporting when every container is dead, which is exactly the case it
# exists to make visible.
python3 -c 'import yaml'  2>/dev/null || apt-get install -y -qq python3-yaml  >/dev/null
python3 -c 'import boto3' 2>/dev/null || apt-get install -y -qq python3-boto3 >/dev/null

# --- 2. the repo, on main, nothing else ------------------------------------
# FLEET GIT RULE: a box tracks origin/main and only origin/main. No box-only
# branches -- work that exists only on a box is invisible to every other box
# and to the next agent, and that is how a fleet silently forks.
# `-d .git` is wrong: a git WORKTREE (and a submodule) has .git as a FILE
# pointing at the real gitdir. box1 is exactly that, so the naive test tried to
# re-clone over a live checkout. Ask git, don't guess at the layout.
if ! git -C "$DIR" rev-parse --git-dir >/dev/null 2>&1; then
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
# Never clobber work that only exists here. The fleet rule says a box holds no
# unique state, but "reset --hard over whatever is there" is how you find out
# the rule was violated by losing the evidence. Refuse and let drift report it.
if [ -n "$(git status --porcelain)" ]; then
  say "REFUSING to reset: working tree is dirty. Land it on main first:"
  git status --short | sed 's/^/    /'
else
  git checkout -q main 2>/dev/null || git checkout -q -B main origin/main
  git reset -q --hard origin/main
fi
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
# Installed UNCONDITIONALLY. It used to be gated on all secrets being present,
# but the documented flow delivers secrets AFTER provision (provision -> setenv
# -> deploy), so a brand-new box got no heartbeat and was invisible on /fleet/
# exactly when you most want to watch it. The unit is Type=oneshot and re-reads
# EnvironmentFile every run, so it simply fails until the secrets land and then
# starts succeeding on its own -- no second provision needed.
if true; then
  say "installing heartbeat timer"
  install -m 0755 "$DIR/scripts/heartbeat.sh" /usr/local/bin/tat-heartbeat.sh
  # The heartbeat keys on the FLEET name (box1/box2/...), not the provider
  # hostname -- fleet.yml is the vocabulary everything else uses, and a
  # heartbeat filed under "srv1856364" is one more thing to translate.
  BOXNAME="$(python3 "$DIR/scripts/box_name.py")"
  say "heartbeat identity: $BOXNAME"
  cat >/etc/systemd/system/tat-heartbeat.service <<UNIT
[Unit]
Description=TAT fleet heartbeat (publishes box health to R2)
[Service]
Type=oneshot
Environment=TAT_BOX_NAME=${BOXNAME}
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
  if [ ${#MISSING[@]} -gt 0 ]; then
    say "  (it will fail until the secrets land -- that is expected, and the"
    say "   box will start reporting on its own once fleet.sh setenv runs)"
  fi
fi

say "done. next:  fleet.sh deploy <box>"
