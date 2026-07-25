#!/usr/bin/env bash
# fleet.sh -- one entry point for every multi-box operation.
#
#   fleet.sh status              per-box: git sha, load, RAM, lane health
#   fleet.sh drift               FAIL if any box is off main / dirty / ahead
#   fleet.sh provision <box>     idempotent full build-out of a box
#   fleet.sh deploy <box|all>    pull main, rebuild image, recreate its lanes
#   fleet.sh setenv <box|all> K=V   write a secret to .env and restart lanes
#   fleet.sh lanes <box>         what fleet.yml says this box should run
#
# Reads fleet.yml as the assignment map -- never hardcode a box or a lane here.
# Run from a checkout of tat-satellite-render (Codespace or any box).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLEET="$REPO_DIR/fleet.yml"
SSH_KEY="${TAT_BOX_KEY:-$HOME/.ssh/tat_box}"
SSH_OPTS=(-i "$SSH_KEY" -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)

py() { python3 -c "$1" "${@:2}"; }

_boxes() {   # -> name<TAB>host
  py '
import sys,yaml
f=yaml.safe_load(open(sys.argv[1]))
for b in f["boxes"]: print(b["name"]+"\t"+b["host"])
' "$FLEET"
}

_host_of() {
  py '
import sys,yaml
f=yaml.safe_load(open(sys.argv[1]))
for b in f["boxes"]:
    if b["name"]==sys.argv[2]: print(b["host"]); break
else: sys.exit("unknown box: "+sys.argv[2])
' "$FLEET" "$1"
}

# lanes assigned to a box, as "project<TAB>composefile"
_lanes_of() {
  py '
import sys,yaml
f=yaml.safe_load(open(sys.argv[1]))
for l in f["lanes"]:
    if l["box"]==sys.argv[2]: print(l["project"]+"\t"+l["compose"])
' "$FLEET" "$1"
}

_env_keys() {
  py 'import sys,yaml;print("\n".join(yaml.safe_load(open(sys.argv[1]))["env_keys"]))' "$FLEET"
}

on() { local host="$1"; shift; ssh "${SSH_OPTS[@]}" "$host" "$@"; }

cmd_lanes() {
  local box="${1:?box}"
  _lanes_of "$box" | while IFS=$'\t' read -r proj comp; do echo "  $proj  ($comp)"; done
}

cmd_status() {
  _boxes | while IFS=$'\t' read -r name host; do
    echo "=== $name  $host"
    on "$host" '
      cd /root/tsr-s2 2>/dev/null && \
        echo "  repo   $(git rev-parse --short HEAD) on $(git branch --show-current) $(git status --porcelain | head -1 | sed "s/^/DIRTY:/")" || echo "  repo   ABSENT"
      echo "  load  $(uptime | sed "s/.*load average: //")   mem $(free -g | awk "/Mem:/{print \$3\"/\"\$2\"GB used, \"\$7\"GB avail\"}")"
      n=$(docker ps --format "{{.Names}}" | grep -c "^tat-s2-" || true)
      echo "  lanes  $n running: $(docker ps --format "{{.Label \"com.docker.compose.project\"}}" | grep "^tat-s2-" | sort -u | tr "\n" " ")"
    ' 2>&1 | sed 's/^/ /'
  done
}

# DRIFT: the fleet git rule, enforced. Any box not exactly on origin/main and
# clean is an incident -- a box-only commit is invisible to every other box and
# to the next agent.
cmd_drift() {
  local bad=0
  _boxes | while IFS=$'\t' read -r name host; do
    out=$(on "$host" '
      cd /root/tsr-s2 || { echo "NOREPO"; exit 0; }
      git fetch -q origin main 2>/dev/null || true
      b=$(git branch --show-current)
      d=$(git status --porcelain | wc -l)
      a=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
      be=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
      echo "$b|$d|$a|$be|$(git rev-parse --short HEAD)"
    ' 2>/dev/null)
    IFS='|' read -r branch dirty ahead behind sha <<<"$out"
    if [ "$out" = "NOREPO" ]; then
      echo "DRIFT $name: no /root/tsr-s2"; continue
    fi
    msg=""
    [ "$branch" != "main" ] && msg="$msg on '$branch' not main;"
    [ "${dirty:-0}" -gt 0 ] && msg="$msg $dirty uncommitted file(s);"
    [ "${ahead:-0}" -gt 0 ] && msg="$msg $ahead commit(s) NOT pushed to main;"
    [ "${behind:-0}" -gt 0 ] && msg="$msg $behind commit(s) behind main;"
    if [ -n "$msg" ]; then echo "DRIFT $name ($sha):$msg"; else echo "ok    $name ($sha) clean on main"; fi
  done
}

cmd_provision() {
  local box="${1:?box}" host; host="$(_host_of "$box")"
  echo "[provision] $box $host"
  # the base build-out is a script in this repo so box N+1 is a step, not a build
  ssh "${SSH_OPTS[@]}" "$host" 'bash -s' < "$REPO_DIR/scripts/provision_box.sh"
  echo "[provision] $box: now run  fleet.sh setenv $box KEY=VALUE  for each missing secret, then deploy"
}

cmd_deploy() {
  local target="${1:?box|all}"
  _boxes | while IFS=$'\t' read -r name host; do
    [ "$target" != "all" ] && [ "$target" != "$name" ] && continue
    echo "=== deploy $name"
    local lanes; lanes="$(_lanes_of "$name")"
    if [ -z "$lanes" ]; then echo "  (no lanes assigned)"; continue; fi
    on "$host" 'cd /root/tsr-s2 && git fetch -q origin main && git checkout -q main && git reset -q --hard origin/main && echo "  at $(git rev-parse --short HEAD)" && docker compose -f docker-compose.s2.yml build -q emit-cron && echo "  image built"'
    while IFS=$'\t' read -r proj comp; do
      [ -z "$proj" ] && continue
      echo "  -> $proj"
      on "$host" "cd /root/tsr-s2 && docker compose -p $proj -f docker-compose.s2.yml -f $comp --profile cron up -d --no-build emit-cron 2>&1 | tail -1"
    done <<<"$lanes"
  done
}

# Secret propagation: write/replace a key in every target box's .env, then
# restart that box's lanes so the new value is actually in the containers.
cmd_setenv() {
  local target="${1:?box|all}" kv="${2:?KEY=VALUE}"
  local key="${kv%%=*}"
  _boxes | while IFS=$'\t' read -r name host; do
    [ "$target" != "all" ] && [ "$target" != "$name" ] && continue
    on "$host" "touch /root/tsr-s2/.env && chmod 600 /root/tsr-s2/.env && sed -i '/^${key}=/d' /root/tsr-s2/.env && printf '%s\n' '$kv' >> /root/tsr-s2/.env && echo '  $name: $key set'"
  done
  echo "[setenv] restart lanes to pick it up:  fleet.sh deploy $target"
}

case "${1:-}" in
  status)    shift; cmd_status "$@" ;;
  drift)     shift; cmd_drift "$@" ;;
  provision) shift; cmd_provision "$@" ;;
  deploy)    shift; cmd_deploy "$@" ;;
  setenv)    shift; cmd_setenv "$@" ;;
  lanes)     shift; cmd_lanes "$@" ;;
  *) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
