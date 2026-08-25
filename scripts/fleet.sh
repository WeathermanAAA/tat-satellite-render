#!/usr/bin/env bash
# fleet.sh -- one entry point for every multi-box operation.
#
#   fleet.sh status              per-box: git sha, load, RAM, lane health
#   fleet.sh drift               FAIL if any box is off main / dirty / ahead
#   fleet.sh provision <box>     idempotent full build-out of a box
#   fleet.sh deploy <box|all>    pull main, rebuild image, recreate its lanes
#   fleet.sh setenv <box|all> K=V   write a secret to .env and restart lanes
#   fleet.sh lanes <box>         what fleet.yml says this box should run
#   fleet.sh writes off|on [box|all]   R2 WRITE kill switch: set TAT_R2_WRITES
#                                and recreate every write lane, same image; verifies
#                                env + guard presence per container (~30-60 s)
#   fleet.sh breaker status|arm|disarm|trip|reset   the r2-breaker Worker
#                                (admin verbs need TAT_BREAKER_ADMIN_KEY)
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

# -n is load-bearing: without it ssh swallows the caller's stdin, and a
# `while read` loop over lanes silently deploys only its first entry.
on() { local host="$1"; shift; ssh -n "${SSH_OPTS[@]}" "$host" "$@"; }

cmd_lanes() {
  local box="${1:?box}"
  _lanes_of "$box" | while IFS=$'\t' read -r proj comp; do echo "  $proj  ($comp)"; done
}

cmd_status() {
  # `|| echo UNREACHABLE` is load-bearing: ssh exits 255 for a down box OR a
  # bad key, and under `set -e` that aborted the whole run -- so one dead box
  # blinded you to every box after it, which is the opposite of what a fleet
  # status is for.
  while IFS=$'\t' read -r name host; do
    echo "=== $name  $host"
    on "$host" '
      cd /root/tsr-s2 2>/dev/null && \
        echo "  repo   $(git rev-parse --short HEAD) on $(git branch --show-current) $(git status --porcelain | head -1 | sed "s/^/DIRTY:/")" || echo "  repo   ABSENT"
      echo "  load  $(uptime | sed "s/.*load average: //")   mem $(free -g | awk "/Mem:/{print \$3\"/\"\$2\"GB used, \"\$7\"GB avail\"}")"
      n=$(docker ps --format "{{.Names}}" | grep -c "^tat-s2-" || true)
      echo "  lanes  $n running: $(docker ps --format "{{.Label \"com.docker.compose.project\"}}" | grep "^tat-s2-" | sort -u | tr "\n" " ")"
    ' 2>&1 | sed 's/^/ /' || echo "  UNREACHABLE (ssh failed)"
  done < <(_boxes)
}

# DRIFT: the fleet git rule, enforced. Any box not exactly on origin/main and
# clean is an incident -- a box-only commit is invisible to every other box and
# to the next agent.
cmd_drift() {
  # Two things this has to get right to be worth running:
  #   * a box that is DOWN is drift too, not a silent skip. ssh exits 255 for
  #     both unreachable and bad-key, and a bare assignment under `set -e`
  #     aborted the whole loop -- so a dead box hid every box behind it.
  #   * it must EXIT non-zero when it finds drift, or it cannot gate anything.
  #     The old loop ran in a pipeline subshell, so a flag set inside died with
  #     the subshell; the input is redirected instead so `bad` survives.
  local bad=0
  while IFS=$'\t' read -r name host; do
    out=$(on "$host" '
      cd /root/tsr-s2 2>/dev/null || { echo "NOREPO"; exit 0; }
      git fetch -q origin main 2>/dev/null || true
      b=$(git branch --show-current)
      d=$(git status --porcelain | wc -l)
      a=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
      be=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
      echo "$b|$d|$a|$be|$(git rev-parse --short HEAD)"
    ' 2>/dev/null) || out="UNREACHABLE"
    if [ "$out" = "UNREACHABLE" ] || [ -z "$out" ]; then
      echo "DRIFT $name: UNREACHABLE (ssh failed)"; bad=1; continue
    fi
    if [ "$out" = "NOREPO" ]; then
      echo "DRIFT $name: no /root/tsr-s2"; bad=1; continue
    fi
    IFS='|' read -r branch dirty ahead behind sha <<<"$out"
    msg=""
    [ "$branch" != "main" ] && msg="$msg on '$branch' not main;"
    [ "${dirty:-0}" -gt 0 ] && msg="$msg $dirty uncommitted file(s);"
    [ "${ahead:-0}" -gt 0 ] && msg="$msg $ahead commit(s) NOT pushed to main;"
    [ "${behind:-0}" -gt 0 ] && msg="$msg $behind commit(s) behind main;"
    if [ -n "$msg" ]; then
      echo "DRIFT $name ($sha):$msg"; bad=1
    else
      echo "ok    $name ($sha) clean on main"
    fi
  done < <(_boxes)
  return "$bad"
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
  while IFS=$'\t' read -r name host; do
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
    # RECONCILE. Deploy used to only bring lanes UP, so a lane reassigned to
    # another box kept running on BOTH -- two emitters racing one product's
    # manifest geometry, which is the failure the lane headers warn about.
    # "Moving a lane is an edit plus a deploy" is only true if the box the lane
    # LEFT also converges to the map.
    local want; want="$(echo "$lanes" | cut -f1 | tr '\n' ' ')"
    on "$host" "cd /root/tsr-s2 && for p in \$(docker ps --format '{{.Label \"com.docker.compose.project\"}}' | grep '^tat-s2-' | sort -u); do
        case ' $want ' in *\" \$p \"*) ;; *) echo \"  <- stopping \$p (not assigned to $name)\"; docker compose -p \$p down >/dev/null 2>&1 || true ;; esac
      done"
  done < <(_boxes)
}

# write/replace one KEY=VALUE in a box's .env (mode 600). The value never
# echoes; the caller reports success/failure per box.
_setenv_on() {
  local host="$1" kv="$2" key="${2%%=*}"
  on "$host" "touch /root/tsr-s2/.env && chmod 600 /root/tsr-s2/.env && sed -i '/^${key}=/d' /root/tsr-s2/.env && printf '%s\n' '$kv' >> /root/tsr-s2/.env"
}

# Secret propagation: write/replace a key in every target box's .env, then
# restart that box's lanes so the new value is actually in the containers.
cmd_setenv() {
  local target="${1:?box|all}" kv="${2:?KEY=VALUE}"
  local key="${kv%%=*}"
  # A PARTIALLY applied credential is the worst outcome here -- some boxes on
  # the new secret, some on the old, and no signal. So each box reports, a
  # failure does not abort the rest, and the command exits non-zero if any box
  # was missed.
  local bad=0 n=0
  while IFS=$'\t' read -r name host; do
    [ "$target" != "all" ] && [ "$target" != "$name" ] && continue
    n=$((n + 1))
    if _setenv_on "$host" "$kv"; then
      echo "  $name: $key set"
    else
      echo "  $name: FAILED to set $key (ssh/write error)"; bad=1
    fi
  done < <(_boxes)
  [ "$n" -eq 0 ] && { echo "[setenv] no box matched '$target'"; return 1; }
  if [ "$bad" -ne 0 ]; then
    echo "[setenv] INCOMPLETE -- $key is not uniform across the fleet. Re-run for the failed box before deploying."
  else
    echo "[setenv] restart lanes to pick it up:  fleet.sh deploy $target"
  fi
  return "$bad"
}

# ---- R2 write kill switch (PHASE0 spec, 2026-08-25) -------------------------
# The 30-second manual path: put TAT_R2_WRITES=0|1 into every .env the write
# lanes load, then bring each write container back up from its own compose
# labels with the SAME image (--no-build, --no-deps, --force-recreate), so
# every process re-imports tat_killswitch and reads the new env. The box half
# is scripts/box_writes.sh (piped over ssh like provision_box.sh); read its
# header for why a plain `docker restart` cannot do this (compose bakes
# env_file in at create time) and how the result is VERIFIED per container:
# the env value AND a probe that runs tat_killswitch inside the container
# (a lane whose image predates the file would take the env and keep
# writing; that reports `guard: ABSENT` and fails the box).
# Derived from `docker ps` with name filters: an absent container is skipped,
# never an error, so the same command is right on an emit-only box and box1.
#
# WRITE_RE is a positive allowlist of the lanes that PUT to R2; NEVER_RE is a
# second gate over everything that SERVES (caddy, render, meso-render*,
# s1-render, radar-render, centerfix/objfix) plus prune. Recreating a serving
# stack to stop a bill would trade a cost incident for an outage, so a name
# has to pass BOTH. tests/test_killswitch.py pins both lists against the
# container inventory in the spec.
WRITE_RE='^(tat-s2-[a-z0-9-]+-emit-cron-[0-9]+|tat-render-(floater|intensity|guidance)-poller-[0-9]+|tat-s1-s1-ingest(-[a-z0-9]+)?-[0-9]+|tat-satellite-render-meso-poller-[0-9]+|tat-overlays-[a-z0-9]+-poller-[0-9]+)$'
NEVER_RE='^(tat-s2-prune-|tat-render-(caddy|render|centerfix-plot|objfix-collector|ens-watchdog)-|tat-satellite-render-meso-render|tat-s1-s1-render|tat-radar-render)'

cmd_writes() {
  local mode="${1:-}" target="${2:-all}" val
  case "$mode" in
    off) val=0 ;;
    on)  val=1 ;;
    *) echo "usage: fleet.sh writes off|on [box|all]"; return 2 ;;
  esac
  # Same discipline as setenv: every box reports, one failure never aborts
  # the rest, and the exit code says whether the fleet is uniform.
  local bad=0 n=0
  while IFS=$'\t' read -r name host; do
    [ "$target" != "all" ] && [ "$target" != "$name" ] && continue
    n=$((n + 1))
    echo "=== writes $mode  $name"
    # ssh's stdin IS the script here, so it cannot swallow the box list (the
    # -n trap documented on `on`); the regexes travel as env assignments.
    if ssh "${SSH_OPTS[@]}" "$host" "VAL=$val WRITE_RE='$WRITE_RE' NEVER_RE='$NEVER_RE' bash -s" < "$REPO_DIR/scripts/box_writes.sh"; then
      echo "  $name: writes $mode (env + guard verified in every write container)"
    else
      echo "  $name: INCOMPLETE -- see the lines above; a lane may still hold the old setting, or lacks the guard and still writes"; bad=1
    fi
  done < <(_boxes)
  [ "$n" -eq 0 ] && { echo "[writes] no box matched '$target'"; return 1; }
  if [ "$bad" -ne 0 ]; then
    echo "[writes] INCOMPLETE -- re-run for the failed box; check with: fleet.sh status"
  else
    echo "[writes] R2 writes $mode on $target (heartbeats keep publishing: fleet/* is allowlisted; deletes/prune untouched)"
  fi
  return "$bad"
}

# ---- the r2-breaker Worker --------------------------------------------------
# `status` is public. arm/disarm/trip/reset POST with X-Admin-Key, read from
# the caller's env (TAT_BREAKER_ADMIN_KEY) or else from box1's .env (put it
# there with: fleet.sh setenv box1 TAT_BREAKER_ADMIN_KEY=<key>). Never printed.
# Remember: `reset` is the ONLY path that re-enables writes; `disarm` just
# drops to alert-only and does NOT re-enable them.
_breaker_call() {   # method path [admin-key] -> prints the body; rc 0 iff 2xx
  local method="$1" path="$2" key="${3:-}" base="${TAT_BREAKER_URL:-https://triple-a-tropics.com/r2-breaker}"
  local out code
  if [ -n "$key" ]; then
    out="$(curl -sS -m 15 -X "$method" -H "X-Admin-Key: $key" -w $'\n%{http_code}' "$base$path")" || { echo "[breaker] curl failed for $path"; return 1; }
  else
    out="$(curl -sS -m 15 -X "$method" -w $'\n%{http_code}' "$base$path")" || { echo "[breaker] curl failed for $path"; return 1; }
  fi
  code="${out##*$'\n'}"
  printf '%s\n' "${out%$'\n'*}"
  case "$code" in 2*) return 0 ;; *) echo "[breaker] HTTP $code from $path"; return 1 ;; esac
}

cmd_breaker() {
  local verb="${1:-}"
  case "$verb" in
    status) _breaker_call GET /status ;;
    arm|disarm|trip|reset)
      local key="${TAT_BREAKER_ADMIN_KEY:-}"
      if [ -z "$key" ]; then
        key="$(on "$(_host_of box1)" "sed -n 's/^TAT_BREAKER_ADMIN_KEY=//p' /root/tsr-s2/.env | head -1" 2>/dev/null || true)"
      fi
      if [ -z "$key" ]; then
        echo "[breaker] no admin key: export TAT_BREAKER_ADMIN_KEY, or  fleet.sh setenv box1 TAT_BREAKER_ADMIN_KEY=<key>"
        return 1
      fi
      [ "$verb" = "trip" ] && echo "[breaker] TRIP: writes_enabled=false in ANY mode (alert too); every box stops R2 writes within ~2 min (mirror + poll). Only 'reset' re-enables ('disarm' does not)."
      _breaker_call POST "/$verb" "$key" ;;
    *) echo "usage: fleet.sh breaker status|arm|disarm|trip|reset"; return 2 ;;
  esac
}

case "${1:-}" in
  status)    shift; cmd_status "$@" ;;
  drift)     shift; cmd_drift "$@" ;;
  provision) shift; cmd_provision "$@" ;;
  deploy)    shift; cmd_deploy "$@" ;;
  setenv)    shift; cmd_setenv "$@" ;;
  lanes)     shift; cmd_lanes "$@" ;;
  writes)    shift; cmd_writes "$@" ;;
  breaker)   shift; cmd_breaker "$@" ;;
  *) sed -n '2,/^set -euo pipefail/{/^set -euo pipefail/!p}' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
