#!/usr/bin/env bash
# box_writes.sh -- the BOX half of `fleet.sh writes off|on` (PHASE0 spec,
# 2026-08-25). Piped in over ssh like provision_box.sh
# (`VAL=0 WRITE_RE=... NEVER_RE=... bash -s < scripts/box_writes.sh`), so it
# must not depend on the repo being anywhere in particular.
#
# Env in:  VAL=0|1   WRITE_RE   NEVER_RE   [DEFAULT_ENV=/root/tsr-s2/.env]
#
# 1. Finds every RUNNING write container: `docker ps` names through the two
#    filters fleet.sh owns (WRITE_RE is a positive allowlist, NEVER_RE a
#    second gate over everything that serves, plus prune). A lane that is
#    not running is skipped, never an error.
# 2. Writes TAT_R2_WRITES=$VAL into every .env those containers' compose
#    projects load -- `env_file: .env` resolves against the project working
#    dir, and box1 runs stacks out of more than one checkout
#    (/root/tsr-s2, /root/tat-satellite-render, /root/tat-sat-s1,
#    /root/tat-overlays) -- plus DEFAULT_ENV, the fleet's canonical file.
# 3. RECREATES each write container from its own compose labels, one
#    `docker compose up` per project: same image (--no-build), nothing else
#    touched (--no-deps: floater-poller must not drag `render` along),
#    --force-recreate so the outcome does not depend on compose's config
#    hash. `docker restart` is NOT enough and is not used: compose bakes
#    env_file into the container at CREATE time, and a restart keeps the
#    old env (verified 2026-08-25: env file changed, restart, value
#    unchanged). tat_killswitch reads the env once at import, so the
#    process has to be created with the new value.
# 4. VERIFIES, per container, (a) TAT_R2_WRITES=$VAL in its env (docker
#    inspect) AND (b) that the guard is actually there and honours it: the
#    container's own python runs tat_killswitch from the container's own
#    cwd and must refuse a probe key (VAL=0) / import and allow (VAL=1).
#    Env alone is necessary, not sufficient -- every writer does
#    `try: import tat_killswitch except Exception: tat_killswitch = None`
#    and treats None as ALLOWED, so a lane whose IMAGE predates the file
#    takes the env, would report "verified", and keeps writing (found on
#    box1 2026-08-25 in all four image families: s2, render, meso, s1).
#    Exits non-zero on either failure. "Success" here means the value is
#    in the process AND the process has something that reads it.
set -uo pipefail

VAL="${VAL:?VAL=0|1}"
WRITE_RE="${WRITE_RE:?WRITE_RE (from fleet.sh)}"
NEVER_RE="${NEVER_RE:?NEVER_RE (from fleet.sh)}"
DEFAULT_ENV="${DEFAULT_ENV:-/root/tsr-s2/.env}"
# The tat-overlays pollers run TAT generators out of a sparse checkout at
# this path inside the container (docker-compose.overlays.yml x-cmd REPO);
# they import the TAT copy of tat_killswitch.py from there, not /app.
OVERLAYS_TAT_DIR="${OVERLAYS_TAT_DIR:-/work/tat}"

# guard_probe CONTAINER -> rc 0 iff tat_killswitch imports in the container
# and answers for $VAL. Uses the container's cwd (/app for tsr images) or the
# TAT checkout for overlays pollers; a missing checkout (-w fails), a missing
# file (ImportError), no python at all -> non-zero -> reported as ABSENT.
# stdout/stderr dropped: the module prints its DISABLED transition line.
guard_probe() {
  local c="$1" py wd=()
  case "$c" in tat-overlays-*) wd=(-w "$OVERLAYS_TAT_DIR") ;; esac
  if [ "$VAL" = "0" ]; then
    py='import sys, tat_killswitch as k; sys.exit(0 if not k.writes_allowed("probe/box_writes") else 1)'
  else
    py='import sys, tat_killswitch as k; sys.exit(0 if k.state().get("writes_allowed") else 1)'
  fi
  docker exec ${wd[@]+"${wd[@]}"} "$c" sh -c \
    'if command -v python3 >/dev/null 2>&1; then exec python3 -c "$1"; else exec python -c "$1"; fi' _ "$py" \
    >/dev/null 2>&1
}

label() { docker inspect -f "{{index .Config.Labels \"$2\"}}" "$1" 2>/dev/null; }

names=$(docker ps --format '{{.Names}}' | grep -E "$WRITE_RE" | grep -vE "$NEVER_RE" || true)
if [ -z "$names" ]; then
  echo "  (no write containers running here)"
  exit 0
fi

# --- 1+2. every env file a write container loads ----------------------------
rc=0
envs="$DEFAULT_ENV"
for c in $names; do
  wd=$(label "$c" com.docker.compose.project.working_dir)
  [ -n "$wd" ] && envs="$envs"$'\n'"$wd/.env"
done
while read -r f; do
  [ -z "$f" ] && continue
  d=$(dirname "$f")
  [ -d "$d" ] || { echo "  SKIP      $f (no such directory)"; continue; }
  if touch "$f" && chmod 600 "$f" && sed -i '/^TAT_R2_WRITES=/d' "$f" \
      && printf 'TAT_R2_WRITES=%s\n' "$VAL" >> "$f"; then
    echo "  env       $f  TAT_R2_WRITES=$VAL"
  else
    echo "  FAILED    $f (could not write)"; rc=1
  fi
done < <(printf '%s\n' "$envs" | sort -u)
if [ "$rc" -ne 0 ]; then
  echo "  aborting before any recreate: an env file could not be written"
  exit 1
fi

# --- 3. group the write containers by compose project ---------------------
declare -A svcs=() cname=()
for c in $names; do
  proj=$(label "$c" com.docker.compose.project)
  svc=$(label "$c" com.docker.compose.service)
  wd=$(label "$c" com.docker.compose.project.working_dir)
  cfs=$(label "$c" com.docker.compose.project.config_files)
  if [ -z "$proj" ] || [ -z "$svc" ] || [ -z "$wd" ] || [ -z "$cfs" ]; then
    echo "  SKIP      $c (not compose-managed: recreate it by hand)"; rc=1; continue
  fi
  k="$proj"$'\t'"$wd"$'\t'"$cfs"
  svcs["$k"]="${svcs[$k]:-} $svc"
  cname["$k|$svc"]="$c"
done

# --- 3+4. recreate per project, verify per container -----------------------
for k in "${!svcs[@]}"; do
  IFS=$'\t' read -r proj wd cfs <<<"$k"
  fargs=()
  IFS=, read -ra cfarr <<<"$cfs"
  for f in "${cfarr[@]}"; do fargs+=(-f "$f"); done
  # shellcheck disable=SC2086
  if (cd "$wd" && docker compose -p "$proj" "${fargs[@]}" up -d --no-build --no-deps --force-recreate ${svcs[$k]} >/dev/null 2>&1); then
    for s in ${svcs[$k]}; do
      c="${cname[$k|$s]}"
      if docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" 2>/dev/null | grep -qx "TAT_R2_WRITES=$VAL"; then
        if guard_probe "$c"; then
          echo "  recreated $c  (TAT_R2_WRITES=$VAL verified in the container; guard: present)"
        else
          echo "  recreated $c  (TAT_R2_WRITES=$VAL in its env; guard: ABSENT -- no working tat_killswitch.py in its image/checkout, so this lane STILL WRITES; rebuild it: RUNBOOK-FLEET.md 'guard presence')"; rc=1
        fi
      else
        echo "  recreated $c  but TAT_R2_WRITES=$VAL is NOT in its env -- check that project's env_file"; rc=1
      fi
    done
  else
    echo "  FAILED    $proj: docker compose up failed for:${svcs[$k]}"; rc=1
  fi
done
exit "$rc"
