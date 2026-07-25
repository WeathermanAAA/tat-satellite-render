#!/usr/bin/env bash
# heartbeat.sh -- publish this box's health to R2 once a minute.
#
# WHY: a box that dies quietly is worse than one that dies loudly. Its lanes
# simply stop publishing and the site just looks a bit stale -- exactly like a
# slow satellite or a quiet basin. The heartbeat separates "no new data" from
# "no box", by making the BOX assert liveness independently of whether its
# lanes happened to produce a frame.
#
# Contract: writes fleet/<box>.json to the R2 bucket with a 60 s cache. The
# object's own age is the signal -- a stale file means a dead box even if its
# contents look healthy, which is why every consumer must read `ts` and
# compare it to now rather than trusting `ok`.
#
# Run by the systemd timer that provision_box.sh installs. Reads R2_* from
# /root/tsr-s2/.env via the unit's EnvironmentFile.
set -uo pipefail

BOX="${TAT_BOX_NAME:-$(hostname)}"
PREFIX="${R2_HEARTBEAT_PREFIX:-fleet}"
KEY="${PREFIX}/${BOX}.json"

cd /root/tsr-s2 2>/dev/null || exit 0

read -r load1 load5 load15 _ < /proc/loadavg
cores=$(nproc)
read -r memtotal memavail < <(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print t, a}' /proc/meminfo)
disk_pct=$(df --output=pcent / | tail -1 | tr -dc '0-9')
sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
branch=$(git branch --show-current 2>/dev/null || echo unknown)
dirty=$(git status --porcelain 2>/dev/null | wc -l)

# lane inventory: what is actually up right now, and anything that exited
lanes_up=$(docker ps --filter "name=^/tat-s2-" --format '{{.Label "com.docker.compose.project"}}' 2>/dev/null | sort -u | paste -sd, -)
lanes_dead=$(docker ps -a --filter "name=^/tat-s2-" --filter "status=exited" --format '{{.Label "com.docker.compose.project"}}' 2>/dev/null | sort -u | paste -sd, -)
# OOM kills since boot are the failure mode that matters most on an emit box
ooms=$(dmesg -T 2>/dev/null | grep -c "Out of memory: Killed process" || echo 0)

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
json=$(cat <<JSON
{"box":"${BOX}","ts":"${ts}","sha":"${sha}","branch":"${branch}","dirty":${dirty:-0},
 "cores":${cores},"load1":${load1},"load5":${load5},"load15":${load15},
 "load_per_core":$(awk -v l="$load1" -v c="$cores" 'BEGIN{printf "%.2f", l/c}'),
 "mem_total_mb":$((memtotal/1024)),"mem_avail_mb":$((memavail/1024)),
 "disk_pct":${disk_pct:-0},"oom_kills":${ooms:-0},
 "lanes_up":"${lanes_up}","lanes_exited":"${lanes_dead}"}
JSON
)

HB_JSON="$json" python3 - "$KEY" <<'PY'
import os, sys
sys.path.insert(0, "/root/tsr-s2")
try:
    import s1_ingest
    s1_ingest.R2().put_bytes(sys.argv[1], os.environ["HB_JSON"].encode(),
                             "application/json", "max-age=60")
except Exception as e:
    print("heartbeat put failed:", e, file=sys.stderr)
    sys.exit(1)
PY
