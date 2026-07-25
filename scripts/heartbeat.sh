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
# `grep -c` prints its count AND exits 1 when the count is zero, so
# `|| echo 0` appends a SECOND zero and the JSON becomes "0\n0" -- which
# published malformed heartbeats that every consumer then failed to parse.
# `|| true` keeps grep's own output and nothing else.
ooms=$(dmesg -T 2>/dev/null | grep -c "Out of memory: Killed process" || true)
ooms=${ooms:-0}

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

# A malformed heartbeat is worse than a missing one: consumers show the box as
# broken-in-an-unknown-way instead of plainly stale. Validate before we ship it.
if ! printf '%s' "$json" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
  echo "heartbeat: refusing to publish malformed JSON" >&2
  printf '%s\n' "$json" >&2
  exit 1
fi

# Publish the ROSTER too, from fleet.yml. Without this the health page has to
# hardcode the box list, so box3 would heartbeat into the void until somebody
# remembered to edit a file in the OTHER repo -- a new box invisible on the
# health page is precisely the failure this whole mechanism exists to prevent.
# Every box is on the same main sha, so whoever writes it writes the same bytes.
ROSTER=$(python3 - <<'PYR'
import json
try:
    import yaml
    f = yaml.safe_load(open("/root/tsr-s2/fleet.yml"))
    def blurb(b):
        # fleet.yml roles carry the full placement rationale; the card wants a
        # line, not an essay. First sentence, and never a truncated word.
        r = " ".join(str(b.get("role", "")).split())
        head = r.split(". ")[0].rstrip(".")
        return (head + ".") if head else ""
    print(json.dumps({"boxes": [{"name": b["name"], "role": blurb(b),
                                 "tier": b.get("tier", "")}
                                for b in f["boxes"]]}, separators=(",", ":")))
except Exception:
    print("")
PYR
)
if [ -n "$ROSTER" ]; then
  HB_ROSTER="$ROSTER" python3 - <<'PYR2'
import os, sys
sys.path.insert(0, "/root/tsr-s2")
try:
    import s1_ingest
    s1_ingest.R2().put_bytes("fleet/index.json", os.environ["HB_ROSTER"].encode(),
                             "application/json", "max-age=60")
except Exception as e:
    print("roster put failed:", e, file=sys.stderr)
PYR2
fi

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
