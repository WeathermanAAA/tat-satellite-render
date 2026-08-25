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
# The checkout this heartbeat reports on and imports s1_ingest from. Only the
# shell-level test overrides it (a sandbox with a stub s1_ingest); the systemd
# unit sets nothing, so production is /root/tsr-s2 as before.
TSR_DIR="${TAT_TSR_DIR:-/root/tsr-s2}"
export TSR_DIR
# The r2-breaker Worker's status endpoint (PHASE0 spec, 2026-08-25). This is
# the ONLY box->Worker traffic there is: one GET per box per minute. Every
# lane reads the CDN mirror this script publishes, never the Worker.
BREAKER_URL="${TAT_BREAKER_STATUS_URL:-https://triple-a-tropics.com/r2-breaker/status}"

cd "$TSR_DIR" 2>/dev/null || exit 0

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

# --- R2 breaker: read the Worker's status once; used twice below ------------
# (a) breaker_age_s / breaker_mode ride in this heartbeat so /fleet/ and the
#     liveness workflow can see the breaker through the box's eyes;
# (b) after the heartbeat PUT succeeds, the document is mirrored to
#     fleet/breaker.json on R2 -- the CDN copy every writer's kill switch
#     polls (tat_killswitch.py) instead of hitting the free-plan Worker.
# On a failed fetch or anything that is not a v1 status document we write
# NOTHING to the mirror (the last good copy must never be clobbered with
# garbage) and report breaker_age_s as null here. An unparseable mirror
# fails OPEN on the boxes, so a stale-but-valid copy is strictly safer than
# an honest-looking blank.
breaker_raw=$(curl -fsS -m 5 "$BREAKER_URL" 2>/dev/null || true)
IFS=$'\t' read -r breaker_ok breaker_age_s breaker_mode_json < <(
  HB_BREAKER="$breaker_raw" python3 - <<'PYB'
import datetime as dt, json, os
try:
    d = json.loads(os.environ.get("HB_BREAKER", ""))
    if not isinstance(d, dict) or d.get("v") != 1:
        raise ValueError("not a v1 breaker status")
    wts = str(d["worker_ts"]).strip()
    if wts[-1] in "Zz":
        wts = wts[:-1] + "+00:00"
    w = dt.datetime.fromisoformat(wts)
    if w.tzinfo is None:
        w = w.replace(tzinfo=dt.timezone.utc)
    age = int((dt.datetime.now(dt.timezone.utc) - w).total_seconds())
    mode = d.get("mode")
    print("1", age, json.dumps(mode if isinstance(mode, str) else None), sep="\t")
except Exception:
    print("0", "null", "null", sep="\t")
PYB
)
breaker_ok=${breaker_ok:-0}; breaker_age_s=${breaker_age_s:-null}; breaker_mode_json=${breaker_mode_json:-null}

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
json=$(cat <<JSON
{"box":"${BOX}","ts":"${ts}","sha":"${sha}","branch":"${branch}","dirty":${dirty:-0},
 "cores":${cores},"load1":${load1},"load5":${load5},"load15":${load15},
 "load_per_core":$(awk -v l="$load1" -v c="$cores" 'BEGIN{printf "%.2f", l/c}'),
 "mem_total_mb":$((memtotal/1024)),"mem_avail_mb":$((memavail/1024)),
 "disk_pct":${disk_pct:-0},"oom_kills":${ooms:-0},
 "lanes_up":"${lanes_up}","lanes_exited":"${lanes_dead}",
 "breaker_age_s":${breaker_age_s},"breaker_mode":${breaker_mode_json}}
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
import json, os
try:
    import yaml
    f = yaml.safe_load(open(os.path.join(os.environ["TSR_DIR"], "fleet.yml")))
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
sys.path.insert(0, os.environ["TSR_DIR"])
try:
    import s1_ingest
    s1_ingest.R2().put_bytes("fleet/index.json", os.environ["HB_ROSTER"].encode(),
                             "application/json", "max-age=60")
except Exception as e:
    print("roster put failed:", e, file=sys.stderr)
PYR2
fi

HB_JSON="$json" python3 - "$KEY" <<'PY' || exit 1
import os, sys
sys.path.insert(0, os.environ["TSR_DIR"])
try:
    import s1_ingest
    ok = s1_ingest.R2().put_bytes(sys.argv[1], os.environ["HB_JSON"].encode(),
                                  "application/json", "max-age=60")
    if not ok:
        raise RuntimeError("put_bytes returned False")
except Exception as e:
    print("heartbeat put failed:", e, file=sys.stderr)
    sys.exit(1)
PY

# --- R2 breaker mirror: only after the heartbeat landed, only a valid doc ---
# fleet/breaker.json is what tat_killswitch polls on every box (via the CDN).
# It is written under the fleet/ prefix so the kill switch's allowlist lets it
# through even while writes are stopped -- otherwise a tripped breaker could
# never publish its own reset. Adds mirrored_at/mirrored_by so a reader can
# tell a fresh mirror from a fresh Worker tick. Failure here is logged, never
# fatal: the heartbeat already succeeded and its exit code says so.
if [ "$breaker_ok" = "1" ]; then
  HB_BREAKER="$breaker_raw" HB_BOX="$BOX" python3 - <<'PYM'
import datetime as dt, json, os, sys
sys.path.insert(0, os.environ["TSR_DIR"])
try:
    d = json.loads(os.environ["HB_BREAKER"])
    if not isinstance(d, dict) or d.get("v") != 1:
        raise ValueError("not a v1 breaker status")
    d["mirrored_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    d["mirrored_by"] = os.environ["HB_BOX"]
    import s1_ingest
    ok = s1_ingest.R2().put_bytes("fleet/breaker.json",
                                  json.dumps(d, separators=(",", ":")).encode(),
                                  "application/json", "max-age=30")
    if not ok:
        raise RuntimeError("put_bytes returned False")
except Exception as e:
    print("breaker mirror put failed:", e, file=sys.stderr)
PYM
fi
