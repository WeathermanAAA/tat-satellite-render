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

# --- CPU steal + NIC throughput sample -----------------------------------
# WHY (2026-09-02 incident): box1 sat 35 h under a host-side ingress shaper
# (eth0 rx pinned at exactly 1280 KiB/s) and then 33+ h under Hostinger's
# automated CPU limiter (steal 94-96 % on every vCPU), and /fleet/ showed
# "load 16-49" for both -- indistinguishable from a busy box, because the
# heartbeat read /proc/loadavg only. Load under a cap is the run queue
# backing up behind the hypervisor; steal is the discriminator. A NIC pinned
# at a constant is the shaper's signature. Both are cheap to read and cannot
# be inferred from anything else the heartbeat publishes.
#
# Two reads of /proc/stat and /proc/net/dev, SAMPLE_S apart, and the deltas
# become percentages of the box's total jiffies and KiB/s on the default
# route's interface. Kept as a function that takes FILE PATHS so the
# arithmetic is unit-testable against canned snapshots (`--sample-from`).
HB_SAMPLE_S="${HB_SAMPLE_S:-3}"

hb_default_iface() {
  ip -o route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}'
}

# hb_cpu_net_json STAT1 STAT2 DEV1 DEV2 IFACE SECONDS
# Prints a JSON fragment (no braces): "steal_pct":..,"cpu_busy_pct":..,...
# Any unreadable input yields nulls, never a malformed number.
hb_cpu_net_json() {
  local stat1="$1" stat2="$2" dev1="$3" dev2="$4" iface="$5" secs="$6"
  awk -v iface="$iface" -v secs="$secs" '
    function cpuline(f, arr,   line, n, i, parts) {
      if ((getline line < f) <= 0) { close(f); return 0 }
      close(f)
      n = split(line, parts, " ")
      if (parts[1] != "cpu" || n < 9) return 0
      # user nice system idle iowait irq softirq steal
      for (i = 2; i <= 9; i++) arr[i-1] = parts[i] + 0
      return 1
    }
    function devline(f, arr,   line, name, rest, p, n) {
      while ((getline line < f) > 0) {
        gsub(/^ +/, "", line)
        if (index(line, iface ":") == 1) {
          rest = substr(line, length(iface) + 2)
          n = split(rest, p, " ")
          if (n >= 9) { arr[1] = p[1] + 0; arr[2] = p[9] + 0; close(f); return 1 }
        }
      }
      close(f); return 0
    }
    BEGIN {
      okc = cpuline(ARGV[1], a) && cpuline(ARGV[2], b)
      okn = (iface != "") && devline(ARGV[3], r1) && devline(ARGV[4], r2)
      if (okc) {
        tot = 0
        for (i = 1; i <= 8; i++) { d[i] = b[i] - a[i]; if (d[i] < 0) d[i] = 0; tot += d[i] }
        if (tot > 0) {
          busy = 100 * (d[1] + d[2] + d[3] + d[6] + d[7]) / tot
          idle = 100 * (d[4] + d[5]) / tot
          steal = 100 * d[8] / tot
          printf "\"steal_pct\":%.1f,\"cpu_busy_pct\":%.1f,\"cpu_idle_pct\":%.1f,", steal, busy, idle
        } else okc = 0
      }
      if (!okc) printf "\"steal_pct\":null,\"cpu_busy_pct\":null,\"cpu_idle_pct\":null,"
      if (okn && secs + 0 > 0) {
        rx = (r2[1] - r1[1]) / 1024 / secs; tx = (r2[2] - r1[2]) / 1024 / secs
        if (rx < 0) rx = 0; if (tx < 0) tx = 0
        printf "\"net_if\":\"%s\",\"rx_kib_s\":%.1f,\"tx_kib_s\":%.1f,", iface, rx, tx
      } else printf "\"net_if\":null,\"rx_kib_s\":null,\"tx_kib_s\":null,"
      printf "\"sample_s\":%s", (secs + 0 > 0) ? secs + 0 : "null"
      exit 0
    }' "$stat1" "$stat2" "$dev1" "$dev2"
}

# Test hook: `heartbeat.sh --sample-from STAT1 STAT2 DEV1 DEV2 IFACE SECS`
# prints the fragment and exits without touching R2, docker, or git.
if [ "${1:-}" = "--sample-from" ]; then
  shift
  if [ "$#" -ne 6 ]; then
    echo "usage: heartbeat.sh --sample-from STAT1 STAT2 DEV1 DEV2 IFACE SECONDS" >&2
    exit 2
  fi
  hb_cpu_net_json "$@"; echo
  exit 0
fi

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

# steal + NIC: two snapshots HB_SAMPLE_S apart (see the header). Taken AFTER
# the docker/git reads above so their own CPU does not land in the window.
# If mktemp fails there is no scratch dir: publish nulls for the sample and
# never fall back to a shared path (a fallback of /tmp followed by rm -rf is
# exactly the kind of line that deletes /tmp on a bad day).
HB_NULLS='"steal_pct":null,"cpu_busy_pct":null,"cpu_idle_pct":null,"net_if":null,"rx_kib_s":null,"tx_kib_s":null,"sample_s":null'
cpu_net="$HB_NULLS"
if HB_TMP=$(mktemp -d "${TMPDIR:-/tmp}/tat-hb.XXXXXX" 2>/dev/null) && [ -d "$HB_TMP" ]; then
  HB_IF=$(hb_default_iface)
  # An interface name may legally contain " or \ (only / and whitespace are
  # forbidden); either would break the JSON and the validator would drop the
  # whole heartbeat. Publish net nulls rather than a dead-looking box.
  case "$HB_IF" in *[\"\\]*) HB_IF="" ;; esac
  # Divide the byte deltas by the MEASURED window, not the nominal sleep: under
  # a CPU cap every fork between the two snapshots is slow, and dividing by 3 s
  # would turn "pinned at exactly 1280 KiB/s" into jitter. $EPOCHREALTIME is a
  # bash builtin (no fork). The CPU percentages are ratios and need no window.
  hb_t0=$EPOCHREALTIME
  head -1 /proc/stat >"$HB_TMP/s1" 2>/dev/null; cat /proc/net/dev >"$HB_TMP/d1" 2>/dev/null
  sleep "$HB_SAMPLE_S"
  head -1 /proc/stat >"$HB_TMP/s2" 2>/dev/null; cat /proc/net/dev >"$HB_TMP/d2" 2>/dev/null
  hb_t1=$EPOCHREALTIME
  hb_secs=$(awk -v a="$hb_t0" -v b="$hb_t1" 'BEGIN{d=b-a; if (d<=0) d=0; printf "%.2f", d}')
  cpu_net=$(hb_cpu_net_json "$HB_TMP/s1" "$HB_TMP/s2" "$HB_TMP/d1" "$HB_TMP/d2" "$HB_IF" "$hb_secs")
  rm -rf -- "$HB_TMP" 2>/dev/null
  # awk absent or crashed -> an empty fragment would leave a bare comma in the
  # JSON and the validator would refuse the whole heartbeat (box looks dead).
  case "$cpu_net" in *'"steal_pct":'*) ;; *) cpu_net="$HB_NULLS" ;; esac
fi

# run_s is this run's wall time. The timer is OnUnitActiveSec=60 on a oneshot
# unit, so runs never overlap: when a run outlives 60 s (65 s was measured
# under the 2026-09 CPU cap) the period simply stretches to the run time.
# No lock is needed here, and a long run_s is itself a starvation signal.
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
json=$(cat <<JSON
{"box":"${BOX}","ts":"${ts}","sha":"${sha}","branch":"${branch}","dirty":${dirty:-0},
 "cores":${cores},"load1":${load1},"load5":${load5},"load15":${load15},
 "load_per_core":$(awk -v l="$load1" -v c="$cores" 'BEGIN{printf "%.2f", l/c}'),
 "mem_total_mb":$((memtotal/1024)),"mem_avail_mb":$((memavail/1024)),
 "disk_pct":${disk_pct:-0},"oom_kills":${ooms:-0},
 ${cpu_net},"run_s":${SECONDS},
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
