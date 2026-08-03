#!/usr/bin/env bash
# s2_canary_watchdog.sh — AUTOMATIC rollback for the container-publish canary
# (2026-08-03 R2 cost incident). "A canary that needs someone awake to catch
# it isn't a canary."
#
# Watches the tat-s2-conus-fast lane after S2_CONTAINER_TILES_CONUS_FAST=1
# and reverts to per-frame publishing ON ITS OWN if any trip condition fires:
#
#   C1 index-missing : a manifest-advertised newest frame of a hinted product
#                      has no fetchable tiles.z{maxzoom}.json (2 consecutive
#                      rounds, so a mid-upload race can't false-trip)
#   C2 byte-mismatch : a ranged tile read is not byte-identical to the same
#                      window sliced from the whole block, or is not WebP
#                      (trips IMMEDIATELY — corruption gets no second round)
#   C3 stale         : any canary product's manifest.latest is older than
#                      STALE_S (default 2100 s = 7x the 5-min cadence)
#   C4 lane-broken   : lane log shows container PUT failures, or >= 6
#                      consecutive [FAIL] lines
#
# Trip action: flag -> 0 in .env, force-recreate the lane (next pass publishes
# per-tile again), DELETE canary-product container frames newer than
# REBUILD_REACH_S (the backfill window re-renders them per-tile; older
# container frames stay — they are readable by design via the sticky manifest
# hint, and only bounded-recent frames can be corrupt because C2 trips within
# one round), then exit 42. PASS: WATCH_S elapsed with no trip -> exit 0,
# flag stays on.
#
# Run on box2 as a transient unit so it survives SSH:
#   systemd-run --unit s2canary --working-directory=/root/tsr-s2 \
#     --property=Restart=on-failure --property=RestartSec=30 \
#     --property=SuccessExitStatus="0 42" \
#     /root/tsr-s2/scripts/s2_canary_watchdog.sh
# (Restart=on-failure means a watchdog crash resumes watching; the trip exit
# code 42 is mapped to SuccessExitStatus so a TRIP does not restart-loop.)
set -u
cd "$(dirname "$0")/.."

CDN="https://cdn.triple-a-tropics.com"
PRODUCTS="goes19/conus/ir goes19/conus/irbd goes19/conus/truecolor goes19/conus/c02"
LANE_PROJ="tat-s2-conus-fast"
LANE_COMPOSE="docker-compose.s2.conus-fast.yml"
LANE_CONTAINER="tat-s2-conus-fast-emit-cron-1"
ENV_FILE=".env"
FLAG="S2_CONTAINER_TILES_CONUS_FAST"
VERDICT="/root/s2_canary_verdict.txt"
INTERVAL_S="${S2_CANARY_INTERVAL_S:-120}"
WATCH_S="${S2_CANARY_WATCH_S:-43200}"          # 12 h then PASS
STALE_S="${S2_CANARY_STALE_S:-2100}"           # 35 min
REBUILD_REACH_S="${S2_CANARY_REACH_S:-2400}"   # 40 min: inside backfill reach

log() { echo "[canary $(date -u +%H:%M:%S)] $*"; }
TRIP_REASON=""   # set by lane_log_bad (same shell); check_round runs in a
                 # command-substitution SUBSHELL, so its reason is derived
                 # from the captured output + exit code instead

check_round() {  # -> writes trip reason to $TRIP_REASON, returns 0 if healthy
  TRIP_REASON=""
  python3 - "$CDN" "$STALE_S" $PRODUCTS <<'PYEOF'
import calendar, json, random, sys, time, urllib.request

cdn, stale_s, products = sys.argv[1], int(sys.argv[2]), sys.argv[3:]

UA = {"User-Agent": "tat-canary/1 (+https://triple-a-tropics.com)"}

def get(url, headers=None, timeout=25):
    # a REAL User-Agent is load-bearing: Cloudflare 403s the default
    # Python-urllib UA from the box IP, which false-tripped the first
    # canary run within seconds of arming (2026-08-03 22:59Z)
    h = dict(UA); h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()

missing = []
for p in products:
    try:
        _s, raw = get(f"{cdn}/shadow/sat/{p}/latest_times.json")
        m = json.loads(raw)
    except Exception as e:
        # manifest unreachable FROM THIS VANTAGE is not a canary failure
        # (a CDN outage breaks everyone regardless of publish format);
        # report it and let the parent escalate only if it persists
        print(f"UNREACHABLE {p}: {e}")
        sys.exit(4)
    latest = m.get("latest")
    if latest:
        # timegm, not mktime: the stamp is UTC and the box TZ must not
        # influence the age (a TZ-shifted age would false-trip or mask)
        t = calendar.timegm(time.strptime(latest, "%Y%m%dT%H%M%SZ"))
        age = time.time() - t
        if age > stale_s:
            print(f"TRIP stale {p} latest={latest} age={int(age)}s")
            sys.exit(3)
    if not m.get("containers"):
        continue                    # not flipped/hinted yet: nothing to check
    mz = m.get("maxzoom")
    idx_url = f"{cdn}/shadow/sat/{p}/{latest}/tiles.z{mz}.json"
    try:
        st, raw = get(idx_url)
        idx = json.loads(raw)
    except Exception:
        # TRANSITION CASE (false-trip caught live 2026-08-03 23:00Z): a
        # hinted product's newest frame can legitimately be a LEGACY
        # per-tile frame (hint went sticky before the first container frame
        # published). Missing index + present per-tile z0 object = healthy
        # legacy frame, exactly what the viewer's fallback reads. Missing
        # BOTH = the frame is truly unreadable -> count toward C1.
        try:
            st0, _ = get(f"{cdn}/shadow/sat/{p}/{latest}/0/0/0.webp")
            legacy_ok = (st0 == 200)
        except Exception:
            legacy_ok = False
        if not legacy_ok:
            missing.append(f"{p}/{latest}")
        continue
    tiles = list(idx.get("tiles", {}).items())
    if not tiles:
        missing.append(f"{p}/{latest} (empty index)")
        continue
    # C2: one full byte-match + one spot ranged read
    name, (bkey, off, ln) = random.choice(tiles)
    burl = f"{cdn}/shadow/sat/{p}/{latest}/{bkey}"
    st, ranged = get(burl, {"Range": f"bytes={off}-{off+ln-1}"})
    if st == 200 and len(ranged) > ln:
        ranged = ranged[off:off+ln]
    _st, whole = get(burl)
    if ranged != whole[off:off+ln] or len(ranged) != ln:
        print(f"TRIP byte-mismatch {p}/{latest}/{name} got={len(ranged)} want={ln}")
        sys.exit(2)
    if not (ranged[:4] == b"RIFF" and ranged[8:12] == b"WEBP"):
        print(f"TRIP byte-mismatch {p}/{latest}/{name} not-webp")
        sys.exit(2)
print("INDEXMISS " + ";".join(missing) if missing else "OK")
sys.exit(0)
PYEOF
  return $?    # 0 ok | 2 byte-mismatch | 3 stale | 4 unreachable | other transient
}

lane_log_bad() {
  local since="$1"
  local bad
  bad=$(docker logs --since "$since" "$LANE_CONTAINER" 2>&1 |
        grep -cE "container (block|index) PUT failed" || true)
  [ "${bad:-0}" -gt 0 ] && { TRIP_REASON="lane-broken (container PUT failures: $bad)"; return 0; }
  local fails
  fails=$(docker logs --since "$since" "$LANE_CONTAINER" 2>&1 |
          grep -c "^\[FAIL\]" || true)
  [ "${fails:-0}" -ge 6 ] && { TRIP_REASON="lane-broken (FAILs: $fails)"; return 0; }
  return 1
}

do_rollback() {
  local reason="$1"
  log "TRIP: $reason -- rolling back to per-frame publishing"
  if grep -q "^${FLAG}=" "$ENV_FILE"; then
    sed -i "s/^${FLAG}=.*/${FLAG}=0/" "$ENV_FILE"
  else
    echo "${FLAG}=0" >> "$ENV_FILE"
  fi
  docker compose -p "$LANE_PROJ" -f docker-compose.s2.yml -f "$LANE_COMPOSE" \
    --profile cron up -d --no-build --force-recreate emit-cron
  # delete recent container-format frames so the backfill re-renders them
  # per-tile (bounded to REBUILD_REACH_S; older container frames stay and are
  # readable via the sticky hint)
  docker run --rm --env-file "$ENV_FILE" -e REACH_S="$REBUILD_REACH_S" \
    --entrypoint python tat-s2:latest -c '
import datetime as dt, os, sys
sys.path.insert(0, "/app")
import s1_ingest
store = s1_ingest.R2()
reach = int(os.environ.get("REACH_S", "2400"))
floor = (dt.datetime.utcnow() - dt.timedelta(seconds=reach)).strftime("%Y%m%dT%H%M%SZ")
for p in ["goes19/conus/ir", "goes19/conus/irbd",
          "goes19/conus/truecolor", "goes19/conus/c02"]:
    root = f"shadow/sat/{p}/"
    for pre in store.list_prefixes(root, start_after=root + floor):
        stamp = pre[len(root):].strip("/")
        _dirs, keys = store.list_level(pre)
        if any("tiles.z" in k and k.endswith(".json") for k in keys):
            all_keys = store.list_keys(pre)
            store.delete(all_keys)
            print(f"deleted container frame {p}/{stamp} ({len(all_keys)} keys)")
' 2>&1 | sed 's/^/[canary-cleanup] /'
  {
    echo "verdict: TRIPPED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "reason: $reason"
    echo "action: ${FLAG}=0, lane recreated, container frames newer than ${REBUILD_REACH_S}s deleted for re-render"
  } > "$VERDICT"
  exit 42
}

log "watching lane=$LANE_PROJ flag=$FLAG interval=${INTERVAL_S}s window=${WATCH_S}s"
start=$(date +%s)
misses=0
unreachable=0
while :; do
  now=$(date +%s)
  if [ $((now - start)) -ge "$WATCH_S" ]; then
    { echo "verdict: PASS $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "watched: ${WATCH_S}s, no trip; flag stays ON"; } > "$VERDICT"
    log "PASS -- ${WATCH_S}s without a trip; container publishing stays on"
    exit 0
  fi
  out=$(check_round 2>&1); rc=$?
  echo "$out" | tail -2 | sed 's/^/[canary-check] /'
  case $rc in
    2) do_rollback "byte-mismatch: $(echo "$out" | grep TRIP | head -1)" ;;
    3) do_rollback "stale: $(echo "$out" | grep TRIP | head -1)" ;;
    4) unreachable=$((unreachable + 1))
       log "manifest unreachable round $unreachable/5 (CDN vantage, no trip yet)"
       [ "$unreachable" -ge 5 ] && \
         do_rollback "unreachable (5 consecutive rounds: cannot verify canary health)"
       ;;
    0) unreachable=0 ;;
    *) log "checker transient rc=$rc (no trip)" ;;
  esac
  if echo "$out" | grep -q "^INDEXMISS .*[a-z]"; then
    misses=$((misses + 1))
    log "index-missing round $misses/2: $(echo "$out" | grep INDEXMISS)"
    [ "$misses" -ge 2 ] && do_rollback "index-missing (2 consecutive rounds)"
  else
    misses=0
  fi
  if lane_log_bad "${INTERVAL_S}s"; then
    do_rollback "$TRIP_REASON"
  fi
  sleep "$INTERVAL_S"
done
