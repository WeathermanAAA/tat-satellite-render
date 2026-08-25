#!/usr/bin/env bash
# s2_cov_canary_watchdog.sh -- AUTOMATIC rollback for the coverage-from-manifest
# canary (change C of the 2026-08-25 cost audit). Same principle as
# s2_canary_watchdog.sh: a canary that needs someone awake is not a canary.
#
# Watches ONE lane (default tat-s2-hwfd-leads on box2) that was recreated with
# the tat-s2:cov image (compose override /root/canary-cov.yml). Every ROUND_S:
#   C1 stale       : any canary product's latest_times.json `latest` on the CDN
#                    is older than STALE_S (default 4200 s = 7x the 10-min step)
#   C2 broken      : the lane log shows a Python traceback or >= 6 consecutive
#                    [FAIL] lines in the last round
#   C3 ghost       : the newest advertised frame of a product has no readable
#                    _ready.json (advertised-but-missing = the manifest lies)
#   C4 no-progress : no `[done]` line at all in the lane log for STALE_S
# Trip: recreate the lane WITHOUT the override (back to tat-s2:latest, the old
# code), write the verdict file, exit 42. PASS: WATCH_S elapsed with no trip ->
# verdict PASS, exit 0 (the lane stays on the canary image).
#
# Run on box2 as a transient unit:
#   systemd-run --unit s2covcanary --working-directory=/root/tsr-s2 \
#     --property=SuccessExitStatus="0 42" /root/s2_cov_canary_watchdog.sh
set -u
cd "${S2_CANARY_REPO:-/root/tsr-s2}"
CDN="https://cdn.triple-a-tropics.com"
PRODUCTS="${S2_CANARY_PRODUCTS:-himawari9/fd/ir himawari9/fd/irbd himawari9/fd/sandwich himawari9/fd/airmass himawari9/fd/b08}"
LANE_PROJ="${S2_CANARY_LANE_PROJ:-tat-s2-hwfd-leads}"
LANE_COMPOSE="${S2_CANARY_LANE_COMPOSE:-docker-compose.s2.hw-fd-leads.yml}"
LANE_CONTAINER="${LANE_PROJ}-emit-cron-1"
STALE_S="${S2_CANARY_STALE_S:-4200}"
WATCH_S="${S2_CANARY_WATCH_S:-7200}"
ROUND_S="${S2_CANARY_ROUND_S:-300}"
VERDICT="/root/s2_cov_canary_verdict_${LANE_PROJ}.txt"
LOG="/root/s2_cov_canary_${LANE_PROJ}.log"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
trip() {
  say "TRIP $1 -- rolling the lane back to tat-s2:latest"
  docker compose -p "$LANE_PROJ" -f docker-compose.s2.yml -f "$LANE_COMPOSE" --profile cron \
    up -d --no-build --no-deps --force-recreate emit-cron >> "$LOG" 2>&1
  docker ps -a --filter "name=$LANE_CONTAINER" --format '{{.Names}} {{.Status}} {{.Image}}' >> "$LOG"
  echo "TRIP $(date -u +%FT%TZ) $1" > "$VERDICT"
  exit 42
}
stamp_epoch() { # 20260825T181000Z -> epoch
  local s="$1"; date -u -d "${s:0:4}-${s:4:2}-${s:6:2}T${s:9:2}:${s:11:2}:${s:13:2}Z" +%s 2>/dev/null || echo 0
}
say "watch start lane=$LANE_PROJ products=[$PRODUCTS] stale=${STALE_S}s watch=${WATCH_S}s round=${ROUND_S}s image=$(docker inspect -f '{{.Config.Image}}' "$LANE_CONTAINER" 2>/dev/null)"
t0=$(date +%s)
while :; do
  now=$(date +%s)
  # C2: lane health from its own log (last round)
  tb=$(docker logs --since "${ROUND_S}s" "$LANE_CONTAINER" 2>&1 | grep -c "Traceback (most recent call last)")
  fails=$(docker logs --since "${ROUND_S}s" "$LANE_CONTAINER" 2>&1 | grep -E '^\[FAIL\]' | wc -l)
  [ "$tb" -gt 0 ] && trip "C2 traceback in lane log ($tb)"
  [ "$fails" -ge 6 ] && trip "C2 $fails [FAIL] lines in one round"
  # C4: progress
  if [ $((now - t0)) -ge "$STALE_S" ]; then
    done_n=$(docker logs --since "${STALE_S}s" "$LANE_CONTAINER" 2>&1 | grep -c '^\[done\]')
    [ "$done_n" -eq 0 ] && trip "C4 no [done] line in ${STALE_S}s"
  fi
  # C1 + C3 per product
  for p in $PRODUCTS; do
    m=$(curl -fsS -m 20 "$CDN/shadow/sat/$p/latest_times.json?t=$now" 2>/dev/null) || { say "warn: manifest fetch failed for $p"; continue; }
    latest=$(printf '%s' "$m" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("latest") or "")' 2>/dev/null)
    [ -z "$latest" ] && trip "C1 $p manifest has no latest"
    age=$((now - $(stamp_epoch "$latest")))
    [ "$age" -gt "$STALE_S" ] && trip "C1 $p latest $latest is ${age}s old (> ${STALE_S}s)"
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 "$CDN/shadow/sat/$p/$latest/_ready.json?t=$now")
    [ "$code" != "200" ] && trip "C3 $p newest advertised frame $latest has no _ready.json (HTTP $code)"
  done
  # ops line for the record: list_pages on this round's passes
  lp=$(docker logs --since "${ROUND_S}s" "$LANE_CONTAINER" 2>&1 | grep -oE 'list_pages=[0-9]+' | awk -F= '{s+=$2} END{print s+0}')
  passes=$(docker logs --since "${ROUND_S}s" "$LANE_CONTAINER" 2>&1 | grep -c '\[ops\]')
  cover=$(docker logs --since "${ROUND_S}s" "$LANE_CONTAINER" 2>&1 | grep -c '\[cover\] .*manifest (')
  say "round ok: passes=$passes list_pages=$lp manifest-cover-lines=$cover elapsed=$((now - t0))s"
  if [ $((now - t0)) -ge "$WATCH_S" ]; then
    say "PASS: ${WATCH_S}s without a trip; lane stays on $(docker inspect -f '{{.Config.Image}}' "$LANE_CONTAINER")"
    echo "PASS $(date -u +%FT%TZ)" > "$VERDICT"
    exit 0
  fi
  sleep "$ROUND_S"
done
