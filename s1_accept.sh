#!/usr/bin/env bash
# =============================================================================
# s1_accept.sh -- one-shot/--watch §8 acceptance gate for S1, REMOTE.
#
# Runs both remote checks (no SSH, no box access -- R2 creds if present, else the
# PUBLIC CDN + anonymous NOAA bucket) and prints a single-screen PASS/FAIL
# against the §8 S1 gate:
#     (1) never-miss: ZERO missed slots over the window (s1_audit.py --remote)
#     (2) byte-identity: REAL pixel delta == 0 (s1_pixeldiff.py --remote);
#         the lossy-WebP cross-build floor is allowed and reported.
#
#   ./s1_accept.sh                       # one shot, last 6 h, sample 30
#   ./s1_accept.sh --hours 24 --sample 50
#   ./s1_accept.sh --watch 300           # re-check every 300 s, compact one-liner
#
# Exit (one-shot): 0 = GREEN (gate passed), 2 = FAIL (real miss or real delta),
#                  3 = PENDING (no failure, but still draining / not enough data).
# --watch loops forever (Ctrl-C to stop), printing one timestamped line per cycle.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
HOURS=6
SAMPLE=30
PREFIX="${S1_R2_PREFIX:-shadow}"
WATCH=""
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours)  HOURS="$2"; shift 2;;
    --sample) SAMPLE="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --watch)  WATCH="$2"; shift 2;;
    *) PASSTHRU+=("$1"); shift;;
  esac
done

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Returns: sets globals AUDIT_VERDICT, AUDIT_LINE, PIX_VERDICT, PIX_LINE, GATE.
run_cycle() {
  local af pf
  af="$(mktemp)"; pf="$(mktemp)"
  "$PY" "$HERE/s1_audit.py" --remote --hours "$HOURS" --prefix "$PREFIX" \
      ${PASSTHRU[@]+"${PASSTHRU[@]}"} >"$af" 2>&1; AUDIT_RC=$?
  "$PY" "$HERE/s1_pixeldiff.py" --remote --sample "$SAMPLE" --prefix "$PREFIX" \
      >"$pf" 2>&1; PIX_RC=$?
  AUDIT_OUT="$(cat "$af")"; PIX_OUT="$(cat "$pf")"
  rm -f "$af" "$pf"

  # Classify each from its verdict marker.
  if   grep -q "NEVER-MISS FAIL" <<<"$AUDIT_OUT"; then AUDIT_VERDICT=FAIL
  elif grep -q "NEVER-MISS PASS (zero missed)" <<<"$AUDIT_OUT"; then AUDIT_VERDICT=PASS
  elif grep -q "NEVER-MISS PASS-with-pending" <<<"$AUDIT_OUT"; then AUDIT_VERDICT=PASS_PENDING
  else AUDIT_VERDICT=PENDING; fi
  AUDIT_LINE="$(grep -E 'covered:|0 shadow frames|publish latency' <<<"$AUDIT_OUT" | tr '\n' ' ' | sed 's/  */ /g')"

  if   grep -q "PIXEL-DIFF FAIL" <<<"$PIX_OUT"; then PIX_VERDICT=FAIL
  elif grep -q "PIXEL-DIFF PASS" <<<"$PIX_OUT"; then PIX_VERDICT=PASS
  else PIX_VERDICT=PENDING; fi
  PIX_LINE="$(grep -E 'REAL source delta|cross-build|0 shadow frames|no common' <<<"$PIX_OUT" | tr '\n' ' ' | sed 's/  */ /g')"

  # Compose the §8 gate.
  if [[ "$AUDIT_VERDICT" == FAIL || "$PIX_VERDICT" == FAIL ]]; then GATE=FAIL
  elif [[ "$AUDIT_VERDICT" == PASS && "$PIX_VERDICT" == PASS ]]; then GATE=GREEN
  else GATE=PENDING; fi
}

gate_exit() {
  case "$1" in GREEN) return 0;; FAIL) return 2;; *) return 3;; esac
}

if [[ -n "$WATCH" ]]; then
  echo "watching S1 §8 gate every ${WATCH}s (Ctrl-C to stop) | hours=$HOURS sample=$SAMPLE prefix=$PREFIX"
  while true; do
    run_cycle
    printf "[%s] gate=%-7s audit=%-12s pixel=%-7s | %s\n" \
      "$(ts)" "$GATE" "$AUDIT_VERDICT" "$PIX_VERDICT" "$PIX_LINE"
    sleep "$WATCH"
  done
else
  run_cycle
  echo "================= S1 §8 ACCEPTANCE (remote) ================="
  echo "$AUDIT_OUT"
  echo "------------------------------------------------------------"
  echo "$PIX_OUT"
  echo "============================================================"
  echo "  never-miss : $AUDIT_VERDICT"
  echo "  byte-id    : $PIX_VERDICT"
  echo ">> §8 GATE: $GATE  (GREEN=passed, PENDING=draining/insufficient data, FAIL=real miss/delta)"
  gate_exit "$GATE"; exit $?
fi
