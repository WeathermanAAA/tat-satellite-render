#!/usr/bin/env python3
"""s2_ops_sweep.py — sum the fleet's measured R2 request counters.

Runs ON a box. Parses every tat-s2-*-emit-cron-1 container's `[ops]` log
lines (exact per-request counters from s1_ingest.R2, one line per pass) over
a window and reports Class A (list_pages + put) and Class B (get + head)
totals, extrapolated to a daily rate. This is the evidence line for the
2026-08 R2 cost incident: the 16-day bill averaged ~10.7M Class A ops/day;
the post-fix projection was ~2.25M/day before the container flips and
~0.6M/day after. Compare THIS number, not vibes, and let the billing page
be the independent confirmation.

Usage (per box):  python3 scripts/s2_ops_sweep.py [--hours 6]
Prints one JSON line: totals, per-lane breakdown, daily extrapolation.
Notes: DeleteObjects requests are free (delete_req excluded from Class A);
the prune lane logs no [ops] lines (raw boto client) — its LIST cost is the
walk itself, reported separately by its own log lines.
"""
import argparse
import json
import re
import subprocess
import sys

OPS_RE = re.compile(r"\[ops\] (.+?) pass: (.+)$")
KV_RE = re.compile(r"(\w+)=(\d+)")
CLASS_A = ("list_pages", "put")
CLASS_B = ("get", "head")


def containers():
    out = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=True).stdout.split()
    return sorted(c for c in out
                  if c.startswith("tat-s2-") and "emit-cron" in c
                  and "prune" not in c)


def sweep(hours: float) -> dict:
    """Per-lane daily rates use each container's ACTUAL log span, not the
    requested window: a force-recreated lane has minutes of log, and
    dividing its sum by the full window silently understates it (caught on
    the first post-flip verification sweep, 2026-08-04)."""
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    lanes = {}
    a_daily = b_daily = 0.0
    for c in containers():
        log = subprocess.run(
            ["docker", "logs", "-t", "--since", f"{int(hours * 3600)}s", c],
            capture_output=True, text=True).stdout
        agg = {}
        passes = 0
        first_ts = None
        for line in log.splitlines():
            if first_ts is None and len(line) > 30:
                try:
                    first_ts = dt.datetime.fromisoformat(
                        line.split()[0].replace("Z", "+00:00")[:32])
                except ValueError:
                    pass
            m = OPS_RE.search(line)
            if not m:
                continue
            passes += 1
            for k, v in KV_RE.findall(m.group(2)):
                agg[k] = agg.get(k, 0) + int(v)
        span_h = hours
        if first_ts is not None:
            span_h = min(hours, max((now - first_ts).total_seconds() / 3600,
                                    0.05))
        la = sum(agg.get(k, 0) for k in CLASS_A)
        lb = sum(agg.get(k, 0) for k in CLASS_B)
        lanes[c] = {"passes": passes, "span_h": round(span_h, 2),
                    "class_a_daily": int(la * 24 / span_h), **agg}
        a_daily += la * 24 / span_h
        b_daily += lb * 24 / span_h
    tot = {}
    for v in lanes.values():
        for k, n in v.items():
            if k not in ("passes", "span_h", "class_a_daily"):
                tot[k] = tot.get(k, 0) + n
    return {
        "window_h": hours,
        "class_a_daily": int(a_daily),
        "class_b_daily": int(b_daily),
        "detail": tot,
        "lanes": lanes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0,
                    help="log window to sum (default 6h)")
    args = ap.parse_args()
    print(json.dumps(sweep(args.hours)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
