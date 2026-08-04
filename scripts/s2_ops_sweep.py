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
    lanes = {}
    for c in containers():
        log = subprocess.run(
            ["docker", "logs", "--since", f"{int(hours * 3600)}s", c],
            capture_output=True, text=True).stdout
        agg = {}
        passes = 0
        for line in log.splitlines():
            m = OPS_RE.search(line)
            if not m:
                continue
            passes += 1
            for k, v in KV_RE.findall(m.group(2)):
                agg[k] = agg.get(k, 0) + int(v)
        lanes[c] = {"passes": passes, **agg}
    tot = {}
    for v in lanes.values():
        for k, n in v.items():
            if k != "passes":
                tot[k] = tot.get(k, 0) + n
    a = sum(tot.get(k, 0) for k in CLASS_A)
    b = sum(tot.get(k, 0) for k in CLASS_B)
    return {
        "window_h": hours,
        "class_a": a, "class_b": b,
        "class_a_daily": int(a * 24 / hours),
        "class_b_daily": int(b * 24 / hours),
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
