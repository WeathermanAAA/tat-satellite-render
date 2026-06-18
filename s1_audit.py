#!/usr/bin/env python3
"""S1 never-miss audit (SATELLITE-REARCH §8 S1 gate).

Independent ground truth: list the NOAA CMIPM2-C13 objects over a window
(ListObjectsV2, anonymous) -> the set of slots that SHOULD have produced a
shadow frame. Compare to the published shadow frames in R2. ZERO missed slots is
the gate. (S1's meso-2 is near-1:1 raw-slot->frame, so the raw-object audit IS
the frame-coverage audit here -- §3.3; S3 adds the storm-crop join.)

  python s1_audit.py --hours 6            # audit the last 6 hours
  python s1_audit.py --start ... --end ...

Runs on the box (R2 creds from the env) + NOAA anonymously. The set-comparison
core (audit_compare) is pure + unit-tested.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

import s1_slots as S

UTC = dt.timezone.utc


def audit_compare(ground_truth_stamps, published_stamps, no_data_stamps=()):
    """Pure set comparison. ``covered`` = published OR explicitly logged no-data
    (§3.3). Returns {missing, extra, covered_count, gt_count}."""
    gt = set(ground_truth_stamps)
    pub = set(published_stamps)
    nodata = set(no_data_stamps)
    covered = pub | nodata
    missing = sorted(gt - covered)
    extra = sorted(pub - gt)   # published but not in the (windowed) ground truth
    return {
        "missing": missing,
        "extra": extra,
        "gt_count": len(gt),
        "published_count": len(pub),
        "covered_in_window": len(gt & covered),
    }


def list_noaa_ground_truth(start: dt.datetime, end: dt.datetime) -> list[str]:
    """ListObjectsV2 the NOAA CMIPM prefix (anonymous) across the window's hour
    partitions; return the CMIPM2-C13 stamps in [start, end]."""
    s3 = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
    stamps: set[str] = set()
    h = start.replace(minute=0, second=0, microsecond=0)
    while h <= end:
        prefix = f"ABI-L2-CMIPM/{h.year}/{h.strftime('%j')}/{h.hour:02d}/"
        token = None
        while True:
            kw = dict(Bucket=S.S1_BUCKET, Prefix=prefix)
            if token:
                kw["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kw)
            for o in resp.get("Contents", []):
                slot = S.parse_goes_key(o["Key"])
                if S.is_s1_slot(slot) and start <= slot.scan_start <= end:
                    stamps.add(slot.stamp)
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        h += dt.timedelta(hours=1)
    return sorted(stamps)


def list_shadow_published(prefix: str) -> list[str]:
    """List the shadow frames in R2 -> their stamps."""
    s3 = boto3.client(
        "s3", endpoint_url=os.environ.get("R2_ENDPOINT"),
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"))
    bucket = os.environ.get("R2_BUCKET", "triple-a-tropics-media")
    key_prefix = f"{prefix.strip('/')}/{S.S1_PRODUCT_PATH}/"
    stamps: set[str] = set()
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=key_prefix)
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            st = S.stamp_from_frame_key(o["Key"])
            if st:
                stamps.add(st)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(stamps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S1 never-miss audit")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--start", help="ISO start (overrides --hours)")
    ap.add_argument("--end", help="ISO end (default now)")
    ap.add_argument("--prefix", default=os.environ.get("S1_R2_PREFIX", "shadow"))
    ap.add_argument("--lag-min", type=float, default=3.0,
                    help="trailing-edge guard: ignore slots newer than this many "
                         "minutes (publish+render+SQS latency window).")
    a = ap.parse_args(argv)

    end = (dt.datetime.fromisoformat(a.end.replace("Z", "+00:00")) if a.end
           else dt.datetime.now(UTC))
    end = end - dt.timedelta(minutes=a.lag_min)   # don't fault the in-flight edge
    start = (dt.datetime.fromisoformat(a.start.replace("Z", "+00:00")) if a.start
             else end - dt.timedelta(hours=a.hours))

    print(f"S1 never-miss audit | window {start.isoformat()} .. {end.isoformat()} "
          f"(lag guard {a.lag_min} min) | prefix={a.prefix}")
    gt = list_noaa_ground_truth(start, end)
    # Clip ground truth to the lag-guarded end as well.
    gt = [s for s in gt
          if dt.datetime.strptime(s, S.STAMP_FMT).replace(tzinfo=UTC) <= end]
    pub = list_shadow_published(a.prefix)
    pub_in = [s for s in pub
              if start <= dt.datetime.strptime(s, S.STAMP_FMT).replace(tzinfo=UTC) <= end]

    r = audit_compare(gt, pub_in)
    print(f"  NOAA CMIPM2-C13 ground-truth slots : {r['gt_count']}")
    print(f"  shadow frames published (in window): {len(pub_in)}")
    print(f"  covered (frame present)            : {r['covered_in_window']}")
    print(f"  MISSED slots                       : {len(r['missing'])}")
    if r["missing"]:
        print("  -- missing stamps --")
        for s in r["missing"]:
            print("    ", s)
    if r["extra"]:
        print(f"  (note: {len(r['extra'])} shadow frames not in the windowed "
              f"ground truth -- usually NOAA objects aged out of the listing)")
    ok = len(r["missing"]) == 0 and r["gt_count"] > 0
    print(f">> NEVER-MISS {'PASS (zero missed slots)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
