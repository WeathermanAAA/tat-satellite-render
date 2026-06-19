#!/usr/bin/env python3
"""S1 never-miss audit (SATELLITE-REARCH §8 S1 gate).

Independent ground truth: list the NOAA CMIPM2-C13 objects over a window
(ListObjectsV2, anonymous) -> the set of slots that SHOULD have produced a
shadow frame. Compare to the published shadow frames. ZERO missed slots is the
gate. (S1's meso-2 is near-1:1 raw-slot->frame, so the raw-object audit IS the
frame-coverage audit here -- §3.3; S3 adds the storm-crop join.)

Two modes:
  * on-box (default): shipped set via R2 ListObjectsV2 (R2 creds from the env).
      python s1_audit.py --hours 6
  * --remote:         shipped set via R2 creds IF present, else the PUBLIC CDN
      latest_times.json (no creds). NOAA ground truth is always anonymous. So
      the full gate can be evaluated from anywhere -- no SSH, no box access --
      during/after the manual deploy. Classifies covered/pending/missed so a
      backlog drain reads PENDING not MISS, and reports publish latency.
      python s1_audit.py --remote --hours 6

The classification core (s1_slots.classify_coverage) is pure + unit-tested.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

import s1_slots as S
import s1_sources as SRC

UTC = dt.timezone.utc
CDN = "https://cdn.triple-a-tropics.com"


# ---------------------------------------------------------------------------
# Pure comparison (on-box) -- kept for the existing on-box path + its test
# ---------------------------------------------------------------------------
def audit_compare(ground_truth_stamps, published_stamps, no_data_stamps=()):
    """Pure set comparison. ``covered`` = published OR explicitly logged no-data
    (§3.3). Returns {missing, extra, gt_count, published_count, covered_in_window}."""
    gt = set(ground_truth_stamps)
    pub = set(published_stamps)
    nodata = set(no_data_stamps)
    covered = pub | nodata
    return {
        "missing": sorted(gt - covered),
        "extra": sorted(pub - gt),
        "gt_count": len(gt),
        "published_count": len(pub),
        "covered_in_window": len(gt & covered),
    }


# ---------------------------------------------------------------------------
# NOAA ground truth (anonymous, no creds)
# ---------------------------------------------------------------------------
def list_noaa_ground_truth_detailed(src: SRC.SatSource, start: dt.datetime,
                                    end: dt.datetime):
    """ListObjectsV2 the source's NOAA prefixes (anonymous) across [start,end];
    return [(stamp, noaa_lastmodified)] for its COMPLETE slots. ABI: each accepted
    object is a complete slot. AHI FLDK: a stamp counts only when all 10 segments
    are present (never a half-scan), and its LastModified is the NEWEST segment's
    (when the scan became complete) -> the latency baseline."""
    s3 = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
    # stamp -> {item: lastmodified}; complete when required.issubset(items).
    acc: dict[str, dict] = {}
    lookback_min = int((end - start).total_seconds() // 60) + 15
    for prefix in SRC.noaa_prefixes(src, end, lookback_min):
        token = None
        while True:
            kw = dict(Bucket=src.bucket, Prefix=prefix)
            if token:
                kw["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kw)
            for o in resp.get("Contents", []):
                slot = SRC.parse(src, o["Key"])
                if SRC.is_ours(src, slot) and start <= slot.scan_start <= end:
                    lm = o["LastModified"]
                    lm = (lm if lm.tzinfo else lm.replace(tzinfo=UTC)).astimezone(UTC)
                    acc.setdefault(slot.stamp, {})[SRC.gate_item(src, slot)] = lm
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
    required = SRC.gate_required(src)
    out = {st: max(items.values()) for st, items in acc.items()
           if required.issubset(items.keys())}   # complete scans only
    return sorted(out.items())


def list_noaa_ground_truth(src: SRC.SatSource, start: dt.datetime,
                           end: dt.datetime) -> list[str]:
    """Just the stamps (on-box path)."""
    return [s for s, _ in list_noaa_ground_truth_detailed(src, start, end)]


# ---------------------------------------------------------------------------
# Shipped set (R2 ListObjectsV2 with creds, OR public CDN latest_times.json)
# ---------------------------------------------------------------------------
def _r2_creds_present() -> bool:
    return bool(os.environ.get("R2_ENDPOINT")
                and (os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"))
                and (os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")))


def _cdn_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tat-s1-audit/1.0"})
    return urllib.request.urlopen(req, timeout=30).read()


def _cdn_head_lastmod(url: str):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "tat-s1-audit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        lm = r.headers.get("Last-Modified")
    if not lm:
        return None
    try:
        from email.utils import parsedate_to_datetime
        d = parsedate_to_datetime(lm)
        return d.astimezone(UTC) if d.tzinfo else d.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def list_shadow_r2(prefix: str, product_path: str = S.S1_PRODUCT_PATH):
    """R2 ListObjectsV2 of the shadow product -> {stamp: put_time}."""
    s3 = boto3.client(
        "s3", endpoint_url=os.environ.get("R2_ENDPOINT"),
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"))
    bucket = os.environ.get("R2_BUCKET", "triple-a-tropics-media")
    key_prefix = f"{prefix.strip('/')}/{product_path}/"
    out: dict[str, dt.datetime] = {}
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=key_prefix)
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            st = S.stamp_from_frame_key(o["Key"])
            if st:
                lm = o["LastModified"]
                out[st] = (lm if lm.tzinfo else lm.replace(tzinfo=UTC)).astimezone(UTC)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return out


def list_shadow_cdn(prefix: str, product_path: str = S.S1_PRODUCT_PATH):
    """Public CDN latest_times.json -> {stamp: None} (no creds). Returns {} if the
    manifest is absent (worker not writing yet). PUT times are filled lazily via
    CDN HEAD by the caller (latency on a sample)."""
    url = f"{CDN}/{S.latest_times_key(prefix, product_path)}"
    try:
        body = _cdn_get(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}
        raise
    times = json.loads(body).get("times", [])
    return {st: None for st in times}


def list_shadow_remote(prefix: str, product_path: str = S.S1_PRODUCT_PATH,
                       latency_sample: int = 50):
    """Shipped set for --remote: R2 creds if present (authoritative + per-frame
    PUT time), else the public CDN manifest (+ CDN HEAD latency on a sample).
    Returns ({stamp: put_time_or_None}, source_label)."""
    if _r2_creds_present():
        return list_shadow_r2(prefix, product_path), "R2 ListObjectsV2 (creds)"
    shipped = list_shadow_cdn(prefix, product_path)
    # Fill PUT times for the most-recent `latency_sample` covered frames via HEAD.
    for st in sorted(shipped)[-latency_sample:]:
        try:
            shipped[st] = _cdn_head_lastmod(
                f"{CDN}/{S.shadow_frame_key(prefix, st, product_path)}")
        except Exception:  # noqa: BLE001 - latency is best-effort
            pass
    return shipped, "public CDN latest_times.json (no creds)"


def _fmt_latency(latencies):
    if not latencies:
        return "n/a"
    latencies = sorted(latencies)
    n = len(latencies)
    med = latencies[n // 2]
    return f"min={latencies[0]:.0f}s median={med:.0f}s max={latencies[-1]:.0f}s (n={n})"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _window(a):
    end = (dt.datetime.fromisoformat(a.end.replace("Z", "+00:00")) if a.end
           else dt.datetime.now(UTC))
    end = end - dt.timedelta(minutes=a.lag_min)
    start = (dt.datetime.fromisoformat(a.start.replace("Z", "+00:00")) if a.start
             else end - dt.timedelta(hours=a.hours))
    return start, end


def run_onbox(a) -> int:
    src = SRC.get_source(a.source)
    start, end = _window(a)
    print(f"S1 never-miss audit (on-box) | source={src.key} | {start.isoformat()} "
          f".. {end.isoformat()} | prefix={a.prefix}")
    gt = [s for s in list_noaa_ground_truth(src, start, end)
          if dt.datetime.strptime(s, S.STAMP_FMT).replace(tzinfo=UTC) <= end]
    # on-box shipped set via R2 (creds required here)
    pub = list(list_shadow_r2(a.prefix, src.product_path).keys())
    pub_in = [s for s in pub
              if start <= dt.datetime.strptime(s, S.STAMP_FMT).replace(tzinfo=UTC) <= end]
    r = audit_compare(gt, pub_in)
    print(f"  ground-truth slots: {r['gt_count']}  shadow: {len(pub_in)}  "
          f"covered: {r['covered_in_window']}  MISSED: {len(r['missing'])}")
    for s in r["missing"]:
        print("    missing:", s)
    ok = len(r["missing"]) == 0 and r["gt_count"] > 0
    print(f">> NEVER-MISS {'PASS (zero missed)' if ok else 'FAIL'}")
    return 0 if ok else 1


def run_remote(a) -> int:
    src = SRC.get_source(a.source)
    start, end = _window(a)
    print(f"S1 never-miss audit (REMOTE) | source={src.key} | {start.isoformat()} "
          f".. {end.isoformat()} (lag guard {a.lag_min} min) | prefix={a.prefix}")
    gt = [(s, lm) for s, lm in list_noaa_ground_truth_detailed(src, start, end)
          if dt.datetime.strptime(s, S.STAMP_FMT).replace(tzinfo=UTC) <= end]
    gt_times = {s: lm for s, lm in gt}
    gt_stamps = list(gt_times)
    shipped, source = list_shadow_remote(a.prefix, src.product_path)
    print(f"  shipped-set source: {source}")
    shadow_in = {s: t for s, t in shipped.items()
                 if start <= dt.datetime.strptime(s, S.STAMP_FMT).replace(tzinfo=UTC) <= end}

    # Empty/partial tolerance: 0 shadow frames -> worker not writing yet, exit 0.
    if not shipped:
        print(f"  0 shadow frames -- worker not writing yet, {len(gt_stamps)} "
              f"ground-truth slots waiting (PENDING, not a fail).")
        print(">> NEVER-MISS PENDING (pre-deploy / no frames yet) -- exit 0")
        return 0

    now = dt.datetime.now(UTC)
    cls = S.classify_coverage(gt_stamps, list(shadow_in), now, settle_s=a.settle_s)
    first_shadow = min(shipped) if shipped else None

    # Publish latency on covered slots with a known PUT time.
    lats = []
    for s in cls["covered"]:
        put = shadow_in.get(s)
        pub_t = gt_times.get(s)
        if put and pub_t:
            lats.append((put - pub_t).total_seconds())

    print(f"  first shadow frame seen: {first_shadow}")
    print(f"  ground-truth slots: {len(gt_stamps)}  shadow(in-window): {len(shadow_in)}")
    print(f"  covered: {len(cls['covered'])}  pending: {len(cls['pending'])}  "
          f"MISSED: {len(cls['missed'])}")
    print(f"  publish latency (NOAA publish -> R2 PUT): {_fmt_latency(lats)}")
    if cls["pending"]:
        print(f"  pending (in-flight / backlog drain / pre-worker): "
              f"{len(cls['pending'])}  e.g. {cls['pending'][:3]}")
    for s in cls["missed"]:
        print("    MISSED:", s)
    ok = len(cls["missed"]) == 0
    verdict = ("PASS (zero missed)" if ok and not cls["pending"]
               else "PASS-with-pending (zero missed; backlog still draining)"
               if ok else "FAIL (real misses)")
    print(f">> NEVER-MISS {verdict}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S1 never-miss audit")
    ap.add_argument("--remote", action="store_true",
                    help="evaluate from anywhere (R2 creds OR public CDN manifest)")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--start", help="ISO start (overrides --hours)")
    ap.add_argument("--end", help="ISO end (default now)")
    ap.add_argument("--prefix", default=os.environ.get("S1_R2_PREFIX", "shadow"))
    ap.add_argument("--source", default=os.environ.get("S1_SOURCE", "goes19"),
                    help="goes19 | goes18 | himawari9 (the satellite to audit)")
    ap.add_argument("--lag-min", type=float, default=3.0,
                    help="ignore slots newer than this many minutes (in-flight edge)")
    ap.add_argument("--settle-s", type=float, default=180.0,
                    help="a missing slot newer than now-settle_s is PENDING not MISS")
    a = ap.parse_args(argv)
    return run_remote(a) if a.remote else run_onbox(a)


if __name__ == "__main__":
    sys.exit(main())
