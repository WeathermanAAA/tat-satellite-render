#!/usr/bin/env python3
"""S1 shadow-vs-prod pixel diff (SATELLITE-REARCH §7.2/§9 strict-identity tier).

For each slot present in BOTH the shadow product (shadow/sat/goes19/meso2/ir/)
and the prod meso product (meso/goes19-m2/ir/), decode both WebP frames and
compare: byte-equal, then decoded-pixel-equal, then (if not) the max abs channel
diff + differing-pixel fraction -- so a sub-pixel cross-host AA shift is told
apart from real renderer drift. For a "by construction" product ANY decoded diff
on a same-bbox slot is a wrapper bug, not a tolerance to widen.

NOTE on the bbox-timing confound (ZVC-7, applied to operator-steered meso): prod
renders "latest" with its LAST-DISCOVERED bbox (<=120 s stale); shadow renders
the slot's OWN extent. When the operator box is stable across that window the
bbox is identical and the frames are byte-identical; right after an operator
move they can differ (prod used the stale bbox). This tool reports the identical
fraction and flags the rest for inspection rather than asserting 100%.

  python s1_pixeldiff.py --sample 30          # 30 most-recent common slots
  python s1_pixeldiff.py --sample 30 --cdn     # read prod from the public CDN
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import sys

import boto3

import s1_slots as S

UTC = dt.timezone.utc
CDN = "https://cdn.triple-a-tropics.com"


def _cdn_get(url: str) -> bytes:
    """GET a CDN object. Cloudflare 403s the default Python-urllib UA, so send a
    browser-y one."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "tat-s1-pixeldiff/1.0"})
    return urllib.request.urlopen(req, timeout=30).read()


def _r2():
    return boto3.client(
        "s3", endpoint_url=os.environ.get("R2_ENDPOINT"),
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"))


def _list_stamps(s3, bucket, key_prefix):
    out, token = set(), None
    while True:
        kw = dict(Bucket=bucket, Prefix=key_prefix)
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            st = S.stamp_from_frame_key(o["Key"])
            if st:
                out.add(st)
        if r.get("IsTruncated"):
            token = r.get("NextContinuationToken")
        else:
            break
    return out


def diff_frames(a: bytes, b: bytes) -> dict:
    """Compare two WebP frames. Returns byte_equal / pixel_equal / max_abs_diff /
    diff_frac / shape match."""
    import numpy as np
    from PIL import Image
    if a == b:
        return {"byte_equal": True, "pixel_equal": True, "max_abs_diff": 0,
                "diff_frac": 0.0, "shape_match": True}
    ia = np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), dtype=np.int16)
    ib = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.int16)
    if ia.shape != ib.shape:
        return {"byte_equal": False, "pixel_equal": False, "max_abs_diff": None,
                "diff_frac": None, "shape_match": False,
                "shape_a": ia.shape, "shape_b": ib.shape}
    d = np.abs(ia - ib)
    diff_px = int((d.max(axis=-1) > 0).sum())
    total_px = ia.shape[0] * ia.shape[1]
    return {"byte_equal": False, "pixel_equal": bool(d.max() == 0),
            "max_abs_diff": int(d.max()), "diff_frac": diff_px / total_px,
            "shape_match": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S1 shadow-vs-prod pixel diff")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--prefix", default=os.environ.get("S1_R2_PREFIX", "shadow"))
    ap.add_argument("--cdn", action="store_true",
                    help="read prod meso frames from the public CDN instead of R2")
    a = ap.parse_args(argv)

    s3 = _r2()
    bucket = os.environ.get("R2_BUCKET", "triple-a-tropics-media")
    shadow_stamps = _list_stamps(s3, bucket, f"{a.prefix.strip('/')}/{S.S1_PRODUCT_PATH}/")

    if a.cdn:
        import json
        band = json.loads(_cdn_get(f"{CDN}/meso/goes19-m2/manifest.json")
                          ).get("bands", {}).get("ir", {})
        prod_stamps = {S.stamp_from_frame_key(f["key"]) for f in band.get("frames", [])}
        prod_stamps = {s for s in prod_stamps if s}
    else:
        prod_stamps = _list_stamps(s3, bucket, f"{S.S1_PROD_PRODUCT_PATH}/")

    common = sorted(shadow_stamps & prod_stamps)[-a.sample:]
    print(f"shadow slots={len(shadow_stamps)} prod slots={len(prod_stamps)} "
          f"common={len(shadow_stamps & prod_stamps)} | diffing {len(common)}")
    if not common:
        print(">> no common slots yet (let both pipelines run); cannot diff")
        return 1

    def fetch(key):
        if a.cdn and key.startswith("meso/"):
            return _cdn_get(f"{CDN}/{key}")
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    n_byte = n_pixel = n_diff = 0
    worst = []
    for st in common:
        try:
            sb = fetch(S.shadow_frame_key(a.prefix, st))
            pb = fetch(S.prod_frame_key(st))
        except Exception as e:  # noqa: BLE001
            print(f"  {st}: fetch error {e}")
            continue
        r = diff_frames(sb, pb)
        if r["byte_equal"]:
            n_byte += 1
            n_pixel += 1
        elif r["pixel_equal"]:
            n_pixel += 1
        else:
            n_diff += 1
            worst.append((st, r))
    print(f"  byte-identical : {n_byte}/{len(common)}")
    print(f"  pixel-identical: {n_pixel}/{len(common)}  (decoded array equal)")
    print(f"  differing      : {n_diff}/{len(common)}")
    for st, r in worst[:10]:
        print(f"    {st}: max_abs_diff={r['max_abs_diff']} "
              f"diff_frac={r['diff_frac']} shape_match={r['shape_match']}")
    ok = n_pixel == len(common)
    print(f">> PIXEL-DIFF {'PASS (all common slots pixel-identical)' if ok else 'see differing slots (likely bbox-timing, inspect)'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
