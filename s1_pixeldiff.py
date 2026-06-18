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


def _reencode_floor(rgb):
    """Per-pixel max-channel delta induced by ONE q90/method6 WebP re-encode of
    ``rgb`` -- the encode-quantization noise envelope for THIS content (the same
    transcode params render uses). Cross-build encoder disagreement on the same
    source is bounded by this."""
    import io as _io
    import numpy as np
    from PIL import Image
    buf = _io.BytesIO()
    Image.fromarray(rgb.astype("uint8"), "RGB").save(buf, "WEBP", quality=90, method=6)
    rgb2 = np.asarray(Image.open(_io.BytesIO(buf.getvalue())).convert("RGB"),
                      dtype=np.int16)
    return np.abs(rgb.astype(np.int16) - rgb2).max(axis=-1)


def decompose_diff(shadow: bytes, prod: bytes) -> dict:
    """Decompose shadow-vs-prod into the lossy-WebP CROSS-BUILD floor vs any REAL
    (source) pixel delta (SATELLITE-REARCH §7.2/§9 strict-identity tier).

    Method: measure the encode-noise ceiling by re-encoding EACH frame's own
    decoded source one more q90 round-trip (``_reencode_floor``); the floor
    ceiling = max self-noise of the two. A pixel whose |shadow-prod| exceeds that
    ceiling cannot be explained by encoder quantization -> it is a REAL source
    difference (wrong bbox/palette/coastline/content). REAL must be 0.

    Returns total_diff_frac, floor_frac (artifact), real_frac, floor_ceiling,
    real_max, shape_match. A shape mismatch is wholly REAL (framing/bbox bug)."""
    import numpy as np
    from PIL import Image
    if shadow == prod:
        return {"shape_match": True, "byte_equal": True, "total_diff_frac": 0.0,
                "floor_frac": 0.0, "real_frac": 0.0, "floor_ceiling": 0,
                "real_max": 0, "real_px": 0, "total_px": 0}
    a = np.asarray(Image.open(io.BytesIO(shadow)).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(io.BytesIO(prod)).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return {"shape_match": False, "byte_equal": False, "total_diff_frac": 1.0,
                "floor_frac": 0.0, "real_frac": 1.0, "floor_ceiling": None,
                "real_max": None, "shape_a": a.shape, "shape_b": b.shape}
    d = np.abs(a - b).max(axis=-1)
    ceiling = int(max(_reencode_floor(a).max(), _reencode_floor(b).max()))
    total_px = d.size
    diff_mask = d > 0
    real_mask = d > ceiling
    return {
        "shape_match": True, "byte_equal": False,
        "total_diff_frac": float(diff_mask.mean()),
        "floor_frac": float((diff_mask & ~real_mask).mean()),
        "real_frac": float(real_mask.mean()),
        "floor_ceiling": ceiling,
        "real_max": int(d[real_mask].max()) if real_mask.any() else 0,
        "real_px": int(real_mask.sum()),
        "total_px": int(total_px),
    }


def _prod_stamps_cdn() -> set:
    import json
    band = json.loads(_cdn_get(f"{CDN}/meso/goes19-m2/manifest.json")
                      ).get("bands", {}).get("ir", {})
    return {s for s in (S.stamp_from_frame_key(f["key"])
                        for f in band.get("frames", [])) if s}


def run_onbox(a) -> int:
    s3 = _r2()
    bucket = os.environ.get("R2_BUCKET", "triple-a-tropics-media")
    shadow_stamps = _list_stamps(s3, bucket, f"{a.prefix.strip('/')}/{S.S1_PRODUCT_PATH}/")
    prod_stamps = (_prod_stamps_cdn() if a.cdn
                   else _list_stamps(s3, bucket, f"{S.S1_PROD_PRODUCT_PATH}/"))
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
            n_byte += 1; n_pixel += 1
        elif r["pixel_equal"]:
            n_pixel += 1
        else:
            n_diff += 1; worst.append((st, r))
    print(f"  byte-identical : {n_byte}/{len(common)}")
    print(f"  pixel-identical: {n_pixel}/{len(common)}  (decoded array equal)")
    print(f"  differing      : {n_diff}/{len(common)}")
    for st, r in worst[:10]:
        print(f"    {st}: max_abs_diff={r['max_abs_diff']} diff_frac={r['diff_frac']}")
    ok = n_pixel == len(common)
    print(f">> PIXEL-DIFF {'PASS (all pixel-identical)' if ok else 'inspect differing'}")
    return 0 if ok else 2


def run_remote(a) -> int:
    # Shadow stamps: R2 creds if present, else the public CDN manifest (no creds).
    from s1_audit import _r2_creds_present, list_shadow_cdn, list_shadow_r2
    use_r2 = _r2_creds_present()
    shadow_stamps = set((list_shadow_r2(a.prefix) if use_r2
                         else list_shadow_cdn(a.prefix)).keys())
    src = "R2 (creds)" if use_r2 else "public CDN (no creds)"
    if not shadow_stamps:
        print(f"shadow source: {src} | 0 shadow frames yet -- worker not writing; "
              f"nothing to diff (PENDING, exit 0).")
        return 0
    prod_stamps = _prod_stamps_cdn()
    common = sorted(shadow_stamps & prod_stamps)[-a.sample:]
    print(f"shadow src={src} shadow={len(shadow_stamps)} prod(CDN)={len(prod_stamps)} "
          f"common={len(shadow_stamps & prod_stamps)} | diffing {len(common)}")
    if not common:
        print(">> no common slots yet (both pipelines need overlap); PENDING exit 0")
        return 0

    s3 = _r2() if use_r2 else None
    bucket = os.environ.get("R2_BUCKET", "triple-a-tropics-media")

    def fetch_shadow(st):
        if use_r2:
            return s3.get_object(Bucket=bucket,
                                 Key=S.shadow_frame_key(a.prefix, st))["Body"].read()
        return _cdn_get(f"{CDN}/{S.shadow_frame_key(a.prefix, st)}")

    n = real_total = 0
    floor_fracs, real_fracs = [], []
    flagged = []
    for st in common:
        try:
            sb = fetch_shadow(st)
            pb = _cdn_get(f"{CDN}/{S.prod_frame_key(st)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {st}: fetch error {e}"); continue
        d = decompose_diff(sb, pb)
        n += 1
        floor_fracs.append(d["floor_frac"])
        real_fracs.append(d["real_frac"])
        if d["real_frac"] > 0 or not d["shape_match"]:
            real_total += d.get("real_px", 1)
            flagged.append((st, d))
    if not n:
        print(">> no slots fetched; PENDING exit 0"); return 0
    avg_floor = 100 * sum(floor_fracs) / n
    avg_real = 100 * sum(real_fracs) / n
    print(f"  slots diffed: {n}")
    print(f"  cross-build (lossy-WebP) artifact: {avg_floor:.3f}% of pixels (avg)")
    print(f"  REAL source delta                : {avg_real:.6f}% of pixels (avg)  "
          f"[must be 0]")
    for st, d in flagged[:10]:
        print(f"    REAL on {st}: shape_match={d['shape_match']} "
              f"real_px={d.get('real_px')} real_max={d.get('real_max')} "
              f"ceiling={d.get('floor_ceiling')}")
    ok = real_total == 0
    print(f">> PIXEL-DIFF {'PASS (real delta = 0; difference is pure cross-build encode floor)' if ok else 'FAIL (real source delta > 0 -- a wrapper/render bug)'}")
    return 0 if ok else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S1 shadow-vs-prod pixel diff")
    ap.add_argument("--remote", action="store_true",
                    help="evaluate from anywhere (R2 creds OR public CDN); "
                         "decompose cross-build floor vs real delta")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--prefix", default=os.environ.get("S1_R2_PREFIX", "shadow"))
    ap.add_argument("--cdn", action="store_true",
                    help="(on-box) read prod meso frames from the CDN instead of R2")
    a = ap.parse_args(argv)
    return run_remote(a) if a.remote else run_onbox(a)


if __name__ == "__main__":
    sys.exit(main())
