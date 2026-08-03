#!/usr/bin/env python3
"""Object-level TTL prune for the shadow satellite prefixes (SATELLITE-REARCH
§4.3 retention, reworked 2026-07-09): delete every frame under ``shadow/sat/**``
older than N days using ONLY ListObjectsV2 + DeleteObject(s) + Get/PutObject.

WHY NOT BUCKET LIFECYCLE: the box R2 token has object read/write but NOT
Get/PutBucketLifecycleConfiguration (scripts/s2_shadow_lifecycle.py
AccessDenied'd on the box, 2026-07-08), and token scopes are frozen by
decision. This prune needs no extra permissions and replaces that approach
outright -- do not reintroduce the lifecycle rule.

What it deletes (age = the frame STAMP parsed from the key, not LastModified,
so a re-uploaded old frame still ages out on data time):
  * tiled pyramid frames   {prefix}/{product}/{stamp}/{z}/{x}/{y}.webp
                           + {stamp}/bt.png + {stamp}/_ready.json
                           (the READY MARKER IS DELETED FIRST so an
                           interrupted prune can never leave a frame that
                           ``complete_stamps`` still advertises)
  * single-frame products  {prefix}/{product}/{stamp}.webp  (the S1 shape --
                           s1_ingest already prunes its own on an hours
                           window; this catches restart orphans only)
What it never touches: any key without a valid stamp segment/basename --
latest_times.json, products.json, health.json, colorbars, foreign prefixes.

Safety rails: dry-run by DEFAULT (``--apply`` deletes); refuses any prefix
that does not start with ``shadow/``; keeps the newest ``--keep-min`` stamps
per product regardless of age (an emitter outage must never empty a product);
after deleting, rewrites any latest_times.json that still lists a pruned
stamp. NOTE (2026-08-03): active products no longer rebuild their manifest
from ``complete_stamps`` on every emit -- they append incrementally and
filter times[] to the S2_PRUNE_DAYS horizon, full-rebuilding only on the
S2_MANIFEST_HEAL_S tick -- so this rewrite is the main between-tick
correction for RETIRED products, and the emitter's horizon filter is what
prevents an in-flight append from resurrecting stamps this prune deletes.

Box usage (same R2 env as the emit services; see docker-compose.s2.yml):
  python s2_prune.py                          # dry-run report, 14d/keep-2
  python s2_prune.py --apply                  # actually delete
  python s2_prune.py --days 14 --keep-min 2 --prefix shadow/sat/ --apply
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass, field

UTC = dt.timezone.utc
STAMP_FMT = "%Y%m%dT%H%M%SZ"          # mirrors s1_slots.STAMP_FMT
_STAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")
READY_MARKER = "_ready.json"          # mirrors s2_registry.READY_MARKER
MANIFEST = "latest_times.json"
DELETE_BATCH = 1000                   # DeleteObjects hard cap


def parse_stamp(seg: str):
    """A segment/basename that is a frame stamp -> aware datetime, else None."""
    if not _STAMP_RE.match(seg):
        return None
    try:
        return dt.datetime.strptime(seg, STAMP_FMT).replace(tzinfo=UTC)
    except ValueError:
        return None


def classify_key(key: str):
    """Key -> (product_dir, stamp, is_marker) or None for non-frame keys.

    product_dir is the full key prefix up to (not including) the stamp, e.g.
    ``shadow/sat/goes19/conus/ir``. Tiled frames carry the stamp as a path
    SEGMENT; single-frame products carry it as the file basename.
    """
    segs = key.split("/")
    # tiled: first directory segment that parses as a stamp
    for i, seg in enumerate(segs[:-1]):
        if parse_stamp(seg):
            return "/".join(segs[:i]), seg, segs[-1] == READY_MARKER
    # single-frame: basename is {stamp}{ext}
    base = segs[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    if parse_stamp(stem):
        return "/".join(segs[:-1]), stem, False
    return None


@dataclass
class ProductPlan:
    product: str
    condemned: list = field(default_factory=list)   # stamps to delete (old->new)
    kept_old: list = field(default_factory=list)    # over-age but keep-min-protected
    keys: list = field(default_factory=list)        # delete order: markers first
    bytes: int = 0
    total_stamps: int = 0


def plan_prune(objects, *, days: int, keep_min: int, now: dt.datetime) -> list:
    """objects: iterable of (key, size). Returns [ProductPlan] with work to do.

    A stamp is condemned iff parse(stamp) < now - days AND it is not among the
    newest keep_min stamps of its product (stamp strings sort chronologically).
    """
    cutoff = now - dt.timedelta(days=days)
    per: dict = {}            # product -> stamp -> {"keys":[(key,is_marker,size)]}
    for key, size in objects:
        c = classify_key(key)
        if c is None:
            continue
        product, stamp, is_marker = c
        per.setdefault(product, {}).setdefault(stamp, []).append((key, is_marker, size))

    plans = []
    for product, stamps in sorted(per.items()):
        ordered = sorted(stamps)                       # chronological
        protected = set(ordered[-keep_min:]) if keep_min > 0 else set()
        plan = ProductPlan(product=product, total_stamps=len(ordered))
        for stamp in ordered:
            t = parse_stamp(stamp)
            if t is None or t >= cutoff:
                continue
            if stamp in protected:
                plan.kept_old.append(stamp)
                continue
            plan.condemned.append(stamp)
            entries = stamps[stamp]
            # ready marker FIRST: an interrupted prune leaves an un-advertised
            # partial (cleaned next run), never an advertised half-frame.
            entries.sort(key=lambda e: (not e[1], e[0]))
            plan.keys.extend(k for k, _m, _s in entries)
            plan.bytes += sum(s for _k, _m, s in entries)
        if plan.condemned or plan.kept_old:
            plans.append(plan)
    return plans


def rewrite_manifest(manifest: dict, condemned: set):
    """Drop condemned stamps from a latest_times manifest (single-frame and
    tiled share times/latest/count). Returns the new dict, or None if no
    referenced stamp was pruned (no write needed)."""
    times = manifest.get("times") or []
    keep = [t for t in times if t not in condemned]
    if len(keep) == len(times):
        return None
    out = dict(manifest)
    out["times"] = keep
    out["latest"] = keep[-1] if keep else None
    out["count"] = len(keep)
    return out


# --------------------------------------------------------------------------
# boto3 layer (kept thin + injectable so tests use a FakeS3, repo convention)
# --------------------------------------------------------------------------
def make_client():
    import boto3
    from botocore.config import Config as BotoConfig
    ep = os.environ.get("R2_ENDPOINT")
    if not ep:
        sys.exit("ERROR: R2_ENDPOINT not set. Run on the box (or export the R2 env).")
    return boto3.client(
        "s3", endpoint_url=ep,
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}))


def list_objects(s3, bucket: str, prefix: str):
    """Yield (key, size) for every object under prefix (paginated)."""
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            yield o["Key"], o.get("Size", 0)
        if not resp.get("IsTruncated"):
            return
        token = resp.get("NextContinuationToken")


def delete_keys(s3, bucket: str, keys: list) -> int:
    """Batch DeleteObjects; on AccessDenied fall back to per-key DeleteObject
    (some narrowly-scoped tokens allow the singular op only). Preserves the
    caller's marker-first ordering in both paths."""
    from botocore.exceptions import ClientError
    deleted = 0
    batch_ok = True
    i = 0
    while i < len(keys):
        chunk = keys[i:i + DELETE_BATCH]
        if batch_ok:
            try:
                s3.delete_objects(Bucket=bucket,
                                  Delete={"Objects": [{"Key": k} for k in chunk],
                                          "Quiet": True})
                deleted += len(chunk)
                i += len(chunk)
                continue
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") not in ("AccessDenied", "NotImplemented"):
                    raise
                batch_ok = False
                print("[prune] DeleteObjects denied; falling back to per-key DeleteObject")
        for k in chunk:
            s3.delete_object(Bucket=bucket, Key=k)
            deleted += 1
        i += len(chunk)
    return deleted


def fix_manifest(s3, bucket: str, product: str, condemned: set, apply: bool) -> bool:
    """Rewrite {product}/latest_times.json if it references a pruned stamp."""
    from botocore.exceptions import ClientError
    key = f"{product}/{MANIFEST}"
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        manifest = json.loads(body)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return False
        raise
    except (json.JSONDecodeError, KeyError):
        print(f"[prune] WARNING: unparseable manifest left untouched: {key}")
        return False
    out = rewrite_manifest(manifest, condemned)
    if out is None:
        return False
    if apply:
        s3.put_object(Bucket=bucket, Key=key,
                      Body=json.dumps(out).encode(),
                      ContentType="application/json",
                      CacheControl="public, max-age=30")
    return True


def main(argv=None, s3=None):
    ap = argparse.ArgumentParser(description="shadow TTL prune (object ops only)")
    ap.add_argument("--prefix", default="shadow/sat/",
                    help="prune under this key prefix (must start with shadow/)")
    ap.add_argument("--days", type=int, default=14, help="TTL in days (default 14)")
    ap.add_argument("--keep-min", type=int, default=2,
                    help="always keep the newest N stamps per product (default 2)")
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", "triple-a-tropics-media"))
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry-run report only)")
    args = ap.parse_args(argv)

    if not args.prefix.startswith("shadow/"):
        sys.exit(f"REFUSED: prefix {args.prefix!r} is outside shadow/ -- this "
                 f"tool never touches production prefixes.")
    if args.days < 1:
        sys.exit("REFUSED: --days must be >= 1")

    s3 = s3 or make_client()
    now = dt.datetime.now(UTC)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[prune] {mode} prefix={args.prefix} ttl={args.days}d "
          f"keep-min={args.keep_min} bucket={args.bucket}")

    plans = plan_prune(list_objects(s3, args.bucket, args.prefix),
                       days=args.days, keep_min=args.keep_min, now=now)
    todo = [p for p in plans if p.condemned]
    n_keys = sum(len(p.keys) for p in todo)
    n_bytes = sum(p.bytes for p in todo)
    for p in plans:
        note = f" (keep-min held {len(p.kept_old)} over-age)" if p.kept_old else ""
        if p.condemned:
            print(f"  {p.product}: {len(p.condemned)}/{p.total_stamps} stamps -> "
                  f"{len(p.keys)} objects, {p.bytes/1e6:.1f} MB{note}")
        elif p.kept_old:
            print(f"  {p.product}: nothing to delete{note}")

    if not todo:
        print("[prune] nothing older than the TTL; done.")
        return 0
    print(f"[prune] TOTAL: {sum(len(p.condemned) for p in todo)} stamps, "
          f"{n_keys} objects, {n_bytes/1e6:.1f} MB")
    if args.apply:
        for p in todo:
            deleted = delete_keys(s3, args.bucket, p.keys)
            fixed = fix_manifest(s3, args.bucket, p.product, set(p.condemned), True)
            print(f"  deleted {deleted} objects from {p.product}"
                  + (" + manifest rewritten" if fixed else ""))
    else:
        for p in todo:
            if fix_manifest(s3, args.bucket, p.product, set(p.condemned), False):
                print(f"  (would also rewrite {p.product}/{MANIFEST})")
        print("[prune] dry-run only -- re-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
