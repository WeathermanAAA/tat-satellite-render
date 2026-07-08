#!/usr/bin/env python3
"""Apply a MERGE-SAFE R2 object-lifecycle TTL to the Stage-2 shadow pyramid prefix
(SATELLITE-REARCH §4.3: lifecycle is the retention floor; the app-side prune still
runs). Adds/updates ONE rule (id below) scoped to shadow/sat/goes19/** without
touching any existing bucket lifecycle rules (meso/floater/feeds keep theirs).

Reuses the box R2 env (R2_ENDPOINT / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY /
R2_BUCKET) -- the SAME vars s1_ingest uses. Run on the Hostinger box; NO creds are
needed in a Codespace.

  python scripts/s2_shadow_lifecycle.py            # 10-day TTL on shadow/sat/goes19/
  python scripts/s2_shadow_lifecycle.py --days 14 --prefix shadow/sat/goes19/
  python scripts/s2_shadow_lifecycle.py --show     # print current rules, change nothing
"""
import argparse
import os
import sys

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

RULE_ID = "s2-shadow-sat-goes19-ttl"


def client():
    ep = os.environ.get("R2_ENDPOINT")
    if not ep:
        sys.exit("ERROR: R2_ENDPOINT not set. Run on the box (or export the R2 env).")
    return boto3.client(
        "s3", endpoint_url=ep,
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}))


def get_rules(s3, bucket):
    try:
        return s3.get_bucket_lifecycle_configuration(Bucket=bucket).get("Rules", [])
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchLifecycleConfiguration",
                                           "NoSuchLifecycleConfigurationError"):
            return []
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="R2 shadow lifecycle TTL (merge-safe)")
    ap.add_argument("--days", type=int, default=10, help="expiration days (7-14; default 10)")
    ap.add_argument("--prefix", default="shadow/sat/goes19/", help="prefix to expire")
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", "triple-a-tropics-media"))
    ap.add_argument("--show", action="store_true", help="print current rules, make no change")
    args = ap.parse_args(argv)

    s3 = client()
    existing = get_rules(s3, args.bucket)
    print(f"[lifecycle] bucket={args.bucket}  existing rules: "
          f"{[r.get('ID') for r in existing]}")
    if args.show:
        for r in existing:
            print("  ", r)
        return 0

    # keep every OTHER rule verbatim; replace only ours (idempotent).
    kept = [r for r in existing if r.get("ID") != RULE_ID]
    ours = {
        "ID": RULE_ID,
        "Status": "Enabled",
        "Filter": {"Prefix": args.prefix},
        "Expiration": {"Days": args.days},
        # also age out any interrupted multipart uploads under the prefix
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 2},
    }
    rules = kept + [ours]
    s3.put_bucket_lifecycle_configuration(
        Bucket=args.bucket, LifecycleConfiguration={"Rules": rules})
    print(f"[lifecycle] applied '{RULE_ID}': Expire {args.days}d on '{args.prefix}' "
          f"(kept {len(kept)} other rule(s)).")
    print(f"[lifecycle] rules now: {[r.get('ID') for r in get_rules(s3, args.bucket)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
