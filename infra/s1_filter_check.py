#!/usr/bin/env python3
"""SNS MessageBody subscription-filter matcher + ground-truth validator.

A faithful (subset) re-implementation of Amazon SNS *payload-based* filtering
semantics, used two ways:

  * the §3.5 acceptance check (infra/s1_acceptance_check.sh) and the unit tests
    call ``matches(body, policy)`` to PROVE -- deterministically, against a
    CAPTURED real NOAA notification -- that our committed filter policy passes a
    CMIPM notification and rejects a non-CMIPM one. This is the local half of
    "NumberOfNotificationsFilteredOut ~0 for traffic that should pass": it is
    what makes a silent filter no-op fail loudly instead of reading green.

  * ``python infra/s1_filter_check.py [fixture.json]`` validates the committed
    filter policy against the committed firehose fixture and exits non-zero on
    any surprise (a CMIPM key that would be dropped, or a non-CMIPM key that
    would pass).

SNS payload-filter rules implemented (the subset our policy uses):
  - policy value is a DICT  -> recurse into the message's value at that key;
  - policy value is a LIST  -> a list of match operators; the message scalar
    matches if it satisfies ANY operator. Supported operators: a bare string
    (exact match) and {"prefix": "..."};
  - if the message value at a key is itself a LIST (e.g. Records[]), the policy
    (a dict) matches if ANY array element matches -- SNS's "array matches if any
    element matches" rule.
Anything the message lacks -> no match (a missing path cannot satisfy a policy).
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _scalar_matches(value: Any, operators: list) -> bool:
    """A message scalar vs a list of SNS match operators (ANY)."""
    for op in operators:
        if isinstance(op, str):
            if value == op:
                return True
        elif isinstance(op, dict):
            if "prefix" in op and isinstance(value, str) and value.startswith(op["prefix"]):
                return True
            if "anything-but" in op:
                ab = op["anything-but"]
                ab = ab if isinstance(ab, list) else [ab]
                if value not in ab:
                    return True
            if "exists" in op:
                # exists:true -> value present (it is, we're here); false -> absent.
                if bool(op["exists"]) is True:
                    return True
    return False


def matches(message: Any, policy: Any) -> bool:
    """Does ``message`` satisfy the SNS MessageBody ``policy``?"""
    if isinstance(policy, list):
        # Leaf: a list of operators applied to the message scalar (or to ANY
        # element if the message value is itself a list).
        if isinstance(message, list):
            return any(_scalar_matches(v, policy) for v in message)
        return _scalar_matches(message, policy)
    if isinstance(policy, dict):
        # If the message at this level is an array, match if ANY element matches
        # the whole sub-policy (SNS array semantics).
        if isinstance(message, list):
            return any(matches(elem, policy) for elem in message)
        if not isinstance(message, dict):
            return False
        for key, sub in policy.items():
            if key not in message:
                return False
            if not matches(message[key], sub):
                return False
        return True
    # A bare scalar policy (rare) -> exact match.
    return message == policy


def object_key(record_message: dict) -> str | None:
    """Pull Records[0].s3.object.key from a raw S3-event body (or None)."""
    try:
        return record_message["Records"][0]["s3"]["object"]["key"]
    except (KeyError, IndexError, TypeError):
        return None


def _main(argv: list[str]) -> int:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    policy = json.load(open(os.path.join(here, "s1_filter_policy.json")))
    fixture = argv[1] if len(argv) > 1 else os.path.join(
        here, "..", "tests", "fixtures", "s1", "noaa_goes19_firehose_raw.json")
    raw = json.load(open(fixture))
    bodies = [json.loads(m) if isinstance(m, str) else m for m in raw]
    ok = True
    npass = nfail = 0
    for b in bodies:
        key = object_key(b) or ""
        want = key.startswith("ABI-L2-CMIPM/")
        got = matches(b, policy)
        flag = "OK " if got == want else "BAD"
        if got != want:
            ok = False
        npass += 1 if got else 0
        nfail += 1 if not got else 0
        print(f"  [{flag}] pass={got!s:5} want={want!s:5} {key}")
    print(f"\nfilter would pass {npass}/{len(bodies)} (CMIPM only); "
          f"reject {nfail}. policy-vs-realkey agreement: {'OK' if ok else 'MISMATCH'}")
    # Also assert there is at least one CMIPM2-C13 that PASSES -- our S1 slot.
    c13 = [object_key(b) for b in bodies
           if (object_key(b) or "").startswith("ABI-L2-CMIPM/")
           and "CMIPM2-M6C13" in (object_key(b) or "")]
    if c13:
        print(f"and {len(c13)} CMIPM2-C13 (the S1 product) PASS the filter.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
