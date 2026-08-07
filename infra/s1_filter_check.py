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
    (exact match), {"prefix": "..."}, and {"wildcard": "..."} (the band-tight
    policies -- see s1_sources.filter_policy);
  - if the message value at a key is itself a LIST (e.g. Records[]), the policy
    (a dict) matches if ANY array element matches -- SNS's "array matches if any
    element matches" rule.
Anything the message lacks -> no match (a missing path cannot satisfy a policy).

OPERATOR-COVERAGE TRAP: an operator this matcher does not implement falls
through every branch and silently returns False -- exactly how adopting the
wildcard policy would have turned acceptance-check part C into a false
MISMATCH. If s1_sources.filter_policy ever grows a new operator, add it HERE
in the same commit (test_s1_filter.py locks the pairing).
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _wildcard_match(pattern: str, value: str) -> bool:
    """AWS SNS ``wildcard`` semantics: ``*`` matches any character run
    (including empty); the pattern is anchored at BOTH ends (a whole-string
    match, not a search); there is no escape syntax."""
    parts = pattern.split("*")
    if len(parts) == 1:
        return value == pattern
    head, tail = parts[0], parts[-1]
    # The anchored head and tail must fit without overlapping each other.
    if len(value) < len(head) + len(tail):
        return False
    if not value.startswith(head) or not value.endswith(tail):
        return False
    pos, end_limit = len(head), len(value) - len(tail)
    for seg in (p for p in parts[1:-1] if p):
        i = value.find(seg, pos, end_limit)
        if i < 0:
            return False
        pos = i + len(seg)
    return True


def _scalar_matches(value: Any, operators: list) -> bool:
    """A message scalar vs a list of SNS match operators (ANY)."""
    for op in operators:
        if isinstance(op, str):
            if value == op:
                return True
        elif isinstance(op, dict):
            if "prefix" in op and isinstance(value, str) and value.startswith(op["prefix"]):
                return True
            if ("wildcard" in op and isinstance(value, str)
                    and _wildcard_match(op["wildcard"], value)):
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


def _source_fixture_bodies(source_key: str, here: str) -> list[dict]:
    """Per-source validation bodies. goes19: the CAPTURED real firehose batch.
    goes18: the same capture with the sat token swapped (the key grammar is
    identical -- only G19->G18 differs). himawari9: synthesized from the real
    AHI FLDK key grammar, every band x two segments (no captured AHI batch)."""
    import os
    if source_key in ("goes19", "goes18"):
        fixture = os.path.join(here, "..", "tests", "fixtures", "s1",
                               "noaa_goes19_firehose_raw.json")
        raw = json.load(open(fixture))
        bodies = [json.loads(m) if isinstance(m, str) else m for m in raw]
        if source_key == "goes18":
            bodies = [json.loads(json.dumps(b).replace("_G19_", "_G18_"))
                      for b in bodies]
        # The captured batch happens to hold NO C13 -- without positives a
        # match-nothing filter validates silently (the trap this checker
        # exists to catch). Synthesize them from a REAL captured CMIPM2 key
        # so the grammar stays authentic: the C13 twin (the S1 product), an
        # M3 scan-mode twin (the filter must NOT pin M6), and a wrong-sat
        # C13 (must reject).
        base = next(k for k in (object_key(b) for b in bodies)
                    if k and "-CMIPM2-M6C" in k)
        other = "_G18_" if source_key == "goes19" else "_G19_"
        mine = "_G18_" if source_key == "goes18" else "_G19_"
        import re
        c13 = re.sub(r"(-CMIPM2-M\d)C\d\d", r"\1C13", base)
        for extra in (c13,
                      c13.replace("-CMIPM2-M6", "-CMIPM2-M3"),
                      c13.replace(mine, other)):
            bodies.append({"Records": [{"s3": {"object": {"key": extra}}}]})
        return bodies
    keys = [(f"AHI-L1b-FLDK/2026/06/19/1850/"
             f"HS_H09_20260619_1850_B{band:02d}_FLDK_R20_S{seg:02d}10.DAT.bz2")
            for band in range(1, 17) for seg in (1, 10)]
    keys.append("AHI-L1b-Target/2026/06/19/1850/"
                "HS_H09_20260619_1850_B13_R302_S0101.DAT.bz2")  # non-FLDK
    return [{"Records": [{"s3": {"object": {"key": k}}}]} for k in keys]


def _main(argv: list[str]) -> int:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    import s1_sources as SRC
    source_keys = ([argv[argv.index("--source") + 1]] if "--source" in argv
                   else sorted(SRC.SOURCES))
    ok = True
    for skey in source_keys:
        source = SRC.get_source(skey)
        policy = SRC.filter_policy(source)
        print(f"== {skey}  policy={json.dumps(policy, separators=(',', ':'))}")
        npass = nfail = 0
        for b in _source_fixture_bodies(skey, here):
            key = object_key(b) or ""
            # Ground truth = the worker's own parsed-key predicate, so the
            # filter provably passes exactly the product the worker keeps
            # (no more hardcoded prefix drifting from what "ours" means).
            want = SRC.is_ours(source, SRC.parse(source, key))
            got = matches(b, policy)
            flag = "OK " if got == want else "BAD"
            if got != want:
                ok = False
            npass += 1 if got else 0
            nfail += 1 if not got else 0
            print(f"  [{flag}] pass={got!s:5} want={want!s:5} {key}")
        print(f"  filter passes {npass}, rejects {nfail} "
              f"(ground truth: s1_sources.is_ours)")
        if not npass:
            print(f"  !! FAIL: zero positive matches for {skey} -- an "
                  "over-narrow filter would starve the SQS path to backfill.")
            ok = False
    print(f"\npolicy-vs-is_ours agreement: {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
