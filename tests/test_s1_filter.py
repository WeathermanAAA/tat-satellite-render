#!/usr/bin/env python3
"""The committed SNS MessageBody filter policy vs CAPTURED real NOAA traffic.

Proves the body-path filter (Records[].s3.object.key prefix ABI-L2-CMIPM/)
passes exactly the CMIPM keys and rejects the rest -- the deterministic half of
the §3.5 acceptance check (a silent no-op or a mis-scoped policy fails here)."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, ROOT)
import s1_filter_check as F  # noqa: E402

POLICY = json.load(open(os.path.join(ROOT, "infra", "s1_filter_policy.json")))
FIXTURE = os.path.join(HERE, "fixtures", "s1", "noaa_goes19_firehose_raw.json")


def _bodies():
    raw = json.load(open(FIXTURE))
    return [json.loads(m) if isinstance(m, str) else m for m in raw]


class TestFilterPolicy(unittest.TestCase):
    def test_scope_is_message_body_in_committed_iac(self):
        # The IaC must set MessageBody scope; assert the readable policy is the
        # nested body-path shape (a flat attribute policy would be the bug).
        self.assertIn("Records", POLICY)
        self.assertEqual(POLICY["Records"]["s3"]["object"]["key"],
                         [{"prefix": "ABI-L2-CMIPM/"}])

    def test_matches_agree_with_real_keys(self):
        bodies = _bodies()
        self.assertGreaterEqual(len(bodies), 5, "fixture should hold real traffic")
        saw_cmipm = saw_other = False
        for b in bodies:
            key = F.object_key(b) or ""
            want = key.startswith("ABI-L2-CMIPM/")
            self.assertEqual(F.matches(b, POLICY), want, f"mismatch on {key}")
            saw_cmipm = saw_cmipm or want
            saw_other = saw_other or not want
        # The fixture must exercise BOTH branches (else the test proves nothing).
        self.assertTrue(saw_cmipm, "fixture has no CMIPM key")
        self.assertTrue(saw_other, "fixture has no non-CMIPM key to reject")

    def test_synthetic_cmipm2_c13_passes(self):
        body = {"Records": [{"s3": {"object": {"key":
                "ABI-L2-CMIPM/2026/169/21/OR_ABI-L2-CMIPM2-M6C13_G19_s20261692100572_e_c.nc"}}}]}
        self.assertTrue(F.matches(body, POLICY))

    def test_non_cmipm_rejected(self):
        for key in ("ABI-L1b-RadC/2026/169/21/x.nc",
                    "ABI-L2-MCMIPM/2026/169/21/x.nc",   # MCMIP is NOT our CMIPM prefix
                    "GLM-L2-LCFA/2026/169/21/x.nc"):
            body = {"Records": [{"s3": {"object": {"key": key}}}]}
            self.assertFalse(F.matches(body, POLICY), key)

    def test_array_any_element_semantics(self):
        # SNS: an array property matches if ANY element matches.
        body = {"Records": [
            {"s3": {"object": {"key": "ABI-L1b-RadC/x.nc"}}},
            {"s3": {"object": {"key": "ABI-L2-CMIPM/y.nc"}}},
        ]}
        self.assertTrue(F.matches(body, POLICY))


if __name__ == "__main__":
    unittest.main()
