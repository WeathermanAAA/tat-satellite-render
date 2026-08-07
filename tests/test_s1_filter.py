#!/usr/bin/env python3
"""The generated SNS MessageBody filter policies vs CAPTURED real NOAA traffic.

Proves each source's band-tight wildcard policy (s1_sources.filter_policy)
passes exactly what the worker keeps (s1_sources.is_ours) and rejects the
rest -- the deterministic half of the §3.5 acceptance check (a silent no-op,
a mis-scoped policy, or an unimplemented operator fails here). Also locks the
policy-generator/matcher pairing: filter_policy may only use operators
s1_filter_check implements, or acceptance part C goes blind."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "infra"))
sys.path.insert(0, ROOT)
import s1_filter_check as F  # noqa: E402
import s1_sources as SRC  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "s1", "noaa_goes19_firehose_raw.json")

G19 = SRC.get_source("goes19")
G18 = SRC.get_source("goes18")
H9 = SRC.get_source("himawari9")
POLICY_G19 = SRC.filter_policy(G19)


def _bodies():
    raw = json.load(open(FIXTURE))
    return [json.loads(m) if isinstance(m, str) else m for m in raw]


def _body(key):
    return {"Records": [{"s3": {"object": {"key": key}}}]}


C13_KEY = ("ABI-L2-CMIPM/2026/169/21/"
           "OR_ABI-L2-CMIPM2-M6C13_G19_s20261692100572_e_c.nc")


class TestWildcardMatcher(unittest.TestCase):
    """AWS SNS wildcard semantics: '*' = any run (incl. empty), pattern
    anchored at BOTH ends, no escape syntax."""

    def test_anchored_both_ends(self):
        self.assertTrue(F._wildcard_match("a*c", "abc"))
        self.assertTrue(F._wildcard_match("a*c", "ac"))        # empty run
        self.assertFalse(F._wildcard_match("a*c", "abcd"))     # tail anchored
        self.assertFalse(F._wildcard_match("a*c", "xabc"))     # head anchored

    def test_no_star_is_exact(self):
        self.assertTrue(F._wildcard_match("abc", "abc"))
        self.assertFalse(F._wildcard_match("abc", "abcd"))

    def test_head_tail_may_not_overlap(self):
        # "ab*ba" must NOT match "aba" (the b would be shared).
        self.assertFalse(F._wildcard_match("ab*ba", "aba"))
        self.assertTrue(F._wildcard_match("ab*ba", "abba"))

    def test_middle_segments_in_order(self):
        self.assertTrue(F._wildcard_match("a*b*c", "a1b2c"))
        self.assertFalse(F._wildcard_match("a*b*c", "acb"))    # out of order
        # A middle segment may not borrow from the anchored tail.
        self.assertFalse(F._wildcard_match("a*bc*bc", "abc"))
        self.assertTrue(F._wildcard_match("a*bc*bc", "abcbc"))


class TestFilterPolicy(unittest.TestCase):
    def test_scope_is_nested_body_path_wildcard(self):
        # The generated policy must be the nested body-path shape (a flat
        # attribute policy would be the INGEST-1 bug) with a wildcard leaf.
        for src in (G19, G18, H9):
            pol = SRC.filter_policy(src)
            ops = pol["Records"]["s3"]["object"]["key"]
            self.assertEqual(len(ops), 1)
            self.assertIn("wildcard", ops[0])
            # AWS caps wildcards at 3 stars per pattern.
            self.assertLessEqual(ops[0]["wildcard"].count("*"), 3)

    def test_policy_uses_only_operators_the_matcher_implements(self):
        # The pairing lock: a policy operator the local matcher doesn't know
        # falls through to False and turns acceptance part C into a false
        # MISMATCH (or worse, validates a match-nothing policy).
        implemented = {"prefix", "wildcard", "anything-but", "exists"}
        for src in (G19, G18, H9):
            for op in SRC.filter_policy(src)["Records"]["s3"]["object"]["key"]:
                if isinstance(op, dict):
                    self.assertTrue(set(op) & implemented,
                                    f"unimplemented operator in {op}")

    def test_matches_agree_with_is_ours_on_real_traffic(self):
        bodies = _bodies()
        self.assertGreaterEqual(len(bodies), 5, "fixture should hold real traffic")
        for b in bodies:
            key = F.object_key(b) or ""
            want = SRC.is_ours(G19, SRC.parse(G19, key))
            self.assertEqual(F.matches(b, POLICY_G19), want, f"mismatch on {key}")

    def test_synthetic_cmipm2_c13_passes(self):
        self.assertTrue(F.matches(_body(C13_KEY), POLICY_G19))

    def test_scan_mode_not_hardcoded(self):
        # M6 -> M3: if ABI leaves mode 6 the filter must keep matching
        # (pinning M6 would silently zero the SQS path).
        m3 = C13_KEY.replace("-CMIPM2-M6", "-CMIPM2-M3")
        self.assertTrue(F.matches(_body(m3), POLICY_G19))

    def test_rejects_other_bands_sectors_sats_products(self):
        for key in (C13_KEY.replace("C13", "C10"),          # other band
                    C13_KEY.replace("CMIPM2", "CMIPM1"),    # meso-1 sector
                    C13_KEY.replace("_G19_", "_G18_"),      # other sat
                    "ABI-L1b-RadC/2026/169/21/x.nc",
                    "ABI-L2-MCMIPM/2026/169/21/x.nc",       # MCMIP != CMIPM
                    "GLM-L2-LCFA/2026/169/21/x.nc"):
            self.assertFalse(F.matches(_body(key), POLICY_G19), key)

    def test_ahi_policy_passes_b13_only_all_segments(self):
        pol = SRC.filter_policy(H9)
        for seg in (1, 5, 10):
            key = (f"AHI-L1b-FLDK/2026/06/19/1850/"
                   f"HS_H09_20260619_1850_B13_FLDK_R20_S{seg:02d}10.DAT.bz2")
            self.assertTrue(F.matches(_body(key), pol), key)
        for key in ("AHI-L1b-FLDK/2026/06/19/1850/"
                    "HS_H09_20260619_1850_B08_FLDK_R20_S0110.DAT.bz2",  # band
                    "AHI-L1b-Target/2026/06/19/1850/"
                    "HS_H09_20260619_1850_B13_R302_S0101.DAT.bz2"):     # sector
            self.assertFalse(F.matches(_body(key), pol), key)

    def test_array_any_element_semantics(self):
        # SNS: an array property matches if ANY element matches.
        body = {"Records": [
            {"s3": {"object": {"key": "ABI-L1b-RadC/x.nc"}}},
            {"s3": {"object": {"key": C13_KEY}}},
        ]}
        self.assertTrue(F.matches(body, POLICY_G19))

    def test_checker_cli_agrees_for_every_source(self):
        # The acceptance check's part C, exactly as the script runs it.
        import subprocess
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, "infra", "s1_filter_check.py")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
