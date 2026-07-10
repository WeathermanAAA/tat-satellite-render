"""Writer side of the cross-repo edge contract (Stage-0 gate companion).

The deployed cyclolab-router Worker (v8b77a818) reads bucket
triple-a-tropics-media at cyclolab/{sid}/index.html expecting text/html.
These tests pin the writer to those exact strings; the main repo's
tests/test_cyclolab_router.py pins the Worker's resolve() to the same.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyclolab_pages import adv_key, page_key, page_url_path  # noqa: E402
from storm_ids import InvestSidError  # noqa: E402


class TestEdgeContract(unittest.TestCase):

    def test_page_key_exact_shape(self):
        self.assertEqual(page_key("NHC_EP012026"),
                         "cyclolab/NHC_EP012026/index.html")
        self.assertEqual(page_key("NHC_AL052026"),
                         "cyclolab/NHC_AL052026/index.html")

    def test_shadow_prefix_is_unreachable_namespace(self):
        # Shadow writes compose under a prefix the Worker never reads.
        self.assertEqual(page_key("NHC_EP012026", prefix="shadow/cyclolab"),
                         "shadow/cyclolab/NHC_EP012026/index.html")

    def test_adv_key_matches_source_writes(self):
        # Must equal what cyclolab_advisories writes ({prefix}/adv/{sid}.json).
        self.assertEqual(adv_key("NHC_EP012026"),
                         "cyclolab/adv/NHC_EP012026.json")

    def test_public_path(self):
        self.assertEqual(page_url_path("NHC_EP012026"),
                         "/cyclolab/NHC_EP012026/")

    def test_invests_now_get_page_keys(self):
        # Stage C: invests ARE page-able (grey / red-X subset page).
        self.assertEqual(page_key("NHC_EP902026"),
                         "cyclolab/NHC_EP902026/index.html")
        self.assertEqual(page_url_path("NHC_EP902026"), "/cyclolab/NHC_EP902026/")

    def test_write_html_content_type_and_default_bucket(self):
        # R2Sink.write_html must PUT text/html to the Worker's bucket.
        import intensity_poller as ip
        with mock.patch("boto3.client") as mc:
            sink = ip.R2Sink()
            self.assertEqual(sink.bucket, "triple-a-tropics-media")
            sink.write_html("cyclolab/NHC_EP012026/index.html", "<html/>")
            kwargs = mc.return_value.put_object.call_args.kwargs
            self.assertEqual(kwargs["Key"], "cyclolab/NHC_EP012026/index.html")
            self.assertEqual(kwargs["ContentType"], "text/html; charset=utf-8")
            self.assertEqual(kwargs["Bucket"], "triple-a-tropics-media")


if __name__ == "__main__":
    unittest.main(verbosity=2)
