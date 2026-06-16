"""Phase 4: coastal watches/warnings overlay — parser + advisory-JSON wiring.

The official NHC ``windWatchesWarnings`` KMZ (one Placemark/LineString per
watch/warning, connecting the breakpoints) is parsed into a ``ww`` array of
``{type, geometry}`` on the SAME adv/{sid}.json that carries the cone. It is
ADDITIVE + GRACEFUL: a missing or malformed WW layer degrades to ``[]`` and
NEVER breaks the cone the page needs. Fixture = the real AL012026 advisory-1
WW KMZ (a single Tropical Storm Watch along the upper Texas coast).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kml_advisories as K  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "cyclolab"
WW = (FIX / "AL012026_001adv_WW.kmz").read_bytes()
CONE = (FIX / "EP012026_013adv_CONE.kmz").read_bytes()
TRACK = (FIX / "EP012026_013adv_TRACK.kmz").read_bytes()


class ParseWWTests(unittest.TestCase):
    def test_real_ww_kmz_one_ts_watch(self):
        segs = K.parse_ww_kmz(WW)
        self.assertEqual(len(segs), 1)
        seg = segs[0]
        self.assertEqual(seg["type"], "TS_WATCH")
        self.assertGreaterEqual(len(seg["geometry"]), 2)
        # geometry is [[lon, lat], ...] in range, along the TX/LA coast.
        for lon, lat in seg["geometry"]:
            self.assertTrue(-180 <= lon <= 180 and -90 <= lat <= 90)
        self.assertAlmostEqual(seg["geometry"][0][0], -95.6, places=2)

    def test_type_mapping_by_name_and_style(self):
        self.assertEqual(K._ww_type("Tropical Storm Warning", "#TWR"), "TS_WARNING")
        self.assertEqual(K._ww_type("Hurricane Watch", "#HWA"), "HU_WATCH")
        self.assertEqual(K._ww_type("Hurricane Warning", "#HWR"), "HU_WARNING")
        self.assertEqual(K._ww_type("Storm Surge Warning", ""), "SS_WARNING")
        # styleUrl fallback when the name is absent/unknown.
        self.assertEqual(K._ww_type("", "#TWA"), "TS_WATCH")
        # an unknown segment keeps its (cleaned) name -> never silently dropped.
        self.assertEqual(K._ww_type("Some New Advisory", "#ZZ"),
                         "SOME NEW ADVISORY")

    def test_malformed_raises_for_caller_to_catch(self):
        with self.assertRaises(K.AdvisoryParseError):
            K.parse_ww_kmz(b"not a zip")


class BuildAdvisoryWWTests(unittest.TestCase):
    def test_ww_present_when_kmz_given(self):
        adv = K.build_advisory_json("NHC_EP012026", CONE, TRACK,
                                    ww_kmz_bytes=WW)
        self.assertIn("ww", adv)
        self.assertEqual(len(adv["ww"]), 1)
        self.assertEqual(adv["ww"][0]["type"], "TS_WATCH")
        # provenance records the WW bytes when present.
        self.assertIn("ww_sha256", adv["provenance"])
        # the cone is intact (the overlay never displaces the cone contract).
        self.assertGreaterEqual(len(adv["cone"]), 1000)
        self.assertEqual(adv["cone"][0], adv["cone"][-1])

    def test_ww_empty_when_no_kmz(self):
        adv = K.build_advisory_json("NHC_EP012026", CONE, TRACK)
        self.assertEqual(adv["ww"], [])
        self.assertNotIn("ww_sha256", adv["provenance"])

    def test_malformed_ww_degrades_gracefully(self):
        # A malformed WW layer must NOT raise here and must NOT break the cone.
        adv = K.build_advisory_json("NHC_EP012026", CONE, TRACK,
                                    ww_kmz_bytes=b"not a zip")
        self.assertEqual(adv["ww"], [])
        self.assertGreaterEqual(len(adv["cone"]), 1000)   # cone still written


if __name__ == "__main__":
    unittest.main()
