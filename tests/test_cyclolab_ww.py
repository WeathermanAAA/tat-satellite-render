"""Phase 4: coastal watches/warnings overlay — parser + advisory-JSON wiring.

The official NHC ``windWatchesWarnings`` KMZ (one Placemark/LineString per
watch/warning, connecting the breakpoints) is parsed into a ``ww`` array of
``{type, geometry}`` on the SAME adv/{sid}.json that carries the cone. It is
ADDITIVE + GRACEFUL: a missing or malformed WW layer degrades to ``[]`` and
NEVER breaks the cone the page needs. Fixture = the real AL012026 advisory-1
WW KMZ (a single Tropical Storm Watch along the upper Texas coast).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kml_advisories as K  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "cyclolab"
WW = (FIX / "AL012026_001adv_WW.kmz").read_bytes()
CONE = (FIX / "EP012026_013adv_CONE.kmz").read_bytes()
TRACK = (FIX / "EP012026_013adv_TRACK.kmz").read_bytes()
ALERTS = json.loads((FIX / "nws_tc_alerts_sample.json").read_text())


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


class ParseZonesTests(unittest.TestCase):
    """INLAND county/zone FILLS from the NWS alerts API (Phase 4 follow-up).
    Fixture = a REAL api.weather.gov/alerts/active capture (2026-06-16) of a
    Gulf TC: TXZ/LAZ land forecast-zones carry embedded polygons, plus a marine
    GMZ feature with no geometry that must be excluded (inland fills only)."""

    def test_land_zones_only_marine_excluded(self):
        z = K.parse_nws_alert_zones(ALERTS)
        # 4 land features (2 TS Warning LAZ + 2 TS Watch TXZ); the GMZ marine
        # feature (null geometry) is dropped.
        self.assertEqual(len(z), 4)
        self.assertEqual({e["type"] for e in z}, {"TS_WARNING", "TS_WATCH"})
        for e in z:
            self.assertGreaterEqual(len(e["geometry"]), 3)
            self.assertEqual(e["geometry"][0], e["geometry"][-1])   # closed ring
            self.assertFalse(e["ugc"].startswith("GM"))             # not marine
            for lon, lat in e["geometry"]:
                self.assertTrue(-180 <= lon <= 180 and -90 <= lat <= 90)

    def test_event_to_type_mapping(self):
        self.assertEqual(K._NWS_EVENT_TO_TYPE["hurricane warning"], "HU_WARNING")
        self.assertEqual(K._NWS_EVENT_TO_TYPE["storm surge watch"], "SS_WATCH")
        # a non-TC event is ignored entirely.
        other = {"features": [{"geometry": {"type": "Polygon",
                 "coordinates": [[[-80, 25], [-80.1, 25.1], [-80.2, 25], [-80, 25]]]},
                 "properties": {"event": "Flood Warning",
                                "geocode": {"UGC": ["FLC086"]}}}]}
        self.assertEqual(K.parse_nws_alert_zones(other), [])

    def test_cone_bbox_attribution(self):
        gulf = (-96.0, 28.0, -91.0, 31.0)
        far = (-70.0, 40.0, -69.0, 41.0)
        self.assertEqual(len(K.parse_nws_alert_zones(ALERTS, cone_box=gulf,
                                                     margin_deg=1.0)), 4)
        self.assertEqual(len(K.parse_nws_alert_zones(ALERTS, cone_box=far,
                                                     margin_deg=1.0)), 0)

    def test_zone_resolver_fallback_for_null_geometry(self):
        # When a LAND feature carries no embedded geometry, the affected-zone
        # URLs are resolved via the injected resolver (the caller caches them).
        synth = {"features": [{"geometry": None, "properties": {
            "event": "Hurricane Warning", "geocode": {"UGC": ["FLC086"]},
            "affectedZones": ["https://api.weather.gov/zones/county/FLC086"]}}]}

        def resolver(url):
            self.assertIn("FLC086", url)
            return {"type": "Polygon", "coordinates":
                    [[[-80, 25], [-80.1, 25.1], [-80.2, 25], [-80, 25]]]}

        z = K.parse_nws_alert_zones(synth, resolve_zone=resolver)
        self.assertEqual(len(z), 1)
        self.assertEqual(z[0]["type"], "HU_WARNING")

    def test_marine_zone_url_not_resolved(self):
        # A marine affectedZone URL must not be resolved even on the fallback.
        synth = {"features": [{"geometry": None, "properties": {
            "event": "Tropical Storm Warning", "geocode": {"UGC": ["GMZ355"]},
            "affectedZones": ["https://api.weather.gov/zones/forecast/GMZ355"]}}]}
        called = []
        K.parse_nws_alert_zones(
            synth, resolve_zone=lambda u: called.append(u))
        self.assertEqual(called, [])           # marine-only -> skipped entirely

    def test_one_bad_feature_never_sinks_the_rest(self):
        bad = {"features": [
            {"properties": None},                      # malformed
            ALERTS["features"][0],                      # a good land feature
        ]}
        z = K.parse_nws_alert_zones(bad)
        self.assertEqual(len(z), 1)

    def test_rdp_simplifies_but_preserves_endpoints(self):
        ring = ([[0.0, 0.0]] +
                [[i * 0.001, 0.00005 * (i % 2)] for i in range(1, 150)] +
                [[0.15, 0.0], [0.0, 0.0]])
        simp = K._rdp(ring, 0.01)
        self.assertLess(len(simp), len(ring))
        self.assertEqual(simp[0], ring[0])
        self.assertEqual(simp[-1], ring[-1])


if __name__ == "__main__":
    unittest.main()
