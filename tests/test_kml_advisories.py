"""Parser tests for kml_advisories against AMANDA's REAL adv #13 artifacts.

Fixtures in ``tests/fixtures/cyclolab/`` are the genuine NHC GIS products
for EP012026 advisory #13 (cone + track KMZ). amanda_parsed_proposal.json
is a prior agent's parse - used here as a CROSS-REFERENCE for the
load-bearing facts (advisory 13, 9 points, the tau list, first-point
intensity, dev labels, cone vertex count), not as a byte-for-byte oracle
(its valid_utc rounding is lossy; our contract derives valid_utc =
issued_utc + tau, which is deterministic).

Runnable via ``python -m pytest tests/test_kml_advisories.py -q`` AND
``python -m unittest tests.test_kml_advisories``.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kml_advisories import (  # noqa: E402
    AdvisoryParseError,
    build_advisory_json,
    parse_cone_kmz,
    parse_track_kmz,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "cyclolab"
CONE_KMZ = FIX / "EP012026_013adv_CONE.kmz"
TRACK_KMZ = FIX / "EP012026_013adv_TRACK.kmz"
PROPOSAL = FIX / "amanda_parsed_proposal.json"

EXPECTED_TAUS = [0, 12, 24, 36, 48, 60, 72, 96, 120]
SID = "NHC_EP012026"
TEXT_URLS = {
    "tcp_url": "https://www.nhc.noaa.gov/text/MIATCPEP1.shtml",
    "tcd_url": "https://www.nhc.noaa.gov/text/MIATCDEP1.shtml",
}


def _read(p: Path) -> bytes:
    return p.read_bytes()


def _make_kmz(kml_text: str, name: str = "doc.kml") -> bytes:
    """Build a tiny in-memory KMZ carrying one kml member."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, kml_text)
    return buf.getvalue()


# A minimal but valid cone KML (square ring, namespaced).
_GOOD_CONE_KML = """<?xml version='1.0' encoding='UTF-8'?>
<kml xmlns='http://earth.google.com/kml/2.1'><Document><Placemark>
<Polygon><outerBoundaryIs><LinearRing><coordinates>
-140.0,8.0,0 -139.0,8.0,0 -139.0,9.0,0 -140.0,9.0,0 -140.0,8.0,0
</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark></Document></kml>"""

# Same ring but NOT closed (last != first) - parser must close it.
_OPEN_CONE_KML = """<?xml version='1.0' encoding='UTF-8'?>
<kml xmlns='http://earth.google.com/kml/2.1'><Document><Placemark>
<Polygon><outerBoundaryIs><LinearRing><coordinates>
-140.0,8.0,0 -139.0,8.0,0 -139.0,9.0,0 -140.0,9.0,0
</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark></Document></kml>"""


class TestConeParser(unittest.TestCase):

    def setUp(self):
        self.ring = parse_cone_kmz(_read(CONE_KMZ))

    def test_real_cone_is_large_closed_ring(self):
        # The real ring is ~1,272 vertices (design §8.3 / proposal).
        self.assertGreaterEqual(len(self.ring), 1000)
        self.assertEqual(self.ring[0], self.ring[-1])     # closed

    def test_real_cone_vertex_count_matches_proposal(self):
        proposal = json.loads(PROPOSAL.read_text())
        self.assertEqual(len(self.ring), proposal["cone_vertex_count"])

    def test_real_cone_lonlat_sane_for_ep(self):
        lons = [p[0] for p in self.ring]
        lats = [p[1] for p in self.ring]
        # Eastern North Pacific, well west of Mexico, north of the equator.
        self.assertTrue(all(-160.0 < x < -120.0 for x in lons))
        self.assertTrue(all(0.0 < y < 30.0 for y in lats))

    def test_lon_first_ordering(self):
        # Cone coordinates are lon,lat - lon must be the negative one here.
        self.assertLess(self.ring[0][0], 0)              # lon (~ -140)
        self.assertGreater(self.ring[0][1], 0)           # lat (~ +8)

    def test_synthetic_square_ring(self):
        ring = parse_cone_kmz(_make_kmz(_GOOD_CONE_KML))
        self.assertEqual(ring[0], ring[-1])
        self.assertEqual(len(ring), 5)

    def test_open_ring_is_closed(self):
        ring = parse_cone_kmz(_make_kmz(_OPEN_CONE_KML))
        self.assertEqual(ring[0], ring[-1])
        # 4 distinct + 1 closing vertex.
        self.assertEqual(len(ring), 5)


class TestConeRejections(unittest.TestCase):

    def test_truncated_zip_raises(self):
        good = _read(CONE_KMZ)
        with self.assertRaises(AdvisoryParseError):
            parse_cone_kmz(good[: len(good) // 2])

    def test_not_bytes_raises(self):
        with self.assertRaises(AdvisoryParseError):
            parse_cone_kmz("not bytes")          # type: ignore[arg-type]

    def test_kml_without_polygon_raises(self):
        kml = ("<kml xmlns='http://earth.google.com/kml/2.1'><Document>"
               "<Placemark><Point><coordinates>-140,8,0</coordinates>"
               "</Point></Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            parse_cone_kmz(_make_kmz(kml))

    def test_empty_coordinates_raises(self):
        kml = ("<kml xmlns='http://earth.google.com/kml/2.1'><Document>"
               "<Placemark><Polygon><outerBoundaryIs><LinearRing>"
               "<coordinates>   </coordinates></LinearRing>"
               "</outerBoundaryIs></Polygon></Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            parse_cone_kmz(_make_kmz(kml))

    def test_too_few_vertices_raises(self):
        kml = ("<kml xmlns='http://earth.google.com/kml/2.1'><Document>"
               "<Placemark><Polygon><outerBoundaryIs><LinearRing>"
               "<coordinates>-140,8,0 -139,8,0</coordinates></LinearRing>"
               "</outerBoundaryIs></Polygon></Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            parse_cone_kmz(_make_kmz(kml))

    def test_out_of_range_lat_raises(self):
        kml = ("<kml xmlns='http://earth.google.com/kml/2.1'><Document>"
               "<Placemark><Polygon><outerBoundaryIs><LinearRing>"
               "<coordinates>-140,8,0 -139,8,0 -139,999,0 -140,9,0 -140,8,0"
               "</coordinates></LinearRing></outerBoundaryIs></Polygon>"
               "</Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            parse_cone_kmz(_make_kmz(kml))

    def test_malformed_xml_raises(self):
        with self.assertRaises(AdvisoryParseError):
            parse_cone_kmz(_make_kmz("<kml><Document><unclosed></kml>"))

    def test_zip_without_kml_member_raises(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "no kml here")
        with self.assertRaises(AdvisoryParseError):
            parse_cone_kmz(buf.getvalue())


class TestTrackParser(unittest.TestCase):

    def setUp(self):
        self.track = parse_track_kmz(_read(TRACK_KMZ))
        self.proposal = json.loads(PROPOSAL.read_text())

    def test_advisory_number_is_13(self):
        self.assertEqual(self.track["advisory"], 13)
        self.assertEqual(self.track["advisory"],
                         int(self.proposal["advisory"]))

    def test_issued_utc_is_2100z(self):
        self.assertEqual(self.track["issued_utc"], "2026-06-05T21:00:00Z")
        self.assertEqual(self.track["issued_utc"],
                         self.proposal["issued_utc"])

    def test_nine_points(self):
        self.assertEqual(len(self.track["points"]), 9)

    def test_taus_exact(self):
        self.assertEqual([p["tau_h"] for p in self.track["points"]],
                         EXPECTED_TAUS)

    def test_first_point_intensity_matches_proposal(self):
        first = self.track["points"][0]
        self.assertEqual(first["intensity_kt"], 35)
        self.assertEqual(first["intensity_kt"],
                         self.proposal["points"][0]["intensity_kt"])

    def test_first_point_position(self):
        first = self.track["points"][0]
        self.assertAlmostEqual(first["lat"], 13.1)
        self.assertAlmostEqual(first["lon"], -134.1)

    def test_dev_labels_match_proposal(self):
        got = [p["dev_label"] for p in self.track["points"]]
        want = [p["dev_label"] for p in self.proposal["points"]]
        self.assertEqual(got, want)

    def test_intensity_series_matches_proposal(self):
        got = [p["intensity_kt"] for p in self.track["points"]]
        want = [p["intensity_kt"] for p in self.proposal["points"]]
        self.assertEqual(got, want)

    def test_valid_utc_is_issued_plus_tau(self):
        # Our deterministic contract: valid = issued + tau hours.
        from datetime import datetime, timedelta, timezone
        issued = datetime(2026, 6, 5, 21, 0, tzinfo=timezone.utc)
        for p in self.track["points"]:
            want = (issued + timedelta(hours=p["tau_h"])).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            self.assertEqual(p["valid_utc"], want)

    def test_all_points_lonlat_sane_for_ep(self):
        for p in self.track["points"]:
            self.assertTrue(-160.0 < p["lon"] < -120.0)
            self.assertTrue(0.0 < p["lat"] < 30.0)


class TestTrackRejections(unittest.TestCase):

    def test_truncated_zip_raises(self):
        good = _read(TRACK_KMZ)
        with self.assertRaises(AdvisoryParseError):
            parse_track_kmz(good[: len(good) // 2])

    def test_single_point_track_parses_but_fails_build(self):
        # A lone point parses (parser does no >=2 check) - build enforces it.
        kml = ("<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
               "<Placemark><ExtendedData>"
               "<Data name='advisoryNum'><value>13</value></Data>"
               "<Data name='stormType'><value>TS</value></Data>"
               "<Data name='pubAdvTime'>"
               "<value>1100 AM HST Fri Jun 05 2026</value></Data>"
               "</ExtendedData></Placemark>"
               "<Placemark><styleUrl>#initial_point</styleUrl>"
               "<description>Advisory Information Maximum Wind: 35 knots"
               "</description>"
               "<Point><coordinates>-134.1,13.1,0</coordinates></Point>"
               "</Placemark></Document></kml>")
        track = parse_track_kmz(_make_kmz(kml))
        self.assertEqual(len(track["points"]), 1)

    def test_no_points_raises(self):
        kml = ("<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
               "<Placemark><ExtendedData>"
               "<Data name='advisoryNum'><value>13</value></Data>"
               "<Data name='pubAdvTime'>"
               "<value>1100 AM HST Fri Jun 05 2026</value></Data>"
               "</ExtendedData></Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            parse_track_kmz(_make_kmz(kml))

    def test_missing_advisory_num_raises(self):
        kml = ("<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
               "<Placemark><ExtendedData>"
               "<Data name='pubAdvTime'>"
               "<value>1100 AM HST Fri Jun 05 2026</value></Data>"
               "</ExtendedData></Placemark>"
               "<Placemark><styleUrl>#s_point</styleUrl>"
               "<description>12 hr Forecast Maximum Wind: 35 knots"
               "</description>"
               "<Point><coordinates>-134.5,12.9,0</coordinates></Point>"
               "</Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            parse_track_kmz(_make_kmz(kml))

    def test_missing_pub_adv_time_raises(self):
        kml = ("<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
               "<Placemark><ExtendedData>"
               "<Data name='advisoryNum'><value>13</value></Data>"
               "</ExtendedData></Placemark>"
               "<Placemark><styleUrl>#s_point</styleUrl>"
               "<description>12 hr Forecast Maximum Wind: 35 knots"
               "</description>"
               "<Point><coordinates>-134.5,12.9,0</coordinates></Point>"
               "</Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            parse_track_kmz(_make_kmz(kml))

    def test_point_without_forecast_hour_raises(self):
        kml = ("<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
               "<Placemark><ExtendedData>"
               "<Data name='advisoryNum'><value>13</value></Data>"
               "<Data name='pubAdvTime'>"
               "<value>1100 AM HST Fri Jun 05 2026</value></Data>"
               "</ExtendedData></Placemark>"
               "<Placemark><styleUrl>#s_point</styleUrl>"
               "<description>no hour here Maximum Wind: 35 knots"
               "</description>"
               "<Point><coordinates>-134.5,12.9,0</coordinates></Point>"
               "</Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            parse_track_kmz(_make_kmz(kml))


class TestBuildAdvisoryJson(unittest.TestCase):

    def setUp(self):
        self.cone_b = _read(CONE_KMZ)
        self.track_b = _read(TRACK_KMZ)
        self.adv = build_advisory_json(SID, self.cone_b, self.track_b,
                                       text_urls=TEXT_URLS)

    def test_exact_contract_keys(self):
        self.assertEqual(
            set(self.adv.keys()),
            {"sid", "advisory", "issued_utc", "source", "method",
             "cone", "points", "text", "provenance"})

    def test_scalar_fields(self):
        self.assertEqual(self.adv["sid"], SID)
        self.assertEqual(self.adv["advisory"], 13)
        self.assertEqual(self.adv["issued_utc"], "2026-06-05T21:00:00Z")
        self.assertEqual(self.adv["source"], "nhc")
        self.assertEqual(self.adv["method"], "official-cone")

    def test_cone_and_points(self):
        self.assertGreaterEqual(len(self.adv["cone"]), 1000)
        self.assertEqual(self.adv["cone"][0], self.adv["cone"][-1])
        self.assertEqual([p["tau_h"] for p in self.adv["points"]],
                         EXPECTED_TAUS)

    def test_text_block(self):
        self.assertEqual(self.adv["text"], {
            "tcp_url": TEXT_URLS["tcp_url"],
            "tcd_url": TEXT_URLS["tcd_url"],
        })

    def test_text_is_none_without_urls(self):
        adv = build_advisory_json(SID, self.cone_b, self.track_b)
        self.assertIsNone(adv["text"])

    def test_provenance_keys_and_clock_free(self):
        prov = self.adv["provenance"]
        self.assertEqual(
            set(prov.keys()),
            {"cone_sha256", "track_sha256", "cone_bytes", "track_bytes",
             "parsed_utc"})
        self.assertIsNone(prov["parsed_utc"])     # caller stamps the clock
        self.assertEqual(prov["cone_bytes"], len(self.cone_b))
        self.assertEqual(prov["track_bytes"], len(self.track_b))
        self.assertEqual(len(prov["cone_sha256"]), 64)
        self.assertEqual(len(prov["track_sha256"]), 64)

    def test_output_is_json_dumps_able(self):
        s = json.dumps(self.adv)
        round_trip = json.loads(s)
        self.assertEqual(round_trip["advisory"], 13)
        self.assertEqual(len(round_trip["cone"]), len(self.adv["cone"]))

    def test_build_rejects_single_point_track(self):
        # < 2 forecast points must fail the build validation.
        kml = ("<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
               "<Placemark><ExtendedData>"
               "<Data name='advisoryNum'><value>13</value></Data>"
               "<Data name='stormType'><value>TS</value></Data>"
               "<Data name='pubAdvTime'>"
               "<value>1100 AM HST Fri Jun 05 2026</value></Data>"
               "</ExtendedData></Placemark>"
               "<Placemark><styleUrl>#initial_point</styleUrl>"
               "<description>Advisory Information Maximum Wind: 35 knots"
               "</description>"
               "<Point><coordinates>-134.1,13.1,0</coordinates></Point>"
               "</Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            build_advisory_json(SID, self.cone_b, _make_kmz(kml))

    def test_build_rejects_non_monotonic_taus(self):
        # Two points whose taus decrease must fail the build.
        kml = ("<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
               "<Placemark><ExtendedData>"
               "<Data name='advisoryNum'><value>13</value></Data>"
               "<Data name='stormType'><value>TS</value></Data>"
               "<Data name='pubAdvTime'>"
               "<value>1100 AM HST Fri Jun 05 2026</value></Data>"
               "</ExtendedData></Placemark>"
               "<Placemark><styleUrl>#s_point</styleUrl>"
               "<description>24 hr Forecast Maximum Wind: 35 knots"
               "</description>"
               "<Point><coordinates>-134.5,12.9,0</coordinates></Point>"
               "</Placemark>"
               "<Placemark><styleUrl>#d_point</styleUrl>"
               "<description>12 hr Forecast Maximum Wind: 30 knots"
               "</description>"
               "<Point><coordinates>-134.9,12.5,0</coordinates></Point>"
               "</Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            build_advisory_json(SID, self.cone_b, _make_kmz(kml))

    def test_build_rejects_bad_cone(self):
        bad = _make_kmz(
            "<kml xmlns='http://earth.google.com/kml/2.1'><Document>"
            "<Placemark><Point><coordinates>-140,8,0</coordinates>"
            "</Point></Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            build_advisory_json(SID, bad, self.track_b)


if __name__ == "__main__":
    unittest.main()


class TestCrossReviewRegressions(unittest.TestCase):
    """Pins the two cross-review findings (cyclolab-stage1-build review).

    The REAL NHC document order puts the Point placemarks BEFORE the
    LineString that carries the stormType ExtendedData - a single
    forward pass resolved the tau-0 initial point's dev_label while
    storm_type was still None, silently mislabeling a tau-0 HU/TD as
    "TS" (masked by Amanda genuinely being a TS)."""

    def _hurricane_track_points_first(self):
        # Real-NHC ordering: initial Point FIRST, ExtendedData LAST.
        return _make_kmz(
            "<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            "<Placemark><styleUrl>#initial_point</styleUrl>"
            "<description>Advisory Information Maximum Wind: 80 knots"
            "</description>"
            "<Point><coordinates>-134.1,13.1,0</coordinates></Point>"
            "</Placemark>"
            "<Placemark><styleUrl>#h_point</styleUrl>"
            "<description>12 hr Forecast Maximum Wind: 85 knots"
            "</description>"
            "<Point><coordinates>-135.0,13.5,0</coordinates></Point>"
            "</Placemark>"
            "<Placemark><ExtendedData>"
            "<Data name='advisoryNum'><value>7</value></Data>"
            "<Data name='stormType'><value>HU</value></Data>"
            "<Data name='pubAdvTime'>"
            "<value>1100 AM HST Fri Jun 05 2026</value></Data>"
            "</ExtendedData></Placemark>"
            "</Document></kml>")

    def test_tau0_inherits_true_storm_type_despite_document_order(self):
        track = parse_track_kmz(self._hurricane_track_points_first())
        self.assertEqual(track["points"][0]["tau_h"], 0)
        self.assertEqual(track["points"][0]["dev_label"], "HU",
                         "tau-0 must inherit the run's TRUE stormType "
                         "(forward-pass resolution saw None -> 'TS')")

    def test_windless_point_rejected_by_build(self):
        cone = _read(CONE_KMZ)
        kml = _make_kmz(
            "<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            "<Placemark><ExtendedData>"
            "<Data name='advisoryNum'><value>7</value></Data>"
            "<Data name='stormType'><value>TS</value></Data>"
            "<Data name='pubAdvTime'>"
            "<value>1100 AM HST Fri Jun 05 2026</value></Data>"
            "</ExtendedData></Placemark>"
            "<Placemark><styleUrl>#initial_point</styleUrl>"
            "<description>Advisory Information Maximum Wind: 35 knots"
            "</description>"
            "<Point><coordinates>-134.1,13.1,0</coordinates></Point>"
            "</Placemark>"
            "<Placemark><styleUrl>#s_point</styleUrl>"
            "<description>12 hr Forecast</description>"
            "<Point><coordinates>-135.0,13.5,0</coordinates></Point>"
            "</Placemark></Document></kml>")
        with self.assertRaises(AdvisoryParseError):
            build_advisory_json("NHC_EP012026", cone, kml)
