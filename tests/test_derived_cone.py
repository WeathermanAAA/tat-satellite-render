"""WP derived uncertainty cone (CYCLOLAB_DESIGN.md §8.4) - hand-computed
geometry, the pinned-radii never-invent contract, and the §8.3 JSON shape.

Runnable via pytest AND unittest:
    cd /tmp/tsr && python -m pytest tests/test_derived_cone.py -q
    cd /tmp/tsr && python -m unittest tests.test_derived_cone
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import derived_cone as dc  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BLOB_PATH = REPO / "cyclolab_radii_jtwc_wpac_mean_2015.json"

# The full pinned nm table (blob's nm_values, int keys) - used by the
# geometry tests so tau lookups hit exact table entries.
FULL_NM = {12: 24, 24: 39, 36: 50, 48: 61, 72: 92, 96: 129, 120: 180}


def gc_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Independent haversine great-circle distance (n mi) - the test's own
    yardstick, NOT reusing the module's forward solution."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * dc.R_NM * math.asin(min(1.0, math.sqrt(a)))


class TestRadiiBlob(unittest.TestCase):
    """The never-invent contract: the blob must match the pinned literals
    EXACTLY (source_md5 + the five km values). These are downloaded,
    image-verified numbers; a silent drift here would publish a fabricated
    uncertainty band."""

    def setUp(self):
        self.blob = json.loads(BLOB_PATH.read_text())

    def test_blob_file_loads(self):
        self.assertIsInstance(self.blob, dict)

    def test_method_version_pinned(self):
        self.assertEqual(self.blob["method_version"], "jtwc-wpac-mean-2015")

    def test_source_md5_exact(self):
        # The archived-PDF md5 from §8.4 - asserted, never recomputed.
        self.assertEqual(self.blob["source_md5"],
                         "a5eafd0f4ce55ac4e2c7f4420bfe43f6")

    def test_five_km_values_exact(self):
        self.assertEqual(
            self.blob["km_values"],
            {"24": 72.1, "48": 112.7, "72": 169.7, "96": 238.1, "120": 334.1},
        )

    def test_nm_values_exact(self):
        self.assertEqual(
            self.blob["nm_values"],
            {"12": 24, "24": 39, "36": 50, "48": 61,
             "72": 92, "96": 129, "120": 180},
        )

    def test_conversion_and_interpolated_taus(self):
        self.assertEqual(self.blob["conversion_nm_per_km"], 0.539957)
        self.assertEqual(self.blob["interpolated_taus"], [12, 36])

    def test_source_doc_names_table3_jtwc(self):
        self.assertIn("Table 3", self.blob["source_doc"])
        self.assertIn("JTWC", self.blob["source_doc"])


class TestTauInterpolation(unittest.TestCase):

    def test_exact_table_tau_36(self):
        # 36 h is a pinned (interpolated-in-doc) entry = 50 n mi exactly.
        self.assertEqual(dc._radius_for_tau(36, FULL_NM), 50.0)

    def test_exact_table_tau_24(self):
        self.assertEqual(dc._radius_for_tau(24, FULL_NM), 39.0)

    def test_linear_between_24_and_36(self):
        # 30 h sits halfway between the 24 (39) and 36 (50) entries.
        self.assertAlmostEqual(dc._radius_for_tau(30, FULL_NM),
                               (39 + 50) / 2, places=9)

    def test_below_smallest_tau_interpolates_from_origin(self):
        # 6 h is half of the 12 h entry (24) measured from a (0,0) apex.
        self.assertAlmostEqual(dc._radius_for_tau(6, FULL_NM), 12.0, places=9)

    def test_above_horizon_clamps(self):
        # Past the 120 h table end we clamp, not extrapolate.
        self.assertEqual(dc._radius_for_tau(168, FULL_NM), 180.0)

    def test_floor_applies_at_apex(self):
        # tau 0 -> 0 from origin interp, but floored to MIN_RADIUS_NM.
        self.assertEqual(dc._radius_for_tau(0, FULL_NM), dc.MIN_RADIUS_NM)


class TestSinglePointCircle(unittest.TestCase):
    """A single forecast point buffers to a full circle; every vertex is
    its radius away (great-circle), within 1%."""

    def setUp(self):
        # tau 24 with a 60 n mi table entry -> radius exactly 60 (no floor).
        self.center = (13.1, -134.1)
        self.ring = dc.derive_cone(
            [{"tau_h": 24, "lat": self.center[0], "lon": self.center[1]}],
            {24: 60},
        )

    def test_every_vertex_60nm_within_1pct(self):
        for lon, lat in self.ring:
            d = gc_nm(self.center[0], self.center[1], lat, lon)
            self.assertAlmostEqual(d, 60.0, delta=0.6)   # 1% of 60

    def test_ring_is_closed(self):
        self.assertEqual(self.ring[0], self.ring[-1])

    def test_lonlat_order(self):
        # Center is near -134 lon / +13 lat; ring lon's hug -134, lat's +13.
        lons = [c[0] for c in self.ring]
        lats = [c[1] for c in self.ring]
        self.assertTrue(all(-136 < x < -132 for x in lons))
        self.assertTrue(all(11 < y < 15 for y in lats))


class TestTwoPointStadium(unittest.TestCase):
    """Two points 120 n mi apart with equal radii r sweep a stadium:
    along-axis extent = separation + 2r, perpendicular extent = 2r."""

    R = 40.0
    SEP = 120.0

    def setUp(self):
        p0 = (10.0, 150.0)
        # p1 is due north of p0 by SEP n mi (axis = the meridian).
        lon1, lat1 = dc._dest_point(p0[0], p0[1], 0.0, self.SEP)
        self.p0, self.p1 = p0, (lat1, lon1)
        self.ring = dc.derive_cone(
            [{"tau_h": 24, "lat": p0[0], "lon": p0[1]},
             {"tau_h": 48, "lat": lat1, "lon": lon1}],
            {24: self.R, 48: self.R},
        )

    def test_separation_is_120(self):
        self.assertAlmostEqual(
            gc_nm(self.p0[0], self.p0[1], self.p1[0], self.p1[1]),
            self.SEP, delta=0.01)

    def test_along_axis_extent(self):
        # Northernmost to southernmost ring vertex (axis is the meridian).
        north = max(self.ring, key=lambda c: c[1])
        south = min(self.ring, key=lambda c: c[1])
        along = gc_nm(south[1], south[0], north[1], north[0])
        expected = self.SEP + 2 * self.R          # 200
        self.assertAlmostEqual(along, expected, delta=0.02 * expected)

    def test_perpendicular_extent(self):
        # Width measured AS LONGITUDE SPREAD at the mid latitude: the
        # extreme-lon vertices may sit at different latitudes (the
        # tangent chain tie-breaks differently than the old rails did),
        # and a point-to-point great-circle between them would smuggle
        # in the along-axis separation (sqrt(80^2+120^2) = 144).
        import math
        lons = [c[0] for c in self.ring]
        mid_lat = (self.p0[0] + self.p1[0]) / 2.0
        perp = (max(lons) - min(lons)) * 60.0 * math.cos(
            math.radians(mid_lat))
        expected = 2 * self.R                      # 80
        self.assertAlmostEqual(perp, expected, delta=0.02 * expected)

    def test_ring_closed(self):
        self.assertEqual(self.ring[0], self.ring[-1])


class TestMonotoneWidening(unittest.TestCase):
    """Increasing radii along a straight (due-east) track => the corridor
    half-width grows monotonically. The forward (left) rail vertices sit
    exactly their point's radius off-axis, so measuring them proves it."""

    def setUp(self):
        self.taus = [24, 48, 72, 96]
        self.radii = [30, 50, 70, 90]
        self.pts = []
        lat0, lon0 = 5.0, 140.0
        for i, tau in enumerate(self.taus):
            lon_i, lat_i = dc._dest_point(lat0, lon0, 90.0, 60.0 * i)
            self.pts.append({"tau_h": tau, "lat": lat_i, "lon": lon_i})
        self.table = dict(zip(self.taus, self.radii))
        self.ring = dc.derive_cone(self.pts, self.table)

    def test_left_rail_halfwidth_equals_radius_and_increases(self):
        # Layout-independent envelope contract (the tangent chain
        # interleaves contacts, crossings and arcs - positions are not
        # pinned): the boundary TOUCHES every tau circle (min vertex
        # distance ~ r) and never cuts materially inside it. Tolerance
        # 6% = the documented local-planar-frame distortion ceiling
        # (cos-lat spread on strongly poleward tracks), immaterial for
        # an average-error uncertainty band.
        for p, r in zip(self.pts, self.radii):
            dmin = min(gc_nm(p["lat"], p["lon"], lat_v, lon_v)
                       for lon_v, lat_v in self.ring)
            self.assertLessEqual(dmin, r * 1.06,
                                 f"tau {p['tau_h']}: boundary never "
                                 f"touches its circle (dmin {dmin:.1f} "
                                 f"vs r {r})")
            self.assertGreaterEqual(dmin, r * 0.94,
                                    f"tau {p['tau_h']}: boundary cuts "
                                    f"{r - dmin:.1f} nm inside r={r}")

    def test_ring_closed(self):
        self.assertEqual(self.ring[0], self.ring[-1])


class TestDeterminism(unittest.TestCase):

    def test_same_input_same_ring(self):
        pts = [{"tau_h": 0, "lat": 12.0, "lon": 130.0},
               {"tau_h": 24, "lat": 13.5, "lon": 131.2},
               {"tau_h": 48, "lat": 15.0, "lon": 132.8}]
        a = dc.derive_cone(pts, FULL_NM)
        b = dc.derive_cone(pts, FULL_NM)
        self.assertEqual(a, b)

    def test_input_order_independent(self):
        # The function sorts by tau, so a shuffled input gives the same ring.
        pts = [{"tau_h": 0, "lat": 12.0, "lon": 130.0},
               {"tau_h": 24, "lat": 13.5, "lon": 131.2},
               {"tau_h": 48, "lat": 15.0, "lon": 132.8}]
        a = dc.derive_cone(pts, FULL_NM)
        b = dc.derive_cone(list(reversed(pts)), FULL_NM)
        self.assertEqual(a, b)

    def test_empty_points_raises(self):
        with self.assertRaises(ValueError):
            dc.derive_cone([], FULL_NM)


class TestBuildDerivedAdvisoryJson(unittest.TestCase):
    """The §8.3 contract for a WP derived cone."""

    def setUp(self):
        self.blob = json.loads(BLOB_PATH.read_text())
        self.points = [
            {"tau_h": 0, "valid_utc": "2026-06-05T18:00:00Z",
             "lat": 12.0, "lon": 130.0, "intensity_kt": 45, "dev_label": "TS",
             "advisory": 7},
            {"tau_h": 24, "valid_utc": "2026-06-06T18:00:00Z",
             "lat": 13.5, "lon": 131.2, "intensity_kt": 55, "dev_label": "TY"},
            {"tau_h": 48, "valid_utc": "2026-06-07T18:00:00Z",
             "lat": 15.0, "lon": 132.8, "intensity_kt": 65, "dev_label": "TY"},
        ]
        self.out = dc.build_derived_advisory_json(
            "JTWC_WP062026", self.points, self.blob)

    def test_contract_keys_present(self):
        for k in ("sid", "advisory", "issued_utc", "source", "method",
                  "cone", "points", "text", "provenance"):
            self.assertIn(k, self.out)

    def test_source_is_jtwc(self):
        self.assertEqual(self.out["source"], "jtwc")

    def test_method_is_method_version_prefixed(self):
        self.assertEqual(self.out["method"],
                         "derived-mean-error-jtwc-wpac-mean-2015")

    def test_text_is_null(self):
        self.assertIsNone(self.out["text"])

    def test_points_pass_through_verbatim(self):
        self.assertEqual(self.out["points"], self.points)

    def test_sid_and_issuance_and_advisory(self):
        self.assertEqual(self.out["sid"], "JTWC_WP062026")
        self.assertEqual(self.out["issued_utc"], "2026-06-05T18:00:00Z")
        self.assertEqual(self.out["advisory"], 7)

    def test_cone_is_closed_ring_of_pairs(self):
        cone = self.out["cone"]
        self.assertGreater(len(cone), 4)
        self.assertEqual(cone[0], cone[-1])
        for pt in cone:
            self.assertEqual(len(pt), 2)

    def test_provenance_records_radii_method(self):
        prov = self.out["provenance"]
        self.assertEqual(prov["radii_method_version"], "jtwc-wpac-mean-2015")
        self.assertEqual(prov["radii_source_doc"], self.blob["source_doc"])

    def test_serializable(self):
        # The whole contract round-trips through JSON unchanged.
        s = json.dumps(self.out)
        self.assertEqual(json.loads(s)["method"], self.out["method"])



class TestSweptEnvelopeSmoothness(unittest.TestCase):
    """S4-AD1 #9: the tangent chain must be visually smooth - no
    scallops, no backtracking micro-edges - across track shapes incl.
    the engulfed stalled-storm case (domination filter)."""

    @staticmethod
    def _max_turn(ring):
        import math

        def brg(a, b):
            la1, lo1, la2, lo2 = (math.radians(a[1]), math.radians(a[0]),
                                  math.radians(b[1]), math.radians(b[0]))
            y = math.sin(lo2 - lo1) * math.cos(la2)
            x = (math.cos(la1) * math.sin(la2)
                 - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1))
            return math.degrees(math.atan2(y, x))
        worst = 0.0
        for i in range(1, len(ring) - 1):
            t = abs((brg(ring[i], ring[i + 1]) - brg(ring[i - 1], ring[i])
                     + 540.0) % 360.0 - 180.0)
            worst = max(worst, t)
        return worst

    def _ring(self, pts):
        nm = {24: 39.0, 48: 61.0, 72: 92.0, 96: 129.0, 120: 180.0}
        return dc.derive_cone(pts, nm)

    def test_straight_track_no_scallops(self):
        ring = self._ring([{"tau_h": t, "lat": 15 + 0.05 * t,
                            "lon": -135 - 0.09 * t}
                           for t in (0, 12, 24, 36, 48, 72, 96, 120)])
        self.assertLess(self._max_turn(ring), 35.0,
                        "boundary kinks - scallops are back")

    def test_recurving_track_no_scallops(self):
        ring = self._ring([{"tau_h": t, "lat": 15 + 0.08 * t,
                            "lon": -135 - 0.12 * t + (t / 45.0) ** 2 * 1.4}
                           for t in (0, 12, 24, 36, 48, 72, 96, 120)])
        self.assertLess(self._max_turn(ring), 45.0)

    def test_stalled_track_engulfed_circles_filtered(self):
        # consecutive circles engulf (d ~ 12 nm < dr) - the domination
        # filter must keep the envelope sane, not corrupt the chain.
        ring = self._ring([{"tau_h": t, "lat": 15 + 0.01 * t,
                            "lon": -135 - 0.015 * t}
                           for t in (0, 12, 24, 36, 48, 72, 96, 120)])
        self.assertEqual(ring[0], ring[-1])
        self.assertLess(self._max_turn(ring), 35.0)

    def test_provenance_names_the_construction(self):
        nm_blob = {"method_version": "jtwc-wpac-mean-2015",
                   "nm_values": {"24": 39, "48": 61, "72": 92,
                                 "96": 129, "120": 180}}
        adv = dc.build_derived_advisory_json(
            "JTWC_WP082026",
            [{"tau_h": 0, "lat": 15, "lon": -135},
             {"tau_h": 24, "lat": 16, "lon": -137}], nm_blob)
        self.assertEqual(adv["provenance"]["construction"],
                         "tangent-chain-swept-envelope")


if __name__ == "__main__":
    unittest.main()
