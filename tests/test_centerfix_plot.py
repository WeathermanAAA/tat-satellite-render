"""Contracts for the storm-centred multi-source centre-fix plot.

The plot's job is showing where the OBJECTIVE fixes disagree with the OFFICIAL
position, so the tests here are mostly honesty properties rather than pixels:

  * SATCON is an INTENSITY consensus and produces no position. It must never
    become a centre marker, and when its own membership rule is unmet the
    header must say "no consensus" rather than relabelling the bare ADT.
  * A REJECTED ARCHER candidate must not carry the headline crosshair, the
    certainty rings or the separation measurement - that would publish a
    number ARCHER itself refused. The plot falls back to the newest ACCEPTED
    fix and discloses that it did.
  * A truncated collector run must be disclosed, because the newest analyzed
    frame is then older than the newest available one.
  * The view window is the requested box, not the data array's extent: a
    geostationary grid this far off nadir is not axis-aligned in lat/lon.
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import centerfix_plot as cf                     # noqa: E402


class _Fake:
    """Minimal stand-in for a FetchResult over a small regular grid."""

    def __init__(self, clat=13.4, clon=170.7, half=3.0, n=48, units="K"):
        lat = np.linspace(clat - half, clat + half, n)
        lon = np.linspace(clon - half, clon + half, n)
        self.lons, self.lats = np.meshgrid(lon, lat)
        r = np.hypot(self.lons - clon, self.lats - clat)
        bt_c = -20.0 - 60.0 * np.exp(-((r - 0.5) ** 2) / 0.15)
        bt_c = np.where(r < 0.25, 12.0, bt_c)
        self.cmi = bt_c + 273.15
        self.units = units
        self.sat_name = "Himawari-9"


def _track(points, truncated=False, satcon=None):
    return {"storm": {"id": "JTWC_WP122026", "name": "DOLPHIN"},
            "points": points, "truncated": truncated, "satcon": satcon,
            "input": "AHI B13 calibrated BT"}


def _pt(t, lat, lon, fix=True, **kw):
    d = {"t": t, "lat": lat, "lon": lon, "fix": fix,
         "confidence_score": kw.get("conf", 1.1),
         "r50_km": kw.get("r50", 16), "r95_km": kw.get("r95", 43),
         "scene": kw.get("scene", "EYE"), "vmax_kt": kw.get("vmax", 100.0),
         "mslp_mb": kw.get("mslp", 950.0)}
    return d


STORM = {"id": "JTWC_WP122026", "name": "DOLPHIN", "slug": "wp12",
         "basin": "WP", "lat": 13.4, "lon": 170.7, "intensity_kt": 100.0}
BBOX = [167.7, 10.4, 173.7, 16.4]      # [W, S, E, N]


class GeometryTests(unittest.TestCase):
    def test_km_between_is_antimeridian_safe(self):
        # 179.7E vs -179.0 is 1.3 deg apart, not 358.7.
        d = cf._km_between(10.0, 179.7, 10.0, -179.0)
        self.assertLess(d, 200.0)

    def test_target_box_is_the_square_about_the_backdrop_centre(self):
        # floater_poller widens the BACKDROP to the render aspect; the target
        # box is the square of side (N-S) about that centre - the same
        # reconstruction the explorer does to georeference a frame.
        man = {"bands": {"ir": {"frames": [
            {"t": "2026-07-28T18:40:00Z", "bounds": [164.7, 7.4, 176.7, 19.4]}
        ]}}}
        box = cf.target_box(man)
        self.assertAlmostEqual(box["n"] - box["s"], 12.0, places=6)
        self.assertAlmostEqual(box["e"] - box["w"], 12.0, places=6)
        self.assertAlmostEqual((box["w"] + box["e"]) / 2, 170.7, places=6)

    def test_target_box_missing_is_none_not_a_guess(self):
        self.assertIsNone(cf.target_box(None))
        self.assertIsNone(cf.target_box({"bands": {}}))


class RenderContractTests(unittest.TestCase):
    """Renders a real PNG each time; asserts on the figure's own text."""

    def _render(self, track, satcon=None, adv=None):
        import matplotlib
        matplotlib.use("Agg")
        png = cf.render(STORM, track, adv, _Fake(), _Fake(),
                        box={"w": 164.7, "s": 7.4, "e": 176.7, "n": 19.4},
                        satcon=satcon, dpi=50, bbox=BBOX)
        self.assertGreater(len(png), 5000)
        self.assertEqual(png[1:4], b"PNG")
        return png

    # -- the header strings are the honesty surface; capture them ---------
    def _texts(self, track, satcon=None, adv=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        seen = []
        orig = plt.Figure.text

        def spy(self, x, y, s, *a, **k):
            seen.append(str(s))
            return orig(self, x, y, s, *a, **k)
        # capture Axes.text too (panel labels / BT readout)
        from matplotlib.axes import Axes
        aorig = Axes.text

        def aspy(self, x, y, s, *a, **k):
            seen.append(str(s))
            return aorig(self, x, y, s, *a, **k)
        plt.Figure.text, Axes.text = spy, aspy
        try:
            self._render(track, satcon=satcon, adv=adv)
        finally:
            plt.Figure.text, Axes.text = orig, aorig
        return "\n".join(seen)

    def test_satcon_absent_says_no_consensus(self):
        txt = self._texts(_track([_pt("2026-07-28T05:30:00.000Z", 13.5, 170.0)]),
                          satcon=None)
        self.assertIn("SATCON", txt)
        self.assertIn("no consensus", txt)

    def test_satcon_present_is_an_intensity_readout(self):
        sc = {"vmax": {"value": 87.7}, "mslp": {"value": 941.0}}
        txt = self._texts(_track([_pt("2026-07-28T05:30:00.000Z", 13.5, 170.0)]),
                          satcon=sc)
        self.assertIn("SATCON", txt)
        self.assertIn("88 kt", txt)
        self.assertNotIn("no consensus", txt)

    def test_adt_and_official_intensities_are_both_shown(self):
        txt = self._texts(_track([_pt("2026-07-28T05:30:00.000Z", 13.5, 170.0)]))
        self.assertIn("ADT", txt)
        self.assertIn("OFFICIAL", txt)

    def test_rejected_newest_candidate_is_disclosed(self):
        # newest frame rejected, an older accepted fix exists -> the crosshair
        # is on the older one and the header must SAY so.
        pts = [_pt("2026-07-28T04:30:00.000Z", 13.5, 170.0, fix=True),
               _pt("2026-07-28T05:30:00.000Z", 11.5, 172.5, fix=False,
                   scene="CURVED BAND", conf=0.39)]
        txt = self._texts(_track(pts))
        self.assertIn("CANDIDATE REJECTED", txt)
        self.assertIn("CROSSHAIR", txt)

    def test_accepted_newest_needs_no_rejection_notice(self):
        pts = [_pt("2026-07-28T05:30:00.000Z", 13.5, 170.0, fix=True)]
        txt = self._texts(_track(pts))
        self.assertNotIn("CANDIDATE REJECTED", txt)

    def test_truncated_run_is_disclosed(self):
        pts = [_pt("2026-07-28T05:30:00.000Z", 13.5, 170.0)]
        txt = self._texts(_track(pts, truncated=True))
        self.assertIn("TRUNCATED", txt)

    def test_bt_readouts_cover_ir_and_wv(self):
        txt = self._texts(_track([_pt("2026-07-28T05:30:00.000Z", 13.5, 170.0)]))
        self.assertIn("IR BT", txt)
        self.assertIn("WV BT", txt)

    def test_disclosure_and_provider_credit_are_present(self):
        txt = self._texts(_track([_pt("2026-07-28T05:30:00.000Z", 13.5, 170.0)]))
        self.assertIn("not official", txt.lower())
        self.assertIn("Himawari-9", txt)

    def test_renders_without_a_forecast_or_a_box(self):
        import matplotlib
        matplotlib.use("Agg")
        png = cf.render(STORM, _track([_pt("2026-07-28T05:30:00.000Z",
                                           13.5, 170.0)]),
                        None, _Fake(), None, box=None, satcon=None,
                        dpi=50, bbox=BBOX)
        self.assertGreater(len(png), 5000)

    def test_no_accepted_fix_still_renders(self):
        # every candidate rejected: no crosshair, no separation number, but a
        # plot that draws the scene and says why is better than no plot.
        pts = [_pt("2026-07-28T05:30:00.000Z", 11.5, 172.5, fix=False,
                   scene="SHEAR", conf=0.14)]
        txt = self._texts(_track(pts))
        self.assertIn("CANDIDATE REJECTED", txt)


class _FakeNoEye(_Fake):
    """A sheared / eyeless system: the coldest cloud sits ON the centre, so
    there is no clearing to score. The honest output is a withheld score, not
    a number computed off an empty eye region."""

    def __init__(self, clat=13.4, clon=170.7, half=3.0, n=48):
        super().__init__(clat, clon, half, n)
        r = np.hypot(self.lons - clon, self.lats - clat)
        self.cmi = (-80.0 + 22.0 * r) + 273.15


def _tiny_png() -> bytes:
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf = _io.BytesIO()
    f = plt.figure(figsize=(2, 1))
    f.savefig(buf, format="png")
    plt.close(f)
    return buf.getvalue()


def _eye_field(clat=18.0, clon=-117.0, half=1.6, n=200,
               eye_r_km=12.0, wall_w_km=14.0, eye_c=-6.0, wall_c=-72.0,
               canopy_c=-58.0, amb_c=-30.0, warm_at_km=0.0):
    """A REALISTIC synthetic storm at ~2 km pixels.

    Warm eye that is warmest at its own centre (so the warmest-pixel search is
    not picking an arbitrary member of a flat disc), a cold eyewall ANNULUS of
    finite width, a cooler canopy outside it, then ambient. An "eyewall"
    modelled as a filled disc is not an eyewall: its inner edge, which is what
    the search looks for, would sit at the eye's own edge.

    ``warm_at_km`` offsets the warm spot, which is how a warm NOTCH against the
    eyewall's inner edge is simulated.
    """
    lat = np.linspace(clat - half, clat + half, n)
    lon = np.linspace(clon - half, clon + half, n)
    LON, LAT = np.meshgrid(lon, lat)
    kmy = (LAT - clat) * 111.0
    kmx = (LON - clon) * 111.0 * np.cos(np.radians(clat))
    r = np.hypot(kmx, kmy)
    wall_out = eye_r_km + wall_w_km
    bt = np.full(r.shape, amb_c, dtype="float64")
    bt[r > wall_out] = canopy_c
    bt[r > wall_out * 2.6] = amb_c
    bt[(r > eye_r_km) & (r <= wall_out)] = wall_c
    rw = np.hypot(kmx - warm_at_km, kmy)
    inside = r <= eye_r_km
    bt[inside] = eye_c - 8.0 * (rw[inside] / max(eye_r_km, 1e-6))
    return bt, LAT, LON, clat, clon


class EyeScoreTests(unittest.TestCase):
    """The eye score is a CONTRAST in °C, not a pixel count: a count is
    resolution-dependent and would not compare across sensors. A system with
    no eye must return nothing rather than a number computed off an empty
    region."""

    def test_finds_the_eyewall_and_scores_a_real_eye(self):
        # THE BUG THIS PINS: the eyewall used to be the coldest ring ANYWHERE
        # in a 140 km profile, which on a large storm is the outer canopy.
        # Verified against GENEVIEVE, whose eye is ~16 km across but whose
        # globally-coldest ring sat at 62 km -- so "inside the eyewall"
        # swallowed most of the CDO and the score measured eye-centre against
        # outer-canopy contrast: a real number answering the wrong question.
        bt, la, lo, clat, clon = _eye_field()
        prof = cf.eye_score(bt, la, lo, clat, clon, scene="EYE")
        self.assertIsNotNone(prof["score"], prof.get("reason"))
        # the eyewall's INNER EDGE is the eye radius (12 km) -- the search must
        # land there, not out in the canopy
        self.assertLess(abs(prof["eyewall_r_km"] - 12.0), 10.0,
                        f"eyewall at {prof['eyewall_r_km']} km, true ~12 km")
        self.assertGreater(prof["score"], 40.0)
        self.assertGreater(prof["eye_warm_c"], prof["eyewall_cold_c"])

    def test_a_cloud_filled_eye_does_not_score_like_a_cleared_one(self):
        bt, la, lo, clat, clon = _eye_field(eye_c=-68.0)
        prof = cf.eye_score(bt, la, lo, clat, clon, scene="EYE")
        if prof["score"] is not None:
            self.assertLess(prof["score"], 15.0)

    def test_a_warm_notch_against_the_eyewall_is_withheld(self):
        # Not a cleared eye: the warm spot is out against the wall, which is
        # also what a mislocated centre looks like.
        bt, la, lo, clat, clon = _eye_field(warm_at_km=10.0)
        prof = cf.eye_score(bt, la, lo, clat, clon, scene="EYE")
        self.assertIsNone(prof["score"])
        self.assertIn("notch", (prof["reason"] or "").lower())

    def test_an_unresolved_eye_is_withheld_not_estimated(self):
        # A few pixels across cannot resolve its own warm minimum; the warmest
        # pixel is then an artefact of where the grid falls.
        bt, la, lo, clat, clon = _eye_field(n=24, eye_r_km=4.0, wall_w_km=5.0)
        prof = cf.eye_score(bt, la, lo, clat, clon, scene="EYE")
        self.assertIsNone(prof["score"])
        self.assertIn("resolved", (prof["reason"] or "").lower())

    def test_the_search_floor_is_adaptive_not_a_bare_12_km(self):
        # A fixed 12 km floor pushed the search PAST the eyewall of a small
        # storm, which is how GENEVIEVE lost its score entirely.
        bt, la, lo, clat, clon = _eye_field()
        prof = cf.eye_score(bt, la, lo, clat, clon, scene="EYE")
        self.assertLessEqual(prof["eyewall_floor_km"], 12.0)

    def test_resolution_is_reported_so_the_number_can_be_judged(self):
        bt, la, lo, clat, clon = _eye_field()
        prof = cf.eye_score(bt, la, lo, clat, clon, scene="EYE")
        self.assertGreater(prof["px_km"], 0.0)
        self.assertGreater(prof["eye_across_px"], cf.EYE_MIN_ACROSS_PX)

    def test_no_eye_withholds_the_score_and_says_why(self):
        f = _FakeNoEye()
        prof = cf.eye_score(cf._bt_celsius(f), f.lats, f.lons, 13.4, 170.7)
        self.assertIsNotNone(prof)          # the PROFILE is still real
        self.assertIsNone(prof["score"])    # ...the score is not
        self.assertTrue(prof["reason"])

    def test_a_non_eye_scene_withholds_the_score(self):
        # THE GATE THAT MATTERS MOST. A storm under a uniform CDO still has a
        # warmest pixel and a coldest ring, and their difference is a real
        # contrast that is NOT an eye score. Live 12W on 2026-07-28 reported
        # 16.8 °C off a "warmest eye pixel" of -62.5 °C -- deep convective
        # cloud -- while the method's own classifier said UNIFORM CDO.
        f = _Fake()
        prof = cf.eye_score(cf._bt_celsius(f), f.lats, f.lons, 13.4, 170.7,
                            scene="UNIFORM CDO")
        self.assertIsNotNone(prof)
        self.assertIsNone(prof["score"])
        self.assertIn("UNIFORM CDO", prof["reason"])

    def test_an_eye_scene_is_still_scored(self):
        bt, la, lo, clat, clon = _eye_field()
        prof = cf.eye_score(bt, la, lo, clat, clon, scene="EYE")
        self.assertIsNotNone(prof["score"], prof.get("reason"))

    def test_the_eyewall_floor_is_what_stops_a_zero_radius_eye(self):
        # Without it an eyeless field reports an "eye" of zero radius, scoring
        # the centre pixel against itself.
        self.assertGreater(cf.EYE_MIN_EYEWALL_KM_MIN, 0.0)


class CompositePlateTests(unittest.TestCase):
    """The 2x2 plate is an ADDITIONAL product. It carries all four panels under
    one header, never silently drops to three, and does not recompute the ACE
    chart it pastes in."""

    PTS = [_pt("2026-07-28T05:30:00.000Z", 13.42, 170.68)]

    def _plate(self, hist=None, ir=None, adv=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        seen = []
        forig, aorig = plt.Figure.text, Axes.text

        def fspy(self, x, y, s, *a, **k):
            seen.append(str(s)); return forig(self, x, y, s, *a, **k)

        def aspy(self, x, y, s, *a, **k):
            seen.append(str(s)); return aorig(self, x, y, s, *a, **k)
        plt.Figure.text, Axes.text = fspy, aspy
        try:
            png = cf.render_composite(
                STORM, _track(self.PTS), adv, ir or _Fake(), _Fake(),
                box=None, satcon=None, dpi=50, bbox=BBOX,
                track_history=hist)
        finally:
            plt.Figure.text, Axes.text = forig, aorig
        self.assertEqual(png[1:4], b"PNG")
        self.assertGreater(len(png), 5000)
        return png, "\n".join(seen)

    def test_plate_carries_all_four_panels(self):
        _png, txt = self._plate()
        self.assertIn("GRAYSCALE + BD-STEP CONTOURS", txt)   # top-left
        self.assertIn("ENHANCED COLOUR", txt)                # top-right
        self.assertIn("WIND, PRESSURE & ACE", txt)           # bottom-left
        self.assertIn("EYE STRUCTURE", txt)                  # bottom-right

    def test_eyeless_storm_keeps_the_panel_and_withholds_the_number(self):
        _png, txt = self._plate(ir=_FakeNoEye())
        self.assertIn("EYE STRUCTURE", txt)
        self.assertNotIn("EYE SCORE", txt)

    def test_a_cdo_scene_draws_the_profile_but_no_score(self):
        # The plate must carry the ADT scene gate through, not just eye_score.
        pts = [_pt("2026-07-28T05:30:00.000Z", 13.42, 170.68,
                   scene="UNIFORM CDO")]
        saved, self.PTS = self.PTS, pts
        try:
            _png, txt = self._plate()
        finally:
            self.PTS = saved
        self.assertIn("EYE STRUCTURE", txt)
        self.assertNotIn("EYE SCORE", txt)
        self.assertIn("UNIFORM CDO", txt)

    def test_the_chart_panel_labels_itself_without_covering_the_chart(self):
        # The captured chart captions its own axes along its top edge, so the
        # panel label needs a reserved strip rather than the image's top-left.
        _png, txt = self._plate()
        self.assertIn("WIND, PRESSURE & ACE", txt)

    def test_plate_header_carries_identity_valid_time_and_forecast_hour(self):
        adv = {"advisory": 8, "points": [
            {"lat": 13.9, "lon": 169.4, "tau_h": 12},
            {"lat": 14.6, "lon": 167.9, "tau_h": 24}]}
        _png, txt = self._plate(adv=adv)
        self.assertIn("DOLPHIN", txt)
        self.assertIn("VALID", txt)
        self.assertIn("T+24H", txt)
        self.assertIn("OFFICIAL", txt)

    def test_band_readouts_are_tagged_ir_wv_and_swir(self):
        # An untagged -60 C invites reading a WV frame as a cloud top, so every
        # band names itself -- including the ones that did not arrive.
        _png, txt = self._plate()
        self.assertIn("IR BT", txt)
        self.assertIn("WV BT", txt)
        self.assertIn("SWIR BT", txt)

    def test_the_two_panel_plot_is_still_its_own_product(self):
        # The plate is additive: render() keeps working on its own inputs and
        # stays the thing published to the existing key.
        import matplotlib
        matplotlib.use("Agg")
        png = cf.render(STORM, _track(self.PTS), None, _Fake(), _Fake(),
                        box=None, satcon=None, dpi=50, bbox=BBOX)
        self.assertEqual(png[1:4], b"PNG")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ProfileMetadataTests(unittest.TestCase):
    """Grid facts belong to the PROFILE, not to a successful score.

    They were assigned only on the scored path, so a withheld panel printed
    "0.0 km pixels" and "beyond a 0 km floor" — quantities that were never
    computed, rendered as though they had been measured. A zero that means
    "unset" is worse than a blank, because it reads as data.
    """

    def _prof(self, scene):
        bt, la, lo, clat, clon = _eye_field()
        return cf.eye_score(bt, la, lo, clat, clon, scene=scene)

    def test_pixel_size_and_floor_are_set_even_when_withheld(self):
        for scene in ("EYE", "UNIFORM CDO", "SHEAR"):
            with self.subTest(scene):
                p = self._prof(scene)
                self.assertIsNotNone(p["px_km"])
                self.assertGreater(p["px_km"], 0.0)
                self.assertIsNotNone(p["eyewall_floor_km"])
                self.assertGreater(p["eyewall_floor_km"], 0.0)

    def test_contours_are_five_levels_not_the_full_ladder(self):
        # Nine levels on a 2 km field is a mesh over the whole frame; the
        # storm-scale curves are what this panel is for.
        self.assertEqual(len(cf.CONTOUR_LEVELS), 5)
        # ascending, as matplotlib requires
        self.assertEqual(cf.CONTOUR_LEVELS,
                         [-80.0, -75.0, -63.0, -53.0, -30.0])
        self.assertEqual(len(cf.CONTOUR_COLORS), len(cf.CONTOUR_LEVELS))

    def test_category_bands_are_a_background_tint(self):
        # Full strength made the panel read as the bands with a line on top.
        self.assertLessEqual(cf.BAND_ALPHA, 0.35)
        self.assertGreater(cf.BAND_ALPHA, 0.15)

    def test_speck_filter_drops_small_closed_loops(self):
        import numpy as _np
        # a 5 km box is a speck; a 40 km box is structure
        small = _np.array([[0, 0], [0.045, 0], [0.045, 0.045], [0, 0.045],
                           [0, 0]])
        big = _np.array([[0, 0], [0.36, 0], [0.36, 0.36], [0, 0.36], [0, 0]])
        self.assertLess(cf._poly_area_km2(small, 15.0), cf.CONTOUR_MIN_AREA_KM2)
        self.assertGreater(cf._poly_area_km2(big, 15.0), cf.CONTOUR_MIN_AREA_KM2)
