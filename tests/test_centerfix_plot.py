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


class EyeScoreTests(unittest.TestCase):
    """The eye score is a CONTRAST in °C, not a pixel count: a count is
    resolution-dependent and would not compare across sensors. A system with
    no eye must return nothing rather than a number computed off an empty
    region."""

    def test_finds_the_eyewall_and_scores_a_real_eye(self):
        f = _Fake()
        prof = cf.eye_score(cf._bt_celsius(f), f.lats, f.lons, 13.4, 170.7)
        self.assertIsNotNone(prof)
        self.assertIsNotNone(prof["score"])
        # the synthetic eyewall ring sits at ~0.5 deg = ~55 km
        self.assertGreater(prof["eyewall_r_km"], 30.0)
        self.assertLess(prof["eyewall_r_km"], 85.0)
        # a warm eye (+12 C) inside a deep cold ring -> a large positive score
        self.assertGreater(prof["score"], 40.0)
        self.assertGreater(prof["eye_warm_c"], prof["eyewall_cold_c"])

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
        f = _Fake()
        prof = cf.eye_score(cf._bt_celsius(f), f.lats, f.lons, 13.4, 170.7,
                            scene="EYE")
        self.assertIsNotNone(prof["score"])

    def test_the_eyewall_floor_is_what_stops_a_zero_radius_eye(self):
        # Without it an eyeless field reports an "eye" of zero radius, scoring
        # the centre pixel against itself.
        self.assertGreater(cf.EYE_MIN_EYEWALL_KM, 0.0)


class CompositePlateTests(unittest.TestCase):
    """The 2x2 plate is an ADDITIONAL product. It carries all four panels under
    one header, never silently drops to three, and does not recompute the ACE
    chart it pastes in."""

    PTS = [_pt("2026-07-28T05:30:00.000Z", 13.42, 170.68)]

    def _plate(self, wpace_png=None, captured=None, ir=None, adv=None):
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
                wpace_png=wpace_png, wpace_captured=captured)
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

    def test_missing_chart_capture_says_so_rather_than_drawing_nothing(self):
        # An empty frame would read as "this storm has no ACE" -- a claim.
        _png, txt = self._plate(wpace_png=None)
        self.assertIn("not captured", txt)

    def test_a_captured_chart_is_pasted_not_recomputed(self):
        _png, txt = self._plate(wpace_png=_tiny_png())
        self.assertIn("WIND, PRESSURE & ACE", txt)
        self.assertNotIn("not captured", txt)

    def test_a_stale_capture_declares_its_age(self):
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)
        _png, txt = self._plate(wpace_png=_tiny_png(), captured=old)
        self.assertIn("CAPTURED", txt)

    def test_a_fresh_capture_does_not_nag(self):
        fresh = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)
        _png, txt = self._plate(wpace_png=_tiny_png(), captured=fresh)
        self.assertNotIn("CAPTURED", txt)

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
        _png, txt = self._plate(wpace_png=_tiny_png())
        self.assertIn("FROM CYCLOLAB", txt)

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
