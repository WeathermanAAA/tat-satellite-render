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


if __name__ == "__main__":
    unittest.main(verbosity=2)
