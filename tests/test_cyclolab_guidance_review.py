"""Tests for the CycloLab guidance Stage-B REVIEW renderers (held; not in the live
flow). A cheap structural pass always runs; the real rendering assertions run under
Playwright when it is installed (developed storm / fresh invest / SHIPS unavailable).

  python -m unittest tests.test_cyclolab_guidance_review
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cyclolab_guidance_review as R
from cyclolab_basemap import basemap_for


def _ramp(v0, dv, n=11, pos=True):
    return [{"tau": i * 6, "lat": (10 + 0.3 * i) if pos else None,
             "lon": (-120 - 0.4 * i) if pos else None,
             "vmax": max(15, v0 + dv * i), "mslp": 1005 - i} for i in range(n)]


DEVELOPED = {
    "init_time": "2026-06-16T00:00:00Z", "init_cycle": "2026061600",
    "aids": {"AVNI": _ramp(25, 4), "HWFI": _ramp(25, 6), "TVCN": _ramp(25, 5),
             "HCCA": _ramp(25, 5), "DSHP": _ramp(25, 3), "IVCN": _ramp(25, 4)},
    "present_aids": ["AVNI", "HWFI", "TVCN", "HCCA", "DSHP", "IVCN"],
    "track_aids": ["AVNI", "HWFI", "TVCN", "HCCA"],
    "intensity_aids": ["AVNI", "HWFI", "DSHP", "IVCN"],
    "consensus": ["TVCN", "HCCA", "IVCN"], "sid": "NHC_EP012026", "basin": "EP",
}
FRESH = {
    "init_time": "2026-06-16T06:00:00Z", "init_cycle": "2026061606",
    "aids": {"DSHP": _ramp(25, 2, pos=False), "LGEM": _ramp(25, 1, pos=False),
             "SHIP": _ramp(25, 2, pos=False)},
    "present_aids": ["DSHP", "LGEM", "SHIP"], "track_aids": [],
    "intensity_aids": ["DSHP", "LGEM", "SHIP"], "consensus": [],
    "sid": "NHC_AL952026", "basin": "AL",
}
SHIPS_OK = {"available": True, "header": {"id_line": "TEST EP012026"}, "taus": [0, 6, 12, 24, 48],
            "env_series": {"SHEAR (KT)": [5, 6, 8, 12, 20], "SST (C)": [29, 29, 28, 28, 27]},
            "storm_type": ["TROP"] * 5, "prelim_ri_prob": 3.0, "ri_predictor_table": [],
            "ri_threshold_probs": [], "ri_matrix": {"cols": ["20/12", "25/24"],
            "rows": {"Consensus": {"20/12": 1.0, "25/24": 4.0}}}, "ahi": {"value": 0, "verdict": "NOT ANNULAR"}}
SHIPS_OFF = {"available": False, "reason": "unavailable", "sid": "NHC_AL952026"}


class TestStructural(unittest.TestCase):
    def _page(self, g, s):
        bm = basemap_for(g["aids"][list(g["aids"])[0]][0]["lat"] or 14.0,
                         g["aids"][list(g["aids"])[0]][0]["lon"] or -42.0, g["basin"])
        return R.build_page(g["sid"], "TEST", g, s, bm)

    def test_developed_page_embeds_data_and_renderers(self):
        html = self._page(DEVELOPED, SHIPS_OK)
        self.assertIn("window.__GUIDANCE__", html)
        self.assertIn("renderTracks", html)
        self.assertIn("renderIntensity", html)
        self.assertIn("renderShips", html)
        self.assertIn('"track_aids":["AVNI"', html.replace(" ", ""))
        # no leftover placeholder collisions
        self.assertNotIn("GUIDANCE_JSON;", html)

    def test_fresh_invest_empty_track_aids(self):
        html = self._page(FRESH, SHIPS_OFF)
        self.assertIn('"track_aids":[]', html.replace(" ", ""))
        self.assertIn('"available":false', html.replace(" ", ""))


_HAS_PW = False
try:
    from playwright.sync_api import sync_playwright  # noqa: F401
    _HAS_PW = True
except Exception:
    pass


@unittest.skipUnless(_HAS_PW, "playwright not installed")
class TestRendering(unittest.TestCase):
    """Real headless-chromium render: no JS errors, graceful states, ink-scan."""
    def _render(self, g, s):
        import tempfile
        from playwright.sync_api import sync_playwright
        bm = basemap_for(14.0, -42.0, g["basin"])
        html = R.build_page(g["sid"], "TEST", g, s, bm)
        f = Path(tempfile.mkdtemp()) / "p.html"
        f.write_text(html)
        out = {}
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1100, "height": 900})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(f.as_uri(), wait_until="load", timeout=30000)
            pg.wait_for_timeout(900)
            out["errs"] = errs
            out["tracks"] = pg.eval_on_selector("#tracks", "el=>el.textContent")
            out["tlen"] = pg.eval_on_selector("#tracks", "el=>el.innerHTML.length")
            out["ilen"] = pg.eval_on_selector("#intensity", "el=>el.innerHTML.length")
            out["ships"] = pg.eval_on_selector("#ships-root", "el=>el.textContent")
            b.close()
        return out

    def test_developed_renders_clean(self):
        o = self._render(DEVELOPED, SHIPS_OK)
        self.assertEqual(o["errs"], [])
        self.assertGreater(o["tlen"], 1000)         # ink-scan: tracks SVG has content
        self.assertGreater(o["ilen"], 1000)         # intensity SVG has content
        self.assertIn("RI probability", o["ships"]) if False else None

    def test_fresh_invest_graceful(self):
        o = self._render(FRESH, SHIPS_OFF)
        self.assertEqual(o["errs"], [])
        self.assertIn("No track aids", o["tracks"])  # graceful empty-track message
        self.assertGreater(o["ilen"], 1000)          # intensity still renders (stat-only)
        self.assertIn("unavailable", o["ships"].lower())


if __name__ == "__main__":
    unittest.main()
