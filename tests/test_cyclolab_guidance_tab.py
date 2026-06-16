"""Integration test for the CycloLab Model-Guidance tab (Stage B ported into
render_page). Structural checks always run; the rendering assertions run under
Playwright (developed storm via routed fixtures / graceful no-guidance)."""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cyclolab_shell as S

STORM = {"sid": "NHC_EP012026", "name": "TEST-ONE", "current_category": "TD",
         "points": [{"lat": 12.0, "lon": -120.0, "wind_kt": 35, "pressure_mb": 1000}]}
FEED = "https://cdn.triple-a-tropics.com/global_storms.geojson"


def _ramp(v0, dv, n=9, pos=True):
    return [{"tau": i * 12, "lat": (12 + 0.3 * i) if pos else None,
             "lon": (-120 - 0.4 * i) if pos else None,
             "vmax": max(15, v0 + dv * i), "mslp": 1005 - i} for i in range(n)]


GUIDANCE = {"init_time": "2026-06-16T00:00:00Z", "init_cycle": "2026061600",
            "aids": {"AVNI": _ramp(25, 5), "HWFI": _ramp(25, 7), "TVCN": _ramp(25, 6),
                     "HCCA": _ramp(25, 6), "DSHP": _ramp(25, 4), "IVCN": _ramp(25, 5)},
            "present_aids": ["AVNI", "HWFI", "TVCN", "HCCA", "DSHP", "IVCN"],
            "track_aids": ["AVNI", "HWFI", "TVCN", "HCCA"],
            "intensity_aids": ["AVNI", "HWFI", "DSHP", "IVCN"],
            "consensus": ["TVCN", "HCCA", "IVCN"]}
SHIPS = {"available": True, "header": {"id_line": "TEST EP012026"}, "taus": [0, 12, 24, 48],
         "env_series": {"SHEAR (KT)": [5, 8, 12, 20], "SST (C)": [29, 29, 28, 27]},
         "storm_type": ["TROP"] * 4, "prelim_ri_prob": 3.0,
         "ri_matrix": {"cols": ["20/12"], "rows": {"Consensus": {"20/12": 1.0}}},
         "ahi": {"value": 0, "verdict": "NOT ANNULAR"}}


class TestStructure(unittest.TestCase):
    def setUp(self):
        self.html = S.render_page(STORM, feed_url=FEED)

    def test_guidance_merged_into_models_tab(self):
        # Phase 3b: model guidance lives INSIDE the Models tab now; the
        # standalone Guidance tab + section are gone.
        for m in ('id="gtracks"', 'id="gintensity"', 'id="gships-root"',
                  "function initGuidance"):
            self.assertIn(m, self.html, m)
        self.assertNotIn('data-sec="guidance"', self.html)
        self.assertNotIn('id="sec-guidance"', self.html)
        # the guidance cards are physically inside the Models section
        i_models = self.html.index('id="sec-models"')
        i_adv = self.html.index('id="sec-advisories"')
        i_gtracks = self.html.index('id="gtracks"')
        self.assertTrue(i_models < i_gtracks < i_adv,
                        "guidance cards must live inside the Models section")
        # opening the Models tab hydrates BOTH HAFS and guidance
        self.assertIn("initModels(); initGuidance();", self.html)

    def test_palette_b_locked_no_options_board(self):
        # tracks color = SSHWS category of peak wind (palette B); the guidance BLOCK
        # itself must not reference WIND_TIER (no fork), and the live page has no
        # options board (that was the held review only).
        self.assertIn("SSHS[sshsCat(gPeak", self.html)
        start = self.html.index("Model guidance (Stage B)")
        end = self.html.index("section nav (lazy init", start)
        block = self.html[start:end]
        self.assertNotIn("WIND_TIER", block)
        self.assertNotIn("options board", self.html)

    @unittest.skipUnless(shutil.which("node"), "node not on PATH")
    def test_embedded_js_parses(self):
        scripts = re.findall(r"<script>(.*?)</script>", self.html, re.S)
        big = max(scripts, key=len)
        f = Path(tempfile.mkdtemp()) / "s.js"
        f.write_text(big)
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


_HAS_PW = False
try:
    from playwright.sync_api import sync_playwright  # noqa: F401
    _HAS_PW = True
except Exception:
    pass


@unittest.skipUnless(_HAS_PW, "playwright not installed")
class TestRender(unittest.TestCase):
    def _open_tab(self, route_status):
        from playwright.sync_api import sync_playwright
        html = S.render_page(STORM, feed_url=FEED)
        f = Path(tempfile.mkdtemp()) / "p.html"
        f.write_text(html)
        out = {}
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))

            def route(r):
                u = r.request.url
                if route_status != 200:
                    return r.fulfill(status=route_status, body="")
                if "guidance.json" in u:
                    return r.fulfill(status=200, content_type="application/json", body=json.dumps(GUIDANCE))
                if "ships.json" in u:
                    return r.fulfill(status=200, content_type="application/json", body=json.dumps(SHIPS))
                return r.continue_()
            pg.route("**/cyclolab/**", route)
            pg.goto(f.as_uri(), wait_until="load", timeout=40000)
            pg.wait_for_timeout(500)
            # Phase 3b: guidance hydrates when the Models tab opens (HAFS loads
            # alongside; its external script 404s under file:// and degrades
            # gracefully via fail(), so it never errors the guidance render).
            pg.click('button[data-sec="models"]')
            pg.wait_for_timeout(1300)
            out["errs"] = errs
            out["tracks_len"] = pg.eval_on_selector("#gtracks", "e=>e.innerHTML.length")
            out["inten_len"] = pg.eval_on_selector("#gintensity", "e=>e.innerHTML.length")
            out["ships"] = pg.eval_on_selector("#gships-root", "e=>e.textContent")
            out["empty"] = pg.eval_on_selector("#gtracks-empty", "e=>e.style.display")
            b.close()
        return out

    def test_developed_renders(self):
        o = self._open_tab(200)
        self.assertEqual(o["errs"], [])
        self.assertGreater(o["tracks_len"], 1000)   # spaghetti drawn
        self.assertGreater(o["inten_len"], 1000)    # intensity drawn
        self.assertIn("RI probability", o["ships"])

    def test_no_guidance_graceful(self):
        o = self._open_tab(403)                     # R2 absent = storm not active
        self.assertEqual(o["errs"], [])
        self.assertEqual(o["empty"], "block")       # "No model guidance" stub shown
        self.assertIn("unavailable", o["ships"].lower())


if __name__ == "__main__":
    unittest.main()
