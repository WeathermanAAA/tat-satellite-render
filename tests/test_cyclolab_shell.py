"""Behavioral tests for the CycloLab per-storm shell - the render_page()
template in cyclolab_shell.py and its REAL inline hydration script, driven
through tests/cyclolab_shell_harness.cjs under node + jsdom.

Two layers, both exercising the actual emitted artifact (no re-implementation):

  * Static render contract (pure-python): OG/meta tags, data-cat, baked JSON,
    feed/adv URL substitution, the ENDED variant, and the invest guard.
  * Inline-script behavior (node harness): hydration apply(), category token
    switching + banner motion, section nav, the odometer state machine, and
    the ENDED page's no-polling contract.

The harness loads the page into jsdom with runScripts:'dangerously', stubs the
DOM surfaces the shell really touches (matchMedia -> reduced-motion false,
fetch -> a fixture feed, requestAnimationFrame, SVGElement.getTotalLength), and
prints one JSON snapshot per op. This wrapper feeds it plans and asserts.

Run: cd /tmp/tsr && NODE_PATH=/workspaces/Triple-A-Tropics/node_modules \
       python -m pytest tests/test_cyclolab_shell.py -q
(NODE_PATH is baked into the subprocess env below, so a bare pytest works too.)
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import cyclolab_shell  # noqa: E402
from storm_ids import InvestSidError  # noqa: E402

HARNESS = HERE / "cyclolab_shell_harness.cjs"
FIXTURE = HERE / "fixtures" / "cyclolab" / "synth_storm.json"
NODE = shutil.which("node")

# The harness needs jsdom on NODE_PATH. Honour an inherited NODE_PATH (the
# documented invocation) but fall back to the main repo's node_modules so a
# bare `pytest` invocation works too.
_DEFAULT_NODE_PATH = "/workspaces/Triple-A-Tropics/node_modules"
FEED_URL = "https://cdn.triple-a-tropics.com/feeds/ep_tracks_data.json"


def load_storm() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _node_env() -> dict:
    env = dict(os.environ)
    existing = env.get("NODE_PATH", "")
    if _DEFAULT_NODE_PATH not in existing.split(os.pathsep):
        env["NODE_PATH"] = (
            existing + os.pathsep + _DEFAULT_NODE_PATH if existing
            else _DEFAULT_NODE_PATH)
    return env


def run_harness(html: str, plan) -> list[dict]:
    """Render `html` + a plan through the node harness; return the list of
    per-op snapshot records ({"op": ..., "state": {...}})."""
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "page.html"
        pj = Path(td) / "plan.json"
        page.write_text(html, encoding="utf-8")
        pj.write_text(json.dumps(plan), encoding="utf-8")
        proc = subprocess.run(
            [NODE, str(HARNESS), str(page), str(pj)],
            capture_output=True, text=True, timeout=120, env=_node_env(),
        )
    if proc.returncode != 0:
        raise RuntimeError(f"node harness failed (rc={proc.returncode}):\n"
                           f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


@unittest.skipIf(NODE is None, "node not on PATH")
class TestRenderContract(unittest.TestCase):
    """Static render contract - pure python, no node."""

    def setUp(self):
        self.storm = load_storm()

    def test_og_and_meta_carry_name_and_path(self):
        html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)
        name = self.storm["name"].upper()  # SYNTH
        # og:title + og:description carry the storm name.
        self.assertIn(f'<meta property="og:title" content="{name}', html)
        self.assertIn('<meta property="og:description"', html)
        self.assertIn(name, html)
        # og:url carries the storm's public page path.
        self.assertIn(
            f'<meta property="og:url" '
            f'content="https://triple-a-tropics.com/cyclolab/{self.storm["sid"]}/"',
            html)
        # the page <title> also carries the name.
        self.assertIn(f"<title>{name}", html)

    def test_data_cat_matches_current_category(self):
        html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)
        self.assertEqual(self.storm["current_category"], "C4")
        self.assertIn('<html lang="en" data-cat="C4">', html)

    def test_unknown_category_falls_back_to_TD(self):
        s = copy.deepcopy(self.storm)
        s["current_category"] = "ZZ"
        html = cyclolab_shell.render_page(s, feed_url=FEED_URL)
        self.assertIn('data-cat="TD"', html)

    def test_baked_json_parses_and_is_the_storm(self):
        html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)
        marker = "var BAKED = "
        i = html.index(marker) + len(marker)
        j = html.index(";", i)
        baked = json.loads(html[i:j])
        self.assertEqual(baked["sid"], self.storm["sid"])
        self.assertEqual(len(baked["points"]), len(self.storm["points"]))
        self.assertAlmostEqual(baked["ace"], self.storm["ace"])

    def test_feed_and_adv_urls_land_in_the_script(self):
        adv = "https://cdn.triple-a-tropics.com/feeds/cyclolab/adv/NHC_EP082026.json"
        html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                          adv_url=adv)
        self.assertIn(f'var FEED_URL = "{FEED_URL}"', html)
        self.assertIn(f'var ADV_URL = "{adv}"', html)
        self.assertIn(f'var SID = "{self.storm["sid"]}"', html)

    def test_adv_url_defaults_to_the_adv_key(self):
        html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)
        # cyclolab_pages.adv_key shape when no adv_url is passed.
        self.assertIn('var ADV_URL = "cyclolab/adv/NHC_EP082026.json"', html)

    def test_live_page_polls_ended_page_does_not(self):
        live = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                          ended=False)
        ended = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                           ended=True)
        self.assertIn("var ENDED = false;", live)
        self.assertIn("var ENDED = true;", ended)

    def test_ended_variant_adds_attr_and_text(self):
        ended = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                           ended=True)
        self.assertIn('<html lang="en" data-ended data-cat="C4">', ended)
        self.assertIn("ENDED", ended)  # the frozen-strip copy
        # the non-ended page's <html> tag carries NO data-ended attribute.
        # (The CSS rule `html[data-ended] .ended-strip{}` always mentions the
        #  string, so assert on the opening tag specifically, not a substring.)
        live = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)
        self.assertIn('<html lang="en" data-cat="C4">', live)
        self.assertNotIn('<html lang="en" data-ended', live)

    def test_invest_sid_raises(self):
        # storm_ids contract: invests (90-99) get no page. render_page parses
        # the sid up front, so the guard propagates.
        s = copy.deepcopy(self.storm)
        s["sid"] = "NHC_EP902026"
        with self.assertRaises(InvestSidError):
            cyclolab_shell.render_page(s, feed_url=FEED_URL)


@unittest.skipIf(NODE is None, "node not on PATH")
class TestHydration(unittest.TestCase):
    """apply() drives the live DOM (node + jsdom)."""

    def setUp(self):
        self.storm = load_storm()
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)

    def test_apply_populates_odometers_and_svgs(self):
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [{"op": "apply", "storm": self.storm}]})
        st = recs[-1]["state"]
        # last fix: wind 120 kt, mslp 925 mb; ace 4.821 -> "4.82".
        self.assertEqual(st["odo"]["vmax"], "120")
        self.assertEqual(st["odo"]["mslp"], "925")
        self.assertEqual(st["odo"]["ace"], "4.82")
        # position + last-fix odometers populated (not the em-dash placeholder).
        self.assertEqual(st["odo"]["pos"], "20.4N 141.2W")
        self.assertTrue(st["odo"]["fix"] and st["odo"]["fix"] != "—")
        # both Overview SVGs have rendered children.
        self.assertGreater(st["trackmapChildCount"], 0)
        self.assertGreater(st["chartChildCount"], 0)


@unittest.skipIf(NODE is None, "node not on PATH")
class TestCategorySwitch(unittest.TestCase):
    """setCategory() flips the token + runs the banner morph, once."""

    def setUp(self):
        self.storm = load_storm()
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)

    def test_switch_to_c5_flips_cat_chip_and_animates(self):
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [{"op": "setCategory", "cat": "C5"}]})
        st = recs[-1]["state"]
        self.assertEqual(st["cat"], "C5")
        self.assertEqual(st["chip"], "Category 5")
        # the category-change choreography: crossfade + one shine sweep.
        self.assertIn("xfade", st["bannerClasses"])
        self.assertIn("shine", st["bannerClasses"])

    def test_same_category_is_a_noop_no_class_churn(self):
        # The page boots at C4 (the fixture's current_category). Re-asserting
        # C4 must not animate; switching to C5 then re-asserting C5 (after the
        # operator clears the anim classes) must not re-add them.
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [
                 {"op": "setCategory", "cat": "C4"},   # same as boot -> no churn
                 {"op": "setCategory", "cat": "C5"},   # change -> xfade+shine
                 {"op": "removeBannerClasses"},        # clear them
                 {"op": "setCategory", "cat": "C5"},   # same -> must NOT re-add
             ]})
        same_c4, to_c5, cleared, same_c5 = (r["state"] for r in recs)
        # 1) re-asserting the current category never adds anim classes.
        self.assertEqual(same_c4["cat"], "C4")
        self.assertEqual(same_c4["bannerClasses"], ["banner"])
        # 2) a real change does animate.
        self.assertEqual(to_c5["cat"], "C5")
        self.assertIn("xfade", to_c5["bannerClasses"])
        # 3) re-asserting the same category after a manual clear is stable -
        #    cat unchanged and no class re-add.
        self.assertEqual(cleared["bannerClasses"], ["banner"])
        self.assertEqual(same_c5["cat"], "C5")
        self.assertEqual(same_c5["bannerClasses"], ["banner"])


@unittest.skipIf(NODE is None, "node not on PATH")
class TestSectionNav(unittest.TestCase):
    """openSec() activates a section + its nav button; bad names don't throw."""

    def setUp(self):
        self.storm = load_storm()
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)

    def test_open_models_activates_section_and_nav(self):
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [{"op": "openSec", "name": "models"}]})
        st = recs[-1]["state"]
        self.assertEqual(st["activeSection"], "sec-models")
        self.assertEqual(st["activeNav"], "models")

    def test_unknown_section_is_a_noop_no_throw(self):
        # An unknown name must not throw (the harness would non-zero exit on a
        # script error); the page stays alive and renders a further snapshot.
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [
                 {"op": "openSec", "name": "overview"},
                 {"op": "openSec", "name": "does-not-exist"},
                 {"op": "snapshot"},
             ]})
        # three records came back -> no crash on the bogus name.
        self.assertEqual(len(recs), 3)
        # the page is still hydrated/usable (baked odometers intact).
        self.assertEqual(recs[-1]["state"]["odo"]["vmax"], "120")


@unittest.skipIf(NODE is None, "node not on PATH")
class TestOdometer(unittest.TestCase):
    """The odometer state machine: data-odo + per-digit column transforms."""

    def setUp(self):
        self.storm = load_storm()
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)

    def _one_point_storm(self, wind: float) -> dict:
        s = copy.deepcopy(self.storm)
        s["points"] = [dict(self.storm["points"][0], wind_kt=wind)]
        return s

    def test_vmax_rolls_from_95_to_110(self):
        s95 = self._one_point_storm(95.0)
        s110 = self._one_point_storm(110.0)
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [
                 {"op": "apply", "storm": s95},
                 {"op": "apply", "storm": s110},
             ]})
        at95, at110 = (r["state"] for r in recs)
        # data-odo flips 95 -> 110.
        self.assertEqual(at95["odo"]["vmax"], "95")
        self.assertEqual(at110["odo"]["vmax"], "110")
        # the digit columns' translateY transforms changed (the roll).
        cols95 = at95["odoColsVmax"]
        cols110 = at110["odoColsVmax"]
        self.assertTrue(all(c.startswith("translateY(") for c in cols95))
        self.assertTrue(all(c.startswith("translateY(") for c in cols110))
        self.assertNotEqual(cols95, cols110)


@unittest.skipIf(NODE is None, "node not on PATH")
class TestEndedPage(unittest.TestCase):
    """The frozen ENDED page: visible strip, baked data, and NO poll timer."""

    def setUp(self):
        self.storm = load_storm()

    def test_ended_page_schedules_no_60s_poll(self):
        html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                          ended=True)
        # Drive the stubbed fetch (a feed IS provided) so any poll() that DID
        # run would resolve and try to re-arm the 60s timer - the ended page
        # must not, since `if (!ENDED) poll()` never starts it.
        recs = run_harness(html,
                           {"feed": {"storms": [self.storm]},
                            "ops": [{"op": "snapshot"}]})
        st = recs[-1]["state"]
        self.assertTrue(st["ended"])
        self.assertTrue(st["endedStripVisible"])
        self.assertNotIn(60000, st["scheduledDelays"])
        # baked snapshot still hydrated the page before freezing.
        self.assertEqual(st["odo"]["vmax"], "120")

    def test_live_page_does_schedule_the_60s_poll(self):
        # The contrast: on the live page the poll resolves (stubbed fetch) and
        # re-arms the 60s timer. Proves the ended-page assertion is meaningful.
        html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                          ended=False)
        recs = run_harness(html,
                           {"feed": {"storms": [self.storm]},
                            "ops": [{"op": "snapshot"}]})
        st = recs[-1]["state"]
        self.assertFalse(st["ended"])
        self.assertIn(60000, st["scheduledDelays"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
