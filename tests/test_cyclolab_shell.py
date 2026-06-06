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
class TestIntegratedCard(unittest.TestCase):
    """AD R2: ONE storm-info bug card - structure, canon labels, baked
    heroes (static render contract, pure python)."""

    def setUp(self):
        self.storm = load_storm()
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)

    def test_one_card_banner_then_body_no_second_box(self):
        # the gradient header and the body live inside the SAME .bug card,
        # in order: bug > banner > bug-body(heroes > vitals).
        h = self.html
        i_bug = h.index('<div class="bug">')
        i_banner = h.index('id="banner"')
        i_body = h.index('<div class="bug-body">')
        i_heroes = h.index('<div class="heroes">')
        i_vitals = h.index('id="vitals"')
        self.assertTrue(i_bug < i_banner < i_body < i_heroes < i_vitals)
        # exactly one bug card, one body, one vitals region.
        self.assertEqual(h.count('<div class="bug">'), 1)
        self.assertEqual(h.count('<div class="bug-body">'), 1)
        # the old standalone vitals card chrome is gone: the vitals base
        # rule carries no own background/border (the .bug owns the card).
        import re
        m = re.search(r"\.vitals \{([^}]*)\}", h)
        self.assertIsNotNone(m)
        self.assertNotIn("background", m.group(1))
        self.assertNotIn("border", m.group(1))

    def test_glyph_carries_canon_label_and_stays_stationary(self):
        # fixture is C4 -> canon label "4", baked into the glyph <text>,
        # which sits OUTSIDE the spinning group (only the path spins).
        self.assertRegex(self.html, r'id="glyph-cat"[^>]*')
        i_spin_close = self.html.index("</g>", self.html.index('class="spin"'))
        i_text = self.html.index('id="glyph-cat"')
        self.assertGreater(i_text, i_spin_close)
        self.assertRegex(self.html,
                         r'id="glyph-cat"[^>]*>\s*4\s*</text>')
        # canonical treatment attrs (the tracks-map / placard canon).
        import re
        text_tag = re.search(r'<text id="glyph-cat".*?>', self.html,
                             re.S).group(0)
        # weight 800, not the canon's 900: Metropolis has no Black face,
        # so 900 would silently clamp to 800 - we declare what renders.
        for attr in ('font-weight="800"', 'dominant-baseline="central"'):
            self.assertIn(attr, text_tag)
        # AD R3: the label wears the CATEGORY COLOR via CSS (follows
        # category changes); thin dark stroke for the bright ramps.
        m = re.search(r"\.banner \.glyph text \{([^}]*)\}", self.html)
        self.assertIsNotNone(m)
        self.assertIn("fill: var(--cat-accent)", m.group(1))
        self.assertIn("paint-order: stroke", m.group(1))

    def test_canon_label_parity_python_sweep(self):
        # python _sshs_label == the documented canon for every category,
        # and the baked glyph + Category-hero cells agree with it.
        want = {"TD": "D", "TS": "S", "C1": "1", "C2": "2", "C3": "3",
                "C4": "4", "C5": "5"}
        for cat, label in want.items():
            self.assertEqual(cyclolab_shell._sshs_label(cat), label)
            s = copy.deepcopy(self.storm)
            s["current_category"] = cat
            h = cyclolab_shell.render_page(s, feed_url=FEED_URL)
            self.assertRegex(h, r'id="glyph-cat"[^>]*>\s*' + label +
                             r'\s*</text>')
            # digits bake as fixed cells; letters carry .ch (auto-width);
            # the container carries the accessible value, cells are
            # presentation-only.
            cell_cls = "digit" if label.isdigit() else "digit ch"
            self.assertIn(f'id="odo-cat" aria-label="{label}">'
                          f'<span class="{cell_cls}" aria-hidden="true">'
                          + label, h)

    def test_heroes_bake_vmax_and_category(self):
        # fixture last wind 120 -> three baked digit cells + the
        # accessible value on the container; no-JS render carries real
        # values.
        self.assertIn('id="odo-vmax" aria-label="120">'
                      '<span class="digit" aria-hidden="true">1</span>'
                      '<span class="digit" aria-hidden="true">2</span>'
                      '<span class="digit" aria-hidden="true">0</span>',
                      self.html)
        self.assertIn('class="hero-cap">Max wind<', self.html)
        self.assertIn('class="hero-cap">Category<', self.html)

    def test_vmax_promoted_out_of_the_inline_vitals(self):
        # Max wind is a HERO now - the JS VITALS list must not rebuild it
        # as an inline row (id collision would double-render).
        self.assertNotIn('{ id: "vmax"', self.html)
        # the inline list starts at Min pressure and keeps the AD R2 order.
        i = self.html.index('var VITALS')
        block = self.html[i:i + 600]
        order = [block.index('"mslp"'), block.index('"ace"'),
                 block.index('"pos"'), block.index('"move"'),
                 block.index('"fix"'), block.index('"next"')]
        self.assertEqual(order, sorted(order))

    def test_missing_wind_bakes_em_dash(self):
        s = copy.deepcopy(self.storm)
        s["points"] = [dict(s["points"][-1], wind_kt=None)]
        h = cyclolab_shell.render_page(s, feed_url=FEED_URL)
        self.assertIn('id="odo-vmax" aria-label="—">'
                      '<span class="digit ch" aria-hidden="true">—</span>',
                      h)

    def test_sshs_label_none_and_empty_match_the_js_guard(self):
        # exact JS-parity edge: sshsLabel(null) / sshsLabel("") -> "D".
        self.assertEqual(cyclolab_shell._sshs_label(None), "D")
        self.assertEqual(cyclolab_shell._sshs_label(""), "D")


@unittest.skipIf(NODE is None, "node not on PATH")
class TestIntegratedCardBehavior(unittest.TestCase):
    """AD R2 behavior: category switches drive the glyph label + the
    Category hero through the same canon (node + jsdom)."""

    def setUp(self):
        self.storm = load_storm()
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)

    def test_category_switch_updates_glyph_and_hero(self):
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [
                 {"op": "snapshot"},                  # boot: baked C4
                 {"op": "setCategory", "cat": "C5"},  # number path
                 {"op": "setCategory", "cat": "TS"},  # letter path
             ]})
        boot, c5, ts = (r["state"] for r in recs)
        # boot keeps the BAKED canon label (setCategory early-returns on
        # the same cat - the python bake must already be right).
        self.assertEqual(boot["glyphCat"].strip(), "4")
        self.assertEqual(boot["heroCatText"], "4")
        self.assertEqual(c5["glyphCat"], "5")
        self.assertEqual(c5["odo"]["cat"], "5")
        self.assertEqual(ts["glyphCat"], "S")
        self.assertEqual(ts["odo"]["cat"], "S")

    def test_vmax_hero_still_rides_the_odometer(self):
        # the promoted hero keeps the odometer state machine (data-odo +
        # rolling columns) - same id, same discipline - and the
        # accessible value tracks it (the rolling 0-9 stacks are
        # aria-hidden presentation).
        s = copy.deepcopy(self.storm)
        s["points"] = [dict(self.storm["points"][-1], wind_kt=140.0)]
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [{"op": "apply", "storm": s}]})
        st = recs[-1]["state"]
        self.assertEqual(st["odo"]["vmax"], "140")
        self.assertTrue(all(c.startswith("translateY(")
                            for c in st["odoColsVmax"]))
        self.assertEqual(st["odoAriaVmax"], "140")

    def test_unknown_category_clamps_to_td_everywhere(self):
        # the hydrated path applies the SAME validation as render_page:
        # garbage current_category clamps to TD on every surface (ramp
        # token, chip, glyph label, Category hero) instead of leaking
        # raw text + the off-category default-blue ramp.
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [{"op": "setCategory", "cat": "ZZ"}]})
        st = recs[-1]["state"]
        self.assertEqual(st["cat"], "TD")
        # S4-AD1 #10: at TD the chip is HIDDEN (it would duplicate the
        # type word); the clamp shows as the hidden chip + TD ramp.
        self.assertFalse(st["chipShown"])
        self.assertEqual(st["glyphCat"], "D")
        self.assertEqual(st["odo"]["cat"], "D")


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
        self.assertTrue(st["chipShown"])   # C1-5 chips stay (#10)
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



@unittest.skipIf(NODE is None, "node not on PATH")
class TestStage3Mounts(unittest.TestCase):
    """Stage 3: the Satellite + Models mounts (CYCLOLAB_DESIGN §7.2/§7.3).
    Models = the componentized /models/ HafsViewer constructed with THIS
    page's element table + the storm lock from the id join; Satellite =
    the storm-scoped floater viewer. Both are LAZY - nothing fetched, no
    script injected, no viewer constructed until the tab first opens."""

    def setUp(self):
        self.storm = load_storm()          # NHC_EP082026 -> 08e / ep082026
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                               loader="")

    @staticmethod
    def _floaters_top(include=True):
        entry = {"id": "ep082026", "slug": "ep08", "name": "SYNTH",
                 "basin": "EP", "bands": ["ir", "wv_up"],
                 "manifest": "floaters/ep08/manifest.json"}
        other = {"id": "ep992026", "slug": "ep99", "name": "OTHER",
                 "basin": "EP", "bands": ["ir"],
                 "manifest": "floaters/ep99/manifest.json"}
        return {"generated_utc": "2026-06-06T20:00:00Z",
                "storms": ([entry] if include else []) + [other]}

    @staticmethod
    def _floater_storm():
        def fr(hhmm):
            return {"t": f"2026-06-06T{hhmm}:00Z",
                    "key": f"floaters/ep08/x/{hhmm.replace(':', '')}.png"}
        return {"id": "ep082026", "slug": "ep08", "name": "SYNTH",
                "basin": "EP",
                "bands": {
                    "ir": {"label": "Clean IR",
                           "frames": [fr("10:00"), fr("10:10"), fr("10:20")]},
                    "wv_up": {"label": "WV (upper)",
                              "frames": [fr("10:02"), fr("10:12"),
                                         fr("10:40")]},
                }}

    def test_mounts_are_lazy_until_tab_opens(self):
        recs = run_harness(self.html, {
            "hafs_stub": True, "floaters": self._floaters_top(),
            "floater_storm": self._floater_storm(),
            "ops": [{"op": "snapshot"}]})
        st = recs[-1]["state"]["stage3"]
        self.assertIsNone(st["hafsCtor"])
        self.assertFalse(st["hafsScriptInjected"])
        self.assertFalse([u for u in st["fetched"] if "floaters" in u],
                         "satellite fetched before the tab opened")

    def test_models_tab_constructs_storm_locked_viewer(self):
        recs = run_harness(self.html, {
            "hafs_stub": True,
            "ops": [{"op": "openSec", "name": "models"}]})
        h = recs[-1]["state"]["stage3"]["hafsCtor"]
        self.assertIsNotNone(h, "HafsViewer not constructed on tab open")
        self.assertEqual(h["rootId"], "cl-hafs-root")
        self.assertEqual(h["stormLock"], "08e")
        self.assertEqual(
            h["manifestUrl"],
            "https://cdn.triple-a-tropics.com/models/hafs/manifest.json")
        self.assertEqual(h["assetBase"],
                         "https://cdn.triple-a-tropics.com/models/hafs/")
        self.assertTrue(h["elsWired"],
                        "els table not wired to the cl-hafs-* elements")
        self.assertEqual(len(h["elsKeys"]), 24)

    def test_models_without_stub_lazy_loads_the_house_script(self):
        recs = run_harness(self.html, {
            "ops": [{"op": "openSec", "name": "models"}]})
        st = recs[-1]["state"]["stage3"]
        self.assertTrue(st["hafsScriptInjected"])
        self.assertIn("https://triple-a-tropics.com/models/hafs.js",
                      st["hafsScriptSrc"])
        self.assertIsNone(st["hafsCtor"],
                          "ctor must wait for the script's onload")

    def test_satellite_tab_mounts_newest_frame_first(self):
        recs = run_harness(self.html, {
            "floaters": self._floaters_top(),
            "floater_storm": self._floater_storm(),
            "ops": [{"op": "openSec", "name": "satellite"}]})
        st = recs[-1]["state"]["stage3"]
        self.assertEqual(st["sat"]["band"], "ir")
        self.assertEqual(st["sat"]["frames"], 3)
        self.assertEqual(st["sat"]["idx"], 2, "newest frame first")
        self.assertIn("floaters/ep08/x/1020.png", st["satImgSrc"])
        self.assertEqual([b["slug"] for b in st["satBands"]],
                         ["ir", "wv_up"])
        self.assertTrue(st["satBands"][0]["active"])
        self.assertIn("2026-06-06 10:20Z", st["satTime"])

    def test_satellite_band_switch_keeps_the_moment(self):
        # at ir 10:20, switching to wv_up must pick the NEAREST frame
        # (10:12, idx 1) - not the band's newest (10:40, idx 2). The
        # availability-aware scrub on a time axis.
        recs = run_harness(self.html, {
            "floaters": self._floaters_top(),
            "floater_storm": self._floater_storm(),
            "ops": [{"op": "openSec", "name": "satellite"},
                    {"op": "clickSatBand", "slug": "wv_up"}]})
        st = recs[-1]["state"]["stage3"]
        self.assertEqual(st["sat"]["band"], "wv_up")
        self.assertEqual(st["sat"]["idx"], 1,
                         "band switch must keep the moment (nearest frame)")
        self.assertIn("floaters/ep08/x/1012.png", st["satImgSrc"])

    def test_satellite_absent_storm_shows_empty_state(self):
        recs = run_harness(self.html, {
            "floaters": self._floaters_top(include=False),
            "floater_storm": self._floater_storm(),
            "ops": [{"op": "openSec", "name": "satellite"}]})
        st = recs[-1]["state"]["stage3"]
        self.assertTrue(st["satEmptyShown"])
        self.assertEqual(st["sat"]["frames"], 0)

    def test_leaving_satellite_tab_pauses_playback(self):
        # pause-on-hide: a hidden tab must not keep its 5fps loop alive.
        recs = run_harness(self.html, {
            "floaters": self._floaters_top(),
            "floater_storm": self._floater_storm(),
            "ops": [{"op": "openSec", "name": "satellite"},
                    {"op": "clickSatPlay"},
                    {"op": "openSec", "name": "overview"}]})
        playing_after_click = recs[1]["state"]["stage3"]["sat"]["playing"]
        playing_after_leave = recs[2]["state"]["stage3"]["sat"]["playing"]
        self.assertTrue(playing_after_click)
        self.assertFalse(playing_after_leave,
                         "leaving the tab must pause satellite playback")

    def test_band_emptied_by_prune_shows_empty_state(self):
        man = self._floater_storm()
        man["bands"]["wv_up"]["frames"] = []     # server prune left the key
        recs = run_harness(self.html, {
            "floaters": self._floaters_top(), "floater_storm": man,
            "ops": [{"op": "openSec", "name": "satellite"},
                    {"op": "clickSatBand", "slug": "wv_up"}]})
        st = recs[-1]["state"]["stage3"]
        self.assertTrue(st["satEmptyShown"],
                        "empty band must show the empty state, not freeze")
        self.assertEqual(st["sat"]["frames"], 0)

    def test_baked_ids_ride_the_template(self):
        self.assertIn('var HAFS_ID = "08e"', self.html)
        self.assertIn('var FLOATER_ID = "ep082026"', self.html)



class TestOgImageTag(unittest.TestCase):
    def test_og_image_emitted_only_with_url(self):
        storm = load_storm()
        with_url = cyclolab_shell.render_page(
            storm, feed_url=FEED_URL,
            og_image_url="https://cdn.x/og/NHC_EP082026.png")
        self.assertIn('<meta property="og:image" content="https://cdn.x/'
                      'og/NHC_EP082026.png">', with_url)
        without = cyclolab_shell.render_page(storm, feed_url=FEED_URL)
        self.assertNotIn("og:image", without)


@unittest.skipIf(NODE is None, "node not on PATH")
class TestStage4Advisories(unittest.TestCase):
    """Stage 4: THE CONE reveal + the intensity cone + advisory text
    panels (CYCLOLAB_DESIGN §7.4/§8). The reveal is structural here
    (choreography elements + stagger delays + the one permitted spin
    loop); the rendered look is the packet's job."""

    def setUp(self):
        self.storm = load_storm()      # EP08 -> EP registry entry baked
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                               loader="")

    @staticmethod
    def _adv(method="official-cone", points=None, text=None):
        if points is None:
            points = [
                {"tau_h": 0, "lat": 15.0, "lon": -135.0,
                 "intensity_kt": 30, "dev_label": "TD",
                 "valid_utc": "2026-06-06T21:00:00Z"},
                {"tau_h": 12, "lat": 15.5, "lon": -136.0,
                 "intensity_kt": 35, "dev_label": "TS",
                 "valid_utc": "2026-06-07T09:00:00Z"},
                {"tau_h": 24, "lat": 16.0, "lon": -137.2,
                 "intensity_kt": 45, "dev_label": "TS",
                 "valid_utc": "2026-06-07T21:00:00Z"},
                {"tau_h": 48, "lat": 17.0, "lon": -139.6,
                 "intensity_kt": 70, "dev_label": "HU",
                 "valid_utc": "2026-06-08T21:00:00Z"},
            ]
        ring = [[-140.0, 13.0], [-134.0, 13.0], [-133.5, 18.0],
                [-140.5, 18.5], [-140.0, 13.0]]
        return {"sid": "NHC_EP082026", "advisory": 21,
                "issued_utc": "2026-06-06T21:00:00Z",
                "source": "nhc", "method": method,
                "cone": ring, "points": points,
                "text": text if text is not None else
                {"tcp": "BULLETIN\nTEST PUBLIC ADVISORY BODY",
                 "tcd": "TEST DISCUSSION BODY",
                 "tcp_url": "u1", "tcd_url": "u2"}}

    def _run(self, ops, adv=None):
        plan = {"ops": [{"op": "applyAdvisory",
                         "adv": adv or self._adv()}] + ops}
        return run_harness(self.html, plan)

    def test_cone_reveal_structure_and_stagger(self):
        recs = self._run([{"op": "openSec", "name": "advisories"}])
        a = recs[-1]["state"]["stage3"]["adv"]
        # jsdom has no WAAPI -> the final-frame branch engages: the
        # growth-front mask stroke sits at dashoffset 0 (fully drawn).
        self.assertEqual(a["coneReveal"], "final")
        self.assertEqual(str(a["coneFrontOffset"]), "0")
        self.assertEqual(a["coneIcons"], 4)       # one per forecast point
        self.assertGreaterEqual(a["coneSpinners"], 4)
        # pops RIDE THE WAVEFRONT: NOW during the hold (~0.4s), then
        # strictly increasing with along-track distance, all after the
        # 1s hold (S4-AD1 #1).
        det = a["coneIconsDetail"]
        self.assertEqual([d["tau"] for d in det], [0, 12, 24, 48])
        self.assertAlmostEqual(det[0]["delay"], 0.4, places=2)
        self.assertGreater(det[1]["delay"], 1.0)
        self.assertLess(det[1]["delay"], det[2]["delay"])
        self.assertLess(det[2]["delay"], det[3]["delay"])
        self.assertLessEqual(det[3]["delay"], 4.5)
        # every point carries a placard; none overlap (S4-AD1 #7)
        pls = a["conePlacards"]
        self.assertEqual(len(pls), 4)
        for i in range(len(pls)):
            for j in range(i + 1, len(pls)):
                p, q = pls[i], pls[j]
                clear = (p["x"] + p["w"] <= q["x"] or
                         q["x"] + q["w"] <= p["x"] or
                         p["y"] + p["h"] <= q["y"] or
                         q["y"] + q["h"] <= p["y"])
                self.assertTrue(clear, f"placards {i}/{j} overlap")
        # basemap + auto-fit viewport (S4-AD1 #2/#3)
        self.assertGreaterEqual(a["coneGraticule"], 2)
        self.assertTrue(a["coneViewBox"].startswith("0 0 1000 "))
        self.assertFalse(a["coneEmptyShown"])

    def test_non_tropical_points_render_white_with_caption(self):
        # S4-AD1 #4/#5: post-tropical/remnant taus render WHITE, carry
        # NO category letter, and the caption gains the one-line key.
        adv = self._adv(points=[
            {"tau_h": 0, "lat": 15.0, "lon": -135.0,
             "intensity_kt": 30, "dev_label": "TD", "valid_utc": "x"},
            {"tau_h": 12, "lat": 15.5, "lon": -136.0,
             "intensity_kt": 35, "dev_label": "TS", "valid_utc": "x"},
            {"tau_h": 24, "lat": 16.0, "lon": -137.2,
             "intensity_kt": 35, "dev_label": "PT", "valid_utc": "x"},
            {"tau_h": 48, "lat": 17.0, "lon": -139.6,
             "intensity_kt": 25, "dev_label": "EX", "valid_utc": "x"},
        ])
        recs = self._run([{"op": "openSec", "name": "advisories"}], adv=adv)
        a = recs[-1]["state"]["stage3"]["adv"]
        det = a["coneIconsDetail"]
        self.assertEqual([d["tropical"] for d in det],
                         [True, True, False, False])
        self.assertEqual([d["hasCatLabel"] for d in det],
                         [True, True, False, False],
                         "white icons carry no SS label")
        self.assertIn("White icons = forecast non-tropical", a["coneNote"])

    def test_all_tropical_omits_the_white_caption(self):
        recs = self._run([{"op": "openSec", "name": "advisories"}])
        self.assertNotIn("White icons",
                         recs[-1]["state"]["stage3"]["adv"]["coneNote"])

    def test_bunched_taus_never_overlap_placards(self):
        # S4-AD1 #7: taus bunched within a fraction of a degree - the
        # collision pass must still separate every placard.
        adv = self._adv(points=[
            {"tau_h": t, "lat": 15.0 + 0.05 * i, "lon": -135.0 - 0.07 * i,
             "intensity_kt": 30 + i, "dev_label": "TS", "valid_utc": "x"}
            for i, t in enumerate((0, 12, 24, 36, 48, 60))])
        recs = self._run([{"op": "openSec", "name": "advisories"}], adv=adv)
        pls = recs[-1]["state"]["stage3"]["adv"]["conePlacards"]
        self.assertEqual(len(pls), 6)
        for i in range(len(pls)):
            for j in range(i + 1, len(pls)):
                p, q = pls[i], pls[j]
                clear = (p["x"] + p["w"] <= q["x"] or
                         q["x"] + q["w"] <= p["x"] or
                         p["y"] + p["h"] <= q["y"] or
                         q["y"] + q["h"] <= p["y"])
                self.assertTrue(clear,
                                f"bunched placards {i}/{j} overlap: {p} {q}")

    def test_official_cone_copy(self):
        recs = self._run([{"op": "openSec", "name": "advisories"}])
        a = recs[-1]["state"]["stage3"]["adv"]
        self.assertIn("Official NHC forecast cone", a["coneNote"])
        self.assertIn("advisory 21", a["coneNote"])
        self.assertIn("official National Hurricane Center",
                      a["coneMethodBody"])

    def test_derived_cone_copy_wp_disclosure(self):
        adv = self._adv(method="derived-mean-error-jtwc-wpac-mean-2015")
        recs = self._run([{"op": "openSec", "name": "advisories"}], adv=adv)
        a = recs[-1]["state"]["stage3"]["adv"]
        self.assertIn("not an official JTWC product", a["coneNote"])
        self.assertIn("no\u2009official" if False else "cone of uncertainty",
                      a["coneMethodBody"])
        self.assertIn("jtwc-wpac-mean-2015", a["coneMethodBody"])
        self.assertIn("interpolated", a["coneMethodBody"])

    def test_intensity_cone_envelope_matches_python_math(self):
        # the in-page JS mirrors cyclolab_intensity.envelope - pin the
        # parity on the EP entry (12h 5.7 / 48h 12.9, tau 0 anchored).
        import cyclolab_intensity as ci
        recs = self._run([{"op": "openSec", "name": "advisories"}])
        a = recs[-1]["state"]["stage3"]["adv"]
        self.assertTrue(a["intRendered"])
        rows = a["intRows"]
        self.assertEqual([r["tau"] for r in rows], [0, 12, 24, 48])
        py = ci.envelope(self._adv()["points"], ci.basin_entry("EP"))
        for js, p in zip(rows, py):
            self.assertAlmostEqual(js["upper"], p["upper"], places=6)
            self.assertAlmostEqual(js["lower"], p["lower"], places=6)
        self.assertEqual(rows[0]["upper"], rows[0]["lower"])  # tau-0 apex
        self.assertIn("nhc-ofcl-5yr-2020-2024", a["intMethodBody"])
        self.assertIn("not a probabilistic bound", a["intMethodBody"])

    def test_intensity_honesty_guard_no_registry_entry(self):
        # an unregistered basin bakes INTENSITY_ERR = null -> the labeled
        # panel renders and NO envelope is drawn. Simulate by patching
        # the baked constant (basin registries are keyed by real basins).
        html = self.html.replace(
            'var INTENSITY_ERR = {', 'var INTENSITY_ERR = null; var _x = {')
        plan = {"ops": [{"op": "applyAdvisory", "adv": self._adv()},
                        {"op": "openSec", "name": "advisories"}]}
        recs = run_harness(html, plan)
        a = recs[-1]["state"]["stage3"]["adv"]
        self.assertTrue(a["intMissingShown"])
        self.assertIn("No published intensity-error statistics",
                      a["intMissingText"])
        self.assertIn("rather than borrowed or invented",
                      a["intMissingText"])
        self.assertFalse(a["intRendered"])

    def test_advisory_text_panels_and_toggle(self):
        recs = self._run([
            {"op": "openSec", "name": "advisories"},
            {"op": "clickAdvTextTab", "prod": "tcd"},
            {"op": "clickAdvTextTab", "prod": "tcp"}])
        states = [r["state"]["stage3"]["adv"]["advText"] for r in recs[-3:]]
        self.assertIn("TEST PUBLIC ADVISORY BODY", states[0])
        self.assertEqual(states[1], "TEST DISCUSSION BODY")
        self.assertIn("TEST PUBLIC ADVISORY BODY", states[2])

    def test_missing_text_products_show_placeholder(self):
        adv = self._adv(text={"tcp_url": "u1", "tcd_url": "u2"})
        recs = self._run([{"op": "openSec", "name": "advisories"}], adv=adv)
        self.assertIn("not available",
                      recs[-1]["state"]["stage3"]["adv"]["advText"])

    def test_no_advisory_shows_cone_empty_state(self):
        plan = {"ops": [{"op": "openSec", "name": "advisories"}]}
        recs = run_harness(self.html, plan)
        a = recs[-1]["state"]["stage3"]["adv"]
        self.assertTrue(a["coneEmptyShown"])
        self.assertEqual(a["coneIcons"], 0)

    def test_reveal_replays_on_every_open(self):
        # re-opening rebuilds the SVG (fresh nodes re-arm the CSS
        # animations) - prove it by node identity proxy: icon count
        # stays right after a second open following a detour.
        recs = self._run([
            {"op": "openSec", "name": "advisories"},
            {"op": "openSec", "name": "overview"},
            {"op": "openSec", "name": "advisories"}])
        self.assertEqual(recs[-1]["state"]["stage3"]["adv"]["coneIcons"], 4)
        self.assertEqual(
            recs[-1]["state"]["stage3"]["adv"]["coneReveal"], "final")
        self.assertEqual(
            str(recs[-1]["state"]["stage3"]["adv"]["coneFrontOffset"]),
            "0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
