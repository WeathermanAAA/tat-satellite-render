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
RADII_FIXTURE = HERE / "fixtures" / "cyclolab" / "synth_storm_radii.json"
NODE = shutil.which("node")


def load_radii_storm() -> dict:
    return json.loads(RADII_FIXTURE.read_text(encoding="utf-8"))

# The harness needs jsdom on NODE_PATH. Honour an inherited NODE_PATH (the
# documented invocation) but fall back to the main repo's node_modules so a
# bare `pytest` invocation works too.
_DEFAULT_NODE_PATH = "/workspaces/Triple-A-Tropics/node_modules"
FEED_URL = "https://cdn.triple-a-tropics.com/feeds/ep_tracks_data.json"

# final-gate-2 #1: the per-storm SST hero meta the poller's
# SstHeroWriter writes (cyclolab_sst.py) - the plan fixture the harness
# serves at SST_BASE/meta.json.
SST_META = {
    "sid": "NHC_EP082026",
    "center": {"lat": 20.4, "lon": -141.2},
    "box": {"hw_lon": 9.0, "hw_lat": 5.175},
    "px": [1200, 690],
    "valid_date": "2026-06-05",
    "updated_utc": "2026-06-06T12:00:00Z",
    "layers": [
        {"slug": "actual", "label": "SST",
         "title": "Sea surface temperature",
         "field": "sea-surface temperature (°C)", "file": "actual.png",
         "source": "NOAA Coral Reef Watch CoralTemp v3.1 (5 km)",
         "valid": "2026-06-05"},
        {"slug": "anomaly", "label": "Anomaly", "title": "SST anomaly",
         "field": "SST anomaly vs the site-wide 1991–2020 baseline (°C)",
         "file": "anomaly.png",
         "source": ("NOAA Coral Reef Watch CoralTemp v3.1 (5 km) minus "
                    "the Triple-A-Tropics 1991–2020 day-of-year "
                    "climatology"),
         "valid": "2026-06-05"},
    ],
}


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
            # final-gate-3 #2: the Category hero bakes as PLAIN TEXT (no
            # digit cells), with the accessible value on the container.
            self.assertIn(f'id="odo-cat" aria-label="{label}">{label}<', h)

    def test_heroes_bake_vmax_and_category(self):
        # final-gate-3 #2: fixture last wind 120 bakes as plain text
        # (no digit cells) with the accessible value on the container.
        self.assertIn('id="odo-vmax" aria-label="120">120<', self.html)
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
        self.assertIn('id="odo-vmax" aria-label="—">—<', h)

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

    def test_vmax_hero_is_plain_text_with_a11y(self):
        # final-gate-3 #2: the promoted hero is PLAIN TEXT - the value,
        # the rendered textContent, and the accessible label all agree,
        # with no digit cells or rolling columns left.
        s = copy.deepcopy(self.storm)
        s["points"] = [dict(self.storm["points"][-1], wind_kt=140.0)]
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.storm]},
             "ops": [{"op": "apply", "storm": s}]})
        st = recs[-1]["state"]
        self.assertEqual(st["odo"]["vmax"], "140")
        self.assertEqual(st["odoText"]["vmax"], "140")
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
             "sst_meta": SST_META,
             "ops": [{"op": "apply", "storm": self.storm}]})
        st = recs[-1]["state"]
        # last fix: wind 120 kt, mslp 925 mb; ace 4.821 -> "4.82".
        self.assertEqual(st["odo"]["vmax"], "120")
        self.assertEqual(st["odo"]["mslp"], "925")
        self.assertEqual(st["odo"]["ace"], "4.82")
        # position + last-fix odometers populated (not the em-dash placeholder).
        self.assertEqual(st["odo"]["pos"], "20.4N 141.2W")
        self.assertTrue(st["odo"]["fix"] and st["odo"]["fix"] != "—")
        # the hero rendered from the PER-STORM render family
        # (final-gate-2 #1): the img points at the baked SST base, the
        # spinning glyph carries the category label, and the caption
        # discloses source + field + valid date.
        hero = st["hero"]
        self.assertIn("/sst/actual.png", hero["imgUrl"])
        self.assertTrue(hero["imgShown"])
        self.assertEqual(hero["layers"], ["actual", "anomaly"])
        self.assertEqual(hero["activeLayer"], "actual")
        self.assertGreater(hero["glyphHtml"], 50)
        self.assertEqual(hero["glyphLabel"], "4")   # C4 synth fixture
        self.assertIn("Coral Reef Watch", hero["caption"])
        self.assertIn("valid 2026-06-05", hero["caption"])
        self.assertIn("storm-centered", hero["caption"])
        self.assertTrue(hero["sub"].endswith("SYNTH"))
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
class TestPlainTextStats(unittest.TestCase):
    """final-gate-3 #2: stats are PLAIN STATIC TEXT. A value change swaps
    the text and plays one subtle fade-in; there is NO rolling, no digit
    cell, no clip machinery anywhere in the DOM."""

    def setUp(self):
        self.storm = load_storm()
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)

    def _one_point_storm(self, wind: float) -> dict:
        s = copy.deepcopy(self.storm)
        s["points"] = [dict(self.storm["points"][0], wind_kt=wind)]
        return s

    def test_value_change_swaps_plain_text_and_fades(self):
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
        # the rendered text IS the value - plain, no cells.
        self.assertEqual(at95["odoText"]["vmax"], "95")
        self.assertEqual(at110["odoText"]["vmax"], "110")
        self.assertEqual(at110["odo"]["vmax"], "110")
        # the only motion left: the fade-in class on the CHANGE.
        self.assertTrue(at110["odoSwapVmax"],
                        "a value change should play the subtle fade")

    def test_no_rolling_machinery_in_the_template(self):
        # the structural proof the odometer is gone: no cell/strip/anchor
        # classes and no clip CSS are ever emitted - there is nothing in
        # the served template that could shear a baseline or ghost a
        # neighbour. (The DOM can only contain what the template + the
        # plain-text odoSet write, and odoSet writes only textContent.)
        for needle in ('class="digit col"', 'class="strip"',
                       'class="anchor"', '.odo .col', '.odo .strip',
                       'odoRoll', 'odoSettle'):
            self.assertNotIn(needle, self.html,
                             f"odometer remnant in template: {needle}")


@unittest.skipIf(NODE is None, "node not on PATH")
class TestWindUnits(unittest.TestCase):
    """final-gate-3 #3: DISPLAY-ONLY wind-unit conversion, applied
    EVERYWHERE (hero, vitals movement, W&P axis label, cone placards),
    persisted + reopenable; the canonical feed value stays in knots."""

    def setUp(self):
        self.storm = load_storm()          # synth vmax 120 kt
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)
        self.adv = json.loads(
            (FIXTURE.parent / "live_adv_ep17.json").read_text())

    def _run(self, ops):
        return run_harness(self.html,
                           {"feed": {"storms": [self.storm]},
                            "adv": self.adv, "ops": ops})

    def test_default_is_kt_and_hero_shows_knots(self):
        st = self._run([{"op": "apply", "storm": self.storm}])[-1]["state"]
        self.assertEqual(st["windUnits"], "kt")
        self.assertEqual(st["vmaxText"], "120")
        self.assertEqual(st["vmaxUnit"], "kt")
        self.assertIn("wind kt", st["chartLabel"])

    def test_switch_to_mph_converts_every_surface(self):
        recs = self._run([{"op": "apply", "storm": self.storm},
                          {"op": "openSec", "name": "advisories"},
                          {"op": "setWindUnits", "unit": "mph"}])
        st = recs[-1]["state"]
        self.assertEqual(st["windUnits"], "mph")
        # 120 kt -> 138.09 mph -> ROUNDED to nearest 5 = 140 (FG-R3 #4,
        # NHC-style; every converted display ends in 0/5).
        self.assertEqual(st["vmaxText"], "140")
        self.assertEqual(st["vmaxUnit"], "mph")
        self.assertIn("wind mph", st["chartLabel"])
        # cone placards carry the converted unit, never a bare "kt"
        labels = [p["label"] for p in st["stage3"]["adv"]["conePlacards"]]
        self.assertTrue(labels)
        self.assertTrue(all("mph" in s for s in labels), labels)
        self.assertFalse(any("kt" in s for s in labels), labels)

    def test_kmh_conversion_is_correct(self):
        recs = self._run([{"op": "apply", "storm": self._mk(100)},
                          {"op": "setWindUnits", "unit": "kmh"}])
        st = recs[-1]["state"]
        # 100 kt -> 185 km/h (1.852)
        self.assertEqual(st["vmaxText"], "185")
        self.assertEqual(st["vmaxUnit"], "km/h")

    def _mk(self, wind):
        s = copy.deepcopy(self.storm)
        s["points"] = [dict(self.storm["points"][-1], wind_kt=wind)]
        return s

    def test_settings_dialog_opens_and_switches(self):
        recs = self._run([{"op": "apply", "storm": self.storm},
                          {"op": "openSettings"},
                          {"op": "clickSettingsUnit", "unit": "kmh"}])
        opened = recs[-2]["state"]
        switched = recs[-1]["state"]
        self.assertTrue(opened["settingsOpen"])
        self.assertEqual(switched["windUnits"], "kmh")
        checked = [u["unit"] for u in switched["settingsUnits"]
                   if u["checked"]]
        self.assertEqual(checked, ["kmh"])

    def test_feed_value_unchanged_display_only(self):
        # the storm dict the page holds is never mutated by a unit switch
        # - conversion is presentation. (vmax baked from the kt feed; a
        # switch only changes what's DISPLAYED.)
        recs = self._run([{"op": "apply", "storm": self.storm},
                          {"op": "setWindUnits", "unit": "mph"},
                          {"op": "setWindUnits", "unit": "kt"}])
        st = recs[-1]["state"]
        self.assertEqual(st["vmaxText"], "120")   # back to the kt value
        self.assertEqual(st["vmaxUnit"], "kt")


@unittest.skipIf(NODE is None, "node not on PATH")
class TestWindRounding(unittest.TestCase):
    """FG-R3 #4: CONVERTED winds round to the nearest 5 (NHC-style) so every
    mph / km-h display ends in 0 or 5; kt stays the raw advisory value."""

    def setUp(self):
        self.html = cyclolab_shell.render_page(load_storm(), feed_url=FEED_URL)

    def _disp_many(self, unit, kts):
        ops = [{"op": "setWindUnits", "unit": unit}]
        ops += [{"op": "callLab", "fn": "windDisp", "args": [k]} for k in kts]
        recs = run_harness(self.html, {"ops": ops})
        res = [r["result"] for r in recs if r.get("op") == "callLab"]
        return dict(zip(kts, res))

    def test_mph_table_matches_nhc_published_conversions(self):
        # NHC advisory conversions, rounded to nearest 5 mph.
        self.assertEqual(
            self._disp_many("mph", [30, 50, 64, 100, 120, 160]),
            {30: "35", 50: "60", 64: "75", 100: "115", 120: "140",
             160: "185"})

    def test_kmh_table_rounds_to_nearest_5(self):
        self.assertEqual(
            self._disp_many("kmh", [34, 64, 100]),
            {34: "65", 64: "120", 100: "185"})

    def test_kt_stays_the_raw_value(self):
        self.assertEqual(
            self._disp_many("kt", [34, 64, 97]),
            {34: "34", 64: "64", 97: "97"})

    def test_no_converted_value_lands_off_a_5_boundary(self):
        # property sweep over the whole advisory wind range, both units.
        kts = list(range(10, 200, 5))
        for unit in ("mph", "kmh"):
            got = self._disp_many(unit, kts)
            for kt, d in got.items():
                self.assertEqual(int(d) % 5, 0,
                                 "%d kt -> %s %s is not on a 5-boundary"
                                 % (kt, d, unit))


@unittest.skipIf(NODE is None, "node not on PATH")
class TestWindTierPalette(unittest.TestCase):
    """FG-R3 #1: the wind tiers carry their OWN palette - explicitly NOT the
    SSHS category tokens - with four candidate treatments resolved per
    product (ring vs swath chosen independently)."""

    REJECTED = {"34": [70, 197, 106], "50": [255, 225, 77], "64": [255, 77, 59]}

    def setUp(self):
        self.html = cyclolab_shell.render_page(load_storm(), feed_url=FEED_URL)

    def _tiers(self, key, which="Ring"):
        recs = run_harness(self.html, {"ops": [
            {"op": "callLab", "fn": "setWindPalette", "args": [key]},
            {"op": "callLab", "fn": "tierColors", "args": [which]}]})
        return recs[-1]["result"]

    def test_palette_B_equals_neon_ramp_samples(self):
        recs = run_harness(self.html, {"ops": [
            {"op": "callLab", "fn": "neonRGB", "args": [34]},
            {"op": "callLab", "fn": "neonRGB", "args": [50]},
            {"op": "callLab", "fn": "neonRGB", "args": [64]},
            {"op": "callLab", "fn": "setWindPalette", "args": ["B"]},
            {"op": "callLab", "fn": "tierColors", "args": ["Ring"]}]})
        n = [recs[i]["result"] for i in range(3)]
        tiers = recs[-1]["result"]
        self.assertEqual([tiers["34"], tiers["50"], tiers["64"]], n)

    def test_no_palette_reproduces_the_rejected_category_tokens(self):
        for key in ("A", "B", "C", "D"):
            self.assertNotEqual(self._tiers(key), self.REJECTED, key)

    def test_four_palettes_have_distinct_cores(self):
        cores = [tuple(self._tiers(k)["64"]) for k in ("A", "B", "C", "D")]
        self.assertEqual(len(set(cores)), 4, cores)

    def test_unknown_palette_falls_back_to_C(self):
        # C (neon blue->gold) is the locked canon / default (FG-R3 art-r2).
        self.assertEqual(self._tiers("Z"), self._tiers("C"))

    def test_ring_and_swath_palettes_resolve_independently(self):
        # shared knob drives both; explicit per-product knobs override.
        recs = run_harness(self.html, {"ops": [
            {"op": "callLab", "fn": "setWindPalette", "args": ["A"]},
            {"op": "callLab", "fn": "tierColors", "args": ["Ring"]},
            {"op": "callLab", "fn": "tierColors", "args": ["Swath"]}]})
        ring, swath = recs[1]["result"], recs[2]["result"]
        self.assertEqual(ring, swath)   # both fall through to the shared "A"


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
        # must not: it does ONE advisory fetch (final-gate-3 #4) with no
        # re-arm and no feed re-apply.
        recs = run_harness(html,
                           {"feed": {"storms": [self.storm]},
                            "ops": [{"op": "snapshot"}]})
        st = recs[-1]["state"]
        self.assertTrue(st["ended"])
        self.assertTrue(st["endedStripVisible"])
        self.assertNotIn(60000, st["scheduledDelays"])
        # baked snapshot still hydrated the page before freezing.
        self.assertEqual(st["odo"]["vmax"], "120")

    def test_ended_page_single_fetches_the_frozen_advisory(self):
        # final-gate-3 #4, the LATENT variant of the user's blank: ENDED
        # pages used to skip the fetch entirely, so advFull stayed null
        # FOREVER - no cone, no intensity chart, and permanently blank
        # advisory panels on every dead-storm page. Now: ONE fetch of the
        # frozen R2 advisory JSON (no re-arm), both products RENDER, and
        # the countdown stays suppressed (a dead storm has no NEXT
        # advisory even though the frozen JSON still carries the field).
        html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                          ended=True)
        adv = json.loads(
            (FIXTURE.parent / "live_adv_ep17.json").read_text())
        self.assertTrue(adv.get("next_advisory_utc"))   # field present...
        recs = run_harness(html, {
            "feed": {"storms": [self.storm]}, "adv": adv,
            "ops": [{"op": "openSec", "name": "advisories"},
                    {"op": "clickAdvTextTab", "prod": "tcd"}]})
        # The default-tab render shows TCP, the tcd click shows TCD.
        tcp_text = recs[-2]["state"]["stage3"]["adv"]["advText"]
        tcd_text = recs[-1]["state"]["stage3"]["adv"]["advText"]
        adv_fetches = [u for u in recs[-1]["state"]["stage3"]["fetched"]
                       if "/adv/" in u]
        self.assertEqual(len(adv_fetches), 1)            # exactly once
        self.assertNotIn(60000, recs[-1]["state"]["scheduledDelays"])
        self.assertIn("BULLETIN", tcp_text)             # TCP product
        self.assertIn("Discussion", tcd_text)           # TCD product
        for body in (tcp_text, tcd_text):
            self.assertNotIn("(advisory text", body)
            self.assertNotIn("(loading", body)

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

    def test_satellite_speed_presets_default_1x_and_switch(self):
        # final-gate-3 #5: four presets, default 1x; a click changes the
        # speed multiplier (which scales the playback cadence uniformly,
        # so the no-stutter contract holds at every speed).
        # the harness snapshots after EVERY op, so index off the ops.
        recs = run_harness(self.html, {
            "floaters": self._floaters_top(),
            "floater_storm": self._floater_storm(),
            "ops": [{"op": "openSec", "name": "satellite"},
                    {"op": "clickSatSpeed", "speed": 4},
                    {"op": "clickSatSpeed", "speed": 0.5}]})
        speeds = [r["state"]["stage3"]["sat"]["speed"] for r in recs]
        self.assertEqual(speeds, [1, 4, 0.5])

    def test_satellite_gif_button_and_hooks_exist(self):
        # the export button mounts and the encoder hook is exposed (the
        # encoder itself is round-trip-validated against the browser's
        # GIF decoder in validate_gif.py - jsdom has no real decoder).
        recs = run_harness(self.html, {
            "floaters": self._floaters_top(),
            "floater_storm": self._floater_storm(),
            "ops": [{"op": "openSec", "name": "satellite"},
                    {"op": "snapshot"}]})
        st = recs[-1]["state"]["stage3"]["sat"]
        self.assertFalse(st["gifBusy"])
        self.assertIn('id="sat-gif"', self.html)
        self.assertIn("function gifEncode", self.html)
        self.assertIn("function gifLzw", self.html)

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
class TestSatellitePlaybackContract(unittest.TestCase):
    """Final-gate #4: the storm-scoped viewer GUARANTEES playback
    order - strictly chronological, deduped, one band per sequence -
    even against a disordered upstream manifest."""

    def setUp(self):
        self.html = cyclolab_shell.render_page(load_storm(),
                                               feed_url=FEED_URL,
                                               loader="")

    def test_shuffled_manifest_plays_chronologically(self):
        frames = [{"t": f"2026-06-06T{h:02d}:00:00Z",
                   "key": f"floaters/ep08/ir/2026{h:02d}.png"}
                  for h in (7, 2, 9, 4, 2, 11, 4)]   # shuffled + dupes
        top = TestStage3Mounts._floaters_top()
        man = {"id": "ep082026", "slug": "ep08", "name": "SYNTH",
               "basin": "EP",
               "bands": {"ir": {"label": "Clean IR", "frames": frames}}}
        recs = run_harness(self.html, {
            "hafs_stub": True, "floaters": top, "floater_storm": man,
            "ops": [{"op": "openSec", "name": "satellite"}]})
        sat = recs[-1]["state"]["stage3"]["sat"]
        ts = sat["frameTimes"]
        self.assertEqual(ts, sorted(ts), "frames not chronological")
        self.assertEqual(len(ts), len(set(ts)), "duplicate frames kept")
        self.assertEqual(len(ts), 5)
        # registration sanity: every frame in the sequence comes from
        # the SAME band path (no zoom/band mixing in one loop)
        prefixes = {k.rsplit("/", 2)[0] + "/" + k.rsplit("/", 2)[1]
                    for k in sat["frameKeys"]}
        self.assertEqual(len(prefixes), 1, f"mixed paths: {prefixes}")


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
        # cone group's clip is removed (fully revealed).
        self.assertEqual(a["coneReveal"], "final")
        self.assertEqual(a["coneClip"], "none")
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

    def test_title_lockup_and_watermark_avoid_everything(self):
        # S4-AD2 #4/#5: the in-plot title lockup exists with the house
        # lockup text, and the watermark landed in open water - outside
        # the cone bbox and clear of every placard.
        recs = self._run([{"op": "openSec", "name": "advisories"}])
        a = recs[-1]["state"]["stage3"]["adv"]
        self.assertIsNotNone(a["coneTitle"])
        self.assertEqual(a["coneTitle"]["head"], "FORECAST CONE")
        self.assertIn("TRIPLE-A-TROPICS", a["coneTitle"]["eyebrow"])
        self.assertIn("CycloLab", a["coneTitle"]["eyebrow"])
        self.assertTrue(a["coneTitle"]["sub"].endswith("SYNTH"))
        self.assertTrue(a["coneFramed"])
        wm = a["coneWatermark"]
        self.assertIsNotNone(wm)
        self.assertEqual(wm["text"], "PACIFIC OCEAN")
        # clear of every placard rect (centre-in-rect check)
        for p in a["conePlacards"]:
            inside = (p["x"] <= wm["x"] <= p["x"] + p["w"] and
                      p["y"] <= wm["y"] <= p["y"] + p["h"])
            self.assertFalse(inside, "watermark sits on a placard")

    def test_live_payload_renders_nonempty_text_both_products(self):
        # FINAL-GATE #3: against the FROZEN LIVE payload (Amanda adv 17
        # exactly as served from R2), both advisory-text products must
        # render non-empty.
        import json as _json
        from pathlib import Path as _P
        live = _json.loads((_P(__file__).parent / "fixtures" / "cyclolab"
                            / "live_adv_ep17.json").read_text())
        recs = self._run([{"op": "openSec", "name": "advisories"},
                          {"op": "clickAdvTextTab", "prod": "tcd"}],
                         adv=live)
        mid = recs[-2]["state"]["stage3"]["adv"]["advText"]
        end = recs[-1]["state"]["stage3"]["adv"]["advText"]
        self.assertGreater(len(mid), 500, "tcp rendered empty")
        self.assertIn("BULLETIN", mid)
        self.assertGreater(len(end), 500, "tcd rendered empty")
        self.assertNotEqual(mid, end, "tcd chip did not switch products")

    def test_cone_failure_never_starves_the_text_panel(self):
        # FINAL-GATE #3 isolation: a poisoned cone ring makes
        # renderAdvCone throw - the advisory text must STILL render.
        # (Negative-controlled: pre-fix, renderAdvTab chained the three
        # renderers unguarded and this assertion fails.)
        adv = self._adv()
        adv["cone"] = adv["cone"][:3] + [None] + adv["cone"][4:]
        recs = self._run([{"op": "openSec", "name": "advisories"}], adv=adv)
        txt = recs[-1]["state"]["stage3"]["adv"]["advText"]
        self.assertGreater(len(txt), 10,
                           "text starved by the cone renderer's throw")

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
        # Honest placeholder states (final-gate-3 #4): URL present but
        # text not yet attached = the posting-lag window the poller's
        # text-heal closes - say so; no URLs at all = the agency
        # publishes no product - say THAT.
        adv = self._adv(text={"tcp_url": "u1", "tcd_url": "u2"})
        recs = self._run([{"op": "openSec", "name": "advisories"}], adv=adv)
        self.assertIn("hasn’t posted yet",
                      recs[-1]["state"]["stage3"]["adv"]["advText"])
        adv2 = self._adv(text={})
        recs2 = self._run([{"op": "openSec", "name": "advisories"}],
                          adv=adv2)
        self.assertIn("no advisory text product",
                      recs2[-1]["state"]["stage3"]["adv"]["advText"])

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
            recs[-1]["state"]["stage3"]["adv"]["coneClip"], "none")


@unittest.skipIf(NODE is None, "node not on PATH")
class TestHeroLayers(unittest.TestCase):
    """final-gate-2 #1 - the storm-centered SST hero layer picker:
    meta-driven, LAZY per selection, disclosure caption, honest
    fallback."""

    def setUp(self):
        self.storm = load_storm()
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL)

    def _run(self, ops, meta=SST_META):
        plan = {"feed": {"storms": [self.storm]}, "ops": ops}
        if meta is not None:
            plan["sst_meta"] = meta
        return run_harness(self.html, plan)

    def test_sst_base_is_baked(self):
        # the live default rides the Worker path for THIS sid
        self.assertIn('var SST_BASE = "/cyclolab/NHC_EP082026/sst"',
                      self.html)
        html2 = cyclolab_shell.render_page(
            self.storm, feed_url=FEED_URL,
            sst_base="https://cdn.example.com/shadow/cyclolab/X/sst")
        self.assertIn(
            'var SST_BASE = "https://cdn.example.com/shadow/cyclolab/X/sst"',
            html2)

    def test_picker_builds_and_defaults_to_first_layer(self):
        recs = self._run([{"op": "apply", "storm": self.storm}])
        hero = recs[-1]["state"]["hero"]
        self.assertEqual(hero["layers"], ["actual", "anomaly"])
        self.assertEqual(hero["activeLayer"], "actual")
        self.assertIn("/sst/actual.png?v=2026-06-06T12", hero["imgUrl"])
        self.assertEqual(hero["head"], "SEA SURFACE TEMPERATURE")

    def test_layer_click_is_lazy_and_discloses_baseline(self):
        recs = self._run([{"op": "apply", "storm": self.storm},
                          {"op": "clickHeroLayer", "slug": "anomaly"}])
        before = recs[0]["state"]["hero"]
        after = recs[-1]["state"]["hero"]
        # lazy: the anomaly PNG was not referenced until picked
        self.assertNotIn("anomaly.png", before["imgUrl"])
        self.assertIn("/sst/anomaly.png?v=", after["imgUrl"])
        self.assertEqual(after["activeLayer"], "anomaly")
        self.assertEqual(after["head"], "SST ANOMALY")
        # final-gate-3 #6a, ONE CANON: the caption discloses the
        # SITE-WIDE 1991-2020 baseline (no divergence note) + valid day
        self.assertIn("1991", after["caption"])
        self.assertNotIn("official CRW climatology", after["caption"])
        self.assertIn("valid 2026-06-05", after["caption"])

    def test_missing_meta_is_an_honest_fallback(self):
        recs = self._run([{"op": "apply", "storm": self.storm}],
                         meta=None)
        hero = recs[-1]["state"]["hero"]
        self.assertFalse(hero["imgShown"])
        self.assertEqual(hero["layers"], [])
        self.assertIn("No storm-centered SST render", hero["caption"])
        # the glyph still renders (centered by CSS - no client math)
        self.assertGreater(hero["glyphHtml"], 50)


def _parse_pairs(d: str, sep: str) -> list[tuple[float, float]]:
    """Parse 'x{sep}y' coordinate pairs out of an SVG path string."""
    import re
    pat = (r"([-\d.]+)\s+([-\d.]+)" if sep == " "
           else r"([-\d.]+),([-\d.]+)")
    return [(float(a), float(b)) for a, b in re.findall(pat, d)]


def _inside(poly: list[tuple[float, float]], x: float, y: float) -> bool:
    n, j, c = len(poly), len(poly) - 1, False
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            c = not c
        j = i
    return c


@unittest.skipIf(NODE is None, "node not on PATH")
class TestConeCorridorContainment(unittest.TestCase):
    """final-gate-2 #3, the geometry half (the pixel half lives in
    test_cyclolab_visual.py): the reveal corridor must CONTAIN the cone
    at full extent - rear bulge behind NOW and lateral bulges included
    - so clip removal at settle is a visual no-op and the revealed
    lateral boundary is always the cone's own casing."""

    @classmethod
    def setUpClass(cls):
        cls.storm = load_storm()
        cls.html = cyclolab_shell.render_page(cls.storm, feed_url=FEED_URL)
        cls.adv = json.loads((HERE / "fixtures" / "cyclolab"
                              / "live_adv_ep17.json").read_text())

    def _seek(self, f):
        recs = run_harness(self.html, {
            "ops": [{"op": "applyAdvisory", "adv": self.adv},
                    {"op": "openSec", "name": "advisories"},
                    {"op": "coneSeek", "f": f}]})
        a = recs[-1]["state"]["stage3"]["adv"]
        clip = _parse_pairs(a["coneRevealD"], " ")
        ring = _parse_pairs(a["coneOutlineD"], ",")
        return a, clip, ring

    def test_full_extent_corridor_contains_every_ring_vertex(self):
        a, clip, ring = self._seek(1.0)
        self.assertTrue(a["coneHooks"])
        self.assertGreater(len(clip), 40)
        self.assertGreater(len(ring), 100)
        outside = [(x, y) for x, y in ring if not _inside(clip, x, y)]
        self.assertEqual(
            len(outside), 0,
            f"{len(outside)}/{len(ring)} ring verts OUTSIDE the full "
            f"corridor (first: {outside[:3]}) - the settle frame would "
            f"pop them in")

    def test_stationary_forecast_renders_unclipped_cone(self):
        # adversarial-review find (reproduced crash): a fully-
        # stationary forecast (every point identical) collapses the
        # growth axis to Ltot=0; halfAt used to index samples[NaN] and
        # the swallowed throw left the cone clipped against an EMPTY
        # path - invisible fill+casing, null hooks. Now: no growth
        # axis means nothing to reveal - the finished cone ships
        # UNCLIPPED with index-staggered pops.
        pts = [{"tau_h": t, "lat": 15.0, "lon": -135.0,
                "intensity_kt": 30, "dev_label": "TS", "valid_utc": "x"}
               for t in (0, 12, 24, 48)]
        ring = [[-137.0, 13.0], [-133.0, 13.0], [-133.0, 17.0],
                [-137.0, 17.0], [-137.0, 13.0]]
        adv = {"sid": "NHC_EP082026", "advisory": 9,
               "issued_utc": "2026-06-06T21:00:00Z", "source": "nhc",
               "method": "official-cone", "cone": ring, "points": pts,
               "text": {"tcp": "X", "tcd": "Y"}}
        recs = run_harness(self.html, {
            "ops": [{"op": "applyAdvisory", "adv": adv},
                    {"op": "openSec", "name": "advisories"},
                    {"op": "coneSeek", "f": 0.5}]})
        a = recs[-1]["state"]["stage3"]["adv"]
        # the cone RENDERED (fill+casing present, not clipped away)
        self.assertEqual(a["coneClip"], "none")
        self.assertEqual(a["coneReveal"], "final")
        self.assertGreater(len(a["coneOutlineD"]), 40)
        self.assertEqual(a["coneIcons"], 4)
        self.assertFalse(a["coneEmptyShown"])
        # hooks exist and answer honestly (degenerate, no throw)
        self.assertTrue(a["coneHooks"])
        self.assertTrue(a["coneSeek"] and a["coneSeek"].get("degenerate"))
        # pop delays are real numbers (no NaN animation-delay)
        for d in a["coneIconsDetail"]:
            self.assertTrue(d["delay"] >= 0.3, d)

    def test_partial_seek_clips_strictly_less(self):
        # teeth: the containment test CAN fail - a part-grown corridor
        # must leave forward ring vertices outside.
        _, clip, ring = self._seek(0.35)
        outside = [p for p in ring if not _inside(clip, p[0], p[1])]
        self.assertGreater(len(outside), 10,
                           "a 35% corridor should not contain the "
                           "whole cone - the probe has no teeth")

    def test_seek_hook_reports_geometry_and_rearms_clip(self):
        a, clip, _ = self._seek(0.5)
        s = a["coneSeek"]
        self.assertIsNotNone(s)
        self.assertGreater(s["Ltot"], 100)
        self.assertAlmostEqual(s["d"], s["Ltot"] * 0.5, places=3)
        self.assertGreater(s["w"], 0)
        # seeking re-arms the clip attribute (jsdom boots at 'final')
        self.assertIn("ac-reveal-clip", a["coneClip"])
        self.assertGreater(a["coneRevealPathLen"], 100)

    def test_seek_tip_advances_along_the_track(self):
        # NOT self-referential (adversarial-review find): the tip must
        # MOVE between seeks and the clip polygon must actually grow -
        # checked against the emitted geometry, not seek's own math.
        recs = run_harness(self.html, {
            "ops": [{"op": "applyAdvisory", "adv": self.adv},
                    {"op": "openSec", "name": "advisories"},
                    {"op": "coneSeek", "f": 0.25},
                    {"op": "coneSeek", "f": 0.75}]})
        a25 = recs[-2]["state"]["stage3"]["adv"]
        a75 = recs[-1]["state"]["stage3"]["adv"]
        s25, s75 = a25["coneSeek"], a75["coneSeek"]
        moved = ((s75["tipX"] - s25["tipX"]) ** 2 +
                 (s75["tipY"] - s25["tipY"]) ** 2) ** 0.5
        self.assertGreater(moved, 50,
                           "the front tip barely moved between f=0.25 "
                           "and f=0.75")
        # the revealed polygon AREA grows with f (shoelace on the
        # emitted clip path - independent of the hook's return values)
        def area(d):
            pts = _parse_pairs(d, " ")
            s = 0.0
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                s += x1 * y2 - x2 * y1
            return abs(s) / 2
        self.assertGreater(area(a75["coneRevealD"]),
                           area(a25["coneRevealD"]) * 1.5)


@unittest.skipIf(NODE is None, "node not on PATH")
class TestNoLeaderLines(unittest.TestCase):
    """final-gate-3 #1: leader lines are GONE. No placard ever draws a
    connector. Pairing is positional - placards stay collision-free AND
    hug their glyph tightly enough to read without a line."""

    @classmethod
    def setUpClass(cls):
        cls.storm = load_storm()
        cls.html = cyclolab_shell.render_page(cls.storm, feed_url=FEED_URL)
        cls.live_adv = json.loads((HERE / "fixtures" / "cyclolab"
                                   / "live_adv_ep17.json").read_text())

    def _state(self, adv):
        recs = run_harness(self.html, {
            "ops": [{"op": "applyAdvisory", "adv": adv},
                    {"op": "openSec", "name": "advisories"}]})
        return recs[-1]["state"]["stage3"]["adv"]

    def _no_overlaps(self, pls, label):
        for i in range(len(pls)):
            for j in range(i + 1, len(pls)):
                a, b = pls[i], pls[j]
                ox = min(a["x"] + a["w"], b["x"] + b["w"]) - \
                    max(a["x"], b["x"])
                oy = min(a["y"] + a["h"], b["y"] + b["h"]) - \
                    max(a["y"], b["y"])
                self.assertFalse(
                    ox > 0 and oy > 0,
                    f"{label}: placards {a['i']} and {b['i']} overlap")

    def test_no_leader_lines_on_the_live_official_cone(self):
        a = self._state(self.live_adv)
        self.assertEqual(a["coneLeaders"], [],
                         "a leader line was rendered - they must be gone")
        self.assertTrue(a["conePlacards"])
        self._no_overlaps(a["conePlacards"], "live-official")
        # pairing without lines: every placard's anchor sits within a
        # tight band of its glyph edge so the eye pairs them by position.
        for p in a["conePlacards"]:
            self.assertLessEqual(
                p["gap"], 120.0,
                f"placard {p['i']} gap {p['gap']:.1f} too far to pair "
                "without a leader line")

    def test_bunched_taus_no_lines_no_overlaps_still_pair(self):
        # The pathological cluster the leader lines used to rescue: six
        # taus packed tight. Must place collision-free, draw ZERO lines,
        # and keep every placard close enough to pair by position.
        pts = [
            {"tau_h": t, "lat": 15.0 + 0.05 * i, "lon": -135.0 - 0.07 * i,
             "intensity_kt": 30 + i, "dev_label": "TS", "valid_utc": "x"}
            for i, t in enumerate((0, 12, 24, 36, 48, 60))]
        ring = [[-140.0, 13.0], [-134.0, 13.0], [-133.5, 18.0],
                [-140.5, 18.5], [-140.0, 13.0]]
        adv = {"sid": "NHC_EP082026", "advisory": 21,
               "issued_utc": "2026-06-06T21:00:00Z", "source": "nhc",
               "method": "official-cone", "cone": ring, "points": pts,
               "text": {"tcp": "X", "tcd": "Y"}}
        a = self._state(adv)
        self.assertEqual(a["coneLeaders"], [], "bunched case drew a line")
        self._no_overlaps(a["conePlacards"], "bunched")
        for p in a["conePlacards"]:
            self.assertLessEqual(
                p["gap"], 150.0,
                f"bunched placard {p['i']} gap {p['gap']:.1f} unpairable")


@unittest.skipIf(NODE is None, "node not on PATH")
class TestMesoSectorSeam(unittest.TestCase):
    """final-gate-2 #6: the Satellite tab reads sectors through a
    SOURCE REGISTRY so mesoscale sectors are an entry, not a rewrite.
    Seam only - one source today."""

    def test_registry_exists_with_the_floater_entry(self):
        # STRUCTURAL markers only (a comment reword must not break the
        # suite): the registry literal, the floater entry, its resolver
        # hook, and the viewer iterating the registry.
        html = cyclolab_shell.render_page(load_storm(), feed_url=FEED_URL)
        self.assertIn("var SAT_SOURCES = [{", html)
        self.assertIn('id: "floater"', html)
        self.assertIn("resolve: function (top)", html)
        self.assertIn("SAT_SOURCES.map(", html)


@unittest.skipIf(NODE is None, "node not on PATH")
class TestOverviewPlots(unittest.TestCase):
    """FG-R3 #7/#8/#11: the two art-directed Overview plots (track-history
    + wind-history swath) and the two-column Overview layout. The plots
    are CLIENT-SIDE renderers driven from the existing per-fix feed (no
    new fetch) and wired into apply()'s new-fix block + rerenderUnits()."""

    def setUp(self):
        self.storm = load_radii_storm()          # 14 fixes, radii present
        self.bare = load_storm()                  # synth_storm.json, NO radii
        self.html = cyclolab_shell.render_page(self.storm, feed_url=FEED_URL,
                                               loader="")

    def _ov(self, ops, storm=None):
        plan = {"feed": {"storms": [storm or self.storm]}, "ops": ops}
        recs = run_harness(self.html, plan)
        return recs

    # ---- #7 track-history plot ------------------------------------------
    def test_trackplot_renders_dots_three_shapes_and_colorbar(self):
        st = self._ov([{"op": "apply", "storm": self.storm}])[-1]["state"]
        ov = st["overview"]
        self.assertTrue(ov["trackRendered"])
        # one dot per fix (14); legend key shapes use a separate class.
        self.assertEqual(ov["trackDotCount"], len(self.storm["points"]))
        sh = ov["trackShapes"]
        # the fixture mixes SS (square) at the start, TS (circle), ET
        # (triangle) at the end -> all three marker shapes present.
        self.assertGreater(sh["circle"], 0)
        self.assertGreater(sh["square"], 0)
        self.assertGreater(sh["triangle"], 0)
        # the LABELED colorbar (gradient + ticks).
        self.assertTrue(ov["trackColorbar"])
        self.assertGreaterEqual(ov["trackColorbarTicks"], 6)
        # title lockup + map furniture + legend.
        self.assertEqual(ov["trackTitle"], "TRACK HISTORY")
        self.assertTrue(ov["trackLegend"])
        self.assertGreaterEqual(ov["trackGraticule"], 2)

    def test_trackplot_windfield_present_with_radii(self):
        # the current fix carries radii -> the four-quadrant arcs render
        # (one path per non-empty threshold) and the caption cites them.
        st = self._ov([{"op": "apply", "storm": self.storm}])[-1]["state"]
        ov = st["overview"]
        self.assertGreater(ov["trackWindField"], 0)
        self.assertIn("wind radii", ov["trackNote"])
        self.assertIn("best-track deck", ov["trackNote"])

    def test_trackplot_windfield_absent_is_graceful(self):
        # a feed with NO radii anywhere: the plot still renders fully from
        # wind + nature; the wind-field is skipped and the caption says so.
        st = self._ov([{"op": "apply", "storm": self.bare}],
                      storm=self.bare)[-1]["state"]
        ov = st["overview"]
        self.assertTrue(ov["trackRendered"])
        self.assertEqual(ov["trackWindField"], 0)
        self.assertEqual(ov["trackDotCount"], len(self.bare["points"]))
        self.assertIn("wind radii unavailable", ov["trackNote"])

    def test_trackplot_colorbar_converts_with_units(self):
        # units-aware colorbar: the underlying ramp GEOMETRY stays in kt,
        # only the tick labels convert. 185 kt -> 213 mph appears as a tick.
        recs = self._ov([{"op": "apply", "storm": self.storm},
                         {"op": "setWindUnits", "unit": "mph"}])
        # the plot was re-rendered by rerenderUnits with mph labels.
        ov = recs[-1]["state"]["overview"]
        self.assertTrue(ov["trackRendered"])
        # the caption's threshold note now reads "mph", never a bare "kt".
        self.assertIn("mph", ov["trackNote"])

    # ---- #8 wind-history swath plot -------------------------------------
    def test_swath_renders_filled(self):
        # FG-R3 verdict 2: the swath ALWAYS renders filled - the outlined /
        # hatched variant and its toggle were dropped entirely. Both the
        # 34-kt field and the 64-kt core render as filled bands.
        recs = self._ov([{"op": "apply", "storm": self.storm}])
        applied = recs[-1]["state"]["overview"]
        self.assertTrue(applied["swathRendered"])
        self.assertGreater(applied["swath34"], 0)
        self.assertGreater(applied["swath64"], 0)

    def test_swath_derived_caption_and_method_panel(self):
        st = self._ov([{"op": "apply", "storm": self.storm}])[-1]["state"]
        ov = st["overview"]
        self.assertEqual(ov["swathTitle"], "WIND HISTORY")
        self.assertTrue(ov["swathDerivedShown"], "derived caption hidden")
        self.assertTrue(ov["swathMethodShown"], "method <details> hidden")
        body = ov["swathMethodBody"] or ""
        self.assertIn("interpolat", body)          # interpolation disclosed
        self.assertIn("best-track", body)          # source cited
        self.assertIn("not an official", body)     # honesty
        self.assertIn("Derived", ov["swathNote"])

    def test_swath_honest_empty_state_without_radii(self):
        # a storm with no analyzed radii: never a blank/broken panel -
        # the SVG hides and the honest empty state shows instead.
        recs = run_harness(
            self.html,
            {"feed": {"storms": [self.bare]},
             "ops": [{"op": "apply", "storm": self.bare}]})
        ov = recs[-1]["state"]["overview"]
        self.assertFalse(ov["swathRendered"])
        self.assertEqual(ov["swathDisplay"], "none")
        self.assertTrue(ov["swathEmptyShown"])
        self.assertIn("not yet available", ov["swathEmptyText"])
        # the derived caption + method panel stay hidden in the empty state.
        self.assertFalse(ov["swathDerivedShown"])
        self.assertFalse(ov["swathMethodShown"])

    # ---- #11 two-column Overview layout ---------------------------------
    def test_overview_two_column_grid(self):
        st = self._ov([{"op": "apply", "storm": self.storm}])[-1]["state"]
        ov = st["overview"]
        # desktop = a real two-column CSS grid on the .wipe, two ov-col
        # wrappers (left = hero+chart, right = track+swath).
        self.assertIn("1fr 1fr", ov["ovGridCols"])
        self.assertEqual(ov["ovCols"], 2)
        # all four overview cards are present.
        for c in ov["ovCardOrder"]:
            self.assertTrue(c["present"], f"missing {c['id']}")

    def test_overview_mobile_stack_order_in_css(self):
        # the mobile media query defines the explicit stack order
        # hero -> track -> W&P -> swath via CSS `order` (jsdom does not
        # apply media queries, so assert the served rule text).
        h = self.html
        self.assertIn("grid-template-columns: 1fr;", h)   # single column
        self.assertIn("#sec-overview #card-hero  { order: 1; }", h)
        self.assertIn("#sec-overview #card-track { order: 2; }", h)
        self.assertIn("#sec-overview #card-wp    { order: 3; }", h)
        self.assertIn("#sec-overview #card-swath { order: 4; }", h)

    def test_plots_render_on_new_fix_via_apply(self):
        # wiring parity with renderChart: a fresh fix triggers both plots
        # (and the chart) inside apply()'s new-fix block.
        st = self._ov([{"op": "apply", "storm": self.storm}])[-1]["state"]
        self.assertGreater(st["chartChildCount"], 0)
        self.assertTrue(st["overview"]["trackRendered"])
        self.assertTrue(st["overview"]["swathRendered"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
