"""VISUAL regression tests for the CycloLab shell - these render the
real page in a real engine and measure INK PIXELS, not DOM attributes.

Born in AD Round 3 (the R2 suite was green while the rendered card had
flat-sheared digit bottoms and superscript-looking units) and rebuilt
in Round 3b, when pixel measurement showed the R3 odometer STILL broke
the vitals text two ways the suite could not see:
  (1) digits sank and wobbled BY VALUE - translateY in fractional ems
      rounded differently per digit (the "9" sat 2px low);
  (2) the rest-state clip sheared the round digits' baseline overshoot
      flat. Round glyphs (0,3,5,6,8,9) are SUPPOSED to dip 1-2px below
      the baseline - that overshoot is correct typography. The old
      tests asserted UNIFORM ink bottoms, so they actively REWARDED
      the clipping; TestVitalsTypography replaces that expectation.

Calibration facts (measured on Metropolis-800 at 200px, threshold 90):
  * round-digit overshoot is 1% of em (sub-device-pixel at vitals
    sizes on DPR 1 - so the ground truth for "unclipped" is pixel
    EQUALITY with a static twin string, not a fixed offset);
  * W's pointed vertex dips ~1.5% em below baseline - NOT a flat
    reference glyph;
  * "1"'s narrow stem AA flickers +-1-2px around the threshold, so it
    is excluded from the modal baseline and given a looser band.

Skips cleanly when playwright/browsers/Pillow are unavailable, and
when the self-hosted Metropolis webfont cannot load (offline CI) -
fallback-font metrics would make every number meaningless. Run locally
before any visual sign-off.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import cyclolab_shell  # noqa: E402
from ace_core import SSHS_COLORS  # noqa: E402

FIXTURE = HERE / "fixtures" / "cyclolab" / "synth_storm.json"
FEED_URL = "https://cdn.triple-a-tropics.com/feeds/ep_tracks_data.json"
DPR = 3
THRESH = 90                       # absolute luminance, per the R3b scan
FLAT_DIGITS = set("1247")
ROUND_DIGITS = set("035689")
FLAT_LETTERS = set("NUTHKMILEFknmhitlu")      # no W (vertex dips)
SKIP_CHARS = set(" .-:·,W")
FAST_MOTION = ":root{--motion-slow:0.05s!important;}"

try:
    from PIL import Image
    from playwright.sync_api import sync_playwright
    with sync_playwright() as _p:
        _b = _p.chromium.launch()
        _b.close()
    HAVE_RENDERER = True
except Exception:  # noqa: BLE001 - any miss (import, browser) -> skip
    HAVE_RENDERER = False


def ink_rows(im, box, dpr=DPR):
    """(top, bottom) device-pixel rows containing ink (pixels clearly
    brighter than the panel background) within a CSS-px box."""
    l, t, r, b = (int(v * dpr) for v in box)
    px = im.crop((l, t, r, b)).convert("L")
    w, h = px.size
    rows = [max(px.getpixel((x, y)) for x in range(w)) for y in range(h)]
    bg = sorted(rows)[max(0, len(rows) // 10)]
    ink = [y for y, v in enumerate(rows) if v > bg + 60]
    return (t + ink[0], t + ink[-1]) if ink else None


# ---------------------------------------------------------------------------
# R3b per-glyph scan core (mirrors the forensic scanner methodology)
# ---------------------------------------------------------------------------

def classify(ch):
    if ch in FLAT_DIGITS or ch in FLAT_LETTERS:
        return "flat"
    if ch in ROUND_DIGITS:
        return "round"
    return "other"


def cell_bottoms(im, cells, dpr):
    """cells = [{char, box, em}] css px -> [{char, kind, bottom,
    bottom_sub}]: ``bottom`` is the last row with any pixel above the
    binary threshold (the R3b spec scan, device px); ``bottom_sub`` is
    a SUB-PIXEL bottom-edge estimate (last solid row + fractional AA
    coverage of the rows below it, from per-row peak luminance). The
    round digits' baseline overshoot is ~1% of em - often under one
    device pixel - so cross-context comparisons must be sub-pixel or
    they flicker with whatever pixel-grid fraction each context lands
    on. Band: 0.30em above the box to 0.28em below (contains all legit
    overshoot without reaching the next text row)."""
    out = []
    W, H = im.size
    BG = 16.0
    for c in cells:
        l, t, r, b = c["box"]
        em = c["em"]
        x0, x1 = int(l * dpr) + 1, int(r * dpr) - 1
        y0 = max(0, int((t - 0.30 * em) * dpr))
        y1 = min(H, int((b + 0.28 * em) * dpr))
        if x1 <= x0 or y1 <= y0:
            out.append(None)
            continue
        px = im.crop((x0, y0, x1, y1)).convert("L")
        w, h = px.size
        peaks = [max(px.getpixel((x, y)) for x in range(w))
                 for y in range(h)]
        ink = [y for y, v in enumerate(peaks) if v > THRESH]
        solid = [y for y, v in enumerate(peaks) if v >= 245]
        sub = None
        if solid:
            sub = float(solid[-1])
            for y in range(solid[-1] + 1, h):
                fr = (peaks[y] - BG) / (255.0 - BG)
                if fr <= 0.04:
                    break
                sub += min(1.0, max(0.0, fr))
        out.append({"char": c["char"], "kind": classify(c["char"]),
                    "bottom": y0 + ink[-1] if ink else None,
                    "bottom_sub": y0 + sub if sub is not None else None})
    return out


def baseline_verdict(rows):
    """Modal flat baseline (excluding the AA-flickery '1'; ties break
    low on screen) on the binary-threshold bottoms, plus a SUB-PIXEL
    modal (median of the non-'1' flats' bottom_sub) for cross-context
    comparisons. flat_dev/one_dev are threshold-based (the R3b wobble
    spec); rounds/rounds_sub are offsets below the respective modal."""
    flats = [(r["char"], r["bottom"]) for r in rows
             if r and r["kind"] == "flat" and r["bottom"] is not None]
    core = [b for c, b in flats if c != "1"] or [b for _, b in flats]
    if not core:
        return None
    modal = max(sorted(set(core)), key=core.count)
    dev = max((abs(b - modal) for c, b in flats if c != "1"), default=0)
    dev1 = max((abs(b - modal) for c, b in flats if c == "1"), default=0)
    rounds = [(r["char"], r["bottom"] - modal) for r in rows
              if r and r["kind"] == "round" and r["bottom"] is not None]
    subs = sorted(r["bottom_sub"] for r in rows
                  if r and r["kind"] == "flat" and r["char"] != "1"
                  and r.get("bottom_sub") is not None)
    modal_sub = subs[len(subs) // 2] if subs else None
    rounds_sub = [(r["char"], r["bottom_sub"] - modal_sub) for r in rows
                  if r and r["kind"] == "round"
                  and r.get("bottom_sub") is not None] \
        if modal_sub is not None else []
    return {"modal": modal, "flat_dev": dev, "one_dev": dev1,
            "rounds": rounds, "modal_sub": modal_sub,
            "rounds_sub": rounds_sub}


JS_CELLS = """(id) => {
  // final-gate-3 #2: the value is now PLAIN TEXT (no .digit cells). To
  // measure per-glyph boxes we wrap each character in a BARE inline
  // span - tnum digits don't kern, so bare spans render pixel-identical
  // to the plain text already in the screenshot; the boxes line up.
  const el = document.getElementById(id);
  const em = parseFloat(getComputedStyle(el).fontSize);
  const want = el.getAttribute('data-odo') || el.textContent;
  el.textContent = '';
  for (let i = 0; i < want.length; i++) {
    const s = document.createElement('span');
    s.textContent = want[i];
    el.appendChild(s);
  }
  return [...el.children].map((c, i) => {
    const r = c.getBoundingClientRect();
    return {char: want[i], box: [r.left, r.top, r.right, r.bottom], em};
  });
}"""

JS_TWIN = """(id) => {
  // GROUND-TRUTH TWIN, isolated: the value string re-rendered as plain
  // text in a fixed overlay with the odometer's exact font context and
  // an opaque panel background. In-flow twins collide with real
  // content in tight layouts (the hero twin overlapped the Category
  // hero) and inherit alien line boxes in flex rows - so the twin is
  // measured in isolation and compared RELATIVELY (per-glyph offset
  // from each side's own modal flat baseline).
  const el = document.getElementById(id);
  const cs = getComputedStyle(el);          // .odo is plain text now
  let ov = document.getElementById('twin-overlay');
  if (!ov) {
    ov = document.createElement('div');
    ov.id = 'twin-overlay';
    ov.style.cssText = 'position:fixed;top:24px;left:740px;z-index:9999;' +
      'background:#0b0e13;padding:18px 24px;text-align:left;';
    document.body.appendChild(ov);
  }
  let twin = document.getElementById(id + '-twin');
  if (!twin) {
    twin = document.createElement('div');
    twin.id = id + '-twin';
    twin.style.marginBottom = '36px';
    ov.appendChild(twin);
  }
  twin.style.cssText = 'color:#fff;white-space:pre;margin-bottom:36px;' +
    `font-family:${cs.fontFamily};font-size:${cs.fontSize};` +
    `font-weight:${cs.fontWeight};line-height:${cs.lineHeight};` +
    'font-feature-settings:"tnum";font-variant-numeric:tabular-nums;';
  const want = el.getAttribute('data-odo') || el.textContent;
  // per-char SPANS, not Range rects: the Range box of a line-final
  // character collapses to a sliver in both engines. Digits do not
  // kern and tnum keeps tabular advances, so spans are lossless here.
  twin.textContent = '';
  const em = parseFloat(getComputedStyle(twin).fontSize);
  const out = [];
  for (let i = 0; i < want.length; i++) {
    const s = document.createElement('span');
    s.textContent = want[i];
    twin.appendChild(s);
  }
  for (let i = 0; i < want.length; i++) {
    const r = twin.children[i].getBoundingClientRect();
    out.push({char: want[i], box: [r.left, r.top, r.right, r.bottom], em});
  }
  return out;
}"""


def usable(cells):
    return [c for c in cells
            if c["char"] not in SKIP_CHARS and not c["char"].isspace()]


@unittest.skipUnless(HAVE_RENDERER, "playwright chromium + Pillow required")
class TestRenderedCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        storm = json.loads(FIXTURE.read_text(encoding="utf-8"))
        html = cyclolab_shell.render_page(storm, feed_url=FEED_URL,
                                          loader="")  # no loader overlay
        cls._td = tempfile.TemporaryDirectory()
        page_file = Path(cls._td.name) / "page.html"
        page_file.write_text(html, encoding="utf-8")
        shot = Path(cls._td.name) / "shot.png"
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1380, "height": 860},
                            device_scale_factor=DPR)
            pg.goto(f"file://{page_file}")
            pg.wait_for_timeout(3000)   # boot apply + font load + settle
            # unclipped same-font reference next to the hero (measures
            # what an UNCLIPPED "120" looks like in the same engine).
            pg.evaluate("""() => {
              const src = document.getElementById('odo-vmax');
              const cs = getComputedStyle(src);
              const ref = document.createElement('span');
              ref.id = 'ref-digits'; ref.textContent = '120';
              ref.style.cssText =
                'position:absolute; left:420px; top:150px; ' +
                'background:#0b0e13; color:#ffffff; ' +
                `font-family:${cs.fontFamily}; font-size:${cs.fontSize}; ` +
                `font-weight:${cs.fontWeight}; ` +
                'font-feature-settings:"tnum"; line-height:1.15em;';
              document.body.appendChild(ref);
            }""")
            pg.wait_for_timeout(150)
            pg.screenshot(path=str(shot))
            cls.boxes = pg.evaluate("""() => {
              const g = s => { const r = document.querySelector(s)
                .getBoundingClientRect();
                return [r.left, r.top, r.right, r.bottom]; };
              return { heroOdo: g('#odo-vmax'),
                       heroUnit: g('#odo-vmax + .unit'),
                       ref: g('#ref-digits'),
                       glyph: g('.banner .glyph') };
            }""")
            # final-gate-3 #2: the value is plain text - measure each
            # .odo element's own box (its rendered value) and the band
            # just below it.
            cls.rest_cells = pg.evaluate("""() =>
              [...document.querySelectorAll('.odo')]
                .filter(c => (c.textContent || '').trim().length)
                .map(c => {
                  const r = c.getBoundingClientRect();
                  return {box: [r.left, r.top, r.right, r.bottom],
                          em: parseFloat(getComputedStyle(c).fontSize),
                          odo: c.id || '?'};
                })""")
            # the structural guard the odometer is gone: zero cell/strip
            # machinery anywhere, ever.
            cls.col_count = pg.evaluate(
                "document.querySelectorAll('.odo .col, .odo .strip')"
                ".length")
            b.close()
        cls.im = Image.open(shot)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _ink(self, name):
        r = ink_rows(self.im, self.boxes[name])
        self.assertIsNotNone(r, f"no ink found in {name}")
        return r

    def test_hero_digits_are_not_clipped(self):
        # ink HEIGHT of the settled odometer "120" == unclipped
        # reference "120" (same engine, same font). A box-edge clip
        # shears the bottom overshoot and fails this by device pixels.
        odo_t, odo_b = self._ink("heroOdo")
        ref_t, ref_b = self._ink("ref")
        self.assertLessEqual(abs((odo_b - odo_t) - (ref_b - ref_t)), 2,
                             "hero digit ink height differs from the "
                             "unclipped reference - glyphs are clipped")

    def test_hero_unit_shares_the_digit_baseline(self):
        # digits and the uppercase KT unit both bottom out at the
        # baseline; their ink bottoms must agree within 2 device px
        # (round-digit overshoot can add 1 below the flat baseline).
        _, digit_bottom = self._ink("heroOdo")
        _, unit_bottom = self._ink("heroUnit")
        self.assertLessEqual(abs(digit_bottom - unit_bottom), 2,
                             "KT does not sit on the hero digits' baseline")

    def test_glyph_label_painted_in_category_color(self):
        # the synth fixture is C4: the canon "4" on the glyph must be
        # painted in the canonical C4 accent (not white).
        accent = SSHS_COLORS["C4"].lstrip("#")
        ar, ag, ab = (int(accent[i:i + 2], 16) for i in (0, 2, 4))
        l, t, r, b = (int(v * DPR) for v in self.boxes["glyph"])
        px = self.im.crop((l, t, r, b)).convert("RGB")
        w, h = px.size
        hits = sum(
            1 for x in range(0, w, 2) for y in range(0, h, 2)
            if abs(px.getpixel((x, y))[0] - ar) < 40
            and abs(px.getpixel((x, y))[1] - ag) < 40
            and abs(px.getpixel((x, y))[2] - ab) < 40)
        self.assertGreater(hits, 5,
                           "no category-accent ink found on the glyph "
                           "label - is it still white?")

    def test_rest_state_is_plain_text_with_no_ghosts(self):
        # R3b contract: settled odometers are PLAIN TEXT - zero rolling
        # .col elements anywhere - and the band just below each digit
        # cell's box is pure background (overshoot lives INSIDE the box;
        # neighbor-digit ghosts can no longer exist at rest because
        # there is no strip at rest).
        self.assertEqual(self.col_count, 0,
                         "rolling .col elements present at rest - the "
                         "settle-to-plain-text contract is broken")
        vitals = [c for c in self.rest_cells if c["em"] <= 20]
        self.assertGreater(len(vitals), 0, "no vitals-sized cells found")
        for col in vitals:
            l, t, r, b = col["box"]
            band = (int(l * DPR), int(b * DPR) + 2,
                    int(r * DPR), int((b + 0.20 * col["em"]) * DPR))
            px = self.im.crop(band).convert("L")
            w, h = px.size
            bright = sum(1 for x in range(w) for y in range(h)
                         if px.getpixel((x, y)) > 100)
            self.assertEqual(
                bright, 0,
                f"ink below a settled cell of #{col['odo']} "
                f"(band {band}): ghost or stray clip artifact")


@unittest.skipUnless(HAVE_RENDERER, "playwright browsers + Pillow required")
class TestVitalsTypography(unittest.TestCase):
    """R3b: per-glyph baseline discipline on rendered pixels, across
    engines and DPRs. See the module docstring for the calibration.

      (a) flat glyphs share the row's modal baseline within 1 device
          px ('1' gets 2);
      (b) settled odometer text is pixel-equal per glyph to a STATIC
          TWIN of the same string (catches BOTH clipping and sinking),
          and where the twin's round-digit overshoot is measurable
          (>=1 dev px below modal) it survives in the odometer;
      (c) value cycle 925/999/111/140/000 on the MSLP odometer: the
          same digit lands on the same bottom every time;
      (d) zero .col elements at rest, every cycle step.
    """
    ENGINES = ("chromium", "webkit")
    DPRS = (1, 2, 3)
    ODOS = ("odo-vmax", "odo-mslp", "odo-pos", "odo-move", "odo-fix")
    CYCLE = ("925", "999", "111", "140", "000")

    @classmethod
    def setUpClass(cls):
        storm = json.loads(FIXTURE.read_text(encoding="utf-8"))
        html = cyclolab_shell.render_page(storm, feed_url=FEED_URL,
                                          loader="")
        cls._td = tempfile.TemporaryDirectory()
        cls.page_file = Path(cls._td.name) / "page.html"
        cls.page_file.write_text(html, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _scan(self, pg, im, oid, dpr, twin=False):
        cells = usable(pg.evaluate(JS_TWIN if twin else JS_CELLS, oid))
        return cell_bottoms(im, cells, dpr)

    def test_baselines_overshoot_and_value_stability(self):
        with sync_playwright() as p:
            for engine in self.ENGINES:
                browser = getattr(p, engine).launch()
                for dpr in self.DPRS:
                    with self.subTest(engine=engine, dpr=dpr):
                        self._run_one(browser, engine, dpr)
                browser.close()

    def _run_one(self, browser, engine, dpr):
        td = Path(self._td.name)
        pg = browser.new_page(viewport={"width": 1380, "height": 900},
                              device_scale_factor=dpr)
        pg.goto(f"file://{self.page_file}")
        pg.add_style_tag(content=FAST_MOTION)
        pg.wait_for_timeout(3000)
        if not pg.evaluate(
                "document.fonts.check('800 15px Metropolis')"):
            pg.close()
            self.skipTest("Metropolis webfont unavailable (offline?)")
        shot = td / f"s_{engine}_{dpr}.png"
        pg.screenshot(path=str(shot))
        im = Image.open(shot)

        odo_rows = {oid: self._scan(pg, im, oid, dpr) for oid in self.ODOS}
        # twins go in AFTER the main shot so they don't disturb it
        twin_cells = {oid: usable(pg.evaluate(JS_TWIN, oid))
                      for oid in self.ODOS}
        pg.wait_for_timeout(120)
        shot2 = td / f"s_{engine}_{dpr}_twin.png"
        pg.screenshot(path=str(shot2))
        im2 = Image.open(shot2)
        twin_rows = {oid: cell_bottoms(im2, twin_cells[oid], dpr)
                     for oid in self.ODOS}

        for oid in self.ODOS:
            v = baseline_verdict(odo_rows[oid])
            if v:                                     # (a)
                self.assertLessEqual(
                    v["flat_dev"], 1,
                    f"{engine}/dpr{dpr} #{oid}: flat glyphs wobble "
                    f"{v['flat_dev']}px off the modal baseline")
                self.assertLessEqual(
                    v["one_dev"], 2,
                    f"{engine}/dpr{dpr} #{oid}: '1' {v['one_dev']}px "
                    f"off the modal baseline")
            tv = baseline_verdict(twin_rows[oid])
            if v and tv and v["modal_sub"] is not None \
                    and tv["modal_sub"] is not None:
                # (b) RELATIVE per-glyph equality, SUB-PIXEL: each
                # glyph's bottom-edge offset from its own side's modal
                # flat baseline must match the isolated plain-text
                # twin's within 0.75 device px ('1' gets 1.5 - narrow-
                # stem AA). Catches clipping (odo round pinned at the
                # baseline vs twin dipping) AND per-value sink (odo
                # round 2px deep vs twin 1) in either direction,
                # without flickering on pixel-grid fractions the way a
                # binary threshold does (overshoot is ~1% of em).
                for o, t in zip(odo_rows[oid], twin_rows[oid]):
                    if not o or not t or o.get("bottom_sub") is None \
                            or t.get("bottom_sub") is None:
                        continue
                    o_rel = o["bottom_sub"] - v["modal_sub"]
                    t_rel = t["bottom_sub"] - tv["modal_sub"]
                    tol = 1.5 if o["char"] == "1" else 0.75
                    self.assertLessEqual(
                        abs(o_rel - t_rel), tol,
                        f"{engine}/dpr{dpr} #{oid} '{o['char']}': "
                        f"sub-px baseline offset {o_rel:+.2f} != "
                        f"plain-text twin {t_rel:+.2f} - settled cell "
                        f"is not rendering as plain text")
                # (b) overshoot SURVIVES: a round digit whose twin
                # dips measurably below the flat baseline must dip in
                # the odometer too (one half-pixel of quantization
                # allowed). Pinned at the baseline while the twin
                # overshoots = clipped.
                t_over = dict(tv["rounds_sub"])
                for ch, over in v["rounds_sub"]:
                    if t_over.get(ch, 0) >= 0.5:
                        self.assertGreaterEqual(
                            over, t_over[ch] - 0.5,
                            f"{engine}/dpr{dpr} #{oid} '{ch}': twin "
                            f"overshoots {t_over[ch]:.2f}px below the "
                            f"baseline but the odometer render shows "
                            f"{over:.2f} - overshoot clipped")

        # (c)+(d): value cycle on MSLP
        seen = {}
        for val in self.CYCLE:
            pg.evaluate("""(v) => window.__lab.odoSet(
                document.getElementById('odo-mslp'), v)""", val)
            pg.wait_for_timeout(600)
            self.assertEqual(
                pg.evaluate("document.querySelectorAll("
                            "'.odo .col, .odo .strip').length"),
                0, f"{engine}/dpr{dpr}: cell/strip machinery present "
                   f"(value {val}) - the odometer must be gone")
            sv = td / f"c_{engine}_{dpr}_{val}.png"
            pg.screenshot(path=str(sv))
            rows = self._scan(pg, Image.open(sv), "odo-mslp", dpr)
            v = baseline_verdict(rows)
            if v:
                self.assertLessEqual(
                    v["flat_dev"], 1,
                    f"{engine}/dpr{dpr} value {val}: flat wobble "
                    f"{v['flat_dev']}px")
            for r in rows:
                if r and r["bottom"] is not None:
                    seen.setdefault(r["char"], set()).add(r["bottom"])
        for ch, bots in seen.items():                 # (c)
            self.assertLessEqual(
                max(bots) - min(bots), 1,
                f"{engine}/dpr{dpr}: digit '{ch}' landed on different "
                f"bottoms across values: {sorted(bots)} - per-value "
                f"fractional offsets are back")
        pg.close()


# ---------------------------------------------------------------------------
# FINAL-GATE R2 #3 - per-frame casing coverage (the PIXEL half; the
# geometry half is TestConeCorridorContainment in test_cyclolab_shell).
#
# THE POSITIVE CONTRACT: frame-extract the reveal at >=10 arc fractions
# (through the deterministic __lab.cone().seek hook - no wall-clock
# racing). In EVERY frame, walk the revealed region's pixel boundary:
# every sample outside the advancing-front cap zone must have white
# casing ink within N px - the revealed region looks FINISHED at every
# frame, fill + its boundary together. The settle frame is checked with
# NO exclusion zone: one complete closed loop.
#
# Negative controls: (a) in-suite PROBE-TEETH - hiding the boundary
# stroke must collapse the measured coverage (the probe can fail);
# (b) dev-time - the probe run against the pre-fix build (linear
# corridor narrower than the cone) fails mid-frames + settle, evidence
# JSON in the review packet.
# ---------------------------------------------------------------------------

ADV_FIXTURE = HERE / "fixtures" / "cyclolab" / "live_adv_ep17.json"
OCEAN_RGB = (0x10, 0x1a, 0x2c)

HIDE_FURNITURE_JS = """() => {
  // display:none, NOT opacity - the ac-pop/ac-spin CSS animations
  // (fill: both) override inline opacity, but nothing overrides
  // display.
  const svg = document.getElementById('advcone');
  for (const sel of ['.ac-icon', '.ac-title', '.ac-ocean',
                     '.ac-graticule', '.ac-land', '.ac-coast', '.ac-border',
                     '.ac-frame', 'line[data-role="leader"]']) {
    svg.querySelectorAll(sel).forEach(el => {
      el.style.display = 'none';
    });
  }
  // R3 #2: the title lockup is now an HTML overlay OUTSIDE the SVG, so hide
  // it here too (its white text would otherwise count as a revealed boundary
  // with no casing, off in the corner away from the cone).
  const lk = document.getElementById('advcone-lockup');
  if (lk) lk.style.display = 'none';
}"""


def casing_coverage(im, *, exclude=None, n_px=9, sample_step=3):
    """Fraction of revealed-boundary samples with white casing ink
    within n_px, + the sample count. exclude = (cx, cy, r) image-px
    circle around the advancing front tip."""
    import numpy as np
    px = np.asarray(im.convert("RGB"), dtype=np.int16)
    ocean = np.array(OCEAN_RGB, dtype=np.int16)
    # tolerance 18: the cone fill blend sits 24+ off the ocean, while
    # the card-panel bg (#11161f, max-channel diff 13) that can bleed
    # into the clip box at element edges stays below it.
    revealed = np.abs(px - ocean).max(axis=2) > 18
    ink = ((px[:, :, 0] > 165) & (px[:, :, 1] > 185)
           & (px[:, :, 2] > 205))
    # the stage's rounded corners + element-edge AA leak the panel bg
    # into the screenshot border; the cone sits 200+px inside the
    # auto-fit margins, so a 32px frame drop costs nothing.
    revealed[:32, :] = False
    revealed[-32:, :] = False
    revealed[:, :32] = False
    revealed[:, -32:] = False
    er = revealed.copy()
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        er &= np.roll(revealed, sh, axis=ax)
    boundary = revealed & ~er
    cov = ink.copy()
    for _ in range(n_px):
        cov |= (np.roll(cov, 1, 0) | np.roll(cov, -1, 0)
                | np.roll(cov, 1, 1) | np.roll(cov, -1, 1))
    ys, xs = np.nonzero(boundary)
    if exclude is not None:
        cx, cy, r = exclude
        keep = (xs - cx) ** 2 + (ys - cy) ** 2 > r * r
        ys, xs = ys[keep], xs[keep]
    ys, xs = ys[::sample_step], xs[::sample_step]
    if ys.size == 0:
        return 1.0, 0
    return float(cov[ys, xs].mean()), int(ys.size)


@unittest.skipUnless(HAVE_RENDERER, "playwright/Pillow unavailable")
class TestConeCasingRidesTheFront(unittest.TestCase):
    ENGINES = ("chromium", "webkit")
    FRACS = (0.08, 0.17, 0.26, 0.35, 0.44, 0.53, 0.62, 0.71, 0.84, 0.92)
    MIN_COV = 0.985

    def test_casing_coverage_per_frame_and_settled_loop(self):
        storm = json.loads(FIXTURE.read_text())
        adv = json.loads(ADV_FIXTURE.read_text())
        html = cyclolab_shell.render_page(storm, feed_url=FEED_URL,
                                          loader="")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p, \
                tempfile.TemporaryDirectory() as td:
            page_file = Path(td) / "page.html"
            page_file.write_text(html, encoding="utf-8")
            for engine in self.ENGINES:
                browser = getattr(p, engine).launch()
                try:
                    self._run_engine(browser, engine, page_file, adv, td)
                finally:
                    browser.close()

    def _shot(self, pg, box, path):
        pg.screenshot(path=str(path), clip=box)
        return Image.open(path)

    def _run_engine(self, browser, engine, page_file, adv, td):
        pg = browser.new_page(viewport={"width": 1280, "height": 900},
                              device_scale_factor=2)
        pg.goto(f"file://{page_file}")
        # The widget-size pass caps #advcone at --panel-max-h (a live-layout
        # concern that letterboxes the SVG); this test measures the casing's
        # GEOMETRY along the reveal front, so render the cone at its full
        # auto-fit scale here by lifting the cap.
        pg.evaluate("() => document.documentElement.style"
                    ".setProperty('--panel-max-h', 'none')")
        pg.wait_for_timeout(1200)
        pg.evaluate("a => window.__lab.applyAdvisory(a)", adv)
        pg.evaluate("window.__lab.openSec('advisories')")
        pg.wait_for_timeout(400)
        pg.evaluate(HIDE_FURNITURE_JS)
        box = pg.evaluate("""() => {
          const r = document.getElementById('advcone')
            .getBoundingClientRect();
          return {x: r.x, y: r.y, width: r.width, height: r.height};
        }""")
        shot = Path(td) / f"casing_{engine}.png"
        checked = 0
        for f in self.FRACS:
            seek = pg.evaluate(
                "f => window.__lab.cone().seek(f)", f)
            pg.wait_for_timeout(80)         # commit the paint
            im = self._shot(pg, box, shot)
            scale = im.size[0] / seek["W"]
            tip = (seek["tipX"] * scale, seek["tipY"] * scale,
                   (seek["w"] * 1.6 + 24) * scale)
            cov, n = casing_coverage(im, exclude=tip)
            if n < 60:
                continue        # barely-grown frames: nothing to walk
            checked += 1
            self.assertGreaterEqual(
                cov, self.MIN_COV,
                f"{engine} f={f}: only {cov:.1%} of {n} revealed-"
                f"boundary samples have casing ink within 9px - the "
                f"revealed region does not look finished")
        # the skip guard must not have eaten the test: most frames
        # carry a walkable boundary on a real cone.
        self.assertGreaterEqual(
            checked, 7,
            f"{engine}: only {checked}/10 frames had a walkable "
            f"boundary - the probe is not seeing the reveal")
        # the settle frame: one complete closed loop, no exclusions
        pg.evaluate("window.__lab.cone().settle()")
        pg.wait_for_timeout(80)
        im = self._shot(pg, box, shot)
        cov, n = casing_coverage(im, exclude=None)
        self.assertGreater(n, 200, f"{engine}: settled cone too small")
        self.assertGreaterEqual(
            cov, self.MIN_COV,
            f"{engine} settle: casing loop incomplete ({cov:.1%})")
        # PROBE TEETH (in-suite negative control): hiding the WHITE casing
        # highlight must collapse coverage - the probe can fail. maps-pass:
        # the cone group is now glass-fill[0], bevel dark[1]/blue[2]/light
        # WHITE highlight[3], centerline casing[4], white centerline[5] - so
        # the white-casing ink the coverage detector keys on is child[3]
        # (was child[1], the old single white boundary stroke).
        pg.evaluate("""() => {
          document.querySelector('.ac-conegrp').children[3]
            .style.opacity = '0';
        }""")
        pg.wait_for_timeout(80)
        im = self._shot(pg, box, shot)
        cov_teeth, _ = casing_coverage(im, exclude=None)
        self.assertLess(
            cov_teeth, 0.6,
            f"{engine}: probe kept reporting {cov_teeth:.1%} with the "
            f"casing hidden - it has no teeth")
        pg.close()


# ---------------------------------------------------------------------------
# FINAL-GATE R2 #5 - the satellite PRESENTED-frame cadence contract:
# canvas + ImageBitmap presentation on a rAF clock, measured on a 4x
# CPU-throttled chromium (CDP Emulation.setCPUThrottlingRate) plus an
# unthrottled webkit cross-engine pass. Contract: over a full loop
# (after a one-loop warmup) no presented interval exceeds 2x the
# median, and the median holds the 200 ms frame grid.
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_RENDERER, "playwright/Pillow unavailable")
class TestSatellitePresentedCadence(unittest.TestCase):
    N_FRAMES = 20
    FRAME_MS = 200

    @classmethod
    def _frame_pngs(cls):
        import io as _io

        import numpy as np
        rng = np.random.default_rng(11)
        out = []
        for _ in range(cls.N_FRAMES):
            arr = rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)
            buf = _io.BytesIO()
            Image.fromarray(arr, "RGB").save(buf, "PNG")
            out.append(buf.getvalue())
        return out

    def _route(self, pg, frames):
        man = {"bands": {"ir": {"label": "IR", "frames": [
            {"t": f"2026-06-06T{i // 60:02d}:{i % 60:02d}:00Z",
             "key": f"floaters/ep082026/ir/f{i:03d}.png"}
            for i in range(self.N_FRAMES)]}}}
        top = {"storms": [{"id": "ep082026",
                           "manifest": "floaters/ep082026/manifest.json"}]}

        def handler(route):
            # strip the cache-bust query before matching
            path = route.request.url.split("?")[0]
            hdrs = {"Access-Control-Allow-Origin": "*"}
            if path.endswith("floaters/manifest.json"):
                route.fulfill(json=top, headers=hdrs)
            elif path.endswith("/manifest.json"):
                route.fulfill(json=man, headers=hdrs)
            elif path.endswith(".png"):
                i = int(path.rsplit("/f", 1)[1][:3])
                route.fulfill(body=frames[i], headers=hdrs,
                              content_type="image/png")
            else:
                route.abort()

        pg.route("https://cdn.triple-a-tropics.com/floaters/**", handler)

    def _measure(self, p, engine, page_file, frames, throttle):
        browser = getattr(p, engine).launch()
        try:
            pg = browser.new_page(
                viewport={"width": 1100, "height": 760})
            self._route(pg, frames)
            pg.goto(f"file://{page_file}")
            pg.wait_for_timeout(1200)
            pg.evaluate("window.__lab.openSec('satellite')")
            pg.wait_for_timeout(1500)        # mount + initial decode
            st0 = pg.evaluate("window.__lab.satState()")
            self.assertEqual(st0["frames"], self.N_FRAMES, engine)
            self.assertEqual(st0["mode"], "canvas",
                             f"{engine}: presentation path is not the "
                             f"canvas/ImageBitmap pipeline")
            if throttle:
                cdp = pg.context.new_cdp_session(pg)
                cdp.send("Emulation.setCPUThrottlingRate", {"rate": 4})
            pg.evaluate(
                "document.getElementById('sat-play').click()")
            pg.wait_for_timeout(
                int(self.N_FRAMES * self.FRAME_MS * 2.6) + 1200)
            st = pg.evaluate("window.__lab.satState()")
            pg.close()
        finally:
            browser.close()
        pres = st["presented"]
        self.assertGreaterEqual(
            len(pres), self.N_FRAMES * 2,
            f"{engine}{' @4x' if throttle else ''}: only {len(pres)} "
            f"presented frames - playback starved")
        # drop the first loop (decode warmup), measure the steady loop
        iv = [b - a for a, b in zip(pres[self.N_FRAMES:],
                                    pres[self.N_FRAMES + 1:])]
        self.assertGreaterEqual(len(iv), self.N_FRAMES - 2)
        iv_sorted = sorted(iv)
        med = iv_sorted[len(iv_sorted) // 2]
        label = f"{engine}{' @4x' if throttle else ''}"
        self.assertLess(abs(med - self.FRAME_MS), self.FRAME_MS * 0.5,
                        f"{label}: median interval {med:.0f}ms is off "
                        f"the {self.FRAME_MS}ms grid")
        self.assertLessEqual(
            max(iv), 2 * med,
            f"{label}: worst presented interval {max(iv):.0f}ms > 2x "
            f"median {med:.0f}ms - cadence contract broken "
            f"(intervals: {[round(x) for x in iv_sorted[-5:]]})")
        return iv

    def _measure_with_retry(self, p, engine, page_file, frames,
                            throttle):
        # shared-hardware flake guard (adversarial-review find): one
        # foreign-process stall can blow a single 4x-throttled run.
        # The contract must hold on at least one of two INDEPENDENT
        # measurements - a real regression fails both.
        try:
            return self._measure(p, engine, page_file, frames,
                                 throttle=throttle)
        except AssertionError as first:
            print(f"[cadence] {engine} first run failed "
                  f"({first}); re-measuring once")
            return self._measure(p, engine, page_file, frames,
                                 throttle=throttle)

    def test_even_cadence_throttled_chromium_and_webkit(self):
        storm = json.loads(FIXTURE.read_text())
        html = cyclolab_shell.render_page(storm, feed_url=FEED_URL,
                                          loader="")
        frames = self._frame_pngs()
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p, \
                tempfile.TemporaryDirectory() as td:
            page_file = Path(td) / "page.html"
            page_file.write_text(html, encoding="utf-8")
            # THE contract run: 4x CPU throttle on chromium (CDP).
            self._measure_with_retry(p, "chromium", page_file, frames,
                                     throttle=True)
            # cross-engine canvas-path coverage (webkit has no CDP
            # throttle; unthrottled contract still pins the pipeline).
            self._measure_with_retry(p, "webkit", page_file, frames,
                                     throttle=False)


RADII_FIXTURE = HERE / "fixtures" / "cyclolab" / "synth_storm_radii.json"


def _radii_storm():
    """The radii fixture with the current fix bumped to an active major
    hurricane (all three ring tiers + 64-kt core render); the committed
    fixture on disk is left honest/weakening."""
    s = json.loads(RADII_FIXTURE.read_text(encoding="utf-8"))
    cur = s["points"][-1]
    cur["wind_kt"] = 90.0
    cur["pressure_mb"] = 958.0
    cur["nature"] = "TS"
    cur["radii"] = {"34": [110, 100, 80, 92], "50": [62, 54, 40, 50],
                    "64": [34, 28, 20, 26]}
    s["current_category"] = "C4"          # major hurricane -> orange accent
    return s


@unittest.skipUnless(HAVE_RENDERER, "playwright + Pillow required")
class TestLockupRailAndContainment(unittest.TestCase):
    """FG-R3 #3a/#3b: every overview-plot lockup rail is the house blue
    (#3fa4ff) regardless of storm category, and the backing panel CONTAINS
    every line - eyebrow included - even for the longest worst-case name."""

    ENGINES = ("chromium", "webkit")
    RAIL_RGB = "rgb(63, 164, 255)"        # #3fa4ff, the house blue

    @classmethod
    def setUpClass(cls):
        html = cyclolab_shell.render_page(_radii_storm(), feed_url=FEED_URL,
                                          loader="")
        cls._td = tempfile.TemporaryDirectory()
        cls.page = Path(cls._td.name) / "page.html"
        cls.page.write_text(html, encoding="utf-8")

    def _measure(self, engine, name, typ):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = getattr(p, engine).launch()
            pg = b.new_page(viewport={"width": 1380, "height": 1700},
                            device_scale_factor=2)
            pg.goto(f"file://{self.page}")
            pg.wait_for_timeout(3200)
            out = pg.evaluate(
                """([nm, tp]) => {
                  document.getElementById('storm-name').textContent = nm;
                  document.getElementById('storm-type').textContent = tp;
                  // C4 is already set from the bake; keep it (orange accent)
                  window.__lab.renderTrackPlot();
                  window.__lab.renderSwathPlot();
                  const res = [];
                  // maps-pass R5: the lockup is an HTML overlay (.map-lockup)
                  // with a border-left RAIL; every .ml-* line must sit inside
                  // its box (no overflow).
                  ['trackplot', 'swathplot'].forEach(id => {
                    const box = document.getElementById(id + '-lockup');
                    if (!box || box.hidden) return;
                    const br = box.getBoundingClientRect();
                    const rail = getComputedStyle(box).borderLeftColor;
                    const lines = [];
                    box.querySelectorAll('.ml-eyebrow,.ml-head,.ml-sub')
                      .forEach(t => {
                        const tb = t.getBoundingClientRect();
                        lines.push({ txt: t.textContent.trim(),
                                     left: tb.left, right: tb.right });
                      });
                    res.push({ sel: '#' + id,
                      bgLeft: br.left, bgRight: br.right,
                      railFill: rail, lines });
                  });
                  return res;
                }""", [name, typ])
            b.close()
        return out

    def _check(self, engine, name, typ):
        plots = self._measure(engine, name, typ)
        self.assertEqual(len(plots), 2, f"{engine}: both plots present")
        for pl in plots:
            self.assertEqual(
                pl["railFill"], self.RAIL_RGB,
                f"{engine} {pl['sel']}: rail must be the house blue, "
                f"got {pl['railFill']}")
            for ln in pl["lines"]:
                self.assertLessEqual(
                    ln["right"], pl["bgRight"] + 1.5,
                    f"{engine} {pl['sel']}: '{ln['txt']}' overflows the "
                    f"panel right ({ln['right']:.1f} > {pl['bgRight']:.1f})")
                self.assertGreaterEqual(
                    ln["left"], pl["bgLeft"] - 1.5,
                    f"{engine} {pl['sel']}: '{ln['txt']}' spills past the "
                    f"panel left")

    def test_rail_is_house_blue_and_lines_fit_default_name(self):
        for e in self.ENGINES:
            self._check(e, "SYNTH", "HURRICANE")

    def test_long_name_eyebrow_and_sub_stay_inside_the_panel(self):
        # the worst-case long name: the sub line is the widest line.
        for e in self.ENGINES:
            self._check(e, "EIGHTEEN-E", "TROPICAL DEPRESSION")


@unittest.skipUnless(HAVE_RENDERER, "playwright + Pillow required")
class TestWindRoundingSweep(unittest.TestCase):
    """FG-R3 #4: with a converted unit selected, NO wind value renders off a
    5-boundary ANYWHERE we draw winds - hero, vitals, cone placards,
    intensity + W&P axis ticks, track legend / colorbar, field labels, plot
    captions. (windDisp is unit-tested exhaustively in the jsdom suite; this
    proves nothing on the live page bypasses it.) NHC's verbatim advisory
    text is excluded - it is already agency-rounded, not our conversion."""

    def _sweep(self, unit):
        adv = json.loads(ADV_FIXTURE.read_text(encoding="utf-8"))
        html = cyclolab_shell.render_page(_radii_storm(), feed_url=FEED_URL,
                                          loader="")
        td = tempfile.TemporaryDirectory()
        page = Path(td.name) / "page.html"
        page.write_text(html, encoding="utf-8")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1380, "height": 2000},
                            device_scale_factor=2)
            pg.goto(f"file://{page}?units={unit}")    # boot in the unit
            pg.wait_for_timeout(3200)
            nums = pg.evaluate(
                """(adv) => {
                  window.__lab.applyAdvisory(adv);
                  window.__lab.openSec('advisories');
                  window.__lab.openSec('overview');
                  window.__lab.renderTrackPlot();
                  window.__lab.renderSwathPlot();
                  const winds = [];
                  // skip NHC's verbatim advisory text (id/class 'advtext')
                  const inAdvText = (el) => {
                    while (el) {
                      const id = el.id || '';
                      const cl = (el.getAttribute &&
                        el.getAttribute('class')) || '';
                      if (/advtext/i.test(id) || /advtext/i.test(cl))
                        return true;
                      el = el.parentElement;
                    }
                    return false;
                  };
                  const tw = document.createTreeWalker(
                    document.body, NodeFilter.SHOW_TEXT);
                  let node;
                  while ((node = tw.nextNode())) {
                    if (inAdvText(node.parentElement)) continue;
                    const re = /(\\d+)\\s*(?:mph|km\\/h)/g; let m;
                    while ((m = re.exec(node.textContent || '')))
                      winds.push(+m[1]);
                  }
                  // bare-number SVG wind labels (number + unit are separate
                  // <text> nodes, so the inline regex misses them).
                  document.querySelectorAll(
                    '.tp-cbar-tick,.tp-field-lbl,.mwk-tier,'
                    + '.wp-ytick,.in-ytick').forEach((t) => {
                      const v = (t.textContent || '').trim();
                      if (/^\\d+$/.test(v)) winds.push(+v);
                    });
                  return winds;
                }""", adv)
            b.close()
        return nums

    def test_mph_no_value_off_a_5_boundary(self):
        nums = self._sweep("mph")
        self.assertGreaterEqual(len(nums), 12,
                                f"too few wind values swept: {nums}")
        bad = sorted({n for n in nums if n % 5 != 0})
        self.assertEqual(bad, [], f"mph wind values off a 5-boundary: {bad}")

    def test_kmh_no_value_off_a_5_boundary(self):
        nums = self._sweep("kmh")
        self.assertGreaterEqual(len(nums), 12,
                                f"too few wind values swept: {nums}")
        bad = sorted({n for n in nums if n % 5 != 0})
        self.assertEqual(bad, [], f"km/h wind values off a 5-boundary: {bad}")


@unittest.skipUnless(HAVE_RENDERER, "playwright + Pillow required")
class TestSettingsControlPlacement(unittest.TestCase):
    """The in-app settings cog lives in the SIDEBAR FOOTER, never overlapping
    the banner's corner cyclone glyph - at every category and panel size."""

    CATS = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
    SIZES = [("desktop", 1280, 900), ("mobile", 390, 850)]

    def _measure(self, pg, cat):
        return pg.evaluate(
            """(cat) => {
              if (window.__lab && window.__lab.setCategory) __lab.setCategory(cat);
              const g = document.querySelector('.banner .glyph');
              const s = document.getElementById('settings-btn');
              const foot = document.querySelector('.side-foot');
              const banner = document.querySelector('.banner');
              if (!g || !s) return { missing: true };
              const rg = g.getBoundingClientRect(), rs = s.getBoundingClientRect();
              const ov = !(rg.right <= rs.left || rs.right <= rg.left ||
                           rg.bottom <= rs.top || rs.bottom <= rg.top);
              return { overlap: ov,
                       inFooter: !!(foot && foot.contains(s)),
                       inBanner: !!(banner && banner.contains(s)),
                       sVisible: rs.width > 0 && rs.height > 0 };
            }""", cat)

    def test_settings_never_overlaps_banner_glyph(self):
        from playwright.sync_api import sync_playwright
        storm = json.loads(FIXTURE.read_text(encoding="utf-8"))
        html = cyclolab_shell.render_page(storm, feed_url=FEED_URL, loader="")
        with tempfile.TemporaryDirectory() as td:
            page = Path(td) / "page.html"
            page.write_text(html, encoding="utf-8")
            with sync_playwright() as p:
                b = p.chromium.launch()
                for label, w, h in self.SIZES:
                    pg = b.new_page(viewport={"width": w, "height": h},
                                    device_scale_factor=2)
                    pg.goto(f"file://{page}")
                    pg.wait_for_timeout(1200)
                    for cat in self.CATS:
                        m = self._measure(pg, cat)
                        pg.wait_for_timeout(60)
                        self.assertFalse(m.get("missing"),
                                         f"{label}/{cat}: glyph or settings missing")
                        self.assertTrue(m["sVisible"], f"{label}/{cat}: settings hidden")
                        self.assertTrue(m["inFooter"],
                                        f"{label}/{cat}: settings not in .side-foot")
                        self.assertFalse(m["inBanner"],
                                         f"{label}/{cat}: settings still inside .banner")
                        self.assertFalse(m["overlap"],
                                         f"{label}/{cat}: settings overlaps the "
                                         "banner glyph")
                    pg.close()
                b.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# Satellite AUTO-REFRESH (the manifest re-poll): a new manifest frame must
# appear within one poll WITHOUT a reload; scrub/pause state survives the
# append; follow-live advances on newest; the inactive note shows on a stale
# floater; an ENDED archive page never polls.
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_RENDERER, "playwright/Pillow unavailable")
class TestSatelliteAutoRefresh(unittest.TestCase):
    @classmethod
    def _pngs(cls, n):
        import io as _io

        import numpy as np
        rng = np.random.default_rng(7)
        out = []
        for _ in range(n):
            arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            buf = _io.BytesIO()
            Image.fromarray(arr, "RGB").save(buf, "PNG")
            out.append(buf.getvalue())
        return out

    @staticmethod
    def _times(n, newest_age_min):
        import datetime as dt
        newest = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            minutes=newest_age_min)
        return [(newest - dt.timedelta(minutes=15 * (n - 1 - i)))
                .strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)]

    def _route(self, pg, state, frames):
        def handler(route):
            path = route.request.url.split("?")[0]
            hdrs = {"Access-Control-Allow-Origin": "*"}
            n = state["n"]
            times = state["times"]
            if path.endswith("floaters/manifest.json"):
                route.fulfill(json={"storms": [{"id": "ep082026",
                              "manifest": "floaters/ep082026/manifest.json"}]},
                              headers=hdrs)
            elif path.endswith("/manifest.json"):
                man = {"bands": {"ir": {"label": "IR", "frames": [
                    {"t": times[i], "key": f"floaters/ep082026/ir/f{i:03d}.png"}
                    for i in range(n)]}}}
                route.fulfill(json=man, headers=hdrs)
            elif path.endswith(".png"):
                i = int(path.rsplit("/f", 1)[1][:3])
                route.fulfill(body=frames[min(i, len(frames) - 1)],
                              headers=hdrs, content_type="image/png")
            else:
                route.abort()
        pg.route("https://cdn.triple-a-tropics.com/floaters/**", handler)

    def _boot(self, p, total, shown, newest_age_min=2, ended=False):
        """render + open the Satellite tab showing `shown` of `total` frames;
        returns (browser, pg, state, frames)."""
        storm = json.loads(FIXTURE.read_text())
        html = cyclolab_shell.render_page(storm, feed_url=FEED_URL,
                                          loader="", ended=ended)
        frames = self._pngs(total)
        state = {"n": shown, "times": self._times(total, newest_age_min)}
        browser = p.chromium.launch()
        td = tempfile.mkdtemp()
        page_file = Path(td) / "page.html"
        page_file.write_text(html, encoding="utf-8")
        pg = browser.new_page(viewport={"width": 1100, "height": 760})
        self._route(pg, state, frames)
        pg.goto(f"file://{page_file}")
        pg.wait_for_timeout(900)
        pg.evaluate("window.__lab.openSec('satellite')")
        pg.wait_for_timeout(1400)
        return browser, pg, state, frames

    def test_new_frame_appears_within_one_poll_no_reload(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b, pg, state, _ = self._boot(p, total=12, shown=8)
            try:
                st0 = pg.evaluate("window.__lab.satState()")
                self.assertEqual(st0["frames"], 8)
                # a new frame lands on the CDN; one poll picks it up.
                state["n"] = 9
                pg.evaluate("window.__lab.satPollNow()")
                pg.wait_for_timeout(300)
                st1 = pg.evaluate("window.__lab.satState()")
                self.assertEqual(st1["frames"], 9,
                                 "new manifest frame did not append on poll")
                self.assertEqual(len(st1["frameKeys"]), 9)
            finally:
                b.close()

    def test_follow_live_advances_when_on_newest(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b, pg, state, _ = self._boot(p, total=12, shown=8)
            try:
                # land on newest (default) + paused; a new frame -> advance.
                st0 = pg.evaluate("window.__lab.satState()")
                self.assertEqual(st0["idx"], 7)
                state["n"] = 9
                pg.evaluate("window.__lab.satPollNow()")
                pg.wait_for_timeout(300)
                st1 = pg.evaluate("window.__lab.satState()")
                self.assertEqual(st1["frames"], 9)
                self.assertEqual(st1["idx"], 8, "did not follow-live to newest")
            finally:
                b.close()

    def test_scrub_state_survives_append(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b, pg, state, _ = self._boot(p, total=12, shown=8)
            try:
                # scrub back to an OLDER frame (paused); append must NOT jump.
                pg.evaluate("window.__lab.satState && "
                            "document.getElementById('sat-scrub')")
                pg.eval_on_selector("#sat-scrub", "el => { el.value = '3'; "
                                    "el.dispatchEvent(new Event('input')); }")
                pg.wait_for_timeout(150)
                self.assertEqual(pg.evaluate("window.__lab.satState()")["idx"], 3)
                state["n"] = 10
                pg.evaluate("window.__lab.satPollNow()")
                pg.wait_for_timeout(300)
                st = pg.evaluate("window.__lab.satState()")
                self.assertEqual(st["frames"], 10, "frames did not extend")
                self.assertEqual(st["idx"], 3,
                                 "scrub position jumped on append")
            finally:
                b.close()

    def test_inactive_note_when_floater_is_stale(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            # newest frame is 3 h old -> floater inactive.
            b, pg, _, _ = self._boot(p, total=8, shown=8, newest_age_min=180)
            try:
                st = pg.evaluate("window.__lab.satState()")
                self.assertTrue(st["inactive"], "stale floater not flagged")
                note = pg.eval_on_selector(
                    "#sat-inactive", "el => ({hidden: el.hidden, "
                    "text: el.textContent})")
                self.assertFalse(note["hidden"])
                self.assertIn("Floater inactive", note["text"])
            finally:
                b.close()

    def test_ended_archive_never_polls(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b, pg, _, _ = self._boot(p, total=8, shown=8, ended=True)
            try:
                st = pg.evaluate("window.__lab.satState()")
                self.assertFalse(st["polling"],
                                 "ENDED archive page armed a manifest poll")
            finally:
                b.close()
