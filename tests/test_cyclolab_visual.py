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
  const el = document.getElementById(id);
  const em = parseFloat(getComputedStyle(el).fontSize);
  const want = el.getAttribute('data-odo') || el.textContent;
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
  const ref = el.querySelector('.digit') || el;
  const cs = getComputedStyle(ref);
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
            cls.rest_cells = pg.evaluate("""() =>
              [...document.querySelectorAll('.odo .digit:not(.ch)')]
                .filter(c => !c.closest('.col'))
                .map(c => {
                  const r = c.getBoundingClientRect();
                  return {box: [r.left, r.top, r.right, r.bottom],
                          em: parseFloat(getComputedStyle(c).fontSize),
                          odo: (c.closest('.odo') || {}).id || '?'};
                })""")
            cls.col_count = pg.evaluate(
                "document.querySelectorAll('.odo .col').length")
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
                pg.evaluate("document.querySelectorAll('.odo .col').length"),
                0, f"{engine}/dpr{dpr}: cols still mounted after settle "
                   f"(value {val})")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
