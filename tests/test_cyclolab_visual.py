"""VISUAL regression tests for the CycloLab shell - these render the
real page in a real engine and measure INK PIXELS, not DOM attributes.

Born from AD Round 3: the R2 suite was green while the rendered card
had flat-sheared digit bottoms, superscript-looking units and a muddy
wordmark. Unit tests assert what the code SAYS; these assert what the
user SEES:

  * no-clip: the hero odometer's digit ink must be pixel-equal in
    height to an unclipped same-font reference (a clipped "0" loses its
    baseline overshoot and measures shorter);
  * true baselines: digit ink bottoms == unit ink bottoms == label ink
    bottoms (digits + the uppercase/no-descender units used here bottom
    out exactly at the baseline);
  * glyph category label: actually painted in the category accent.

Skips cleanly when playwright/chromium or Pillow are unavailable (CI
without browsers); run locally before any visual sign-off.
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

try:
    from PIL import Image
    from playwright.sync_api import sync_playwright
    with sync_playwright() as _p:
        _b = _p.chromium.launch()
        _b.close()
    HAVE_RENDERER = True
except Exception:  # noqa: BLE001 - any miss (import, browser) -> skip
    HAVE_RENDERER = False


def ink_rows(im, box):
    """(top, bottom) device-pixel rows containing ink (pixels clearly
    brighter than the panel background) within a CSS-px box."""
    l, t, r, b = (int(v * DPR) for v in box)
    px = im.crop((l, t, r, b)).convert("L")
    w, h = px.size
    rows = [max(px.getpixel((x, y)) for x in range(w)) for y in range(h)]
    bg = sorted(rows)[max(0, len(rows) // 10)]
    ink = [y for y, v in enumerate(rows) if v > bg + 60]
    return (t + ink[0], t + ink[-1]) if ink else None


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
            pg.wait_for_timeout(3000)   # boot apply + font load + roll
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
                       mslpLbl: g('#vrow-mslp .lbl'),
                       mslpOdo: g('#odo-mslp'),
                       mslpUnit: g('#vrow-mslp .unit'),
                       glyph: g('.banner .glyph') };
            }""")
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
        # ink HEIGHT of the rolled odometer "120" == unclipped reference
        # "120" (same engine, same font). A box-edge clip shears the
        # bottom overshoot and fails this by several device px.
        odo_t, odo_b = self._ink("heroOdo")
        ref_t, ref_b = self._ink("ref")
        self.assertLessEqual(abs((odo_b - odo_t) - (ref_b - ref_t)), 2,
                             "hero digit ink height differs from the "
                             "unclipped reference - glyphs are clipped")

    def test_hero_unit_shares_the_digit_baseline(self):
        # digits and the uppercase KT unit both bottom out at the
        # baseline; their ink bottoms must agree within 2 device px.
        _, digit_bottom = self._ink("heroOdo")
        _, unit_bottom = self._ink("heroUnit")
        self.assertLessEqual(abs(digit_bottom - unit_bottom), 2,
                             "KT does not sit on the hero digits' baseline")

    def test_vitals_row_shares_one_baseline(self):
        _, lbl_b = self._ink("mslpLbl")
        _, val_b = self._ink("mslpOdo")
        _, unit_b = self._ink("mslpUnit")
        self.assertLessEqual(abs(val_b - lbl_b), 2,
                             "label and value baselines differ")
        self.assertLessEqual(abs(val_b - unit_b), 2,
                             "value and unit baselines differ")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
