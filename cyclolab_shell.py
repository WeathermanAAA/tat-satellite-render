"""CycloLab per-storm shell - template + renderer (Stage 2, AD round 1).

The poller renders this template on storm birth and PUTs it via
R2Sink.write_html at cyclolab_pages.page_key(sid) (the Worker-pinned
edge contract). One self-contained document per storm: inline CSS
(category token sets + shell layout + motion), inline vanilla JS
(hydration on the poll+diff-merge discipline), real per-storm OG tags.
No CDN deps; typography is self-hosted Metropolis (vendored in the main
repo at /assets/fonts/metropolis, public domain) referenced absolutely
so both the live origin and the cdn-origin shadow pages load it (Pages
serves ACAO:*).

AD ROUND 1 (2026-06-06):
  * Metropolis everywhere; wordmark "CycloLab" in Extra Bold with the
    BRAND CASING RULE - always literally "CycloLab", never uppercased.
  * Banner = broadcast storm-info card: eyebrow / storm-type word (from
    the advisory's tau-0 dev_label - the second-pass-fixed path - with a
    category+basin fallback) / dominant NAME / chip, plus a corner
    spinning hurricane glyph (white, category-tinted glow, CONSTANT
    2.6s rotation - intensity is carried by color, never speed; the one
    permitted continuous loop, reduced-motion = static; same gate the
    cone's forecast icons will share in Stage 4).
  * Ramps re-based on the CANONICAL TAT Saffir-Simpson palette
    (ace_core.SSHS_COLORS - the same single source the tracks maps and
    category pills use); gloss derived from the canonical base. NO ink
    flip anywhere: labels stay light, legibility on the bright C1/C2
    ramps comes from a soft dark text-shadow scrim.
  * Layout inverted: vitals live in a PERSISTENT sidebar card (visible
    in every section) incl. Movement and the advisory countdown; the
    main stage's home view is MAP + CONE (cone polygon rendered from
    the Stage-1 advisory JSON in the brand blue/white treatment; the
    Stage-4 choreography animates it later). Mobile: vitals collapse to
    a slim card above the map; bottom tabs unchanged.
  * Advisory countdown ticks toward the advisory's OWN stated
    next-advisory time (parsed by the poller into the adv JSON - never
    wall-clock cadence); past due -> "ADVISORY DUE · updating".
  * Loader framework with four prototype variants (a intensifying
    ramp / b eye opens / c wordmark shine / d broadcast sweep) for
    art-direction selection; default stays the plain wipe until a pick.

AD ROUND 2 (2026-06-06) - the integrated storm-info "bug" card:
  * Banner + vitals merged into ONE card (.bug): the gradient identity
    header zone (eyebrow / type word / NAME / chip / corner spinning
    glyph, glyph stays top-RIGHT) flows directly into a panel body -
    no second box, no gap.
  * The corner glyph carries the storm's category number/letter,
    reusing the CANONICAL icon treatment from the tracks maps + storm
    cards (generate_tracks_plot.py sshs_label + spinnerSvg: stationary
    white 900-weight text w/ dark stroke, D / S / 1-5, only the path
    spins). One canon, no new style.
  * Max Wind + Category promoted to HERO numbers in the card body
    (broadcast-reference treatment); both ride the odometer.
  * Remaining vitals (Min Pressure / Storm ACE / Position / Movement /
    Last Fix / Next Advisory) are compact LEFT-ALIGNED label-value
    pairs on a fixed label column - the split-justified
    value-pinned-right layout is gone (it read as ragged baselines).
  * Mobile keeps the AD-1 rotation: .bug flattens (display: contents)
    so the banner is the same sticky top bar; heroes + vitals stay ONE
    slim card (.bug-body) above the map.

AD ROUND 3 (2026-06-06) - render-verified fixes (the R2 unit tests were
green while the RENDERED output stayed broken; every item below was
reproduced and re-verified by screenshot/ink-scan, cross-engine):
  * Odometer rebuilt: in-flow invisible anchor digit per rolling cell
    (TRUE baseline) + abspos strip + per-cell clip slack. Kills BOTH
    the flat-sheared digit bottoms (box-edge clipping) and the
    synthesized-baseline drift that made units look superscripted.
  * Glyph category label wears the CATEGORY COLOR (--cat-accent) with
    a thin dark stroke - CSS-owned so it follows category changes.
  * Header glyph + loader eye ride an OVERSIZED viewBox (±44 vs ink
    reach ~41.4) so a full rotation never clips the swirl tails; box
    sizes compensate to keep ink size unchanged.
  * Eyebrow scrim scaled to type size (the 16px glow read as mud at
    10.5px - the "wordmark looks off" root cause).
  * Spin slowed: 2.6s -> 3.2s, header + loader coherent.
  * Loader B = THE pick (render_page default): iris reveal as approved,
    category-colored eye + glow, loader C's letter-build + shine
    wordmark (one shared builder).

AD ROUND 3b (2026-06-06) - the vitals text was STILL broken, pixel-
measured: (1) digits sank/wobbled BY VALUE (translateY in fractional
ems: the "9" sat 2px low) and (2) every round digit's baseline
overshoot was sheared flat by the rest-state clip (perfect bottom
uniformity = the FAILURE signature; healthy 0/3/5/6/8/9 dip 1-2px
below the baseline). Rebuild contract: REST IS PLAIN TEXT (a settled
cell is pixel-identical to static type, overshoot intact); strips
exist only DURING a roll, on a whole-device-pixel entry grid (1.7em
pitch snapped per DPR), inside a mask with >=0.25em slack beyond glyph
extents both sides (entry SPACING owns ghost prevention, never
clip-shaving); cells snap back to plain text on settle.
"""
from __future__ import annotations

import html as _html
import json

from ace_core import SSHS_COLORS  # the canonical category palette
from cyclolab_pages import adv_key, page_url_path
from cyclolab_basemap import basemap_for
from cyclolab_intensity import basin_entry
from storm_ids import parse_sid

FONT_BASE = "https://triple-a-tropics.com/assets/fonts/metropolis"

# The site's hurricane glyph (the same path the maps + banners spin).
HURRICANE_PATH = "M 16.37,-28.27 C 13.58,-28.13 11.51,-27.90 9.23,-27.49 C 1.27,-26.06 -5.88,-22.70 -10.92,-18.02 C -14.83,-14.40 -17.41,-10.06 -18.49,-5.32 C -18.95,-3.30 -19.15,-1.42 -19.15,0.91 C -19.15,2.53 -19.09,3.28 -18.89,4.45 C -18.38,7.38 -17.47,9.46 -15.41,12.37 C -13.88,14.54 -13.43,15.31 -13.20,16.13 C -13.11,16.44 -13.09,16.62 -13.09,17.14 C -13.10,17.93 -13.20,18.32 -13.67,19.28 C -15.30,22.59 -18.65,24.93 -23.49,26.14 C -25.26,26.58 -27.29,26.87 -29.18,26.95 L -30.00,26.98 L -29.65,27.06 C -27.33,27.62 -24.41,28.05 -21.57,28.27 C -20.04,28.38 -16.31,28.38 -14.80,28.27 C -12.93,28.13 -11.43,27.95 -9.77,27.67 C -0.59,26.14 7.56,22.03 12.68,16.37 C 16.22,12.45 18.28,8.10 18.93,3.13 C 19.64,-2.25 18.99,-6.47 16.84,-10.16 C 16.48,-10.80 15.79,-11.82 14.99,-12.95 C 13.61,-14.89 13.18,-15.77 13.12,-16.83 C 13.07,-17.61 13.23,-18.26 13.71,-19.23 C 14.97,-21.79 17.38,-23.84 20.67,-25.16 C 23.13,-26.14 26.24,-26.77 29.15,-26.87 L 30.00,-26.90 L 29.67,-26.98 C 29.13,-27.12 27.57,-27.44 26.66,-27.58 C 24.96,-27.87 23.39,-28.05 21.66,-28.18 C 20.72,-28.25 17.16,-28.30 16.37,-28.27 Z"


# --------------------------------------------------------------------------
# Category token sets, DERIVED from the canonical palette (no eyeballing).
# --------------------------------------------------------------------------
def _shade(hex_color: str, factor: float) -> str:
    """Scale a hex color toward black (factor 0..1 of original light)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        round(r * factor), round(g * factor), round(b * factor))


def _derive_tokens() -> dict[str, dict]:
    """edge/mid derived from each category's CANONICAL accent
    (ace_core.SSHS_COLORS) so the ramp reads as the real TAT category
    color with the approved gloss construction. Ink is ALWAYS light
    (AD rule: no black-text flip; scrims handle the bright ramps)."""
    out = {}
    for cat, accent in SSHS_COLORS.items():
        out[cat] = {"edge": _shade(accent, 0.30), "mid": _shade(accent, 0.62),
                    "accent": accent, "ink": "#ffffff"}
    return out


CAT_TOKENS: dict[str, dict] = _derive_tokens()


def _ramp(c: dict) -> str:
    return (f"linear-gradient(180deg,{c['edge']} 0%,{c['mid']} 22%,"
            f"{c['accent']} 50%,{c['mid']} 78%,{c['edge']} 100%)")


def cat_css() -> str:
    rules = []
    for cat, c in CAT_TOKENS.items():
        rules.append(
            f'html[data-cat="{cat}"] {{ --cat-ramp: {_ramp(c)}; '
            f"--cat-accent: {c['accent']}; --cat-ink: {c['ink']}; }}")
    return "\n  ".join(rules)


def font_css() -> str:
    faces = []
    for w in (400, 500, 600, 700, 800):
        faces.append(
            "@font-face { font-family: Metropolis; font-style: normal; "
            f"font-weight: {w}; font-display: swap; "
            f"src: url('{FONT_BASE}/metropolis-latin-{w}-normal.woff2') "
            "format('woff2'), "
            f"url('{FONT_BASE}/metropolis-latin-{w}-normal.woff') "
            "format('woff'); }")
    return "\n  ".join(faces)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-cat="__CAT__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__NAME__ · CycloLab · Triple-A-Tropics</title>
<meta name="description" content="__OG_DESC__">
<meta property="og:title" content="__OG_TITLE__">
<meta property="og:description" content="__OG_DESC__">
<meta property="og:type" content="website">
<meta property="og:url" content="https://triple-a-tropics.com__PAGE_PATH__">__OG_IMAGE__
<link rel="icon" type="image/svg+xml" href="/logo.svg">
<style>
  __FONT_CSS__

  :root {
    --bg: #0b0e13; --panel: #11161f; --border: #232a36;
    --fg: #e8eef5; --muted: #8ea2bd; --navy-deep: #0a1a2e;
    --cat-ramp: linear-gradient(180deg,#132c40,#3fa4ff,#132c40);
    --cat-accent: #3fa4ff; --cat-ink: #ffffff;
    --motion-slow: 4.5s; --motion-med: 1.2s; --motion-fast: 0.6s;
    /* the soft dark scrim that holds light text on bright ramps */
    --ink-scrim: 0 1px 2px rgba(0,0,0,0.65), 0 0 16px rgba(0,0,0,0.38);
  }
  __CAT_CSS__

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
    font-family: Metropolis, "Helvetica Neue", Arial, sans-serif; }
  a { color: #5dd3ff; text-decoration: none; }

  /* ---- shell: sidebar + stage (one DOM, CSS-rearranged on phones) ---- */
  .lab { display: flex; min-height: 100vh; }
  .side { width: 288px; flex: 0 0 288px; display: flex; flex-direction: column;
    background: linear-gradient(180deg, #0a1a2e 0%, #0d1420 30%, #0b0e13 100%);
    border-right: 1px solid var(--border); }
  .stage { flex: 1 1 auto; min-width: 0; padding: 26px 30px 60px; }

  /* ---- the integrated storm-info bug card (AD R2): ONE card -
         gradient identity header zone + hero numbers + inline vitals.
         No second box, no gap. ---- */
  .bug { margin: 8px; background: var(--panel);
    border: 1px solid var(--border); border-radius: 12px;
    overflow: hidden; }

  /* gradient identity header zone on the approved ramp. Corner glyph
     spins constantly (the one permitted loop); intensity is color,
     never speed. */
  .banner { background: var(--cat-ramp); color: var(--cat-ink);
    padding: 16px 16px 15px; position: relative; overflow: hidden;
    border-bottom: 1px solid rgba(255,255,255,0.85);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.14),
                inset 0 -1px 0 rgba(0,0,0,0.40); }
  /* small-text scrim: the full --ink-scrim's 16px glow (tuned for the
     30px loader word) reads as MUD at 10.5px and made the wordmark
     look off vs the loader's clean render - scale the shadow to the
     type size so the header "CycloLab" matches the loader exactly. */
  .banner .eyebrow { font-size: 10.5px; font-weight: 600;
    letter-spacing: 1.6px; opacity: 0.92;
    text-shadow: 0 1px 1px rgba(0,0,0,0.6), 0 0 5px rgba(0,0,0,0.30); }
  /* BRAND CASING RULE: "CycloLab" is always rendered literally - no
     text-transform here, ever. */
  .banner .eyebrow .brand { font-weight: 800; letter-spacing: 0.4px; }
  .banner .storm-type { font-size: 12px; font-weight: 700;
    letter-spacing: 2.2px; text-transform: uppercase; margin-top: 9px;
    opacity: 0.95; text-shadow: var(--ink-scrim); }
  .banner .storm-name { font-size: 30px; font-weight: 800; line-height: 1.05;
    letter-spacing: 0.5px; text-transform: uppercase; margin: 2px 0 9px;
    text-shadow: var(--ink-scrim); }
  .chip { display: inline-block; padding: 4px 12px; border-radius: 999px;
    background: rgba(0,0,0,0.32); color: #ffffff; font-size: 12px;
    font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.4);
    text-shadow: 0 1px 1px rgba(0,0,0,0.5); }
  /* glyph box is OVERSIZED relative to the ink (viewBox ±44 vs path
     reach ~41.4 when rotated) so a full 360° spin never clips the
     swirl tails; position compensates to keep the ink center put. */
  .banner .glyph { position: absolute; top: 3px; right: 3px;
    width: 60px; height: 60px; z-index: 1;
    filter: drop-shadow(0 0 7px var(--cat-accent))
            drop-shadow(0 1px 2px rgba(0,0,0,0.45)); }
  .banner .glyph .spin { animation: lab-spin 3.2s linear infinite;
    transform-origin: 0 0; }
  /* the canon category label rides the CATEGORY COLOR (canonical TAT
     scale via --cat-accent), not white-on-white; a thin dark stroke
     keeps the bright C1/C2 hues legible on the white glyph. CSS owns
     the treatment so it follows every category change. */
  .banner .glyph text { fill: var(--cat-accent);
    stroke: rgba(0,0,0,0.45); stroke-width: 1.1px;
    paint-order: stroke; }
  @keyframes lab-spin { from { transform: rotate(360deg); }
                        to { transform: rotate(0deg); } }
  .banner::after { content: ""; position: absolute; top: 0; bottom: 0;
    width: 55%; left: -60%; transform: skewX(-18deg);
    background: linear-gradient(90deg, transparent,
      rgba(255,255,255,0.32), transparent); pointer-events: none;
    opacity: 0; }
  .banner.shine::after { animation: lab-shine var(--motion-med) ease-out 1; }
  @keyframes lab-shine { 0% { opacity: 1; left: -60%; }
                         100% { opacity: 1; left: 110%; } }
  .banner .old-ramp { position: absolute; inset: 0;
    background: var(--old-ramp, none); opacity: 0; pointer-events: none; }
  .banner.xfade .old-ramp { opacity: 1;
    animation: lab-xfade calc(var(--motion-slow) * 0.5) ease-in-out 1 forwards; }
  @keyframes lab-xfade { from { opacity: 1; } to { opacity: 0; } }
  .banner > .b-inner { position: relative; z-index: 2; padding-right: 48px; }

  /* ---- card body: HERO numbers (Max Wind + Category, the broadcast
         reference treatment) over compact inline vitals ---- */
  .heroes { display: grid; grid-template-columns: 1fr auto 1fr;
    padding: 13px 16px 11px;
    border-bottom: 1px solid rgba(255,255,255,0.06); }
  .hero { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
  .hero .hero-val { font-size: 40px; font-weight: 800; line-height: 1.05;
    color: #ffffff; font-feature-settings: "tnum";
    font-variant-numeric: tabular-nums;
    display: flex; align-items: baseline; gap: 5px; }
  .hero .hero-val .unit { font-size: 11px; color: var(--muted);
    font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }
  .hero .hero-cap { font-size: 9.5px; font-weight: 700;
    letter-spacing: 1.4px; text-transform: uppercase; color: var(--muted); }
  .hero-div { width: 1px; background: rgba(255,255,255,0.08);
    margin: 1px 16px; }

  /* inline vitals: LEFT-ALIGNED "Label  value" pairs on a fixed label
     column + shared baseline (no split justification - the AD R2 fix
     for values reading at different levels). */
  .vitals { padding: 3px 16px 9px; }
  .vrow { display: grid; grid-template-columns: 102px 1fr;
    align-items: baseline; gap: 10px; padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05); }
  .vrow:last-child { border-bottom: 0; }
  .vrow .lbl { font-size: 10px; font-weight: 700; letter-spacing: 1.3px;
    text-transform: uppercase; color: var(--muted); }
  .vrow .val { font-size: 15.5px; font-weight: 700; color: #ffffff;
    font-feature-settings: "tnum"; font-variant-numeric: tabular-nums;
    white-space: nowrap; justify-self: start;
    display: inline-flex; align-items: baseline; gap: 4px; }
  .vrow .unit { font-size: 10.5px; color: var(--muted); font-weight: 700; }
  .vrow.due .val { color: var(--cat-accent); font-size: 12.5px;
    letter-spacing: 0.6px; }

  /* odometer (AD R3b rebuild - REST IS PLAIN TEXT).
     R2 clipped ink at the box edge (sheared bottoms + synthesized
     baseline). R3 anchored the baseline but kept every digit inside a
     clipped, translateY'd strip even when settled: the 1.15em cell
     height is fractional in device pixels, so translateY(-d x 1.15em)
     rounded differently per digit VALUE (the "9" sat 2px low) - and
     the rest-state clip sheared the round digits' baseline overshoot
     flat (0/3/5/6/8/9 are SUPPOSED to dip 1-2px below the baseline;
     that overshoot is correct typography, not a defect).
     The contract now:
       * AT REST a cell is a plain inline span on the row's natural
         baseline - no box, no clip, no transform. A settled 0 is
         pixel-identical to a 0 typed as static text, overshoot
         included. The card spends ~99% of its life here.
       * DURING A ROLL ONLY, the cell swaps in a .col: in-flow
         INVISIBLE anchor (keeps layout + true baseline) + an abspos
         .strip of ten SPACED entries (1.7em pitch, snapped to whole
         DEVICE pixels for the live DPR - integer geometry, so digit
         resting offsets can never accumulate per-value fractions)
         transitioning translateY. On settle the cell snaps back to
         plain text (timer = transition duration + slack).
       * the roll-time mask is inset(-0.3em 0): >=0.25em slack beyond
         the glyph extents on BOTH sides, so overshoot is CONTAINED,
         never clipped; the mask edges land mid-gap between the spaced
         strip entries, so neighbor entries stay clear of the window -
         entry SPACING owns ghost prevention, never clip-shaving.
     Digits ride fixed 0.62em cells (tnum); LETTER cells (.ch) are
     auto-width (W in "141.2W" must not clip); white-space:pre keeps
     space cells real. */
  .odo { display: inline-block; white-space: pre;
    font-feature-settings: "tnum"; font-variant-numeric: tabular-nums; }
  .odo .digit { display: inline-block; width: 0.62em; text-align: center;
    line-height: 1.15em; }
  .odo .digit.ch { width: auto; min-width: 0.3em; }
  .odo .col { position: relative; display: inline-block; width: 0.62em;
    height: 1.15em; line-height: 1.15em; text-align: center;
    clip-path: inset(-0.3em 0); }
  .odo .col .anchor { visibility: hidden; }
  .odo .col .strip { position: absolute; left: 0; top: 0; width: 100%;
    display: block;
    transition: transform calc(var(--motion-slow) * 0.35)
      cubic-bezier(0.22, 1, 0.36, 1); }
  .odo .col .strip .digit { display: block; width: 100%;
    height: var(--odo-eh, 1.7em); line-height: 1.15em; }

  /* section nav with the accent rail */
  .nav-secs { display: flex; flex-direction: column; padding: 10px 0;
    flex: 1 1 auto; }
  .sec-btn { display: flex; align-items: center; gap: 10px;
    padding: 13px 18px; min-height: 48px; background: transparent;
    color: var(--muted); border: 0; border-left: 3px solid transparent;
    font: inherit; font-size: 13.5px; font-weight: 700;
    letter-spacing: 1.1px; text-transform: uppercase; cursor: pointer;
    text-align: left; }
  .sec-btn:hover { color: var(--fg); }
  .sec-btn.active { color: #ffffff; border-left-color: var(--cat-accent);
    background: linear-gradient(90deg, rgba(255,255,255,0.05), transparent); }
  .back-map { padding: 16px 18px; border-top: 1px solid var(--border);
    font-size: 12.5px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; min-height: 48px; display: flex;
    align-items: center; }

  .ended-strip { display: none; background: #2a2f3a; color: #e8eef5;
    padding: 10px 16px; font-size: 13px; font-weight: 700;
    letter-spacing: 0.6px; text-align: center;
    border-bottom: 1px solid var(--border); }
  html[data-ended] .ended-strip { display: block; }

  /* ---- sections ---- */
  .sec { display: none; }
  .sec.active { display: block; }
  .sec-title { font-size: 17px; font-weight: 800; letter-spacing: 1.2px;
    text-transform: uppercase; color: #ffffff; margin: 0 0 14px; }
  .sec.active .wipe { animation: lab-wipe var(--motion-fast) ease-out 1; }
  @keyframes lab-wipe { from { opacity: 0; transform: translateX(14px); }
                        to   { opacity: 1; transform: translateX(0); } }

  .card { background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px; margin-bottom: 20px; }
  .card h3 { margin: 0 0 10px; font-size: 11.5px; font-weight: 800;
    letter-spacing: 1.4px; text-transform: uppercase; color: var(--muted); }
  .card svg { width: 100%; height: auto; display: block;
    touch-action: pan-y; }
  .card .note { font-size: 10.5px; color: var(--muted); margin-top: 8px;
    letter-spacing: 0.3px; }
  .draw path.series { stroke-dasharray: var(--len, 2000);
    stroke-dashoffset: var(--len, 2000);
    animation: lab-draw calc(var(--motion-slow) * 0.55) ease-out 1 forwards; }
  @keyframes lab-draw { to { stroke-dashoffset: 0; } }
  .draw .fill { opacity: 0;
    animation: lab-fill var(--motion-med) ease-out 1 forwards;
    animation-delay: calc(var(--motion-slow) * 0.45); }
  @keyframes lab-fill { to { opacity: 1; } }

  /* ---- Stage 3 mounts: shared viewer chrome (hafs.js vocabulary). The
     Models tab hosts the componentized /models/ HafsViewer (one impl, two
     mounts - CYCLOLAB_DESIGN §7.3); the Satellite tab reuses the same
     seg/btn/stage idiom for the floater viewer (§7.2). Colors ride the
     category tokens - the app wears the storm here too. */
  .hafs-group { display: flex; flex-direction: column; gap: 4px; }
  .hafs-group > label { font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: var(--muted); }
  .hafs-controls { display: flex; flex-wrap: wrap; gap: 12px 18px;
    padding: 0 0 12px; align-items: flex-end; }
  .hafs-controls select { background: var(--bg); color: #e8eef5;
    border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px;
    font: inherit; font-size: 13px; }
  .hafs-seg-group { display: flex; flex-wrap: wrap; gap: 0; }
  .hafs-seg { background: var(--bg); color: var(--muted);
    border: 1px solid var(--border); border-right: none;
    padding: 7px 12px; font: inherit; font-size: 12.5px; font-weight: 600;
    cursor: pointer; }
  .hafs-seg:first-child { border-radius: 6px 0 0 6px; }
  .hafs-seg:last-child { border-radius: 0 6px 6px 0;
    border-right: 1px solid var(--border); }
  .hafs-seg:hover { color: #ffffff; }
  .hafs-seg.active { background: var(--cat-accent); color: var(--cat-ink);
    border-color: var(--cat-accent); }
  .hafs-stage { position: relative; background: #0a0d12; min-height: 280px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 8px; overflow: hidden; }
  .hafs-stage img { display: block; max-width: 100%; max-height: 72vh;
    width: auto; height: auto; }
  .hafs-statusbox { position: absolute; inset: 0; display: none;
    align-items: center; justify-content: center; gap: 10px;
    color: var(--muted); background: rgba(10,13,18,0.6); font-size: 13.5px; }
  .hafs-spinner { width: 20px; height: 20px; border-radius: 50%;
    border: 3px solid var(--border); border-top-color: var(--cat-accent);
    animation: lab-hafs-spin 0.9s linear infinite; }
  @keyframes lab-hafs-spin { to { transform: rotate(360deg); } }
  #cl-hafs-buffer { position: absolute; left: 10px; bottom: 10px;
    display: none; background: rgba(10,13,18,0.7); color: var(--muted);
    border: 1px solid var(--border); border-radius: 5px; padding: 3px 8px;
    font-size: 10.5px; letter-spacing: 0.4px; }
  .hafs-player { display: flex; align-items: center; gap: 12px;
    padding: 12px 2px; flex-wrap: wrap; }
  .hafs-btn { background: var(--bg); color: #e8eef5;
    border: 1px solid var(--border); border-radius: 6px; padding: 7px 12px;
    font: inherit; font-size: 12.5px; font-weight: 600; cursor: pointer;
    white-space: nowrap; }
  .hafs-btn:hover { border-color: var(--cat-accent); color: var(--cat-accent); }
  .hafs-play { min-width: 86px; }
  .hafs-readout { display: flex; flex-direction: column; gap: 2px;
    min-width: 150px; flex: 1 1 auto;
    font-feature-settings: "tnum"; font-variant-numeric: tabular-nums; }
  .hafs-readout span:first-child { font-size: 14px; font-weight: 700; }
  .hafs-readout span:last-child { font-size: 11.5px; color: var(--muted); }
  .hafs-hours { display: grid;
    grid-template-columns: repeat(auto-fill, minmax(44px, 1fr));
    gap: 4px; padding: 2px 0 10px; }
  .hafs-hr { background: var(--bg); color: #e8eef5;
    border: 1px solid var(--border); border-radius: 5px; padding: 4px 0;
    font: inherit; font-size: 11.5px; font-weight: 600; text-align: center;
    font-variant-numeric: tabular-nums; cursor: pointer;
    transition: background 0.2s, color 0.2s, opacity 0.2s; }
  .hafs-hr.lit:hover { border-color: var(--cat-accent);
    color: var(--cat-accent); }
  .hafs-hr.pending { color: var(--muted); background: transparent;
    opacity: 0.55; cursor: default; }
  .hafs-hr.current { background: var(--cat-accent); color: var(--cat-ink);
    border-color: var(--cat-accent); font-weight: 700; }
  .hafs-caption { color: var(--muted); font-size: 12px; line-height: 1.5;
    margin: 8px 0 0; }
  .hafs-footer { display: flex; align-items: center; gap: 10px;
    flex-wrap: wrap; color: var(--muted); font-size: 11.5px;
    padding-top: 6px; }
  .hafs-pill, .hafs-badge { border: 1px solid var(--border);
    border-radius: 99px; padding: 2px 10px; font-size: 11px; }
  .hafs-badge .hafs-badge-btn { background: none; border: none;
    color: var(--cat-accent); font: inherit; cursor: pointer;
    text-decoration: underline; padding: 0 0 0 4px; }
  .hafs-preannounce { color: var(--muted); }
  #sat-scrub { flex: 2 1 160px; accent-color: var(--cat-accent);
    min-width: 120px; }
  @media (max-width: 700px) {
    .hafs-hours { grid-template-columns: repeat(auto-fill, minmax(38px, 1fr)); }
    .hafs-stage { min-height: 200px; }
  }

  /* ---- Stage 4: THE CONE reveal + advisory panels (§8.1/8.2/8.5).
     All reveal motion is transform/opacity/clip only; the icon spin is
     the ONE permitted continuous loop; reduced-motion = final frame. */
  .adv-cone-stage { background: #0a1019; border-radius: 8px;
    overflow: hidden; }
  /* S4-AD2 #2/#3: the map reads as a SURFACE - ocean lifted one clear
     step above the panel chrome, hairline inset border, land one step
     above the ocean, graticule legible-but-subtle. */
  .ac-ocean-fill { fill: #101a2c; }
  .ac-frame { fill: none; stroke: #2c3a52; stroke-width: 1.5; }
  .ac-land { fill: #1b2536; stroke: #384964;
    stroke-width: 1.2; }
  .ac-graticule line { stroke: #223048; stroke-width: 1; }
  .ac-graticule text { fill: #4d5f7d; font-size: 13px;
    font-feature-settings: "tnum"; font-variant-numeric: tabular-nums; }
  .ac-ocean { fill: rgba(202,217,240,0.14); font-size: 17px;
    font-weight: 700; letter-spacing: 4.5px; }
  .ac-title .ac-eyebrow { fill: #8fa2bd; font-size: 11.5px;
    font-weight: 700; letter-spacing: 1.6px; }
  .ac-title .ac-head { fill: #ffffff; font-size: 21px;
    font-weight: 800; letter-spacing: 0.6px; }
  .ac-title .ac-sub { fill: #b9c6da; font-size: 13px;
    font-weight: 700; letter-spacing: 1.1px; }
  #advcone, #intensity { display: block; width: 100%; height: auto; }
  .ac-zoom { animation: ac-pushin calc(var(--motion-med) * 0.85)
    ease-out 1 both; }
  @keyframes ac-pushin { from { transform: scale(0.94); }
                         to { transform: scale(1); } }
  .ac-icon { transform-box: fill-box; transform-origin: center;
    animation: ac-pop 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) 1 both; }
  @keyframes ac-pop { from { transform: scale(0); opacity: 0; }
                      to { transform: scale(1); opacity: 1; } }
  .ac-spin { transform-box: fill-box; transform-origin: center;
    animation: lab-spin 3.2s linear infinite; }
  .adv-method { margin: 10px 0 0; color: var(--muted); font-size: 12.5px; }
  .adv-method summary { cursor: pointer; color: #9fc6f5;
    font-weight: 600; font-size: 12px; letter-spacing: 0.4px; }
  .adv-method div { margin-top: 8px; line-height: 1.55;
    border-left: 2px solid var(--border); padding-left: 12px; }
  .advtext { background: #0a1019; border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 16px; margin: 12px 0 0;
    font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas,
      monospace; color: #dfe6ee; white-space: pre-wrap;
    max-height: 480px; overflow: auto; }
  @media (prefers-reduced-motion: reduce) {
    .ac-zoom, .ac-icon { animation: none !important; }
    .ac-spin { animation: none !important; }
  }

  .stub { color: var(--muted); font-size: 14px; padding: 30px 0;
    text-align: center; }

  /* ---- loader framework (variant prototypes a-d + default wipe) ---- */
  .loader { position: fixed; inset: 0; z-index: 50; display: flex;
    align-items: center; justify-content: center; pointer-events: none;
    background: var(--navy-deep); }
  .loader .word { font-size: 30px; font-weight: 800; color: #ffffff;
    letter-spacing: 0.5px; text-shadow: var(--ink-scrim); z-index: 2; }
  .loader .word .lite { font-weight: 500; opacity: 0.85; }
  .loader.done { animation: lab-launch var(--motion-med) ease-in-out 1 forwards; }
  @keyframes lab-launch { to { transform: scaleY(0); } }
  .loader { transform-origin: top; }
  /* a) intensifying ramp: JS steps --cat vars; bg follows the ramp */
  .loader[data-variant="a"] { background: var(--cat-ramp);
    transition: background 0.65s ease; }
  /* b) eye opens - THE CHOSEN LOADER (AD R3). Iris reveal approved
     as-is. The eye wears the storm's CATEGORY color (--cat-accent,
     canonical TAT scale) with a matching glow - Cat 2 and Cat 5 look
     different. Oversized viewBox (±44 vs ink reach ~41.4) so the
     spinning swirl tails NEVER clip; box 155px keeps the ink the
     visual size the 120px/68-box prototype had. Spin duration matches
     the header glyph (coherent). Wordmark = loader C's letter-build +
     shine, shared markup + shared rules below. */
  .loader[data-variant="b"] .eye { width: 155px; height: 155px;
    margin-bottom: 18px;
    filter: drop-shadow(0 0 14px var(--cat-accent)); }
  .loader[data-variant="b"] .eye path { fill: var(--cat-accent); }
  .loader[data-variant="b"] .eye .spin { animation: lab-spin 3.2s linear infinite;
    transform-origin: 0 0; }
  .loader[data-variant="b"] { flex-direction: column; }
  .loader[data-variant="b"].done { animation: none;
    clip-path: circle(150% at 50% 46%);
    animation: lab-iris var(--motion-med) ease-in-out 1 forwards; }
  @keyframes lab-iris { from { clip-path: circle(150% at 50% 46%); }
                        to { clip-path: circle(0% at 50% 46%); } }
  /* b+c) wordmark build + shine (ONE canon - b reuses c's treatment) */
  .loader[data-variant="b"] .word span.ch,
  .loader[data-variant="c"] .word span.ch { display: inline-block;
    opacity: 0; transform: translateY(12px);
    animation: lab-ch 0.55s ease-out forwards;
    animation-delay: calc(var(--i) * 0.09s); }
  @keyframes lab-ch { to { opacity: 1; transform: translateY(0); } }
  .loader[data-variant="b"] .word,
  .loader[data-variant="c"] .word { position: relative; overflow: hidden; }
  .loader[data-variant="b"] .word::after,
  .loader[data-variant="c"] .word::after { content: ""; position: absolute;
    top: 0; bottom: 0; width: 40%; left: -50%; transform: skewX(-18deg);
    background: linear-gradient(90deg, transparent,
      rgba(255,255,255,0.5), transparent);
    animation: lab-shine 1.1s ease-out 1; animation-delay: 1.3s; }
  /* d) broadcast sweep: luminous scanline; text fades up in its wake */
  .loader[data-variant="d"] .scan { position: absolute; top: 0; bottom: 0;
    width: 3px; left: 0; background: #ffffff;
    box-shadow: 0 0 18px 6px rgba(140,200,255,0.75);
    animation: lab-scan 1.8s cubic-bezier(0.6,0,0.35,1) 1 forwards; }
  @keyframes lab-scan { from { left: -2%; } to { left: 102%; } }
  .loader[data-variant="d"] .word { opacity: 0;
    animation: lab-fadeup 1s ease-out 0.9s 1 forwards; }
  @keyframes lab-fadeup { from { opacity: 0; transform: translateY(8px); }
                          to { opacity: 1; transform: translateY(0); } }
  .loader[data-variant="d"].done { animation: lab-wipeleft
    var(--motion-med) ease-in-out 1 forwards; }
  @keyframes lab-wipeleft { to { transform: translateX(100%); } }

  /* ---- mobile rotation: banner -> top bar; the card body (heroes +
         vitals) -> ONE slim card above the map; nav -> bottom tabs
         (same DOM; .bug flattens so the banner sticks exactly as
         AD-1 approved) ---- */
  @media (max-width: 640px) {
    .lab { flex-direction: column; }
    .side { width: 100%; flex: 0 0 auto; border-right: 0; }
    .bug { display: contents; }
    .banner { position: sticky; top: 0; z-index: 20; display: flex;
      align-items: center; gap: 10px; padding: 9px 14px; }
    .banner .eyebrow, .banner .storm-type { display: none; }
    .banner .storm-name { font-size: 17px; margin: 0; flex: 1 1 auto; }
    .banner .glyph { position: static; width: 39px; height: 39px;
      order: -1; flex: 0 0 auto; }
    .banner > .b-inner { display: flex; align-items: center; gap: 10px;
      flex: 1 1 auto; padding-right: 0; }
    .bug-body { margin: 10px 12px; background: var(--panel);
      border: 1px solid var(--border); border-radius: 12px;
      overflow: hidden; position: relative; }
    .bug-body::before { content: ""; position: absolute; left: 0;
      right: 0; top: 0; height: 3px; background: var(--cat-ramp); }
    .heroes { padding: 13px 14px 9px; }
    .hero .hero-val { font-size: 26px; }
    .vitals { padding: 1px 14px 6px; }
    .vrow { padding: 4px 0; }
    .vrow .val { font-size: 13.5px; }
    .nav-secs { position: fixed; bottom: 0; left: 0; right: 0; z-index: 30;
      flex-direction: row; background: var(--navy-deep);
      border-top: 1px solid var(--border); padding: 0; }
    .sec-btn { flex: 1 1 25%; justify-content: center; padding: 12px 4px;
      min-height: 52px; font-size: 10.5px; border-left: 0;
      border-top: 3px solid transparent; }
    .sec-btn.active { border-left: 0; border-top-color: var(--cat-accent); }
    .back-map { border-top: 0; padding: 8px 12px; min-height: 0; }
    .stage { padding: 4px 14px 86px; }
  }

  @media (prefers-reduced-motion: reduce) {
    .loader, .loader::after, .loader .scan, .loader .word,
    .loader .word span.ch, .banner.shine::after, .banner.xfade .old-ramp,
    .sec.active .wipe, .draw path.series, .draw .fill {
      animation-duration: 0.001s !important;
      animation-delay: 0s !important; }
    .banner .glyph .spin, .loader .eye .spin { animation: none !important; }
    .odo .col .strip { transition-duration: 0.001s !important; }
  }
</style>
</head>
<body>
<div class="loader" id="loader" data-variant="__LOADER__"></div>
<div class="ended-strip">THIS STORM HAS ENDED · final data below · CycloLab archive view</div>
<div class="lab">
  <aside class="side">
    <div class="bug">
    <div class="banner" id="banner">
      <div class="old-ramp"></div>
      <svg class="glyph" viewBox="-44 -44 88 88" aria-hidden="true">
        <g class="spin"><path d="__HPATH__" fill="#ffffff"
          stroke="rgba(0,0,0,0.30)" stroke-width="1"/></g>
        <!-- canonical icon label (tracks-map / storm-card canon:
             generate_tracks_plot.py sshs_label + spinnerSvg) - D / S /
             1-5, stationary while only the path spins. Weight 800, not
             the canon's 900: Metropolis ships no Black face, so 900
             would silently clamp to 800 anyway - declare what renders.
             Fill/stroke live in CSS (.banner .glyph text): the label
             wears the CATEGORY COLOR per AD R3. -->
        <text id="glyph-cat" y="0" text-anchor="middle"
          dominant-baseline="central" font-size="22" font-weight="800"
          stroke-linejoin="round">__CAT_LABEL__</text>
      </svg>
      <div class="b-inner">
        <div class="eyebrow">TRIPLE-A-TROPICS · <span class="brand">CycloLab</span></div>
        <div class="storm-type" id="storm-type">__TYPE_WORD__</div>
        <div class="storm-name" id="storm-name">__NAME__</div>
        <span class="chip" id="chip"__CHIP_STYLE__>__CHIP__</span>
      </div>
    </div>
    <div class="bug-body">
      <div class="heroes">
        <div class="hero">
          <span class="hero-val"><span class="odo" id="odo-vmax" aria-label="__VMAX_A11Y__">__VMAX_ODO__</span><span class="unit">kt</span></span>
          <span class="hero-cap">Max wind</span>
        </div>
        <div class="hero-div"></div>
        <div class="hero">
          <span class="hero-val"><span class="odo" id="odo-cat" aria-label="__CAT_LABEL__">__CAT_ODO__</span></span>
          <span class="hero-cap">Category</span>
        </div>
      </div>
      <div class="vitals" id="vitals"></div>
    </div>
    </div>

    <nav class="nav-secs" id="secnav">
      <button class="sec-btn active" data-sec="overview">Overview</button>
      <button class="sec-btn" data-sec="satellite">Satellite</button>
      <button class="sec-btn" data-sec="models">Models</button>
      <button class="sec-btn" data-sec="advisories">Advisories</button>
    </nav>
    <a class="back-map" href="/global_tracks.html">← Back to map</a>
  </aside>

  <main class="stage">
    <section class="sec active" id="sec-overview">
      <div class="wipe">
        <div class="card"><h3>Track &amp; forecast cone</h3>
          <svg id="trackmap" viewBox="0 0 1000 620"
               preserveAspectRatio="xMidYMid meet"></svg>
          <div class="note" id="cone-note" hidden></div></div>
        <div class="card"><h3>Wind &amp; pressure</h3>
          <svg id="chart" viewBox="0 0 1000 320"
               preserveAspectRatio="xMidYMid meet"></svg></div>
      </div>
    </section>
    <section class="sec" id="sec-satellite"><div class="wipe">
      <h2 class="sec-title">Satellite</h2>
      <div class="card" id="sat-card" tabindex="0">
        <div class="hafs-controls">
          <div class="hafs-group"><label>Band</label>
            <div id="sat-bands" class="hafs-seg-group" role="group"
                 aria-label="Satellite band"></div></div>
        </div>
        <div class="hafs-stage" id="sat-stage">
          <img id="sat-img" alt="Storm floater satellite frame">
          <div id="sat-status" class="hafs-statusbox">
            <div class="hafs-spinner"></div><span>Loading…</span></div>
        </div>
        <div class="hafs-player">
          <button id="sat-step-back" class="hafs-btn" type="button"
                  title="Previous frame (&#8592;)">&#9664;</button>
          <button id="sat-play" class="hafs-btn hafs-play" type="button"
                  title="Play / pause (space)">&#9654; Play</button>
          <button id="sat-step-fwd" class="hafs-btn" type="button"
                  title="Next frame (&#8594;)">&#9654;</button>
          <input id="sat-scrub" type="range" min="0" max="0" value="0"
                 step="1" aria-label="Frame time">
          <div class="hafs-readout"><span id="sat-time">&#8212;</span>
            <span id="sat-band-label"></span></div>
        </div>
        <div id="sat-empty" class="stub" style="display:none">No floater
          imagery for this storm right now.</div>
        <p class="hafs-caption">GOES floater imagery centered on the storm,
          newest frame first. Frames land every few minutes while the
          floater is active.</p>
      </div>
    </div></section>
    <section class="sec" id="sec-models"><div class="wipe">
      <h2 class="sec-title">Models</h2>
      <div class="card" id="cl-hafs-root" tabindex="0">
        <div id="cl-hafs-controls" class="hafs-controls">
          <div id="cl-hafs-cycle-group" class="hafs-group" style="display:none">
            <label>Cycle</label>
            <div id="cl-hafs-cycles" class="hafs-seg-group"></div></div>
          <div class="hafs-group" style="display:none">
            <label for="cl-hafs-storm">Storm</label>
            <select id="cl-hafs-storm"></select></div>
          <div class="hafs-group"><label>Model</label>
            <div id="cl-hafs-models" class="hafs-seg-group"></div></div>
          <div class="hafs-group"><label>Domain</label>
            <div id="cl-hafs-domains" class="hafs-seg-group"></div></div>
          <div class="hafs-group"><label>Product</label>
            <div id="cl-hafs-products" class="hafs-seg-group"></div></div>
        </div>
        <div id="cl-hafs-stage" class="hafs-stage">
          <img id="cl-hafs-img" alt="HAFS forecast frame for this storm">
          <div id="cl-hafs-status" class="hafs-statusbox">
            <div class="hafs-spinner"></div><span>Loading&#8230;</span></div>
          <div id="cl-hafs-buffer"></div>
        </div>
        <div id="cl-hafs-player" class="hafs-player">
          <button id="cl-hafs-step-back" class="hafs-btn" type="button"
                  title="Previous hour (&#8592;)">&#9664;</button>
          <button id="cl-hafs-play" class="hafs-btn hafs-play" type="button"
                  title="Play / pause (space)">&#9654; Play</button>
          <button id="cl-hafs-step-fwd" class="hafs-btn" type="button"
                  title="Next hour (&#8594;)">&#9654;</button>
          <div class="hafs-readout">
            <span id="cl-hafs-fhour">F000</span>
            <span id="cl-hafs-valid"></span></div>
          <div class="hafs-group"><label for="cl-hafs-speed">Speed</label>
            <select id="cl-hafs-speed"></select></div>
        </div>
        <div id="cl-hafs-hours" class="hafs-hours" role="group"
             aria-label="Forecast hour"></div>
        <div id="cl-hafs-empty" class="stub" style="display:none">No model
          guidance for this storm in the current cycles.</div>
        <p id="cl-hafs-caption" class="hafs-caption">HAFS guidance scoped to
          this storm - the storm-nest domain follows the cyclone, so playback
          is roughly storm-centered. Same renders as the site-wide
          <a href="/models/">/models/</a> viewer.</p>
        <div class="hafs-footer">
          <span id="cl-hafs-meta"></span>
          <span id="cl-hafs-pill" class="hafs-pill" style="display:none"></span>
          <span id="cl-hafs-badge" class="hafs-badge" style="display:none"></span>
        </div>
      </div>
    </div></section>
    <section class="sec" id="sec-advisories"><div class="wipe">
      <h2 class="sec-title">Advisories</h2>
      <div class="card">
        <h3>Forecast cone</h3>
        <div class="adv-cone-stage">
          <svg id="advcone" viewBox="0 0 1000 620"
               preserveAspectRatio="xMidYMid meet" role="img"
               aria-label="Forecast track and uncertainty cone"></svg>
        </div>
        <p class="hafs-caption" id="advcone-note"></p>
        <details class="adv-method" id="advcone-method">
          <summary>How is this drawn?</summary>
          <div id="advcone-method-body"></div>
        </details>
        <div id="advcone-empty" class="stub" style="display:none">No
          advisory geometry yet for this storm.</div>
      </div>
      <div class="card">
        <h3>Intensity forecast</h3>
        <svg id="intensity" viewBox="0 0 1000 380"
             preserveAspectRatio="xMidYMid meet" role="img"
             aria-label="Forecast intensity with published error range"></svg>
        <p class="hafs-caption" id="intensity-note" hidden>Derived
          intensity range &#8212; not an official forecast product.</p>
        <details class="adv-method" id="intensity-method" hidden>
          <summary>How is this derived?</summary>
          <div id="intensity-method-body"></div>
        </details>
        <div id="intensity-missing" class="stub" style="display:none"></div>
      </div>
      <div class="card">
        <h3>Advisory text</h3>
        <div class="hafs-seg-group" id="advtext-tabs" role="group"
             aria-label="Advisory product">
          <button type="button" class="hafs-seg active"
                  data-prod="tcp">Public Advisory</button>
          <button type="button" class="hafs-seg"
                  data-prod="tcd">Discussion</button>
        </div>
        <pre id="advtext" class="advtext">(advisory text loads with the next data poll)</pre>
      </div>
    </div></section>
  </main>
</div>

<script>
(function () {
  "use strict";
  var SID = "__SID__";
  var FEED_URL = "__FEED_URL__";
  var ADV_URL = "__ADV_URL__";
  var ENDED = __ENDED__;
  var BASIN = "__BASIN__";
  var HAFS_ID = "__HAFS_ID__";        // storm_ids join: 01e
  var FLOATER_ID = "__ATCF_LONG__";   // storm_ids join: ep012026
  var CDN = "https://cdn.triple-a-tropics.com";
  // Per-basin published intensity-error entry (null = the honesty-guard
  // case: a labeled "no published statistics" panel, never a borrowed
  // or invented envelope).
  var INTENSITY_ERR = __INTENSITY_ERR__;
  // Storm-window basemap (S4-AD1 #2): vendored Natural Earth land,
  // clipped + antimeridian-normalized at bake time. No runtime fetch.
  var BASEMAP = __BASEMAP__;
  // Python-derived category ramp tokens (THE approved gloss recipe -
  // same edge/mid/accent stops as the banner ramps and LIVE STATUS
  // chrome; one canon, baked not re-derived).
  var CAT_TOKENS = __CAT_TOKENS__;
  var SITE_BASE = "https://triple-a-tropics.com";
  var POLL_MS = 60000;
  var SSHS = __SSHS_JSON__;
  var CHIP_LABEL = { TD: "Tropical Depression", TS: "Tropical Storm",
    C1: "Category 1", C2: "Category 2", C3: "Category 3",
    C4: "Category 4", C5: "Category 5" };
  // storm-type word: advisory dev_label first (the tau-0 second-pass
  // path), category+basin fallback otherwise.
  var DEV_WORD = { TD: "Tropical Depression", TS: "Tropical Storm",
    HU: "Hurricane", MH: "Major Hurricane", STS: "Subtropical Storm",
    PTC: "Post-Tropical Cyclone", L: "Low" };
  function catWord(cat) {
    if (cat === "TD") return "Tropical Depression";
    if (cat === "TS") return "Tropical Storm";
    return BASIN === "WP" ? "Typhoon" : "Hurricane";
  }
  // canonical icon letter/number - the SAME treatment the tracks maps +
  // storm-card placard spinners use (generate_tracks_plot.py sshs_label
  // / sshsLabel; mirrored in python _sshs_label). One canon, no new style.
  function sshsLabel(cls) {
    if (cls === "TD") return "D";
    if (cls === "TS") return "S";
    return (cls || "").replace("C", "") || "D";
  }
  var reduced = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- loader -------------------------------------------------------------
  var loader = document.getElementById("loader");
  (function initLoader() {
    var v = loader.getAttribute("data-variant");
    var word = '<div class="word">Loading <span>CycloLab</span></div>';
    // the C-progression wordmark (letter build + shine) - ONE builder,
    // shared by variants b and c per AD R3. Brand casing rule applies:
    // always literally "CycloLab".
    function wordC() {
      var chars = "CycloLab".split("").map(function (ch, i) {
        return '<span class="ch" style="--i:' + i + '">' + ch + "</span>";
      }).join("");
      return '<div class="word"><span class="lite">Loading&nbsp;</span>' +
        chars + "</div>";
    }
    if (v === "b") {
      loader.innerHTML =
        '<svg class="eye" viewBox="-44 -44 88 88"><g class="spin">' +
        '<path d="__HPATH__"/></g></svg>' + wordC();
    } else if (v === "c") {
      loader.innerHTML = wordC();
    } else if (v === "d") {
      loader.innerHTML = '<div class="scan"></div>' + word;
    } else {
      loader.innerHTML = word;
    }
    if (v === "a") {           // intensifying ramp up to the real category
      var seq = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"];
      var target = document.documentElement.getAttribute("data-cat");
      var stop = seq.indexOf(target); if (stop < 0) stop = 0;
      var i = 0;
      var iv = setInterval(function () {
        document.documentElement.setAttribute("data-cat", seq[i]);
        if (i >= stop) { clearInterval(iv);
          document.documentElement.setAttribute("data-cat", target); }
        i++;
      }, reduced ? 1 : 480);
    }
    var hold = reduced ? 50 : (v ? 2700 : 900);
    setTimeout(function () {
      loader.classList.add("done");
      setTimeout(function () { if (loader.parentNode) loader.remove(); },
                 reduced ? 60 : 1400);
    }, hold);
  })();

  // ---- section nav (lazy init on first open) ------------------------------
  var inited = {};
  function openSec(name) {
    if (!document.getElementById("sec-" + name)) return;  // unknown: no-op
    document.querySelectorAll(".sec").forEach(function (s) {
      s.classList.toggle("active", s.id === "sec-" + name);
    });
    document.querySelectorAll(".sec-btn").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-sec") === name);
    });
    if (!inited[name]) {
      inited[name] = true;
      // Stage 3: nothing is fetched until the tab opens (lazy mounts).
      if (name === "models") initModels();
      else if (name === "satellite") initSatellite();
    }
    // THE CONE reveal plays once per tab OPEN (not once per session):
    // rebuilding the SVG re-arms every CSS animation naturally.
    if (name === "advisories") renderAdvTab();
    // pause-on-hide: a hidden tab must not keep its playback loop alive
    // (the satellite interval kept swapping img.src at 5 fps display:none,
    // and both viewers' timers could run concurrently).
    if (name !== "advisories" && acRaf) {
      cancelAnimationFrame(acRaf); acRaf = null;
    }
    if (name !== "satellite") satPause();
    if (name !== "models" && hafsViewer && hafsViewer._pause) {
      hafsViewer._pause();
    }
    var w = document.querySelector("#sec-" + name + " .wipe");
    if (w) { w.style.animation = "none"; void w.offsetWidth; w.style.animation = ""; }
  }
  document.getElementById("secnav").addEventListener("click", function (e) {
    var b = e.target.closest(".sec-btn");
    if (b) openSec(b.getAttribute("data-sec"));
  });

  // ---- odometer ------------------------------------------------------------
  // REST IS PLAIN TEXT: a settled cell is a plain span on the row's
  // natural baseline (pixel-identical to static type, overshoot
  // intact). A .col (anchor + SPACED strip on a whole-device-pixel
  // grid) exists ONLY while a digit is actually rolling; on settle
  // (timer = transition duration + slack) the cell snaps back to
  // plain text. Mid-roll retargets (the 1s countdown) reuse the live
  // strip - CSS transitions retarget from the interpolated position.
  function odoCellChar(cell) {
    var t = cell.getAttribute("data-ch");
    return t != null ? t : cell.textContent;
  }
  function odoSettle(cell) {
    if (cell._settleT) { clearTimeout(cell._settleT); cell._settleT = null; }
    var ch = odoCellChar(cell);
    cell.className = "digit" + (/^[0-9]$/.test(ch) ? "" : " ch");
    cell.removeAttribute("data-ch");
    cell.textContent = ch;
  }
  function odoRoll(el, cell, from, to) {
    var strip, eh;
    if (!cell.classList.contains("col")) {
      // build the roll-time col: integer geometry - entry pitch is
      // 1.7em SNAPPED TO WHOLE DEVICE PIXELS, every resting offset is
      // -digit x (integer device px) so no per-value fractions.
      var fs = parseFloat(getComputedStyle(el).fontSize) || 16;
      var dpr = window.devicePixelRatio || 1;
      eh = Math.max(1, Math.round(1.7 * fs * dpr)) / dpr;
      cell.className = "digit col";
      cell.textContent = "";
      var anchor = document.createElement("span");
      anchor.className = "anchor"; anchor.textContent = to;
      cell.appendChild(anchor);
      strip = document.createElement("span");
      strip.className = "strip";
      strip.style.setProperty("--odo-eh", eh + "px");
      strip.setAttribute("data-eh", String(eh));
      for (var d = 0; d <= 9; d++) {
        var s = document.createElement("span");
        s.className = "digit"; s.textContent = String(d);
        strip.appendChild(s);
      }
      cell.appendChild(strip);
      strip.style.transition = "none";
      strip.style.transform = "translateY(" + (-from * eh) + "px)";
      void strip.offsetWidth;             // commit the start frame
      strip.style.transition = "";
    } else {
      strip = cell.querySelector(".strip");
      eh = parseFloat(strip.getAttribute("data-eh")) || 0;
      cell.querySelector(".anchor").textContent = to;
    }
    cell.setAttribute("data-ch", to);
    strip.style.transform = "translateY(" + (-Number(to) * eh) + "px)";
    if (cell._settleT) clearTimeout(cell._settleT);
    var dur = parseFloat(getComputedStyle(strip).transitionDuration) || 0;
    cell._settleT = setTimeout(function () { odoSettle(cell); },
                               dur * 1000 + 150);
  }
  function odoSet(el, text) {
    var want = String(text);
    if (el.getAttribute("data-odo") === want) return;
    el.setAttribute("data-odo", want);
    // AT reads the value; the cells are aria-hidden presentation.
    el.setAttribute("aria-label", want);
    while (el.children.length > want.length) {
      var last = el.lastChild;
      if (last._settleT) clearTimeout(last._settleT);
      el.removeChild(last);
    }
    for (var i = 0; i < want.length; i++) {
      var ch = want[i];
      var cell = el.children[i];
      if (!cell) {
        cell = document.createElement("span");
        cell.className = "digit" + (/[0-9]/.test(ch) ? "" : " ch");
        cell.setAttribute("aria-hidden", "true");
        cell.textContent = ch;
        el.appendChild(cell);
        continue;
      }
      var cur = odoCellChar(cell);
      if (cur === ch) {
        // settled-and-right or already rolling to the right target.
        if (!cell.classList.contains("col")) {
          cell.classList.toggle("ch", !/[0-9]/.test(ch));
        }
        continue;
      }
      if (/^[0-9]$/.test(ch) && /^[0-9]$/.test(cur) && !reduced) {
        odoRoll(el, cell, Number(cur), ch);
      } else {
        // letters, mixed transitions, reduced motion: straight to rest.
        if (cell._settleT) { clearTimeout(cell._settleT); cell._settleT = null; }
        cell.className = "digit" + (/[0-9]/.test(ch) ? "" : " ch");
        cell.removeAttribute("data-ch");
        cell.textContent = ch;
      }
    }
  }

  // ---- vitals (inline rows in the bug card body; Max Wind + Category
  //      live above them as the HERO numbers, baked into the template) ----
  var VITALS = [
    { id: "mslp", lbl: "Min pressure", unit: "mb" },
    { id: "ace", lbl: "Storm ACE", unit: "" },
    { id: "pos", lbl: "Position", unit: "" },
    { id: "move", lbl: "Movement", unit: "" },
    { id: "fix", lbl: "Last fix", unit: "UTC" },
    { id: "next", lbl: "Next advisory", unit: "" },
  ];
  function buildVitals() {
    document.getElementById("vitals").innerHTML = VITALS.map(function (s) {
      return '<div class="vrow" id="vrow-' + s.id + '">' +
        '<span class="lbl">' + s.lbl + '</span>' +
        '<span class="val"><span class="odo" id="odo-' + s.id + '"></span>' +
        (s.unit ? '<span class="unit">' + s.unit + '</span>' : "") +
        "</span></div>";
    }).join("");
  }
  function fmtPos(lat, lon) {
    if (lat == null || lon == null) return "—";
    return Math.abs(lat).toFixed(1) + (lat >= 0 ? "N" : "S") + " " +
           Math.abs(lon).toFixed(1) + (lon >= 0 ? "E" : "W");
  }
  function movement(pts) {
    if (!pts || pts.length < 2) return "—";
    for (var i = pts.length - 2; i >= 0; i--) {
      var a = pts[i], b = pts[pts.length - 1];
      var dtH = (new Date(b.t) - new Date(a.t)) / 3600000;
      if (dtH < 1) continue;
      var latm = (b.lat - a.lat) * 60;
      var lonm = (b.lon - a.lon) * 60 *
        Math.cos((a.lat + b.lat) / 2 * Math.PI / 180);
      var dist = Math.sqrt(latm * latm + lonm * lonm);
      if (dist < 0.5) return "Stationary";
      var dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                  "S","SSW","SW","WSW","W","WNW","NW","NNW"];
      var brg = (Math.atan2(lonm, latm) * 180 / Math.PI + 360) % 360;
      return dirs[Math.round(brg / 22.5) % 16] + " " +
        Math.round(dist / dtH) + " kt";
    }
    return "—";
  }

  // ---- advisory countdown (source of truth = the advisory's OWN stated
  //      next-advisory time, parsed by the poller; never wall-clock) -----
  var nextAdvUtc = null;
  function tickCountdown() {
    var el = document.getElementById("odo-next");
    var row = document.getElementById("vrow-next");
    if (!el) return;
    if (!nextAdvUtc) { odoSet(el, "—"); return; }
    var ms = new Date(nextAdvUtc) - new Date();
    if (ms <= 0) {
      row.classList.add("due");
      odoSet(el, "DUE · updating");
      return;
    }
    row.classList.remove("due");
    var h = Math.floor(ms / 3600000);
    var m = Math.floor(ms % 3600000 / 60000);
    var s = Math.floor(ms % 60000 / 1000);
    odoSet(el, h > 0 ? (h + "h " + (m < 10 ? "0" : "") + m + "m")
                     : (m + "m " + (s < 10 ? "0" : "") + s + "s"));
  }
  setInterval(tickCountdown, 1000);

  // ---- category + type word -------------------------------------------------
  var curCat = document.documentElement.getAttribute("data-cat");
  var advTypeWord = null;   // advisory-sourced word wins when present
  function setCategory(cat) {
    // unknown/garbage categories clamp to TD - the same validation
    // render_page applies to the static render, so the hydrated
    // ramp/chip/glyph/hero can never desync from the baked page.
    if (cat && !CHIP_LABEL[cat]) cat = "TD";
    if (!cat || cat === curCat) return;
    var banner = document.getElementById("banner");
    var oldRamp = getComputedStyle(document.documentElement)
      .getPropertyValue("--cat-ramp");
    document.documentElement.setAttribute("data-cat", cat);
    // chip ONLY when it adds information (S4-AD1 #10): CATEGORY 1-5
    // for hurricanes; at TD/TS it duplicated the type word.
    var chipEl = document.getElementById("chip");
    if (cat === "TD" || cat === "TS") {
      chipEl.style.display = "none";
    } else {
      chipEl.style.display = "";
      chipEl.textContent = CHIP_LABEL[cat] || cat;
    }
    // the canon label rides the corner glyph + the Category hero.
    document.getElementById("glyph-cat").textContent = sshsLabel(cat);
    odoSet(document.getElementById("odo-cat"), sshsLabel(cat));
    if (!advTypeWord) {
      document.getElementById("storm-type").textContent =
        catWord(cat).toUpperCase();
    }
    curCat = cat;
    if (reduced) return;
    banner.querySelector(".old-ramp").style.setProperty("--old-ramp", oldRamp);
    banner.classList.remove("xfade", "shine"); void banner.offsetWidth;
    banner.classList.add("xfade", "shine");
  }

  // ---- map + cone (the home view showstopper) -------------------------------
  var coneRing = null, coneAdv = null, conePts = null;
  function renderTrack(storm) {
    var svg = document.getElementById("trackmap");
    var pts = storm.points || [];
    if (!pts.length && !coneRing) { svg.innerHTML = ""; return; }
    var lats = [], lons = [];
    pts.forEach(function (p) { lats.push(p.lat); lons.push(p.lon); });
    (coneRing || []).forEach(function (c) { lons.push(c[0]); lats.push(c[1]); });
    var pad = 1.8;
    var la0 = Math.min.apply(null, lats) - pad, la1 = Math.max.apply(null, lats) + pad;
    var lo0 = Math.min.apply(null, lons) - pad, lo1 = Math.max.apply(null, lons) + pad;
    var W = 1000, H = 620;
    function X(lon) { return (lon - lo0) / (lo1 - lo0) * W; }
    function Y(lat) { return H - (lat - la0) / (la1 - la0) * H; }
    var parts = ['<rect width="' + W + '" height="' + H + '" fill="#0a1019"/>'];
    // CONE: brand blue/white, never category-colored (uncertainty area).
    if (coneRing && coneRing.length > 3) {
      var dC = coneRing.map(function (c, i) {
        return (i ? "L" : "M") + X(c[0]).toFixed(1) + "," + Y(c[1]).toFixed(1);
      }).join(" ") + " Z";
      parts.push('<defs><filter id="cone-soft" x="-20%" y="-20%" ' +
        'width="140%" height="140%"><feGaussianBlur stdDeviation="5"/>' +
        "</filter></defs>");
      parts.push('<path d="' + dC + '" fill="none" stroke="#8cc8ff" ' +
        'stroke-width="9" stroke-opacity="0.35" filter="url(#cone-soft)"/>');
      parts.push('<path d="' + dC + '" fill="rgba(255,255,255,0.12)" ' +
        'stroke="#0a1a2e" stroke-width="2"/>');
      if (conePts && conePts.length > 1) {
        var dF = conePts.map(function (p, i) {
          return (i ? "L" : "M") + X(p.lon).toFixed(1) + "," + Y(p.lat).toFixed(1);
        }).join(" ");
        parts.push('<path d="' + dF + '" fill="none" stroke="#ffffff" ' +
          'stroke-width="1.6" stroke-dasharray="2 5" stroke-opacity="0.7"/>');
      }
    }
    if (pts.length) {
      var d = pts.map(function (p, i) {
        return (i ? "L" : "M") + X(p.lon).toFixed(1) + "," + Y(p.lat).toFixed(1);
      }).join(" ");
      parts.push('<path d="' + d + '" fill="none" stroke="var(--cat-accent)" ' +
        'stroke-width="2.5" stroke-opacity="0.85" stroke-linejoin="round" ' +
        'stroke-linecap="round"/>');
      pts.forEach(function (p) {
        parts.push('<circle cx="' + X(p.lon).toFixed(1) + '" cy="' +
          Y(p.lat).toFixed(1) + '" r="4.5" fill="' +
          (SSHS[p.cls] || SSHS.TD) + '" stroke="#fff" stroke-width="1"/>');
      });
      var last = pts[pts.length - 1];
      parts.push('<circle cx="' + X(last.lon).toFixed(1) + '" cy="' +
        Y(last.lat).toFixed(1) + '" r="10" fill="none" ' +
        'stroke="var(--cat-accent)" stroke-width="3"/>');
    }
    svg.innerHTML = parts.join("");
  }

  var chartDrawn = false;
  function renderChart(storm) {
    var svg = document.getElementById("chart");
    var pts = (storm.points || []).filter(function (p) {
      return p.wind_kt != null; });
    if (pts.length < 2) { svg.innerHTML = ""; return; }
    var W = 1000, H = 320, padL = 56, padR = 56, padT = 16, padB = 28;
    var wMax = Math.max(140, Math.max.apply(null, pts.map(function (p) {
      return p.wind_kt; })) + 10);
    var prs = pts.map(function (p) { return p.pressure_mb; })
      .filter(function (v) { return v != null; });
    var p0 = Math.min.apply(null, prs.concat([1000])) - 6;
    var p1 = Math.max.apply(null, prs.concat([1014])) + 6;
    function Xi(i) { return padL + i / (pts.length - 1) * (W - padL - padR); }
    function Yw(w) { return H - padB - (w / wMax) * (H - padT - padB); }
    function Yp(p) { return H - padB - ((p - p0) / (p1 - p0)) * (H - padT - padB); }
    var parts = ['<rect width="' + W + '" height="' + H + '" fill="#0a1019"/>'];
    [[34, "TS"], [64, "C1"], [83, "C2"], [96, "C3"], [113, "C4"], [137, "C5"]]
      .forEach(function (b) {
        parts.push('<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' +
          Yw(b[0]).toFixed(1) + '" y2="' + Yw(b[0]).toFixed(1) +
          '" stroke="' + SSHS[b[1]] + '" stroke-opacity="0.25" ' +
          'stroke-dasharray="3 5"/>');
        parts.push('<text x="' + (W - padR + 6) + '" y="' +
          (Yw(b[0]) + 4).toFixed(1) + '" fill="' + SSHS[b[1]] +
          '" font-size="11" opacity="0.8">' + b[1] + "</text>");
      });
    var dWind = pts.map(function (p, i) {
      return (i ? "L" : "M") + Xi(i).toFixed(1) + "," + Yw(p.wind_kt).toFixed(1);
    }).join(" ");
    var area = dWind + " L" + Xi(pts.length - 1).toFixed(1) + "," + (H - padB) +
      " L" + padL + "," + (H - padB) + " Z";
    parts.push('<path class="fill" d="' + area +
      '" fill="var(--cat-accent)" fill-opacity="0.13"/>');
    parts.push('<path class="series" d="' + dWind +
      '" fill="none" stroke="var(--cat-accent)" stroke-width="3" ' +
      'stroke-linejoin="round" stroke-linecap="round"/>');
    var pp = pts.filter(function (p) { return p.pressure_mb != null; });
    if (pp.length >= 2) {
      var dPres = pp.map(function (p, i) {
        var gi = pts.indexOf(p);
        return (i ? "L" : "M") + Xi(gi).toFixed(1) + "," +
          Yp(p.pressure_mb).toFixed(1);
      }).join(" ");
      parts.push('<path d="' + dPres + '" fill="none" stroke="#8ea2bd" ' +
        'stroke-width="2" stroke-dasharray="5 4" stroke-opacity="0.9"/>');
    }
    parts.push('<text x="' + padL + '" y="13" fill="#8ea2bd" font-size="11">' +
      'wind kt (solid) · pressure mb (dashed)</text>');
    svg.innerHTML = parts.join("");
    if (!chartDrawn && !reduced) {
      var series = svg.querySelector("path.series");
      var len = series.getTotalLength ? series.getTotalLength() : 2000;
      series.style.setProperty("--len", len);
      svg.classList.add("draw");
    }
    chartDrawn = true;
  }

  // ---- hydration (poll + diff-merge: grow state, never reset the user) ----
  var lastFixKey = null;
  var lastStorm = null;
  function apply(storm) {
    lastStorm = storm;
    setCategory(storm.current_category || "TD");
    document.getElementById("storm-name").textContent =
      (storm.name || SID).toUpperCase();
    var pts = storm.points || [];
    var last = pts[pts.length - 1] || {};
    var fixKey = last.t || null;
    odoSet(document.getElementById("odo-vmax"),
           last.wind_kt != null ? String(Math.round(last.wind_kt)) : "—");
    odoSet(document.getElementById("odo-mslp"),
           last.pressure_mb != null ? String(Math.round(last.pressure_mb)) : "—");
    odoSet(document.getElementById("odo-ace"),
           storm.ace != null ? storm.ace.toFixed(2) : "0.00");
    odoSet(document.getElementById("odo-pos"), fmtPos(last.lat, last.lon));
    odoSet(document.getElementById("odo-move"), movement(pts));
    odoSet(document.getElementById("odo-fix"),
           fixKey ? fixKey.slice(5, 16).replace("T", " ") : "—");
    if (fixKey !== lastFixKey) {
      renderTrack(storm);
      renderChart(storm);
      lastFixKey = fixKey;
    }
  }

  var advFull = null;
  var acRaf = null;          // the cone reveal's rAF loop handle
  function applyAdvisory(adv) {
    if (!adv) return;
    var changed = adv.advisory !== coneAdv;
    advFull = adv;
    coneAdv = adv.advisory;
    coneRing = adv.cone || null;
    conePts = adv.points || null;
    nextAdvUtc = adv.next_advisory_utc || null;
    tickCountdown();
    if (adv.points && adv.points.length &&
        DEV_WORD[adv.points[0].dev_label]) {
      advTypeWord = DEV_WORD[adv.points[0].dev_label];
      document.getElementById("storm-type").textContent =
        advTypeWord.toUpperCase();
    }
    var note = document.getElementById("cone-note");
    if (note && coneRing) {
      note.hidden = false;
      note.textContent = (adv.method === "official-cone")
        ? "Cone: official NHC forecast cone, advisory " + coneAdv + "."
        : "Derived uncertainty envelope (advisory " + coneAdv +
          ") — not an official JTWC product. Method: " +
          (adv.method || "derived") + ".";
    }
    // isolated like renderAdvTab (final-gate #3): a bad cone payload
    // in one renderer never blocks the other surfaces
    if (changed && lastStorm) {
      try { renderTrack(lastStorm); } catch (e) {
        try { console.warn("[cyclolab] track render failed:", e); }
        catch (e2) {}
      }
    }
    if (changed && inited.advisories) renderAdvTab();
  }

  function fetchJson(url) {
    return fetch(url + (url.indexOf("?") >= 0 ? "&" : "?") + "t=" + Date.now(),
                 { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }
  function poll() {
    Promise.all([fetchJson(FEED_URL), fetchJson(ADV_URL)])
      .then(function (res) {
        var feed = res[0], adv = res[1];
        if (feed && feed.storms) {
          var storm = feed.storms.filter(function (s) {
            return s && s.sid === SID; })[0];
          if (storm) apply(storm);
        }
        applyAdvisory(adv);
      })
      .then(function () { if (!ENDED) setTimeout(poll, POLL_MS); });
  }

  buildVitals();
  var BAKED = __BAKED__;
  if (BAKED) apply(BAKED);
  if (!ENDED) poll();

  // ---- Stage 4: THE CONE reveal + intensity cone + advisory text ----------
  // (§8.1/8.2) The showpiece cone: same projection + brand-blue/white
  // styling as the Overview map, plus the reveal choreography - push-in
  // zoom, current-intensity placard pop, the cone drawing itself outward
  // from the present position (an expanding clip circle), forecast-point
  // icons popping in sequence as SLOW-SPINNING category-colored cyclone
  // glyphs with glossy tau+kt placards. Category color lives ONLY in the
  // icons/placards - the cone itself is uncertainty, never tinted.
  function catForKt(kt) {
    if (kt == null) return "TD";
    if (kt < 34) return "TD";
    if (kt < 64) return "TS";
    if (kt < 83) return "C1";
    if (kt < 96) return "C2";
    if (kt < 113) return "C3";
    if (kt < 137) return "C4";
    return "C5";
  }
  var WP_METHOD_COPY =
    "JTWC issues a forecast track for western Pacific storms but no " +
    "official \u201ccone of uncertainty.\u201d This envelope is drawn by " +
    "buffering each forecast point by JTWC\u2019s published average " +
    "track-forecast error at that lead time (ESCAP/WMO Typhoon Committee " +
    "verification report, 2015 season, Table 3) and sweeping the " +
    "boundary. It reflects the historical AVERAGE error of past " +
    "forecasts \u2014 it is not a probabilistic bound, and the storm can " +
    "travel outside it. 12 h and 36 h radii are interpolated between " +
    "published values. Method jtwc-wpac-mean-2015.";
  var NHC_METHOD_COPY =
    "This is the official National Hurricane Center forecast cone for " +
    "this advisory, drawn from NHC\u2019s published GIS product. The cone " +
    "contains the probable track of the storm center (roughly two-thirds " +
    "of historical official track errors) \u2014 it says nothing about " +
    "the storm\u2019s size or impacts, and the storm can travel outside " +
    "it. Forecast-point icons are colored by the forecast intensity.";
  // Dev labels that count as TROPICAL for icon coloring; everything
  // else (EX/PT/LO/DB/remnants) is the user-ordered WHITE emphasis -
  // "white = forecast non-tropical" (S4-AD1 #4).
  var TROPICAL_DEV = { TD: 1, TS: 1, HU: 1, TY: 1, ST: 1, SD: 1, SS: 1 };
  function pointCat(p) {
    if (p.dev_label && SSHS[p.dev_label]) return p.dev_label;
    return catForKt(p.intensity_kt);
  }
  function pointTropical(p) {
    if (!p.dev_label) return true;            // no label: color by kt
    return !!TROPICAL_DEV[p.dev_label];
  }
  function renderAdvCone() {
    var svg = document.getElementById("advcone");
    var note = document.getElementById("advcone-note");
    var empty = document.getElementById("advcone-empty");
    var method = document.getElementById("advcone-method");
    if (!advFull || !coneRing || coneRing.length < 4 || !conePts ||
        !conePts.length) {
      svg.innerHTML = ""; empty.style.display = "block";
      note.textContent = ""; method.hidden = true;
      return;
    }
    empty.style.display = "none"; method.hidden = false;
    var pts = conePts;

    // ---- shared lon frame: everything joins the BASEMAP window ------
    var frameLon = (BASEMAP.window[2] + BASEMAP.window[3]) / 2.0;
    function normLon(lon) {
      while (lon - frameLon > 180) lon -= 360;
      while (lon - frameLon < -180) lon += 360;
      return lon;
    }

    // ---- uniform-scale auto-fit projection (S4-AD1 #3) --------------
    // nm-ish planar units (lon scaled by cos mid-lat) -> fitted into a
    // fixed-width 1000-unit canvas with margin; the viewBox HEIGHT
    // follows the content so the envelope is NEVER clipped at any
    // panel aspect (meet-scaling does the rest).
    var lats = [], lons = [];
    pts.forEach(function (p) { lats.push(p.lat); lons.push(normLon(p.lon)); });
    coneRing.forEach(function (c) {
      lons.push(normLon(c[0])); lats.push(c[1]);
    });
    var latMid = (Math.min.apply(null, lats) +
                  Math.max.apply(null, lats)) / 2;
    var K = Math.max(0.2, Math.cos(latMid * Math.PI / 180));
    function pxu(lon) { return lon * 60 * K; }
    function pyu(lat) { return -lat * 60; }
    var xs = lons.map(pxu), ys = lats.map(pyu);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var W = 1000, MARGIN = 110;
    var S = (W - 2 * MARGIN) / Math.max(1e-6, x1 - x0);
    var H = Math.max(540, Math.min(1500,
        Math.round((y1 - y0) * S + 2 * MARGIN)));
    var offY = (H - (y1 - y0) * S) / 2;
    function X(lon) { return (pxu(normLon(lon)) - x0) * S + MARGIN; }
    function Y(lat) { return (pyu(lat) - y0) * S + offY; }
    function lonAt(x) { return ((x - MARGIN) / S + x0) / (60 * K); }
    function latAt(y) { return -((y - offY) / S + y0) / 60; }

    var parts = ['<rect class="ac-ocean-fill" width="' + W +
                 '" height="' + H + '"/>'];

    // ---- basemap: land, graticule, ocean watermark (S4-AD1 #2) ------
    var lonL = lonAt(0), lonR = lonAt(W);
    var latT = latAt(0), latB = latAt(H);
    parts.push('<g class="ac-graticule">');
    for (var gl = Math.ceil(lonL / 5) * 5; gl <= lonR; gl += 5) {
      var gx = X(gl);
      parts.push('<line x1="' + gx.toFixed(1) + '" y1="0" x2="' +
        gx.toFixed(1) + '" y2="' + H + '"/>');
      var gn = ((gl % 360) + 360) % 360;
      var glab = gn > 180 ? (360 - gn) + "\u00b0W"
                          : (gn === 0 || gn === 180 ? gn + "\u00b0"
                                                    : gn + "\u00b0E");
      if (gx > 30 && gx < W - 60) {
        parts.push('<text x="' + (gx + 6).toFixed(1) + '" y="' +
          (H - 12) + '">' + glab + "</text>");
      }
    }
    for (var ga = Math.ceil(latB / 5) * 5; ga <= latT; ga += 5) {
      var gy = Y(ga);
      parts.push('<line x1="0" y1="' + gy.toFixed(1) + '" x2="' + W +
        '" y2="' + gy.toFixed(1) + '"/>');
      if (gy > 40 && gy < H - 34) {
        parts.push('<text x="10" y="' + (gy - 6).toFixed(1) + '">' +
          Math.abs(ga) + "\u00b0" + (ga >= 0 ? "N" : "S") + "</text>");
      }
    }
    parts.push("</g>");
    (BASEMAP.land || []).forEach(function (ring) {
      var d = ring.map(function (c, i) {
        return (i ? "L" : "M") + X(c[0]).toFixed(1) + "," +
          Y(c[1]).toFixed(1);
      }).join(" ") + " Z";
      parts.push('<path class="ac-land" d="' + d + '"/>');
    });


    // ---- reveal (S4-AD2 #1): ONE continuous arc-length-parameterized
    // progress drives a clip-path POLYGON keyframe animation on the
    // cone group - compositor-friendly (no per-frame mask re-raster;
    // the round-1 mask-stroke reveal repainted the masked group every
    // frame and visibly stepped). Constant vertex count across
    // keyframes; revealed vertices are static, the leading-edge cap
    // unfolds; a sine ease over GROW_MS=3600 keeps the peak advance
    // ~1.44% of track length per 30fps frame (spec <= 1.5%).
    var tp = pts.map(function (p) { return [X(p.lon), Y(p.lat)]; });
    var lastSeg = tp.length > 1 ? [tp[tp.length - 1][0] - tp[tp.length - 2][0],
                                   tp[tp.length - 1][1] - tp[tp.length - 2][1]]
                                : [1, 0];
    var lsLen = Math.max(1e-6, Math.hypot(lastSeg[0], lastSeg[1]));
    function distToTrack(x, y) {
      var best = Infinity;
      for (var si = 1; si < tp.length; si++) {
        var ax = tp[si - 1][0], ay = tp[si - 1][1];
        var bx2 = tp[si][0], by2 = tp[si][1];
        var ex = bx2 - ax, ey = by2 - ay;
        var L2 = ex * ex + ey * ey;
        var t2 = L2 ? Math.max(0, Math.min(1,
            ((x - ax) * ex + (y - ay) * ey) / L2)) : 0;
        var qx = ax + ex * t2, qy = ay + ey * t2;
        best = Math.min(best, Math.hypot(x - qx, y - qy));
      }
      return best;
    }
    var halfW = 0;
    coneRing.forEach(function (c) {
      halfW = Math.max(halfW, distToTrack(X(c[0]), Y(c[1])));
    });
    var ext = halfW + 60;
    var tpExt = tp.concat([[
      tp[tp.length - 1][0] + lastSeg[0] / lsLen * ext,
      tp[tp.length - 1][1] + lastSeg[1] / lsLen * ext]]);
    var cum = [0];
    for (var ci = 1; ci < tpExt.length; ci++) {
      cum.push(cum[ci - 1] + Math.hypot(tpExt[ci][0] - tpExt[ci - 1][0],
                                        tpExt[ci][1] - tpExt[ci - 1][1]));
    }
    var Ltot = cum[cum.length - 1];
    var HOLD_MS = 1000, GROW_MS = 4000;
    var frontW0 = 64;
    var frontW = 2 * halfW + 110;
    function halfAt(d) {
      return (frontW0 + (frontW - frontW0) * (d / Ltot)) / 2;
    }
    // resample the extended track at uniform arc steps
    var SAMP = 26;
    function pointAt(d) {
      d = Math.max(0, Math.min(Ltot, d));
      for (var k2 = 1; k2 < tpExt.length; k2++) {
        if (d <= cum[k2] || k2 === tpExt.length - 1) {
          var f2 = (d - cum[k2 - 1]) /
                   Math.max(1e-6, cum[k2] - cum[k2 - 1]);
          var x2 = tpExt[k2 - 1][0] +
                   (tpExt[k2][0] - tpExt[k2 - 1][0]) * f2;
          var y2 = tpExt[k2 - 1][1] +
                   (tpExt[k2][1] - tpExt[k2 - 1][1]) * f2;
          var ux2 = (tpExt[k2][0] - tpExt[k2 - 1][0]) /
                    Math.max(1e-6, cum[k2] - cum[k2 - 1]);
          var uy2 = (tpExt[k2][1] - tpExt[k2 - 1][1]) /
                    Math.max(1e-6, cum[k2] - cum[k2 - 1]);
          return { x: x2, y: y2, nx: -uy2, ny: ux2 };
        }
      }
      return { x: tpExt[0][0], y: tpExt[0][1], nx: 0, ny: -1 };
    }
    var samples = [];
    for (var sj = 0; sj < SAMP; sj++) {
      var sd = Ltot * sj / (SAMP - 1);
      var sp = pointAt(sd);
      samples.push({ d: sd, x: sp.x, y: sp.y, nx: sp.nx, ny: sp.ny });
    }
    var CAPV = 5;                       // leading-edge cap vertices
    function polyAt(d) {
      // corridor polygon: left chain forward, leading-edge cap, right
      // chain back. THE CAP'S MAXIMUM FORWARD EXTENT IS EXACTLY d -
      // shoulders sit BEHIND at d - 0.45w and the arc rises to the tip
      // at d. (The first cut bulged the tip 0.55w BEYOND d; the
      // high-contrast 1.8px boundary line painting inside that bulge
      // read as an outline ghosting ahead of the pops - the final-gate
      // bug. Now ink and pops share one front.) Beyond-front samples
      // collapse onto the shoulders so interpolation unfolds them.
      var w = halfAt(d);
      var back = 0.45 * w;
      var f = pointAt(Math.max(0, d - back));   // shoulder base
      var ft = pointAt(d);                       // the tip - AT d
      var shL = [f.x + f.nx * w, f.y + f.ny * w];
      var shR = [f.x - f.nx * w, f.y - f.ny * w];
      var L = [], R = [];
      for (var j2 = 0; j2 < SAMP; j2++) {
        var s2 = samples[j2];
        if (s2.d <= d) {
          var wj = halfAt(s2.d);
          L.push((s2.x + s2.nx * wj).toFixed(1) + " " +
                 (s2.y + s2.ny * wj).toFixed(1));
          R.push((s2.x - s2.nx * wj).toFixed(1) + " " +
                 (s2.y - s2.ny * wj).toFixed(1));
        } else {
          L.push(shL[0].toFixed(1) + " " + shL[1].toFixed(1));
          R.push(shR[0].toFixed(1) + " " + shR[1].toFixed(1));
        }
      }
      var cap = [];
      for (var cv = 0; cv < CAPV; cv++) {
        var th = Math.PI * (cv + 1) / (CAPV + 1);
        // ellipse arc from the LEFT shoulder rising to the TIP AT d,
        // back down to the RIGHT shoulder - never past d
        var fwd = Math.sin(th) * back;
        var lat2 = Math.cos(th) * w;
        var tx = f.ny, ty = -f.nx;      // (nx,ny) rotated -90 = tangent
        cap.push((f.x + f.nx * lat2 + tx * fwd).toFixed(1) + " " +
                 (f.y + f.ny * lat2 + ty * fwd).toFixed(1));
      }
      // an SVG path (M..L..Z), not a CSS polygon: Chromium never
      // repaints style clip-path mutations on SVG containers (probe:
      // interpolated computed values, 108 painted px mid-growth) -
      // updating a real <clipPath><path d> invalidates reliably.
      return "M" + L.concat(cap, R.reverse()).join(" L ") + " Z";
    }
    // TRAPEZOID ease (S4-AD2 #1): quadratic accel/decel over the first
    // and last 15%, LINEAR plateau between - the sine ease's peak
    // velocity (1.57x average) measured 1.63%/frame against the 1.5%
    // smoothness budget once vsync jitter stacked on it; the trapezoid
    // plateau holds ~0.98%/frame nominal with gentle ends.
    var EASE_A = 0.15;
    var EASE_V = 1 / (1 - EASE_A);          // plateau velocity
    function easeS(t) {
      t = Math.max(0, Math.min(1, t));
      if (t < EASE_A) return EASE_V * t * t / (2 * EASE_A);
      if (t > 1 - EASE_A) {
        var u = 1 - t;
        return 1 - EASE_V * u * u / (2 * EASE_A);
      }
      return EASE_V * (t - EASE_A / 2);
    }
    function invEaseS(f) {
      f = Math.max(0, Math.min(1, f));
      var fa = EASE_V * EASE_A / 2;         // progress at the corners
      if (f < fa) return Math.sqrt(2 * EASE_A * f / EASE_V);
      if (f > 1 - fa) return 1 - Math.sqrt(2 * EASE_A * (1 - f) / EASE_V);
      return f / EASE_V + EASE_A / 2;
    }


    // ---- the cone (S4-AD1 #8 restyle): crisp navy/white boundary,
    // subtle white-blue interior, NO glow filters -----------------------
    var dC = coneRing.map(function (c, i) {
      return (i ? "L" : "M") + X(c[0]).toFixed(1) + "," +
        Y(c[1]).toFixed(1);
    }).join(" ") + " Z";
    var dF = pts.map(function (p, i) {
      return (i ? "L" : "M") + X(p.lon).toFixed(1) + "," +
        Y(p.lat).toFixed(1);
    }).join(" ");
    // pill ramp gradients (final-gate #5): THE banner/LIVE-STATUS
    // chrome recipe (edge 0/mid 22/accent 50/mid 78/edge 100) as SVG
    // gradients - the pill is ONE rounded rect FILLED by the ramp, so
    // the sheen is clipped to the pill geometry by construction (the
    // old half-height overlay rect bulged past the corner radius).
    var gdefs = "";
    Object.keys(CAT_TOKENS).forEach(function (k) {
      var c = CAT_TOKENS[k];
      gdefs += '<linearGradient id="pillg-' + k +
        '" x1="0" y1="0" x2="0" y2="1">' +
        '<stop offset="0%" stop-color="' + c.edge + '"/>' +
        '<stop offset="22%" stop-color="' + c.mid + '"/>' +
        '<stop offset="50%" stop-color="' + c.accent + '"/>' +
        '<stop offset="78%" stop-color="' + c.mid + '"/>' +
        '<stop offset="100%" stop-color="' + c.edge + '"/>' +
        "</linearGradient>";
    });
    parts.push('<defs>' + gdefs + '<clipPath id="ac-reveal-clip" ' +
      'clipPathUnits="userSpaceOnUse">' +
      '<path class="ac-reveal-path" d=""/></clipPath></defs>');
    parts.push('<g class="ac-conegrp" clip-path="url(#ac-reveal-clip)">' +
      '<path d="' + dC + '" fill="rgba(205,228,255,0.13)" ' +
      'stroke="#0a1a2e" stroke-width="4.5"/>' +
      '<path d="' + dC + '" fill="none" stroke="#dceaff" ' +
      'stroke-width="1.8"/>' +
      '<path d="' + dF + '" fill="none" stroke="#ffffff" ' +
      'stroke-width="1.6" stroke-dasharray="2 5" ' +
      'stroke-opacity="0.7"/></g>');

    // ---- icons + placards (S4-AD1 #4/5/6/7) --------------------------
    // collision-aware placard layout: alternate sides of the track,
    // push outward on overlap, leader line when pushed far.
    var rects = [];      // occupied rects: title, icons, placards
    // ---- in-plot TITLE LOCKUP (S4-AD2 #5): reserve its corner before
    // anything else places - the side with less cone overlap wins.
    var coneXs = coneRing.map(function (c) { return X(c[0]); });
    var coneYs = coneRing.map(function (c) { return Y(c[1]); });
    var coneBox = { x: Math.min.apply(null, coneXs),
                    y: Math.min.apply(null, coneYs),
                    w: Math.max.apply(null, coneXs) -
                       Math.min.apply(null, coneXs),
                    h: Math.max.apply(null, coneYs) -
                       Math.min.apply(null, coneYs) };
    var TIT_W = 285, TIT_H = 86, TIT_PAD = 22;
    function overlapArea(a, b) {
      var ox = Math.max(0, Math.min(a.x + a.w, b.x + b.w) -
                           Math.max(a.x, b.x));
      var oy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) -
                           Math.max(a.y, b.y));
      return ox * oy;
    }
    var titleLeft = { x: TIT_PAD, y: TIT_PAD, w: TIT_W, h: TIT_H };
    var titleRight = { x: W - TIT_W - TIT_PAD, y: TIT_PAD,
                       w: TIT_W, h: TIT_H };
    var titleRect = (overlapArea(titleLeft, coneBox) <=
                     overlapArea(titleRight, coneBox))
      ? titleLeft : titleRight;
    rects.push(titleRect);
    function overlaps(r) {
      for (var k = 0; k < rects.length; k++) {
        var o = rects[k];
        if (r.x < o.x + o.w && r.x + r.w > o.x &&
            r.y < o.y + o.h && r.y + r.h > o.y) return true;
      }
      return false;
    }
    var iconR = [];      // per-point icon half-size (canvas units)
    pts.forEach(function (p, i) {
      var half = (i === 0 ? 42 : 20);
      iconR.push(half);
      rects.push({ x: tp[i][0] - half, y: tp[i][1] - half,
                   w: 2 * half, h: 2 * half });
    });
    var placards = [];
    pts.forEach(function (p, i) {
      var pw = (i === 0 ? 112 : 86), ph = (i === 0 ? 26 : 22);
      // local track direction -> perpendicular placement sides
      var a = tp[Math.max(0, i - 1)], b = tp[Math.min(tp.length - 1, i + 1)];
      var vx = b[0] - a[0], vy = b[1] - a[1];
      var vl = Math.max(1e-6, Math.hypot(vx, vy));
      var nx = -vy / vl, ny = vx / vl;
      var side = (i % 2 === 0) ? 1 : -1;
      if (i === 0) side = -1;            // NOW placard prefers up-track
      var placed = null, leader = false;
      var offs = [34, 48, 62, 76, 92, 110];
      var nudges = [0, 30, -30, 60, -60, 90, -90];
      // preferred side first, then the opposite; outward pushes, then
      // along-track nudges - placards NEVER overlap (S4-AD1 #7).
      var sides = [side, -side];
      for (var si = 0; si < sides.length && !placed; si++) {
        for (var oi = 0; oi < offs.length && !placed; oi++) {
          for (var ni = 0; ni < nudges.length && !placed; ni++) {
            var cx2 = tp[i][0] + nx * sides[si] * offs[oi]
                      + (vx / vl) * nudges[ni];
            var cy2 = tp[i][1] + ny * sides[si] * offs[oi]
                      + (vy / vl) * nudges[ni];
            var r2 = { x: cx2 - pw / 2, y: cy2 - ph / 2, w: pw, h: ph };
            r2.x = Math.max(6, Math.min(W - pw - 6, r2.x));
            r2.y = Math.max(6, Math.min(H - ph - 6, r2.y));
            if (!overlaps(r2)) {
              placed = r2;
              leader = offs[oi] > 48 || Math.abs(nudges[ni]) > 30;
            }
          }
        }
      }
      if (!placed) {                     // pathological: park it below
        placed = { x: Math.max(6, Math.min(W - pw - 6,
                       tp[i][0] - pw / 2)),
                   y: Math.min(H - ph - 6, 6 + i * (ph + 4)),
                   w: pw, h: ph };
        leader = true;
      }
      rects.push(placed);
      placards.push({ rect: placed, leader: leader });
    });

    var anyNonTropical = false;
    pts.forEach(function (p, i) {
      var px2 = tp[i][0], py2 = tp[i][1];
      var tropical = pointTropical(p);
      if (!tropical) anyNonTropical = true;
      var cat = pointCat(p);
      var col = tropical ? (SSHS[cat] || SSHS.TD) : "#ffffff";
      // pops ride the wavefront EXACTLY: the cap tip sits at the
      // front distance, so delay = hold + grow * invease(d_i / L).
      var delayMs = (i === 0)
        ? 400
        : Math.round(HOLD_MS + GROW_MS *
                     invEaseS(Math.max(0.02, cum[i] / Ltot)));
      var scale = (i === 0 ? 0.95 : 0.42);   // NOW is the hero (#6)
      var tau = Math.round(p.tau_h || 0);
      parts.push('<g class="ac-icon" data-tau="' + tau +
        '" data-tropical="' + (tropical ? 1 : 0) +
        '" style="animation-delay:' + (delayMs / 1000).toFixed(2) +
        's">' + '<g transform="translate(' + px2.toFixed(1) + " " +
        py2.toFixed(1) + ') scale(' + scale + ')">' +
        '<g class="ac-spin"><path d="__HPATH__" fill="' + col +
        '" stroke="rgba(0,0,0,0.35)" stroke-width="2"/></g>' +
        (tropical
          ? '<text class="ac-cat" y="12" text-anchor="middle" ' +
            'font-size="34" font-weight="800" fill="#ffffff" ' +
            'stroke="rgba(0,0,0,0.45)" stroke-width="1">' +
            sshsLabel(cat) + "</text>"
          : "") +
        "</g>");
      // placard (skip none - NOW gets one too), with leader if pushed
      var pl = placards[i];
      var pillGrad = "url(#pillg-" + (tropical ? cat : "NEUTRAL") + ")";
      var label = (i === 0)
        ? "NOW \u00b7 " + Math.round(p.intensity_kt || 0) + "kt"
        : "+" + tau + "h \u00b7 " + Math.round(p.intensity_kt || 0) + "kt";
      var pw2 = pl.rect.w, ph2 = pl.rect.h;
      if (pl.leader) {
        parts.push('<line x1="' + px2.toFixed(1) + '" y1="' +
          py2.toFixed(1) + '" x2="' + (pl.rect.x + pw2 / 2).toFixed(1) +
          '" y2="' + (pl.rect.y + ph2 / 2).toFixed(1) +
          '" stroke="rgba(255,255,255,0.45)" stroke-width="1.2"/>');
      }
      parts.push('<g data-role="placard" data-x="' +
        pl.rect.x.toFixed(1) + '" data-y="' + pl.rect.y.toFixed(1) +
        '" data-w="' + pw2 + '" data-h="' + ph2 +
        '" transform="translate(' + pl.rect.x.toFixed(1) +
        " " + pl.rect.y.toFixed(1) + ')">' +
        '<rect width="' + pw2 + '" height="' + ph2 + '" rx="' +
        (ph2 / 2) + '" fill="' + pillGrad +
        '" stroke="rgba(0,0,0,0.3)"/>' +
        '<text x="' + (pw2 / 2) + '" y="' + (ph2 - 7) +
        '" text-anchor="middle" font-size="' + (i === 0 ? 13.5 : 12.5) +
        '" font-weight="' + (i === 0 ? 800 : 700) +
        // canon ink rule: ALWAYS light on the category ramps (same
        // stroke treatment as the icon SS labels)
        '" fill="#ffffff" stroke="rgba(0,0,0,0.4)" stroke-width="0.8" ' +
        'paint-order="stroke">' + label + "</text></g>");
      parts.push("</g>");
    });

    // ---- title lockup markup (#5) -----------------------------------
    var stormName = (document.getElementById("storm-name") || {})
      .textContent || "";
    var typeWord = (document.getElementById("storm-type") || {})
      .textContent || "";
    var tx0 = titleRect.x, ty0 = titleRect.y;
    var anchorRight = titleRect.x > W / 2;
    var taX = anchorRight ? (titleRect.x + titleRect.w) : tx0;
    var taA = anchorRight ? "end" : "start";
    var railX = anchorRight ? (titleRect.x + titleRect.w + 8) : (tx0 - 8);
    parts.push('<g class="ac-title">' +
      '<rect x="' + (railX - 1.5) + '" y="' + ty0 +
      '" width="3" height="' + TIT_H +
      '" rx="1.5" fill="var(--cat-accent)"/>' +
      '<text class="ac-eyebrow" x="' + taX + '" y="' + (ty0 + 14) +
      '" text-anchor="' + taA +
      '">TRIPLE-A-TROPICS \u00b7 CycloLab</text>' +
      '<text class="ac-head" x="' + taX + '" y="' + (ty0 + 42) +
      '" text-anchor="' + taA + '">FORECAST CONE</text>' +
      '<text class="ac-sub" x="' + taX + '" y="' + (ty0 + 66) +
      '" text-anchor="' + taA + '">' + typeWord.toUpperCase() +
      " " + stormName.toUpperCase() + "</text></g>");

    // ---- ocean watermark (#4): small, auto-placed in the EMPTIEST
    // open water - never behind the cone, placards or the title ------
    var wmW = ((BASEMAP.ocean || "").length * 13) + 30, wmH = 26;
    var best = null, bestScore = -1;
    for (var gy2 = 0.14; gy2 <= 0.9; gy2 += 0.095) {
      for (var gx2 = 0.08; gx2 <= 0.92; gx2 += 0.105) {
        var cxw = gx2 * W, cyw = gy2 * H;
        var r3 = { x: cxw - wmW / 2, y: cyw - wmH / 2, w: wmW, h: wmH };
        if (r3.x < 8 || r3.x + r3.w > W - 8 ||
            r3.y < 8 || r3.y + r3.h > H - 30) continue;
        var bad = overlapArea(r3, coneBox) > 0;
        if (!bad) {
          for (var rk = 0; rk < rects.length && !bad; rk++) {
            if (overlapArea(r3, rects[rk]) > 0) bad = true;
          }
        }
        if (!bad) {                       // open WATER means not on land
          for (var lk = 0; lk < (BASEMAP.land || []).length && !bad;
               lk++) {
            var lr = BASEMAP.land[lk];
            var lx0 = Infinity, lx1 = -Infinity,
                ly0 = Infinity, ly1 = -Infinity;
            for (var lv = 0; lv < lr.length; lv++) {
              var lpx = X(lr[lv][0]), lpy = Y(lr[lv][1]);
              lx0 = Math.min(lx0, lpx); lx1 = Math.max(lx1, lpx);
              ly0 = Math.min(ly0, lpy); ly1 = Math.max(ly1, lpy);
            }
            if (overlapArea(r3, { x: lx0, y: ly0, w: lx1 - lx0,
                                  h: ly1 - ly0 }) > 0) bad = true;
          }
        }
        if (bad) continue;
        // score: distance from the cone box centre (deep open water)
        var dxw = cxw - (coneBox.x + coneBox.w / 2);
        var dyw = cyw - (coneBox.y + coneBox.h / 2);
        var sc = dxw * dxw + dyw * dyw;
        if (sc > bestScore) { bestScore = sc; best = { x: cxw, y: cyw }; }
      }
    }
    if (best) {
      parts.push('<text class="ac-ocean" x="' + best.x.toFixed(0) +
        '" y="' + best.y.toFixed(0) + '" text-anchor="middle">' +
        (BASEMAP.ocean || "") + "</text>");
    }

    // hairline frame: the map has edges (#2)
    parts.push('<rect class="ac-frame" x="0.75" y="0.75" width="' +
      (W - 1.5) + '" height="' + (H - 1.5) + '" rx="2"/>');

    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.innerHTML = parts.join("");

    // arm the growth front (WAAPI dashoffset); reduced motion / jsdom
    // jump to the final frame
    var grp = svg.querySelector(".ac-conegrp");
    // ONE continuous arc-length-parameterized progress drives the clip
    // polygon from a rAF loop with INLINE STYLE writes. WAAPI/CSS
    // animation of a basic-shape clip on an SVG container interpolates
    // in computed style but Chromium never repaints until it finishes
    // (probe-verified: computed values moved, pixels snapped at the
    // end) - manual style mutation forces the recalc+paint every
    // frame. Polygon rebuild per tick is ~57 vertices: trivial.
    // (grp.animate doubles as the capability gate: jsdom lacks it and
    // its immediate-rAF stub would otherwise spin the loop.)
    if (acRaf) { cancelAnimationFrame(acRaf); acRaf = null; }
    var revealPath = svg.querySelector(".ac-reveal-path");
    if (!reduced && grp.animate) {
      revealPath.setAttribute("d", polyAt(0));
      grp.setAttribute("data-reveal", "animated");
      var t0 = performance.now() + HOLD_MS;
      var tickFn = function () {
        var tt = (performance.now() - t0) / GROW_MS;
        if (tt >= 1) {
          grp.removeAttribute("clip-path");
          acRaf = null;
          return;
        }
        if (tt > 0) {
          revealPath.setAttribute("d", polyAt(easeS(tt) * Ltot));
        }
        acRaf = requestAnimationFrame(tickFn);
      };
      acRaf = requestAnimationFrame(tickFn);
    } else {
      grp.removeAttribute("clip-path");  // final frame: fully revealed
      grp.setAttribute("data-reveal", "final");
    }

    var official = advFull.method === "official-cone";
    note.textContent = (official
      ? "Official NHC forecast cone \u00b7 advisory " + advFull.advisory +
        " \u00b7 icons colored by forecast intensity."
      : "Derived uncertainty envelope \u2014 not an official JTWC " +
        "product \u00b7 advisory " + advFull.advisory + " \u00b7 method " +
        (advFull.method || "derived") + ".") +
      (anyNonTropical ? " White icons = forecast non-tropical." : "");
    document.getElementById("advcone-method-body").textContent =
      official ? NHC_METHOD_COPY : WP_METHOD_COPY;
  }

  // (§8.6) THE INTENSITY CONE: forecast-VMAX center line + published-
  // error envelope on SSHWS bands. Brand-toned translucent envelope -
  // category color lives ONLY in the icons and bands. Honesty guard:
  // no registry entry => a labeled panel, never a borrowed envelope.
  function maeAt(entry, tau) {
    var table = {};
    var taus = [];
    for (var k in entry.mae_kt) {
      if (entry.mae_kt.hasOwnProperty(k)) {
        table[+k] = +entry.mae_kt[k]; taus.push(+k);
      }
    }
    taus.sort(function (a, b) { return a - b; });
    if (tau <= 0) return 0;
    if (table[tau] != null) return table[tau];
    var lo = 0, loV = 0;
    for (var i = 0; i < taus.length; i++) {
      if (taus[i] > tau) {
        return loV + (table[taus[i]] - loV) * (tau - lo) / (taus[i] - lo);
      }
      lo = taus[i]; loV = table[taus[i]];
    }
    return table[taus[taus.length - 1]];
  }
  function intensityRows() {
    if (!INTENSITY_ERR || !advFull || !advFull.points) return null;
    var rows = [];
    advFull.points.forEach(function (p) {
      if (p.intensity_kt == null) return;
      var tau = +(p.tau_h || 0);
      var m = maeAt(INTENSITY_ERR, tau);
      var kt = +p.intensity_kt;   // coerce BEFORE arithmetic - a string
      rows.push({ tau: tau, center: kt,        // kt would concatenate
                  upper: Math.min(200, kt + m),
                  lower: Math.max(0, kt - m), mae: m });
    });
    return rows;
  }
  function renderIntensity() {
    var svg = document.getElementById("intensity");
    var note = document.getElementById("intensity-note");
    var method = document.getElementById("intensity-method");
    var missing = document.getElementById("intensity-missing");
    if (!INTENSITY_ERR) {
      svg.innerHTML = ""; note.hidden = true; method.hidden = true;
      missing.style.display = "block";
      missing.textContent = "No published intensity-error statistics " +
        "for this basin \u2014 an intensity range is not shown rather " +
        "than borrowed or invented.";
      return;
    }
    var rows = intensityRows();
    if (!rows || rows.length < 2) {
      svg.innerHTML = ""; note.hidden = true; method.hidden = true;
      missing.style.display = "block";
      missing.textContent = "No forecast intensity points yet for this " +
        "storm.";
      return;
    }
    missing.style.display = "none"; note.hidden = false;
    method.hidden = false;
    var W = 1000, H = 380, padL = 56, padR = 26, padT = 18, padB = 40;
    var tMax = rows[rows.length - 1].tau || 120;
    var kMax = 80;
    rows.forEach(function (r) { if (r.upper + 12 > kMax) kMax = r.upper + 12; });
    function Xt(t) { return padL + t / tMax * (W - padL - padR); }
    function Yk(k) { return H - padB - k / kMax * (H - padT - padB); }
    var parts = ['<rect width="' + W + '" height="' + H +
                 '" fill="#0a1019"/>'];
    // SSHWS bands from the canonical palette
    var bands = [[0, 34, "TD"], [34, 64, "TS"], [64, 83, "C1"],
                 [83, 96, "C2"], [96, 113, "C3"], [113, 137, "C4"],
                 [137, 999, "C5"]];
    bands.forEach(function (b) {
      var top = Math.min(b[1], kMax), bot = b[0];
      if (bot >= kMax) return;
      parts.push('<rect x="' + padL + '" y="' + Yk(top).toFixed(1) +
        '" width="' + (W - padL - padR) + '" height="' +
        (Yk(bot) - Yk(top)).toFixed(1) + '" fill="' + SSHS[b[2]] +
        '" fill-opacity="0.10"/>');
      if (b[1] <= kMax) {
        parts.push('<text x="' + (W - padR - 6) + '" y="' +
          (Yk(b[1]) + 12).toFixed(1) + '" text-anchor="end" ' +
          'font-size="11" fill="' + SSHS[b[2]] + '" fill-opacity="0.8">' +
          b[2] + "</text>");
      }
    });
    // axes ticks
    rows.forEach(function (r) {
      parts.push('<text x="' + Xt(r.tau).toFixed(1) + '" y="' + (H - 14) +
        '" text-anchor="middle" font-size="11.5" fill="#8b95a5">+' +
        Math.round(r.tau) + "h</text>");
    });
    [0, 25, 50, 75, 100, 125, 150].forEach(function (k) {
      if (k > kMax) return;
      parts.push('<text x="' + (padL - 8) + '" y="' +
        (Yk(k) + 4).toFixed(1) + '" text-anchor="end" font-size="11" ' +
        'fill="#8b95a5">' + k + "</text>");
    });
    // envelope (brand-toned translucent, never category-colored)
    var up = rows.map(function (r, i) {
      return (i ? "L" : "M") + Xt(r.tau).toFixed(1) + "," +
        Yk(r.upper).toFixed(1);
    }).join(" ");
    var down = rows.slice().reverse().map(function (r) {
      return "L" + Xt(r.tau).toFixed(1) + "," + Yk(r.lower).toFixed(1);
    }).join(" ");
    parts.push('<path d="' + up + " " + down +
      ' Z" fill="rgba(255,255,255,0.10)" stroke="#8cc8ff" ' +
      'stroke-width="1.5" stroke-opacity="0.45"/>');
    // center line
    var center = rows.map(function (r, i) {
      return (i ? "L" : "M") + Xt(r.tau).toFixed(1) + "," +
        Yk(r.center).toFixed(1);
    }).join(" ");
    parts.push('<path d="' + center + '" fill="none" stroke="#ffffff" ' +
      'stroke-width="2.5" stroke-linejoin="round"/>');
    // spinning category icons at each forecast point
    rows.forEach(function (r, i) {
      var col = SSHS[catForKt(r.center)] || SSHS.TD;
      parts.push('<g class="ac-icon" style="animation-delay:' +
        (0.3 + i * 0.15).toFixed(2) + 's"><g transform="translate(' +
        Xt(r.tau).toFixed(1) + " " + Yk(r.center).toFixed(1) +
        ') scale(0.3)"><g class="ac-spin"><path d="__HPATH__" fill="' +
        col + '" stroke="rgba(0,0,0,0.35)" stroke-width="2"/></g></g></g>');
    });
    svg.innerHTML = parts.join("");
    var body = "Center line: the official forecast intensity (VMAX) " +
      "from advisory " + advFull.advisory + ". Shaded range: \u00b1 the " +
      (INTENSITY_ERR.agency || "") + " published mean absolute intensity " +
      "error (" + (INTENSITY_ERR.error_type || "MAE") + ", " +
      (INTENSITY_ERR.window || "") + " window) at each lead time \u2014 " +
      "method " + INTENSITY_ERR.method_version + ". Lead times without a " +
      "published value are interpolated linearly. It reflects the " +
      "historical average error of past forecasts \u2014 it is not a " +
      "probabilistic bound, and the storm\u2019s intensity can fall " +
      "outside it. Do not use for life-safety decisions \u2014 see the " +
      "official advisory text." +
      (INTENSITY_ERR.staleness_note ? " " + INTENSITY_ERR.staleness_note
                                    : "") +
      (INTENSITY_ERR.vintage_note ? " " + INTENSITY_ERR.vintage_note : "");
    document.getElementById("intensity-method-body").textContent = body;
  }

  // (§7.4) advisory text panels - monospace, never tinted
  var advTextProd = "tcp";
  function renderAdvText() {
    var pre = document.getElementById("advtext");
    var t = (advFull && advFull.text) || {};
    var body = advTextProd === "tcp" ? t.tcp : t.tcd;
    pre.textContent = body ||
      "(advisory text not available for this advisory)";
    var host = document.getElementById("advtext-tabs");
    for (var i = 0; i < host.children.length; i++) {
      var b = host.children[i];
      b.classList.toggle("active",
                         b.getAttribute("data-prod") === advTextProd);
    }
  }
  document.getElementById("advtext-tabs").addEventListener("click",
    function (e) {
      var b = e.target.closest(".hafs-seg");
      if (!b) return;
      advTextProd = b.getAttribute("data-prod");
      renderAdvText();
    });
  function renderAdvTab() {
    // ISOLATED (final-gate #3): a throw in any one renderer must never
    // starve the others - the user-visible failure was the advisory
    // text staying blank because renderAdvCone ran first in this chain.
    var fns = [renderAdvCone, renderIntensity, renderAdvText];
    for (var i = 0; i < fns.length; i++) {
      try { fns[i](); } catch (e) {
        try { console.warn("[cyclolab] adv renderer failed:", e); }
        catch (e2) {}
      }
    }
  }

  // ---- Stage 3: Models mount (componentized /models/ HafsViewer) ----------
  // One impl, two mounts (CYCLOLAB_DESIGN §7.3): hafs.js is lazy-loaded on
  // first tab open from the house origin, then constructed with THIS page's
  // element table and the storm lock from the id join (sid -> hafs id). The
  // /models/ auto-boot keys off #hafs-viewer, absent here - no double-boot.
  var hafsViewer = null;
  function withHafsViewer(cb, onerr) {
    if (window.HafsViewer) { cb(); return; }
    var s = document.createElement("script");
    s.src = SITE_BASE + "/models/hafs.js";
    s.onload = function () {
      if (window.HafsViewer) cb(); else onerr();
    };
    s.onerror = onerr;
    document.head.appendChild(s);
  }
  function initModels() {
    function cl(id) { return document.getElementById("cl-hafs-" + id); }
    var status = cl("status");
    status.style.display = "flex";
    function fail() {
      status.querySelector("span").textContent =
        "Model viewer failed to load - reload to retry.";
    }
    withHafsViewer(function () {
      hafsViewer = new window.HafsViewer(cl("root") ||
          document.getElementById("cl-hafs-root"), {
        manifestUrl: CDN + "/models/hafs/manifest.json",
        assetBase: CDN + "/models/hafs/",
        stormLock: HAFS_ID,
        els: { stage: cl("stage"), img: cl("img"), status: status,
               empty: cl("empty"), controls: cl("controls"),
               cycleGroup: cl("cycle-group"), cycles: cl("cycles"),
               stormSel: cl("storm"), models: cl("models"),
               domains: cl("domains"), products: cl("products"),
               hours: cl("hours"), play: cl("play"),
               stepB: cl("step-back"), stepF: cl("step-fwd"),
               speed: cl("speed"), fhour: cl("fhour"), valid: cl("valid"),
               meta: cl("meta"), badge: cl("badge"), pill: cl("pill"),
               buffer: cl("buffer"), player: cl("player"),
               caption: cl("caption") }
      });
    }, fail);
  }

  // ---- Stage 3: Satellite mount (storm-scoped floater viewer, §7.2) -------
  // Lazy (nothing fetched until the tab opens); newest frame first, then a
  // small bounded backward preload window; band switches keep the MOMENT
  // (nearest frame in the new band's availability - the hour-grid idiom on
  // a time axis).
  var sat = { man: null, band: null, frames: [], idx: 0, playing: false,
              timer: null, gen: 0, preloaded: {} };
  function satEl(id) { return document.getElementById("sat-" + id); }
  function satStatus(show, msg) {
    var box = satEl("status");
    box.style.display = show ? "flex" : "none";
    if (msg != null) box.querySelector("span").textContent = msg;
  }
  function satShow(i) {
    var n = sat.frames.length;
    if (!n) return;
    if (i < 0) i = 0;
    if (i >= n) i = n - 1;
    sat.idx = i;
    var f = sat.frames[i];
    satEl("img").src = CDN + "/" + f.key;
    satEl("scrub").value = String(i);
    satEl("time").textContent =
      f.t.slice(0, 16).replace("T", " ") + "Z";
  }
  function satPreload() {
    // newest-backwards, bounded window, ~3 in flight (P3-b loader pattern).
    var gen = ++sat.gen;
    var want = [];
    for (var k = sat.frames.length - 1;
         k >= 0 && want.length < 12; k--) want.push(sat.frames[k]);
    var inflight = 0, qi = 0;
    function next() {
      if (gen !== sat.gen) return;
      while (inflight < 3 && qi < want.length) {
        var f = want[qi++];
        var u = CDN + "/" + f.key;
        if (sat.preloaded[u]) continue;
        inflight++;
        var im = new Image();
        sat.preloaded[u] = im;
        im.onload = im.onerror = function () { inflight--; next(); };
        im.src = u;
      }
    }
    next();
  }
  function satNearest(t) {
    var best = sat.frames.length - 1, dBest = Infinity;
    var want = new Date(t).getTime();
    for (var i = 0; i < sat.frames.length; i++) {
      var d = Math.abs(new Date(sat.frames[i].t).getTime() - want);
      if (d < dBest) { dBest = d; best = i; }
    }
    return best;
  }
  function satSelectBand(slug, keepTime) {
    var prevT = (keepTime && sat.frames.length) ? sat.frames[sat.idx].t : null;
    sat.band = slug;
    var b = (sat.man && sat.man.bands && sat.man.bands[slug]) || null;
    sat.frames = (b && b.frames) || [];
    satEl("scrub").max = String(Math.max(0, sat.frames.length - 1));
    satEl("band-label").textContent = b ? (b.label || slug) : "";
    // a band key can survive a server-side prune with zero frames left:
    // honest per-band empty state instead of freezing on the old frame.
    satEl("empty").style.display = sat.frames.length ? "none" : "block";
    if (!sat.frames.length) {
      satEl("img").removeAttribute("src");
      satEl("time").textContent = "\u2014";
    }
    var host = satEl("bands");
    for (var i = 0; i < host.children.length; i++) {
      var btn = host.children[i];
      btn.classList.toggle("active", btn.getAttribute("data-slug") === slug);
    }
    satShow(prevT != null ? satNearest(prevT) : sat.frames.length - 1);
    satPreload();
  }
  function satStep(d) { satPause(); satShow(sat.idx + d); }
  function satPause() {
    sat.playing = false;
    if (sat.timer) { clearInterval(sat.timer); sat.timer = null; }
    satEl("play").innerHTML = "&#9654; Play";
  }
  function satTogglePlay() {
    if (sat.playing) { satPause(); return; }
    if (!sat.frames.length) return;
    sat.playing = true;
    satEl("play").innerHTML = "&#10074;&#10074; Pause";
    sat.timer = setInterval(function () {
      satShow(sat.idx >= sat.frames.length - 1 ? 0 : sat.idx + 1);
    }, 200);
  }
  function initSatellite() {
    satStatus(true, "Loading\u2026");
    fetchJson(CDN + "/floaters/manifest.json").then(function (top) {
      var entry = null;
      var storms = (top && top.storms) || [];
      for (var i = 0; i < storms.length; i++) {
        if (String(storms[i].id).toLowerCase() === FLOATER_ID) {
          entry = storms[i]; break;
        }
      }
      if (!entry || !entry.manifest) return null;
      return fetchJson(CDN + "/" + entry.manifest);
    }).then(function (man) {
      if (!man || !man.bands) {
        satStatus(false);
        satEl("empty").style.display = "block";
        return;
      }
      sat.man = man;
      satStatus(false);
      var host = satEl("bands");
      host.innerHTML = "";
      var slugs = [];
      for (var slug in man.bands) {
        if (!man.bands.hasOwnProperty(slug)) continue;
        slugs.push(slug);
        var b = document.createElement("button");
        b.type = "button";
        b.className = "hafs-seg";
        b.setAttribute("data-slug", slug);
        b.textContent = man.bands[slug].label || slug;
        b.addEventListener("click", function () {
          satPause();
          satSelectBand(this.getAttribute("data-slug"), true);
        });
        host.appendChild(b);
      }
      if (!slugs.length) {
        satEl("empty").style.display = "block";
        return;
      }
      satSelectBand(slugs.indexOf("ir") >= 0 ? "ir" : slugs[0], false);
      satEl("step-back").addEventListener("click", function () { satStep(-1); });
      satEl("step-fwd").addEventListener("click", function () { satStep(1); });
      satEl("play").addEventListener("click", satTogglePlay);
      satEl("scrub").addEventListener("input", function () {
        satPause(); satShow(Number(this.value));
      });
      satEl("card").addEventListener("keydown", function (e) {
        // native control focused (scrub slider, buttons): let it own keys
        // except the arrows on the slider, where step == native move.
        if (/^(select|textarea)$/i.test(e.target.tagName)) return;
        if (e.target.tagName === "BUTTON" &&
            (e.key === " " || e.key === "Spacebar")) return;
        if (e.key === "ArrowLeft") { satStep(-1); e.preventDefault(); }
        else if (e.key === "ArrowRight") { satStep(1); e.preventDefault(); }
        else if (e.key === " " || e.key === "Spacebar") {
          satTogglePlay(); e.preventDefault();
        }
      });
    });
  }

  window.__lab = { openSec: openSec, setCategory: setCategory,
                   apply: apply, applyAdvisory: applyAdvisory,
                   odoSet: odoSet,
                   // Stage-3 deterministic hooks (tests + ops)
                   renderAdvTab: renderAdvTab,
                   intensityRows: intensityRows,
                   satState: function () {
                     return { band: sat.band, idx: sat.idx,
                              frames: sat.frames.length,
                              playing: sat.playing };
                   },
                   hafsViewer: function () { return hafsViewer; } };
})();
</script>
</body>
</html>
"""


def _esc(s) -> str:
    return _html.escape(str(s if s is not None else ""), quote=True)


def _type_word(cat: str, basin: str) -> str:
    if cat == "TD":
        return "Tropical Depression"
    if cat == "TS":
        return "Tropical Storm"
    return "Typhoon" if basin == "WP" else "Hurricane"


def _sshs_label(cat: str) -> str:
    """Canonical icon letter/number, D / S / 1-5 - exact mirror of the
    inline JS sshsLabel (the placard-spinner fallback variant of the
    canon, generate_tracks_plot.py:1326; ace_core.sshs_label is the
    strict-dict form), including the None/empty -> "D" guard."""
    if cat == "TD":
        return "D"
    if cat == "TS":
        return "S"
    return (cat or "").replace("C", "") or "D"


def _odo_static(text) -> str:
    """Bake an odometer's initial cells (static .digit spans; non-digits
    carry .ch = auto-width, mirroring odoSet's REST state) so the no-JS
    render carries real values. The live odoSet() leaves matching cells
    untouched (rest IS this plain-text form); only a digit that actually
    changes swaps in a rolling column, then settles back to this."""
    return "".join(
        f'<span class="digit{"" if ch.isdigit() else " ch"}"'
        f' aria-hidden="true">{_esc(ch)}</span>'
        for ch in str(text))


def render_page(storm: dict, *, feed_url: str, adv_url: str | None = None,
                ended: bool = False, loader: str = "b",
                og_image_url: str | None = None) -> str:
    """Render one storm's shell. ``loader`` selects the loading screen:
    "b" (eye opens - THE CHOSEN loader, AD R3 default) or the other
    prototypes "a"/"c"/"d"; "" = plain wipe."""
    ids = parse_sid(storm["sid"])
    cat = storm.get("current_category") or "TD"
    if cat not in CAT_TOKENS:
        cat = "TD"
    name = (storm.get("name") or ids.nhc_id).upper()
    last = (storm.get("points") or [{}])[-1]
    wind = last.get("wind_kt")
    chip = {"TD": "Tropical Depression", "TS": "Tropical Storm",
            "C1": "Category 1", "C2": "Category 2", "C3": "Category 3",
            "C4": "Category 4", "C5": "Category 5"}.get(cat, cat)
    type_word = _type_word(cat, ids.basin)
    og_title = f"{name} · {chip} · CycloLab"
    bits = []
    if wind is not None:
        bits.append(f"{round(float(wind))} kt")
    if last.get("pressure_mb") is not None:
        bits.append(f"{round(float(last['pressure_mb']))} mb")
    state = "Final track and statistics" if ended else "Live tracking"
    og_desc = (f"{state} for {name}"
               + (f" — {' · '.join(bits)}" if bits else "")
               + " · Triple-A-Tropics CycloLab")

    baked = json.dumps(storm, separators=(",", ":"))

    html = (HTML_TEMPLATE
            .replace("__FONT_CSS__", font_css())
            .replace("__CAT_CSS__", cat_css())
            .replace("__HPATH__", HURRICANE_PATH)
            .replace("__CAT__", cat)
            .replace("__CAT_LABEL__", _esc(_sshs_label(cat)))
            .replace("__CAT_ODO__", _odo_static(_sshs_label(cat)))
            .replace("__VMAX_A11Y__", _esc(
                round(float(wind)) if wind is not None else "—"))
            .replace("__VMAX_ODO__", _odo_static(
                round(float(wind)) if wind is not None else "—"))
            .replace("__NAME__", _esc(name))
            .replace("__TYPE_WORD__", _esc(type_word.upper()))
            .replace("__CHIP__", _esc(chip))
            .replace("__CHIP_STYLE__",
                     ' style="display:none"' if cat in ("TD", "TS") else "")
            .replace("__OG_TITLE__", _esc(og_title))
            .replace("__OG_DESC__", _esc(og_desc))
            .replace("__PAGE_PATH__", _esc(page_url_path(storm["sid"])))
            .replace("__SID__", _esc(storm["sid"]))
            .replace("__FEED_URL__", _esc(feed_url))
            .replace("__HAFS_ID__", _esc(ids.hafs_id))
            .replace("__OG_IMAGE__",
                     ('\n<meta property="og:image" content="' +
                      _esc(og_image_url) + '">') if og_image_url else "")
            .replace("__CAT_TOKENS__", json.dumps(
                {**{k: {kk: v[kk] for kk in ("edge", "mid", "accent")}
                    for k, v in CAT_TOKENS.items()},
                 "NEUTRAL": {"edge": _shade("#8b95a5", 0.30),
                             "mid": _shade("#8b95a5", 0.62),
                             "accent": "#8b95a5"}},
                separators=(",", ":")))
            .replace("__BASEMAP__",
                     json.dumps(basemap_for(
                         float(last.get("lat") or 15.0),
                         float(last.get("lon") or -140.0), ids.basin),
                         separators=(",", ":")))
            .replace("__INTENSITY_ERR__",
                     json.dumps(basin_entry(ids.basin),
                                separators=(",", ":")))
            .replace("__ATCF_LONG__", _esc(ids.atcf_long))
            .replace("__ADV_URL__", _esc(adv_url or adv_key(storm["sid"])))
            .replace("__BASIN__", ids.basin)
            .replace("__LOADER__", _esc(loader))
            .replace("__SSHS_JSON__", json.dumps(SSHS_COLORS))
            .replace("__ENDED__", "true" if ended else "false")
            .replace("__BAKED__", baked))
    if ended:
        html = html.replace("<html lang=\"en\" data-cat=",
                            "<html lang=\"en\" data-ended data-cat=")
    return html
