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
    /* ONE sizing token (FG-R3 widget-size pass): every content panel/SVG
       caps its height here so the auto-fit cone + plots stay "a panel, not a
       poster" - the whole content fits WITHOUT scrolling, composing with the
       sidebar on one screen. The cone's auto-fit viewBox scales DOWN to the
       cap (preserveAspectRatio meet = no clip), just a reasonable scale. */
    --panel-max-h: 62vh;
    /* FG-R3 #3a: the lockup accent rail is ALWAYS the house blue - never a
       category accent. Defined once at :root and NOT overridden by the
       per-category token rules, so every lockup rail (cone / overview /
       hero) reads the same blue at any storm intensity. */
    --ac-rail: #3fa4ff;
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
  /* ---- Stage C: NHC formation-chance pill (invests) ----
     The KEY invest metric: 48-hour + 7-day genesis odds, colour-coded by the
     canonical NHC low/medium/high scheme (<=30 yellow, 40-60 orange, >=70 red)
     - a pop of forecast colour on the otherwise-grey invest banner. */
  .formation-pill { display: inline-flex; align-items: center; gap: 8px;
    margin-top: 6px; padding: 3px 10px 3px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.04em;
    border: 1px solid currentColor; width: fit-content; }
  /* The genesis-odds pill is invest/PTC-only. For a designated storm it is
     left [hidden] + empty, but the .formation-pill display above outranks the
     [hidden] attribute, leaving a stray empty capsule (the "blank pill"). Hide
     it whenever it carries no odds; a populated invest pill (not empty, not
     hidden) still shows. */
  .formation-pill[hidden], .formation-pill:empty { display: none; }
  /* "FORMATION" with "chance" stacked beneath it */
  .formation-pill .fp-eyebrow { display: inline-flex; flex-direction: column;
    line-height: 1.04; font-size: 9px; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; opacity: 0.95; }
  .formation-pill .fp-eyebrow .fp-e2 { font-size: 8px; font-weight: 700; opacity: 0.8; }
  /* the two windows sit CLOSE together with a thin vertical divider between them */
  .formation-pill .fp-wins { display: inline-flex; align-items: center; gap: 6px; }
  .formation-pill .fp-div { flex: 0 0 auto; width: 1px; align-self: stretch;
    background: currentColor; opacity: 0.5; }
  /* both windows (48h left, 7-day right) styled IDENTICALLY */
  .formation-pill .fp-win { color: #f3f7fc; font-weight: 700; white-space: nowrap;
    font-variant-numeric: tabular-nums; }
  .formation-pill .fp-win b { font-weight: 800; }
  .formation-pill .fp-dot { flex: 0 0 auto; width: 6px; height: 6px;
    border-radius: 50%; background: currentColor; box-shadow: 0 0 6px currentColor; }
  .formation-pill[data-level="low"]    { color: #f5c842; background: rgba(245,200,66,0.14); }
  .formation-pill[data-level="medium"] { color: #ff9a4d; background: rgba(255,140,61,0.16); }
  .formation-pill[data-level="high"]   { color: #ff6b6b; background: rgba(255,77,77,0.18); }
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
  /* ---- Stage C: INVEST + PTC grey identity + giant red X ----
     data-invest (a 90-99 invest) AND data-ptc (a Potential Tropical Cyclone:
     a DESIGNATED system NHC is advising on while still a DB/DS disturbance)
     SHARE the grey identity: both OVERRIDE the category vars to grey, so the
     banner ramp, accent, sec-btn, vitals, chips - everything keyed on --cat-* -
     reads grey with no per-element edits, and both swap the spinning cyclone
     glyph for the red X. They DIVERGE on official products: an invest has none,
     so it hides the cone/advisories section + nav button and the next-advisory
     vital; a PTC KEEPS them (NHC is actively advising). Both hide the ACE vital
     (a PTC accrues no ACE; an invest none either). The Models tab is shown for
     BOTH (guidance lives there now - Phase 3b); HAFS degrades gracefully when a
     storm has no run. */
  .banner .glyph .invest-x { display: none; }
  html[data-invest], html[data-ptc] {
    --cat-ramp: linear-gradient(180deg,#2a2f3a,#8b95a5,#2a2f3a);
    --cat-accent: #9aa6b6; --cat-ink: #ffffff; }
  html[data-invest] .banner .glyph,
  html[data-ptc] .banner .glyph { filter: drop-shadow(0 0 7px rgba(255,59,59,0.55))
    drop-shadow(0 1px 2px rgba(0,0,0,0.45)); }
  html[data-invest] .banner .glyph .spin,
  html[data-invest] .banner .glyph #glyph-cat,
  html[data-ptc] .banner .glyph .spin,
  html[data-ptc] .banner .glyph #glyph-cat { display: none; }
  html[data-invest] .banner .glyph .invest-x,
  html[data-ptc] .banner .glyph .invest-x { display: block; }
  /* INVEST-ONLY: no official cone/advisories + no next-advisory countdown. */
  html[data-invest] [data-sec="advisories"] { display: none; }
  html[data-invest] #sec-advisories { display: none !important; }
  html[data-invest] #vrow-next { display: none; }
  /* SHARED: neither an invest nor a PTC accrues ACE. */
  html[data-invest] #vrow-ace,
  html[data-ptc] #vrow-ace { display: none; }
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

  /* numeric stats - PLAIN STATIC TEXT (final-gate-3 #2: the odometer
     is gone). Every value renders as ordinary type on the row's natural
     baseline at all times: a 9 is pixel-identical to a 9 typed as
     static text, round-digit baseline overshoot (0/3/5/6/8/9 dip
     1-2px - correct typography) intact, with NO box, clip, strip, or
     transform anywhere. A value CHANGE is marked by one subtle
     fade-in of the new text (reduced motion / no-JS = instant swap).
     tnum keeps the digits tabular so a changing number never reflows
     its neighbours; white-space:pre keeps space cells real. The baked
     no-JS form is the same plain text the live updater writes. */
  .odo { display: inline-block; white-space: pre;
    font-feature-settings: "tnum"; font-variant-numeric: tabular-nums; }
  .odo.odo-swap { animation: odo-swap calc(var(--motion-fast) * 0.5)
    ease-out 1; }
  @keyframes odo-swap { from { opacity: 0.3; } to { opacity: 1; } }

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
  .back-map { color: var(--muted); text-decoration: none;
    font-size: 12.5px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; display: inline-flex; align-items: center;
    white-space: nowrap; }
  .back-map:hover { color: var(--fg); }

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
  .card svg { width: 100%; height: auto; max-height: var(--panel-max-h);
    display: block; touch-action: pan-y; }
  /* maps-pass R4 #1: the track + swath maps get a DEFINITE height (the
     shared aspect-fill rule) so their basemap COVERS edge-to-edge like the
     cone, no letterbox; respects the 62vh no-scroll cap. */
  #trackplot, #swathplot { height: clamp(320px, 56vh, 560px); }
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
  /* ---- Gutter layout (CycloLab polish #3): the rendered frame is ~square, so
     a full-width stack left a big dead band beside it. Reclaim it - image in the
     main column, controls stacked in a side gutter, frame strip / scrubber full
     width beneath. This rule set is a MIRROR of the one in /models/ index.html;
     keep the two identical (one canon, no drift). Color-agnostic - the storm
     tokens ride the inner chrome, not this layout. */
  .vw-grid { display: grid;
    grid-template-columns: minmax(0, max-content) clamp(214px, 28%, 326px);
    justify-content: start; align-items: start; gap: 16px 22px; }
  .vw-grid > .hafs-stage { grid-column: 1; grid-row: 1; min-width: 0; }
  .vw-grid > .vw-aside { grid-column: 2; grid-row: 1;
    display: flex; flex-direction: column; gap: 16px; min-width: 0; }
  .vw-grid > .vw-below,
  .vw-grid > .hafs-caption,
  .vw-grid > .hafs-footer,
  .vw-grid > .stub { grid-column: 1 / -1; min-width: 0; }
  /* gutter chrome: tidy wrapping CHIPS + ONE compact transport row, readout
     on its own line (mirror of /models/ index.html - keep identical). */
  .vw-aside .hafs-controls { flex-direction: column; align-items: stretch;
    gap: 14px; padding: 0; background: none; border: none; }
  .vw-aside .hafs-controls.sat-tools { align-items: stretch; }
  .vw-aside .sat-gif { align-self: stretch; }   /* full-gutter, not orphaned */
  .vw-aside .hafs-seg-group { flex-wrap: wrap; gap: 6px; }
  .vw-aside .hafs-seg-group .hafs-seg { border: 1px solid var(--border);
    border-radius: 7px; padding: 6px 11px; font-size: 11.5px; }
  .vw-aside .hafs-seg-group .hafs-seg.active { border-color: var(--cat-accent); }
  .vw-aside .hafs-player { flex-direction: row; flex-wrap: wrap;
    align-items: center; gap: 8px; padding: 0; background: none; border: none; }
  .vw-aside .hafs-player .hafs-play { flex: 1 1 auto; }
  .vw-aside .hafs-player .hafs-readout,
  .vw-aside .hafs-player .hafs-group { flex: 1 1 100%; }
  .vw-aside .hafs-group select { width: 100%; min-width: 0; }
  .vw-below .hafs-hours { padding-left: 0; padding-right: 0; }
  .vw-below #sat-scrub { width: 100%; }
  @media (max-width: 760px) {
    /* no gutter room: stack; the controls become a compact horizontal bar */
    .vw-grid { display: block; }
    .vw-aside { flex-direction: row; flex-wrap: wrap; align-items: flex-end;
      gap: 10px 14px; margin-top: 12px; }
    .vw-aside .hafs-controls,
    .vw-aside .hafs-player { flex-direction: row; flex-wrap: wrap;
      align-items: flex-end; }
    .vw-aside .hafs-group select { width: auto; }
    .vw-below { margin-top: 10px; }
  }
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
  .hafs-stage img, .hafs-stage canvas { display: block; max-width: 100%;
    max-height: var(--panel-max-h); width: auto; height: auto; }
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
  /* in-app settings reopen-control (final-gate-3 #3): lives in the SIDEBAR
     FOOTER beside "Back to map" - NEVER over the banner corner glyph. */
  .side-foot { display: flex; align-items: center;
    justify-content: space-between; gap: 10px; padding: 13px 18px;
    border-top: 1px solid var(--border); }
  .settings-btn { display: inline-flex; align-items: center; gap: 7px;
    background: rgba(255,255,255,0.04); color: var(--muted);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 7px 11px; font: inherit; font-size: 12px; font-weight: 700;
    letter-spacing: 0.6px; text-transform: uppercase; cursor: pointer;
    transition: color 0.15s, border-color 0.15s; }
  .settings-btn:hover { color: var(--fg); border-color: var(--cat-accent); }
  .settings-btn svg { display: block; }
  .settings-pop { position: fixed; inset: 0; z-index: 60;
    display: flex; align-items: flex-start; justify-content: center;
    padding-top: 12vh; background: rgba(4,8,14,0.55); }
  .settings-pop[hidden] { display: none; }
  .settings-card { width: min(360px, calc(100vw - 32px));
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.5); }
  .settings-head { font-size: 12px; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }
  .settings-row { display: flex; flex-direction: column; gap: 8px; }
  .settings-lbl { font-size: 13px; font-weight: 700; color: var(--fg); }
  .seg-units { display: flex; gap: 0; }
  .seg-units button { flex: 1 1 0; background: var(--bg); color: var(--muted);
    border: 1px solid var(--border); padding: 9px 0; font: inherit;
    font-size: 13px; font-weight: 700; cursor: pointer; }
  .seg-units button:first-child { border-radius: 8px 0 0 8px; }
  .seg-units button:last-child { border-radius: 0 8px 8px 0;
    border-left: 0; }
  .seg-units button:not(:first-child):not(:last-child) { border-left: 0; }
  .seg-units button[aria-checked="true"] { background: var(--cat-accent);
    color: var(--cat-ink); border-color: var(--cat-accent); }
  .settings-note { margin: 14px 0 0; font-size: 11.5px; line-height: 1.5;
    color: var(--muted); }
  /* satellite speed + GIF export tools (final-gate-3 #5) */
  .sat-tools { align-items: flex-end; padding-top: 2px; }
  .sat-gif { align-self: flex-end; }
  .sat-gif[disabled] { opacity: 0.5; cursor: progress; }
  .sat-gif-prog { display: flex; align-items: center; gap: 8px;
    flex: 1 1 160px; font-size: 11.5px; color: var(--muted);
    font-variant-numeric: tabular-nums; }
  .sat-gif-prog[hidden] { display: none; }
  .sat-gif-bar { flex: 1 1 auto; height: 6px; border-radius: 3px;
    background: var(--bg); border: 1px solid var(--border);
    overflow: hidden; }
  .sat-gif-bar i { display: block; height: 100%; width: 0;
    background: var(--cat-accent); transition: width 0.15s linear; }
  /* floater-inactive note (auto-refresh #4): shown instead of implying a
     live feed once the floater stops producing frames. */
  .sat-inactive { margin: 8px 0 0; font-size: 11.5px; font-weight: 600;
    letter-spacing: 0.3px; color: #d2a93f; }
  .sat-inactive[hidden] { display: none; }
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
  /* --- Model guidance (Stage B; merged into the Models tab in Phase 3b, so
         the guidance SVGs are now scoped by id rather than by #sec-guidance) --- */
  #gtracks, #gintensity { width: 100%; height: auto; display: block;
    background: #101a2c; border-radius: 10px; }
  .g-legend { display: flex; flex-wrap: wrap; gap: 5px 14px; margin-top: 9px;
    font-size: 11.5px; color: var(--muted); font-weight: 600; align-items: center; }
  .g-legend .lg { display: inline-flex; align-items: center; gap: 5px; }
  .g-legend .lg b { color: var(--fg); }
  .g-legend .sw { width: 15px; height: 3px; border-radius: 2px; display: inline-block; }
  .g-ships-head { display: flex; flex-wrap: wrap; gap: 7px 10px; margin-bottom: 10px; }
  .g-chip { background: var(--navy-deep); border: 1px solid var(--border);
    border-radius: 8px; padding: 5px 9px; font-size: 12px; color: var(--muted);
    font-weight: 600; font-variant-numeric: tabular-nums; }
  .g-chip b { color: var(--fg); font-weight: 800; }
  .g-chip.ri b { color: var(--cat-accent); }
  .g-sm-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .g-sm { background: var(--navy-deep); border: 1px solid var(--border);
    border-radius: 9px; padding: 8px 8px 4px; }
  .g-sm .t { font-size: 11px; font-weight: 700; color: var(--fg); }
  .g-sm .v { font-size: 10.5px; color: var(--muted); font-weight: 600; }
  .g-sm svg { background: none; margin-top: 2px; }
  .g-ri { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 11.5px;
    font-variant-numeric: tabular-nums; }
  .g-ri th, .g-ri td { padding: 5px 6px; text-align: center;
    border-bottom: 1px solid var(--border); }
  .g-ri th { color: var(--muted); font-weight: 700; font-size: 10px;
    text-transform: uppercase; letter-spacing: .03em; }
  .g-ri td.rn { text-align: left; color: var(--fg); font-weight: 700; }
  .g-ri caption { text-align: left; font-size: 12px; font-weight: 800;
    color: var(--fg); padding-bottom: 6px; }
  @media (max-width: 720px) { .g-sm-grid { grid-template-columns: repeat(2, 1fr); } }
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
  /* maps-pass: the cone map is FULL-BLEED to the card edges (one
     continuous basemap, no panel-in-panel). The stage background is the
     OCEAN navy, so any meet-scaling letterbox blends into the ocean
     instead of reading as a second tone. */
  .adv-cone-stage { background: #101a2c; margin: 2px -14px 0;
    border-top: 1px solid var(--border); overflow: hidden;
    position: relative;
    /* maps-pass R4 #1: a DEFINITE height (respecting the 62vh no-scroll cap)
       so the SVG below can COVER it edge-to-edge and the fill extent has a
       real card aspect to measure. */
    height: clamp(340px, 62vh, 600px); }
  /* maps-pass R3 #2: the title lockup is an HTML overlay pinned to the
     panel's TOP-LEFT corner (the SVG is meet-scaled + CENTERED, so an
     in-SVG lockup floats inset from the card edge). Eyebrow + storm name on
     a dark backing; the panel <h3> is the canonical "Forecast cone" head. */
  /* maps-pass R4 #2: the box GROWS to its widest line (up to nearly the
     panel width); a long name steps its type down in JS so it never escapes
     the box. width:max-content keeps the box hugging its content. */
  .adv-lockup { position: absolute; top: 12px; left: 12px; z-index: 2;
    background: rgba(8,13,22,0.82); border: 1px solid rgba(44,58,82,0.55);
    border-left: 3px solid var(--ac-rail); border-radius: 8px;
    padding: 7px 13px 8px; pointer-events: none;
    width: max-content; max-width: calc(100% - 20px); }
  .adv-lockup[hidden] { display: none; }
  .adv-lockup .al-eyebrow { color: #8fa2bd; font-size: 11px;
    font-weight: 700; letter-spacing: 1.5px; white-space: nowrap; }
  .adv-lockup .al-name { color: #ffffff; font-size: 18px; font-weight: 800;
    letter-spacing: 0.5px; margin-top: 2px; white-space: nowrap; }
  /* maps-pass R5: track + swath corner furniture as HTML overlays (the cone's
     contained-box lockup treatment), pinned to the map's top-left corner in a
     FLEX STACK so the title + wind-field key never collide; the PACIFIC OCEAN
     watermark is dropped on these panels. */
  .map-stage { position: relative; line-height: 0; }
  .map-corner-tl { position: absolute; top: 12px; left: 12px; z-index: 2;
    display: flex; flex-direction: column; align-items: flex-start; gap: 8px;
    max-width: calc(100% - 24px); pointer-events: none; line-height: 1.25; }
  .map-lockup, .map-windkey, .map-stats { background: rgba(8,13,22,0.82);
    border: 1px solid rgba(44,58,82,0.55); border-radius: 8px;
    width: max-content; max-width: 100%; }
  .map-lockup, .map-stats { border-left: 3px solid var(--ac-rail); }
  .map-lockup { padding: 7px 13px 8px; }
  .map-lockup[hidden], .map-windkey[hidden], .map-stats[hidden] {
    display: none; }
  /* maps-pass R6: the CURRENT stats card, pinned BOTTOM-LEFT of .map-stage in
     its own flex stack - the same dark-box treatment as the title lockup, so
     it anchors to the panel corner instead of floating in map space. */
  .map-corner-bl { position: absolute; bottom: 12px; left: 12px; z-index: 2;
    display: flex; flex-direction: column; align-items: flex-start; gap: 8px;
    max-width: calc(100% - 24px); pointer-events: none; line-height: 1.3; }
  .map-stats { padding: 7px 12px 8px; }
  .map-stats .ms-h { color: #8fa2bd; font-size: 10px; font-weight: 700;
    letter-spacing: 1.2px; }
  .map-stats .ms-row { color: #cdd9ea; font-size: 12px; margin-top: 3px;
    white-space: nowrap; }
  .map-stats .ms-key { display: flex; gap: 11px; margin-top: 6px; }
  .map-stats .ms-k { display: inline-flex; align-items: center; gap: 4px;
    color: #b9c6da; font-size: 10.5px; }
  .map-stats .ms-mk { width: 9px; height: 9px; display: block; }
  .map-lockup .ml-eyebrow { color: #8fa2bd; font-size: 11px; font-weight: 700;
    letter-spacing: 1.5px; white-space: nowrap; }
  .map-lockup .ml-head { color: #ffffff; font-size: 19px; font-weight: 800;
    letter-spacing: 0.5px; margin-top: 2px; white-space: nowrap; }
  .map-lockup .ml-sub { color: #9fb0c8; font-size: 12px; font-weight: 700;
    letter-spacing: 0.8px; margin-top: 2px; white-space: nowrap; }
  .map-windkey { padding: 6px 11px 7px; display: flex; align-items: center;
    gap: 9px; }
  .map-windkey .mwk-h { color: #8fa2bd; font-size: 10px; font-weight: 700;
    letter-spacing: 1.2px; white-space: nowrap; }
  .map-windkey .mwk-tier { display: inline-flex; align-items: center;
    gap: 4px; font-size: 11.5px; font-weight: 800; }
  .map-windkey .mwk-ring { width: 9px; height: 9px; border-radius: 50%;
    border: 2px solid currentColor; box-sizing: border-box; }
  .map-windkey .mwk-u { color: #b9c6da; font-size: 10.5px; }
  /* Overview hero (final-gate-2 #1/#2): a storm-centered SST render
     from SOURCE data - the per-storm PNGs the poller bakes (storm at
     the EXACT pixel center, native 5 km CRW detail, house recipe +
     labeled isotherms) - with the big spinning category glyph and a
     base-layer picker. The PNG shares the panel's 16/9.2 aspect, so
     object-fit:cover is a 1:1 mapping and registration needs NO
     client crop math: the storm is always at 50%/50%.
     #2: the hero is a PANEL, not a poster - the overview column caps
     its width so the bug card + hero + W&P chart compose on one
     comfortable screen.
     FG-R3 #11: the Overview is now a TWO-COLUMN composition - LEFT =
     SST hero + W&P chart, RIGHT = track-history + wind-swath. The wipe
     widens to host both columns; each PANEL still caps small (widget,
     not poster). Mobile collapses to one column with an explicit stack
     order via CSS `order` (hero -> track -> W&P -> swath). */
  #sec-overview .wipe { max-width: 1180px; display: grid;
    grid-template-columns: 1fr 1fr; gap: 0 20px; align-items: start; }
  #sec-overview .ov-col { min-width: 0; }
  #sec-overview .ov-col .card:last-child { margin-bottom: 0; }
  .sst-hero { position: relative; overflow: hidden; border-radius: 8px;
    aspect-ratio: 16 / 9.2; background: #0a1019;
    border: 1px solid #2c3a52; }
  .sst-hero img { position: absolute; inset: 0; width: 100%;
    height: 100%; object-fit: cover; user-select: none;
    pointer-events: none; }
  /* PART D: lat/long lattice over the storm-centered CRW raster. The SST box
     is exactly 16:9.2 (== the container), so object-fit:cover does not crop and
     a linear lat/lon->px overlay registers. Reuses the cone graticule styling
     (.ac-graticule .grat-* below). Sits over the raster, under the title/scrim. */
  .sst-grat { position: absolute; inset: 0; width: 100%; height: 100%;
    pointer-events: none; }
  .sst-hero-layers { position: absolute; top: 10px; right: 10px; }
  .sst-hero-layers .hafs-seg { font-size: 10.5px; font-weight: 700;
    padding: 4px 10px; background: rgba(10,16,25,0.78);
    border-color: rgba(255,255,255,0.22); color: #cdd9ea; }
  .sst-hero-layers .hafs-seg.active { background: var(--cat-accent);
    color: var(--cat-ink); border-color: var(--cat-accent); }
  .sst-hero-scrim { position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(155deg, rgba(8,12,20,0.62) 0%,
      rgba(8,12,20,0.18) 30%, transparent 55%); }
  .sst-hero-title { position: absolute; top: 14px; left: 16px;
    padding-left: 11px; }
  .sst-hero-title .hero-rail { position: absolute; left: 0; top: 2px;
    bottom: 2px; width: 3px; border-radius: 1.5px;
    background: var(--ac-rail); }
  .sst-hero-title .hero-eyebrow { color: #aebdd4; font-size: 11px;
    font-weight: 700; letter-spacing: 1.6px; }
  .sst-hero-title .hero-head { color: #ffffff; font-size: 19px;
    font-weight: 800; letter-spacing: 0.6px; margin-top: 2px;
    text-shadow: 0 1px 6px rgba(0,0,0,0.5); }
  .sst-hero-title .hero-sub { color: #cdd9ea; font-size: 12.5px;
    font-weight: 700; letter-spacing: 1.1px; margin-top: 2px; }
  .sst-hero-glyph { position: absolute; left: 50%; top: 50%;
    width: 120px; height: 120px;
    margin: -60px 0 0 -60px; pointer-events: none;
    filter: drop-shadow(0 3px 10px rgba(0,0,0,0.55)); }
  .sst-hero-glyph svg { width: 100%; height: 100%; }
  /* S4-AD2 #2/#3: the map reads as a SURFACE - ocean lifted one clear
     step above the panel chrome, hairline inset border, land one step
     above the ocean, graticule legible-but-subtle. */
  .ac-ocean-fill { fill: #101a2c; }
  /* maps-pass: the inner hairline frame is RETIRED (it boxed the map into
     a nested panel-in-panel); the card edge is the only frame now. */
  .ac-frame { fill: none; stroke: none; }
  /* maps-pass basemap canon (ne_10m): LIGHT-GRAY land, clearly lighter
     than the #101a2c ocean so landmasses stand out, with NO stroke (so
     abutting ne_10m country fills merge into one continuous landmass with
     no interior borders); the coast is a SEPARATE white stroke from the
     ne_10m coastline polylines. Land paints OVER the graticule, so the
     graticule only shows on open water. */
  .ac-land { fill: #a7b2c4; stroke: none; }
  /* coast + borders are FINE HAIRLINES (phase-4 v2 #3), retuned together: the
     old 2.6 coast / 1.4 border / 0.8 state read as heavy clutter at the cone
     auto-fit zoom. Now coast 1.3, country 0.7, state 0.5 - clean lines that
     read without burying the subject. (The borders are clipped to land in the
     bake so they never run into the water - cyclolab_basemap._clip_lines_to_land.) */
  .ac-coast { fill: none; stroke: #ffffff; stroke-width: 1.3;
    stroke-linejoin: round; stroke-linecap: round; }
  /* country + state borders are SLATE, not white (phase-4 C): a bright white
     internal border fought the white coast and the subject layer for
     attention; slate keeps them legible over the light land yet recessive -
     furniture, not subject. ONE shared rule -> every CycloLab map (cone /
     guidance track / overview track+swath) inherits it, no per-map fork. */
  .ac-border { fill: none; stroke: rgba(71,85,105,0.92);
    stroke-width: 0.7; stroke-linejoin: round; stroke-linecap: round; }
  /* state/province boundaries (ne_10m admin_1). v3: the v2 thinning made these
     too faint - BOLDER now (width 0.5 -> 0.9, opacity 0.60 -> 0.9) so the
     landfall state lines read clearly. The country border + coast stay thin
     (the "too thick" feedback still holds for those - this is ONLY the states). */
  .ac-state { fill: none; stroke: rgba(71,85,105,0.9);
    stroke-width: 0.9; stroke-linejoin: round; stroke-linecap: round; }
  /* maps-pass R3 #3: a CASING/HALO graticule - a dark hairline UNDER a light
     line - so every line reads over BOTH the light-gray land AND the dark
     ocean (a flat light line vanished over the light land). Labels get the
     same dark-casing paint-order stroke and sit on all four edges. */
  .ac-graticule .grat-cas { stroke: rgba(8,14,26,0.55); stroke-width: 2.6; }
  .ac-graticule .grat-lin { stroke: rgba(228,236,250,0.6); stroke-width: 1; }
  .ac-graticule .grat-lab { fill: #e7eef9; font-size: 12.5px;
    font-variant-numeric: tabular-nums; paint-order: stroke;
    stroke: rgba(8,14,26,0.82); stroke-width: 2.6; stroke-linejoin: round; }
  /* (.ac-ocean watermark + .ac-title SVG lockup retired R2/R5 - the cone +
     track/swath titles are HTML overlays: .adv-lockup / .map-lockup.) */
  /* maps-pass R4 #1: the cone SVG COVERS its definite-height stage (slice),
     full-bleed; the intensity chart keeps its natural aspect (meet). */
  #advcone { display: block; width: 100%; height: 100%; }
  #intensity { display: block; width: 100%; height: auto;
    max-height: var(--panel-max-h); }
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
  /* ---- Phase 4: coastal watches/warnings overlay control + legend ----
     Official NHC data (windWatchesWarnings KMZ); no derived disclosure.
     ART-GATED palette - canonical NHC TCWW colors, awaiting sign-off. */
  .ac-ww-bar { display: flex; flex-wrap: wrap; align-items: center;
    gap: 6px 16px; margin: 9px 0 0; }
  .ac-ww-toggle { display: inline-flex; align-items: center; gap: 6px;
    cursor: pointer; font-size: 12px; font-weight: 700; color: #cfe0f2;
    letter-spacing: 0.3px; user-select: none; }
  .ac-ww-toggle input { accent-color: #9fc6f5; cursor: pointer; }
  .ac-ww-legend { display: flex; flex-wrap: wrap; gap: 5px 14px;
    align-items: center; font-size: 11.5px; color: var(--muted);
    font-weight: 600; }
  .ac-ww-legend .ww-lg { display: inline-flex; align-items: center; gap: 5px; }
  .ac-ww-legend .ww-sw { width: 16px; height: 4px; border-radius: 2px;
    display: inline-block; box-shadow: 0 0 0 1px rgba(6,12,22,0.7); }
  .ac-ww .ww-cas { fill: none; stroke: rgba(6,12,22,0.72);
    stroke-linecap: round; stroke-linejoin: round; }
  .ac-ww .ww-lin { fill: none; stroke-linecap: round; stroke-linejoin: round; }
  /* (the v2 inland county/zone FILL layer was removed in v3 - the W/W presence
     is now only the coastal breakpoint lines above.) */
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

  /* ---- Overview plots (FG-R3 #7 track history + #8 wind swath): two
     art-directed client-side panels that REUSE the cone's map furniture
     (.ac-ocean-fill / .ac-land / .ac-graticule / .ac-coast / .ac-border /
     .ac-frame) and add panel-specific chrome. The track dots + colorbar are
     SSHS-anchored category bands (the canonical 7 hues), the color breaking
     on the Saffir-Simpson thresholds (34/64/83/96/113/137 kt). */
  .tp-track { fill: none; stroke: rgba(150,180,220,0.42);
    stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }
  .tp-dot { stroke: rgba(8,12,20,0.55); stroke-width: 1; }
  .tp-dot.tp-now { stroke: #ffffff; stroke-width: 1.6;
    filter: drop-shadow(0 0 6px currentColor); }
  .tp-cbar-tick { fill: #ffffff; font-size: 11px;
    font-variant-numeric: tabular-nums; paint-order: stroke;
    stroke: rgba(7,12,22,0.85); stroke-width: 2px;
    stroke-linejoin: round; }
  .tp-cbar-frame { fill: none; stroke: #3a4d6e; stroke-width: 1; }
  .tp-field-lbl { font-size: 11px; font-weight: 700;
    font-variant-numeric: tabular-nums; paint-order: stroke;
    stroke: rgba(6,11,20,0.92); stroke-width: 2.6px;
    stroke-linejoin: round; }
  .tp-radii-lab { fill: #aebdd4; font-size: 11px; font-weight: 600; }
  .sw-caption-d { display: inline-block; margin: 8px 0 0;
    font-size: 10.5px; font-weight: 700; letter-spacing: 0.8px;
    color: #9fc6f5; text-transform: uppercase; }
  .sw-empty { color: var(--muted); font-size: 12.5px; padding: 26px 14px;
    text-align: center; line-height: 1.6; }

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
    .sec-btn { flex: 1 1 20%; justify-content: center; padding: 12px 3px;
      min-height: 52px; font-size: 10px; border-left: 0;
      border-top: 3px solid transparent; }
    .sec-btn.active { border-left: 0; border-top-color: var(--cat-accent); }
    .side-foot { padding: 8px 12px; }
    .stage { padding: 4px 14px 86px; }
    /* FG-R3 #11: mobile Overview = ONE column, explicit stack order
       hero -> track -> W&P -> swath (the DOM stays logical: left col =
       hero+chart, right col = track+swath; CSS `order` interleaves). */
    #sec-overview .wipe { grid-template-columns: 1fr; gap: 0; }
    #sec-overview .ov-col { display: contents; }
    #sec-overview #card-hero  { order: 1; }
    #sec-overview #card-track { order: 2; }
    #sec-overview #card-wp    { order: 3; }
    #sec-overview #card-swath { order: 4; }
    /* FG-R2 review notes (two rounds): the hero lockup must read as a
       small corner signature on a ~366px panel. The eyebrow drops on
       phones - same precedent as the banner's mobile compaction - and
       head/sub shrink to caption scale. */
    .sst-hero-title { top: 9px; left: 11px; padding-left: 7px; }
    .sst-hero-title .hero-eyebrow { display: none; }
    .sst-hero-title .hero-head { font-size: 10.5px;
      letter-spacing: 0.5px; margin-top: 0; }
    .sst-hero-title .hero-sub { font-size: 8.5px;
      letter-spacing: 0.6px; margin-top: 1px; }
    .sst-hero-title .hero-rail { width: 2px; }
    .sst-hero-layers .hafs-seg { font-size: 9px; padding: 2px 7px; }
  }

  @media (prefers-reduced-motion: reduce) {
    .loader, .loader::after, .loader .scan, .loader .word,
    .loader .word span.ch, .banner.shine::after, .banner.xfade .old-ramp,
    .sec.active .wipe, .draw path.series, .draw .fill {
      animation-duration: 0.001s !important;
      animation-delay: 0s !important; }
    .banner .glyph .spin, .loader .eye .spin { animation: none !important; }
    .odo.odo-swap { animation: none !important; }
  }

  /* Right-click "copy as PNG" affordance (overview plots) + the result toast. */
  .cl-copyable { cursor: context-menu; -webkit-touch-callout: none;
    -webkit-user-select: none; user-select: none; }  /* svg, img, stage divs;
    callout:none suppresses the iOS image menu during a long-press copy */
  .cl-toast { position: fixed; left: 50%; bottom: 26px;
    transform: translateX(-50%) translateY(10px);
    background: #0d1626; color: #eaf2ff; border: 1px solid #2b3b57;
    border-radius: 9px; padding: 9px 16px; z-index: 9999;
    font: 600 13px/1.2 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    box-shadow: 0 8px 28px rgba(0, 0, 0, .45); opacity: 0; pointer-events: none;
    transition: opacity .18s ease, transform .18s ease; }
  .cl-toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  @media (prefers-reduced-motion: reduce) { .cl-toast { transition: opacity .1s; } }
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
        <!-- Stage C: invests show a GIANT RED X here instead of the spinning
             cyclone glyph + category label (CSS toggles on html[data-invest]). -->
        <g class="invest-x"><path d="M -22,-22 L 22,22 M 22,-22 L -22,22"
          fill="none" stroke="#ff3b3b" stroke-width="11"
          stroke-linecap="round"/></g>
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
        <!-- Stage C: NHC formation-chance pill (invests only; populated by
             loadFormation() from cyclolab/{sid}/formation.json). -->
        <div class="formation-pill" id="formation-pill" hidden></div>
      </div>
    </div>
    <div class="bug-body">
      <div class="heroes">
        <div class="hero">
          <span class="hero-val"><span class="odo" id="odo-vmax" aria-label="__VMAX_A11Y__">__VMAX_ODO__</span><span class="unit" id="vmax-unit">kt</span></span>
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
      <button class="sec-btn" data-sec="recon">Recon</button>
      <button class="sec-btn" data-sec="models">Models</button>
      <button class="sec-btn" data-sec="advisories">Advisories</button>
    </nav>
    <div class="side-foot">
      <button type="button" id="settings-btn" class="settings-btn"
              title="Settings" aria-label="Settings"
              aria-haspopup="dialog" aria-expanded="false">
        <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
          <path fill="currentColor" fill-rule="evenodd" d="M9.81,4.62 L9.71,1.24 L14.29,1.24 L14.19,4.62 L15.67,5.23 L17.99,2.77 L21.23,6.01 L18.77,8.33 L19.38,9.81 L22.76,9.71 L22.76,14.29 L19.38,14.19 L18.77,15.67 L21.23,17.99 L17.99,21.23 L15.67,18.77 L14.19,19.38 L14.29,22.76 L9.71,22.76 L9.81,19.38 L8.33,18.77 L6.01,21.23 L2.77,17.99 L5.23,15.67 L4.62,14.19 L1.24,14.29 L1.24,9.71 L4.62,9.81 L5.23,8.33 L2.77,6.01 L6.01,2.77 L8.33,5.23 Z M8.3,12.0 A3.7,3.7 0 1 0 15.7,12.0 A3.7,3.7 0 1 0 8.3,12.0 Z"/></svg>
        <span>Settings</span>
      </button>
      <a class="back-map" href="/global_tracks.html">← Back to map</a>
    </div>
  </aside>

  <main class="stage">
    <section class="sec active" id="sec-overview">
      <div class="wipe">
        <div class="card" id="card-map" style="grid-column:1/-1">
          <h3>Storm map</h3>
          <div id="overview-map"></div>
          <div class="note" id="overview-map-note">Interactive track &amp; layers. Satellite and model imagery stack in as layers when published.</div>
        </div>
        <div class="ov-col ov-left">
        <div class="card" id="card-hero">
          <div class="sst-hero" id="sst-hero">
            <img id="sst-hero-img" alt="Sea-surface temperature around the storm"
                 draggable="false">
            <svg class="sst-grat ac-graticule" id="sst-grat"
                 preserveAspectRatio="none" aria-hidden="true"></svg>
            <div class="sst-hero-scrim"></div>
            <div class="sst-hero-title">
              <div class="hero-rail"></div>
              <div class="hero-eyebrow">TRIPLE-A-TROPICS &middot; CycloLab</div>
              <div class="hero-head" id="sst-hero-head">SEA SURFACE TEMPERATURE</div>
              <div class="hero-sub" id="sst-hero-sub"></div>
            </div>
            <div class="hafs-seg-group sst-hero-layers" id="sst-hero-layers"
                 role="group" aria-label="Base layer"></div>
            <div class="sst-hero-glyph" id="sst-hero-glyph"></div>
          </div>
          <div class="note" id="sst-hero-note"></div></div>
        <div class="card" id="card-wp"><h3>Wind &amp; pressure</h3>
          <svg id="chart" viewBox="0 0 1000 320"
               preserveAspectRatio="xMidYMid meet"></svg></div>
        </div>
        <div class="ov-col ov-right">
        <div class="card" id="card-track">
          <div class="map-stage">
            <div class="map-corner-tl">
              <div class="map-lockup" id="trackplot-lockup" hidden>
                <div class="ml-eyebrow">TRIPLE-A-TROPICS · CycloLab</div>
                <div class="ml-head" id="trackplot-lockup-head">TRACK
                  HISTORY</div>
                <div class="ml-sub" id="trackplot-lockup-sub"></div>
              </div>
              <div class="map-windkey" id="trackplot-windkey" hidden></div>
            </div>
            <div class="map-corner-bl">
              <div class="map-stats" id="trackplot-stats" hidden>
                <div class="ms-h">CURRENT</div>
                <div class="ms-row" id="trackplot-stats-vmax"></div>
                <div class="ms-row" id="trackplot-stats-pmin"></div>
                <div class="ms-row" id="trackplot-stats-ace"></div>
                <div class="ms-key">
                  <span class="ms-k"><svg class="ms-mk" viewBox="0 0 12 12">
                    <circle cx="6" cy="6" r="4.6" fill="#cdd9ea"/></svg>trop</span>
                  <span class="ms-k"><svg class="ms-mk" viewBox="0 0 12 12">
                    <rect x="1.6" y="1.6" width="8.8" height="8.8"
                      fill="#cdd9ea"/></svg>sub</span>
                  <span class="ms-k"><svg class="ms-mk" viewBox="0 0 12 12">
                    <path d="M6 1.4 L10.6 10.6 L1.4 10.6 Z"
                      fill="#cdd9ea"/></svg>non-trop</span>
                </div>
              </div>
            </div>
            <svg id="trackplot" viewBox="0 0 1000 620"
                 preserveAspectRatio="xMidYMid meet"></svg>
          </div>
          <div class="note" id="trackplot-note"></div></div>
        <div class="card" id="card-swath">
          <div class="map-stage">
            <div class="map-corner-tl">
              <div class="map-lockup" id="swathplot-lockup" hidden>
                <div class="ml-eyebrow">TRIPLE-A-TROPICS · CycloLab</div>
                <div class="ml-head" id="swathplot-lockup-head">WIND
                  HISTORY</div>
                <div class="ml-sub" id="swathplot-lockup-sub"></div>
              </div>
            </div>
            <svg id="swathplot" viewBox="0 0 1000 620"
                 preserveAspectRatio="xMidYMid meet"></svg>
          </div>
          <div class="sw-empty" id="swath-empty" style="display:none"></div>
          <span class="sw-caption-d" id="swath-derived" hidden>Derived
            product</span>
          <div class="note" id="swathplot-note"></div>
          <details class="adv-method" id="swath-method" hidden>
            <summary>How is this drawn?</summary>
            <div id="swath-method-body"></div></details></div>
        </div>
      </div>
    </section>
    <section class="sec" id="sec-satellite"><div class="wipe">
      <h2 class="sec-title">Satellite</h2>
      <div class="card vw-grid" id="sat-card" tabindex="0">
        <div class="hafs-stage" id="sat-stage">
          <img id="sat-img" alt="Storm floater satellite frame">
          <canvas id="sat-canvas" style="display:none" role="img"
                  aria-label="Storm floater satellite frame"></canvas>
          <div id="sat-status" class="hafs-statusbox">
            <div class="hafs-spinner"></div><span>Loading…</span></div>
        </div>
        <div class="vw-aside">
          <div class="hafs-controls">
            <div class="hafs-group"><label>Band</label>
              <div id="sat-bands" class="hafs-seg-group" role="group"
                   aria-label="Satellite band"></div></div>
          </div>
          <div class="hafs-player">
            <button id="sat-step-back" class="hafs-btn" type="button"
                    title="Previous frame (&#8592;)">&#9664;</button>
            <button id="sat-play" class="hafs-btn hafs-play" type="button"
                    title="Play / pause (space)">&#9654; Play</button>
            <button id="sat-step-fwd" class="hafs-btn" type="button"
                    title="Next frame (&#8594;)">&#9654;</button>
            <div class="hafs-readout"><span id="sat-time">&#8212;</span>
              <span id="sat-band-label"></span></div>
          </div>
          <div class="hafs-controls sat-tools">
            <div class="hafs-group"><label>Speed</label>
              <div id="sat-speed" class="hafs-seg-group" role="group"
                   aria-label="Playback speed"></div></div>
            <button id="sat-gif" class="hafs-btn sat-gif" type="button"
                    title="Export the loaded frames as a GIF">
              &#8681; GIF</button>
            <div id="sat-gif-prog" class="sat-gif-prog" hidden
                 role="status" aria-live="polite">
              <div class="sat-gif-bar"><i></i></div>
              <span class="sat-gif-pct">0%</span></div>
          </div>
        </div>
        <div class="vw-below">
          <input id="sat-scrub" type="range" min="0" max="0" value="0"
                 step="1" aria-label="Frame time"></div>
        <div id="sat-empty" class="stub" style="display:none">No floater
          imagery for this storm right now.</div>
        <div id="sat-inactive" class="sat-inactive" hidden
             role="status" aria-live="polite"></div>
        <p class="hafs-caption">GOES floater imagery centered on the storm,
          newest frame first. Frames land every few minutes while the
          floater is active.</p>
      </div>
    </div></section>
    <section class="sec" id="sec-recon"><div class="wipe">
      <h2 class="sec-title">Recon</h2>
      <div class="card" id="recon-viewer" tabindex="0">
        <div id="recon-status" class="hafs-statusbox">
          <div class="hafs-spinner"></div><span>Loading recon&#8230;</span></div>
      </div>
      <p class="hafs-caption">Hurricane-hunter aircraft observations (HDOB
        flight-level + SFMR surface wind, vortex fixes, dropsondes) for this
        storm. SFMR is unreliable in heavy rain and at very high wind; obs are
        point-in-time.</p>
    </div></section>
    <section class="sec" id="sec-models"><div class="wipe">
      <h2 class="sec-title">Models</h2>
      <div class="card vw-grid" id="cl-hafs-root" tabindex="0">
        <div id="cl-hafs-stage" class="hafs-stage">
          <img id="cl-hafs-img" alt="HAFS forecast frame for this storm">
          <div id="cl-hafs-status" class="hafs-statusbox">
            <div class="hafs-spinner"></div><span>Loading&#8230;</span></div>
          <div id="cl-hafs-buffer"></div>
        </div>
        <div class="vw-aside">
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
        </div>
        <div class="vw-below">
          <div id="cl-hafs-hours" class="hafs-hours" role="group"
               aria-label="Forecast hour"></div>
        </div>
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
      <!-- Model guidance (merged into Models - Phase 3b): named storms get
           HAFS above + guidance here; invests + PTCs get guidance with HAFS
           gracefully absent (cl-hafs-empty). -->
      <div class="card">
        <h3>Model forecast tracks</h3>
        <svg id="gtracks" viewBox="0 0 1000 560"
             preserveAspectRatio="xMidYMid meet" role="img"
             aria-label="Model forecast track guidance"></svg>
        <div class="g-legend" id="gtracks-legend"></div>
        <p class="hafs-caption">Operational track aids, NHC ATCF aid_public.
          Colored by each model's peak forecast wind (SSHWS category).
          Consensus aids (TVCN, HCCA) are drawn heavier.</p>
        <div id="gtracks-empty" class="stub" style="display:none">No model
          guidance for this storm yet.</div>
      </div>
      <div class="card">
        <h3>Model forecast intensity</h3>
        <svg id="gintensity" viewBox="0 0 1000 380"
             preserveAspectRatio="xMidYMid meet" role="img"
             aria-label="Model forecast intensity guidance"></svg>
        <div class="g-legend" id="gintensity-legend"></div>
        <p class="hafs-caption">Intensity aids vs forecast hour over the SSHWS
          category bands. Regional hurricane models are emphasized; the global
          and statistical aids are drawn lighter.</p>
        <div id="gintensity-empty" class="stub" style="display:none"></div>
      </div>
      <div class="card">
        <h3>SHIPS output diagram</h3>
        <div id="gships-root"></div>
        <p class="hafs-caption">Statistical Hurricane Intensity Prediction
          Scheme: environment, rapid-intensification probabilities, annularity.</p>
      </div>
    </div></section>
    <section class="sec" id="sec-advisories"><div class="wipe">
      <h2 class="sec-title">Advisories</h2>
      <div class="card">
        <h3>Forecast cone</h3>
        <div class="adv-cone-stage">
          <div class="adv-lockup" id="advcone-lockup" hidden>
            <div class="al-eyebrow">TRIPLE-A-TROPICS · CycloLab</div>
            <div class="al-name" id="advcone-lockup-name"></div>
          </div>
          <svg id="advcone" viewBox="0 0 1000 620"
               preserveAspectRatio="xMidYMid meet" role="img"
               aria-label="Forecast track and uncertainty cone"></svg>
        </div>
        <div class="ac-ww-bar" id="advcone-ww" hidden>
          <label class="ac-ww-toggle"><input type="checkbox" id="advcone-ww-chk" checked>
            <span>Watches &amp; warnings</span></label>
          <div class="ac-ww-legend" id="advcone-ww-legend"></div>
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
        <pre id="advtext" class="advtext">(loading advisory data…)</pre>
      </div>
    </div></section>
  </main>
</div>

<!-- in-app settings (final-gate-3 #3): reopen the wind-unit choice from
     the gear; an extensible dialog, units are its first control. -->
<div class="settings-pop" id="settings-pop" hidden role="dialog"
     aria-label="CycloLab settings" aria-modal="false">
  <div class="settings-card">
    <div class="settings-head">Settings</div>
    <div class="settings-row">
      <div class="settings-lbl">Wind units</div>
      <div class="seg-units" id="settings-units" role="radiogroup"
           aria-label="Wind units"></div>
    </div>
    <div class="settings-row">
      <div class="settings-lbl">Map time</div>
      <div class="seg-units" id="settings-maptime" role="radiogroup"
           aria-label="Map time mode"></div>
    </div>
    <p class="settings-note">Display only. Agency forecasts are issued in
      knots; other units are converted here.</p>
  </div>
</div>

<script>
(function () {
  "use strict";
  var SID = "__SID__";
  var IS_INVEST = __IS_INVEST__;     // Stage C: grey/red-X subset page
  var IS_PTC = __IS_PTC__;           // Potential Tropical Cyclone: grey/red-X
                                     // identity but KEEPS cone/advisories/Models
  var SPAWN_SID = "__SPAWN_SID__";   // PTC: sid of the invest it spawned (the
                                     // NHC TWO formation odds live there)
  // ---- live PTC identity (durable) -----------------------------------------
  // The PTC "dress" (grey scheme, red-X glyph, formation-chance pill, hidden
  // ACE row) is a TRANSIENT pre-genesis state, NOT a value frozen at page
  // birth. NHC designates a disturbance as a Potential Tropical Cyclone, then
  // NAMES it the moment it becomes a TS+ - at which point the page must SHED
  // the dress and wear the real category (and, the reverse, re-wear it if a
  // system ever drops back). We re-evaluate this EVERY poll off the LIVE feed
  // (is_ptc + name + current_category), never off __IS_PTC__ alone. IS_PTC is
  // a `var` so setPtc() can flip it live; PTC_BAKED keeps the birth value as a
  // feed-omitted fallback.
  var PTC_BAKED = IS_PTC;
  // NHC's spelled-out designation numbers ("ONE".."FIFTY-NINE") - the
  // placeholder name a depression/PTC carries before it is named. A real name
  // (ARTHUR) is none of these.
  var NUMBER_NAME = (function () {
    var ones = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN",
      "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN",
      "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"];
    var tens = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY"];
    var s = {};
    for (var n = 1; n < 60; n++) {
      s[n < 20 ? ones[n]
        : tens[Math.floor(n / 10)] + (n % 10 ? "-" + ones[n % 10] : "")] = true;
    }
    return s;
  })();
  function isNamedTC(storm) {
    // A genuine named/designated TC: TS-or-stronger SSHWS AND a REAL NHC name
    // (not the "ONE"/"TWO" designation placeholder, an invest, or the raw sid).
    var cat = (storm && storm.current_category) || "";
    if (cat !== "TS" && !/^C[1-5]$/.test(cat)) return false;
    var nm = ((storm && storm.name) || "").trim().toUpperCase();
    if (!nm || !/^[A-Z][A-Z'\- ]*$/.test(nm)) return false;   // letters -> a name
    if (NUMBER_NAME[nm]) return false;                        // "ONE".."FIFTY-NINE"
    return nm !== "INVEST" && nm !== "UNNAMED" && nm !== "NAMELESS";
  }
  function ptcNow(storm) {
    // A named TS+ system is, by definition, no longer "potential" - veto any
    // feed/bake lag (the operational b-deck NATURE can trail the classification).
    if (isNamedTC(storm)) return false;
    // Otherwise trust the LIVE feed's flag; fall back to the page-birth value
    // only when the feed omits it (older feeds / a fetch gap).
    return (storm && storm.is_ptc != null) ? !!storm.is_ptc : PTC_BAKED;
  }
  function setPtc(on, storm) {
    // Idempotent: act ONLY on a true transition. Returns whether the identity
    // flipped, so apply() can force the plots to re-render (their titles/colors
    // follow the identity, and a name->TC relabel can land on the same fix).
    if (on === IS_PTC) return false;
    IS_PTC = on;
    if (on) document.documentElement.setAttribute("data-ptc", "");
    else document.documentElement.removeAttribute("data-ptc");
    // Force setCategory() to FULLY re-apply (ramp + glyph letter + Category
    // hero + type word): the baked data-cat can already equal the live category
    // (a PTC with TS winds), so setCategory's no-op guard would otherwise leave
    // the frozen "PTC" dress. apply() calls setCategory right after us, by which
    // point IS_PTC is updated and these labels resolve to the real category.
    curCat = null;
    var pill = document.getElementById("formation-pill");
    if (pill) {
      if (on) { loadFormation(); }       // reverse edge: re-arm the chance pill
      else { pill.hidden = true; pill.innerHTML = ""; }
    }
    return true;
  }
  var FEED_URL = "__FEED_URL__";
  var ADV_URL = "__ADV_URL__";
  // per-storm SST hero layer base (final-gate-2 #1): meta.json +
  // {layer}.png live under it, written by the poller's SST hero writer.
  var SST_BASE = "__SST_BASE__";
  var ENDED = __ENDED__;
  var BASIN = "__BASIN__";
  var HAFS_ID = "__HAFS_ID__";        // storm_ids join: 01e
  var FLOATER_ID = "__ATCF_LONG__";   // storm_ids join: ep012026 / wp072026
  var FLOATER_SLUG = "__FLOATER_SLUG__"; // floater index slug: wp07 / ep01 / wp91
  var CDN = "https://cdn.triple-a-tropics.com";
  // Per-basin published intensity-error entry (null = the honesty-guard
  // case: a labeled "no published statistics" panel, never a borrowed
  // or invented envelope).
  var INTENSITY_ERR = __INTENSITY_ERR__;
  // Storm-window basemap (S4-AD1 #2): vendored Natural Earth land,
  // clipped + antimeridian-normalized at bake time. No runtime fetch.
  var BASEMAP = __BASEMAP__;
  // v3 dedup: the coast is DERIVED from the land rings (their boundary MINUS the
  // window-edge segments) instead of being stored in the bake - so the GSHHG
  // high-res coast costs nothing beyond the land it already shares vertices
  // with. Byte-for-byte mirror of cyclolab_basemap.coast_from_land / _ring_coast.
  // Computed ONCE here; all three basemap render sites draw BASEMAP_COAST.
  function coastFromLand(land, win) {
    if (!win || !land) return [];
    var la0 = win[0], la1 = win[1], lo0 = win[2], lo1 = win[3], eps = 0.02;
    function onEdge(a, b) {
      return (Math.abs(a[0] - lo0) < eps && Math.abs(b[0] - lo0) < eps) ||
             (Math.abs(a[0] - lo1) < eps && Math.abs(b[0] - lo1) < eps) ||
             (Math.abs(a[1] - la0) < eps && Math.abs(b[1] - la0) < eps) ||
             (Math.abs(a[1] - la1) < eps && Math.abs(b[1] - la1) < eps);
    }
    var out = [];
    land.forEach(function (ring) {
      var n = ring.length, cur = [];
      for (var i = 0; i < n; i++) {
        var a = ring[i], b = ring[(i + 1) % n];
        if (a[0] === b[0] && a[1] === b[1]) continue;
        if (onEdge(a, b)) {
          if (cur.length >= 2) out.push(cur);
          cur = [];
        } else if (cur.length &&
                   cur[cur.length - 1][0] === a[0] &&
                   cur[cur.length - 1][1] === a[1]) {
          cur.push(b);
        } else {
          if (cur.length >= 2) out.push(cur);
          cur = [a, b];
        }
      }
      if (cur.length >= 2) out.push(cur);
    });
    return out;
  }
  var BASEMAP_COAST = coastFromLand(BASEMAP.land, BASEMAP.window);
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

  // ---- display settings (final-gate-3 #3): WIND UNITS ---------------------
  // DISPLAY-ONLY conversion - the canonical feeds stay in knots; agency
  // forecasts ARE issued in knots (the methodology panels say so). The
  // setting is read on boot from ?units= (the launch dialog's hand-off)
  // or localStorage, persisted, and changeable from the in-app gear.
  // Extensible: the framework is a settings object, units are its first
  // member.
  var WIND_UNITS = {
    kt:  { label: "kt",   conv: function (kt) { return kt; } },
    mph: { label: "mph",  conv: function (kt) { return kt * 1.1507794; } },
    kmh: { label: "km/h", conv: function (kt) { return kt * 1.852; } }
  };
  var SETTINGS_KEY = "cyclolab:settings";
  var settings = { windUnits: "kt", mapTime: "synced" };
  function loadSettings() {
    var s = {};
    try { s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") || {}; }
    catch (e) { s = {}; }
    // a ?units= launch param (from the pre-launch dialog) wins for THIS
    // load and is then persisted as the remembered choice.
    var q = null;
    try { q = new URLSearchParams(location.search).get("units"); }
    catch (e2) { q = null; }
    var u = (q && WIND_UNITS[q]) ? q
          : (WIND_UNITS[s.windUnits] ? s.windUnits : "kt");
    settings.windUnits = u;
    settings.mapTime = (s.mapTime === "independent") ? "independent" : "synced";
    if (q) saveSettings();
  }
  function saveSettings() {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); }
    catch (e) {}
  }
  function windUnitLabel() { return WIND_UNITS[settings.windUnits].label; }
  function unitsSourceNote() {
    // methodology honesty (final-gate-3 #3): agency forecasts ARE in
    // knots; when the user views another unit, say the conversion is
    // ours and display-only.
    if (settings.windUnits === "kt") return "";
    return " Agency forecasts are issued in knots; the " +
      windUnitLabel() + " values shown here are a display-only " +
      "conversion.";
  }
  function windNum(kt) {
    if (kt == null || isNaN(kt)) return null;
    return WIND_UNITS[settings.windUnits].conv(+kt);
  }
  // agency-convention rounding (final-gate-3 #4): CONVERTED wind values
  // (mph / km-h) ALWAYS round to the nearest 5 - NHC-style, so every
  // converted display ends in 0 or 5 (30 kt -> 35, 64 kt -> 75, 100 kt ->
  // 115 mph). kt stays the RAW advisory value (nearest 1; the feeds are
  // already issued in 5-kt steps). This applies EVERYWHERE winds display
  // because every wind surface routes through windDisp().
  function round5(v) { return Math.round(v / 5) * 5; }
  function windDisp(kt) {            // rounded display number (string)
    var v = windNum(kt);
    if (v == null) return "—";
    return String(settings.windUnits === "kt" ? Math.round(v) : round5(v));
  }
  function setWindUnits(u) {
    if (!WIND_UNITS[u] || u === settings.windUnits) return;
    settings.windUnits = u;
    saveSettings();
    rerenderUnits();
    syncSettingsUI();
  }
  function setMapTime(m) {
    var mode = (m === "independent") ? "independent" : "synced";
    settings.mapTime = mode;
    saveSettings();
    if (clMap && clMap.setTimeMode) clMap.setTimeMode(mode);
    syncSettingsUI();
  }
  function rerenderUnits() {
    // DISPLAY-ONLY: re-render every wind surface from the retained data.
    // apply() refreshes the hero + vitals; the W&P chart and the
    // advisories tab are gated on a NEW fix, so re-render them directly.
    try { if (lastStorm) { apply(lastStorm); renderChart(lastStorm); } }
    catch (e) {}
    // FG-R3 #7/#8: the two Overview plots carry units-aware colorbar +
    // legend labels - convert them directly on a unit change (same as the
    // W&P chart; apply() is gated on a NEW fix so it won't redraw them).
    try { if (lastStorm) renderTrackPlot(lastStorm); } catch (e3) {}
    try { if (lastStorm) renderSwathPlot(lastStorm); } catch (e4) {}
    try { if (inited.advisories && advFull) renderAdvTab(); } catch (e2) {}
  }
  function syncSettingsUI() {
    var host = document.getElementById("settings-units");
    if (host) {
      for (var i = 0; i < host.children.length; i++) {
        var b = host.children[i];
        b.setAttribute("aria-checked",
          b.getAttribute("data-unit") === settings.windUnits ? "true" : "false");
      }
    }
    var mt = document.getElementById("settings-maptime");
    if (mt) {
      for (var j = 0; j < mt.children.length; j++) {
        var c = mt.children[j];
        c.setAttribute("aria-checked",
          c.getAttribute("data-maptime") === settings.mapTime ? "true" : "false");
      }
    }
  }
  function buildSettingsUI() {
    var host = document.getElementById("settings-units");
    if (!host) return;
    host.innerHTML = "";
    ["kt", "mph", "kmh"].forEach(function (u) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "seg-unit";
      b.setAttribute("role", "radio");
      b.setAttribute("data-unit", u);
      b.textContent = WIND_UNITS[u].label;
      b.addEventListener("click", function () {
        setWindUnits(this.getAttribute("data-unit"));
      });
      host.appendChild(b);
    });
    var mtHost = document.getElementById("settings-maptime");
    if (mtHost) {
      mtHost.innerHTML = "";
      [["synced", "Synced"], ["independent", "Independent"]].forEach(function (m) {
        var mb = document.createElement("button");
        mb.type = "button"; mb.className = "seg-unit";
        mb.setAttribute("role", "radio");
        mb.setAttribute("data-maptime", m[0]);
        mb.textContent = m[1];
        mb.addEventListener("click", function () { setMapTime(this.getAttribute("data-maptime")); });
        mtHost.appendChild(mb);
      });
    }
    syncSettingsUI();
    var pop = document.getElementById("settings-pop");
    var btn = document.getElementById("settings-btn");
    function openSettings() {
      pop.hidden = false; btn.setAttribute("aria-expanded", "true");
    }
    function closeSettings() {
      pop.hidden = true; btn.setAttribute("aria-expanded", "false");
    }
    if (btn) btn.addEventListener("click", function () {
      pop.hidden ? openSettings() : closeSettings();
    });
    if (pop) pop.addEventListener("click", function (e) {
      if (e.target === pop) closeSettings();   // backdrop dismiss
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !pop.hidden) closeSettings();
    });
  }
  loadSettings();

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

  // ===================== Model guidance (Stage B) ==========================
  // Three renderers hydrating from the live Stage-A JSON (cyclolab/{SID}/
  // guidance.json + ships.json). REUSE, no forks: fitProjection + graticule
  // (the cone's projection math), BASEMAP (baked), SSHS + sshsCat (ace_core
  // single source), the cone basemap classes (.ac-*). Track color = SSHWS
  // category of each model's PEAK forecast wind (Andrew's pick, palette B).
  var GDATA = null, SHDATA = null, gDrawn = false;
  function gPeak(pts) { var m = null; pts.forEach(function (p) {
    if (p.vmax != null && (m == null || p.vmax > m)) m = p.vmax; }); return m; }
  function gEsc(s) { return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
  function gBasemap(pr) {
    function pathOf(rings, close) {
      return rings.map(function (r) {
        return "M" + r.map(function (p) {
          return pr.X(p[0]).toFixed(1) + "," + pr.Y(p[1]).toFixed(1);
        }).join("L") + (close ? "Z" : "");
      }).join("");
    }
    return '<rect class="ac-ocean-fill" x="0" y="0" width="' + pr.W +
      '" height="' + pr.H + '"/>' +
      '<path class="ac-land" d="' + pathOf(BASEMAP.land, true) + '"/>' +
      '<path class="ac-state" d="' + pathOf(BASEMAP.states || [], false) + '"/>' +
      '<path class="ac-border" d="' + pathOf(BASEMAP.borders, false) + '"/>' +
      '<path class="ac-coast" d="' + pathOf(BASEMAP_COAST, false) + '"/>';
  }
  var G_TAUS = [0, 24, 48, 72, 96, 120];
  function gTracks() {
    var svg = document.getElementById("gtracks"),
        empty = document.getElementById("gtracks-empty"),
        leg = document.getElementById("gtracks-legend");
    var taids = (GDATA && GDATA.track_aids) || [], aids = (GDATA && GDATA.aids) || {};
    var cons = {}; ((GDATA && GDATA.consensus) || []).forEach(function (c) { cons[c] = 1; });
    var ext = [];
    taids.forEach(function (t) { (aids[t] || []).forEach(function (p) {
      if (p.lat != null) ext.push({ lat: p.lat, lon: p.lon }); }); });
    if (ext.length < 2) {                       // fresh invest / no dynamical tracks
      svg.innerHTML = ""; if (leg) leg.innerHTML = "";
      empty.style.display = "block";
      empty.textContent = GDATA ? "No track aids yet (statistical-only / fresh invest)."
                                : "No model guidance for this storm yet.";
      return;
    }
    empty.style.display = "none";
    var lats = ext.map(function (p) { return p.lat; }), lons = ext.map(function (p) { return p.lon; });
    ext.push({ lat: Math.min.apply(null, lats) - 0.8, lon: Math.min.apply(null, lons) - 0.8 });
    ext.push({ lat: Math.max.apply(null, lats) + 0.8, lon: Math.max.apply(null, lons) + 0.8 });
    var pr = fitProjection(ext, 1000, 360, 560, 18), H = pr.H;
    svg.setAttribute("viewBox", "0 0 1000 " + H);
    var body = [gBasemap(pr), graticule(pr)];
    var ordered = taids.slice().sort(function (a, b) { return (cons[a] ? 1 : 0) - (cons[b] ? 1 : 0); });
    ordered.forEach(function (t) {
      var pts = (aids[t] || []).filter(function (p) { return p.lat != null; });
      if (pts.length < 2) return;
      var col = SSHS[sshsCat(gPeak(pts))] || "#8ea2bd", isC = cons[t];
      var d = "M" + pts.map(function (p) { return pr.X(p.lon).toFixed(1) + "," + pr.Y(p.lat).toFixed(1); }).join("L");
      if (isC) body.push('<path d="' + d + '" fill="none" stroke="#0a1320" stroke-width="6" stroke-linejoin="round" stroke-linecap="round" stroke-opacity="0.85"/>');
      body.push('<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="' + (isC ? 3.4 : 1.7) + '" stroke-opacity="' + (isC ? 1 : 0.82) + '" stroke-linejoin="round" stroke-linecap="round"/>');
      pts.forEach(function (p) { body.push('<circle cx="' + pr.X(p.lon).toFixed(1) + '" cy="' + pr.Y(p.lat).toFixed(1) + '" r="' + (isC ? 2.4 : 1.5) + '" fill="' + col + '"/>'); });
    });
    var spine = aids.TVCN || aids.HCCA || aids[taids[0]] || [];
    spine.forEach(function (p) {
      if (p.lat == null || G_TAUS.indexOf(p.tau) < 0) return;
      var x = pr.X(p.lon), y = pr.Y(p.lat);
      body.push('<g><rect x="' + (x + 5).toFixed(1) + '" y="' + (y - 8).toFixed(1) + '" width="' + (p.tau >= 100 ? 23 : 17) + '" height="13" rx="3" fill="rgba(7,16,28,0.86)" stroke="rgba(120,140,170,0.4)" stroke-width="0.7"/><text x="' + (x + 7).toFixed(1) + '" y="' + (y + 1.5).toFixed(1) + '" fill="#e8eef5" font-size="9.5" font-weight="700">' + p.tau + '</text></g>');
    });
    var c0 = (aids.TVCN || aids[taids[0]] || []).filter(function (p) { return p.tau === 0 && p.lat != null; })[0];
    if (c0) body.push('<circle cx="' + pr.X(c0.lon).toFixed(1) + '" cy="' + pr.Y(c0.lat).toFixed(1) + '" r="4.5" fill="#fff" stroke="#0a1320" stroke-width="1.5"/>');
    svg.innerHTML = body.join("");
    if (leg) leg.innerHTML = taids.map(function (t) {
      var isC = cons[t]; return '<span class="lg"><span class="sw" style="background:' + (SSHS[sshsCat(gPeak(aids[t] || []))] || "#8ea2bd") + ';height:' + (isC ? 4 : 3) + 'px"></span>' + (isC ? '<b>' + gEsc(t) + '</b> (consensus)' : gEsc(t)) + '</span>'; }).join("");
  }
  var G_SSHS_BANDS = [[0, 34, "TD"], [34, 64, "TS"], [64, 83, "C1"], [83, 96, "C2"], [96, 113, "C3"], [113, 137, "C4"], [137, 999, "C5"]];
  function gIntensity() {
    var svg = document.getElementById("gintensity"),
        leg = document.getElementById("gintensity-legend"),
        empty = document.getElementById("gintensity-empty");
    var iaids = (GDATA && GDATA.intensity_aids) || [], aids = (GDATA && GDATA.aids) || {};
    var W = 1000, H = 380, mL = 46, mR = 16, mT = 14, mB = 30, pw = W - mL - mR, ph = H - mT - mB;
    var taus = [], vs = [];
    iaids.forEach(function (t) { (aids[t] || []).forEach(function (p) { if (p.vmax != null) { taus.push(p.tau); vs.push(p.vmax); } }); });
    if (!taus.length) { svg.innerHTML = ""; if (leg) leg.innerHTML = ""; empty.style.display = "block"; empty.textContent = GDATA ? "No intensity aids yet." : ""; return; }
    empty.style.display = "none";
    var tmax = Math.max(120, Math.max.apply(null, taus)), vmax = Math.max(80, Math.ceil((Math.max.apply(null, vs) + 10) / 20) * 20);
    function X(t) { return mL + (t / tmax) * pw; } function Y(v) { return mT + ph - (v / vmax) * ph; }
    var body = ['<rect x="0" y="0" width="' + W + '" height="' + H + '" fill="#101a2c"/>'];
    G_SSHS_BANDS.forEach(function (b) {
      if (b[0] >= vmax) return;
      var y1 = Y(Math.min(b[1], vmax)), y0 = Y(b[0]);
      body.push('<rect x="' + mL + '" y="' + y1.toFixed(1) + '" width="' + pw + '" height="' + (y0 - y1).toFixed(1) + '" fill="' + SSHS[b[2]] + '" fill-opacity="0.12"/>');
      if (b[0] > 0) body.push('<line x1="' + mL + '" y1="' + y0.toFixed(1) + '" x2="' + (mL + pw) + '" y2="' + y0.toFixed(1) + '" stroke="' + SSHS[b[2]] + '" stroke-opacity="0.3" stroke-width="1"/><text x="' + (mL + pw - 4) + '" y="' + (y0 - 3).toFixed(1) + '" text-anchor="end" fill="' + SSHS[b[2]] + '" font-size="9.5" font-weight="700" opacity="0.85">' + b[2] + '</text>');
    });
    for (var v = 0; v <= vmax; v += 20) body.push('<text x="' + (mL - 7) + '" y="' + (Y(v) + 3).toFixed(1) + '" text-anchor="end" fill="#8ea2bd" font-size="10" font-weight="600">' + v + '</text>');
    body.push('<text x="14" y="' + (mT + 4) + '" fill="#8ea2bd" font-size="10" font-weight="700">kt</text>');
    for (var t = 0; t <= tmax; t += 24) body.push('<line x1="' + X(t).toFixed(1) + '" y1="' + mT + '" x2="' + X(t).toFixed(1) + '" y2="' + (mT + ph) + '" stroke="rgba(150,170,200,0.12)" stroke-width="1"/><text x="' + X(t).toFixed(1) + '" y="' + (H - 10) + '" text-anchor="middle" fill="#8ea2bd" font-size="10" font-weight="600">' + t + '</text>');
    body.push('<text x="' + (mL + pw / 2) + '" y="' + (H - 0.5) + '" text-anchor="middle" fill="#8ea2bd" font-size="9.5" font-weight="600">forecast hour</text>');
    var HIRES = { HFAI: "#46c56a", HFBI: "#2bd4c0", HWFI: "#ffe14d", HMNI: "#ff9a2f" };
    function st(t) {
      if (t === "IVCN") return { c: "#ffffff", w: 3.2, op: 1, dash: "", cons: 1, tier: "consensus" };
      if (HIRES[t]) return { c: HIRES[t], w: 2.2, op: 0.95, dash: "", tier: "hi-res" };
      if (t === "DSHP" || t === "LGEM" || t === "SHIP") return { c: "#8ea2bd", w: 1.4, op: 0.85, dash: "3,3", tier: "statistical" };
      return { c: "#5d6b80", w: 1.2, op: 0.7, dash: "", tier: "global" };
    }
    var ordered = iaids.slice().sort(function (a, b) { return (a === "IVCN" ? 1 : 0) - (b === "IVCN" ? 1 : 0); });
    ordered.forEach(function (t) {
      var pts = (aids[t] || []).filter(function (p) { return p.vmax != null; }); if (pts.length < 2) return;
      var s = st(t), d = "M" + pts.map(function (p) { return X(p.tau).toFixed(1) + "," + Y(p.vmax).toFixed(1); }).join("L");
      if (s.cons) body.push('<path d="' + d + '" fill="none" stroke="#0a1320" stroke-width="5.4" stroke-linejoin="round" stroke-opacity="0.8"/>');
      body.push('<path d="' + d + '" fill="none" stroke="' + s.c + '" stroke-width="' + s.w + '" stroke-opacity="' + s.op + '" stroke-dasharray="' + s.dash + '" stroke-linejoin="round" stroke-linecap="round"/>');
      pts.forEach(function (p) { body.push('<circle cx="' + X(p.tau).toFixed(1) + '" cy="' + Y(p.vmax).toFixed(1) + '" r="' + (s.cons ? 2.2 : 1.4) + '" fill="' + s.c + '"/>'); });
    });
    body.push('<rect x="' + mL + '" y="' + mT + '" width="' + pw + '" height="' + ph + '" fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>');
    svg.innerHTML = body.join("");
    if (leg) leg.innerHTML = ordered.slice().reverse().map(function (t) { var s = st(t); return '<span class="lg"><span class="sw" style="background:' + s.c + ';height:' + Math.max(3, s.w) + 'px"></span>' + (s.cons ? '<b>' + gEsc(t) + '</b>' : gEsc(t)) + ' (' + s.tier + ')</span>'; }).join("");
  }
  function gSpark(vals, taus, w, h, color) {
    var ok = []; vals.forEach(function (v, i) { if (v != null) ok.push({ v: v, t: taus[i] }); });
    if (ok.length < 2) return '<svg viewBox="0 0 ' + w + ' ' + h + '"><text x="' + (w / 2) + '" y="' + (h / 2) + '" text-anchor="middle" fill="#566b80" font-size="9">no data</text></svg>';
    var vv = ok.map(function (o) { return o.v; }), lo = Math.min.apply(null, vv), hi = Math.max.apply(null, vv);
    if (hi === lo) { hi += 1; lo -= 1; }
    var tmax = Math.max.apply(null, taus), mb = 11, mt = 4;
    function X(t) { return 2 + (t / tmax) * (w - 4); } function Y(v) { return mt + (h - mt - mb) * (1 - (v - lo) / (hi - lo)); }
    var d = "M" + ok.map(function (o) { return X(o.t).toFixed(1) + "," + Y(o.v).toFixed(1); }).join("L");
    return '<svg viewBox="0 0 ' + w + ' ' + h + '"><line x1="2" y1="' + (h - mb).toFixed(1) + '" x2="' + (w - 2) + '" y2="' + (h - mb).toFixed(1) + '" stroke="rgba(150,170,200,0.18)"/><path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.8" stroke-linejoin="round"/><circle cx="' + X(ok[0].t).toFixed(1) + '" cy="' + Y(ok[0].v).toFixed(1) + '" r="2" fill="' + color + '"/><text x="2" y="' + (h - 2) + '" fill="#566b80" font-size="8">' + lo.toFixed(0) + '</text><text x="' + (w - 2) + '" y="' + (h - 2) + '" text-anchor="end" fill="#566b80" font-size="8">' + hi.toFixed(0) + '</text></svg>';
  }
  function gShips() {
    var root = document.getElementById("gships-root"); if (!root) return;
    var s = SHDATA;
    if (!s || s.available === false) { root.innerHTML = '<div class="stub">SHIPS unavailable for this system' + (s && s.reason ? " (" + gEsc(s.reason) + ")" : "") + ".</div>"; return; }
    var taus = s.taus || [], env = s.env_series || {}, head = [];
    head.push('<span class="g-chip"><b>' + gEsc((s.header || {}).id_line || s.sid || "") + '</b></span>');
    if (s.ahi) head.push('<span class="g-chip">Annularity (AHI) <b>' + gEsc(s.ahi.value) + '</b>' + (s.ahi.verdict ? " &middot; " + gEsc(String(s.ahi.verdict).split(",")[0]) : "") + '</span>');
    if (s.prelim_ri_prob != null) head.push('<span class="g-chip ri">Prelim RI prob <b>' + gEsc(s.prelim_ri_prob) + '%</b></span>');
    var stype = (s.storm_type || [])[0]; if (stype) head.push('<span class="g-chip">Storm type <b>' + gEsc(stype) + '</b></span>');
    var WANT = [["SHEAR (KT)", "#ffd24d"], ["SST (C)", "#ff7a59"], ["700-500 MB RH", "#46c56a"], ["POT. INT. (KT)", "#5aa9ff"], ["HEAT CONTENT", "#ff9a2f"], ["200 MB DIV", "#7aa0ff"], ["STM SPEED (KT)", "#8ea2bd"], ["V (KT) NO LAND", "#e8eef5"], ["TH_E DEV (C)", "#c08bff"]];
    var cells = WANT.filter(function (p) { return env[p[0]]; }).map(function (p) {
      var v = env[p[0]], cur = null; for (var i = 0; i < v.length; i++) { if (v[i] != null) { cur = v[i]; break; } }
      return '<div class="g-sm"><div class="t">' + gEsc(p[0]) + '</div><div class="v">now ' + (cur == null ? "n/a" : cur) + '</div>' + gSpark(v, taus, 200, 60, p[1]) + '</div>';
    }).join("");
    var rm = s.ri_matrix || { cols: [], rows: {} }, tbl = "";
    if (rm.cols && rm.cols.length) {
      tbl = '<table class="g-ri"><caption>RI probability matrix (% in next, vs threshold/hours)</caption><tr><th>RI (kt/h)</th>' + rm.cols.map(function (c) { return '<th>' + gEsc(c) + '</th>'; }).join("") + '</tr>' +
        Object.keys(rm.rows).map(function (rn) { return '<tr><td class="rn">' + gEsc(rn) + '</td>' + rm.cols.map(function (c) { var val = rm.rows[rn][c]; return '<td>' + (val == null ? "&middot;" : gEsc(val) + "%") + '</td>'; }).join("") + '</tr>'; }).join("") + '</table>';
    }
    root.innerHTML = '<div class="g-ships-head">' + head.join("") + '</div><div class="g-sm-grid">' + cells + '</div>' + tbl;
  }
  function gRenderAll() { try { gTracks(); } catch (e) {} try { gIntensity(); } catch (e2) {} try { gShips(); } catch (e3) {} }
  function initGuidance() {
    if (gDrawn) return; gDrawn = true;
    var base = CDN + "/cyclolab/" + encodeURIComponent(SID) + "/";
    Promise.all([fetchJson(base + "guidance.json"), fetchJson(base + "ships.json")])
      .then(function (r) { GDATA = r[0]; SHDATA = r[1]; gRenderAll(); })
      .catch(function () { gRenderAll(); });
  }
  // expose for tests/manual re-render
  window.__gRenderAll = gRenderAll;

  // NHC formation-chance pill (invests only) - eager (not lazy): the genesis
  // odds belong in the banner, not behind a tab. Reads cyclolab/{SID}/
  // formation.json (the poller's parse of the Tropical Weather Outlook).
  function loadFormation() {
    // Invests AND Potential Tropical Cyclones carry the NHC formation-chance
    // pill (same TWO source). For a PTC the TWO may have transitioned it out
    // once advisories began — fetchJson + the null-guard degrade gracefully
    // (the pill simply stays hidden); we NEVER fabricate odds.
    if (!IS_INVEST && !IS_PTC) return;
    var pill = document.getElementById("formation-pill");
    if (!pill) return;
    // Freshness guard: the poller re-stamps formation.json's generated_at every
    // poll while the system is live, so a PROVABLY-stale timestamp means the
    // poller stopped (or NHC dropped the system from the TWO) - a genuinely
    // FROZEN pill must HIDE rather than show stale odds. An ABSENT/unparseable
    // timestamp cannot disprove freshness, so it stays lenient (shows) - the
    // poller always writes generated_at, so this only loosens for legacy/edge.
    var STALE_MS = 12 * 3600 * 1000;   // 2 TWO cycles
    function fresh(ts) {
      if (!ts) return true;
      var t = Date.parse(ts);
      if (isNaN(t)) return true;
      return (Date.now() - t) < STALE_MS;
    }
    function render(f) {
      // Never paint the chance pill once the system is a named/designated TC -
      // a late-resolving fetch (kicked off while still a PTC) must not re-show
      // the pill after setPtc() shed the dress. IS_PTC is live.
      if (!IS_PTC && !IS_INVEST) return false;
      if (!f || (f.p48 == null && f.p7 == null)) return false;
      if (!fresh(f.generated_at)) return false;   // frozen pill -> hide, not stale
      var p7 = (f.p7 != null) ? f.p7 + "%" : "n/a";
      var p48 = (f.p48 != null) ? f.p48 + "%" : "n/a";
      pill.setAttribute("data-level", f.level || "low");
      pill.innerHTML = '<span class="fp-dot"></span>' +
        '<span class="fp-eyebrow"><span>Formation</span>' +
          '<span class="fp-e2">chance</span></span>' +
        '<span class="fp-wins">' +
          '<span class="fp-win">48h <b>' + p48 + '</b></span>' +
          '<span class="fp-div"></span>' +
          '<span class="fp-win">7-day <b>' + p7 + '</b></span>' +
        '</span>';
      pill.hidden = false;
      return true;
    }
    function tryUrl(sid) {
      return fetchJson(CDN + "/cyclolab/" +
                       encodeURIComponent(sid) + "/formation.json");
    }
    // A PTC's OWN formation.json usually 404s — once NHC designates the
    // system the TWO odds stay under the INVEST it spawned (SPAWN_SID). Try
    // own first, then fall back to the spawning invest. Invests read their
    // own. Always graceful: no data anywhere -> the pill stays hidden.
    tryUrl(SID).then(function (f) {
      if (render(f)) return;
      if (IS_PTC && SPAWN_SID) tryUrl(SPAWN_SID).then(render);
    });
  }
  window.__loadFormation = loadFormation;

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
    if (name === "overview" && clMap) clMap.resize();
    if (!inited[name]) {
      inited[name] = true;
      // Stage 3: nothing is fetched until the tab opens (lazy mounts).
      // Phase 3b: the Models tab now hosts BOTH HAFS and the model guidance
      // (the standalone Guidance tab is gone), so opening Models hydrates both.
      if (name === "models") { initModels(); initGuidance(); }
      else if (name === "satellite") initSatellite();
      else if (name === "recon") initRecon();
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
    if (name === "satellite") satStartPoll();   // resume the manifest poll
    else { satPause(); satStopPoll(); }          // leaving: stop both
    if (name !== "models" && hafsViewer && hafsViewer._pause) {
      hafsViewer._pause();
    }
    // recon viewer polls current.json; only let it run while its tab is up
    if (reconViewer) {
      if (name === "recon" && reconViewer._resume) reconViewer._resume();
      else if (reconViewer._pause) reconViewer._pause();
    }
    var w = document.querySelector("#sec-" + name + " .wipe");
    if (w) { w.style.animation = "none"; void w.offsetWidth; w.style.animation = ""; }
  }
  document.getElementById("secnav").addEventListener("click", function (e) {
    var b = e.target.closest(".sec-btn");
    if (b) openSec(b.getAttribute("data-sec"));
  });
  // auto-refresh #5: a BACKGROUNDED page must not keep polling (or playing);
  // resume the manifest poll on return if the Satellite tab is still active.
  if (typeof document !== "undefined" && document.addEventListener) {
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) { satStopPoll(); satPause(); if (clMap) clMap._pause(); }
      else { satStartPoll(); if (clMap) clMap._resume(); }
    });
  }

  // ---- numeric stats: PLAIN STATIC TEXT (final-gate-3 #2) -------------------
  // The odometer is deleted. odoSet just writes the value as plain text
  // (the element keeps its name for the call sites and the baked no-JS
  // form). A real CHANGE plays one subtle fade-in of the new text;
  // first paint and reduced motion swap instantly. No digit cells, no
  // clip, no strip, no timers - the rest state IS the only state, so
  // there is nothing left to shear a baseline or ghost a neighbour.
  function odoSet(el, text) {
    var want = String(text);
    if (el.getAttribute("data-odo") === want) return;
    var first = !el.hasAttribute("data-odo");
    el.setAttribute("data-odo", want);
    el.setAttribute("aria-label", want);
    el.textContent = want;
    if (!first && !reduced) {
      el.classList.remove("odo-swap");
      void el.offsetWidth;                // restart the fade
      el.classList.add("odo-swap");
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
    // Stage C: an invest has no advisory and no ACE; those rows are HIDDEN via
    // CSS (html[data-invest] #vrow-ace/#vrow-next) rather than dropped, so
    // apply()'s odometer writes still find every element.
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
      // forward speed rides the SAME display unit as wind (NHC TCPs give
      // movement in mph; one unit choice everywhere).
      return dirs[Math.round(brg / 22.5) % 16] + " " +
        windDisp(dist / dtH) + " " + windUnitLabel();
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
    // the canon label rides the corner glyph + the Category hero. A PTC shows
    // "PTC" (no category accrues); a real TC shows its SSHWS letter. IS_PTC is
    // LIVE (setPtc flips it), so the label follows the identity automatically -
    // once shed, this re-applies the real category in place of the baked "PTC".
    var catLbl = IS_PTC ? "PTC" : sshsLabel(cat);
    document.getElementById("glyph-cat").textContent = catLbl;
    odoSet(document.getElementById("odo-cat"), catLbl);
    if (!advTypeWord) {
      document.getElementById("storm-type").textContent =
        (IS_PTC ? "POTENTIAL TROPICAL CYCLONE" : catWord(cat).toUpperCase());
    }
    curCat = cat;
    if (reduced) return;
    banner.querySelector(".old-ramp").style.setProperty("--old-ramp", oldRamp);
    banner.classList.remove("xfade", "shine"); void banner.offsetWidth;
    banner.classList.add("xfade", "shine");
  }

  // ---- map + cone (the home view showstopper) -------------------------------
  var coneRing = null, coneAdv = null, conePts = null;
  // ---- Overview hero (final-gate-2 #1): the poller renders dedicated
  // STORM-CENTERED SST products from SOURCE data (cyclolab_sst.py -
  // the house CRW CoralTemp recipe at native 5 km, labeled isotherms,
  // the storm at the EXACT pixel center) and writes them beside the
  // advisory JSON. The shell reads meta.json for the available layers,
  // builds the base-layer picker (LAZY: only the selected layer's PNG
  // is ever fetched) and pins the spinning category glyph at 50%/50% -
  // the render itself is storm-centered, the old client crop math is
  // gone.
  var heroMeta = null;        // {layers:[...], valid_date, updated_utc}
  var heroLayer = null;       // selected layer slug (sticky across polls)
  var heroFixKey = null;      // meta refetches when the fix advances
  var chartDrawn = false;     // W&P draw-on animation runs once

  function heroNote(msg) {
    document.getElementById("sst-hero-note").textContent = msg;
  }
  function heroFallback(msg) {
    // honest fallback: navy panel + centered glyph, no fake field
    heroMeta = null;
    var img = document.getElementById("sst-hero-img");
    img.removeAttribute("src");
    img.removeAttribute("data-url");
    img.style.display = "none";
    document.getElementById("sst-hero-layers").innerHTML = "";
    var g = document.getElementById("sst-grat");
    if (g) g.innerHTML = "";          // PART D: clear the lattice with the raster
    heroNote(msg);
  }
  function heroLayerEntry(slug) {
    var hit = null;
    ((heroMeta && heroMeta.layers) || []).forEach(function (l) {
      if (l.slug === slug) hit = l;
    });
    return hit;
  }
  function heroCaption() {
    var L = heroLayerEntry(heroLayer);
    if (!L) return;
    // disclosure caption: source \u00b7 field \u00b7 valid date (+ layer note,
    // e.g. the anomaly baseline disclosure). Layers can carry their
    // own valid day (the SSTA file can lag the SST file by hours).
    var vd = L.valid || heroMeta.valid_date;
    var d = vd ? " \u00b7 valid " + vd : "";
    heroNote((L.source || "NOAA Coral Reef Watch CoralTemp v3.1 (5 km)") +
      " \u00b7 " + (L.field || L.label || heroLayer) + d +
      " \u00b7 storm-centered render" +
      (L.note ? " \u00b7 " + L.note : "") + ".");
  }
  function heroSetLayer(slug) {
    var L = heroLayerEntry(slug);
    if (!L) return;
    heroLayer = slug;
    var img = document.getElementById("sst-hero-img");
    img.style.display = "";
    var url = SST_BASE + "/" + L.file + "?v=" +
      encodeURIComponent(heroMeta.updated_utc || "");
    if (img.getAttribute("data-url") !== url) {
      img.setAttribute("data-url", url);
      img.onerror = function () {
        heroFallback(
          "SST base layer unavailable \u00b7 NOAA Coral Reef Watch.");
      };
      img.src = url;            // lazy: fetched on selection only
    }
    var host = document.getElementById("sst-hero-layers");
    for (var i = 0; i < host.children.length; i++) {
      var b = host.children[i];
      b.classList.toggle("active", b.getAttribute("data-slug") === slug);
    }
    var head = document.getElementById("sst-hero-head");
    if (head) {
      head.textContent =
        (L.title || L.label || "Sea surface temperature").toUpperCase();
    }
    heroCaption();
    renderSstGraticule();
  }
  // PART D: lat/long lattice over the storm-centered CRW raster, via the SAME
  // graticule() helper as the cone/track/swath (one source). The SST meta
  // carries center + box{hw_lon,hw_lat}; the render is a PlateCarree box
  // [clon+-hw_lon, clat+-hw_lat] at 16:9.2 == the container, so a linear
  // lat/lon->px projection adapter registers exactly. Degrades to empty (no
  // crash) when the meta lacks center/box.
  function renderSstGraticule() {
    var el = document.getElementById("sst-grat");
    if (!el) return;
    if (!heroMeta || !heroMeta.center || !heroMeta.box) { el.innerHTML = ""; return; }
    var clon = heroMeta.center.lon, clat = heroMeta.center.lat;
    var hwLon = heroMeta.box.hw_lon, hwLat = heroMeta.box.hw_lat;
    if (!(hwLon > 0 && hwLat > 0)) { el.innerHTML = ""; return; }
    var W = (heroMeta.px && heroMeta.px[0]) || 1000;
    var H = (heroMeta.px && heroMeta.px[1]) || Math.round(W * hwLat / hwLon);
    el.setAttribute("viewBox", "0 0 " + W + " " + H);
    var pr = {
      W: W, H: H, x0: 0, y0: 0, x1: W, y1: H,
      X: function (lon) { return (lon - (clon - hwLon)) / (2 * hwLon) * W; },
      Y: function (lat) { return ((clat + hwLat) - lat) / (2 * hwLat) * H; },
      lonAt: function (x) { return (clon - hwLon) + x / W * 2 * hwLon; },
      latAt: function (y) { return (clat + hwLat) - y / H * 2 * hwLat; }
    };
    try { el.innerHTML = graticule(pr); }
    catch (e) { el.innerHTML = ""; }
  }
  function heroBuildPicker() {
    var host = document.getElementById("sst-hero-layers");
    host.innerHTML = "";
    ((heroMeta && heroMeta.layers) || []).forEach(function (l) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "hafs-seg";
      b.setAttribute("data-slug", l.slug);
      b.textContent = l.label || l.slug;
      b.addEventListener("click", function () {
        heroSetLayer(this.getAttribute("data-slug"));
      });
      host.appendChild(b);
    });
  }
  function renderHero(storm) {
    var pts = storm.points || [];
    var last = pts[pts.length - 1];
    if (!last || last.lat == null) return;
    document.getElementById("sst-hero-sub").textContent =
      ((document.getElementById("storm-type") || {}).textContent || "")
        .toUpperCase() + " " + (storm.name || "").toUpperCase();
    // glyph: the cone-hero treatment - spinning path, stationary label;
    // ALWAYS at the panel center (the render is storm-centered). A PTC is NOT
    // a depression: the glyph reads GREY with "PTC" (not a category color +
    // "D"), matching the storm's grey invest identity.
    var cat = storm.current_category || "TD";
    var gFill = IS_PTC ? "#9aa6b6" : (SSHS[cat] || SSHS.TD);
    var gLabel = IS_PTC ? "PTC" : sshsLabel(cat);
    var gSize = IS_PTC ? 25 : 34;   // "PTC" (3 glyphs) needs a smaller size
    document.getElementById("sst-hero-glyph").innerHTML =
      '<svg viewBox="-62 -62 124 124">' +
      '<g class="ac-spin"><path d="__HPATH__" fill="' + gFill +
      '" stroke="rgba(0,0,0,0.35)" stroke-width="2"/></g>' +
      '<text class="ac-cat" y="11" text-anchor="middle" font-size="' + gSize +
      '" font-weight="800" fill="#ffffff" stroke="rgba(0,0,0,0.45)" ' +
      'stroke-width="1">' + gLabel + "</text></svg>";
    var fixKey = (last.t || "") + "|" + cat;
    if (heroMeta && heroFixKey === fixKey) { heroCaption(); return; }
    fetchJson(SST_BASE + "/meta.json").then(function (m) {
      if (!m || !m.layers || !m.layers.length) {
        if (!heroMeta) {
          heroFallback("No storm-centered SST render available yet " +
            "\u00b7 NOAA Coral Reef Watch.");
        }
        return;
      }
      heroFixKey = fixKey;
      var slugs = function (mm) {
        return ((mm && mm.layers) || []).map(function (l) {
          return l.slug; }).join(",");
      };
      var rebuild = !heroMeta || slugs(heroMeta) !== slugs(m);
      heroMeta = m;
      if (rebuild) heroBuildPicker();
      heroSetLayer(heroLayerEntry(heroLayer) ? heroLayer
                                             : m.layers[0].slug);
    });
  }

  function renderChart(storm) {
    // ONE CANON (final-gate #1): this chart mirrors the site's
    // existing wind-history graphic (renderWindChart in the basin
    // pages) - SSHS bands at 0.38, white series line, dark dots with
    // white stroke, threshold y-ticks - scaled to this panel, with
    // the dashed pressure series + its mb axis on the right (the
    // pressure-axis backlog item, closed here since it rode along).
    var svg = document.getElementById("chart");
    var pts = (storm.points || []).filter(function (p) {
      return p.wind_kt != null; });
    if (pts.length < 2) { svg.innerHTML = ""; return; }
    var W = 1000, H = 320, padL = 56, padR = 56, padT = 16, padB = 30;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var wMax = Math.max(160, Math.max.apply(null, pts.map(function (p) {
      return p.wind_kt; })) + 10);
    var times = pts.map(function (p) { return new Date(p.t).getTime(); });
    var tMin = Math.min.apply(null, times);
    var tMax = Math.max.apply(null, times);
    function Xt(t) {
      if (tMax === tMin) return padL + plotW / 2;
      return padL + (t - tMin) / (tMax - tMin) * plotW;
    }
    function Yw(w) { return padT + plotH - (w / wMax) * plotH; }
    var prs = pts.map(function (p) { return p.pressure_mb; })
      .filter(function (v) { return v != null; });
    var p0 = Math.min.apply(null, prs.concat([1000])) - 6;
    var p1 = Math.max.apply(null, prs.concat([1014])) + 6;
    function Yp(p) { return padT + plotH - ((p - p0) / (p1 - p0)) * plotH; }
    var parts = ['<rect width="' + W + '" height="' + H +
                 '" fill="#0a1019"/>'];
    // canon SSHS bands
    [[0, 34, "TD"], [34, 64, "TS"], [64, 83, "C1"], [83, 96, "C2"],
     [96, 113, "C3"], [113, 137, "C4"], [137, wMax, "C5"]]
      .forEach(function (b) {
        var y1 = Yw(Math.min(b[1], wMax)), y2 = Yw(b[0]);
        parts.push('<rect x="' + padL + '" y="' + y1.toFixed(1) +
          '" width="' + plotW + '" height="' + (y2 - y1).toFixed(1) +
          '" fill="' + SSHS[b[2]] + '" fill-opacity="0.38"/>');
      });
    // canon threshold y-axis (tick GEOMETRY stays in kt; the LABELS
    // display in the chosen wind unit - final-gate-3 #3)
    [0, 35, 65, 85, 100, 115, 140, 160].forEach(function (v) {
      if (v > wMax) return;
      var y = Yw(v);
      parts.push('<line x1="' + (padL - 5) + '" y1="' + y.toFixed(1) +
        '" x2="' + padL + '" y2="' + y.toFixed(1) +
        '" stroke="#3a4d6e" stroke-width="1"/>');
      parts.push('<text class="wp-ytick" x="' + (padL - 9) + '" y="' +
        (y + 4).toFixed(1) +
        '" text-anchor="end" font-size="12" fill="#8ea2bd">' +
        windDisp(v) + "</text>");
    });
    // x labels (canon: 3 calendar ticks)
    var MN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
              "Sep", "Oct", "Nov", "Dec"];
    for (var xi = 0; xi < 3; xi++) {
      var tt = tMin + xi * (tMax - tMin) / 2;
      var d2 = new Date(tt);
      parts.push('<text x="' + Xt(tt).toFixed(1) + '" y="' + (H - 8) +
        '" text-anchor="middle" font-size="12" fill="#8ea2bd">' +
        MN[d2.getUTCMonth()] + " " + d2.getUTCDate() + "</text>");
    }
    // pressure series + right mb axis
    var pp = pts.filter(function (p) { return p.pressure_mb != null; });
    if (pp.length >= 2) {
      var dPres = pp.map(function (p, i) {
        return (i ? "L" : "M") + Xt(new Date(p.t).getTime()).toFixed(1) +
          "," + Yp(p.pressure_mb).toFixed(1);
      }).join(" ");
      parts.push('<path d="' + dPres + '" fill="none" stroke="#8ea2bd" ' +
        'stroke-width="2" stroke-dasharray="5 4" stroke-opacity="0.9"/>');
      [p0 + 6, (p0 + p1) / 2, p1 - 6].forEach(function (v) {
        var y = Yp(v);
        parts.push('<line x1="' + (W - padR) + '" y1="' + y.toFixed(1) +
          '" x2="' + (W - padR + 5) + '" y2="' + y.toFixed(1) +
          '" stroke="#3a4d6e" stroke-width="1"/>');
        parts.push('<text x="' + (W - padR + 9) + '" y="' +
          (y + 4).toFixed(1) + '" font-size="12" fill="#8ea2bd">' +
          Math.round(v) + "</text>");
      });
    }
    // canon wind series: white line + dark dots, white stroke
    var dWind = pts.map(function (p, i) {
      return (i ? "L" : "M") + Xt(new Date(p.t).getTime()).toFixed(1) +
        "," + Yw(p.wind_kt).toFixed(1);
    }).join(" ");
    parts.push('<path class="series" d="' + dWind +
      '" fill="none" stroke="#ffffff" stroke-width="2.8" ' +
      'stroke-linejoin="round" stroke-linecap="round"/>');
    pts.forEach(function (p) {
      parts.push('<circle cx="' +
        Xt(new Date(p.t).getTime()).toFixed(1) + '" cy="' +
        Yw(p.wind_kt).toFixed(1) +
        '" r="4" fill="#0a1324" stroke="#ffffff" stroke-width="1.6"/>');
    });
    parts.push('<text x="' + padL + '" y="13" fill="#8ea2bd" ' +
      'font-size="11">wind ' + windUnitLabel() +
      ' (solid) \u00b7 pressure mb (dashed, right axis)</text>');
    svg.innerHTML = parts.join("");
    if (!chartDrawn && !reduced) {
      var series = svg.querySelector("path.series");
      var len = series.getTotalLength ? series.getTotalLength() : 2000;
      series.style.setProperty("--len", len);
      svg.classList.add("draw");
    }
    chartDrawn = true;
  }

  // ======================================================================
  // FG-R3 #7/#8: the two Overview art-directed plots - track history +
  // wind-history swath. Both REUSE the cone's auto-fit projection math
  // (renderAdvCone ~line 1731) and the cone's map furniture (.ac-ocean-
  // fill / .ac-land / .ac-graticule / .ac-coast / .ac-border),
  // but fit the projection to the OBSERVED-TRACK + RADII extent (not the
  // forecast cone). The helper math is COPIED, not shared (no cone state).
  // ======================================================================

  // continuous VIBRANT NEON ramp (NOT the 7 SSHS hues - item 7): 0 -> 185+
  // kt, deep indigo -> electric blue -> cyan -> neon green -> chartreuse ->
  // hot orange -> red -> magenta -> near-white. Stops are [kt, r,g,b].
  var NEON_RAMP = [
    [0,   46,  18, 120],   // deep indigo
    [25,  40,  90, 235],   // electric blue
    [45,  20, 200, 245],   // cyan
    [64,  20, 240, 150],   // neon green
    [83, 170, 255,  40],   // chartreuse
    [96, 255, 215,  30],   // yellow
    [113,255, 140,  20],   // hot orange
    [137,255,  50,  40],   // red
    [165,255,  40, 200],   // magenta
    [185,255, 235, 255]    // near-white
  ];
  // sample the continuous ramp -> [r,g,b]. neonColor() wraps this; palette B
  // (FG-R3 #1) reuses it so the wind tiers are EXACT samples of the colorbar.
  function neonRGB(kt) {
    var v = (kt == null) ? 0 : kt;
    if (v <= NEON_RAMP[0][0]) {
      return [NEON_RAMP[0][1], NEON_RAMP[0][2], NEON_RAMP[0][3]];
    }
    var last = NEON_RAMP[NEON_RAMP.length - 1];
    if (v >= last[0]) { return [last[1], last[2], last[3]]; }
    for (var i = 1; i < NEON_RAMP.length; i++) {
      var a = NEON_RAMP[i - 1], b = NEON_RAMP[i];
      if (v <= b[0]) {
        var f = (v - a[0]) / (b[0] - a[0]);
        return [Math.round(a[1] + (b[1] - a[1]) * f),
                Math.round(a[2] + (b[2] - a[2]) * f),
                Math.round(a[3] + (b[3] - a[3]) * f)];
      }
    }
    return [last[1], last[2], last[3]];
  }
  function neonColor(kt) {
    var c = neonRGB(kt);
    return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
  }

  // ====================================================================
  // SSHS-ANCHORED WIND RAMP (backlog E) - the per-fix track dots AND the
  // right-side colorbar SHARE this ramp, so the color BREAKS land EXACTLY
  // on the Saffir-Simpson thresholds (34/64/83/96/113/137 kt). DISCRETE
  // category bands with a thin NEON EDGE separator at each threshold (the
  // approved treatment). Hues are the CANONICAL ace_core.SSHS_COLORS (the
  // baked `SSHS` map) the chips, markers and category pills already use - NO
  // new colors, no drift. Does NOT touch the WIND_TIER blue->gold palette
  // (rings/swath = its own canon) or neonColor (kept for the wind-tier 'B'
  // knob).
  var SSHS_BANDS = [            // [loKt, hiKt, catCode]
    [0, 34, "TD"], [34, 64, "TS"], [64, 83, "C1"], [83, 96, "C2"],
    [96, 113, "C3"], [113, 137, "C4"], [137, 999, "C5"]
  ];
  var SSHS_THRESH = [34, 64, 83, 96, 113, 137];
  var SSHS_CBAR_MAX = 185;      // bar domain top (kt); C5 fills 137..185
  function sshsCat(kt) {        // kt -> category code (step)
    var v = (kt == null || isNaN(kt)) ? 0 : +kt;
    for (var i = 0; i < SSHS_BANDS.length; i++) {
      if (v < SSHS_BANDS[i][1]) return SSHS_BANDS[i][2];
    }
    return "C5";
  }
  function sshsDotColor(kt) {   // discrete: each dot snaps to its category hue
    return SSHS[sshsCat(kt)];
  }
  // Build the right-side colorbar: discrete category bands + a thin neon edge
  // separator at each threshold + frame + units-aware threshold ticks. Shares
  // the ramp with the dots so the legend can never desync; tick labels are
  // tabular-nums (the .tp-cbar-tick CSS) and convert with the unit setting.
  function sshsColorbar(cbX, cbY, cbW, cbH) {
    var out = [];
    var ktY = function (kt) {
      return cbY + cbH - (Math.min(kt, SSHS_CBAR_MAX) / SSHS_CBAR_MAX) * cbH;
    };
    SSHS_BANDS.forEach(function (b) {
      var y1 = ktY(Math.min(b[1], SSHS_CBAR_MAX)), y0 = ktY(b[0]);
      out.push('<rect x="' + cbX + '" y="' + y1.toFixed(1) + '" width="' +
        cbW + '" height="' + (y0 - y1).toFixed(1) + '" fill="' +
        SSHS[b[2]] + '"/>');
    });
    // thin NEON edge separator at each interior threshold: a dark casing
    // under a bright cyan hairline so it reads over light AND hot bands.
    SSHS_THRESH.forEach(function (kt) {
      var ty = ktY(kt);
      out.push('<line x1="' + cbX + '" y1="' + ty.toFixed(1) + '" x2="' +
        (cbX + cbW) + '" y2="' + ty.toFixed(1) +
        '" stroke="rgba(6,11,20,0.85)" stroke-width="2.6"/>');
      out.push('<line x1="' + cbX + '" y1="' + ty.toFixed(1) + '" x2="' +
        (cbX + cbW) + '" y2="' + ty.toFixed(1) +
        '" stroke="#5ef6ff" stroke-width="1.3"/>');
    });
    out.push('<rect class="tp-cbar-frame" x="' + cbX + '" y="' + cbY +
      '" width="' + cbW + '" height="' + cbH + '"/>');
    SSHS_THRESH.forEach(function (kt) {
      var ty = ktY(kt);
      out.push('<line x1="' + (cbX - 4) + '" y1="' + ty.toFixed(1) + '" x2="' +
        cbX + '" y2="' + ty.toFixed(1) + '" stroke="#3a4d6e" ' +
        'stroke-width="1"/>');
      out.push('<text class="tp-cbar-tick" x="' + (cbX - 7) + '" y="' +
        (ty + 4).toFixed(1) + '" text-anchor="end">' + windDisp(kt) +
        "</text>");
    });
    out.push('<text class="tp-cbar-tick" x="' + (cbX + cbW / 2) + '" y="' +
      (cbY - 8) + '" text-anchor="middle">' + windUnitLabel() + "</text>");
    return out.join("");
  }

  // nature -> marker SHAPE: tropical = circle, subtropical = square,
  // non-tropical = triangle. The deck's nature codes (per-fix p.nature):
  // tropical TS/TD/HU/TY/ST/DB (or empty); subtropical SS/SD; everything
  // else (ET/EX/LO/DS/PT/WV/remnant) = non-tropical.
  var SUBTROP_NAT = { SS: 1, SD: 1 };
  var TROP_NAT = { TS: 1, TD: 1, HU: 1, TY: 1, ST: 1, DB: 1 };
  function natureShape(nat) {
    var n = (nat || "").toUpperCase();
    if (!n || TROP_NAT[n]) return "circle";
    if (SUBTROP_NAT[n]) return "square";
    return "triangle";
  }
  function shapeMarker(shape, cx, cy, r, fill, cls, styleColor) {
    // emit one marker glyph centered at cx,cy with radius r. styleColor
    // (optional) sets the element's `color` - the currentColor the CSS glow
    // reads - decoupled from the fill, so the NOW marker keeps its neon-by-
    // wind fill but glows in the active tier palette (FG-R3 #1).
    var c = 'class="' + cls + '" fill="' + fill + '" style="color:' +
            (styleColor || fill) + '"';
    if (shape === "square") {
      var s = r * 1.78;        // visually-matched area
      return '<rect ' + c + ' x="' + (cx - s / 2).toFixed(1) + '" y="' +
        (cy - s / 2).toFixed(1) + '" width="' + s.toFixed(1) +
        '" height="' + s.toFixed(1) + '" rx="1.5"/>';
    }
    if (shape === "triangle") {
      var t = r * 1.28;
      var p1 = cx + "," + (cy - t).toFixed(1);
      var p2 = (cx - t * 0.92).toFixed(1) + "," + (cy + t * 0.74).toFixed(1);
      var p3 = (cx + t * 0.92).toFixed(1) + "," + (cy + t * 0.74).toFixed(1);
      return '<polygon ' + c + ' points="' + p1 + " " + p2 + " " + p3 +
        '"/>';
    }
    return '<circle ' + c + ' cx="' + cx.toFixed(1) + '" cy="' +
      cy.toFixed(1) + '" r="' + r.toFixed(1) + '"/>';
  }

  // shared auto-fit projection (COPIED from renderAdvCone #3) fitted to a
  // list of {lat,lon} extent points + an optional degree pad (radii reach).
  // FOLLOW-UP: BASEMAP.land is baked for the FORECAST extent - if the
  // observed track runs beyond it some edge land may be missing; a
  // track-extent basemap is a v1.x follow-up.
  function fitProjection(extent, viewW, hMin, hMax, margin) {
    var frameLon = (BASEMAP.window[2] + BASEMAP.window[3]) / 2.0;
    function normLon(lon) {
      while (lon - frameLon > 180) lon -= 360;
      while (lon - frameLon < -180) lon += 360;
      return lon;
    }
    var lats = [], lons = [];
    extent.forEach(function (p) {
      lats.push(p.lat); lons.push(normLon(p.lon));
    });
    var latMid = (Math.min.apply(null, lats) +
                  Math.max.apply(null, lats)) / 2;
    var K = Math.max(0.2, Math.cos(latMid * Math.PI / 180));
    function pxu(lon) { return lon * 60 * K; }
    function pyu(lat) { return -lat * 60; }
    var xs = lons.map(pxu), ys = lats.map(pyu);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var W = viewW, MARGIN = margin;
    var spanX = Math.max(1e-6, x1 - x0), spanY = Math.max(1e-6, y1 - y0);
    // FIT-TO-CONTAIN (not width-only): pick the scale that keeps BOTH
    // the width budget and the height cap, so a tall recurving track
    // never overflows the panel and pushes the current fix + wind field
    // off-edge. H follows the content within [hMin, hMax]; both axes are
    // centered. (The cone fits to width because it grows H to 1500; a
    // PANEL caps H, so contain is the right discipline here.)
    var sW = (W - 2 * MARGIN) / spanX;
    var sH = (hMax - 2 * MARGIN) / spanY;
    var S = Math.min(sW, sH);
    var H = Math.max(hMin, Math.min(hMax,
        Math.round(spanY * S + 2 * MARGIN)));
    var offX = (W - spanX * S) / 2;
    var offY = (H - spanY * S) / 2;
    return {
      W: W, H: H, K: K, S: S, normLon: normLon,
      X: function (lon) { return (pxu(normLon(lon)) - x0) * S + offX; },
      Y: function (lat) { return (pyu(lat) - y0) * S + offY; },
      lonAt: function (x) { return ((x - offX) / S + x0) / (60 * K); },
      latAt: function (y) { return -((y - offY) / S + y0) / 60; }
    };
  }

  // Sparse-track fallback (PART D): a single-fix (or near-zero-extent) track
  // projects to a degenerate sub-degree window - no land, no graticule lines
  // (the blank Track History a fresh JTWC TD showed). Pad a too-small extent to
  // >= minDeg around its center so the basemap + graticule always draw. Shared
  // by the track + swath maps; the data points are unchanged (only the
  // projection window widens to a sensible storm-centered default).
  function ensureMinExtent(extent, minDeg) {
    minDeg = minDeg || 8;
    if (!extent || !extent.length) return extent;
    var lats = extent.map(function (p) { return p.lat; });
    var lons = extent.map(function (p) { return p.lon; });
    var loLat = Math.min.apply(null, lats), hiLat = Math.max.apply(null, lats);
    var loLon = Math.min.apply(null, lons), hiLon = Math.max.apply(null, lons);
    if ((hiLat - loLat) >= minDeg && (hiLon - loLon) >= minDeg) return extent;
    var cLat = (loLat + hiLat) / 2, cLon = (loLon + hiLon) / 2, h = minDeg / 2;
    return extent.concat([{ lat: cLat + h, lon: cLon },
                          { lat: cLat - h, lon: cLon },
                          { lat: cLat, lon: cLon + h },
                          { lat: cLat, lon: cLon - h }]);
  }

  // 5-deg graticule (maps-pass R3 #3): drawn as the TOP-MOST layer with a
  // CASING/HALO - a dark hairline UNDER a light line - so every line reads
  // over BOTH the light-gray land AND the dark ocean (a flat light line
  // vanished over the light land in R2). Degree labels on ALL FOUR edges,
  // each with the same dark-casing paint-order stroke.
  // maps-pass R4 #1: the graticule spans the FILL EXTENT [x0,y0..x1,y1] - the
  // viewBox after it is widened/heightened to the card aspect, which reaches
  // BEYOND the data box [0,W]x[0,H]. So lat/lon lines + edge labels reach
  // every PANEL edge (no lineless band). Defaults to the data box.
  function graticule(pr) {
    var W = pr.W, H = pr.H;
    var x0 = (pr.x0 != null) ? pr.x0 : 0, y0 = (pr.y0 != null) ? pr.y0 : 0;
    var x1 = (pr.x1 != null) ? pr.x1 : W, y1 = (pr.y1 != null) ? pr.y1 : H;
    var lonL = pr.lonAt(x0), lonR = pr.lonAt(x1);
    var latT = pr.latAt(y0), latB = pr.latAt(y1);
    var sy0 = y0.toFixed(1), sy1 = y1.toFixed(1);
    var sx0 = x0.toFixed(1), sx1 = x1.toFixed(1);
    var cas = [], lin = [], lab = [];
    for (var gl = Math.ceil(lonL / 5) * 5; gl <= lonR; gl += 5) {
      var gx = pr.X(gl);
      if (gx < x0 + 1 || gx > x1 - 1) continue;
      var gxs = gx.toFixed(1);
      cas.push('<line class="grat-cas" x1="' + gxs + '" y1="' + sy0 +
        '" x2="' + gxs + '" y2="' + sy1 + '"/>');
      lin.push('<line class="grat-lin" x1="' + gxs + '" y1="' + sy0 +
        '" x2="' + gxs + '" y2="' + sy1 + '"/>');
      var gn = ((gl % 360) + 360) % 360;
      var glab = gn > 180 ? (360 - gn) + "°W"
                          : (gn === 0 || gn === 180 ? gn + "°" : gn + "°E");
      if (gx > x0 + 24 && gx < x1 - 24) {
        lab.push('<text class="grat-lab" x="' + gxs + '" y="' +
          (y0 + 17).toFixed(1) + '" text-anchor="middle">' + glab + "</text>");
        lab.push('<text class="grat-lab" x="' + gxs + '" y="' +
          (y1 - 8).toFixed(1) + '" text-anchor="middle">' + glab + "</text>");
      }
    }
    for (var ga = Math.ceil(latB / 5) * 5; ga <= latT; ga += 5) {
      var gy = pr.Y(ga);
      if (gy < y0 + 1 || gy > y1 - 1) continue;
      var gys = gy.toFixed(1);
      cas.push('<line class="grat-cas" x1="' + sx0 + '" y1="' + gys +
        '" x2="' + sx1 + '" y2="' + gys + '"/>');
      lin.push('<line class="grat-lin" x1="' + sx0 + '" y1="' + gys +
        '" x2="' + sx1 + '" y2="' + gys + '"/>');
      var llab = Math.abs(ga) + "°" + (ga >= 0 ? "N" : "S");
      if (gy > y0 + 22 && gy < y1 - 22) {
        var ly = (gy - 5).toFixed(1);
        lab.push('<text class="grat-lab" x="' + (x0 + 7).toFixed(1) +
          '" y="' + ly + '" text-anchor="start">' + llab + "</text>");
        lab.push('<text class="grat-lab" x="' + (x1 - 7).toFixed(1) +
          '" y="' + ly + '" text-anchor="end">' + llab + "</text>");
      }
    }
    return '<g class="ac-graticule">' + cas.join("") + lin.join("") +
      lab.join("") + "</g>";
  }

  // maps-pass R4 #1: aspect-fill. The data lives in [0,W]x[0,H]; expand THAT
  // box (plus PAD) out to the card's measured aspect so the basemap fills
  // the panel edge-to-edge with NO letterbox gap, while the cone / icons /
  // placards / title (all inside [0,W]x[0,H]) are NEVER cropped - the only
  // thing the fill ever reveals is more ocean/land/graticule. Returns the
  // viewBox extent {x0,y0,x1,y1,vw,vh}. Falls back to the data aspect when
  // the stage is not measurable yet (e.g. a hidden tab).
  function fillExtent(W, H, stageEl, pad) {
    pad = (pad == null) ? 30 : pad;
    var cw = stageEl ? stageEl.clientWidth : 0;
    var chh = stageEl ? stageEl.clientHeight : 0;
    var a = (cw > 4 && chh > 4) ? (cw / chh) : (W / H);
    var dW = W + 2 * pad, dH = H + 2 * pad;
    var VW, VH;
    if (a >= dW / dH) { VH = dH; VW = a * dH; }     // landscape card: widen
    else { VW = dW; VH = dW / a; }                  // portrait card: heighten
    var cx = W / 2, cy = H / 2;
    return { x0: cx - VW / 2, y0: cy - VH / 2,
             x1: cx + VW / 2, y1: cy + VH / 2, vw: VW, vh: VH };
  }

  // shared fill applier (maps-pass R4 #1): recompute the card-aspect fill
  // extent + restyle the SVG viewBox (slice = cover, full-bleed), the ocean
  // rect, and the graticule - WITHOUT touching the data layers. gratPr is the
  // projection object the graticule reads (X/Y/lonAt/latAt/W/H). Called once
  // at mount and again on every resize via svg._refit.
  function mapRefit(svg, W, H, gratPr) {
    var fe = fillExtent(W, H, svg);
    svg.setAttribute("viewBox", fe.x0.toFixed(1) + " " + fe.y0.toFixed(1) +
      " " + fe.vw.toFixed(1) + " " + fe.vh.toFixed(1));
    svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
    var oc = svg.querySelector(".ac-ocean-fill");
    if (oc) {
      oc.setAttribute("x", fe.x0.toFixed(1));
      oc.setAttribute("y", fe.y0.toFixed(1));
      oc.setAttribute("width", fe.vw.toFixed(1));
      oc.setAttribute("height", fe.vh.toFixed(1));
    }
    var og = svg.querySelector(".ac-graticule");
    if (og) {
      gratPr.x0 = fe.x0; gratPr.y0 = fe.y0;
      gratPr.x1 = fe.x1; gratPr.y1 = fe.y1;
      og.outerHTML = graticule(gratPr);
    }
    return fe;
  }

  // shared resize re-fit: each rendered map stores svg._refit; debounced.
  var _refitT = null;
  function refitAllMaps() {
    ["advcone", "trackplot", "swathplot"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el && typeof el._refit === "function") {
        try { el._refit(); } catch (e) { /* map not mounted */ }
      }
    });
  }
  if (typeof window !== "undefined" && window.addEventListener) {
    window.addEventListener("resize", function () {
      if (_refitT) clearTimeout(_refitT);
      _refitT = setTimeout(refitAllMaps, 180);
    });
  }

  // basemap furniture (ocean -> land -> borders -> coast -> graticule-on-top)
  // - REUSE the cone classes so one CSS canon dresses all three maps.
  function mapFurniture(pr) {
    var W = pr.W, H = pr.H;
    // ocean fills the FILL EXTENT (maps-pass R4 #1) so it reaches every
    // panel edge under the aspect-fill viewBox; defaults to the data box.
    var ox = ((pr.x0 != null) ? pr.x0 : 0).toFixed(1);
    var oy = ((pr.y0 != null) ? pr.y0 : 0).toFixed(1);
    var ow = ((pr.vw != null) ? pr.vw : W).toFixed(1);
    var oh = ((pr.vh != null) ? pr.vh : H).toFixed(1);
    var parts = ['<rect class="ac-ocean-fill" x="' + ox + '" y="' + oy +
                 '" width="' + ow + '" height="' + oh + '"/>'];
    (BASEMAP.land || []).forEach(function (ring) {
      var d = ring.map(function (c, i) {
        return (i ? "L" : "M") + pr.X(c[0]).toFixed(1) + "," +
          pr.Y(c[1]).toFixed(1);
      }).join(" ") + " Z";
      parts.push('<path class="ac-land" d="' + d + '"/>');
    });
    // state/province lines UNDER the country borders (dimmer .ac-state).
    (BASEMAP.states || []).forEach(function (line) {
      var d = line.map(function (c, i) {
        return (i ? "L" : "M") + pr.X(c[0]).toFixed(1) + "," +
          pr.Y(c[1]).toFixed(1);
      }).join(" ");
      parts.push('<path class="ac-state" d="' + d + '"/>');
    });
    // maps-pass R2: thin white country borders UNDER the thick white coast
    // (both open ne_10m polylines, no trailing Z), over the land fill.
    (BASEMAP.borders || []).forEach(function (line) {
      var d = line.map(function (c, i) {
        return (i ? "L" : "M") + pr.X(c[0]).toFixed(1) + "," +
          pr.Y(c[1]).toFixed(1);
      }).join(" ");
      parts.push('<path class="ac-border" d="' + d + '"/>');
    });
    (BASEMAP_COAST || []).forEach(function (line) {
      var d = line.map(function (c, i) {
        return (i ? "L" : "M") + pr.X(c[0]).toFixed(1) + "," +
          pr.Y(c[1]).toFixed(1);
      }).join(" ");
      parts.push('<path class="ac-coast" d="' + d + '"/>');
    });
    parts.push(graticule(pr));
    return parts;
  }

  // ------------------------------------------------------------- FG-R3 #1
  // WIND-TIER PALETTE - its OWN palette, deliberately NOT the SSHS category
  // tokens (the green/red category recolor was rejected: wind tiers must not
  // borrow category hues at all). FOUR candidate treatments; the live build
  // picks the ring palette and the swath palette INDEPENDENTLY (per-product)
  // via the knobs below. Each entry maps the 34/50/64-kt threshold -> [r,g,b].
  //   A House blues  - incumbent: deep blue 34 field -> lighter cyan 64 core.
  //   B Neon samples - the track colorbar sampled AT 34/50/64 kt, so the
  //                    rings/swath harmonize with the per-fix dots + legend.
  //   C Blue -> Gold - house blue 34/50, warm gold 64 "hurricane" core.
  //   D Violet ramp  - one non-category hue family, light -> hot = stronger.
  var WIND_TIER_PALETTES = {
    A: { "34": [38, 104, 200], "50": [44, 150, 235], "64": [60, 200, 235] },
    B: { "34": neonRGB(34),    "50": neonRGB(50),    "64": neonRGB(64) },
    C: { "34": [38, 104, 200], "50": [44, 168, 240], "64": [255, 190, 52] },
    D: { "34": [150, 128, 232], "50": [168, 86, 214], "64": [200, 48, 150] }
  };
  // resolve a product's palette: explicit JS knob -> URL param -> shared
  // wind knob -> default A (the incumbent). Separate ring/swath knobs so the
  // user's two independent picks both wire cleanly; the board sets the shared
  // __labWindPalette to drive both at once.
  function _palKey(which) {
    var sp; try { sp = new URLSearchParams(location.search); }
    catch (e) { sp = { get: function () { return null; } }; }
    var k = (window["__lab" + which + "Palette"] ||
             sp.get(which.toLowerCase() + "pal") ||
             window.__labWindPalette || sp.get("wtier") || "C");
    k = String(k).toUpperCase();
    return WIND_TIER_PALETTES[k] ? k : "C";
  }
  function resolveTierPalette(which) {
    return WIND_TIER_PALETTES[_palKey(which)];
  }
  // FG-R3 art-r2 verdict: Option C (blue band -> gold core) is the LOCKED
  // wind-tier canon. RINGS render board-C EXACTLY: crisp strokes, no glow/
  // bloom (only the NOW-marker halo follows the palette). The SWATH uses a
  // VIVID saturated variant (electric blue + gold) as bright FLAT fills with
  // crisp smooth edges - see emitSwath (brightness from saturation, not blur).
  // A/B/D stay selectable knobs for the rings.
  var _tierPal = WIND_TIER_PALETTES.C;       // set per-render by each plot
  function tierRGBA(tier, a) {
    var c = _tierPal[String(tier)] || [255, 255, 255];
    return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")";
  }
  function tierHex(tier) {
    var c = _tierPal[String(tier)] || [255, 255, 255];
    return "#" + c.map(function (v) {
      return ("0" + Math.round(v).toString(16)).slice(-2); }).join("");
  }
  // the center NOW-marker halo takes the active palette's strongest tier
  // (64-kt core), replacing the neon-yellow drop-shadow the dot's own fill
  // produced (FG-R3 #1: "the yellow halo is part of the problem").
  function tierGlow() { return tierHex("64"); }


  // maps-pass R5: populate a track/swath HTML title overlay (the cone's
  // contained-box lockup treatment): eyebrow + panel head + storm-name sub,
  // box grows to its widest line, type steps down on a narrow card so it
  // never overflows. The flex stack keeps it clear of the wind-field key.
  function mapLockup(svgId, head) {
    var box = document.getElementById(svgId + "-lockup");
    var headEl = document.getElementById(svgId + "-lockup-head");
    var subEl = document.getElementById(svgId + "-lockup-sub");
    if (!box || !headEl) return;
    var stormName = (document.getElementById("storm-name") || {})
      .textContent || "";
    var typeWord = (document.getElementById("storm-type") || {})
      .textContent || (document.getElementById("chip") || {})
      .textContent || "";
    headEl.textContent = head;
    if (subEl) {
      subEl.textContent = (typeWord.toUpperCase() + " " +
                           stormName.toUpperCase()).trim();
    }
    box.hidden = false;
    box.style.transform = "";
    var svg = document.getElementById(svgId);
    var avail = (svg && svg.clientWidth ? svg.clientWidth : 360) - 24 - 26;
    var maxLine = 0;
    var lines = box.querySelectorAll(".ml-eyebrow,.ml-head,.ml-sub");
    for (var i = 0; i < lines.length; i++) {
      maxLine = Math.max(maxLine, lines[i].scrollWidth);
    }
    if (maxLine > avail && maxLine > 0) {
      box.style.transformOrigin = "top left";
      box.style.transform = "scale(" +
        Math.max(0.62, avail / maxLine).toFixed(3) + ")";
    }
  }

  // maps-pass R5: populate the wind-field key HTML overlay (one ring per tier
  // that rendered, tier-colored) + the unit - fixing the orphaned "kt", which
  // now sits with its legend at the end of the row.
  function mapWindKey(boxId, fieldKey) {
    var box = document.getElementById(boxId);
    if (!box) return;
    if (!fieldKey || !fieldKey.length) { box.hidden = true; return; }
    var html = '<span class="mwk-h">WIND FIELD</span>';
    fieldKey.forEach(function (fk) {
      html += '<span class="mwk-tier" style="color:' + fk[1] + '">' +
        '<span class="mwk-ring"></span>' + windDisp(Number(fk[0])) + "</span>";
    });
    html += '<span class="mwk-u">' + windUnitLabel() + "</span>";
    box.innerHTML = html;
    box.hidden = false;
  }

  // maps-pass R6: populate the CURRENT stats card HTML overlay (Vmax / Pmin /
  // ACE) - same content as the retired in-SVG tp-legend, now pinned to the
  // panel's bottom-left corner. The static trop/sub/non-trop key lives in the
  // template. fix = the latest track point; ace = storm.ace.
  function mapStats(svgId, fix, ace) {
    var box = document.getElementById(svgId + "-stats");
    if (!box) return;
    var set = function (k, txt) {
      var el = document.getElementById(svgId + "-stats-" + k);
      if (el) el.textContent = txt;
    };
    var vmaxKt = fix ? fix.wind_kt : null;
    set("vmax", "Vmax " + (vmaxKt != null
      ? windDisp(vmaxKt) + " " + windUnitLabel() : "—"));
    set("pmin", "Pmin " + (fix && fix.pressure_mb != null
      ? Math.round(fix.pressure_mb) + " mb" : "—"));
    set("ace", "ACE " + (ace != null ? Number(ace).toFixed(2) : "0.00"));
    box.hidden = false;
  }


  // four-quadrant wind-radii arcs around a center (NE/SE/SW/NW). radii =
  // [ne,se,sw,nw] in NAUTICAL MILES -> degrees lat = nm/60. Each quadrant
  // is a 90deg pie sector; a 0/absent quadrant draws nothing. Returns one
  // <path> string (or "" if every quadrant is 0).
  function quadrantArcs(pr, lat, lon, radii, attrs) {
    if (!radii) return "";
    var cx = pr.X(lon), cy = pr.Y(lat);
    // quadrant compass spans (from North, clockwise): NE 0-90, SE 90-180,
    // SW 180-270, NW 270-360.
    var quads = [[0, 90], [90, 180], [180, 270], [270, 360]];
    var dseg = [];
    for (var q = 0; q < 4; q++) {
      var rn = radii[q];
      if (!rn || rn <= 0) continue;
      var rdeg = rn / 60.0;                  // nm -> degrees latitude
      // sample the arc; project each point (lon scales with cos lat).
      var a0 = quads[q][0], a1 = quads[q][1];
      var seg = ['M' + cx.toFixed(1) + "," + cy.toFixed(1)];
      // 4-deg arc sampling (FG-R3 #2): fine enough that the swept envelope's
      // curved edges read smooth, same quality bar as the cone's chain.
      for (var aa = a0; aa <= a1 + 0.001; aa += 4) {
        var th = aa * Math.PI / 180;          // compass angle from North
        var dlat = rdeg * Math.cos(th);
        var dlon = rdeg * Math.sin(th) /
                   Math.max(0.2, Math.cos(lat * Math.PI / 180));
        seg.push("L" + pr.X(lon + dlon).toFixed(1) + "," +
          pr.Y(lat + dlat).toFixed(1));
      }
      seg.push("Z");
      dseg.push(seg.join(" "));
    }
    if (!dseg.length) return "";
    return '<path d="' + dseg.join(" ") + '" ' + attrs +
      ' fill-rule="evenodd"/>';
  }

  // SMOOTH wind-field blob for the SWATH only (FG-R3 #2). The quadrant arcs
  // above are pie SECTORS (center spike + radial edges + a hard radius step
  // at each 90deg boundary); sweeping them unions into a faceted, scalloped
  // band. This emits ONE closed smooth curve per fix instead: the four
  // quadrant radii sit at the quadrant CENTERS (NE 45 / SE 135 / SW 225 /
  // NW 315) and the radius is cosine-interpolated continuously around the
  // compass, sampled every 3deg. No center point, no radial facets - the
  // dense along-track union reads as one clean band, cone-tangent-chain
  // quality. (Stepped quadrant arcs stay correct for the visible rings.)
  function swathBlob(pr, lat, lon, radii) {
    if (!radii) return "";
    var rv = [radii[0] || 0, radii[1] || 0, radii[2] || 0, radii[3] || 0];
    if (Math.max(rv[0], rv[1], rv[2], rv[3]) <= 0) return "";
    var coslat = Math.max(0.2, Math.cos(lat * Math.PI / 180));
    function rAt(th) {
      var a = (((th - 45) % 360) + 360) % 360;   // 0 at the NE center
      var seg = Math.floor(a / 90);              // 0..3
      var f = (a - seg * 90) / 90;               // 0..1 between centers
      var w = 0.5 - 0.5 * Math.cos(f * Math.PI); // smoothstep
      return rv[seg % 4] + (rv[(seg + 1) % 4] - rv[seg % 4]) * w;
    }
    var pts = [];
    for (var th = 0; th < 360; th += 3) {
      var rdeg = rAt(th) / 60.0, t = th * Math.PI / 180;
      pts.push(pr.X(lon + rdeg * Math.sin(t) / coslat).toFixed(1) + "," +
               pr.Y(lat + rdeg * Math.cos(t)).toFixed(1));
    }
    return "M" + pts.join("L") + "Z";
  }

  function stormHasRadii(storm) {
    return (storm.points || []).some(function (p) {
      return p && p.radii && (p.radii["34"] || p.radii["50"] ||
        p.radii["64"]);
    });
  }

  // ---------------------------------------------------------------- #7
  function renderTrackPlot(storm) {
    var svg = document.getElementById("trackplot");
    var note = document.getElementById("trackplot-note");
    if (!svg) return;
    _tierPal = resolveTierPalette("Ring");      // FG-R3 #1: ring palette
    var pts = (storm.points || []).filter(function (p) {
      return p && p.lat != null && p.lon != null; });
    if (pts.length < 1) { svg.innerHTML = ""; if (note) note.textContent = "";
      return; }
    var last = pts[pts.length - 1];
    // projection extent: every track point + the current wind-field reach.
    var extent = pts.map(function (p) { return { lat: p.lat, lon: p.lon }; });
    var lastRad = last.radii || null;
    var maxNm = 0;
    if (lastRad) {
      ["34", "50", "64"].forEach(function (k) {
        (lastRad[k] || []).forEach(function (v) {
          if (v && v > maxNm) maxNm = v; });
      });
    }
    if (maxNm > 0) {
      var pad = maxNm / 60.0 * 1.15;
      extent = extent.concat([
        { lat: last.lat + pad, lon: last.lon },
        { lat: last.lat - pad, lon: last.lon },
        { lat: last.lat, lon: last.lon + pad /
          Math.max(0.2, Math.cos(last.lat * Math.PI / 180)) },
        { lat: last.lat, lon: last.lon - pad /
          Math.max(0.2, Math.cos(last.lat * Math.PI / 180)) }]);
    }
    var pr = fitProjection(ensureMinExtent(extent, 8), 1000, 440, 760, 92);
    var W = pr.W, H = pr.H;
    // maps-pass R4 #1: aspect-fill - the SHARED basemap rule (same as the
    // cone): expand the viewBox to the card aspect so the basemap + graticule
    // fill the panel edge-to-edge; the track dots / radii (in [0,W]x[0,H]) are
    // never cropped.
    var fe = fillExtent(W, H, svg);
    pr.x0 = fe.x0; pr.y0 = fe.y0; pr.x1 = fe.x1; pr.y1 = fe.y1;
    pr.vw = fe.vw; pr.vh = fe.vh;
    var parts = mapFurniture(pr);

    // current wind-field arcs (drawn BEFORE the track dots / hero marker so
    // the marker sits on top but the arcs fan visibly beyond it). Three
    // tiered rings (34 outer / 50 / 64 inner), ~2px strokes, bright on the
    // dark canvas, all from the active wind-tier palette (FG-R3 #1, the
    // user's ring pick) via tierRGBA/tierHex - NOT the SSHS category hues.
    //   Each non-empty tier gets an outer-edge label and an entry in a tiny
    //   inline key. tiers = [thr, fill, stroke, strokeWidth, labelHex, bearing]
    var hasField = false;
    var fieldKey = [];
    if (lastRad) {
      // each tier carries a distinct outer-edge label bearing (compass deg)
      // so the three numbers fan AROUND the ring instead of stacking on one
      // edge: 34 -> NE, 50 -> E, 64 -> N.
      var tiers = [
        ["34", tierRGBA("34", 0.12), tierRGBA("34", 0.95), 2,
          tierHex("34"), 50],
        ["50", tierRGBA("50", 0.14), tierRGBA("50", 0.97), 2,
          tierHex("50"), 95],
        ["64", tierRGBA("64", 0.18), tierRGBA("64", 1), 2.2,
          tierHex("64"), 10]];
      var coslat = Math.max(0.2, Math.cos(last.lat * Math.PI / 180));
      var ccx = pr.X(last.lon), ccy = pr.Y(last.lat);
      tiers.forEach(function (t) {
        var arc = quadrantArcs(pr, last.lat, last.lon, lastRad[t[0]],
          'fill="' + t[1] + '" stroke="' + t[2] + '" stroke-width="' +
          t[3] + '" stroke-linejoin="round"');
        if (!arc) return;
        hasField = true;
        parts.push(arc);
        fieldKey.push([t[0], t[4]]);
        // outer-edge label along this tier's bearing, in the radius of the
        // quadrant the bearing falls in, nudged a few px past the stroke.
        var bearing = t[5];
        var quad = Math.floor((bearing % 360) / 90);   // 0=NE 1=SE 2=SW 3=NW
        var rr = lastRad[t[0]] || [];
        var rn = rr[quad] || 0;
        if (rn > 0) {
          var th = bearing * Math.PI / 180;
          var rdeg = rn / 60.0;
          var lx = pr.X(last.lon + (rdeg * Math.sin(th)) / coslat);
          var ly = pr.Y(last.lat + rdeg * Math.cos(th));
          // push ~9px farther out along the same bearing, in screen space.
          var vx = lx - ccx, vy = ly - ccy;
          var vlen = Math.max(1, Math.sqrt(vx * vx + vy * vy));
          var ox = lx + (vx / vlen) * 9, oy = ly + (vy / vlen) * 9;
          parts.push('<text class="tp-field-lbl" x="' + ox.toFixed(1) +
            '" y="' + (oy + 4).toFixed(1) + '" fill="' + t[4] +
            '" text-anchor="middle">' + windDisp(Number(t[0])) + "</text>");
        }
      });
    }

    // faint track polyline.
    var tp = pts.map(function (p) {
      return [pr.X(p.lon), pr.Y(p.lat)]; });
    var dline = tp.map(function (c, i) {
      return (i ? "L" : "M") + c[0].toFixed(1) + "," + c[1].toFixed(1);
    }).join(" ");
    // maps-pass: SOLID WHITE observed-track connector (was pale blue), with
    // a thin dark casing so it stays crisp over the light-gray land too.
    // The inline stroke beats the shared .tp-track CSS, so the swath plot's
    // connector (its OWN inline override below) is unaffected.
    parts.push('<path d="' + dline + '" fill="none" ' +
      'stroke="rgba(9,22,42,0.5)" stroke-width="3.8" ' +
      'stroke-linecap="round" stroke-linejoin="round"/>');
    parts.push('<path class="tp-track" d="' + dline +
      '" stroke="#ffffff"/>');

    // per-fix dots: neon ramp by wind_kt, shape by nature, latest = hero.
    var shapesSeen = {};
    pts.forEach(function (p, i) {
      var isNow = (i === pts.length - 1);
      // Stage C: an invest's CURRENT-position marker is a RED X (matching the
      // banner identity), not the wind-coloured hurricane dot. Historical track
      // dots are unchanged; named-storm now-markers are unchanged.
      if (isNow && IS_INVEST) {
        var nx = tp[i][0], ny = tp[i][1], s = 10;
        parts.push('<path d="M' + (nx - s).toFixed(1) + ',' + (ny - s).toFixed(1) +
          'L' + (nx + s).toFixed(1) + ',' + (ny + s).toFixed(1) + 'M' +
          (nx + s).toFixed(1) + ',' + (ny - s).toFixed(1) + 'L' + (nx - s).toFixed(1) +
          ',' + (ny + s).toFixed(1) + '" stroke="#ff3b3b" stroke-width="3.6" ' +
          'stroke-linecap="round" fill="none" ' +
          'style="filter:drop-shadow(0 0 5px rgba(255,59,59,0.7));"/>');
        return;
      }
      var col = sshsDotColor(p.wind_kt);
      var shape = natureShape(p.nature);
      shapesSeen[shape] = 1;
      var r = isNow ? 11 : 5.4;
      var cls = "tp-dot" + (isNow ? " tp-now" : "");
      // NOW marker: neon-by-wind fill, but the halo (currentColor) is the
      // active tier palette's core hue, not the neon-yellow fill (FG-R3 #1).
      parts.push(shapeMarker(shape, tp[i][0], tp[i][1], r, col, cls,
        isNow ? tierGlow() : null));
    });

    // LABELED COLORBAR - SSHS category bands + neon edges, sharing the ramp
    // with the per-fix dots (backlog E) so the legend can never desync.
    var cbX = W - 34, cbY = 96, cbW = 16, cbH = H - 200;
    parts.push(sshsColorbar(cbX, cbY, cbW, cbH));

    // maps-pass R6: the CURRENT stats card (Vmax / Pmin / ACE + the
    // trop/sub/non-trop key) was an in-SVG block floating in map space; it is
    // now the bottom-left HTML corner overlay, populated by mapStats() after
    // innerHTML (content unchanged). The in-SVG tp-legend is retired.

    // maps-pass R5: the title lockup + the WIND-FIELD key + the watermark
    // used to STACK in this top-left corner (and the fill insets in-SVG
    // furniture). They are now HTML overlays in a flex stack (de-collided,
    // contained boxes), and the PACIFIC OCEAN watermark is DROPPED - so
    // nothing here but the hairline frame.
    parts.push('<rect class="ac-frame" x="0.75" y="0.75" width="' +
      (W - 1.5) + '" height="' + (H - 1.5) + '" rx="2"/>');

    svg.setAttribute("viewBox", fe.x0.toFixed(1) + " " + fe.y0.toFixed(1) +
      " " + fe.vw.toFixed(1) + " " + fe.vh.toFixed(1));
    svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
    svg.innerHTML = parts.join("");
    svg._refit = function () { mapRefit(svg, W, H, pr); };   // R4 #1 resize
    mapLockup("trackplot", "TRACK HISTORY");     // R5: HTML corner lockup
    mapWindKey("trackplot-windkey", fieldKey);   // R5: HTML wind-field key
    mapStats("trackplot", last, storm.ace);      // R6: HTML CURRENT card

    // disclosure caption (cite the radii source).
    if (note) {
      var cap = "Observed best-track positions colored by 1-min wind; ";
      cap += hasField
        ? "current four-quadrant wind radii from the latest advisory/ATCF " +
          "best-track deck (" + windDisp(34) + "/" + windDisp(50) + "/" +
          windDisp(64) + " " + windUnitLabel() + " thresholds shown)."
        : "wind radii unavailable for this storm.";
      note.textContent = cap;
    }
  }

  // ---------------------------------------------------------------- #8
  // The swath ALWAYS renders FILLED (verdict 2 - the outlined/hatched
  // variant was dropped; filled won). No treatment toggle.
  function renderSwathPlot(storm) {
    var svg = document.getElementById("swathplot");
    var empty = document.getElementById("swath-empty");
    var derived = document.getElementById("swath-derived");
    var method = document.getElementById("swath-method");
    var mbody = document.getElementById("swath-method-body");
    var note = document.getElementById("swathplot-note");
    if (!svg) return;
    var pts = (storm.points || []).filter(function (p) {
      return p && p.lat != null && p.lon != null; });

    // honest empty state when there are no analyzed radii at all.
    if (!stormHasRadii(storm) || pts.length < 1) {
      svg.innerHTML = ""; svg.style.display = "none";
      if (empty) {
        empty.style.display = "block";
        empty.textContent = "Wind-swath needs analyzed wind radii — " +
          "not yet available for this storm.";
      }
      if (derived) derived.hidden = true;
      if (method) method.hidden = true;
      if (note) note.textContent = "";
      return;
    }
    svg.style.display = "block";
    if (empty) empty.style.display = "none";
    if (derived) derived.hidden = false;
    if (method) method.hidden = false;
    if (mbody) {
      mbody.textContent =
        "Per-advisory analyzed four-quadrant wind radii (34 kt and 64 kt) " +
        "are swept along the storm’s track, linearly interpolating " +
        "the radii between consecutive fixes (the standard NHC wind-swath " +
        "construction), and the swept wind-field polygons are accumulated " +
        "into a single swath. Source = the best-track / advisory radii " +
        "deck; the inter-fix interpolation is ours. This is a DERIVED " +
        "product, not an official NHC wind-swath graphic.";
    }

    // projection extent: track + the maximum radii reach anywhere.
    var extent = pts.map(function (p) { return { lat: p.lat, lon: p.lon }; });
    pts.forEach(function (p) {
      if (!p.radii) return;
      ["34", "64"].forEach(function (k) {
        var arr = p.radii[k]; if (!arr) return;
        var mx = Math.max.apply(null, arr.map(function (v) {
          return v || 0; }));
        if (mx > 0) {
          var pad = mx / 60.0;
          var kc = Math.max(0.2, Math.cos(p.lat * Math.PI / 180));
          extent.push({ lat: p.lat + pad, lon: p.lon });
          extent.push({ lat: p.lat - pad, lon: p.lon });
          extent.push({ lat: p.lat, lon: p.lon + pad / kc });
          extent.push({ lat: p.lat, lon: p.lon - pad / kc });
        }
      });
    });
    var pr = fitProjection(ensureMinExtent(extent, 8), 1000, 440, 760, 92);
    var W = pr.W, H = pr.H;
    // maps-pass R4 #1: aspect-fill (shared basemap rule) - basemap + graticule
    // full-bleed; the wind swath (in [0,W]x[0,H]) is never cropped.
    var fe = fillExtent(W, H, svg);
    pr.x0 = fe.x0; pr.y0 = fe.y0; pr.x1 = fe.x1; pr.y1 = fe.y1;
    pr.vw = fe.vw; pr.vh = fe.vh;
    var parts = mapFurniture(pr);

    // sweep one threshold's wind-field along the track, INTERPOLATING the
    // 4 quadrant radii between consecutive fixes (NHC swath method). Each
    // fix + interpolated sub-step emits the four-quadrant polygon; all are
    // drawn with one fill so overlaps visually UNION.
    function sweep(thr) {
      var withR = pts.filter(function (p) {
        return p.radii && p.radii[thr] &&
          Math.max.apply(null, p.radii[thr].map(function (v) {
            return v || 0; })) > 0; });
      if (withR.length === 0) return [];
      var polys = [];
      function emit(lat, lon, radii) {
        // FG-R3 #2: smooth closed blob (not a faceted quadrant sector).
        var d = swathBlob(pr, lat, lon, radii);
        if (d) polys.push(d);
      }
      // build a dense per-fix list (in track order, only ones with radii
      // for this threshold), interpolating sub-steps between neighbours.
      var ordered = pts.filter(function (p) {
        return p.radii && p.radii[thr] &&
          Math.max.apply(null, p.radii[thr].map(function (v) {
            return v || 0; })) > 0; });
      for (var i = 0; i < ordered.length; i++) {
        var a = ordered[i];
        emit(a.lat, a.lon, a.radii[thr]);
        if (i + 1 < ordered.length) {
          var b = ordered[i + 1];
          // FG-R3 #2: interpolation density scales with the ON-SCREEN gap
          // between fixes (~ one sub-step every 7 px) so the swept envelope
          // has clean edges instead of per-fix scallops on the 64-kt core;
          // capped so a long jump can't explode the polygon count.
          var dpx = Math.sqrt(
            Math.pow(pr.X(b.lon) - pr.X(a.lon), 2) +
            Math.pow(pr.Y(b.lat) - pr.Y(a.lat), 2));
          var STEPS = Math.max(6, Math.min(80, Math.ceil(dpx / 4)));
          for (var s = 1; s < STEPS; s++) {
            var f = s / STEPS;
            var lat = a.lat + (b.lat - a.lat) * f;
            var lon = a.lon + (b.lon - a.lon) * f;
            var rr = [0, 0, 0, 0];
            for (var q = 0; q < 4; q++) {
              rr[q] = (a.radii[thr][q] || 0) +
                ((b.radii[thr][q] || 0) - (a.radii[thr][q] || 0)) * f;
            }
            emit(lat, lon, rr);
          }
        }
      }
      return polys;
    }

    var sw34 = sweep("34");
    var sw64 = sweep("64");

    // The swath ALWAYS renders FILLED. CLEAN SOLID bands, NO per-polygon
    // stroke and NO glow/bloom (FG-R3 art-r2: bloom read fuzzy - removed):
    // the smooth blobs are all wound the same way so fill-rule="nonzero"
    // unions every overlap into one seam-free shape with crisp smooth edges.
    // VIVID SATURATED Option-C colors as bright FLAT fills - electric blue
    // 34-kt band, gold 64-kt core - brightness from saturation + opacity, not
    // blur. (Any stroke here re-draws every sub-polygon edge -> a mesh.)
    var SWATH_FILL = { "34": "rgba(40,138,255,0.58)",     // electric blue
                       "64": "rgba(255,196,48,0.86)" };   // bright gold
    function emitSwath(polys, fillKey) {
      if (!polys.length) return;
      var d = polys.join(" ");
      parts.push('<path class="sw-' + fillKey + '" d="' + d +
        '" fill="' + SWATH_FILL[fillKey] +
        '" fill-rule="nonzero" stroke="none"/>');
    }
    emitSwath(sw34, "34");
    emitSwath(sw64, "64");

    // a faint center track for context. maps-pass: a thin dark casing under
    // a solid-white line (mirrors the track-history connector) so it stays
    // legible over the light-gray land, not just over the swath fill.
    var tp = pts.map(function (p) { return [pr.X(p.lon), pr.Y(p.lat)]; });
    var dline = tp.map(function (c, i) {
      return (i ? "L" : "M") + c[0].toFixed(1) + "," + c[1].toFixed(1);
    }).join(" ");
    parts.push('<path d="' + dline + '" fill="none" ' +
      'stroke="rgba(9,22,42,0.45)" stroke-width="3.4" ' +
      'stroke-linecap="round" stroke-linejoin="round"/>');
    parts.push('<path class="tp-track" d="' + dline +
      '" stroke="rgba(255,255,255,0.82)"/>');

    // maps-pass R5: title -> HTML corner overlay (matching the cone + track);
    // PACIFIC OCEAN watermark DROPPED. Just the hairline frame in-SVG.
    parts.push('<rect class="ac-frame" x="0.75" y="0.75" width="' +
      (W - 1.5) + '" height="' + (H - 1.5) + '" rx="2"/>');

    svg.setAttribute("viewBox", fe.x0.toFixed(1) + " " + fe.y0.toFixed(1) +
      " " + fe.vw.toFixed(1) + " " + fe.vh.toFixed(1));
    svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
    svg.innerHTML = parts.join("");
    svg._refit = function () { mapRefit(svg, W, H, pr); };   // R4 #1 resize
    mapLockup("swathplot", "WIND HISTORY");      // R5: HTML corner lockup

    if (note) {
      note.textContent =
        "Cumulative tropical-storm-force (" + windDisp(34) + " " +
        windUnitLabel() + ") swath" + (sw64.length ?
        ", with the hurricane-force (" + windDisp(64) + " " +
        windUnitLabel() + ") core overlaid," : "") +
        " swept from the analyzed quadrant wind radii along the observed " +
        "track. Derived from best-track deck radii — not an official " +
        "NHC wind-swath product.";
    }
  }

  // ---- hydration (poll + diff-merge: grow state, never reset the user) ----
  var lastFixKey = null;
  var lastStorm = null;
  function apply(storm) {
    lastStorm = storm;
    // PTC identity is LIVE (see setPtc): shed the grey/red-X dress the moment
    // the feed shows a named/designated TC, re-wear it on the reverse. MUST run
    // before setCategory so the ramp/label/type-word re-apply under the new
    // identity. A flip forces the Overview/SST plots to re-render below even
    // when the fix is unchanged (a name->TC relabel can land on the same fix).
    var ptcFlip = setPtc(ptcNow(storm), storm);
    setCategory(storm.current_category || "TD");
    document.getElementById("storm-name").textContent =
      (storm.name || SID).toUpperCase();
    var pts = storm.points || [];
    var last = pts[pts.length - 1] || {};
    var fixKey = last.t || null;
    odoSet(document.getElementById("odo-vmax"), windDisp(last.wind_kt));
    var vu = document.getElementById("vmax-unit");
    if (vu) vu.textContent = windUnitLabel();
    odoSet(document.getElementById("odo-mslp"),
           last.pressure_mb != null ? String(Math.round(last.pressure_mb)) : "—");
    odoSet(document.getElementById("odo-ace"),
           storm.ace != null ? storm.ace.toFixed(2) : "0.00");
    odoSet(document.getElementById("odo-pos"), fmtPos(last.lat, last.lon));
    odoSet(document.getElementById("odo-move"), movement(pts));
    odoSet(document.getElementById("odo-fix"),
           fixKey ? fixKey.slice(5, 16).replace("T", " ") : "—");
    if (fixKey !== lastFixKey || ptcFlip) {
      // ISOLATED (the final-gate #3 lesson): one renderer's throw must
      // never starve the next.
      try { renderHero(storm); } catch (e) {
        try { console.warn("[cyclolab] hero render failed:", e); }
        catch (e2) {}
      }
      try { renderChart(storm); } catch (e) {
        try { console.warn("[cyclolab] chart render failed:", e); }
        catch (e2) {}
      }
      // FG-R3 #7/#8: the two Overview plots render on a NEW fix - each
      // ISOLATED (one throw must not starve the next), mirroring the
      // hero/chart wiring exactly.
      try { renderTrackPlot(storm); } catch (e) {
        try { console.warn("[cyclolab] track plot failed:", e); }
        catch (e2) {}
      }
      try { renderSwathPlot(storm); } catch (e) {
        try { console.warn("[cyclolab] swath plot failed:", e); }
        catch (e2) {}
      }
      lastFixKey = fixKey;
    }
  }

  var advFull = null;
  var acRaf = null;          // the cone reveal's rAF loop handle
  var coneHooks = null;      // per-render seek/settle (test + ops hooks)
  function applyAdvisory(adv) {
    if (!adv) return;
    var changed = adv.advisory !== coneAdv;
    advFull = adv;
    coneAdv = adv.advisory;
    coneRing = adv.cone || null;
    conePts = adv.points || null;
    // ENDED pages hydrate the advisory tab too (single fetch below),
    // but a dead storm has no NEXT advisory - never show a countdown.
    nextAdvUtc = ENDED ? null : (adv.next_advisory_utc || null);
    tickCountdown();
    if (adv.points && adv.points.length &&
        DEV_WORD[adv.points[0].dev_label]) {
      advTypeWord = DEV_WORD[adv.points[0].dev_label];
      document.getElementById("storm-type").textContent =
        advTypeWord.toUpperCase();
      // the hero sub-title rides the same word - sync it now instead
      // of waiting for the next fix (adversarial-review find).
      var hs = document.getElementById("sst-hero-sub");
      if (hs && lastStorm) {
        hs.textContent = advTypeWord.toUpperCase() + " " +
          (lastStorm.name || "").toUpperCase();
      }
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
    if (changed && inited.advisories) renderAdvTab();
  }

  function fetchJson(url) {
    if (typeof fetch !== "function") return Promise.resolve(null);
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
  buildSettingsUI();
  loadFormation();          // Stage C: eager NHC formation pill (invests only)
  var BAKED = __BAKED__;
  if (BAKED) apply(BAKED);
  // Mount the Overview stacking map once the DOM is ready (IIFE-A scope).
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", initOverviewMap);
  else initOverviewMap();
  // ENDED pages used to skip the fetch entirely, which left advFull
  // null FOREVER - no cone, no intensity chart, blank advisory text on
  // every dead-storm page (final-gate-3 #4, the latent variant of the
  // user's blank). The frozen R2 advisory JSON is the storm's final
  // state: fetch it ONCE (no re-arm, no feed re-apply - the page's
  // vitals stay frozen as baked).
  if (ENDED) { fetchJson(ADV_URL).then(applyAdvisory); }
  else { poll(); }

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

  // ---- Phase 4: coastal watches/warnings overlay (NHC AL/EP/CP only) ------
  // adv.ww = [{type, geometry:[[lon,lat],...]}], parsed from the official NHC
  // windWatchesWarnings KMZ (kml_advisories.parse_ww_kmz). Canonical NHC TCWW
  // colors keyed by type. ART-GATED: exact hues await sign-off. Storm-surge
  // types ride the same overlay if/when their geometry is present.
  var WW_STYLE = {
    TS_WATCH:   { color: "#ffe14d", label: "TS Watch" },             // yellow
    TS_WARNING: { color: "#3a8bff", label: "TS Warning" },           // blue
    HU_WATCH:   { color: "#ff8cf0", label: "Hurricane Watch" },      // pink
    HU_WARNING: { color: "#ff3b3b", label: "Hurricane Warning" },    // red
    SS_WATCH:   { color: "#c89bff", label: "Storm Surge Watch" },    // lt purple
    SS_WARNING: { color: "#ff3bd4", label: "Storm Surge Warning" }   // magenta
  };
  var wwShown = true;   // overlay visibility (toggle); default ON.
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
    // maps-pass: tighter fit so the cone FILLS the panel - MARGIN was 110
    // (a generous 22% border); 72 still clears the enlarged NOW icon
    // (glyph radius ~43 canvas units) while the cone reaches the edges. The
    // H floor was 540 (forced letterbox on short cones); 430 lets a wide
    // short cone sit shorter so meet-scaling fills the panel width instead.
    var W = 1000, MARGIN = 72;
    var S = (W - 2 * MARGIN) / Math.max(1e-6, x1 - x0);
    var H = Math.max(430, Math.min(1500,
        Math.round((y1 - y0) * S + 2 * MARGIN)));
    var offY = (H - (y1 - y0) * S) / 2;
    function X(lon) { return (pxu(normLon(lon)) - x0) * S + MARGIN; }
    function Y(lat) { return (pyu(lat) - y0) * S + offY; }
    function lonAt(x) { return ((x - MARGIN) / S + x0) / (60 * K); }
    function latAt(y) { return -((y - offY) / S + y0) / 60; }

    // maps-pass R4 #1: aspect-fill the panel. The data is in [0,W]x[0,H];
    // expand to the card aspect (basemap fills the margin, data uncropped).
    var fe = fillExtent(W, H, svg);

    var parts = ['<rect class="ac-ocean-fill" x="' + fe.x0.toFixed(1) +
                 '" y="' + fe.y0.toFixed(1) + '" width="' + fe.vw.toFixed(1) +
                 '" height="' + fe.vh.toFixed(1) + '"/>'];

    // ---- basemap: land -> borders -> coast -> graticule ON TOP (S4-AD1 #2;
    // maps-pass R2 #4: graticule LAST so it spans the full extent over land,
    // via the shared graticule() helper + a local-projection adapter) ------
    (BASEMAP.land || []).forEach(function (ring) {
      var d = ring.map(function (c, i) {
        return (i ? "L" : "M") + X(c[0]).toFixed(1) + "," +
          Y(c[1]).toFixed(1);
      }).join(" ") + " Z";
      parts.push('<path class="ac-land" d="' + d + '"/>');
    });
    // state/province lines UNDER the country borders (dimmer .ac-state).
    (BASEMAP.states || []).forEach(function (line) {
      var d = line.map(function (c, i) {
        return (i ? "L" : "M") + X(c[0]).toFixed(1) + "," +
          Y(c[1]).toFixed(1);
      }).join(" ");
      parts.push('<path class="ac-state" d="' + d + '"/>');
    });
    (BASEMAP.borders || []).forEach(function (line) {
      var d = line.map(function (c, i) {
        return (i ? "L" : "M") + X(c[0]).toFixed(1) + "," +
          Y(c[1]).toFixed(1);
      }).join(" ");
      parts.push('<path class="ac-border" d="' + d + '"/>');
    });
    (BASEMAP_COAST || []).forEach(function (line) {
      var d = line.map(function (c, i) {
        return (i ? "L" : "M") + X(c[0]).toFixed(1) + "," +
          Y(c[1]).toFixed(1);
      }).join(" ");
      parts.push('<path class="ac-coast" d="' + d + '"/>');
    });

    // The INLAND county/zone FILL layer was REMOVED in v3 - the W/W presence on
    // the cone is now ONLY the official NHC coastal breakpoint LINES (the ww
    // array) below, which hug the terrain-accurate coast. wwTypes is declared
    // here for the coastal-line block to populate (the legend).
    var wwTypes = {};
    // (the graticule is pushed AFTER the cone group below, so it sits ABOVE
    // the cone too - maps-pass R3 #3 top-most layer.)


    // ---- reveal (S4-AD2 #1): ONE continuous arc-length-parameterized
    // progress drives a clip-path POLYGON keyframe animation on the
    // cone group - compositor-friendly (no per-frame mask re-raster;
    // the round-1 mask-stroke reveal repainted the masked group every
    // frame and visibly stepped). Constant vertex count across
    // keyframes; revealed vertices are static, the leading-edge cap
    // unfolds; a sine ease over GROW_MS=3600 keeps the peak advance
    // ~1.44% of track length per 30fps frame (spec <= 1.5%).
    //
    // FINAL-GATE R2 #3: the corridor must CONTAIN the cone laterally at
    // every revealed arc position. The old linear half-width estimate
    // ran NARROWER than the real cone wherever it bulged, so the
    // lateral edge of the revealed region was a raw clip cut - the
    // cone's own cased edges stayed hidden and "filled in" only when
    // the clip dropped at settle. Now the corridor's per-sample width
    // comes from the RING GEOMETRY itself (perpendicular-line
    // intersection + margin), and the track is extended BEHIND NOW by
    // the ring's rear extent, so at every frame the revealed region's
    // lateral + rear boundary is the finished cone: fill and casing
    // appear together, only the advancing cap is a raw front.
    var tp = pts.map(function (p) { return [X(p.lon), Y(p.lat)]; });
    var lastSeg = tp.length > 1 ? [tp[tp.length - 1][0] - tp[tp.length - 2][0],
                                   tp[tp.length - 1][1] - tp[tp.length - 2][1]]
                                : [1, 0];
    var lsLen = Math.max(1e-6, Math.hypot(lastSeg[0], lastSeg[1]));
    var firstSeg = tp.length > 1 ? [tp[1][0] - tp[0][0],
                                    tp[1][1] - tp[0][1]] : lastSeg;
    var fsLen = Math.max(1e-6, Math.hypot(firstSeg[0], firstSeg[1]));
    // SMOOTH the cone ring ONCE: derive_cone returns the union-of-disks
    // boundary already resampled to a dense polyline, but re-densify it here
    // with a CLOSED centripetal Catmull-Rom (passes THROUGH every vertex,
    // wraps the seam) so the displayed boundary is uniformly smooth at canvas
    // scale and on-curve. CRITICAL: this same dense smooth ring feeds BOTH the
    // rendered cone path (dC) AND the reveal corridor (ringEdgesAt) below, so
    // the corridor contains the cone exactly (no settle-frame pop-in) and the
    // geometry test reads real on-curve vertices. catmullRomClosed is hoisted.
    var ringPx = catmullRomClosed(
      coneRing.map(function (c) { return [X(c[0]), Y(c[1])]; }), 3.5);
    // rear/forward extents: how far the ring reaches BEHIND the first
    // track point / BEYOND the last one (signed projection onto the
    // end-segment directions).
    var rearNeed = 0, fwdNeed = 0;
    ringPx.forEach(function (q) {
      rearNeed = Math.max(rearNeed,
          -((q[0] - tp[0][0]) * firstSeg[0] +
            (q[1] - tp[0][1]) * firstSeg[1]) / fsLen);
      fwdNeed = Math.max(fwdNeed,
          ((q[0] - tp[tp.length - 1][0]) * lastSeg[0] +
           (q[1] - tp[tp.length - 1][1]) * lastSeg[1]) / lsLen);
    });
    var rearExt = Math.max(24, rearNeed + 40);
    var fwdExt = Math.max(60, fwdNeed + 60);
    var tpExt = [[tp[0][0] - firstSeg[0] / fsLen * rearExt,
                  tp[0][1] - firstSeg[1] / fsLen * rearExt]]
      .concat(tp)
      .concat([[tp[tp.length - 1][0] + lastSeg[0] / lsLen * fwdExt,
                tp[tp.length - 1][1] + lastSeg[1] / lsLen * fwdExt]]);
    // maps-pass R2 #7: ARC-LENGTH-SMOOTH reveal. The corridor centerline was
    // the COARSE forecast-point polyline, so the growth front's tangent
    // FLIPPED at each vertex - a hitch through curves / the recurve. Fit a
    // Catmull-Rom spline THROUGH the forecast points and densify to a smooth
    // polyline; the single continuous arc-length progress (trapezoid-ease)
    // then advances the front by a smooth small increment each frame
    // regardless of curvature. The spline passes THROUGH every control point,
    // so forecast point fc_i lands at dense index (i+1)*PER_SEG and the
    // icon-pop arc distances still map cleanly. High-curvature segments get
    // more arc-length -> more uniform-arc samples downstream (natural adapt).
    var PER_SEG = 14;
    function catmullRom(poly, perSeg) {
      var n = poly.length;
      if (n < 3) return poly.slice();
      var out = [poly[0].slice()];
      for (var i = 0; i < n - 1; i++) {
        var p0 = poly[i > 0 ? i - 1 : 0], p1 = poly[i], p2 = poly[i + 1];
        var p3 = poly[i + 2 < n ? i + 2 : n - 1];
        for (var s = 1; s <= perSeg; s++) {
          var t = s / perSeg, t2 = t * t, t3 = t2 * t;
          out.push([
            0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t +
              (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
              (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
            0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t +
              (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
              (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
          ]);
        }
      }
      return out;
    }
    // Closed-ring CENTRIPETAL Catmull-Rom densifier (alpha = 0.5). Centripetal
    // knot spacing (chord^0.5) is overshoot- and cusp-free, unlike the uniform
    // form which spikes where the envelope pinches (the tiny tau-0 apex cap) or
    // turns sharply. Consecutive near-coincident vertices are de-duplicated first
    // (the small apex arc clusters them, and zero-length knots would divide by
    // zero) and a closing duplicate dropped, then the spline wraps the seam.
    // Returns an OPEN dense point list (the caller closes it with Z). Passes
    // THROUGH every surviving vertex, so the corridor built from the same dense
    // list contains it exactly.
    function catmullRomClosed(poly, spacing) {
      var P = [];
      for (var k = 0; k < poly.length; k++) {
        var q = poly[k];
        if (!P.length ||
            Math.hypot(q[0] - P[P.length - 1][0],
                       q[1] - P[P.length - 1][1]) > 0.4) P.push(q);
      }
      if (P.length > 1 &&
          Math.hypot(P[0][0] - P[P.length - 1][0],
                     P[0][1] - P[P.length - 1][1]) < 0.4) P.pop();
      var n = P.length;
      if (n < 3) return poly.slice();
      function at(j) { return P[((j % n) + n) % n]; }
      function knot(ti, a, b) {
        return ti + Math.sqrt(Math.hypot(b[0] - a[0], b[1] - a[1]));
      }
      var out = [];
      for (var i = 0; i < n; i++) {
        var p0 = at(i - 1), p1 = at(i), p2 = at(i + 1), p3 = at(i + 2);
        var t0 = 0, t1 = knot(t0, p0, p1), t2 = knot(t1, p1, p2),
            t3 = knot(t2, p2, p3);
        // ADAPTIVE: emit ~one sample per `spacing` px of this segment, so the
        // long tangent chords (the facets) get many points while the already-
        // fine arc steps get few - uniform ~spacing-px point density, no sub-
        // pixel clutter (which toFixed(1) rounding would turn into false kinks).
        var m = Math.max(1, Math.round(
          Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / spacing));
        for (var s = 0; s < m; s++) {
          var t = t1 + (t2 - t1) * s / m;
          // Barry-Goldman pyramid (de Boor for non-uniform Catmull-Rom)
          var A1 = _lrp(p0, p1, t0, t1, t), A2 = _lrp(p1, p2, t1, t2, t),
              A3 = _lrp(p2, p3, t2, t3, t);
          var B1 = _lrp(A1, A2, t0, t2, t), B2 = _lrp(A2, A3, t1, t3, t);
          out.push(_lrp(B1, B2, t1, t2, t));
        }
      }
      return out;
    }
    function _lrp(a, b, ta, tb, t) {
      if (tb - ta < 1e-9) return [a[0], a[1]];
      var u = (t - ta) / (tb - ta);
      return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
    }
    tpExt = catmullRom(tpExt, PER_SEG);
    var cum = [0];
    for (var ci = 1; ci < tpExt.length; ci++) {
      cum.push(cum[ci - 1] + Math.hypot(tpExt[ci][0] - tpExt[ci - 1][0],
                                        tpExt[ci][1] - tpExt[ci - 1][1]));
    }
    var Ltot = cum[cum.length - 1];
    // DEGENERATE TRACK (adversarial-review find): a fully-stationary
    // forecast (every point at the same position) collapses the end
    // segments to [0,0], the extensions to nothing and Ltot to 0 -
    // halfAt would index samples[NaN] and the swallowed throw left the
    // cone clipped against an EMPTY path (invisible fill+casing, null
    // hooks). No growth axis exists, so there is nothing to reveal:
    // render the finished cone unclipped, no reveal choreography.
    var revealDegenerate = !(Ltot > 1) || !isFinite(Ltot);
    // forecast-point arc distances in the EXTENDED frame (icons pop
    // when the front tip passes them; index 0 of cum is the rear point)
    var cumIcons = pts.map(function (_, i) { return cum[(i + 1) * PER_SEG]; });
    var HOLD_MS = 1000, GROW_MS = 4000;
    // local cone edges at a canvas point P with normal n: the distance to
    // the NEAREST ring crossing in the +n direction and in the -n direction,
    // returned as [wPos, wNeg] (0 if that side has no crossing). NEAREST, not
    // farthest: on a recurve the perpendicular from a centerline sample also
    // stabs the cone's FAR limb, and the old max|t| latched onto it - so the
    // corridor ballooned across open water and the far end of the cone was
    // uncovered (revealed) before the growth front ever reached it. Stopping
    // at the first boundary on each side keeps the band hugging the LOCAL
    // cone, so the reveal stays one connected shape growing from the storm.
    function ringEdgesAt(px3, py3, nx3, ny3) {
      var wp = Infinity, wn = Infinity;
      for (var ri = 0; ri < ringPx.length; ri++) {
        var a = ringPx[ri];
        var b = ringPx[(ri + 1) % ringPx.length];
        var ex2 = b[0] - a[0], ey2 = b[1] - a[1];
        var det = ex2 * ny3 - ey2 * nx3;
        if (Math.abs(det) < 1e-9) continue;
        var rx = a[0] - px3, ry = a[1] - py3;
        var t3 = (ex2 * ry - ey2 * rx) / det;
        var s3 = (nx3 * ry - ny3 * rx) / det;
        if (s3 < 0 || s3 > 1) continue;
        if (t3 >= 0) { if (t3 < wp) wp = t3; }
        else if (-t3 < wn) wn = -t3;
      }
      return [wp === Infinity ? 0 : wp, wn === Infinity ? 0 : wn];
    }
    // resample the (now arc-smooth) extended track at uniform arc steps -
    // more samples than the coarse era so the corridor chains hug the
    // recurve smoothly.
    var SAMP = 48;
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
    // per-sample corridor half-widths = the cone's REAL local edges on EACH
    // side (wL on the +n side, wR on the -n side), probed at the sample and
    // half a step either side (the ring can bulge between samples) + margin,
    // so the clip edge always rides just OUTSIDE the casing where the cone is
    // revealed. Per-side (asymmetric) is required: a symmetric max-of-both
    // band would, on the concave flank of a recurve, reach back across the
    // gap to the far limb - the very balloon ringEdgesAt avoids.
    var CORR_PAD = 16, CORR_MIN = 32;
    var samples = [];
    var sampStep = Ltot / (SAMP - 1);
    for (var sj = 0; sj < SAMP; sj++) {
      var sd = Ltot * sj / (SAMP - 1);
      var sp = pointAt(sd);
      var wLmax = 0, wRmax = 0;
      [-0.5, 0, 0.5].forEach(function (frac) {
        var q = pointAt(sd + frac * sampStep);
        var e = ringEdgesAt(q.x, q.y, q.nx, q.ny);
        wLmax = Math.max(wLmax, e[0]);
        wRmax = Math.max(wRmax, e[1]);
      });
      samples.push({ d: sd, x: sp.x, y: sp.y, nx: sp.nx, ny: sp.ny,
                     wL: Math.max(CORR_MIN, wLmax + CORR_PAD),
                     wR: Math.max(CORR_MIN, wRmax + CORR_PAD) });
    }
    function halfAt(d) {
      if (!(sampStep > 0)) return [CORR_MIN, CORR_MIN];  // degenerate guard
      d = Math.max(0, Math.min(Ltot, d));
      var j3 = Math.min(SAMP - 2, Math.floor(d / sampStep));
      var f3 = (d - samples[j3].d) / sampStep;
      return [samples[j3].wL + (samples[j3 + 1].wL - samples[j3].wL) * f3,
              samples[j3].wR + (samples[j3 + 1].wR - samples[j3].wR) * f3];
    }
    // REVEAL CLIP corridor vertices (canvas px) revealed from the rear up to
    // arc d: the L rail forward, the FLAT perpendicular FRONT CUT at d, then
    // the R rail back. The band is intentionally WIDER than the cone (the
    // cone's OWN smooth cased edge shows through, never the clip edge); the
    // lateral edges are static, only the straight front cut advances, so the
    // growing edge cannot wiggle - the finished cone is simply uncovered.
    function corridorVerts(d) {
      var L = [], R = [];
      for (var j2 = 0; j2 < SAMP; j2++) {
        var s2 = samples[j2];
        if (s2.d > d) break;            // samples are arc-ordered
        L.push([s2.x + s2.nx * s2.wL, s2.y + s2.ny * s2.wL]);
        R.push([s2.x - s2.nx * s2.wR, s2.y - s2.ny * s2.wR]);
      }
      var ft = pointAt(d), w = halfAt(d);
      return L.concat([[ft.x + ft.nx * w[0], ft.y + ft.ny * w[0]],
                       [ft.x - ft.nx * w[1], ft.y - ft.ny * w[1]]],
                      R.reverse());
    }
    function polyAt(d) {
      // An SVG <clipPath><path d>, mutated via setAttribute each tick:
      // Chromium repaints that reliably, unlike CSS/WAAPI clip animation on
      // SVG containers.
      return "M" + corridorVerts(d).map(function (p) {
        return p[0].toFixed(1) + " " + p[1].toFixed(1);
      }).join(" L ") + " Z";
    }
    function pointInPoly(poly, qx, qy) {
      var inside = false, n = poly.length, j = n - 1;
      for (var i = 0; i < n; i++) {
        var xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
        if (((yi > qy) !== (yj > qy)) &&
            (qx < (xj - xi) * (qy - yi) / (yj - yi + 1e-12) + xi))
          inside = !inside;
        j = i;
      }
      return inside;
    }
    // LOOP/CUSP GUARD: the flat-front corridor can only enclose a TUBE-LIKE
    // cone. A track that loops back on itself within ~120 h makes the union
    // cone SELF-OVERLAP (late large disks engulf the early track), and no
    // single L/front/R band contains it - some cone vertices would stay
    // clipped through the whole reveal and POP IN at settle. Detect it (cone
    // vertices outside the FULL-extent corridor) and, like a degenerate
    // track, ship the finished cone UNCLIPPED - a static cone never pops.
    // Real JTWC 120 h forecasts are monotone-ish and never trip this.
    if (!revealDegenerate) {
      var fullCorr = corridorVerts(Ltot);
      var outTol = Math.max(2, ringPx.length * 0.01), outN = 0;
      for (var rg = 0; rg < ringPx.length && outN <= outTol; rg++) {
        if (!pointInPoly(fullCorr, ringPx[rg][0], ringPx[rg][1])) outN++;
      }
      if (outN > outTol) revealDegenerate = true;   // self-overlap -> static
    }
    // EASE-OUT (cubic): the front advances quickest just after the hold and
    // decelerates into the settle, so the cone "draws in" briskly then eases to
    // rest. Paired with the fixed-timestep accumulator below, the per-frame arc
    // advance is uniform regardless of vsync jitter (the same de-choppy approach
    // as the satellite playback). invEaseS is the exact inverse, used to time the
    // icon pops to the wavefront. Monotonic on [0,1] with easeS(0)=0, easeS(1)=1.
    function easeS(t) {
      t = Math.max(0, Math.min(1, t));
      var u = 1 - t;
      return 1 - u * u * u;
    }
    function invEaseS(f) {
      f = Math.max(0, Math.min(1, f));
      return 1 - Math.pow(1 - f, 1 / 3);
    }


    // ---- the cone (S4-AD1 #8 restyle): crisp navy/white boundary,
    // subtle white-blue interior, NO glow filters -----------------------
    // The cone boundary is the SMOOTH dense ring (catmullRomClosed, computed
    // once above and shared with the reveal corridor) rendered as a fine
    // polyline - visually a continuous curve through the recurve, no facets,
    // and every vertex lies ON the curve so the corridor that contains it is
    // exact. The centre track is densified the same way (open Catmull-Rom)
    // so the forecast markers still sit exactly on a smoothly curving line.
    var dC = "M" + ringPx.map(function (q) {
      return q[0].toFixed(1) + "," + q[1].toFixed(1);
    }).join(" L ") + " Z";
    var dF = "M" + catmullRom(pts.map(function (p) {
      return [X(p.lon), Y(p.lat)];
    }), PER_SEG).map(function (q) {
      return q[0].toFixed(1) + "," + q[1].toFixed(1);
    }).join(" L ");
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
    parts.push('<defs>' + gdefs +
      // maps-pass: blue-GLASS cone fill - a vertical translucent brand-blue
      // ramp, lighter "sheen" at the top fading to a deeper blue below, so
      // the cone reads as blue glass (distinctly blue, not white).
      '<linearGradient id="cone-glass" x1="0" y1="0" x2="0" y2="1">' +
        '<stop offset="0%" stop-color="rgba(112,188,255,0.34)"/>' +
        '<stop offset="44%" stop-color="rgba(60,152,240,0.20)"/>' +
        '<stop offset="100%" stop-color="rgba(34,94,178,0.27)"/>' +
      '</linearGradient>' +
      '<clipPath id="ac-reveal-clip" ' +
      'clipPathUnits="userSpaceOnUse">' +
      '<path class="ac-reveal-path" d=""/></clipPath></defs>');
    // a degenerate track has no growth axis: the group ships UNCLIPPED
    // (the finished cone IS the only frame).
    // maps-pass cone restyle: blue-GLASS fill, a BEVELED dark-blue 3D edge
    // (concentric highlight+shadow strokes - video-safe, no filter), and a
    // SOLID WHITE centerline (was a faint dotted line). Drawn fill-first,
    // then the bevel rim outermost->innermost (dark depth -> blue body ->
    // light top highlight), then the casing+white centerline on top.
    parts.push('<g class="ac-conegrp"' +
      (revealDegenerate ? '' : ' clip-path="url(#ac-reveal-clip)"') + '>' +
      '<path d="' + dC + '" fill="url(#cone-glass)" stroke="none"/>' +
      '<path d="' + dC + '" fill="none" stroke="#0a2138" ' +
      'stroke-width="6.5" stroke-linejoin="round"/>' +
      '<path d="' + dC + '" fill="none" stroke="#2f74bd" ' +
      'stroke-width="3.6" stroke-linejoin="round"/>' +
      '<path d="' + dC + '" fill="none" stroke="#cfe6ff" ' +
      'stroke-width="2" stroke-opacity="0.92" stroke-linejoin="round"/>' +
      '<path d="' + dF + '" fill="none" stroke="rgba(9,22,42,0.55)" ' +
      'stroke-width="4.4" stroke-linecap="round" stroke-linejoin="round"/>' +
      '<path d="' + dF + '" fill="none" stroke="#ffffff" ' +
      'stroke-width="2.4" stroke-linecap="round" ' +
      'stroke-linejoin="round"/></g>');

    // graticule ON TOP of the cone too (maps-pass R3 #3): the casing/halo
    // reads over the cone glass + the land + the ocean, edge-to-edge. The
    // icons + placards below are drawn AFTER, so they stay on top of it.
    parts.push(graticule({ W: W, H: H, X: X, Y: Y,
                           lonAt: lonAt, latAt: latAt,
                           x0: fe.x0, y0: fe.y0, x1: fe.x1, y1: fe.y1 }));

    // ---- Phase 4: coastal watches/warnings breakpoint LINES (NHC AL/EP/CP) --
    // The official NHC TCWW segments (advFull.ww) drawn ABOVE the cone + the
    // inland fills (which are on the basemap above), sharing the one palette +
    // legend + toggle. The fills above already populated wwTypes; the lines add
    // to it. A separate toggleable group outside the reveal clip.
    // coastal breakpoint lines ON TOP of the fills.
    var wwSegs = (advFull && advFull.ww) || [];
    if (wwSegs.length) {
      var wwParts = [];
      wwSegs.forEach(function (seg) {
        var st = WW_STYLE[seg.type] ||
                 { color: "#cfd8e6", label: (seg.type || "Advisory area") };
        var d = (seg.geometry || []).map(function (c, i) {
          return (i ? "L" : "M") + X(c[0]).toFixed(1) + "," + Y(c[1]).toFixed(1);
        }).join(" ");
        if (!d) return;
        wwTypes[seg.type] = st;   // legend advertises only types that drew a path
        wwParts.push('<path class="ww-cas" d="' + d + '" stroke-width="7"/>');
        wwParts.push('<path class="ww-lin" d="' + d + '" stroke="' +
                     st.color + '" stroke-width="4"/>');
      });
      parts.push('<g class="ac-ww" id="ac-ww-group"' +
                 (wwShown ? '' : ' style="display:none"') + '>' +
                 wwParts.join("") + '</g>');
    }

    // ---- icons + placards (S4-AD1 #4/5/6/7) --------------------------
    // collision-aware placard layout: alternate sides of the track,
    // push outward on overlap, leader line when pushed far.
    var rects = [];      // occupied rects: title, icons, placards
    // ---- TITLE LOCKUP corner reservation: the lockup is the HTML overlay
    // pinned TOP-LEFT (maps-pass R3 #2), so reserve the top-left rect here
    // and placards avoid it.
    var coneXs = coneRing.map(function (c) { return X(c[0]); });
    var coneYs = coneRing.map(function (c) { return Y(c[1]); });
    var coneBox = { x: Math.min.apply(null, coneXs),
                    y: Math.min.apply(null, coneYs),
                    w: Math.max.apply(null, coneXs) -
                       Math.min.apply(null, coneXs),
                    h: Math.max.apply(null, coneYs) -
                       Math.min.apply(null, coneYs) };
    var TIT_W = 320, TIT_H = 64, TIT_PAD = 18;
    function overlapArea(a, b) {
      var ox = Math.max(0, Math.min(a.x + a.w, b.x + b.w) -
                           Math.max(a.x, b.x));
      var oy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) -
                           Math.max(a.y, b.y));
      return ox * oy;
    }
    var titleRect = { x: TIT_PAD, y: TIT_PAD, w: TIT_W, h: TIT_H };
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
      // maps-pass R2 (legibility): the NOW marker DOMINATES the origin and
      // the forecast glyphs are enlarged so every label reads at a glance.
      var half = (i === 0 ? 80 : 46);
      iconR.push(half);
      rects.push({ x: tp[i][0] - half, y: tp[i][1] - half,
                   w: 2 * half, h: 2 * half });
    });
    // FINAL-GATE R3 #1 - NO LEADER LINES, EVER. Pairing is carried by
    // POSITION alone: a placard hugs its glyph just outside the icon
    // edge and is pushed PERPENDICULAR (straight out from the track)
    // before it ever slides along-track, so the eye reads "this pill
    // belongs to the glyph directly inward from it." The candidate
    // slots are ordered by a cost that penalises along-track nudges
    // heavily and growth gently, and a side flip is the cheapest escape
    // - the placard stays as close and as radial as the cluster allows.
    var placards = [];
    pts.forEach(function (p, i) {
      var pw = (i === 0 ? 188 : 152), ph = (i === 0 ? 46 : 40);
      // local track direction -> perpendicular placement sides
      var a = tp[Math.max(0, i - 1)], b = tp[Math.min(tp.length - 1, i + 1)];
      var vx = b[0] - a[0], vy = b[1] - a[1];
      var vl = Math.max(1e-6, Math.hypot(vx, vy));
      var nx = -vy / vl, ny = vx / vl;
      var ux = vx / vl, uy = vy / vl;
      var side = (i % 2 === 0) ? 1 : -1;
      if (i === 0) side = -1;            // NOW placard prefers up-track
      // The pill sits just clear of the icon EDGE: base gap scales with
      // the glyph so the hero NOW and the small taus hug equally tight.
      var base = iconR[i] + ph / 2 + 6;
      var offs = [base, base + 14, base + 30, base + 50, base + 74,
                  base + 102];
      var nudges = [0, 26, -26, 54, -54];
      // Cost-ordered candidate sweep: smallest perpendicular offset and
      // zero along-track nudge first; growing the offset is cheap, a
      // side flip is cheaper than a big nudge, and along-track sliding
      // is the last resort. Placards NEVER overlap (S4-AD1 #7).
      var cands = [];
      for (var oi = 0; oi < offs.length; oi++) {
        for (var ni = 0; ni < nudges.length; ni++) {
          for (var sj = 0; sj < 2; sj++) {
            var s = (sj === 0) ? side : -side;
            cands.push({ cost: oi * 2 + ni * 3 + sj, off: offs[oi],
                         nud: nudges[ni], s: s });
          }
        }
      }
      cands.sort(function (c1, c2) { return c1.cost - c2.cost; });
      var placed = null;
      for (var ci = 0; ci < cands.length && !placed; ci++) {
        var c = cands[ci];
        var cx2 = tp[i][0] + nx * c.s * c.off + ux * c.nud;
        var cy2 = tp[i][1] + ny * c.s * c.off + uy * c.nud;
        var r2 = { x: cx2 - pw / 2, y: cy2 - ph / 2, w: pw, h: ph };
        r2.x = Math.max(6, Math.min(W - pw - 6, r2.x));
        r2.y = Math.max(6, Math.min(H - ph - 6, r2.y));
        if (!overlaps(r2)) placed = r2;
      }
      if (!placed) {                     // pathological: park it below
        placed = { x: Math.max(6, Math.min(W - pw - 6,
                       tp[i][0] - pw / 2)),
                   y: Math.min(H - ph - 6, 6 + i * (ph + 4)),
                   w: pw, h: ph };
      }
      var gap = Math.hypot(placed.x + pw / 2 - tp[i][0],
                           placed.y + ph / 2 - tp[i][1]) - iconR[i];
      rects.push(placed);
      placards.push({ rect: placed, gap: gap });
    });

    var anyNonTropical = false;
    pts.forEach(function (p, i) {
      var px2 = tp[i][0], py2 = tp[i][1];
      var tropical = pointTropical(p);
      if (!tropical) anyNonTropical = true;
      var cat = pointCat(p);
      var col = tropical ? (SSHS[cat] || SSHS.TD) : "#ffffff";
      // pops ride the wavefront EXACTLY: the cap tip sits at the
      // front distance, so delay = hold + grow * invease(d_i / L)
      // (d_i measured in the rear-extended frame - cumIcons). A
      // degenerate track has no wavefront: plain index stagger.
      var delayMs = (i === 0)
        ? 400
        : (revealDegenerate
           ? 600 + i * 150
           : Math.round(HOLD_MS + GROW_MS *
                        invEaseS(Math.max(0.02, cumIcons[i] / Ltot))));
      var scale = (i === 0 ? 1.9 : 0.98);    // NOW dominates; forecast LEGIBLE
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
      // placard (skip none - NOW gets one too). NO leader line: pairing
      // is positional (the pill hugs its glyph; see the placement sweep).
      var pl = placards[i];
      var pillGrad = "url(#pillg-" + (tropical ? cat : "NEUTRAL") + ")";
      var label = (i === 0)
        ? "NOW \u00b7 " + windDisp(p.intensity_kt || 0) + windUnitLabel()
        : "+" + tau + "h \u00b7 " + windDisp(p.intensity_kt || 0) +
          windUnitLabel();
      var pw2 = pl.rect.w, ph2 = pl.rect.h;
      parts.push('<g data-role="placard" data-i="' + i +
        '" data-gap="' + pl.gap.toFixed(1) +
        '" data-iconr="' + iconR[i] + '" data-x="' +
        pl.rect.x.toFixed(1) + '" data-y="' + pl.rect.y.toFixed(1) +
        '" data-w="' + pw2 + '" data-h="' + ph2 +
        '" transform="translate(' + pl.rect.x.toFixed(1) +
        " " + pl.rect.y.toFixed(1) + ')">' +
        '<rect width="' + pw2 + '" height="' + ph2 + '" rx="' +
        (ph2 / 2) + '" fill="' + pillGrad +
        '" stroke="rgba(0,0,0,0.3)"/>' +
        '<text x="' + (pw2 / 2) + '" y="' + (ph2 / 2) +
        '" text-anchor="middle" dominant-baseline="central" font-size="' +
        (i === 0 ? 25 : 21) +
        '" font-weight="' + (i === 0 ? 800 : 700) +
        // canon ink rule: ALWAYS light on the category ramps (same
        // stroke treatment as the icon SS labels)
        '" fill="#ffffff" stroke="rgba(0,0,0,0.4)" stroke-width="0.8" ' +
        'paint-order="stroke">' + label + "</text></g>");
      parts.push("</g>");
    });

    // ---- title lockup (#5; maps-pass R3 #2) -- an HTML OVERLAY pinned to
    // the panel's TOP-LEFT CORNER (populated here), NOT an in-SVG element:
    // the cone SVG is meet-scaled + CENTERED in the card, so an in-SVG lockup
    // floats inset from the card edge. The eyebrow + storm name; the panel
    // <h3> is the canonical "Forecast cone" head. titleRect (top-left) is
    // still reserved below so placards avoid the lockup corner.
    var stormName = (document.getElementById("storm-name") || {})
      .textContent || "";
    var typeWord = (document.getElementById("storm-type") || {})
      .textContent || "";
    var lkName = document.getElementById("advcone-lockup-name");
    var lkBox = document.getElementById("advcone-lockup");
    if (lkName && lkBox) {
      lkName.textContent = (typeWord.toUpperCase() + " " +
                            stormName.toUpperCase()).trim();
      lkBox.hidden = false;
      // maps-pass R4 #2: the box grows to its widest line; for a long name
      // (e.g. TROPICAL DEPRESSION THREE-E) on a narrow card, STEP THE TYPE
      // DOWN so it never escapes the box. Measure the actual rendered width
      // against the available card width (the box can fill almost the whole
      // panel via max-width: calc(100% - 20px)).
      lkName.style.fontSize = "";          // reset to the 18px base
      var stage = lkBox.parentNode;
      var avail = (stage ? stage.clientWidth : 320) - 20 - 28;  // - margin/pad
      var natW = lkName.scrollWidth;
      if (natW > avail && natW > 0) {
        var fs = Math.max(11, 18 * avail / natW);
        lkName.style.fontSize = fs.toFixed(1) + "px";
      }
    }

    // ---- ocean watermark: RETIRED (maps-pass R2 #6 on the cone, R5 on the
    // track + swath). A panel-filling map leaves no clean open water, and the
    // coastlines + graticule already orient the viewer, so "PACIFIC OCEAN"
    // is dropped everywhere; the auto-placement sweep (oceanWatermark) is gone.

    // hairline frame: the map has edges (#2)
    parts.push('<rect class="ac-frame" x="0.75" y="0.75" width="' +
      (W - 1.5) + '" height="' + (H - 1.5) + '" rx="2"/>');

    // maps-pass R4 #1: the viewBox is the FILL EXTENT (card aspect), sliced
    // to COVER the panel - full-bleed, no letterbox gap. The data box
    // [0,W]x[0,H] is centered inside it, so slice only ever trims ocean/land.
    svg.setAttribute("viewBox", fe.x0.toFixed(1) + " " + fe.y0.toFixed(1) +
      " " + fe.vw.toFixed(1) + " " + fe.vh.toFixed(1));
    svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
    svg.innerHTML = parts.join("");

    // Phase 4: WW legend + toggle - shown only when segments are present.
    // onchange (not addEventListener) so re-rendering the tab never stacks
    // duplicate handlers.
    (function () {
      var bar = document.getElementById("advcone-ww");
      var leg = document.getElementById("advcone-ww-legend");
      var chk = document.getElementById("advcone-ww-chk");
      if (!bar || !leg || !chk) return;
      var types = Object.keys(wwTypes);
      if (!types.length) { bar.hidden = true; leg.innerHTML = ""; return; }
      var ORDER = ["TS_WATCH", "TS_WARNING", "HU_WATCH", "HU_WARNING",
                   "SS_WATCH", "SS_WARNING"];
      types.sort(function (a, b) {
        var ia = ORDER.indexOf(a), ib = ORDER.indexOf(b);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
      });
      leg.innerHTML = types.map(function (t) {
        var st = wwTypes[t];
        return '<span class="ww-lg"><span class="ww-sw" style="background:' +
          st.color + '"></span>' + st.label + '</span>';
      }).join("");
      bar.hidden = false;
      chk.checked = wwShown;
      chk.onchange = function () {
        wwShown = chk.checked;
        // the toggle drives the coastal-line group (the inland fill layer was
        // removed in v3).
        var g = document.getElementById("ac-ww-group");
        if (g) g.style.display = wwShown ? "" : "none";
      };
    })();

    // re-fit on resize / orientation change (the shared resize handler calls
    // svg._refit): recompute the fill extent + restyle the viewBox, ocean,
    // and graticule WITHOUT touching the cone reveal/icons.
    var coneGratPr = { W: W, H: H, X: X, Y: Y, lonAt: lonAt, latAt: latAt };
    svg._refit = function () { mapRefit(svg, W, H, coneGratPr); };

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
    if (revealDegenerate) {
      // no growth axis: ships unclipped above; honest static hooks.
      grp.setAttribute("data-reveal", "final");
      coneHooks = {
        seek: function () {
          return { d: 0, Ltot: 0, degenerate: true, W: W, H: H };
        },
        settle: function () {
          return { Ltot: 0, degenerate: true, W: W, H: H };
        }
      };
    } else if (!reduced && grp.animate) {
      revealPath.setAttribute("d", polyAt(0));
      grp.setAttribute("data-reveal", "animated");
      // FIXED-TIMESTEP reveal clock (same de-choppy approach as the satellite
      // playback): accumulate real elapsed time and advance the simulated clock
      // in fixed FRAME_MS chunks, so the eased arc progress advances by a uniform
      // amount per step regardless of vsync jitter (the variable-timestep version
      // read raw performance.now() and stepped under frame-interval noise). The
      // accumulator is clamped so a tab-switch stall can't fast-forward / spiral.
      // Only the clip's flat front moves; the cone geometry never rebuilds.
      var FRAME_MS = 1000 / 60;
      var simMs = -HOLD_MS;          // hold before the front starts to grow
      var prevTs = null, acc = 0;
      var tickFn = function (ts) {
        if (prevTs === null) prevTs = ts;
        acc += Math.min(100, ts - prevTs);   // clamp: never advance > ~6 frames at once
        prevTs = ts;
        while (acc >= FRAME_MS) { simMs += FRAME_MS; acc -= FRAME_MS; }
        var tt = simMs / GROW_MS;
        if (tt >= 1) {
          revealPath.setAttribute("d", polyAt(Ltot));   // pin the full cone first
          grp.removeAttribute("clip-path");
          grp.setAttribute("data-reveal", "final");
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

    // deterministic reveal hooks (final-gate-2 #3): place the growth
    // front at an exact arc fraction, or jump to the settled frame.
    // Pure geometry through the same polyAt the rAF clock drives -
    // the per-frame casing test extracts frames through these instead
    // of racing wall-clock animation. (Degenerate tracks installed
    // their static hooks above.)
    if (!revealDegenerate) coneHooks = {
      seek: function (f) {
        if (acRaf) { cancelAnimationFrame(acRaf); acRaf = null; }
        var d = Math.max(0, Math.min(1, f)) * Ltot;
        grp.setAttribute("clip-path", "url(#ac-reveal-clip)");
        revealPath.setAttribute("d", polyAt(d));
        var p = pointAt(d);
        var hw = halfAt(d);            // [wL, wR] - report the wider side
        return { d: d, Ltot: Ltot, tipX: p.x, tipY: p.y,
                 w: Math.max(hw[0], hw[1]), W: W, H: H };
      },
      settle: function () {
        if (acRaf) { cancelAnimationFrame(acRaf); acRaf = null; }
        grp.removeAttribute("clip-path");
        grp.setAttribute("data-reveal", "final");
        return { Ltot: Ltot, W: W, H: H };
      }
    };

    var official = advFull.method === "official-cone";
    note.textContent = (official
      ? "Official NHC forecast cone \u00b7 advisory " + advFull.advisory +
        " \u00b7 icons colored by forecast intensity."
      : "Derived uncertainty envelope \u2014 not an official JTWC " +
        "product \u00b7 advisory " + advFull.advisory + " \u00b7 method " +
        (advFull.method || "derived") + ".") +
      (anyNonTropical ? " White icons = forecast non-tropical." : "");
    document.getElementById("advcone-method-body").textContent =
      (official ? NHC_METHOD_COPY : WP_METHOD_COPY) + unitsSourceNote();
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
    // y-axis: tick GEOMETRY in kt, LABELS in the chosen unit (#3)
    [0, 25, 50, 75, 100, 125, 150].forEach(function (k) {
      if (k > kMax) return;
      parts.push('<text class="in-ytick" x="' + (padL - 8) + '" y="' +
        (Yk(k) + 4).toFixed(1) + '" text-anchor="end" font-size="11" ' +
        'fill="#8b95a5">' + windDisp(k) + "</text>");
    });
    parts.push('<text x="' + (padL - 8) + '" y="' + (padT + 2) +
      '" text-anchor="end" font-size="10.5" fill="#5b6677">' +
      windUnitLabel() + "</text>");
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
      (INTENSITY_ERR.vintage_note ? " " + INTENSITY_ERR.vintage_note : "") +
      unitsSourceNote();
    document.getElementById("intensity-method-body").textContent = body;
  }

  // (§7.4) advisory text panels - monospace, never tinted. Placeholder
  // states are HONEST (final-gate-3 #4): a product whose URL exists but
  // whose text hasn't attached yet is "posting" (the poller's text-heal
  // rewrites the advisory JSON within a poll or two and the page poll
  // picks it up - it WILL load); no URL at all means the agency
  // publishes no such product for this storm.
  var advTextProd = "tcp";
  function renderAdvText() {
    var pre = document.getElementById("advtext");
    var t = (advFull && advFull.text) || {};
    var body = advTextProd === "tcp" ? t.tcp : t.tcd;
    var hasUrl = !!(advTextProd === "tcp" ? t.tcp_url : t.tcd_url);
    pre.textContent = body || (!advFull
      ? "(loading advisory data…)"
      : hasUrl
        ? "(advisory text hasn’t posted yet — it loads " +
          "automatically within a few minutes)"
        : "(no advisory text product is published for this storm)");
    var host = document.getElementById("advtext-tabs");
    // Source-aware product labels (CYCLOLAB_DESIGN §8.4): JTWC ships a Warning +
    // Prognostic Reasoning; NHC a Public Advisory + Discussion. Same tcp/tcd
    // slots + playback - only the button text differs.
    var jtwc = !!(advFull && advFull.source === "jtwc");
    for (var i = 0; i < host.children.length; i++) {
      var b = host.children[i];
      var prod = b.getAttribute("data-prod");
      b.classList.toggle("active", prod === advTextProd);
      if (prod === "tcp") {
        b.textContent = jtwc ? "JTWC Warning" : "Public Advisory";
      } else if (prod === "tcd") {
        b.textContent = jtwc ? "Prognostic Reasoning" : "Discussion";
      }
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

  // ---- Recon mount (SAME viewer component as the main /recon/ page) -------
  // Lazy-load the shared TATRegions basemap helper + the ReconViewer class
  // from the main site, then mount it LOCKED to this storm (atcf_long matches
  // the recon manifest slug; name is the fallback match). The viewer hydrates
  // from CDN/recon/ - the isolated aircraft-recon feed.
  var reconViewer = null;
  function _loadScript(src, ready, cb, onerr) {
    if (ready()) { cb(); return; }
    var s = document.createElement("script");
    s.src = src;
    s.onload = function () { ready() ? cb() : onerr(); };
    s.onerror = onerr;
    document.head.appendChild(s);
  }
  // ---- Overview stacking map (lazy-loaded reusable module) ----
  var clMap = null;
  function initOverviewMap() {
    var root = document.getElementById("overview-map");
    if (!root || clMap) return;
    var storm = lastStorm || (typeof BAKED !== "undefined" ? BAKED : null);
    if (!storm) return;
    _loadScript(SITE_BASE + "/cyclolab_map.js",
      function () { return !!window.CycloLabMap; },
      function () {
        try { clMap = new window.CycloLabMap(root, { storm: storm, timeMode: settings.mapTime }); }
        catch (e) { if (window.console) console.warn("overview map failed", e); }
      },
      function () { if (window.console) console.warn("cyclolab_map.js failed to load"); });
  }
  function initRecon() {
    var root = document.getElementById("recon-viewer");
    var status = document.getElementById("recon-status");
    function fail() {
      if (status) status.querySelector("span").textContent =
        "Recon viewer failed to load - reload to retry.";
    }
    _loadScript(SITE_BASE + "/models/regions.js",
      function () { return !!window.TATRegions; },
      function () {
        _loadScript(SITE_BASE + "/recon/recon.js",
          function () { return !!window.ReconViewer; },
          function () {
            try {
              var nmEl = document.getElementById("storm-name");
              reconViewer = new window.ReconViewer(root, {
                base: CDN + "/recon",
                stormLock: FLOATER_ID || SID,
                stormName: nmEl ? (nmEl.textContent || "").trim() : "",
                startTab: "storms"
              });
            } catch (e) { fail(); }
          }, fail);
      }, fail);
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
  //
  // FINAL-GATE R2 #5 - THE PRESENTATION PATH: swapping img.src forces a
  // main-thread decode + raster per frame even when the bytes are
  // cached, and playback still stuttered on real (throttled) hardware.
  // Playback now pre-DECODES frames into ImageBitmaps (fetch ->
  // createImageBitmap, a bounded look-ahead ring) and PRESENTS by
  // drawImage onto a canvas from a rAF clock - the only per-frame
  // main-thread work is one GPU-friendly blit. Presented-frame
  // timestamps are recorded so cadence is MEASURABLE
  // (satState().presented); the contract - no interval > 2x the median
  // over a full loop at 4x CPU throttle - is pinned in tests. Engines
  // without canvas/createImageBitmap (jsdom) keep the img.src path
  // with the decode gate.
  var sat = { man: null, band: null, frames: [], idx: 0, playing: false,
              timer: null, gen: 0, preloaded: {},
              mode: null, bmp: {}, bmpKeys: [], raf: null, lastT: 0,
              holdT0: 0, presented: [], speed: 2,
              // auto-refresh (the manifest re-poll): pollTimer = the next
              // scheduled fetch; inactive = the floater stopped producing
              // frames (storm dropped from the index OR newest frame stale).
              pollTimer: null, inactive: false };
  var SAT_POLL_MS = 60000;          // re-poll the manifest while watching
  var SAT_POLL_BACKOFF_MS = 300000; // slower cadence once the floater is idle
  var SAT_STALE_MIN = 40;           // newest frame older than this = inactive
  var SAT_FRAME_MS = 200, SAT_AHEAD = 12, SAT_BMP_MAX = 28;
  // final-gate-3 #5: playback speed presets. The cadence CONTRACT (no
  // interval > 2x the median over a loop) holds at every speed because
  // the target interval scales uniformly - the median scales with it.
  var SAT_SPEEDS = [0.5, 1, 2, 4];
  function satFrameMs() { return SAT_FRAME_MS / sat.speed; }
  function satEl(id) { return document.getElementById("sat-" + id); }
  function satStatus(show, msg) {
    var box = satEl("status");
    box.style.display = show ? "flex" : "none";
    if (msg != null) box.querySelector("span").textContent = msg;
  }
  function satCanvasOk() {
    if (sat.mode !== null) return sat.mode === "canvas";
    var ok = false;
    try {
      var cv = satEl("canvas");
      ok = !!(window.createImageBitmap && cv && cv.getContext &&
              cv.getContext("2d"));
    } catch (e) { ok = false; }
    sat.mode = ok ? "canvas" : "img";
    return ok;
  }
  function satBitmapFor(i) {
    // decoded ImageBitmap for frame i, or null (decode kicked off).
    var f = sat.frames[i];
    if (!f) return null;
    var u = CDN + "/" + f.key;
    var e = sat.bmp[u];
    if (e) {
      // LRU refresh (adversarial-review find): a hit moves the key to
      // the ring's young end, so steady playback can never evict the
      // frames the clock is about to present.
      var ki = sat.bmpKeys.indexOf(u);
      if (ki >= 0 && ki !== sat.bmpKeys.length - 1) {
        sat.bmpKeys.splice(ki, 1);
        sat.bmpKeys.push(u);
      }
      return e.bm;
    }
    e = sat.bmp[u] = { bm: null };
    sat.bmpKeys.push(u);
    fetch(u).then(function (r) { return r.ok ? r.blob() : null; })
      .then(function (bl) { return bl ? createImageBitmap(bl) : null; })
      .then(function (bm) { e.bm = bm; })
      .catch(function () {});
    // bounded decode ring: evict (and close) the oldest entries.
    while (sat.bmpKeys.length > SAT_BMP_MAX) {
      var old = sat.bmpKeys.shift();
      var oe = sat.bmp[old];
      delete sat.bmp[old];
      if (oe && oe.bm && oe.bm.close) {
        try { oe.bm.close(); } catch (e2) {}
      }
    }
    return null;
  }
  function satAhead(i) {
    // decode-ahead window so the presentation clock never waits on the
    // network mid-loop.
    var n = sat.frames.length;
    if (!n) return;
    for (var k = 1; k <= Math.min(SAT_AHEAD, n - 1); k++) {
      satBitmapFor((i + k) % n);
    }
  }
  function satBlit(bm) {
    var cv = satEl("canvas");
    // Pin the canvas to a STABLE size and SCALE each frame to fill it. The
    // floater occasionally emits frames whose height wobbles by ~1px mid-loop
    // (a render-extent rounding flip, e.g. 1056x1056 <-> 1056x1055 across a
    // contiguous block, in every band) -- re-sizing the canvas to each frame's
    // native dimensions made the live loop visibly resize/reflow + jump on
    // those boundaries (the "seizure"). Re-fix the size only on a REAL change
    // (band switch / >2px); a 1px wobble is then absorbed by a sub-pixel scale,
    // never a layout resize. satFrameToCanvas already scale-fits, so the GIF
    // export was immune -- this aligns the live loop with it.
    if (!cv.width || Math.abs(cv.width - bm.width) > 2 ||
        Math.abs(cv.height - bm.height) > 2) {
      cv.width = bm.width; cv.height = bm.height;
    }
    cv.getContext("2d").drawImage(bm, 0, 0, cv.width, cv.height);
  }
  function satReadout(i) {
    var f = sat.frames[i];
    satEl("scrub").value = String(i);
    satEl("time").textContent =
      f.t.slice(0, 16).replace("T", " ") + "Z";
  }
  function satShow(i) {
    var n = sat.frames.length;
    if (!n) return;
    if (i < 0) i = 0;
    if (i >= n) i = n - 1;
    sat.idx = i;
    var f = sat.frames[i];
    if (satCanvasOk()) {
      satEl("img").style.display = "none";
      satEl("canvas").style.display = "";
      var bm = satBitmapFor(i);
      if (bm) {
        satBlit(bm);
      } else {
        // decode in flight: blit when ready if the user is still on
        // this frame; give up quietly on a dead object (~2.5 s).
        var gen = sat.gen, want = i, tries = 0;
        (function waitBlit() {
          if (gen !== sat.gen || sat.idx !== want || tries++ > 60) return;
          var b2 = satBitmapFor(want);
          if (b2) { satBlit(b2); return; }
          setTimeout(waitBlit, 40);
        })();
      }
      satAhead(i);
    } else {
      satEl("img").src = CDN + "/" + f.key;
    }
    satReadout(i);
  }
  function satPreload() {
    // newest-backwards over the ENTIRE band, ~3 in flight. The old
    // 12-frame window left ~90% of the loop un-cached; playback then
    // raced the network and the floater sector visibly lurched
    // (final-gate #4).
    var gen = ++sat.gen;
    var want = [];
    for (var k = sat.frames.length - 1; k >= 0; k--) want.push(sat.frames[k]);
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
    sat.gen++;          // invalidate in-flight waitBlits on BOTH paths
    var b = (sat.man && sat.man.bands && sat.man.bands[slug]) || null;
    // STRICTLY CHRONOLOGICAL + DEDUPED within the band (final-gate #4):
    // playback order is a hard guarantee here, not an upstream promise.
    var raw = ((b && b.frames) || []).slice();
    raw.sort(function (a, b2) { return a.t < b2.t ? -1 : a.t > b2.t ? 1 : 0; });
    sat.frames = [];
    for (var fi = 0; fi < raw.length; fi++) {
      if (!sat.frames.length ||
          raw[fi].t !== sat.frames[sat.frames.length - 1].t) {
        sat.frames.push(raw[fi]);
      }
    }
    satEl("scrub").max = String(Math.max(0, sat.frames.length - 1));
    satEl("band-label").textContent = b ? (b.label || slug) : "";
    // a band key can survive a server-side prune with zero frames left:
    // honest per-band empty state instead of freezing on the old frame
    // (canvas path included - the blitted pixels would otherwise stay).
    satEl("empty").style.display = sat.frames.length ? "none" : "block";
    if (!sat.frames.length) {
      satEl("img").removeAttribute("src");
      var cv0 = satEl("canvas");
      if (cv0) cv0.style.display = "none";
      satEl("time").textContent = "\u2014";
    }
    var host = satEl("bands");
    for (var i = 0; i < host.children.length; i++) {
      var btn = host.children[i];
      btn.classList.toggle("active", btn.getAttribute("data-slug") === slug);
    }
    satShow(prevT != null ? satNearest(prevT) : sat.frames.length - 1);
    // canvas mode decodes ahead of the playhead; the Image-object
    // warmer only serves the img fallback path.
    if (satCanvasOk()) satAhead(sat.idx); else satPreload();
  }
  function satStep(d) { satPause(); satShow(sat.idx + d); }
  function satPause() {
    sat.playing = false;
    if (sat.timer) { clearInterval(sat.timer); sat.timer = null; }
    if (sat.raf) { cancelAnimationFrame(sat.raf); sat.raf = null; }
    satEl("play").innerHTML = "&#9654; Play";
  }
  function satRafTick(ts) {
    if (!sat.playing) { sat.raf = null; return; }
    sat.raf = requestAnimationFrame(satRafTick);
    var fm = satFrameMs();
    if (ts - sat.lastT < fm - 1) return;
    var j = sat.idx >= sat.frames.length - 1 ? 0 : sat.idx + 1;
    var bm = satBitmapFor(j);
    if (!bm) {
      // hold the clock for an in-flight decode; skip a dead frame
      // after ~2 s so one bad object can't stall the loop.
      if (!sat.holdT0) sat.holdT0 = ts;
      if (ts - sat.holdT0 > 2000) {
        sat.holdT0 = 0;
        sat.idx = j; satReadout(j); satAhead(j);
        sat.lastT = ts;
      }
      return;
    }
    sat.holdT0 = 0;
    sat.idx = j;
    satBlit(bm);
    satReadout(j);
    satAhead(j);
    // drift-resistant cadence: lock to the (speed-scaled) frame grid,
    // resync only if the main thread fell more than a frame behind.
    sat.lastT = (ts - sat.lastT > 2 * fm) ? ts : sat.lastT + fm;
    // PRESENTED-frame log (final-gate-2 #5): the cadence contract is
    // asserted on these, not on swap counts.
    sat.presented.push(ts);
    if (sat.presented.length > 600) sat.presented.shift();
  }
  function satTogglePlay() {
    if (sat.playing) { satPause(); return; }
    if (!sat.frames.length) return;
    sat.playing = true;
    satEl("play").innerHTML = "&#10074;&#10074; Pause";
    if (satCanvasOk()) {
      sat.lastT = 0;
      sat.holdT0 = 0;
      satAhead(sat.idx);
      sat.raf = requestAnimationFrame(satRafTick);
      return;
    }
    sat.holds = 0;
    satArmFallbackTimer();
  }
  function satArmFallbackTimer() {
    if (sat.timer) clearInterval(sat.timer);
    sat.timer = setInterval(function () {
      var j = sat.idx >= sat.frames.length - 1 ? 0 : sat.idx + 1;
      // DECODE GATE (img fallback path): during playback a frame is
      // shown only once its preload has fully decoded - swapping src
      // to an in-flight URL races the network and the sector jumps
      // (final-gate #4). A dead frame is skipped after ~2s so one bad
      // object can't stall the loop.
      var im = sat.preloaded[CDN + "/" + sat.frames[j].key];
      var ready = im && im.complete && im.naturalWidth > 0;
      if (!ready && sat.holds < 10) { sat.holds++; return; }
      sat.holds = 0;
      satShow(j);
    }, satFrameMs());
  }
  // SECTOR-SOURCE REGISTRY (final-gate-2 #6, a seam - no second source
  // is built this round): the Satellite tab discovers per-storm sector
  // manifests through this table, so MESOSCALE SECTORS join as a
  // second entry when the meso pipeline exists (satellite roadmap
  // phase) - a registry entry, not a viewer rewrite. Band slugs merge
  // across sources (first registered source wins a collision). Today:
  // the GOES floater pipeline only.
  var SAT_SOURCES = [{
    id: "floater",
    top: CDN + "/floaters/manifest.json",
    resolve: function (top) {
      var storms = (top && top.storms) || [];
      for (var i = 0; i < storms.length; i++) {
        // FLOATER_ID is the bare atcf_long (e.g. wp072026). The floater index
        // keys NHC storms by that bare id but JTWC/WP storms by the agency-
        // prefixed sid (JTWC_WP072026) and invests by an unprefixed sid
        // (WP912026), so strip a leading agency token before comparing. The
        // bare match alone left every WPAC/JTWC named storm's Satellite tab
        // empty (it never resolved its live floater). Match on slug too as a
        // belt-and-suspenders against any id-format drift.
        var fid = String(storms[i].id).toLowerCase();
        if ((fid === FLOATER_ID
             || fid.replace(/^[a-z]+_/, "") === FLOATER_ID
             || String(storms[i].slug || "").toLowerCase() === FLOATER_SLUG) &&
            storms[i].manifest) {
          return CDN + "/" + storms[i].manifest;
        }
      }
      return null;
    }
  }];
  // ---- speed presets (final-gate-3 #5) ------------------------------------
  function satSetSpeed(mult) {
    sat.speed = mult;
    var host = satEl("speed");
    if (host) {
      for (var i = 0; i < host.children.length; i++) {
        var b = host.children[i];
        b.classList.toggle("active",
          parseFloat(b.getAttribute("data-speed")) === mult);
      }
    }
    // re-arm the running clock at the new cadence (rAF reads satFrameMs
    // live; the img-fallback interval must be rebuilt).
    if (sat.playing && sat.timer) satArmFallbackTimer();
  }
  function satBuildSpeed() {
    var host = satEl("speed");
    if (!host) return;
    host.innerHTML = "";
    SAT_SPEEDS.forEach(function (mult) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "hafs-seg";
      b.setAttribute("data-speed", String(mult));
      b.textContent = (mult === 1 ? "1" :
        (mult < 1 ? "0.5" : String(mult))) + "\u00d7";
      b.addEventListener("click", function () {
        satSetSpeed(parseFloat(this.getAttribute("data-speed")));
      });
      host.appendChild(b);
    });
    satSetSpeed(2);   // default playback speed: 2x (matches the /satellite/ viewers)
  }
  function satNudgeSpeed(dir) {
    var i = SAT_SPEEDS.indexOf(sat.speed);
    if (i < 0) i = 1;
    satSetSpeed(SAT_SPEEDS[Math.max(0,
      Math.min(SAT_SPEEDS.length - 1, i + dir))]);
  }

  // ---- client-side GIF export (final-gate-3 #5) ---------------------------
  // Self-contained GIF89a encoder (no CDN): a fixed 3-3-2 RGB palette so
  // the index is computed per pixel (no quantizer search), standard LZW,
  // looping. Frames come from the ALREADY-LOADED imagery (ImageBitmaps in
  // canvas mode, decoded <img>s in the fallback), so export adds no new
  // fetches. Export is capped + downscaled to keep the file sane on
  // desktop and mobile.
  var SAT_GIF_MAX_FRAMES = 48;     // cap (oldest-trimmed) - sane size
  var SAT_GIF_MAX_PX = 480;        // longest side after downscale
  function gifColorIndex(r, g, b) {
    return ((r >> 5) << 5) | ((g >> 5) << 2) | (b >> 6);  // 3-3-2
  }
  function gifGlobalPalette() {
    var pal = new Uint8Array(256 * 3);
    for (var i = 0; i < 256; i++) {
      var r3 = (i >> 5) & 7, g3 = (i >> 2) & 7, b2 = i & 3;
      pal[i * 3] = Math.round(r3 * 255 / 7);
      pal[i * 3 + 1] = Math.round(g3 * 255 / 7);
      pal[i * 3 + 2] = Math.round(b2 * 255 / 3);
    }
    return pal;
  }
  function gifLzw(indices, minCode, out) {
    // GIF variable-width LZW (faithful port of omggif's proven encoder).
    var clear = 1 << minCode, eoi = clear + 1;
    var next = eoi + 1, width = minCode + 1;
    var dict = new Map();
    var cur = 0, shift = 0;
    var block = [];
    function flushBits() {
      while (shift >= 8) {
        block.push(cur & 255); cur >>= 8; shift -= 8;
        if (block.length === 255) {
          out.push(255);
          for (var k = 0; k < 255; k++) out.push(block[k]);
          block = [];
        }
      }
    }
    function emit(code) { cur |= code << shift; shift += width; flushBits(); }
    emit(clear);
    var ib = indices[0];
    for (var i = 1; i < indices.length; i++) {
      var k = indices[i];
      var key = (ib << 8) | k;
      var code = dict.get(key);
      if (code !== undefined) { ib = code; continue; }
      emit(ib);
      if (next === 4096) {            // table full: reset
        emit(clear);
        dict.clear(); next = eoi + 1; width = minCode + 1;
      } else {
        if (next >= (1 << width)) width++;
        dict.set(key, next++);
      }
      ib = k;
    }
    emit(ib);
    emit(eoi);
    if (shift > 0) { block.push(cur & 255); }
    if (block.length) {
      out.push(block.length);
      for (var k2 = 0; k2 < block.length; k2++) out.push(block[k2]);
    }
    out.push(0);   // block terminator
  }
  function gifEncode(frames, w, h, delayCs) {
    // frames: array of Uint8ClampedArray RGBA (w*h*4). Returns Blob.
    var pal = gifGlobalPalette();
    var b = [];
    function str(s) { for (var i = 0; i < s.length; i++) b.push(s.charCodeAt(i)); }
    function u16(v) { b.push(v & 255, (v >> 8) & 255); }
    str("GIF89a");
    u16(w); u16(h);
    b.push(0xF7, 0, 0);                       // global table, 256 colors
    for (var i = 0; i < pal.length; i++) b.push(pal[i]);
    // NETSCAPE loop extension
    b.push(0x21, 0xFF, 0x0B);
    str("NETSCAPE2.0");
    b.push(0x03, 0x01, 0, 0, 0x00);
    frames.forEach(function (rgba) {
      // graphic control extension (delay, no transparency)
      b.push(0x21, 0xF9, 0x04, 0x00);
      u16(delayCs);
      b.push(0x00, 0x00);
      // image descriptor
      b.push(0x2C); u16(0); u16(0); u16(w); u16(h); b.push(0x00);
      var n = w * h, idx = new Uint8Array(n);
      for (var p = 0; p < n; p++) {
        idx[p] = gifColorIndex(rgba[p * 4], rgba[p * 4 + 1],
                               rgba[p * 4 + 2]);
      }
      b.push(8);                              // LZW minimum code size
      gifLzw(idx, 8, b);
    });
    b.push(0x3B);                             // trailer
    return new Blob([new Uint8Array(b)], { type: "image/gif" });
  }
  var satGifBusy = false;
  function satGifProgress(frac) {
    var box = satEl("gif-prog");
    if (!box) return;
    box.hidden = frac == null;
    if (frac == null) return;
    box.querySelector(".sat-gif-bar i").style.width =
      Math.round(frac * 100) + "%";
    box.querySelector(".sat-gif-pct").textContent =
      Math.round(frac * 100) + "%";
  }
  function satFrameToCanvas(i, cw, ch) {
    var cnv = document.createElement("canvas");
    cnv.width = cw; cnv.height = ch;
    var ctx = cnv.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cw, ch);
    var src = sat.bmp[CDN + "/" + sat.frames[i].key];
    src = src && src.bm;
    if (!src) src = sat.preloaded[CDN + "/" + sat.frames[i].key];
    if (src && (src.width || src.naturalWidth)) {
      try { ctx.drawImage(src, 0, 0, cw, ch); } catch (e) { return null; }
      return ctx.getImageData(0, 0, cw, ch).data;
    }
    return null;
  }
  function satExportGif() {
    if (satGifBusy || !sat.frames.length) return;
    var btn = satEl("gif");
    var first = sat.bmp[CDN + "/" + sat.frames[0].key];
    first = (first && first.bm) ||
            sat.preloaded[CDN + "/" + sat.frames[0].key];
    var sw = (first && (first.width || first.naturalWidth)) || 480;
    var sh = (first && (first.height || first.naturalHeight)) || 480;
    var scale = Math.min(1, SAT_GIF_MAX_PX / Math.max(sw, sh));
    var cw = Math.max(2, Math.round(sw * scale));
    var ch = Math.max(2, Math.round(sh * scale));
    // chronological subset, capped (evenly sampled if over the cap)
    var n = sat.frames.length;
    var order = [];
    for (var i = 0; i < n; i++) order.push(i);
    if (n > SAT_GIF_MAX_FRAMES) {
      var pick = [];
      for (var k = 0; k < SAT_GIF_MAX_FRAMES; k++) {
        pick.push(order[Math.round(k * (n - 1) /
          (SAT_GIF_MAX_FRAMES - 1))]);
      }
      order = pick;
    }
    var wasPlaying = sat.playing;
    satPause();
    satGifBusy = true;
    btn.disabled = true;
    satGifProgress(0);
    var rgbaFrames = [];
    var idx = 0;
    var delayCs = Math.max(2, Math.round(satFrameMs() / 10));
    function step() {
      if (idx >= order.length) { finish(); return; }
      var data = satFrameToCanvas(order[idx], cw, ch);
      if (data) rgbaFrames.push(data);
      idx++;
      satGifProgress(idx / order.length * 0.85);
      (window.requestAnimationFrame || setTimeout)(step);
    }
    function finish() {
      try {
        var blob = gifEncode(rgbaFrames, cw, ch, delayCs);
        satGifProgress(1);
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        var stamp = (sat.frames[sat.idx] &&
          sat.frames[sat.idx].t || "").replace(/[^0-9]/g, "").slice(0, 12);
        a.href = url;
        a.download = "cyclolab_" + (SID || "storm") + "_" +
          (sat.band || "sat") + "_" + (stamp || "loop") + ".gif";
        document.body.appendChild(a); a.click();
        document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
      } catch (e) {
        try { console.warn("[cyclolab] GIF export failed:", e); }
        catch (e2) {}
      }
      satGifBusy = false;
      btn.disabled = false;
      setTimeout(function () { satGifProgress(null); }, 1200);
      if (wasPlaying) satTogglePlay();
    }
    step();
  }

  // ---- auto-refresh: re-poll the floater manifest while watching ----------
  // Fetch the floater index -> this storm's sub-manifest -> merged bands map,
  // exactly as initSatellite's first-open chain. Returns null when the storm
  // is not in the index (resolve -> null) or no source yields bands.
  function satFetchBands() {
    return Promise.all(SAT_SOURCES.map(function (src) {
      return fetchJson(src.top).then(function (top) {
        var mu = top && src.resolve(top);
        return mu ? fetchJson(mu) : null;
      }).catch(function () { return null; });
    })).then(function (mans) {
      var bands = {}, got = false;
      mans.forEach(function (man) {
        if (!man || !man.bands) return;
        for (var bs in man.bands) {
          if (!man.bands.hasOwnProperty(bs) || bands[bs]) continue;
          bands[bs] = man.bands[bs]; got = true;
        }
      });
      return got ? bands : null;
    });
  }
  // chronological-ascending, time-deduped (the SAME contract satSelectBand
  // applies when it first loads a band - playback order is a hard guarantee).
  function satSortDedupe(frames) {
    var raw = (frames || []).slice();
    raw.sort(function (a, b) { return a.t < b.t ? -1 : a.t > b.t ? 1 : 0; });
    var out = [];
    for (var i = 0; i < raw.length; i++) {
      if (!out.length || raw[i].t !== out[out.length - 1].t) out.push(raw[i]);
    }
    return out;
  }
  function satAddBandButton(slug) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "hafs-seg";
    b.setAttribute("data-slug", slug);
    b.textContent = (sat.man.bands[slug] && sat.man.bands[slug].label) || slug;
    b.addEventListener("click", function () {
      satPause(); satSelectBand(this.getAttribute("data-slug"), true);
    });
    satEl("bands").appendChild(b);
  }
  function satFmtTime(t) {
    return String(t).slice(0, 16).replace("T", " ") + "Z";
  }
  // the floater is "inactive" if it dropped from the index (droppedFromIndex)
  // OR its newest frame is older than SAT_STALE_MIN. Show an honest last-frame
  // note instead of a caption that keeps promising fresh imagery.
  function satUpdateInactive(droppedFromIndex) {
    var note = satEl("inactive");
    if (!sat.frames.length) {
      sat.inactive = !!droppedFromIndex;
      if (note) { note.hidden = true; }
      return;
    }
    var newestT = sat.frames[sat.frames.length - 1].t;
    var ageMin = (Date.now() - new Date(newestT).getTime()) / 60000;
    sat.inactive = !!droppedFromIndex || !(ageMin <= SAT_STALE_MIN);
    if (note) {
      if (sat.inactive) {
        note.textContent = "Floater inactive \u2014 last frame " +
          satFmtTime(newestT);
        note.hidden = false;
      } else { note.hidden = true; note.textContent = ""; }
    }
  }
  // merge a freshly-polled bands map: refresh every band's frame list (so a
  // later band switch sees the new frames; a new band gets a rail button),
  // then APPEND frames newer than the loaded tail to the CURRENT band,
  // PRESERVING playback state. Newest+paused -> follow-live to the new
  // newest; playing -> the rAF loop advances into the appended frames on its
  // own (idx becomes < last, so the next tick steps in rather than wrapping);
  // paused/scrubbing on an OLDER frame -> idx stays, only the timeline grows.
  function satRepollApply(bands) {
    if (!bands) { satUpdateInactive(true); return; }
    if (!sat.man) sat.man = { bands: {} };
    for (var bs in bands) {
      if (!bands.hasOwnProperty(bs)) continue;
      var isNew = !sat.man.bands[bs];
      sat.man.bands[bs] = bands[bs];
      if (isNew) satAddBandButton(bs);
    }
    var nb = sat.man.bands[sat.band];
    var fresh = satSortDedupe(nb && nb.frames);
    // the current band was empty (server-pruned to zero) and now has frames:
    // adopt them through the normal loader (lands on newest).
    if (!sat.frames.length) {
      if (fresh.length) satSelectBand(sat.band, false);
      satUpdateInactive();
      return;
    }
    var lastT = sat.frames[sat.frames.length - 1].t;
    var wasNewest = sat.idx >= sat.frames.length - 1;
    var added = [];
    fresh.forEach(function (f) {
      if (f.t > lastT) { sat.frames.push(f); added.push(sat.frames.length - 1); }
    });
    if (added.length) {
      satEl("scrub").max = String(sat.frames.length - 1);
      // DECODE GATE: kick the appended frames' decode NOW so they are
      // presentable before the clock (or a follow-live jump) reaches them.
      if (satCanvasOk()) {
        added.forEach(function (i) { satBitmapFor(i); });
      } else {
        added.forEach(function (i) {
          var u = CDN + "/" + sat.frames[i].key;
          if (!sat.preloaded[u]) {
            var im = new Image(); sat.preloaded[u] = im; im.src = u;
          }
        });
      }
      if (wasNewest && !sat.playing) satShow(sat.frames.length - 1);
    }
    satUpdateInactive();
  }
  // poll only while the Satellite tab is OPEN + the document is VISIBLE;
  // never on ENDED archive pages; back off once the floater is inactive.
  function satPollActive() {
    var s = document.getElementById("sec-satellite");
    var hidden = (typeof document !== "undefined") && document.hidden;
    return !ENDED && !!s && s.classList.contains("active") && !hidden;
  }
  function satStopPoll() {
    if (sat.pollTimer) { clearTimeout(sat.pollTimer); sat.pollTimer = null; }
  }
  function satSchedulePoll(delay) {
    satStopPoll();
    if (ENDED) return;                 // frozen archive: never poll
    sat.pollTimer = setTimeout(function () {
      sat.pollTimer = null;
      if (!satPollActive()) return;    // hidden / off-tab: resume on re-entry
      satFetchBands().then(function (bands) {
        satRepollApply(bands);
        satSchedulePoll(sat.inactive ? SAT_POLL_BACKOFF_MS : SAT_POLL_MS);
      }).catch(function () { satSchedulePoll(SAT_POLL_MS); });
    }, delay);
  }
  function satStartPoll() {
    if (ENDED || sat.pollTimer || !satPollActive()) return;
    satSchedulePoll(sat.inactive ? SAT_POLL_BACKOFF_MS : SAT_POLL_MS);
  }

  function initSatellite() {
    satStatus(true, "Loading\u2026");
    // PER-SOURCE isolation (adversarial-review find): one source's throwing
    // resolve()/rejected fetch settles to null inside satFetchBands and the
    // terminal catch lands on the honest empty state. satFetchBands is the
    // SAME chain the auto-refresh poll reuses.
    satFetchBands().then(function (bands) {
      if (!bands) {
        satStatus(false);
        satEl("empty").style.display = "block";
        return;
      }
      sat.man = { bands: bands };
      satStatus(false);
      satEl("bands").innerHTML = "";
      var slugs = [];
      for (var slug in bands) {
        if (!bands.hasOwnProperty(slug)) continue;
        slugs.push(slug);
        satAddBandButton(slug);
      }
      if (!slugs.length) {
        satEl("empty").style.display = "block";
        return;
      }
      satSelectBand(slugs.indexOf("ir") >= 0 ? "ir" : slugs[0], false);
      satBuildSpeed();
      satEl("step-back").addEventListener("click", function () { satStep(-1); });
      satEl("step-fwd").addEventListener("click", function () { satStep(1); });
      satEl("play").addEventListener("click", satTogglePlay);
      satEl("gif").addEventListener("click", satExportGif);
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
        } else if (e.key === "+" || e.key === "=") {
          satNudgeSpeed(1); e.preventDefault();
        } else if (e.key === "-" || e.key === "_") {
          satNudgeSpeed(-1); e.preventDefault();
        }
      });
      // auto-refresh: flag staleness now + arm the manifest re-poll (no-op on
      // ENDED archive pages or when the tab/document is not visible).
      satUpdateInactive();
      satStartPoll();
    }).catch(function () {
      // never strand the spinner: any unexpected throw in the band
      // build lands on the honest empty state.
      satStatus(false);
      satEl("empty").style.display = "block";
    });
  }

  window.__lab = { openSec: openSec, setCategory: setCategory,
                   apply: apply, applyAdvisory: applyAdvisory,
                   odoSet: odoSet,
                   // Stage-3 deterministic hooks (tests + ops)
                   renderAdvTab: renderAdvTab,
                   intensityRows: intensityRows,
                   cone: function () { return coneHooks; },
                   satState: function () {
                     return { band: sat.band, idx: sat.idx,
                              frames: sat.frames.length,
                              frameTimes: sat.frames.map(function (f) {
                                return f.t; }),
                              frameKeys: sat.frames.map(function (f) {
                                return f.key; }),
                              playing: sat.playing,
                              mode: sat.mode,
                              speed: sat.speed,
                              gifBusy: satGifBusy,
                              inactive: sat.inactive,
                              polling: !!sat.pollTimer,
                              presented: sat.presented.slice() };
                   },
                   // deterministic single manifest re-poll (tests + ops):
                   // fetch + merge/append exactly as the timer would, now.
                   satPollNow: function () {
                     return satFetchBands().then(satRepollApply);
                   },
                   satSetSpeed: function (m) { satSetSpeed(m); },
                   satExportGif: function () { return satExportGif(); },
                   gifEncode: gifEncode,
                   setWindUnits: function (u) { setWindUnits(u); },
                   windUnits: function () { return settings.windUnits; },
                   windDisp: function (kt) { return windDisp(kt); },
                   // FG-R3 #1 palette hooks (board render + tests)
                   setWindPalette: function (k) {
                     window.__labWindPalette = k; },
                   tierColors: function (which) {
                     var p = resolveTierPalette(which || "Ring");
                     return { "34": p["34"], "50": p["50"], "64": p["64"] }; },
                   neonRGB: function (kt) { return neonRGB(kt); },
                   hafsViewer: function () { return hafsViewer; },
                   // FG-R3 #7/#8 overview-plot hooks (tests + ops)
                   renderTrackPlot: function (s) {
                     return renderTrackPlot(s || lastStorm); },
                   renderSwathPlot: function (s) {
                     return renderSwathPlot(s || lastStorm); },
                   // live PTC identity hooks (tests + ops)
                   isPtc: function () { return IS_PTC; },
                   ptcNow: function (s) { return ptcNow(s || lastStorm); } };
})();

// ---- Right-click -> copy an overview plot as a shareable PNG -----------------
// Every hand-rolled SVG overview plot (cone, intensity, wind & pressure, track,
// wind swath) + the SST hero image is copyable by RIGHT-CLICK: render to a @2x
// PNG carrying a title header + the @WeathermanAAA mark, put it on the clipboard
// as image/png (ClipboardItem), and toast "Copied". Safari (no image clipboard)
// downloads the PNG + toasts "Downloaded" -- never a silent fail. cors-safe
// (?cors=1, like the GIF export) so the canvas is not tainted. Only these plots
// opt in (cl-copyable) -- normal right-click everywhere else is untouched.
(function initPlotCopy() {
  var MARK = "@WeathermanAAA";
  var SITE = "cyclolab.triple-a-tropics.com";
  var SC = 2;                  // @2x -- it's a shareable
  var HEAD = 30, FOOT = 26;    // logical-px header/footer bands (x SC on canvas)
  var FONT = "system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif";
  // kind: "stage" = SVG + positioned HTML overlays (capture the whole stage so
  // the title lockup / legends / stat box come along); "svg" = self-contained
  // SVG (all chrome is in-SVG); "hero" = the SST hero (a baked <img> + overlays).
  var PLOTS = [["advcone", "stage"], ["trackplot", "stage"], ["swathplot", "stage"],
               ["intensity", "svg"], ["chart", "svg"], ["sst-hero-img", "hero"]];
  // Clean section titles for the header band (track/swath have no card <h3> --
  // their head lives in the in-stage lockup, which the band must not duplicate).
  var TITLES = {advcone: "Forecast cone", intensity: "Intensity forecast",
    chart: "Wind & pressure", trackplot: "Track history",
    swathplot: "Wind swath", "sst-hero-img": "Sea surface temperature"};
  var STYLE_PROPS = ["fill", "fill-opacity", "stroke", "stroke-width",
    "stroke-dasharray", "stroke-linecap", "stroke-linejoin", "stroke-opacity",
    "opacity", "font-family", "font-size", "font-weight", "font-style",
    "text-anchor", "letter-spacing", "color", "paint-order"];

  function toast(msg) {
    var t = document.getElementById("cl-toast");
    if (!t) {
      t = document.createElement("div"); t.id = "cl-toast";
      t.className = "cl-toast"; document.body.appendChild(t);
    }
    t.textContent = msg; t.classList.add("show");
    clearTimeout(t._h);
    t._h = setTimeout(function () { t.classList.remove("show"); }, 1900);
  }
  function titleFor(el) {
    return TITLES[el.id] || el.getAttribute("aria-label") || "CycloLab";
  }
  function stageFor(el) {
    return (el.closest && (el.closest(".adv-cone-stage") ||
      el.closest(".map-stage") || el.closest(".sst-hero"))) || null;
  }
  // The page CSS, MINUS @font-face + @media (brace-balanced): no external font
  // fetch (-> no canvas taint) and no responsive reflow inside the foreignObject.
  function stripAt(css, at) {
    var out = "", i = 0;
    while (i < css.length) {
      var idx = css.indexOf(at, i);
      if (idx < 0) { out += css.slice(i); break; }
      out += css.slice(i, idx);
      var b = css.indexOf("{", idx);
      if (b < 0) { i = idx + at.length; continue; }
      var depth = 1, j = b + 1;
      while (j < css.length && depth > 0) {
        if (css[j] === "{") depth++; else if (css[j] === "}") depth--;
        j++;
      }
      i = j;
    }
    return out;
  }
  var _css = null;
  function pageCss() {
    if (_css != null) return _css;
    var raw = "", ss = document.querySelectorAll("style");
    for (var i = 0; i < ss.length; i++) raw += ss[i].textContent + "\n";
    _css = stripAt(stripAt(raw, "@font-face"), "@media");
    return _css;
  }
  // Inline the live computed styles onto the clone -- the plots style via page
  // CSS classes (fills/fonts), which a standalone serialized SVG would lose.
  function inlineStyles(src, clone) {
    if (src.nodeType !== 1) return;
    var cs = window.getComputedStyle(src), st = "";
    for (var i = 0; i < STYLE_PROPS.length; i++) {
      var v = cs.getPropertyValue(STYLE_PROPS[i]);
      if (v) st += STYLE_PROPS[i] + ":" + v + ";";
    }
    clone.setAttribute("style", st + (clone.getAttribute("style") || ""));
    var sc = src.children || [], cc = clone.children || [];
    for (var j = 0; j < sc.length && j < cc.length; j++) inlineStyles(sc[j], cc[j]);
  }
  function svgDims(svg) {
    var vb = (svg.getAttribute("viewBox") || "").trim().split(/\s+/).map(Number);
    if (vb.length === 4 && vb[2] > 0 && vb[3] > 0) return [vb[2], vb[3]];
    var r = svg.getBoundingClientRect(); return [r.width || 1000, r.height || 600];
  }
  function compose(img, sw, sh, title, done, guard) {
    var hb = HEAD * SC, fb = FOOT * SC, pw = sw * SC, ph = sh * SC;
    var cv = document.createElement("canvas");
    cv.width = pw; cv.height = hb + ph + fb;
    var ctx = cv.getContext("2d");
    ctx.fillStyle = "#0b1322"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#eaf2ff"; ctx.textAlign = "left";
    ctx.font = "700 " + (15 * SC) + "px " + FONT;
    ctx.fillText(title, 11 * SC, hb / 2);
    ctx.fillStyle = "#7fa6d8"; ctx.textAlign = "right";
    ctx.font = "600 " + (11 * SC) + "px " + FONT;
    ctx.fillText("CycloLab", pw - 11 * SC, hb / 2);
    try { ctx.drawImage(img, 0, hb, pw, ph); } catch (e) { done(null); return; }
    // guard (foreignObject path only): a Safari blank-render or a taint must NOT
    // silently deliver a blank/failed PNG -- bail to the caller's fallback.
    // getImageData throws on a tainted canvas; a uniform mid-strip = a blank
    // render (a real plot's middle always crosses content).
    if (guard) {
      try {
        var mid = ctx.getImageData(0, hb + Math.floor(ph / 2), pw, 2).data;
        var uniform = true;
        for (var k = 4; k < mid.length; k += 4) {
          if (mid[k] !== mid[0] || mid[k + 1] !== mid[1] || mid[k + 2] !== mid[2]) {
            uniform = false; break;
          }
        }
        if (uniform) { done(null); return; }
      } catch (e) { done(null); return; }
    }
    var fy = hb + ph + fb / 2;
    ctx.fillStyle = "#7fa6d8"; ctx.textAlign = "left";
    ctx.font = "600 " + (11 * SC) + "px " + FONT;
    ctx.fillText(SITE, 11 * SC, fy);
    ctx.fillStyle = "#eaf2ff"; ctx.textAlign = "right";
    ctx.font = "700 " + (12 * SC) + "px " + FONT;
    ctx.fillText(MARK, pw - 11 * SC, fy);
    try { cv.toBlob(function (b) { done(b); }, "image/png"); }
    catch (e) { done(null); }      // tainted canvas
  }
  function download(blob, title) {
    if (!blob) { toast("Copy failed"); return; }
    var name = "cyclolab_" + String(title).toLowerCase()
      .replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") + ".png";
    var u = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = u; a.download = name; document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(u); }, 4000);
    toast("Downloaded");
  }
  // One copy path for desktop right-click AND mobile long-press. The clipboard
  // write is registered SYNCHRONOUSLY in the gesture with a Promise<Blob> (a lazy
  // ClipboardItem): the render resolves async WITHOUT losing the user gesture,
  // which is what makes image-copy work on iOS Safari + Android (and hardens
  // desktop Safari). Where image-clipboard is unsupported, or it rejects, the
  // same blob is downloaded -- so mobile always lands on copy OR download.
  function doCopy(el, kind, stage) {
    var title = titleFor(el), p;
    try { p = pngBlob(el, kind, stage); } catch (e) { toast("Copy failed"); return; }
    function dl() { p.then(function (b) { download(b, title); })
      .catch(function () { toast("Copy failed"); }); }
    if (window.ClipboardItem && navigator.clipboard && navigator.clipboard.write) {
      navigator.clipboard.write([new ClipboardItem({ "image/png": p })])
        .then(function () { toast("Copied"); })
        .catch(dl);                       // unsupported/denied image write -> download
    } else { dl(); }
  }
  // Produce the plot PNG as a Promise<Blob>: the rich foreignObject capture for
  // overlay/hero plots, the self-contained SVG for the rest; on FO failure it
  // falls back to the prior SVG-only / bare-img render so a copy is never WORSE
  // than before. Rejects only on total failure.
  function pngBlob(el, kind, stage) {
    var title = titleFor(el);
    return new Promise(function (resolve, reject) {
      function ok(b) { if (b) resolve(b); else reject(new Error("copy failed")); }
      if (kind === "hero")
        heroBlob(stage || el, el, title, function (b) { if (b) resolve(b); else imgBlob(el, title, ok); });
      else if (kind === "stage" && stage)
        stageBlob(stage, title, null, function (b) { if (b) resolve(b); else svgBlob(el, title, ok); });
      else
        svgBlob(el, title, ok);
    });
  }
  // --- blob producers: render the plot, then call done(blob|null). ---
  function svgBlob(svg, title, done) {
    var d = svgDims(svg), sw = d[0], sh = d[1];
    var clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    inlineStyles(svg, clone);
    if (!clone.getAttribute("viewBox")) clone.setAttribute("viewBox", "0 0 " + sw + " " + sh);
    clone.setAttribute("width", sw * SC); clone.setAttribute("height", sh * SC);
    var xml = new XMLSerializer().serializeToString(clone);
    var img = new Image();
    img.onload = function () { compose(img, sw, sh, title, done); };
    img.onerror = function () { done(null); };
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);
  }
  function imgBlob(el, title, done) {
    var src = el.currentSrc || el.src;
    if (!src) { done(null); return; }
    var u = src + (src.indexOf("?") >= 0 ? "&" : "?") + "cors=1";
    var img = new Image(); img.crossOrigin = "anonymous";
    img.onload = function () {
      var w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
      compose(img, w / SC, h / SC, title, function (b) {
        if (b) done(b);
        else fetch(u, { mode: "cors" }).then(function (r) { return r.blob(); })
          .then(function (b2) { done(b2); })   // tainted -> raw PNG blob
          .catch(function () { done(null); });
      });
    };
    img.onerror = function () { done(null); };
    img.src = u;
  }
  // Capture the WHOLE stage (SVG + the positioned HTML overlays -- title lockup,
  // wind-key/SSHS legend, stat box, marker glyphs) via a foreignObject clone, so
  // the copied PNG matches what's on screen. prep(clone) may mutate the clone
  // (the SST hero inlines its <img> as a data-URL). On ANY failure (Safari's
  // foreignObject->canvas is unreliable; a taint; an empty stage) it calls
  // done(null) so the caller can fall back to the prior SVG-only / bare-img path
  // -- the copy is never WORSE than before, only better.
  function stageBlob(stage, title, prep, done) {
    var r = stage.getBoundingClientRect();
    var W = Math.max(1, Math.round(r.width)), H = Math.max(1, Math.round(r.height));
    if (W < 4 || H < 4) { done(null); return; }
    var clone = stage.cloneNode(true);
    if (prep) prep(clone);
    // Pin the inner plot SVG to the stage box: #trackplot/#swathplot are sized in
    // vh (clamp(...,56vh,...)), which re-resolves against the FO viewport (not the
    // page) and would shrink the map away from its overlays. 100% tracks the
    // pinned stage. (advcone is already height:100%; the SST hero has no inner svg.)
    var isvg = clone.querySelector && clone.querySelector("svg");
    if (isvg) isvg.style.cssText = "width:100%;height:100%;display:block;" + (isvg.style.cssText || "");
    // Pin exact px so the foreignObject doesn't reflow on vh/clamp/% rules.
    clone.style.cssText = "position:relative;width:" + W + "px;height:" + H +
      "px;margin:0;overflow:hidden;" + (clone.style.cssText || "");
    // Carry the live per-category custom props onto the wrapper: the cloned
    // overlays use var(--cat-accent)/--ac-rail etc. (set on <html data-cat=...>),
    // which would otherwise fall back to the :root default inside the FO.
    var ds = window.getComputedStyle(document.documentElement), vars = "";
    ["--cat-accent", "--cat-ink", "--cat-ramp", "--ac-rail"].forEach(function (k) {
      var v = ds.getPropertyValue(k); if (v) vars += k + ":" + v.trim() + ";";
    });
    // Suppress animations/transitions: a data:-URL SVG renders CSS animations
    // FROZEN at t=0, so entrance anims (.ac-icon ac-pop -> scale(0)/opacity(0))
    // would capture INVISIBLE; the reduced-motion escape lives in a stripped
    // @media. animation:none -> each element renders at its settled base.
    var settle = "*{animation:none!important;transition:none!important}";
    var body = '<div xmlns="http://www.w3.org/1999/xhtml" style="width:' + W +
      'px;height:' + H + 'px;overflow:hidden;background:#101a2c;' + vars + '"><style>' +
      pageCss() + settle + '</style>' + new XMLSerializer().serializeToString(clone) + '</div>';
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + (W * SC) +
      '" height="' + (H * SC) + '" viewBox="0 0 ' + W + ' ' + H +
      '"><foreignObject width="' + W + '" height="' + H + '">' + body +
      '</foreignObject></svg>';
    var img = new Image();
    img.onload = function () {
      if (!img.naturalWidth || !img.naturalHeight) { done(null); return; }  // Safari blank-load
      compose(img, W, H, title, done, true);
    };
    img.onerror = function () { done(null); };
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }
  // Hero (SST): inline the PNG as a data-URL so the foreignObject has no external
  // fetch (a cross-origin <img> inside a foreignObject taints the canvas), then
  // capture the stage. On any failure done(null) -> caller falls back to imgBlob.
  function heroBlob(stage, imgEl, title, done) {
    var src = imgEl.currentSrc || imgEl.src || "";
    if (!src) { done(null); return; }
    var u = src + (src.indexOf("?") >= 0 ? "&" : "?") + "cors=1";
    fetch(u, { mode: "cors" }).then(function (r) {
      if (!r.ok) throw new Error("hero fetch " + r.status);
      return r.blob();
    }).then(function (blob) {
      var fr = new FileReader();
      fr.onload = function () {
        var dataUrl = fr.result;
        stageBlob(stage, title, function (clone) {
          var imgs = clone.querySelectorAll ? clone.querySelectorAll("img") : [];
          for (var i = 0; i < imgs.length; i++) {
            imgs[i].setAttribute("src", dataUrl);
            imgs[i].removeAttribute("srcset");
            imgs[i].removeAttribute("crossorigin");
          }
        }, done);
      };
      fr.onerror = function () { done(null); };
      fr.readAsDataURL(blob);
    }).catch(function () { done(null); });
  }
  function wire() {
    PLOTS.forEach(function (p) {
      var el = document.getElementById(p[0]);
      if (!el) return;
      var kind = p[1], stage = stageFor(el);
      // Wire the gesture on the STAGE for overlay/hero plots (it catches a
      // right-click anywhere in the stage, incl. one that falls THROUGH the
      // pointer-events:none SST img); on the SVG itself for self-contained plots.
      var target = (kind === "svg" || !stage) ? el : stage;
      target.classList.add("cl-copyable");
      // Desktop: right-click.
      target.addEventListener("contextmenu", function (e) {
        e.preventDefault();
        doCopy(el, kind, stage);
      });
      // Mobile: long-press (parallels right-click; no on-plot UI). Fire on
      // touchend after a stationary >=450ms hold, so the clipboard write lands
      // in a real user-gesture event -- a setTimeout fire would NOT count as
      // one. A touchmove past a few px = a scroll/pan -> cancel. CSS
      // (-webkit-touch-callout:none on .cl-copyable) hides the iOS image-callout
      // during the hold.
      var t0 = 0, x0 = 0, y0 = 0, moved = false;
      target.addEventListener("touchstart", function (e) {
        if (!e.touches || e.touches.length !== 1) { t0 = 0; return; }
        moved = false; t0 = Date.now();
        x0 = e.touches[0].clientX; y0 = e.touches[0].clientY;
      }, { passive: true });
      target.addEventListener("touchmove", function (e) {
        if (!t0 || !e.touches || !e.touches.length) return;
        if (Math.abs(e.touches[0].clientX - x0) > 10 ||
            Math.abs(e.touches[0].clientY - y0) > 10) { moved = true; t0 = 0; }
      }, { passive: true });
      target.addEventListener("touchend", function (e) {
        var held = t0 && !moved && (Date.now() - t0) >= 450;
        t0 = 0;
        if (!held) return;
        e.preventDefault();          // swallow the trailing click / ghost-tap
        doCopy(el, kind, stage);
      });
      target.addEventListener("touchcancel", function () { t0 = 0; moved = false; });
    });
  }
  if (document.readyState !== "loading") wire();
  else document.addEventListener("DOMContentLoaded", wire);
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


_PTC_NUMBER_WORDS = None


def _ptc_number_words() -> frozenset:
    """NHC's spelled-out designation numbers ("ONE".."FIFTY-NINE") - the
    placeholder name a depression/PTC wears before it is named."""
    global _PTC_NUMBER_WORDS
    if _PTC_NUMBER_WORDS is None:
        ones = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN",
                "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN",
                "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN",
                "NINETEEN"]
        tens = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY"]
        s = set()
        for n in range(1, 60):
            s.add(ones[n] if n < 20
                  else tens[n // 10] + (("-" + ones[n % 10]) if n % 10 else ""))
        _PTC_NUMBER_WORDS = frozenset(s)
    return _PTC_NUMBER_WORDS


def _is_named_tc(storm: dict) -> bool:
    """A genuine named/designated tropical cyclone: TS-or-stronger SSHWS AND a
    REAL NHC name (not the "ONE"/"TWO" designation placeholder, an invest, or
    the raw sid). VETOES the is_ptc dress - a named TC is never "potential", so
    a fresh bake follows NHC the moment it names a system even if ace_core's
    is_ptc lags. Durable mirror of the inline JS isNamedTC() - keep in lock-step."""
    cat = (storm.get("current_category") or "").upper()
    if cat != "TS" and not (
            len(cat) == 2 and cat[0] == "C" and cat[1] in "12345"):
        return False
    nm = (storm.get("name") or "").strip().upper()
    if not nm or not nm[0].isalpha():
        return False
    # letters / space / hyphen / apostrophe only -> a real name (the raw sid
    # carries digits + underscore and is excluded here).
    if not all(c.isalpha() or c in " -'" for c in nm):
        return False
    return (nm not in _ptc_number_words()
            and nm not in {"INVEST", "UNNAMED", "NAMELESS"})


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
    """Bake a stat's initial value as PLAIN TEXT (final-gate-3 #2: the
    odometer is gone). The live odoSet() writes exactly this same plain
    text, so the no-JS render and the hydrated render are pixel-identical
    at rest - there is no cell/strip form to reconcile."""
    return _esc(str(text))


def render_page(storm: dict, *, feed_url: str, adv_url: str | None = None,
                ended: bool = False, loader: str = "b",
                og_image_url: str | None = None,
                sst_base: str | None = None) -> str:
    """Render one storm's shell. ``loader`` selects the loading screen:
    "b" (eye opens - THE CHOSEN loader, AD R3 default) or the other
    prototypes "a"/"c"/"d"; "" = plain wipe. ``sst_base`` is the URL
    base of the per-storm SST hero layers (final-gate-2 #1); default =
    the live Worker path."""
    ids = parse_sid(storm["sid"])
    is_invest = ids.is_invest or bool(storm.get("is_invest"))
    # A Potential Tropical Cyclone (ace_core is_ptc): a DESIGNATED system NHC is
    # advising on while still a DB/DS disturbance. It wears the INVEST visual
    # identity (grey + red X + formation pill) under its REAL designation, but —
    # unlike a 90-99 invest — KEEPS its cone + advisories + Models tab, because
    # NHC is actively advising on it. is_invest and is_ptc are mutually exclusive.
    # is_ptc follows the LIVE classification, never a stale flag: a NAMED TS+
    # system is never "potential", so the bake sheds the PTC dress the moment
    # NHC names it - even if the feed's is_ptc still lags (mirror of the inline
    # JS ptcNow()/isNamedTC()).
    is_ptc = (bool(storm.get("is_ptc")) and not is_invest
              and not _is_named_tc(storm))
    cat = storm.get("current_category") or "TD"
    if cat not in CAT_TOKENS:
        cat = "TD"
    name = (storm.get("name") or ids.nhc_id).upper()
    last = (storm.get("points") or [{}])[-1]
    wind = last.get("wind_kt")
    chip = {"TD": "Tropical Depression", "TS": "Tropical Storm",
            "C1": "Category 1", "C2": "Category 2", "C3": "Category 3",
            "C4": "Category 4", "C5": "Category 5"}.get(cat, cat)
    # Stage C - an invest gets a GREY identity (data-invest CSS overrides the
    # category vars) + a red-X glyph; "INVEST AREA" not a category type word. A
    # PTC reuses that grey/X identity (data-ptc) but its banner reads the NHC
    # classification "POTENTIAL TROPICAL CYCLONE", NOT a category-derived word.
    if is_ptc:
        type_word = "POTENTIAL TROPICAL CYCLONE"
    elif is_invest:
        type_word = "INVEST AREA"
    else:
        type_word = _type_word(cat, ids.basin)
    # No category chip for an invest OR a PTC (a PTC accrues no category), nor
    # for a plain TD/TS (chip is reserved for hurricanes C1-C5).
    chip_hidden = is_invest or is_ptc or cat in ("TD", "TS")
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
            .replace("__CAT_LABEL__",
                     _esc("PTC" if is_ptc else _sshs_label(cat)))
            .replace("__CAT_ODO__",
                     _odo_static("PTC" if is_ptc else _sshs_label(cat)))
            .replace("__VMAX_A11Y__", _esc(
                round(float(wind)) if wind is not None else "—"))
            .replace("__VMAX_ODO__", _odo_static(
                round(float(wind)) if wind is not None else "—"))
            .replace("__NAME__", _esc(name))
            .replace("__TYPE_WORD__", _esc(type_word.upper()))
            .replace("__CHIP__", _esc(chip))
            .replace("__CHIP_STYLE__",
                     ' style="display:none"' if chip_hidden else "")
            .replace("__IS_INVEST__", "true" if is_invest else "false")
            .replace("__IS_PTC__", "true" if is_ptc else "false")
            .replace("__OG_TITLE__", _esc(og_title))
            .replace("__OG_DESC__", _esc(og_desc))
            .replace("__PAGE_PATH__", _esc(page_url_path(storm["sid"])))
            .replace("__SID__", _esc(storm["sid"]))
            .replace("__SPAWN_SID__", _esc(storm.get("spawn_sid") or ""))
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
            .replace("__FLOATER_SLUG__",
                     _esc(f"{ids.basin.lower()}{ids.number:02d}"))
            .replace("__ADV_URL__", _esc(adv_url or adv_key(storm["sid"])))
            .replace("__SST_BASE__", _esc(
                (sst_base or f"/cyclolab/{ids.sid}/sst").rstrip("/")))
            .replace("__BASIN__", ids.basin)
            .replace("__LOADER__", _esc(loader))
            .replace("__SSHS_JSON__", json.dumps(SSHS_COLORS))
            .replace("__ENDED__", "true" if ended else "false")
            .replace("__BAKED__", baked))
    attrs = (("data-invest " if is_invest else "")
             + ("data-ptc " if is_ptc else "")
             + ("data-ended " if ended else ""))
    if attrs:
        html = html.replace("<html lang=\"en\" data-cat=",
                            f"<html lang=\"en\" {attrs}data-cat=")
    return html
