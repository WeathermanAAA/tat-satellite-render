#!/usr/bin/env python3
"""
generate_board.py - CycloLab cone forecast-point placard RESTYLE options board.

ART-GATED (TWEAK 1B item 2): renders 2-3 "storm-bug" pill treatments side by
side for Andrew to PICK. Nothing here ships to the live cone - the live placard
markup (cyclolab_shell.py renderAdvCone) is untouched. Once a treatment is
chosen, it gets wired into the cone's existing leaderless / collision-aware
placement sweep (which is a STANDING call and is NOT being restyled).

Faithful to house style: the real rotating cyclone glyph (HURRICANE_PATH), the
canonical ace_core SSHS palette + the approved pill glass ramp (CAT_TOKENS),
Metropolis, tabular-nums, light ink (white + dark stroke, never a black flip),
and only the lab-spin motion. Wind shown in mph (production stays unit-toggle
aware via windDisp/windUnitLabel); times are the point's valid_utc in UTC.
"""
from __future__ import annotations

import datetime as dt
import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ace_core import SSHS_COLORS          # noqa: E402
from cyclolab_shell import HURRICANE_PATH, CAT_TOKENS   # noqa: E402

DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def dow_hour(iso: str) -> str:
    """'2026-06-24T19:00:00Z' -> 'Wed 7 PM' (UTC, the deterministic shell convention)."""
    d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    h = d.hour
    ap = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{DOW[d.weekday() == 6 and 0 or (d.weekday() + 1) % 7]} {h12} {ap}"


def _dow(iso: str) -> str:
    d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return DOW[(d.weekday() + 1) % 7]


def time_label(iso: str) -> str:
    d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    h = d.hour
    ap = "AM" if h < 12 else "PM"
    return f"{_dow(iso)} {h % 12 or 12} {ap}"


def time_label_compact(iso: str) -> str:
    d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    h = d.hour
    ap = "a" if h < 12 else "p"
    return f"{_dow(iso)} {h % 12 or 12}{ap}"


def mph(kt: int) -> int:
    return int(round((kt * 1.1507794) / 5.0) * 5)


def sshs_label(cat: str) -> str:
    if cat == "TD":
        return "D"
    if cat == "TS":
        return "S"
    return cat.replace("C", "") or "D"


# storm-centered ±41 glyph scaled into a badge of radius r (centred at cx,cy).
def glyph(cx: float, cy: float, r: float, fill: str, letter: str,
          letter_px: float) -> str:
    s = (r * 1.78) / 82.0   # path spans ~±41 -> 82; ~0.78 fill of the disc
    return (
        f'<g class="spin" transform="translate({cx},{cy})">'
        f'<g transform="scale({s:.4f})"><path d="{HURRICANE_PATH}" '
        f'fill="{fill}" stroke="rgba(0,0,0,0.35)" stroke-width="2"/></g>'
        f'<text x="0" y="{letter_px*0.34:.1f}" text-anchor="middle" '
        f'font-size="{letter_px:.0f}" font-weight="800" fill="#fff" '
        f'stroke="rgba(0,0,0,0.45)" stroke-width="0.6" paint-order="stroke" '
        f'style="font-variant-numeric:tabular-nums">{letter}</text></g>')


INK = 'fill="#fff" stroke="rgba(0,0,0,0.42)" stroke-width="0.7" paint-order="stroke"'


def pill(treatment: str, cat: str, kt: int, iso: str) -> str:
    """One placard SVG for a treatment + forecast point."""
    tok = CAT_TOKENS[cat]
    accent = SSHS_COLORS[cat]
    glabel = sshs_label(cat)
    wind = f"{mph(kt)} mph"
    when = time_label(iso)

    if treatment == "A":   # Broadcast chip: badge-left, category glass pill, 2-line
        w, h = 182, 52
        svg = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
        svg.append(
            f'<rect x="1" y="1" width="{w-2}" height="{h-2}" rx="{(h-2)/2}" '
            f'fill="url(#pillg-{cat})" stroke="rgba(0,0,0,0.30)" stroke-width="1.2"/>')
        # legibility scrim on the bright ramps (keeps light ink readable)
        svg.append(f'<rect x="1" y="1" width="{w-2}" height="{h-2}" rx="{(h-2)/2}" '
                   f'fill="rgba(8,13,22,0.16)"/>')
        bcx = 27
        svg.append(f'<circle cx="{bcx}" cy="{h/2}" r="17.5" fill="rgba(8,13,22,0.55)" '
                   f'stroke="rgba(255,255,255,0.22)" stroke-width="1"/>')
        svg.append(glyph(bcx, h/2, 17.5, accent, glabel, 19))
        tx = 52
        svg.append(f'<text x="{tx}" y="20" font-size="11.5" font-weight="700" '
                   f'letter-spacing="0.8" fill="rgba(255,255,255,0.9)" '
                   f'style="text-transform:uppercase">{when}</text>')
        svg.append(f'<text x="{tx}" y="40" font-size="21" font-weight="800" '
                   f'{INK} style="font-variant-numeric:tabular-nums">{wind}</text>')
        svg.append('</svg>')
        return "".join(svg)

    if treatment == "B":   # Stat card: dark glass + SSHS accent rail, stacked
        w, h = 132, 74
        svg = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
        svg.append(f'<rect x="1" y="1" width="{w-2}" height="{h-2}" rx="9" '
                   f'fill="rgba(8,13,22,0.88)" stroke="rgba(44,58,82,0.6)" stroke-width="1"/>')
        svg.append(f'<rect x="1" y="1" width="4" height="{h-2}" rx="2" fill="{accent}"/>')
        svg.append(f'<rect x="1" y="1" width="{w-2}" height="3.5" fill="{accent}" opacity="0.85"/>')
        bcx, bcy = 26, 26
        svg.append(f'<circle cx="{bcx}" cy="{bcy}" r="14" fill="rgba(255,255,255,0.05)"/>')
        svg.append(glyph(bcx, bcy, 14, accent, glabel, 15))
        svg.append(f'<text x="48" y="22" font-size="11" font-weight="700" '
                   f'letter-spacing="1.2" fill="{accent}" '
                   f'style="text-transform:uppercase">{"CAT "+glabel if cat[0]=="C" else cat}</text>')
        svg.append(f'<text x="{w/2}" y="51" text-anchor="middle" font-size="22" '
                   f'font-weight="800" {INK} style="font-variant-numeric:tabular-nums">{wind}</text>')
        svg.append(f'<text x="{w/2}" y="67" text-anchor="middle" font-size="11" '
                   f'font-weight="700" letter-spacing="0.5" fill="#8fa2bd" '
                   f'style="text-transform:uppercase">{when}</text>')
        svg.append('</svg>')
        return "".join(svg)

    # treatment C - minimal inline bug: dark pill, accent glyph, single tight row
    w, h = 150, 32
    whenc = time_label_compact(iso)
    svg = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect x="1" y="1" width="{w-2}" height="{h-2}" rx="{(h-2)/2}" '
               f'fill="rgba(8,13,22,0.84)" stroke="{accent}" stroke-width="1.3" '
               f'stroke-opacity="0.85"/>')
    svg.append(f'<circle cx="16" cy="{h/2}" r="11.5" fill="rgba(255,255,255,0.05)"/>')
    svg.append(glyph(16, h/2, 11.5, accent, glabel, 12.5))
    svg.append(f'<text x="33" y="{h/2+4.5}" font-size="13" font-weight="700" {INK} '
               f'style="font-variant-numeric:tabular-nums">'
               f'<tspan fill="rgba(255,255,255,0.82)" font-weight="600">{whenc}</tspan>'
               f'<tspan dx="6" fill="rgba(120,150,180,0.7)">&#183;</tspan>'
               f'<tspan dx="6">{wind}</tspan></text>')
    svg.append('</svg>')
    return "".join(svg)


# --- sample forecast points spanning the SSHS palette (an intensifying recurve) -
POINTS = [
    ("NOW",   "TD", 30,  "2026-06-19T18:00:00Z"),
    ("+24h",  "TS", 55,  "2026-06-20T19:00:00Z"),
    ("+48h",  "C1", 75,  "2026-06-21T19:00:00Z"),
    ("+72h",  "C2", 92,  "2026-06-22T19:00:00Z"),
    ("+96h",  "C3", 110, "2026-06-23T19:00:00Z"),
    ("+120h", "C4", 125, "2026-06-24T19:00:00Z"),
]

TREATMENTS = [
    ("A", "Broadcast chip",
     "Category-glass pill (TV lower-third). Badge-left glyph + stacked time / wind. Boldest, most on-air."),
    ("B", "Stat card",
     "Dark glass card with SSHS accent rail + top bar. Glyph + category eyebrow, big wind, muted time. Data-forward, restrained."),
    ("C", "Minimal bug",
     "Compact dark inline pill, SSHS-edged, accent glyph + one tight row. Smallest footprint - kindest to a crowded cone."),
]


def grad_defs() -> str:
    out = ['<svg width="0" height="0" style="position:absolute"><defs>']
    for k, c in CAT_TOKENS.items():
        out.append(
            f'<linearGradient id="pillg-{k}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{c["edge"]}"/>'
            f'<stop offset="22%" stop-color="{c["mid"]}"/>'
            f'<stop offset="50%" stop-color="{c["accent"]}"/>'
            f'<stop offset="78%" stop-color="{c["mid"]}"/>'
            f'<stop offset="100%" stop-color="{c["edge"]}"/></linearGradient>')
    out.append('</defs></svg>')
    return "".join(out)


def build() -> str:
    cols = []
    for code, name, desc in TREATMENTS:
        rows = []
        for taul, cat, kt, iso in POINTS:
            rows.append(
                f'<div class="row"><div class="tau">{html.escape(taul)}</div>'
                f'<div class="pill">{pill(code, cat, kt, iso)}</div></div>')
        cols.append(
            f'<section class="col"><div class="chead"><span class="ccode">{code}</span>'
            f'<span class="cname">{html.escape(name)}</span></div>'
            f'<p class="cdesc">{html.escape(desc)}</p>{"".join(rows)}</section>')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>
  :root {{ --bg:#0b1622; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:radial-gradient(1200px 600px at 50% -10%, #13243a, #0b1622 60%);
    color:#e8eef6; font-family:Metropolis,"Helvetica Neue",Arial,sans-serif; padding:30px 34px 40px; }}
  h1 {{ font-size:20px; font-weight:800; margin:0 0 4px; letter-spacing:.2px; }}
  h1 .brand {{ color:#2b9cff; }}
  .sub {{ color:#8fa2bd; font-size:12.5px; margin:0 0 22px; max-width:1100px; line-height:1.5; }}
  .sub b {{ color:#cdd9e8; }}
  .board {{ display:grid; grid-template-columns:repeat(3,1fr); gap:26px; max-width:1180px; }}
  .col {{ background:rgba(8,13,22,0.45); border:1px solid rgba(44,58,82,0.5);
    border-radius:12px; padding:16px 16px 20px; }}
  .chead {{ display:flex; align-items:baseline; gap:9px; margin-bottom:4px; }}
  .ccode {{ font-size:13px; font-weight:800; color:#0b1622; background:#2b9cff;
    border-radius:6px; padding:2px 8px; }}
  .cname {{ font-size:15px; font-weight:800; }}
  .cdesc {{ color:#8fa2bd; font-size:11.5px; line-height:1.45; margin:0 0 16px; min-height:46px; }}
  .row {{ display:flex; align-items:center; gap:14px; padding:9px 2px;
    border-top:1px solid rgba(44,58,82,0.32); }}
  .row:first-of-type {{ border-top:0; }}
  .tau {{ width:46px; font-size:11px; font-weight:700; color:#6f87a3;
    text-transform:uppercase; letter-spacing:.5px; font-variant-numeric:tabular-nums; }}
  .pill {{ flex:1; }}
  .foot {{ color:#6f87a3; font-size:11px; margin-top:22px; max-width:1180px; line-height:1.55; }}
  @keyframes lab-spin {{ from {{ transform:rotate(360deg); }} to {{ transform:rotate(0deg); }} }}
  .spin {{ transform-box:fill-box; transform-origin:center; }}
</style></head><body>
{grad_defs()}
<h1><span class="brand">CycloLab</span> &middot; Cone forecast-point placard &middot; storm-bug restyle &mdash; OPTIONS BOARD</h1>
<p class="sub">Pick a pill treatment. All three keep the <b>leaderless, collision-aware</b> placement (no connector lines &mdash; a standing call, not being restyled) and the existing <b>spinning category glyph</b>; they are colour-coded by the canonical SSHWS palette, ink stays light, wind is <b>mph</b> (production follows the kt/mph toggle), and time is each point's <b>valid_utc</b>. Sample track: an intensifying recurve TD&rarr;Cat&nbsp;4. <b>Stills only &mdash; nothing is deployed; holding for your sign-off.</b></p>
<div class="board">{"".join(cols)}</div>
<p class="foot">Glyph, palette and pill glass ramp are the production constants (HURRICANE_PATH / ace_core.SSHS_COLORS / CAT_TOKENS). Font is Metropolis (falls back if the webfont is absent in this still). The chosen treatment will be wired into renderAdvCone's placard markup; the placement sweep is untouched.</p>
</body></html>"""


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "placard_board.html")
    with open(out, "w") as f:
        f.write(build())
    print("wrote", out)
