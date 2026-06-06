"""CycloLab per-storm shell - template + renderer (Stage 2).

The poller renders this template on storm birth and PUTs it via
R2Sink.write_html at cyclolab_pages.page_key(sid) (the Worker-pinned
edge contract). One self-contained document per storm: inline CSS
(the 7 category token sets + shell layout + motion), inline vanilla JS
(hydration on the poll+diff-merge discipline), real per-storm OG tags.
No CDN deps, no external CSS - the page must outlive site refactors.

LAYOUT (CYCLOLAB_DESIGN.md §4): desktop = left sidebar (identity
banner / section nav with accent rail / back-to-map) + content stage;
<=640px the SAME DOM rearranges to a slim top bar + bottom tab bar.

VISUAL (§5): one `data-cat` attribute on <html> switches the token set
(--cat-ramp/--cat-accent/--cat-ink). Ramps are 5-stop banded gradients
(dark edge / lit middle / dark edge - the LIVE-STATUS chrome gloss).
Body text and data tables are NEVER tinted.

MOTION (§6): ~4-5s, transform/opacity only, state-change triggered;
prefers-reduced-motion lands every animation on its final frame. The
hydration JS adds `data-anim` hooks; tests drive them deterministically.

V1 sections: Overview implemented (stats banner + ACE odometer + track
map + wind/pressure SVG timeline); Satellite/Models/Advisories are
stubbed panels wired for lazy init (Stages 3-4 fill them).
"""
from __future__ import annotations

import html as _html
import json

from cyclolab_pages import adv_key, page_url_path
from storm_ids import parse_sid

# --------------------------------------------------------------------------
# §5 - the seven category token sets (5-stop banded ramps, status-head gloss)
# --------------------------------------------------------------------------
# anchor: the flat accent; edge/mid derived shades keep one construction.
CAT_TOKENS: dict[str, dict] = {
    "TD": {"edge": "#16324a", "mid": "#2c5a80", "accent": "#3f7cab", "ink": "#ffffff"},
    "TS": {"edge": "#0d3b2a", "mid": "#1d6f4f", "accent": "#2aa169", "ink": "#ffffff"},
    "C1": {"edge": "#4a3a08", "mid": "#9a7a14", "accent": "#d9a91f", "ink": "#0a1324"},
    "C2": {"edge": "#4d2a0c", "mid": "#a35a1c", "accent": "#e07b28", "ink": "#0a1324"},
    "C3": {"edge": "#46140f", "mid": "#992b21", "accent": "#d23b2e", "ink": "#ffffff"},
    "C4": {"edge": "#471035", "mid": "#9c2273", "accent": "#d62fa0", "ink": "#ffffff"},
    "C5": {"edge": "#2a1454", "mid": "#5829ad", "accent": "#7a3df0", "ink": "#ffffff"},
}


def _ramp(c: dict) -> str:
    return (f"linear-gradient(180deg,{c['edge']} 0%,{c['mid']} 22%,"
            f"{c['accent']} 50%,{c['mid']} 78%,{c['edge']} 100%)")


def cat_css() -> str:
    """The data-cat token sets, one rule per category."""
    rules = []
    for cat, c in CAT_TOKENS.items():
        rules.append(
            f'html[data-cat="{cat}"] {{ --cat-ramp: {_ramp(c)}; '
            f"--cat-accent: {c['accent']}; --cat-ink: {c['ink']}; }}")
    return "\n  ".join(rules)


# --------------------------------------------------------------------------
# The document template. Placeholders are __DOUBLE_UNDERSCORE__ tokens
# (str.replace, no str.format - the CSS/JS braces stay untouched).
# --------------------------------------------------------------------------
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
<meta property="og:url" content="https://triple-a-tropics.com__PAGE_PATH__">
<link rel="icon" type="image/svg+xml" href="/logo.svg">
<style>
  :root {
    --bg: #0b0e13; --panel: #11161f; --border: #232a36;
    --fg: #e8eef5; --muted: #8ea2bd; --navy-deep: #0a1a2e;
    --cat-ramp: linear-gradient(180deg,#16324a,#3f7cab,#16324a);
    --cat-accent: #3f7cab; --cat-ink: #ffffff;
    --motion-slow: 4.5s; --motion-med: 1.2s; --motion-fast: 0.6s;
  }
  __CAT_CSS__

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
    font-family: "Metropolis", "Helvetica Neue", Arial, sans-serif; }
  a { color: #5dd3ff; text-decoration: none; }

  /* ---- shell: sidebar + stage (one DOM, CSS-rearranged on phones) ---- */
  .lab { display: flex; min-height: 100vh; }
  .side { width: 264px; flex: 0 0 264px; display: flex; flex-direction: column;
    background: linear-gradient(180deg, #0a1a2e 0%, #0d1420 30%, #0b0e13 100%);
    border-right: 1px solid var(--border); }
  .stage { flex: 1 1 auto; min-width: 0; padding: 26px 30px 60px; }

  /* identity banner - wears the storm (NO pulsing dot by design) */
  .banner { background: var(--cat-ramp); color: var(--cat-ink);
    padding: 18px 18px 16px; position: relative; overflow: hidden;
    border-bottom: 1px solid rgba(255,255,255,0.85);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.14),
                inset 0 -1px 0 rgba(0,0,0,0.40); }
  .banner .wordmark { font-size: 11px; font-weight: 800; letter-spacing: 2px;
    text-transform: uppercase; opacity: 0.85; }
  .banner .storm-name { font-size: 26px; font-weight: 900; line-height: 1.1;
    letter-spacing: 0.5px; text-transform: uppercase; margin: 6px 0 8px;
    text-shadow: 0 1px 0 rgba(0,0,0,0.35), 0 2px 4px rgba(0,0,0,0.35); }
  .chip { display: inline-block; padding: 4px 12px; border-radius: 999px;
    background: rgba(0,0,0,0.30); color: #ffffff; font-size: 12.5px;
    font-weight: 800; letter-spacing: 1px; text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.35); }
  /* category-change shine sweep (one pass, state-change triggered) */
  .banner::after { content: ""; position: absolute; top: 0; bottom: 0;
    width: 55%; left: -60%; transform: skewX(-18deg);
    background: linear-gradient(90deg, transparent,
      rgba(255,255,255,0.32), transparent); pointer-events: none;
    opacity: 0; }
  .banner.shine::after { animation: lab-shine var(--motion-med) ease-out 1; }
  @keyframes lab-shine { 0% { opacity: 1; left: -60%; }
                         100% { opacity: 1; left: 110%; } }
  /* gradient crossfade: a stacked overlay fades out the OLD ramp */
  .banner .old-ramp { position: absolute; inset: 0; background: var(--old-ramp, none);
    opacity: 0; pointer-events: none; }
  .banner.xfade .old-ramp { opacity: 1;
    animation: lab-xfade calc(var(--motion-slow) * 0.5) ease-in-out 1 forwards; }
  @keyframes lab-xfade { from { opacity: 1; } to { opacity: 0; } }
  .banner > .b-inner { position: relative; z-index: 1; }

  /* section nav with the accent rail */
  .nav-secs { display: flex; flex-direction: column; padding: 14px 0;
    flex: 1 1 auto; }
  .sec-btn { display: flex; align-items: center; gap: 10px;
    padding: 14px 18px; min-height: 48px; background: transparent;
    color: var(--muted); border: 0; border-left: 3px solid transparent;
    font: inherit; font-size: 14px; font-weight: 700;
    letter-spacing: 1.1px; text-transform: uppercase; cursor: pointer;
    text-align: left; }
  .sec-btn:hover { color: var(--fg); }
  .sec-btn.active { color: #ffffff; border-left-color: var(--cat-accent);
    background: linear-gradient(90deg, rgba(255,255,255,0.05), transparent); }
  .back-map { padding: 16px 18px; border-top: 1px solid var(--border);
    font-size: 13px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; min-height: 48px; display: flex;
    align-items: center; }

  /* ENDED banner strip (frozen state) */
  .ended-strip { display: none; background: #2a2f3a; color: #e8eef5;
    padding: 10px 16px; font-size: 13px; font-weight: 700;
    letter-spacing: 0.6px; text-align: center;
    border-bottom: 1px solid var(--border); }
  html[data-ended] .ended-strip { display: block; }

  /* ---- sections ---- */
  .sec { display: none; }
  .sec.active { display: block; }
  .sec-title { font-size: 18px; font-weight: 800; letter-spacing: 1.2px;
    text-transform: uppercase; color: #ffffff; margin: 0 0 16px; }
  /* section wipe-in (state-change: on switch) */
  .sec.active .wipe { animation: lab-wipe var(--motion-fast) ease-out 1; }
  @keyframes lab-wipe { from { opacity: 0; transform: translateX(14px); }
                        to   { opacity: 1; transform: translateX(0); } }

  /* stats banner (Overview) */
  .stats { display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    margin-bottom: 22px; }
  .stat { background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 12px 14px; position: relative;
    overflow: hidden; }
  .stat::before { content: ""; position: absolute; left: 0; right: 0;
    top: 0; height: 3px; background: var(--cat-ramp); }
  .stat .lbl { font-size: 10.5px; font-weight: 800; letter-spacing: 1.2px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
  .stat .val { font-size: 22px; font-weight: 800; color: #ffffff;
    font-variant-numeric: tabular-nums; display: flex; align-items: baseline;
    gap: 5px; white-space: nowrap; }
  .stat .unit { font-size: 12px; color: var(--muted); font-weight: 700; }
  /* text-y stats (position, last fix) run long - smaller type so they
     never clip their card */
  .stat.small .val { font-size: 15px; line-height: 1.5; }

  /* odometer: fixed-height window, digit columns roll via translateY */
  .odo { display: inline-flex; overflow: hidden; height: 1.15em; }
  .odo .digit { display: inline-block; width: 0.62em; text-align: center; }
  .odo .col { display: flex; flex-direction: column; height: 1.15em; }
  .odo .col .digit { flex: 0 0 1.15em; height: 1.15em; line-height: 1.15em; }
  .odo .col { transition: transform calc(var(--motion-slow) * 0.35)
    cubic-bezier(0.22, 1, 0.36, 1); }

  /* track map + chart cards */
  .card { background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px; margin-bottom: 20px; }
  .card h3 { margin: 0 0 10px; font-size: 12px; font-weight: 800;
    letter-spacing: 1.4px; text-transform: uppercase; color: var(--muted); }
  .card svg { width: 100%; height: auto; display: block;
    touch-action: pan-y; }
  /* chart draw-in (state-change: first open) */
  .draw path.series { stroke-dasharray: var(--len, 2000);
    stroke-dashoffset: var(--len, 2000);
    animation: lab-draw calc(var(--motion-slow) * 0.55) ease-out 1 forwards; }
  @keyframes lab-draw { to { stroke-dashoffset: 0; } }
  .draw .fill { opacity: 0;
    animation: lab-fill var(--motion-med) ease-out 1 forwards;
    animation-delay: calc(var(--motion-slow) * 0.45); }
  @keyframes lab-fill { to { opacity: 1; } }

  .stub { color: var(--muted); font-size: 14px; padding: 30px 0;
    text-align: center; }

  /* launch wipe (plays once on load) */
  .launch { position: fixed; inset: 0; z-index: 50; background: var(--cat-ramp);
    transform-origin: top; pointer-events: none;
    animation: lab-launch var(--motion-med) ease-in-out 1 forwards;
    animation-delay: 0.25s; }
  @keyframes lab-launch { to { transform: scaleY(0); } }

  /* mobile rotation: banner -> top bar, nav -> bottom tabs (same DOM) */
  @media (max-width: 640px) {
    .lab { flex-direction: column; }
    .side { width: 100%; flex: 0 0 auto; flex-direction: row;
      flex-wrap: wrap; border-right: 0; position: sticky; top: 0; z-index: 20;
      border-bottom: 1px solid var(--border); }
    .banner { flex: 1 1 100%; padding: 10px 14px; display: flex;
      align-items: center; gap: 10px; }
    .banner .wordmark { display: none; }
    .banner .storm-name { font-size: 17px; margin: 0; flex: 1 1 auto; }
    .nav-secs { order: 3; flex: 1 1 100%; flex-direction: row;
      position: fixed; bottom: 0; left: 0; right: 0; z-index: 30;
      background: var(--navy-deep); border-top: 1px solid var(--border);
      padding: 0; }
    .sec-btn { flex: 1 1 25%; justify-content: center; padding: 12px 4px;
      min-height: 52px; font-size: 11px; border-left: 0;
      border-top: 3px solid transparent; }
    .sec-btn.active { border-left: 0; border-top-color: var(--cat-accent); }
    .back-map { border-top: 0; padding: 10px 12px; min-height: 0; }
    .stage { padding: 16px 14px 86px; }
  }

  /* reduced motion: every animation lands on its final frame */
  @media (prefers-reduced-motion: reduce) {
    .launch, .banner.shine::after, .banner.xfade .old-ramp,
    .sec.active .wipe, .draw path.series, .draw .fill {
      animation-duration: 0.001s !important;
      animation-delay: 0s !important; }
    .odo .col { transition-duration: 0.001s !important; }
  }
</style>
</head>
<body>
<div class="launch" id="launch"></div>
<div class="ended-strip">THIS STORM HAS ENDED · final data below · CycloLab archive view</div>
<div class="lab">
  <aside class="side">
    <div class="banner" id="banner">
      <div class="old-ramp"></div>
      <div class="b-inner">
        <div class="wordmark">Triple-A-Tropics · CycloLab</div>
        <div class="storm-name" id="storm-name">__NAME__</div>
        <span class="chip" id="chip">__CHIP__</span>
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
        <h2 class="sec-title">Overview</h2>
        <div class="stats" id="stats"></div>
        <div class="card"><h3>Track</h3>
          <svg id="trackmap" viewBox="0 0 1000 560"
               preserveAspectRatio="xMidYMid meet"></svg></div>
        <div class="card"><h3>Wind &amp; pressure</h3>
          <svg id="chart" viewBox="0 0 1000 360"
               preserveAspectRatio="xMidYMid meet"></svg></div>
      </div>
    </section>
    <section class="sec" id="sec-satellite"><div class="wipe">
      <h2 class="sec-title">Satellite</h2>
      <div class="stub">Floater imagery lands in Stage 3.</div>
    </div></section>
    <section class="sec" id="sec-models"><div class="wipe">
      <h2 class="sec-title">Models</h2>
      <div class="stub">The storm-scoped HAFS viewer lands in Stage 3.</div>
    </div></section>
    <section class="sec" id="sec-advisories"><div class="wipe">
      <h2 class="sec-title">Advisories</h2>
      <div class="stub">Advisory text + the forecast cone land in Stage 4.</div>
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
  var POLL_MS = 60000;
  var SSHS = { TD:"#3fa4ff", TS:"#46c56a", C1:"#ffe14d", C2:"#ff9a2f",
               C3:"#ff4d3b", C4:"#e33ad4", C5:"#b03bff" };
  var CHIP_LABEL = { TD:"Tropical Depression", TS:"Tropical Storm",
    C1:"Category 1", C2:"Category 2", C3:"Category 3", C4:"Category 4",
    C5:"Category 5" };
  var reduced = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- section nav (lazy init on first open) ----------------------------
  var inited = {};
  function openSec(name) {
    if (!document.getElementById("sec-" + name)) return;  // unknown: no-op
    document.querySelectorAll(".sec").forEach(function (s) {
      s.classList.toggle("active", s.id === "sec-" + name);
    });
    document.querySelectorAll(".sec-btn").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-sec") === name);
    });
    if (!inited[name]) { inited[name] = true; /* lazy hooks: stages 3-4 */ }
    // restart the wipe (state-change animation)
    var w = document.querySelector("#sec-" + name + " .wipe");
    if (w) { w.style.animation = "none"; void w.offsetWidth; w.style.animation = ""; }
  }
  document.getElementById("secnav").addEventListener("click", function (e) {
    var b = e.target.closest(".sec-btn");
    if (b) openSec(b.getAttribute("data-sec"));
  });

  // ---- odometer (transform-only digit rolls) ----------------------------
  function odoSet(el, text) {
    // builds/updates digit columns; non-digits render as static chars
    var want = String(text);
    if (el.getAttribute("data-odo") === want) return;
    el.setAttribute("data-odo", want);
    while (el.children.length > want.length) el.removeChild(el.lastChild);
    for (var i = 0; i < want.length; i++) {
      var ch = want[i];
      var cell = el.children[i];
      if (/[0-9]/.test(ch)) {
        if (!cell || !cell.classList.contains("col")) {
          var col = document.createElement("span");
          col.className = "col";
          for (var d = 0; d <= 9; d++) {
            var s = document.createElement("span");
            s.className = "digit"; s.textContent = String(d);
            col.appendChild(s);
          }
          if (cell) el.replaceChild(col, cell); else el.appendChild(col);
          cell = col;
        }
        cell.style.transform = "translateY(" + (-1.15 * Number(ch)) + "em)";
      } else {
        if (!cell || cell.classList.contains("col")) {
          var st = document.createElement("span");
          st.className = "digit";
          if (cell) el.replaceChild(st, cell); else el.appendChild(st);
          cell = st;
        }
        cell.textContent = ch;
      }
    }
  }

  // ---- stats banner ------------------------------------------------------
  var STATS = [
    { id: "vmax", lbl: "Max wind", unit: "kt" },
    { id: "mslp", lbl: "Min pressure", unit: "mb" },
    { id: "ace", lbl: "Storm ACE", unit: "" },
    { id: "pos", lbl: "Position", unit: "", small: true },
    { id: "fix", lbl: "Last fix", unit: "UTC", small: true },
  ];
  function buildStats() {
    var host = document.getElementById("stats");
    host.innerHTML = STATS.map(function (s) {
      return '<div class="stat' + (s.small ? ' small' : '') + '">' +
        '<div class="lbl">' + s.lbl + '</div>' +
        '<div class="val"><span class="odo" id="odo-' + s.id + '"></span>' +
        (s.unit ? '<span class="unit">' + s.unit + '</span>' : "") +
        '</div></div>';
    }).join("");
  }

  function fmtPos(lat, lon) {
    if (lat == null || lon == null) return "—";
    return Math.abs(lat).toFixed(1) + (lat >= 0 ? "N" : "S") + " " +
           Math.abs(lon).toFixed(1) + (lon >= 0 ? "E" : "W");
  }

  // ---- category state (the app wears the storm) --------------------------
  var curCat = document.documentElement.getAttribute("data-cat");
  function setCategory(cat) {
    if (!cat || cat === curCat) return;
    var banner = document.getElementById("banner");
    var oldRamp = getComputedStyle(document.documentElement)
      .getPropertyValue("--cat-ramp");
    document.documentElement.setAttribute("data-cat", cat);
    document.getElementById("chip").textContent = CHIP_LABEL[cat] || cat;
    curCat = cat;
    if (reduced) return;
    banner.querySelector(".old-ramp").style.setProperty("--old-ramp", oldRamp);
    banner.classList.remove("xfade", "shine"); void banner.offsetWidth;
    banner.classList.add("xfade", "shine");
  }

  // ---- track map (client-rendered; equirect like the basin pages) -------
  function renderTrack(storm) {
    var svg = document.getElementById("trackmap");
    var pts = storm.points || [];
    if (!pts.length) { svg.innerHTML = ""; return; }
    var lats = pts.map(function (p) { return p.lat; });
    var lons = pts.map(function (p) { return p.lon; });
    var pad = 2.5;
    var la0 = Math.min.apply(null, lats) - pad, la1 = Math.max.apply(null, lats) + pad;
    var lo0 = Math.min.apply(null, lons) - pad, lo1 = Math.max.apply(null, lons) + pad;
    var W = 1000, H = 560;
    function X(lon) { return (lon - lo0) / (lo1 - lo0) * W; }
    function Y(lat) { return H - (lat - la0) / (la1 - la0) * H; }
    var d = pts.map(function (p, i) {
      return (i ? "L" : "M") + X(p.lon).toFixed(1) + "," + Y(p.lat).toFixed(1);
    }).join(" ");
    var parts = ['<rect width="' + W + '" height="' + H + '" fill="#0a1019"/>'];
    parts.push('<path d="' + d + '" fill="none" stroke="var(--cat-accent)" ' +
               'stroke-width="2.5" stroke-opacity="0.85" ' +
               'stroke-linejoin="round" stroke-linecap="round"/>');
    pts.forEach(function (p) {
      var c = SSHS[p.cls] || SSHS.TD;
      parts.push('<circle cx="' + X(p.lon).toFixed(1) + '" cy="' +
        Y(p.lat).toFixed(1) + '" r="5" fill="' + c +
        '" stroke="#fff" stroke-width="1"/>');
    });
    var last = pts[pts.length - 1];
    parts.push('<circle cx="' + X(last.lon).toFixed(1) + '" cy="' +
      Y(last.lat).toFixed(1) + '" r="11" fill="none" ' +
      'stroke="var(--cat-accent)" stroke-width="3"/>');
    svg.innerHTML = parts.join("");
  }

  // ---- wind/pressure timeline (hand-rolled, draw-in on first render) ----
  var chartDrawn = false;
  function renderChart(storm) {
    var svg = document.getElementById("chart");
    var pts = (storm.points || []).filter(function (p) {
      return p.wind_kt != null; });
    if (pts.length < 2) { svg.innerHTML = ""; return; }
    var W = 1000, H = 360, padL = 56, padR = 56, padT = 18, padB = 30;
    var wMax = Math.max(140, Math.max.apply(null, pts.map(function (p) {
      return p.wind_kt; })) + 10);
    var prs = pts.map(function (p) { return p.pressure_mb; })
      .filter(function (v) { return v != null; });
    var p0 = Math.min.apply(null, prs.concat([1000])) - 6;
    var p1 = Math.max.apply(null, prs.concat([1014])) + 6;
    function Xi(i) { return padL + i / (pts.length - 1) * (W - padL - padR); }
    function Yw(w) { return H - padB - (w / wMax) * (H - padT - padB); }
    function Yp(p) { return H - padB - ((p - p0) / (p1 - p0)) * (H - padT - padB); }
    var bands = [[34, "TS"], [64, "C1"], [83, "C2"], [96, "C3"],
                 [113, "C4"], [137, "C5"]];
    var parts = ['<rect width="' + W + '" height="' + H + '" fill="#0a1019"/>'];
    bands.forEach(function (b) {
      parts.push('<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' +
        Yw(b[0]).toFixed(1) + '" y2="' + Yw(b[0]).toFixed(1) +
        '" stroke="' + SSHS[b[1]] + '" stroke-opacity="0.25" ' +
        'stroke-dasharray="3 5"/>');
      parts.push('<text x="' + (W - padR + 6) + '" y="' +
        (Yw(b[0]) + 4).toFixed(1) + '" fill="' + SSHS[b[1]] +
        '" font-size="11" opacity="0.8">' + b[1] + '</text>');
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
    parts.push('<text x="' + padL + '" y="14" fill="#8ea2bd" font-size="11">' +
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

  // ---- hydration (poll + diff-merge: grow state, never reset the user) --
  var lastFixKey = null;
  function apply(storm) {
    var cat = storm.current_category || "TD";
    setCategory(cat);
    var name = (storm.name || SID).toUpperCase();
    document.getElementById("storm-name").textContent = name;
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
    odoSet(document.getElementById("odo-fix"),
           fixKey ? fixKey.slice(5, 16).replace("T", " ") : "—");
    if (fixKey !== lastFixKey) {       // NEW FIX -> redraw map; chart grows
      renderTrack(storm);
      renderChart(storm);
      lastFixKey = fixKey;
    }
  }

  function poll() {
    fetch(FEED_URL + (FEED_URL.indexOf("?") >= 0 ? "&" : "?") + "t=" +
          Date.now(), { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (feed) {
        if (!feed || !feed.storms) return;
        var storm = feed.storms.filter(function (s) {
          return s && s.sid === SID; })[0];
        if (storm) apply(storm);
      })
      .catch(function () { /* baked snapshot stands */ })
      .then(function () { if (!ENDED) setTimeout(poll, POLL_MS); });
  }

  buildStats();
  var BAKED = __BAKED__;
  if (BAKED) apply(BAKED);
  if (!ENDED) poll();

  // launch wipe element removes itself after playing
  var l = document.getElementById("launch");
  l.addEventListener("animationend", function () { l.remove(); });
  if (reduced) l.remove();

  // exposed for the node harness + recording rig (deterministic drives)
  window.__lab = { openSec: openSec, setCategory: setCategory,
                   apply: apply, odoSet: odoSet };
})();
</script>
</body>
</html>
"""


def _esc(s) -> str:
    return _html.escape(str(s if s is not None else ""), quote=True)


def render_page(storm: dict, *, feed_url: str, adv_url: str | None = None,
                ended: bool = False) -> str:
    """Render one storm's shell. ``storm`` is the tracks-feed storm dict
    (the baked snapshot); ``feed_url`` the basin feed the page hydrates
    from. ``ended=True`` bakes the frozen archive variant (no polling)."""
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

    # The baked snapshot keeps the page meaningful before/without JS and
    # is the frozen content of an ENDED page.
    baked = json.dumps(storm, separators=(",", ":"))

    html = (HTML_TEMPLATE
            .replace("__CAT_CSS__", cat_css())
            .replace("__CAT__", cat)
            .replace("__NAME__", _esc(name))
            .replace("__CHIP__", _esc(chip))
            .replace("__OG_TITLE__", _esc(og_title))
            .replace("__OG_DESC__", _esc(og_desc))
            .replace("__PAGE_PATH__", _esc(page_url_path(storm["sid"])))
            .replace("__SID__", _esc(storm["sid"]))
            .replace("__FEED_URL__", _esc(feed_url))
            .replace("__ADV_URL__", _esc(adv_url or adv_key(storm["sid"])))
            .replace("__ENDED__", "true" if ended else "false")
            .replace("__BAKED__", baked))
    if ended:
        html = html.replace("<html lang=\"en\" data-cat=",
                            "<html lang=\"en\" data-ended data-cat=")
    return html
