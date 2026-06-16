#!/usr/bin/env python3
"""
cyclolab_guidance_review.py  -- STAGE B, ART-DIRECTION REVIEW ONLY (HELD)
========================================================================
Builds a SELF-CONTAINED review page for the three CycloLab guidance renderers
(Model Forecast Tracks, Model Forecast Intensity, SHIPS Output Diagram) on a live
Stage-A entity, in the CycloLab house style. The track plot is rendered THREE ways
for the color-scale options board (A cyclonicwx kt-rainbow / B TAT SSHWS / C
WIND_TIER blue->gold) - the contested pick is Andrew's, NOT self-chosen.

NOT wired into cyclolab_shell.render_page / the live pages. Reuses, no forks:
  * basemap geometry  -> cyclolab_basemap.basemap_for
  * category palette   -> ace_core.SSHS_COLORS / sshs_class (single source)
  * intensity bands    -> the cone's SSHS_BANDS thresholds (mirrored as data)
  * projection/graticule -> the cone's fitProjection / graticule (house math)
  * house tokens/font  -> mirrored from cyclolab_shell CSS (Metropolis, navy, tnum)

Usage:  python cyclolab_guidance_review.py NHC_EP932026 93E  [outdir]
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from cyclolab_basemap import basemap_for
from ace_core import SSHS_COLORS

CDN = "https://cdn.triple-a-tropics.com"
UA = {"User-Agent": "cyclolab-guidance-review/1.0"}


def fetch_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.loads(r.read())


def storm_name(sid: str) -> str:
    try:
        feed = fetch_json(f"{CDN}/global_storms.geojson")
        for f in feed.get("features", []):
            p = f.get("properties", {})
            if p.get("kind") == "active_marker" and (p.get("storm_id") or p.get("sid")) == sid:
                return p.get("name") or sid
    except Exception:
        pass
    return sid


def center_of(guidance: dict):
    """Analysis (tau 0) position from a consensus/first track aid; fallback mean."""
    aids = guidance.get("aids", {})
    for pref in (guidance.get("consensus") or []) + (guidance.get("track_aids") or []):
        for p in aids.get(pref, []):
            if p["tau"] == 0 and p["lat"] is not None:
                return p["lat"], p["lon"]
    for a in aids.values():
        for p in a:
            if p["lat"] is not None:
                return p["lat"], p["lon"]
    return 15.0, -120.0


# --- house style + JS (kept in companion files so the page stays readable) ----
HERE = Path(__file__).resolve().parent
CSS = (HERE / "cyclolab_guidance_review.css").read_text()
JS = (HERE / "cyclolab_guidance_review.js").read_text()

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CycloLab guidance review - __NAME__</title>
<style>__CSS__</style></head>
<body>
<div class="wrap">
  <header class="pagehead">
    <div class="ph-eyebrow">CycloLab &middot; Model Guidance &middot; <span class="ph-review">STAGE B REVIEW (held)</span></div>
    <div class="ph-title"><span id="storm-name">__NAME__</span> <span class="ph-sub">__SIDLABEL__</span></div>
    <div class="ph-meta" id="ph-meta"></div>
  </header>

  <section class="panel" id="p-tracks">
    <div class="lockup"><div class="lk-eyebrow">Model Forecast Tracks</div>
      <div class="lk-sub">track aids &middot; colored by peak wind &middot; consensus highlighted</div></div>
    <div class="stage"><svg id="tracks" preserveAspectRatio="xMidYMid meet"></svg></div>
    <div class="legend" id="tracks-legend"></div>
  </section>

  <section class="panel board" id="p-board">
    <div class="lockup"><div class="lk-eyebrow">Track color scale &mdash; options board</div>
      <div class="lk-sub">pick one (do not deploy until Andrew signs off)</div></div>
    <div class="board-grid">
      <figure><figcaption>A &middot; cyclonicwx kt-rainbow</figcaption>
        <div class="stage sm"><svg id="boardA" preserveAspectRatio="xMidYMid meet"></svg></div>
        <svg class="demo" id="demoA" preserveAspectRatio="xMidYMid meet"></svg></figure>
      <figure><figcaption>B &middot; TAT SSHWS category</figcaption>
        <div class="stage sm"><svg id="boardB" preserveAspectRatio="xMidYMid meet"></svg></div>
        <svg class="demo" id="demoB" preserveAspectRatio="xMidYMid meet"></svg></figure>
      <figure><figcaption>C &middot; WIND_TIER blue&rarr;gold</figcaption>
        <div class="stage sm"><svg id="boardC" preserveAspectRatio="xMidYMid meet"></svg></div>
        <svg class="demo" id="demoC" preserveAspectRatio="xMidYMid meet"></svg></figure>
    </div>
    <div class="lk-sub" style="margin-top:7px">Lower strip: the same scale on <b>illustrative</b> tracks at fixed peak winds (30&ndash;150 kt) &mdash; this storm is a weak invest, so its real strands all read low.</div>
  </section>

  <section class="panel" id="p-intensity">
    <div class="lockup"><div class="lk-eyebrow">Model Forecast Intensity</div>
      <div class="lk-sub">Vmax vs forecast hour &middot; SSHWS bands &middot; hi-res aids emphasized</div></div>
    <div class="stage tall"><svg id="intensity" preserveAspectRatio="xMidYMid meet"></svg></div>
    <div class="legend" id="intensity-legend"></div>
  </section>

  <section class="panel" id="p-ships">
    <div class="lockup"><div class="lk-eyebrow">SHIPS Output Diagram</div>
      <div class="lk-sub">environment &middot; rapid-intensification &middot; annularity</div></div>
    <div id="ships-root"></div>
  </section>
</div>
<script>
window.__GUIDANCE__ = /*g*/GUIDANCE_JSON;
window.__SHIPS__ = /*s*/SHIPS_JSON;
window.__BASEMAP__ = /*b*/BASEMAP_JSON;
window.__SSHS__ = /*p*/SSHS_JSON;
window.__STORMNAME__ = "__NAME__";
window.__SIDLABEL__ = "__SIDLABEL__";
</script>
<script>__JS__</script>
</body></html>"""


def build_page(sid: str, name: str, guidance: dict, ships: dict, basemap: dict) -> str:
    return (PAGE
            .replace("__CSS__", CSS)
            .replace("__JS__", JS)
            .replace("GUIDANCE_JSON", json.dumps(guidance, separators=(",", ":")))
            .replace("SHIPS_JSON", json.dumps(ships, separators=(",", ":")))
            .replace("BASEMAP_JSON", json.dumps(basemap, separators=(",", ":")))
            .replace("SSHS_JSON", json.dumps(SSHS_COLORS, separators=(",", ":")))
            .replace("__SIDLABEL__", sid)
            .replace("__NAME__", name))


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "NHC_EP932026"
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else (HERE / "_review_out")
    outdir.mkdir(exist_ok=True)
    guidance = fetch_json(f"{CDN}/cyclolab/{sid}/guidance.json")
    try:
        ships = fetch_json(f"{CDN}/cyclolab/{sid}/ships.json")
    except Exception:
        ships = {"available": False, "reason": "unavailable", "sid": sid}
    name = sys.argv[2] if len(sys.argv) > 2 else storm_name(sid)
    lat, lon = center_of(guidance)
    basemap = basemap_for(lat, lon, guidance.get("basin", "EP"))
    html = build_page(sid, name, guidance, ships, basemap)
    out = outdir / f"review_{sid}.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html)/1000:.0f} KB)  center=({lat},{lon})  "
          f"track_aids={len(guidance.get('track_aids',[]))} ships={ships.get('available')}")


if __name__ == "__main__":
    main()
