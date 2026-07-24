#!/usr/bin/env python3
"""Generate the explorer's PRESENTATION CONFIG from the registry: products.js
(the picker table) + per-palette colorbar PNGs, for committing into the main
repo's /satellite/explorer/. One authoring point: a new recipe row in
s2_recipes/s2_registry flows into the viewer by re-running this and committing
the output (the vendored-asset pattern -- the page stays self-contained and
works before the box has emitted anything).

  python s2_products_export.py --out /path/to/Triple-A-Tropics/satellite/explorer
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

import s2_recipes as X
import s2_registry as R
from colormaps import get_enhancement, enhancement_norm

# Colorbar presentation per FROZEN enhancement: (numeric tick values, caption).
# Tick POSITIONS are computed from the palette norm -- (vmax - v)/(vmax - vmin)
# from the top -- so a label sits exactly at its color (the old evenly-spread
# ticks were up to 15 degC off on the -95..40 rainbow_ir span).
CBARS = {
    "rainbow_ir": ([40, 0, -40, -80], "Brightness temperature (°C)"),
    "dvorak":     ([40, 0, -40, -80], "Brightness temperature (°C)"),
    "wv_tat":     ([0, -30, -60, -90], "Brightness temperature (°C)"),
    "grayscale":  ([30, -10, -50, -90], "Brightness temperature (°C)"),
}
REFL_TICKS = ([100, 50, 0], "Reflectance (%)")


def _fmt(v):
    return str(v).replace("-", "−")


def tick_marks(values, vmin, vmax):
    """[{t: label, p: fraction-from-top}] with exact norm placement."""
    return [{"t": _fmt(v), "p": round((vmax - v) / (vmax - vmin), 4)}
            for v in values]

BAR_W, BAR_H = 14, 230


def render_cbar_png(enhancement: str, path: str):
    """Vertical bar from the FROZEN cmap+norm (vmax at top), matching the
    explorer's existing rainbow_ir_cbar.png geometry."""
    cmap = get_enhancement(enhancement)["cmap"]
    norm = enhancement_norm(enhancement)
    vals = np.linspace(norm.vmax, norm.vmin, BAR_H)
    rgba = (np.asarray(cmap(norm(vals))) * 255 + 0.5).astype(np.uint8)  # H x 4
    img = np.repeat(rgba[:, None, :], BAR_W, axis=1)
    Image.fromarray(img, "RGBA").save(path)


def render_swatch_png(enhancement: str, path: str, w: int = 64, h: int = 10):
    """Horizontal palette-preview swatch from the FROZEN cmap+norm (vmin at
    left, vmax at right) — the rail's real-LUT ramp, never hand-drawn."""
    cmap = get_enhancement(enhancement)["cmap"]
    norm = enhancement_norm(enhancement)
    vals = np.linspace(norm.vmin, norm.vmax, w)
    rgba = (np.asarray(cmap(norm(vals))) * 255 + 0.5).astype(np.uint8)  # W x 4
    img = np.repeat(rgba[None, :, :], h, axis=0)
    Image.fromarray(img, "RGBA").save(path)


def render_refl_swatch_png(path: str, w: int = 64, h: int = 10):
    """Reflectance gray swatch: display gray = sqrt(reflectance), 0% left."""
    r = np.linspace(0.0, 1.0, w)
    g = (np.sqrt(r) * 255 + 0.5).astype(np.uint8)
    row = np.dstack([g, g, g, np.full_like(g, 255)])[0]
    img = np.repeat(row[None, :, :], h, axis=0)
    Image.fromarray(img, "RGBA").save(path)


def render_refl_cbar_png(path: str):
    """Reflectance gray bar: display gray = sqrt(reflectance) (the c01-c06
    gamma-2 stretch), 100% at top."""
    r = np.linspace(1.0, 0.0, BAR_H)
    g = (np.sqrt(r) * 255 + 0.5).astype(np.uint8)
    col = np.dstack([g, g, g, np.full_like(g, 255)])[0]      # H x 4
    img = np.repeat(col[:, None, :], BAR_W, axis=1)
    Image.fromarray(img, "RGBA").save(path)


_GEO_EXPORT = {   # global composite rows (no Recipe object): title + palette
    "ir":   ("Clean IR window · multi-satellite", "rainbow_ir"),
    "irbd": ("IR Dvorak BD · multi-satellite", "dvorak"),
    "wv":   ("6.2 µm Water Vapor · multi-satellite", "wv_tat"),
}


def product_row(e) -> dict:
    r = None
    if e.recipe_id:
        try:
            r = X.recipe_for(e.family, e.recipe_id)
        except KeyError:
            r = None
    if r is None and e.sat_key == "geo":
        title, cbar_key = _GEO_EXPORT[e.band_key]
        group, day_only, bt = "channel", False, True
    elif r is None:   # the pre-suite clean-IR row
        title, group, day_only, bt = "C13 · 10.3 µm (Clean IR)", "channel", False, True
        cbar_key = "rainbow_ir"
    else:
        title, group, day_only, bt = r.title, r.group, r.day_only, bool(r.bt_band)
        if r.kind == "single_palette":
            cbar_key = r.enhancement
        elif r.kind == "sandwich":
            cbar_key = "rainbow_ir"      # sandwich colors ARE rainbow_ir hues
        elif r.kind == "rgb_guns" and all(g.kind == "refl" for g in r.guns) \
                and len({g.expr for g in r.guns}) == 1:
            cbar_key = "_refl"           # gray single reflective channel
        else:
            cbar_key = None              # multispectral RGB / truecolor: no scalar bar
    cbar = None
    if cbar_key == "_refl":
        cbar = {"img": "cbars/gray_refl.png",
                "ticks": tick_marks(REFL_TICKS[0], 0, 100), "cap": REFL_TICKS[1]}
    elif cbar_key:
        values, cap = CBARS[cbar_key]
        norm = enhancement_norm(cbar_key)
        cbar = {"img": f"cbars/{cbar_key}.png",
                "ticks": tick_marks(values, norm.vmin, norm.vmax), "cap": cap}
    return {"key": e.band_key, "id": e.product_id, "path": e.product_path,
            "title": title, "group": group, "bt": bt, "dayOnly": day_only,
            "cbar": cbar}


def _sector_rows(sat_key: str, sector: str) -> list:
    entries = [e for e in R.REGISTRY
               if e.tiled and e.sat_key == sat_key and e.sector_key == sector]
    # picker order: the clean-IR default first, then composites, RGBs, channels
    order = {"channel": 2, "rgb": 1, "composite": 0}
    rows = [product_row(e) for e in entries]
    rows.sort(key=lambda p: (p["key"] != "ir", order[p["group"]], p["title"]))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="main-repo satellite/explorer dir")
    ap.add_argument("--sector", default="conus")
    args = ap.parse_args(argv)

    rows = _sector_rows("goes19", args.sector)
    # Himawari-9 rides along as its own product set (wpac is the export
    # sector; the cockpit swaps wpac<->fd in the path exactly like conus<->fd).
    # TVProducts.products stays the goes19 list -- compare.html and any
    # existing reader see an unchanged shape (additive key only).
    hw_rows = _sector_rows("himawari9", "wpac")
    # GK-2A emits full-disk only, so fd IS the export sector (the cockpit's
    # sector substitution is a no-op) -- same additive-nested-key pattern.
    gk2a_rows = _sector_rows("gk2a", "fd")
    # MTG-I1 FCI (Meteosat-12): full-disk only, like GK-2A.
    mtgi1_rows = _sector_rows("mtgi1", "fd")
    geo_rows = _sector_rows("geo", "global")

    cdir = os.path.join(args.out, "cbars")
    os.makedirs(cdir, exist_ok=True)
    for name in CBARS:
        render_cbar_png(name, os.path.join(cdir, f"{name}.png"))
        render_swatch_png(name, os.path.join(cdir, f"swatch_{name}.png"))
    render_refl_cbar_png(os.path.join(cdir, "gray_refl.png"))
    render_refl_swatch_png(os.path.join(cdir, "swatch_gray_refl.png"))

    js = ("/* GENERATED by tat-satellite-render s2_products_export.py -- do not\n"
          " * hand-edit. Regenerate after registry/recipe changes:\n"
          " *   python s2_products_export.py --out <TAT>/satellite/explorer\n"
          " * Presentation mirror of the registry suite (SSOT: s2_recipes +\n"
          " * s2_registry); products.json on R2 carries the same rows. */\n"
          "window.TVProducts = " + json.dumps({
              "base": "https://cdn.triple-a-tropics.com/shadow/",
              "sector": args.sector,
              "products": rows,
              "himawari9": {"sector": "wpac", "products": hw_rows},
              "gk2a": {"sector": "fd", "products": gk2a_rows},
              "mtgi1": {"sector": "fd", "products": mtgi1_rows},
              "geo": {"sector": "global", "products": geo_rows},
          }, indent=2, ensure_ascii=False) + ";\n")
    with open(os.path.join(args.out, "products.js"), "w") as f:
        f.write(js)
    print(f"wrote products.js ({len(rows)} goes19 + {len(hw_rows)} himawari9 "
          f"+ {len(gk2a_rows)} gk2a + {len(mtgi1_rows)} mtgi1 + "
          f"{len(geo_rows)} geo-global products) + "
          f"{len(CBARS)+1} colorbars -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
