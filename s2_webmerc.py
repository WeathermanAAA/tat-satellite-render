#!/usr/bin/env python3
"""Stage-2 WEB-MERCATOR reprojection (SATELLITE-REARCH §5.5) -- the Phase-2b tile
scheme. Turns the Phase-2a chrome-free EQUIRECTANGULAR (PlateCarree) imagery
raster into a GLOBAL Web-Mercator (EPSG:3857) XYZ WebP pyramid that a MapLibre GL
``RasterTileSource`` consumes directly (its native scheme, rooted at the world z0
tile). This is the second *sanctioned renderer-scope exception* (§2/§5.5) beyond
chrome: a reproject placed AFTER the chrome-free render and BEFORE the tile-cut.

Behind the manifest ``scheme`` field: Phase-2a is ``flat-native-xyz`` (pixels
unchanged), this is ``webmercator-xyz`` (resampled to the slippy-map projection).
Web-Mercator clips beyond ±85.051° -- a known, gated framing delta (§5.5).

Pure numpy + scipy + PIL (no MapLibre, no boto3): each output XYZ tile's pixel
grid is mapped back to lon/lat (inverse Web-Mercator) and BILINEAR-sampled from
the source raster (whose pixels map linearly to lon/lat within its bounds). Tiles
that fall entirely outside the data footprint are skipped (transparent), so the
same ``emit_pyramid``/manifest/prune plumbing as the flat scheme applies -- only
the cut function differs.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from s2_pyramid import PyramidSpec, _as_rgba, _encode_webp

# Web-Mercator's usable latitude limit (the projection diverges at the poles).
WEBMERC_LAT_LIMIT = 85.05112877980659
_R2D = 180.0 / math.pi


def merc_y_to_lat(y_norm: float) -> float:
    """Inverse Web-Mercator: normalized world-Y in [0,1] (0=north) -> latitude."""
    return _R2D * math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_norm)))


def lat_to_merc_y(lat: float) -> float:
    """Forward Web-Mercator: latitude -> normalized world-Y in [0,1] (0=north).
    This is the exact inverse of merc_y_to_lat -- y = 0.5 - asinh(tan(lat))/2pi.
    (NOT the Gudermannian 0.5 - atan(sinh(lat))/2pi, which is a DIFFERENT curve
    and mis-registers vectors against the imagery, error growing with latitude.)"""
    lat = max(-WEBMERC_LAT_LIMIT, min(WEBMERC_LAT_LIMIT, lat))
    return 0.5 - math.asinh(math.tan(math.radians(lat))) / (2.0 * math.pi)


def tile_geo_bounds(z: int, x: int, y: int) -> tuple:
    """Geographic (W,S,E,N) of an XYZ tile in the global Web-Mercator scheme."""
    n = float(2 ** z)
    return (x / n * 360.0 - 180.0, merc_y_to_lat((y + 1) / n),
            (x + 1) / n * 360.0 - 180.0, merc_y_to_lat(y / n))


def _lonlat_axes(z: int, x: int, y: int, T: int):
    """Per-column lon and per-row lat of a tile's pixel centers (row 0 = north)."""
    n = float(2 ** z)
    cols = (x + (np.arange(T) + 0.5) / T) / n
    lon = cols * 360.0 - 180.0
    rows = (y + (np.arange(T) + 0.5) / T) / n
    lat = _R2D * np.arctan(np.sinh(np.pi * (1.0 - 2.0 * rows)))
    return lon, lat


def native_max_zoom(bounds, image_px, tile_size: int = 512) -> int:
    """Smallest global-XYZ zoom whose tile density meets the source's finest
    resolution, so z=maxzoom is >= native (full-resolution zoom). Derived from
    the un-stretched lon axis (Mercator only stretches lat), rounded up."""
    W, S, E, N = bounds
    sw, _sh = image_px
    span = max(abs(E - W), 1e-6)
    px_per_deg = sw / span
    z = math.log2(max(px_per_deg * 360.0 / tile_size, 1.0))
    return max(0, int(math.ceil(z)))


def reproject_tile(rgba: np.ndarray, bounds, z: int, x: int, y: int,
                   tile_size: int = 512):
    """Reproject the equirectangular source into ONE Web-Mercator XYZ tile.
    Returns a (T,T,4) uint8 RGBA, or None if the tile is entirely off-data.

    ANTIMERIDIAN-crossing sources (Himawari full disk) carry an UNWRAPPED east
    edge (E > 180); output tile longitudes west of W unwrap by +360 so they
    sample the eastern lobe. E <= 180 sources (all GOES) are bit-identical to
    the pre-wrap behavior (the unwrap condition can never hold)."""
    W, S, E, N = bounds
    sh, sw = rgba.shape[:2]
    lon, lat = _lonlat_axes(z, x, y, tile_size)
    if E > 180.0:
        lon = np.where((lon < W) & (lon + 360.0 <= E + 1e-9), lon + 360.0, lon)
    # Map lon->source column, lat->source row (row 0 = north = N). map_coordinates
    # samples at (row, col); shift by 0.5 so pixel centers align.
    src_col = (lon - W) / (E - W) * sw - 0.5
    src_row = (N - lat) / (N - S) * sh - 0.5
    CC, RR = np.meshgrid(src_col, src_row)
    inb = (CC >= -0.5) & (CC <= sw - 0.5) & (RR >= -0.5) & (RR <= sh - 0.5)
    if not inb.any():
        return None
    coords = np.stack([RR.ravel(), CC.ravel()])
    out = np.empty((tile_size, tile_size, 4), np.uint8)
    for c in range(4):
        samp = map_coordinates(rgba[..., c], coords, order=1, mode="constant", cval=0)
        out[..., c] = np.clip(samp, 0, 255).reshape(tile_size, tile_size)
    out[~inb, 3] = 0                      # hard-zero alpha outside the source extent
    if int(out[..., 3].max()) == 0:
        return None
    return out


def cut_webmerc_pyramid(raster: np.ndarray, bounds, spec: PyramidSpec = PyramidSpec(),
                        *, maxzoom=None) -> dict:
    """Web-Mercator analogue of s2_pyramid.cut_pyramid: reproject the source into
    a global XYZ pyramid z=min_zoom..maxzoom (auto native if maxzoom is None) and
    encode each non-empty tile to WebP. Returns the same dict shape as
    cut_pyramid: {maxzoom, image_px, tiles:{(z,x,y):bytes}, tile_counts}."""
    rgba = _as_rgba(raster)
    H, W = rgba.shape[:2]
    T = spec.tile_size
    # clamp bounds to Web-Mercator's usable latitude band (§5.5).
    w, s, e, n = bounds
    s = max(s, -WEBMERC_LAT_LIMIT); n = min(n, WEBMERC_LAT_LIMIT)
    cb = (w, s, e, n)
    if maxzoom is None:
        maxzoom = native_max_zoom(cb, (W, H), T)

    tiles: dict = {}
    counts: dict = {}
    for z in range(spec.min_zoom, maxzoom + 1):
        nz = 2 ** z
        # Only iterate the tile range that overlaps the data footprint. An
        # antimeridian-crossing source (unwrapped e > 180) overlaps TWO x
        # ranges of the global grid: [x(w)..nz-1] and [0..x(e-360)].
        if e > 180.0:
            x_ranges = [(max(0, int((w + 180.0) / 360.0 * nz)), nz - 1),
                        (0, min(nz - 1, int((e - 360.0 + 180.0) / 360.0 * nz)))]
        else:
            x_ranges = [(max(0, int((w + 180.0) / 360.0 * nz)),
                         min(nz - 1, int((e + 180.0) / 360.0 * nz)))]
        # y grows southward; north edge = smaller y (correct forward Mercator).
        y0 = max(0, int(lat_to_merc_y(min(n, WEBMERC_LAT_LIMIT)) * nz))
        y1 = min(nz - 1, int(lat_to_merc_y(max(s, -WEBMERC_LAT_LIMIT)) * nz))
        # Materialize the coordinate list FIRST (the dedup set is not
        # thread-safe), then cut. Each tile is an independent, deterministic
        # function of the source raster, so the work parallelizes exactly --
        # and BOTH halves release the GIL (scipy map_coordinates inside
        # reproject_tile; Pillow's WebP encoder), which is why threads and not
        # processes: no per-worker copy of a multi-hundred-MB raster.
        # S2_CUT_WORKERS=1 restores the serial loop byte-for-byte.
        seen = set()
        coords = []
        for x0, x1 in x_ranges:
            for ty in range(y0, y1 + 1):
                for tx in range(x0, x1 + 1):
                    if (tx, ty) in seen:
                        continue
                    seen.add((tx, ty))
                    coords.append((tx, ty))

        def _cut_one(txy, _z=z):
            tx, ty = txy
            tile = reproject_tile(rgba, cb, _z, tx, ty, T)
            if tile is None:
                return None
            data = _encode_webp(tile, spec)
            if data is None:
                return None
            return (_z, tx, ty), data

        workers = max(1, int(os.environ.get("S2_CUT_WORKERS", "3")))
        if workers > 1 and len(coords) > 4:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                cuts = list(ex.map(_cut_one, coords))
        else:
            cuts = [_cut_one(t) for t in coords]
        c = 0
        for item in cuts:
            if item is not None:
                tiles[item[0]] = item[1]
                c += 1
        counts[z] = c
    return {"maxzoom": maxzoom, "image_px": [W, H], "tiles": tiles,
            "tile_counts": counts, "scheme": "webmercator-xyz"}
