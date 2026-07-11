#!/usr/bin/env python3
"""Stage-2 SHADOW PYRAMID EMITTER -- pure tiling + manifest core (Phase 2a).

Cuts ONE rendered imagery raster into a **512 px flat-native XYZ WebP tile
pyramid** and emits an operational-monitor-style ``latest_times.json`` manifest, to a SHADOW
R2 prefix, driven by an ``s2_registry.ProductEntry``. This is the *zoomable
product* path (SATELLITE-REARCH §4.1/§4.2/§6): floaters + meso stay single-frame
sequences (§4.2), only genuinely-zoomable wide products get a pyramid.

WHY FLAT-NATIVE (not Web-Mercator) FOR PHASE 2a
-----------------------------------------------
The brief is "the existing render CUT INTO TILES, pixels unchanged": at the max
zoom level the tiles ARE the source render's pixels (only overview levels are
Lanczos-downsampled, the standard pyramid build). The renderer emits an
equirectangular (PlateCarree) view, so pixel (px,py) maps LINEARLY to (lon,lat)
inside ``bounds`` -- fully geo-referenceable, which the Phase-2b viewer +
draw-box picker consume directly. Reprojection geos/PlateCarree -> EPSG:3857
(SATELLITE-REARCH §5.5) is the heavier, *sanctioned renderer exception* tied to
the chrome-free re-render (§6.3); it is a Phase-2b/later layer. The manifest
carries ``scheme`` + ``bounds`` so a ``webmercator-xyz`` variant slots in later
without breaking the contract or wasting this phase.

PURITY: this module imports only numpy + PIL + stdlib. NO boto3, NO matplotlib,
NO satellites -- the R2 write goes through an injected store (the
``s1_ingest.R2`` object on the box; ``FilesystemStore`` in the Codespace; a
hand ``FakeR2`` dict in tests), and the imagery raster is produced by
``s2_imagery`` (matplotlib + the FROZEN renderer's colortable) and passed in.
So the pyramid math is unit-testable with synthetic rasters and no network.

TILE SCHEME (flat-native XYZ, top-left origin, y increases SOUTHWARD)
--------------------------------------------------------------------
* tile size T = 512 px, WebP (RGBA, lossy, method=6). Off-data/space is a
  transparent alpha (the viewer shows through; DIFFERS on purpose from the
  floater renderer's opaque DARK_BG fill -- a map overlay wants transparency).
* maxzoom Z = ceil(log2(max(W,H)/T)); level z image = native scaled by
  1/2^(Z-z). z=Z is native (pixels unchanged); z=0 fits in <=1 tile.
* tile (z,x,y) = the T x T block at (x*T, y*T) of the level-z image, top-left
  aligned onto a transparent T x T canvas (partial edge tiles padded, viewer
  crops via image_px). Fully-transparent tiles are SKIPPED (§4.2 PUT-discipline;
  a missing tile renders transparent -- standard slippy behaviour, the manifest
  is still the SSOT because the grid is *derivable* from image_px+maxzoom).
* R2 key: ``{prefix}/{product_path}/{stamp}/{z}/{x}/{y}.webp``.
"""
from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import shutil
from typing import Iterable, Optional

import numpy as np
from PIL import Image

# Cache-Control + ContentType mirror s1_ingest / meso_poller exactly (§4.3):
# immutable tiles, short-lived manifest. Kept as literals so this module has no
# import dependency on s1_ingest.
CACHE_FRAME = "public, max-age=31536000, immutable"   # tiles: image/webp
CACHE_MANIFEST = "max-age=30"                          # latest_times.json: application/json
TILE_CONTENT_TYPE = "image/webp"


# ---------------------------------------------------------------------------
# PyramidSpec -- the encode/cut knobs (per product; defaults match §4.2)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PyramidSpec:
    tile_size: int = 512          # §4.2 "WebP, 512 px, served @2x"
    quality: int = 90             # §4.2: q80-85 photographic; q90/near-lossless for
                                  # hard colortable edges (rainbow_ir/Dvorak-BD banding)
    method: int = 6               # WebP effort; matches transcode_frame
    lossless: bool = False        # §4.2 near-lossless option for hard palette edges
    skip_empty: bool = True       # drop fully-transparent tiles (§4.2 PUT bound)
    min_zoom: int = 0


# ---------------------------------------------------------------------------
# The cut: raster -> {(z,x,y): webp_bytes}
# ---------------------------------------------------------------------------
def _as_rgba(raster: np.ndarray) -> np.ndarray:
    """Coerce an HxWx3 or HxWx4 uint8 array to a contiguous HxWx4 uint8 RGBA."""
    a = np.asarray(raster)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"raster must be HxWx3 or HxWx4 uint8, got {a.shape}/{a.dtype}")
    if a.shape[2] == 3:
        alpha = np.full(a.shape[:2] + (1,), 255, dtype=np.uint8)
        a = np.concatenate([a, alpha], axis=2)
    return np.ascontiguousarray(a)


def max_zoom_for(width: int, height: int, tile_size: int = 512) -> int:
    """Levels of halving until the long edge fits one tile (z=maxzoom = native)."""
    longest = max(int(width), int(height))
    if longest <= tile_size:
        return 0
    return int(math.ceil(math.log2(longest / tile_size)))


def level_dims(width: int, height: int, maxzoom: int, z: int) -> tuple[int, int]:
    """Pixel dims of the level-z image (z=maxzoom is native W,H)."""
    scale = 2 ** (maxzoom - z)
    return max(1, math.ceil(width / scale)), max(1, math.ceil(height / scale))


def _encode_webp(tile_rgba: np.ndarray, spec: PyramidSpec) -> Optional[bytes]:
    """Encode a T x T RGBA tile to lossy WebP; None if fully transparent + skip."""
    if spec.skip_empty and int(tile_rgba[..., 3].max()) == 0:
        return None
    im = Image.fromarray(tile_rgba, "RGBA")
    buf = io.BytesIO()
    if spec.lossless:
        im.save(buf, "WEBP", lossless=True, method=spec.method)
    else:
        im.save(buf, "WEBP", quality=spec.quality, method=spec.method)
    return buf.getvalue()


def cut_pyramid(raster: np.ndarray, spec: PyramidSpec = PyramidSpec()) -> dict:
    """Cut one imagery raster into a flat-native XYZ WebP pyramid.

    Returns ``{"maxzoom": int, "image_px": (W,H), "tiles": {(z,x,y): bytes},
    "tile_counts": {z: n}}``. Deterministic: identical raster+spec -> identical
    tile bytes (Lanczos + fixed WebP knobs, no randomness) -- the property the
    idempotency + parity tests rely on.
    """
    rgba = _as_rgba(raster)
    H, W = rgba.shape[:2]
    T = spec.tile_size
    maxzoom = max_zoom_for(W, H, T)
    src = Image.fromarray(rgba, "RGBA")

    tiles: dict[tuple[int, int, int], bytes] = {}
    counts: dict[int, int] = {}
    for z in range(spec.min_zoom, maxzoom + 1):
        lw, lh = level_dims(W, H, maxzoom, z)
        level = src if (lw, lh) == (W, H) else src.resize((lw, lh), Image.LANCZOS)
        larr = np.asarray(level)  # lh x lw x 4
        nx = math.ceil(lw / T)
        ny = math.ceil(lh / T)
        n = 0
        for ty in range(ny):
            for tx in range(nx):
                x0, y0 = tx * T, ty * T
                x1, y1 = min(x0 + T, lw), min(y0 + T, lh)
                canvas = np.zeros((T, T, 4), dtype=np.uint8)  # transparent pad
                canvas[: y1 - y0, : x1 - x0] = larr[y0:y1, x0:x1]
                data = _encode_webp(canvas, spec)
                if data is None:
                    continue
                tiles[(z, tx, ty)] = data
                n += 1
        counts[z] = n
    return {"maxzoom": maxzoom, "image_px": (W, H), "tiles": tiles,
            "tile_counts": counts}


# ---------------------------------------------------------------------------
# Object stores -- the injected R2 interface (put_bytes/put_json/head/delete/
# list_keys). FilesystemStore mirrors the R2 key layout on local disk so the
# whole path is verifiable in a Codespace with no R2 credentials; on the box the
# runner injects s1_ingest.R2 (same 5 methods).
# ---------------------------------------------------------------------------
class FilesystemStore:
    """A local-disk stand-in for the R2 client, key-for-key. `root/<key>`."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def put_bytes(self, key: str, data: bytes, content_type: str, cache: str) -> bool:
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)
        return True

    def put_json(self, key: str, obj: dict, cache: str) -> bool:
        return self.put_bytes(
            key, json.dumps(obj, separators=(",", ":")).encode(),
            "application/json", cache)

    def head(self, key: str) -> bool:
        return os.path.isfile(self._path(key))

    def get_bytes(self, key: str) -> Optional[bytes]:
        try:
            with open(self._path(key), "rb") as fh:
                return fh.read()
        except OSError:
            return None

    def delete(self, keys: Iterable[str]) -> None:
        for k in keys:
            try:
                os.remove(self._path(k))
            except OSError:
                pass

    def list_keys(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        # Walk the directory that contains the prefix; return keys (root-relative)
        # that startswith the prefix -- matches R2 list_objects_v2 prefix semantics.
        start = base if os.path.isdir(base) else os.path.dirname(base)
        out: list[str] = []
        for dirpath, _dirs, files in os.walk(start):
            for f in files:
                key = os.path.relpath(os.path.join(dirpath, f), self.root)
                if key.startswith(prefix):
                    out.append(key)
        return out


# ---------------------------------------------------------------------------
# Emit -- write a frame's pyramid + (optionally) the manifest, idempotently
# ---------------------------------------------------------------------------
def emit_pyramid(entry, store, prefix: str, stamp: str, raster: np.ndarray,
                 bounds, spec: PyramidSpec = PyramidSpec(), *,
                 skip_if_present: bool = True,
                 scheme: str = "flat-native-xyz",
                 bt_png: Optional[bytes] = None,
                 max_zoom: Optional[int] = None) -> dict:
    """Cut `raster` and PUT every tile under
    ``{prefix}/{entry.product_path}/{stamp}/{z}/{x}/{y}.webp``, then write the
    per-frame completion marker (``_ready.json``) LAST.

    Idempotent via the marker, NOT via a tile: the marker exists only once every
    tile PUT has succeeded, so a partial/interrupted emit (tiles present, marker
    absent -- an R2 error after boto's retries, or a crash/OOM/SIGTERM mid-loop)
    leaves ``head(ready_key)==False`` and the next run RE-RENDERS instead of
    skipping it as 'duplicate'. This also decouples dedup from a hardcoded z0
    tile, so it is correct for any ``min_zoom`` / ``skip_empty`` outcome. Returns
    a per-frame meta dict (stamp, bounds, image_px, maxzoom, n_tiles, outcome).
    """
    ready_key = entry.ready_key(prefix, stamp)
    if skip_if_present and store.head(ready_key):
        return {"stamp": stamp, "outcome": "duplicate", "n_tiles": 0,
                "bounds": list(bounds), "image_px": None, "maxzoom": None,
                "scheme": scheme}

    if scheme == "webmercator-xyz":
        import s2_webmerc                          # lazy: scipy only when reprojecting
        # max_zoom caps the pyramid (cron pre-renders z0..5; z6+ = on-demand emit).
        cut = s2_webmerc.cut_webmerc_pyramid(raster, bounds, spec, maxzoom=max_zoom)
    else:
        cut = cut_pyramid(raster, spec)
    n = 0
    for (z, x, y), data in cut["tiles"].items():
        if not store.put_bytes(entry.tile_key(prefix, stamp, z, x, y),
                               data, TILE_CONTENT_TYPE, CACHE_FRAME):
            raise IOError(f"tile PUT failed: {entry.tile_key(prefix, stamp, z, x, y)}")
        n += 1
    # Calibrated BT data raster beside the tiles (before the marker, so it is part
    # of the atomically-completed frame the inspector can rely on).
    if bt_png is not None:
        if not store.put_bytes(entry.bt_key(prefix, stamp), bt_png,
                               "image/png", CACHE_FRAME):
            raise IOError(f"BT-raster PUT failed: {entry.bt_key(prefix, stamp)}")
    meta = {"stamp": stamp, "outcome": "rendered", "n_tiles": n,
            "bounds": [float(b) for b in bounds], "image_px": list(cut["image_px"]),
            "maxzoom": cut["maxzoom"], "tile_counts": cut["tile_counts"],
            "scheme": scheme, "has_bt": bt_png is not None}
    # Commit marker LAST -- only now is the frame's pyramid provably whole.
    if not store.put_json(ready_key, {"image_px": meta["image_px"],
                                      "maxzoom": meta["maxzoom"],
                                      "bounds": meta["bounds"],
                                      "tile_size": spec.tile_size}, CACHE_MANIFEST):
        raise IOError(f"ready-marker PUT failed: {ready_key}")
    return meta


def write_tiled_manifest(entry, store, prefix: str, stamps: Iterable[str],
                         bounds, image_px, maxzoom: int, as_of, *,
                         spec: PyramidSpec = PyramidSpec(),
                         scheme: str = "flat-native-xyz",
                         bt: Optional[dict] = None,
                         members: Optional[list] = None) -> dict:
    """Build + PUT the tiled slider manifest (§4.1 tiled variant, superset).

    The viewer NEVER lists the bucket: `tile` is a product-relative path
    template; for flat-native it derives the per-zoom grid from image_px+maxzoom,
    for webmercator-xyz it uses the global XYZ grid + `bounds` to fitBounds.
    """
    lt = entry.build_tiled_latest_times(
        stamps, bounds=bounds, image_px=image_px, maxzoom=maxzoom, as_of=as_of,
        tile_size=spec.tile_size, min_zoom=spec.min_zoom, scheme=scheme, bt=bt,
        members=members)
    store.put_json(entry.latest_times_key(prefix), lt, CACHE_MANIFEST)
    return lt


def prune_tiles(entry, store, prefix: str, dead_stamps: Iterable[str]) -> int:
    """Delete EVERY tile under each dead stamp (a pyramid has many keys/stamp --
    the single-key prune the S1 flow uses would orphan tiles forever)."""
    n = 0
    for stamp in dead_stamps:
        keys = store.list_keys(entry.tile_stamp_prefix(prefix, stamp))
        if keys:
            store.delete(keys)
            n += len(keys)
    return n


def complete_stamps(entry, store, prefix: str) -> list:
    """Recover COMPLETE frames from R2 reality (cold start / manifest rebuild):
    every stamp that has a ``_ready.json`` marker, paired with its pyramid
    maxzoom (the max tile z present). Partial emits (tiles but no marker) are
    excluded so the manifest never advertises an incomplete frame; the maxzoom
    lets the manifest builder keep one geometry per product (drop stamps cut at a
    different pyramid_px). Pure ``list_keys`` + key parsing -- no GET."""
    ready: set = set()
    maxz: dict = {}
    for key in store.list_keys(entry.tile_stamp_prefix(prefix, "")):
        rs = entry.stamp_from_ready_key(key)
        if rs is not None:
            ready.add(rs)
            continue
        ts = entry.stamp_from_tile_key(key)
        if ts is not None:
            z = int(key[: -len(entry.frame_ext)].split("/")[-3])
            if z > maxz.get(ts, -1):
                maxz[ts] = z
    return [(s, maxz.get(s, 0)) for s in sorted(ready)]


def stamps_from_store(entry, store, prefix: str) -> list:
    """Sorted list of COMPLETE stamps (marker present). Convenience over
    complete_stamps for callers that don't need per-frame maxzoom."""
    return [s for s, _ in complete_stamps(entry, store, prefix)]
