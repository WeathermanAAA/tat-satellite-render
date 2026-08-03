#!/usr/bin/env python3
"""Tar-container tile publishing for the s2 pyramids (the hafs_render #27
pattern, ported; 2026-08-03 R2 cost incident).

WHY: publishing every tile as its own object costs one Class A PUT per tile
-- measured ~2.06M tile PUTs/day across the 11 emit lanes ($4.50/M). Packing
each frame's tiles into a handful of uncompressed USTAR blocks collapses a
301-tile full-disk frame from ~305 objects to ~6, while keeping the READ
path per-tile: the frame's index carries every member's exact byte offset,
so the viewer issues ONE HTTP Range request per tile and receives exactly
the WebP it would have fetched before. No client-side tar parsing, no
whole-block downloads, CDN range caching intact.

WHY ZOOM-BANDED, not hafs' geometric-by-arrival: a pyramid's tiles all
exist at once (the cut completes before any upload), so there is no
progressive-publish constraint to honor. What the read path wants is
LOCALITY: a viewer at zoom z fetches a viewport of z-tiles, so tiles of one
zoom belong in one object -- after the first touch the CDN holds the block
and every later tile in that viewport is an edge-cached range. Overview
levels (z0..OVERVIEW_MAX_Z) are tiny and always fetched together, so they
share one block.

KEY LAYOUT (beside, not replacing, the flat scheme):
    {prefix}/{product}/{stamp}/t{first}-{last}.tar     tile blocks
    {prefix}/{product}/{stamp}/tiles.z{maxzoom}.json   member index
    {prefix}/{product}/{stamp}/bt.png                  unchanged
    {prefix}/{product}/{stamp}/_ready.json             unchanged, LAST
The index filename carries the pyramid maxzoom because a container frame
has no {z}/ dirs for complete_stamps' delimiter probe to read -- the
filename IS the geometry answer, at zero extra requests.

Offsets are computed arithmetically (USTAR: 512-byte header + data padded
to 512) and then VERIFIED by re-reading the archive -- a wrong offset would
serve a viewer garbage bytes, so like hafs_container this module never
trusts its own math. Stdlib only.
"""
from __future__ import annotations

import io
import json
import tarfile
from typing import Dict, Tuple

#: Overview zooms z0..this share one block (a full-disk z0-4 band is ~85
#: small tiles the viewer wants together anyway).
OVERVIEW_MAX_Z = 4

#: Index key suffix -- the {maxzoom} segment is load-bearing (see module doc).
INDEX_KEY = "tiles.z{maxzoom}.json"

#: Block key: t{first zoom}-{last zoom}.tar (collision-free within a stamp).
BLOCK_KEY = "t{first}-{last}.tar"


def member_name(z: int, x: int, y: int, ext: str = ".webp") -> str:
    return f"{z}/{x}/{y}{ext}"


def plan_zoom_bands(zooms) -> list:
    """Split the present zoom levels into bands: one overview band
    (z <= OVERVIEW_MAX_Z) plus one band per deeper level. Deterministic,
    operates on the zooms that EXIST."""
    zs = sorted(set(int(z) for z in zooms))
    bands = []
    overview = [z for z in zs if z <= OVERVIEW_MAX_Z]
    if overview:
        bands.append(overview)
    for z in zs:
        if z > OVERVIEW_MAX_Z:
            bands.append([z])
    return bands


def block_key(band) -> str:
    return BLOCK_KEY.format(first=band[0], last=band[-1])


def build_block(members) -> Tuple[bytes, Dict[str, list]]:
    """One uncompressed USTAR tar in memory from [(name, payload_bytes)...],
    returning (tar_bytes, {name: [data_offset, size]}). Offsets computed
    arithmetically, then verified by decode-back before anything is
    published."""
    index: Dict[str, list] = {}
    buf = io.BytesIO()
    offset = 0
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            # Fixed mtime: identical tiles -> byte-identical blocks, so a
            # byte-diff between two builds always means a CONTENT diff.
            info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
            data_off = offset + 512          # USTAR header = one 512B record
            index[name] = [data_off, len(payload)]
            offset = data_off + len(payload) + ((512 - len(payload) % 512) % 512)
    data = buf.getvalue()
    _verify_block(data, index)
    return data, index


def _verify_block(data: bytes, index: Dict[str, list]) -> None:
    """Never trust the offset math: re-open the built tar and demand every
    member's actual data offset and size match the index EXACTLY."""
    seen: Dict[str, list] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        for m in tar.getmembers():
            seen[m.name] = [m.offset_data, m.size]
    if seen != index:
        raise RuntimeError(
            f"container index mismatch: computed {index}, archive says {seen}")


def build_frame_containers(tiles: Dict[tuple, bytes], maxzoom: int,
                           ext: str = ".webp") -> dict:
    """Plan + build ALL blocks for one frame's tile dict {(z,x,y): bytes}.

    Returns {"blocks": {key: tar_bytes}, "index": <the tiles.z{N}.json
    payload>}: index = {"format": "ustar-v1", "read": "range",
    "maxzoom": N, "tile_size"/"count" facts, "blocks": {key: size},
    "tiles": {"z/x/y.webp": [blockKey, data_off, size]}}. skip_empty tiles
    are simply absent from "tiles" -- a missing entry renders transparent,
    the same slippy contract as a missing object today."""
    zooms = sorted(set(z for z, _x, _y in tiles))
    blocks: Dict[str, bytes] = {}
    tile_map: Dict[str, list] = {}
    block_sizes: Dict[str, int] = {}
    for band in plan_zoom_bands(zooms):
        members = [(member_name(z, x, y, ext), tiles[(z, x, y)])
                   for z, x, y in sorted(tiles)
                   if z in band]
        key = block_key(band)
        data, index = build_block(members)
        blocks[key] = data
        block_sizes[key] = len(data)
        for name, (off, size) in index.items():
            tile_map[name] = [key, off, size]
    return {
        "blocks": blocks,
        "index": {
            "format": "ustar-v1",
            "read": "range",
            "maxzoom": int(maxzoom),
            "count": len(tile_map),
            "blocks": block_sizes,
            "tiles": tile_map,
        },
    }


def index_key_name(maxzoom: int) -> str:
    return INDEX_KEY.format(maxzoom=int(maxzoom))


def maxzoom_from_index_key(key: str):
    """Parse the pyramid maxzoom out of a '.../tiles.z{N}.json' key, or None."""
    base = key.rsplit("/", 1)[-1]
    if base.startswith("tiles.z") and base.endswith(".json"):
        mid = base[len("tiles.z"):-len(".json")]
        if mid.isdigit():
            return int(mid)
    return None


def dumps_index(index: dict) -> bytes:
    return json.dumps(index, separators=(",", ":")).encode()
