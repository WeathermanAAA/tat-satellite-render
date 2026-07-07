#!/usr/bin/env python3
"""Stage-2 calibrated BRIGHTNESS-TEMPERATURE data raster (Phase 2c inspector).

The colorized rainbow_ir tiles are many-to-one in BT (you cannot read a
temperature back out of the color). So the pixel/BT inspector reads from a
SEPARATE compact calibrated data raster emitted beside the tiles: one small
equirectangular PNG per frame that packs BT (deg C) losslessly so the viewer can
sample a real temperature at any lon/lat (report §6: "ship a compact calibrated
data raster beside each display tile; hover samples brightness temp, not the
colorized PNG").

ENCODING (u16 packed into R,G of a LOSSLESS PNG; A = validity):
  value_u16 = round((bt_c - OFFSET) / SCALE)   clamped to [0, 65535]
  R = value_u16 >> 8 ; G = value_u16 & 255 ; B = 0 ; A = 255 if finite else 0
  decode: bt_c = (R*256 + G) * SCALE + OFFSET   (A==0 -> no data)
SCALE=0.01, OFFSET=-120 covers -120..+535 deg C at 0.01 deg precision -- lossless
for the whole IR range. PNG (never WebP) so the packed bytes are exact.

Pure numpy + PIL. The equirect resample of the source curvilinear BT lives in
s2_imagery (it owns the fetch); this module only encodes/decodes + describes.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

BT_SCALE = 0.01
BT_OFFSET = -120.0
BT_ENCODING = "u16-rg-hi-lo"
BT_EXT = ".png"


def encode_bt_png(bt_c: np.ndarray) -> bytes:
    """Pack a HxW BT (deg C, NaN = no data) raster into a lossless RGBA PNG."""
    a = np.asarray(bt_c, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"bt_c must be 2D, got {a.shape}")
    valid = np.isfinite(a)
    u = np.clip(np.round((np.where(valid, a, 0.0) - BT_OFFSET) / BT_SCALE),
                0, 65535).astype(np.uint16)
    h, w = a.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = (u >> 8).astype(np.uint8)      # high byte
    rgba[..., 1] = (u & 0xFF).astype(np.uint8)    # low byte
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=False)   # lossless
    return buf.getvalue()


def decode_value(r: int, g: int, a: int):
    """Decode one packed pixel back to BT (deg C), or None if no data."""
    if a == 0:
        return None
    return (int(r) * 256 + int(g)) * BT_SCALE + BT_OFFSET


def decode_bt_png(png: bytes) -> np.ndarray:
    """Decode a packed PNG back to a HxW float BT (deg C) array, NaN off-data."""
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    u = a[..., 0].astype(np.uint32) * 256 + a[..., 1].astype(np.uint32)
    out = u.astype(np.float64) * BT_SCALE + BT_OFFSET
    out[a[..., 3] == 0] = np.nan
    return out


def bt_descriptor(product_path: str, bounds, dims, ext: str = BT_EXT) -> dict:
    """The manifest 'bt' block: how the viewer finds + decodes the BT raster."""
    return {
        "path": f"{product_path}/{{t}}/bt{ext}",
        "encoding": BT_ENCODING,
        "scale": BT_SCALE,
        "offset": BT_OFFSET,
        "units": "degC",
        "dims": [int(dims[0]), int(dims[1])],          # [w, h]
        "bounds": [float(b) for b in bounds],           # [W,S,E,N], equirect
    }
