"""Minimal canonical true-color reference pixels for the M0 safety gate.

These values are executable fixtures, not a copy of the research ledger.  In
particular, the two 8-bit columns are deliberately separate: a floor build is
checked against ``dn_floor_production`` and a round-half-up build against
``dn_round_half_up``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np


REFLECTANCE_ATOL = 1e-6
DISPLAY_ATOL = 1e-6
EIGHT_BIT_DN_ATOL = 0
EIGHT_BIT_DN_SCOPE = (
    "0, against the DN column matching the quantiser the build declares "
    "(floor OR round_half_up). Never compare a floor build against the "
    "round_half_up column."
)

FloorProduction = Literal["floor_production"]
RoundHalfUp = Literal["round_half_up"]
Quantizer = FloorProduction | RoundHalfUp


@dataclass(frozen=True, slots=True)
class Bands:
    """Reflectance values keyed by their physical sensor role."""

    red: float
    blue: float
    veggie: float
    green: float | None = None


@dataclass(frozen=True, slots=True)
class ReferencePixel:
    """The spatially pointwise stages needed to protect the color recipe."""

    id: str
    sensor: str
    cos_sza: float
    reflectance_factor: Bands
    sunz_corrected: Bands
    rayleigh_corrected: Bands
    synthesized_green: float
    linear_rgb: tuple[float, float, float]
    display_rgb: tuple[float, float, float]
    dn_floor_production: tuple[int, int, int]
    dn_round_half_up: tuple[int, int, int]


# Exact values protected by V07/V09/V11/V12 and the reference-pixel package.
# Attribute names intentionally match the production module.
PROTECTED_CONSTANTS = MappingProxyType(
    {
        "SUNZ_LIMIT_DEG": 88.0,
        "SUNZ_MAX_DEG": 95.0,
        "RAYLEIGH_ATMOSPHERE": "us-standard",
        "RAYLEIGH_AEROSOL": "rayleigh_only",
        "RAYLEIGH_REDUCE_LOW_DEG": 70.0,
        "RAYLEIGH_REDUCE_HIGH_DEG": 95.0,
        "RAYLEIGH_REDUCE_STRENGTH": 0.6,
        "ABI_GREEN_FRACTIONS": (0.45, 0.45, 0.10),
        "HYBRID_GREEN_FRACTION": 0.15,
        "NDVI_GREEN_LIMITS": (0.15, 0.05),
        "NDVI_GREEN_STRENGTH": 3.0,
        "_CIRA_LOG_ROOT": -1.6516951369518393,
        "_CIRA_DENOM": 1.9887713527138795,
        "SHARPEN_RATIO_MAX": 1.5,
    }
)


DERIVED_INVARIANTS = MappingProxyType(
    {
        "cira_black_clip_reflectance": 0.0223,
        "cira_white_clip_reflectance": 2.1730825067739694,
        "cira_at_unit_reflectance": 0.8305103222137298,
        "cira_dn_per_octave": 38.59802625855648,
        "sunz_multiplier_at_88deg": 28.653708347843732,
        "li_shibata_at_88deg": 18.20495234998178,
        "li_shibata_at_nadir": 1.0000000000000002,
        "cira_dn_at_unit_reflectance_floor_production": 211,
        "cira_dn_at_unit_reflectance_round_half_up": 212,
        "cira_headroom_dn_above_unit_reflectance_floor_production": 44,
    }
)


REFERENCE_PIXELS = (
    ReferencePixel(
        id="P1_deep_convective_cloud_top",
        sensor="abi",
        cos_sza=0.93969262,
        reflectance_factor=Bands(red=0.883311, blue=0.864517, veggie=0.873914),
        sunz_corrected=Bands(red=0.940000, blue=0.920000, veggie=0.929887),
        rayleigh_corrected=Bands(red=0.938569, blue=0.914916, veggie=0.929887),
        synthesized_green=0.927057,
        linear_rgb=(0.938569, 0.927057, 0.914916),
        display_rgb=(0.816666, 0.813971, 0.811092),
        dn_floor_production=(208, 207, 206),
        dn_round_half_up=(208, 208, 207),
    ),
    ReferencePixel(
        id="P2_clear_tropical_ocean",
        sensor="abi",
        cos_sza=0.81915204,
        reflectance_factor=Bands(red=0.019137, blue=0.079723, veggie=0.001638),
        sunz_corrected=Bands(red=0.023362, blue=0.097324, veggie=0.001999),
        rayleigh_corrected=Bands(red=0.004000, blue=0.025000, veggie=0.001999),
        synthesized_green=0.013250,
        linear_rgb=(0.004000, 0.013250, 0.025000),
        display_rgb=(0.000000, 0.000000, 0.024958),
        dn_floor_production=(0, 0, 6),
        dn_round_half_up=(0, 0, 6),
    ),
    ReferencePixel(
        id="P3_vegetated_land",
        sensor="abi",
        cos_sza=0.86602540,
        reflectance_factor=Bands(red=0.046683, blue=0.085814, veggie=0.346410),
        sunz_corrected=Bands(red=0.053905, blue=0.099089, veggie=0.399878),
        rayleigh_corrected=Bands(red=0.035000, blue=0.030000, veggie=0.399878),
        synthesized_green=0.069238,
        linear_rgb=(0.035000, 0.069238, 0.030000),
        display_rgb=(0.098434, 0.247408, 0.064772),
        dn_floor_production=(25, 63, 16),
        dn_round_half_up=(25, 63, 17),
    ),
    ReferencePixel(
        id="P4_thin_cirrus_highSZA",
        sensor="abi",
        cos_sza=0.20791169,
        reflectance_factor=Bands(red=0.060723, blue=0.093148, veggie=0.046780),
        sunz_corrected=Bands(red=0.292062, blue=0.448018, veggie=0.220563),
        rayleigh_corrected=Bands(red=0.237142, blue=0.263938, veggie=0.220563),
        synthesized_green=0.247542,
        linear_rgb=(0.237142, 0.247542, 0.263938),
        display_rgb=(0.516250, 0.525623, 0.539628),
        dn_floor_production=(131, 134, 137),
        dn_round_half_up=(132, 134, 138),
    ),
    ReferencePixel(
        id="P5_ahi_vegetated_land_native_green",
        sensor="ahi",
        cos_sza=0.86602540,
        reflectance_factor=Bands(
            red=0.046648,
            blue=0.086192,
            green=0.089784,
            veggie=0.346410,
        ),
        sunz_corrected=Bands(
            red=0.053864,
            blue=0.099525,
            green=0.103673,
            veggie=0.399878,
        ),
        rayleigh_corrected=Bands(
            red=0.035000,
            blue=0.030000,
            green=0.055000,
            veggie=0.399878,
        ),
        synthesized_green=0.106732,
        linear_rgb=(0.035000, 0.106732, 0.030000),
        display_rgb=(0.098434, 0.341914, 0.064772),
        dn_floor_production=(25, 87, 16),
        dn_round_half_up=(25, 87, 17),
    ),
)


def expected_dn(pixel: ReferencePixel, quantizer: Quantizer) -> tuple[int, int, int]:
    """Select the explicitly matching authoritative DN column."""

    if quantizer == "floor_production":
        return pixel.dn_floor_production
    if quantizer == "round_half_up":
        return pixel.dn_round_half_up
    raise ValueError(f"unknown reference quantizer: {quantizer!r}")


def quantize_display(display_rgb: np.ndarray, quantizer: Quantizer) -> np.ndarray:
    """Apply a declared validation quantizer once to clipped display RGB."""

    scaled = np.clip(np.asarray(display_rgb, dtype=np.float64), 0.0, 1.0) * 255.0
    if quantizer == "floor_production":
        return np.floor(scaled).astype(np.uint8)
    if quantizer == "round_half_up":
        return np.floor(scaled + 0.5).astype(np.uint8)
    raise ValueError(f"unknown reference quantizer: {quantizer!r}")
