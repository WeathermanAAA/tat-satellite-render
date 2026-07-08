#!/usr/bin/env python3
"""Stage-2 IMAGERY RECIPES -- the declarative band-math library behind the
multi-product explorer suite (TAT_satellite_toolkit_report §2, rgb_recipes.csv).

One Recipe row == one explorer product. The registry (s2_registry) generates a
tiled ProductEntry per row, and s2_imagery renders every row through ONE engine
-- adding a product is a config row here, not code (report §6 "declarative
registry ... products are band-math recipes").

RECIPE FORMULATION (the standard EUMETSAT/NOAA quick-guide scaling):
    gun = clip((x - lo) / (hi - lo), 0, 1) ** (1/gamma)
where x is a band brightness temperature (Kelvin), a band BT difference
(Kelvin), or a reflectance factor (0..1, the CMI value of a reflective band,
NOT solar-zenith normalized -- matching the quick-guide band math). ``lo > hi``
expresses an inverted gun (the same formula flips sign in the denominator).

Numbers below are the GOES-R ABI quick-guide values (NOAA STAR / CIRA /
EUMETSAT heritage); each row cites its guide. The module is STDLIB-ONLY at
import time (s2_registry, which is stdlib-only, imports the table); numpy is
imported lazily inside the compute functions.

KINDS:
  * "rgb_guns"       -- 3-gun band math (Air Mass, Dust, ... and the reflective
                        single channels, expressed as 3 identical gray guns).
  * "single_palette" -- one emissive band colorized with a FROZEN tat_palettes
                        enhancement (the floater/meso products' exact look:
                        rainbow_ir / dvorak / wv_tat / grayscale) -- zero visual
                        change vs the current renders.
  * "sandwich"       -- VIS-luminance x color-enhanced-IR blend (EUMeTrain
                        Sandwich; a blend, not a 3-gun RGB).
  * "truecolor"      -- delegated VERBATIM to the frozen satellites.py
                        fetch_true_color / truecolor.assemble_truecolor path
                        (CIMSS synthetic green + CIRA GeoColor order + night
                        fade). No recipe math here at all.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

# ABI band native resolution (km) -- drives the per-product pyramid size.
BAND_NATIVE_KM = {1: 1.0, 2: 0.5, 3: 1.0, 4: 2.0, 5: 1.0, 6: 2.0,
                  7: 2.0, 8: 2.0, 9: 2.0, 10: 2.0, 11: 2.0, 12: 2.0,
                  13: 2.0, 14: 2.0, 15: 2.0, 16: 2.0}
# Bands 7-16 are emissive (CMI = brightness temperature, K); 1-6 reflective
# (CMI = reflectance factor, 0..1). Deterministic for ABI L2 CMIP.
EMISSIVE_BANDS = frozenset(range(7, 17))

# Sandwich blend knobs: out = ir_rgb * (FLOOR + (1-FLOOR) * clip(vis)**GAMMA).
# The luminance floor keeps the IR enhancement readable where VIS is dark
# (sober, data-forward -- the EUMeTrain "VIS texture, IR color" intent).
SANDWICH_LUMA_FLOOR = 0.30
SANDWICH_VIS_GAMMA = 0.7


@dataclasses.dataclass(frozen=True)
class Gun:
    """One RGB gun: a band (or band-difference) scaled to 0..1."""
    expr: tuple            # ("band", n) | ("diff", a, b)  -> x = Ca or Ca - Cb
    lo: float              # quick-guide range min (lo > hi = inverted gun)
    hi: float
    gamma: float = 1.0
    kind: str = "bt"       # "bt" (Kelvin) | "refl" (reflectance factor 0..1)

    @property
    def bands(self) -> tuple:
        return tuple(self.expr[1:])


@dataclasses.dataclass(frozen=True)
class Recipe:
    key: str               # band_key / URL slug (registry: sat/goes19/conus/{key})
    title: str             # display title (viewer picker)
    group: str             # "composite" | "rgb" | "channel"
    kind: str              # "rgb_guns" | "single_palette" | "sandwich" | "truecolor"
    guns: tuple = ()       # rgb_guns: exactly 3
    band: int = 0          # single_palette: the one band
    enhancement: str = ""  # single_palette: FROZEN tat_palettes name
    bt_band: int = 0       # band backing the BT inspector raster (0 = none)
    day_only: bool = False
    source: str = ""       # primary quick-guide citation

    @property
    def bands(self) -> tuple:
        """Sorted union of ABI bands this recipe needs."""
        need = set()
        if self.kind == "rgb_guns":
            for g in self.guns:
                need.update(g.bands)
        elif self.kind == "single_palette":
            need.add(self.band)
        elif self.kind == "sandwich":
            need.update((2, 13))
        elif self.kind == "truecolor":
            need.update((1, 2, 3, 13))   # red/blue/veggie + clean-IR night fade
        if self.bt_band:
            need.add(self.bt_band)
        return tuple(sorted(need))

    @property
    def finest_km(self) -> float:
        return min(BAND_NATIVE_KM[b] for b in self.bands) if self.bands else 2.0


def _gray_channel(band: int, lo: float = 0.0, hi: float = 1.0,
                  gamma: float = 2.0) -> tuple:
    """A reflective single channel as 3 identical gray guns (sqrt-ish stretch,
    the standard VIS display gamma)."""
    g = Gun(("band", band), lo, hi, gamma, kind="refl")
    return (g, g, g)


RECIPES: tuple[Recipe, ...] = (
    # ---- composites -------------------------------------------------------
    Recipe("truecolor", "True Color (GeoColor-lite)", "composite", "truecolor",
           source="CIMSS synthetic green (Bah et al.) + CIRA GeoColor order; "
                  "frozen tsr truecolor.py pipeline incl. night IR fade"),
    Recipe("sandwich", "Sandwich (VIS × IR)", "composite", "sandwich",
           bt_band=13, day_only=True,
           source="EUMeTrain Sandwich quick guide (VIS texture × color-enhanced "
                  "IR C13; rendered with the locked rainbow_ir)"),

    # ---- multispectral RGBs (GOES-R ABI quick-guide scalings) -------------
    Recipe("airmass", "Air Mass RGB", "rgb", "rgb_guns", guns=(
        Gun(("diff", 8, 10), -26.2, 0.6),
        Gun(("diff", 12, 13), -43.2, 6.7),
        Gun(("band", 8), 243.9, 208.5),          # inverted (warm=dark)
    ), source="NOAA STAR/CIRA ABI Air Mass RGB quick guide"),
    Recipe("dust", "Dust RGB", "rgb", "rgb_guns", guns=(
        Gun(("diff", 15, 13), -6.7, 2.6),
        Gun(("diff", 14, 11), -0.5, 20.0, 2.5),
        Gun(("band", 13), 261.2, 288.7),
    ), source="EUMETSAT/CIRA ABI Dust RGB quick guide"),
    Recipe("firetemp", "Fire Temperature RGB", "rgb", "rgb_guns", guns=(
        Gun(("band", 7), 273.15, 333.15, 0.4),
        Gun(("band", 6), 0.0, 1.0, kind="refl"),
        Gun(("band", 5), 0.0, 0.75, kind="refl"),
    ), day_only=True, source="NOAA/CIRA ABI Fire Temperature RGB quick guide"),
    Recipe("daycloudphase", "Day Cloud Phase Distinction", "rgb", "rgb_guns", guns=(
        Gun(("band", 13), 280.65, 219.65),       # 7.5 .. -53.5 C, inverted
        Gun(("band", 2), 0.0, 0.78, kind="refl"),
        Gun(("band", 5), 0.01, 0.59, kind="refl"),
    ), day_only=True, source="NOAA/CIRA ABI Day Cloud Phase Distinction quick guide"),
    Recipe("nightmicro", "Nighttime Microphysics", "rgb", "rgb_guns", guns=(
        Gun(("diff", 15, 13), -6.7, 2.6),
        Gun(("diff", 13, 7), -3.1, 5.2),
        Gun(("band", 13), 243.55, 292.65),
    ), source="EUMETSAT/CIRA ABI Nighttime Microphysics quick guide"),

    # ---- the 16 ABI single channels (C13 clean-IR = the existing
    # goes19-conus-ir row; irbd adds the corrected Dvorak BD look) ----------
    Recipe("c01", "C01 · 0.47 µm (Blue)", "channel", "rgb_guns",
           guns=_gray_channel(1), day_only=True, source="NCEI ABI L1b band list"),
    Recipe("c02", "C02 · 0.64 µm (Red visible)", "channel", "rgb_guns",
           guns=_gray_channel(2), day_only=True, source="NCEI ABI L1b band list"),
    Recipe("c03", "C03 · 0.86 µm (Veggie NIR)", "channel", "rgb_guns",
           guns=_gray_channel(3), day_only=True, source="NCEI ABI L1b band list"),
    Recipe("c04", "C04 · 1.37 µm (Cirrus)", "channel", "rgb_guns",
           guns=_gray_channel(4), day_only=True, source="NCEI ABI L1b band list"),
    Recipe("c05", "C05 · 1.6 µm (Snow/Ice)", "channel", "rgb_guns",
           guns=_gray_channel(5), day_only=True, source="NCEI ABI L1b band list"),
    Recipe("c06", "C06 · 2.2 µm (Cloud particle)", "channel", "rgb_guns",
           guns=_gray_channel(6), day_only=True, source="NCEI ABI L1b band list"),
    Recipe("c07", "C07 · 3.9 µm (Shortwave IR)", "channel", "single_palette",
           band=7, enhancement="grayscale", bt_band=7,
           source="meso/floater 'swir' product (frozen grayscale enhancement)"),
    Recipe("c08", "C08 · 6.2 µm (Upper-level WV)", "channel", "single_palette",
           band=8, enhancement="wv_tat", bt_band=8,
           source="floater 'wv_up' product (frozen wv_tat enhancement)"),
    Recipe("c09", "C09 · 6.9 µm (Mid-level WV)", "channel", "single_palette",
           band=9, enhancement="wv_tat", bt_band=9,
           source="wv_tat (frozen), band per NCEI ABI L1b list"),
    Recipe("c10", "C10 · 7.3 µm (Low-level WV)", "channel", "single_palette",
           band=10, enhancement="wv_tat", bt_band=10,
           source="floater 'wv_low' product (frozen wv_tat enhancement)"),
    Recipe("c11", "C11 · 8.4 µm (Cloud-top phase)", "channel", "single_palette",
           band=11, enhancement="rainbow_ir", bt_band=11,
           source="rgb_recipes.csv: TAT rainbow_ir default on IR channels"),
    Recipe("c12", "C12 · 9.6 µm (Ozone)", "channel", "single_palette",
           band=12, enhancement="rainbow_ir", bt_band=12,
           source="rgb_recipes.csv: TAT rainbow_ir default on IR channels"),
    Recipe("irbd", "C13 · 10.3 µm (IR, Dvorak BD)", "channel", "single_palette",
           band=13, enhancement="dvorak", bt_band=13,
           source="meso/floater 'irbd' product (frozen corrected Dvorak BD)"),
    Recipe("c14", "C14 · 11.2 µm (IR window)", "channel", "single_palette",
           band=14, enhancement="rainbow_ir", bt_band=14,
           source="rgb_recipes.csv: TAT rainbow_ir default on IR channels"),
    Recipe("c15", "C15 · 12.3 µm (Dirty IR)", "channel", "single_palette",
           band=15, enhancement="rainbow_ir", bt_band=15,
           source="rgb_recipes.csv: TAT rainbow_ir default on IR channels"),
    Recipe("c16", "C16 · 13.3 µm (CO₂)", "channel", "single_palette",
           band=16, enhancement="rainbow_ir", bt_band=16,
           source="rgb_recipes.csv: TAT rainbow_ir default on IR channels"),
)

RECIPES_BY_KEY = {r.key: r for r in RECIPES}


# ---------------------------------------------------------------------------
# Engine (numpy lazy -- the table above stays stdlib-importable)
# ---------------------------------------------------------------------------
def gun_input(gun: Gun, bands: dict):
    """x for a gun from the fetched band dict {band_number: ndarray}.
    BT guns expect Kelvin; refl guns expect reflectance factor 0..1."""
    if gun.expr[0] == "band":
        return bands[gun.expr[1]]
    a, b = gun.expr[1], gun.expr[2]
    return bands[a] - bands[b]


def scale_gun(x, gun: Gun):
    """clip((x-lo)/(hi-lo), 0, 1) ** (1/gamma); lo>hi = inverted. NaN stays NaN
    (off-disk pixels carry into the alpha mask, not into a fake color)."""
    import numpy as np
    with np.errstate(invalid="ignore"):
        v = (x - gun.lo) / (gun.hi - gun.lo)
        v = np.clip(v, 0.0, 1.0)
        if gun.gamma != 1.0:
            v = v ** (1.0 / gun.gamma)
    return v


def compute_rgb(recipe: Recipe, bands: dict):
    """3-gun band math -> float RGB HxWx3 in 0..1 (NaN where any input is NaN)."""
    import numpy as np
    if recipe.kind != "rgb_guns" or len(recipe.guns) != 3:
        raise ValueError(f"{recipe.key} is not a 3-gun recipe")
    chans = [scale_gun(gun_input(g, bands), g) for g in recipe.guns]
    return np.dstack(chans).astype(np.float32)


def sandwich_rgb(vis_refl, ir_rgb):
    """EUMeTrain-style sandwich: the color-enhanced IR modulated by VIS
    luminance (texture from VIS, color from IR). Inputs: vis 0..1 HxW,
    ir_rgb 0..1 HxWx3. NaN VIS (off-disk / no data) -> NaN out."""
    import numpy as np
    luma = SANDWICH_LUMA_FLOOR + (1.0 - SANDWICH_LUMA_FLOOR) * (
        np.clip(vis_refl, 0.0, 1.0) ** SANDWICH_VIS_GAMMA)
    return (ir_rgb * luma[..., None]).astype(np.float32)


def rgba_from_rgb(rgb, valid=None):
    """float RGB 0..1 -> uint8 RGBA; alpha 0 where any channel is NaN (or where
    ``valid`` is False). The pyramid wants transparent off-data (§ s2_imagery)."""
    import numpy as np
    finite = np.isfinite(rgb).all(axis=-1)
    if valid is not None:
        finite &= valid
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    scaled = np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, 1.0) * 255.0
    out[..., :3] = (scaled + 0.5).astype(np.uint8)
    out[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
    return out
