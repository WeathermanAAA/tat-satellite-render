"""Colormaps + enhancements for Triple-A-Tropics satellite rendering.

REBUILD (2026-05-28) — temperature-domain architecture.

Every IR/WV enhancement is defined directly over a brightness-temperature
domain in DEGREES CELSIUS and ships with:
  - a matplotlib Colormap oriented cold->warm (cmap index 0 == coldest color),
  - the °C domain endpoints (vmin_c = coldest, vmax_c = warmest),
  - an explicit list of °C tick values for the right-side colorbar,
  - a human label and the set of channel "domains" it's valid for.

render.py plots the data in °C with the enhancement's cmap + a fresh
Normalize(vmin_c, vmax_c) and draws a labeled vertical colorbar on the right,
restyled to the TAT dark theme. Because the data is plotted in real
temperature units, the colorbar ticks are physical °C with no manual remap.

DEFAULT preset: rainbow_ir (the TAT signature look).

Presets:
  - rainbow_ir : TAT signature rainbow IR (TWC-family but personalized) —
                 deep-blue clear/warm, cyan+teal accent at the cold-cloud edge,
                 green -> yellow -> orange -> red -> maroon -> magenta cold
                 sweep, white-hot overshoot core. The site default.
  - dvorak_bd  : canonical Dvorak BD — stepped grayscale with the diagnostic
                 black (-63..-69) -> white (-69..-75) flip. Breakpoints per
                 Andrew's reference table (+9,-30,-41,-53,-63,-69,-75,-80,-85).
  - tat_neon   : Triple-A-Tropics neon — cyan/violet/magenta/gold on dark.
  - wv_tat     : dedicated water-vapor table (dry=amber, moist=teal/blue/white).
  - ir_gray    : standard IR grayscale (cold = white). Alias: "grayscale".
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


# ---------------------------------------------------------------------------
# builder helper
# ---------------------------------------------------------------------------
def _seg_cmap(name, anchors):
    """Build a LinearSegmentedColormap from (temp_C, r, g, b) anchors.

    `anchors` MUST be sorted by increasing temperature (coldest first). The
    coldest anchor maps to cmap position 0.0, the warmest to 1.0, so a
    Normalize(vmin=coldest, vmax=warmest) lines real temperatures up with the
    right colors. Two anchors at the (near-)same temperature produce a hard
    step / flip; spread them apart for a ramp.
    """
    t_cold = anchors[0][0]
    t_warm = anchors[-1][0]
    span = t_warm - t_cold
    rs, gs, bs = [], [], []
    last_x = -1.0
    for t, r, g, b in anchors:
        x = (t - t_cold) / span
        if x <= last_x:
            x = last_x + 1e-6  # keep strictly increasing -> sharp edge
        last_x = x
        rs.append((x, r, r))
        gs.append((x, g, g))
        bs.append((x, b, b))
    return LinearSegmentedColormap(name, {"red": rs, "green": gs, "blue": bs}, N=1024)


# ---------------------------------------------------------------------------
# rainbow_ir — TAT signature (TWC-family, personalized). DEFAULT.
# ---------------------------------------------------------------------------
# cold -> warm. Approved 2026-05-28: deep-blue clear (no white low-cloud band),
# cyan/teal TAT accent at the cold-cloud edge, green/yellow/orange/red/maroon/
# magenta cold sweep, white-hot overshoot core. Domain -95 .. +40 C.
_RAINBOW_IR_ANCHORS = [
    (-95.0, 1.00, 1.00, 1.00),  # white-hot overshoot core
    (-90.0, 0.88, 0.25, 0.98),  # magenta
    (-84.0, 0.62, 0.06, 0.62),  # purple
    (-78.0, 0.42, 0.03, 0.22),  # maroon-purple
    (-72.0, 0.55, 0.02, 0.06),  # dark maroon
    (-66.0, 0.80, 0.05, 0.03),  # deep red
    (-58.0, 0.96, 0.16, 0.04),  # red
    (-50.0, 1.00, 0.45, 0.05),  # orange
    (-44.0, 1.00, 0.70, 0.08),  # amber
    (-38.0, 1.00, 0.93, 0.18),  # yellow
    (-32.0, 0.72, 0.92, 0.20),  # yellow-green
    (-26.0, 0.18, 0.80, 0.22),  # green
    (-20.0, 0.08, 0.66, 0.48),  # teal-green (TAT)
    (-14.0, 0.16, 0.72, 0.90),  # cyan accent (TAT)
    (-7.0,  0.18, 0.52, 0.85),  # azure
    (0.0,   0.12, 0.38, 0.74),  # medium blue
    (12.0,  0.07, 0.26, 0.58),  # deeper blue
    (25.0,  0.04, 0.16, 0.42),  # dark blue
    (40.0,  0.02, 0.07, 0.22),  # very dark navy (warmest clear)
]

# ---------------------------------------------------------------------------
# dvorak_bd — canonical stepped grayscale
# ---------------------------------------------------------------------------
# Breakpoints (°C): +9, -30, -41, -53, -63, -69, -75, -80, -85 (Andrew's table).
# Ramps for the warm region (> -30); HARD steps for the diagnostic bands so the
# black(-63..-69) -> white(-69..-75) flip is razor sharp. Domain -95 .. +40.
_DVORAK_BD_ANCHORS = [
    (-95.0, 0.12, 0.12, 0.12),   # extreme cold wedge (< -85)
    (-85.0, 0.12, 0.12, 0.12),
    (-85.0, 0.34, 0.34, 0.34),   # CDG  darker gray (-80..-85)
    (-80.0, 0.34, 0.34, 0.34),
    (-80.0, 0.56, 0.56, 0.56),   # CMG  medium gray (-75..-80)
    (-75.0, 0.56, 0.56, 0.56),
    (-75.0, 1.00, 1.00, 1.00),   # W    white       (-69..-75)
    (-69.0, 1.00, 1.00, 1.00),
    (-69.0, 0.00, 0.00, 0.00),   # B    black        (-63..-69)  <<< flip
    (-63.0, 0.00, 0.00, 0.00),
    (-63.0, 0.46, 0.46, 0.46),   # MG   medium-dark  (-53..-63)
    (-53.0, 0.46, 0.46, 0.46),
    (-53.0, 0.76, 0.76, 0.76),   # LG   light gray   (-41..-53)
    (-41.0, 0.76, 0.76, 0.76),
    (-41.0, 0.40, 0.40, 0.40),   # DG   dark gray    (-30..-41)
    (-30.0, 0.40, 0.40, 0.40),
    # Warm tail matched to the cyclonicwx BD scale (low-cloud "pop"): +9 is the
    # lightest of the warm ramp, warm sea/land is dark gray, and it darkens
    # gradually from +9 down to -30 (into the DG step). Non-monotonic on
    # purpose so low/mid cloud reads as a brighter medium gray than the surface.
    (-30.0, 0.50, 0.50, 0.50),   # medium gray, darkening toward -30
    (9.0,   0.64, 0.64, 0.64),   # +9 — lightest warm gray (low-cloud pop)
    (40.0,  0.16, 0.16, 0.16),   # warm sea/land — dark gray
]

# ---------------------------------------------------------------------------
# tat_neon — refined Triple-A-Tropics signature
# ---------------------------------------------------------------------------
_TAT_NEON_ANCHORS = [
    (-90.0,  1.00, 1.00, 1.00),  # white-hot overshoot core
    (-82.0,  1.00, 0.86, 0.42),  # gold
    (-74.0,  1.00, 0.42, 0.58),  # hot pink
    (-66.0,  0.96, 0.22, 0.86),  # magenta
    (-56.0,  0.55, 0.32, 0.96),  # violet
    (-44.0,  0.22, 0.86, 0.88),  # cyan
    (-32.0,  0.12, 0.58, 0.78),  # teal
    (-18.0,  0.12, 0.30, 0.52),  # blue
    (0.0,    0.08, 0.16, 0.30),  # cool navy
    (30.0,   0.03, 0.05, 0.09),  # warm -> near black
]

# ---------------------------------------------------------------------------
# wv_tat — dedicated water-vapor enhancement
# ---------------------------------------------------------------------------
# Dry/sinking air (warm BT) = amber/brown; moist/rising (cold BT) = teal -> blue
# -> white. Domain tuned for WV channels: -85 .. +20 C.
_WV_TAT_ANCHORS = [
    (-85.0, 1.00, 1.00, 1.00),  # coldest / highest moisture -> white
    (-75.0, 0.72, 0.90, 1.00),  # pale blue
    (-62.0, 0.20, 0.55, 0.92),  # blue
    (-50.0, 0.10, 0.70, 0.78),  # teal
    (-40.0, 0.20, 0.62, 0.45),  # green-teal
    (-30.0, 0.45, 0.55, 0.40),  # transition
    (-20.0, 0.70, 0.58, 0.30),  # tan
    (-8.0,  0.85, 0.58, 0.20),  # amber
    (6.0,   0.62, 0.36, 0.12),  # brown (dry)
    (20.0,  0.34, 0.18, 0.06),  # dark brown (driest / warmest)
]

# standard IR grayscale: cold = white, warm = black.
_IR_GRAY_ANCHORS = [
    (-90.0, 1.00, 1.00, 1.00),
    (30.0,  0.00, 0.00, 0.00),
]

RAINBOW_IR_CMAP = _seg_cmap("rainbow_ir", _RAINBOW_IR_ANCHORS)
DVORAK_BD_CMAP = _seg_cmap("dvorak_bd", _DVORAK_BD_ANCHORS)
TAT_NEON_CMAP = _seg_cmap("tat_neon", _TAT_NEON_ANCHORS)
WV_TAT_CMAP = _seg_cmap("wv_tat", _WV_TAT_ANCHORS)
IR_GRAY_CMAP = _seg_cmap("ir_gray", _IR_GRAY_ANCHORS)


# ---------------------------------------------------------------------------
# colorbar tick sets (°C)
# ---------------------------------------------------------------------------
_RAINBOW_TICKS = [40, 30, 20, 10, 0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
_IR_TICKS = [30, 20, 10, 0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
# Dvorak BD: label the exact diagnostic breakpoints + a couple of warm refs.
_BD_TICKS = [30, 9, 0, -30, -41, -53, -63, -69, -75, -80, -85]
_WV_TICKS = [20, 10, 0, -10, -20, -30, -40, -50, -60, -70, -80]

_CBAR_LABEL_BT = "Brightness Temperature (°C)"


# ---------------------------------------------------------------------------
# registry (insertion order = UI order; rainbow_ir first == default)
# ---------------------------------------------------------------------------
# domain: which channel kinds the preset is offered for in the UI.
#   "ir" -> longwave/clean/shortwave IR ;  "wv" -> water vapor channels
ENHANCEMENTS = {
    "rainbow_ir": {
        "label": "Rainbow IR", "cmap": RAINBOW_IR_CMAP,
        "vmin_c": -95.0, "vmax_c": 40.0, "ticks": _RAINBOW_TICKS,
        "domains": ("ir",), "kind": "ir", "cbar_label": _CBAR_LABEL_BT,
    },
    "dvorak_bd": {
        "label": "Dvorak BD", "cmap": DVORAK_BD_CMAP,
        "vmin_c": -95.0, "vmax_c": 40.0, "ticks": _BD_TICKS,
        "domains": ("ir",), "kind": "ir", "cbar_label": _CBAR_LABEL_BT,
    },
    "tat_neon": {
        "label": "TAT Neon", "cmap": TAT_NEON_CMAP,
        "vmin_c": -90.0, "vmax_c": 30.0, "ticks": _IR_TICKS,
        "domains": ("ir",), "kind": "ir", "cbar_label": _CBAR_LABEL_BT,
    },
    "wv_tat": {
        "label": "Water Vapor", "cmap": WV_TAT_CMAP,
        "vmin_c": -85.0, "vmax_c": 20.0, "ticks": _WV_TICKS,
        "domains": ("wv",), "kind": "ir", "cbar_label": _CBAR_LABEL_BT,
    },
    "ir_gray": {
        "label": "Grayscale IR", "cmap": IR_GRAY_CMAP,
        "vmin_c": -90.0, "vmax_c": 30.0, "ticks": _IR_TICKS,
        "domains": ("ir", "wv"), "kind": "gray", "cbar_label": _CBAR_LABEL_BT,
    },
}

# Back-compat alias: the old name "grayscale" maps to ir_gray. The floater
# poller and legacy share-links still use "grayscale".
ENHANCEMENTS["grayscale"] = ENHANCEMENTS["ir_gray"]

# Default preset (used by app.py + frontend).
DEFAULT_ENHANCEMENT = "rainbow_ir"


def get_enhancement(name: str):
    if name not in ENHANCEMENTS:
        raise ValueError(f"unknown enhancement {name!r}")
    return ENHANCEMENTS[name]


def enhancement_norm(name: str) -> Normalize:
    """Fresh Normalize over the preset's °C domain (per-call, not shared)."""
    e = get_enhancement(name)
    return Normalize(vmin=e["vmin_c"], vmax=e["vmax_c"])


def list_enhancements_for_domain(domain: str):
    """UI helper: enhancement keys valid for a channel domain ('ir'|'wv')."""
    return [k for k, e in ENHANCEMENTS.items()
            if k != "grayscale" and domain in e["domains"]]


def normalize_visible(reflectance: np.ndarray) -> np.ndarray:
    """Sqrt-gamma stretch for visible reflectance (0..1)."""
    x = np.clip(reflectance, 0.0, 1.0)
    return np.sqrt(x)
