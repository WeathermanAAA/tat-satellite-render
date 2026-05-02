"""Three colormaps for GOES ABI rendering.

- dvorak_bd: canonical Dvorak BD enhancement curve, breakpoints in Kelvin.
  Standard NHC/AOML scheme: warm grays, then cyan/yellow/red bands at storm tops.
- tat_neon: Triple-A-Tropics neon palette — cyan -> magenta -> hot pink -> white,
  designed to read crisply on dark backgrounds.
- grayscale: matplotlib gray for IR. Visible (Ch02) gets a sqrt-gamma stretch
  applied at render time, not via the cmap itself.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# ---------------------------------------------------------------------------
# Dvorak BD curve
# ---------------------------------------------------------------------------
# Anchor temperatures (K) and corresponding RGB. Below MG (~-30C / 243K) the
# scheme is just gray. Above (colder), discrete bands kick in.
# Reference values from NOAA AOML hurricane-tools BD curve:
#   warm threshold:    +30 C  (303 K) -> black/gray
#   MG threshold:      -30 C  (243 K) -> medium gray
#   CMG threshold:     -42 C  (231 K) -> dark gray
#   DG threshold:      -54 C  (219 K) -> near-black
#   LG threshold:      -64 C  (209 K) -> black/white split
#   M threshold:       -70 C  (203 K) -> white
#   B threshold:       -76 C  (197 K) -> bright bands
#   W threshold:       -80 C  (193 K) -> bright bands
#   CDG threshold:     -84 C  (189 K) -> bright bands
#   coldest:           -90 C  (183 K) -> very cold

def _dvorak_bd_segments():
    # (temp_K, R, G, B) anchors, warm -> cold
    anchors = [
        (310.0, 0.05, 0.05, 0.05),
        (303.0, 0.10, 0.10, 0.10),
        (273.0, 0.50, 0.50, 0.50),
        (243.0, 0.70, 0.70, 0.70),
        (231.0, 0.45, 0.45, 0.45),
        (219.0, 0.20, 0.20, 0.20),
        (209.5, 0.00, 0.00, 0.00),
        (209.0, 1.00, 1.00, 1.00),  # LG -> M switch (white)
        (203.0, 1.00, 1.00, 0.00),  # M  -> B (yellow)
        (197.0, 1.00, 0.40, 0.00),  # B  -> W (orange)
        (193.0, 0.90, 0.00, 0.00),  # W  -> CDG (red)
        (189.0, 1.00, 0.00, 1.00),  # CDG (magenta)
        (183.0, 0.20, 0.00, 0.40),  # very cold
    ]
    return anchors


def _build_dvorak_bd():
    anchors = _dvorak_bd_segments()
    # Map temps to 0..1 over the rendering range we'll normalize against
    # (warmest -> x=0, coldest -> x=1)
    t_warm, t_cold = anchors[0][0], anchors[-1][0]
    span = t_warm - t_cold

    def x_of(t):
        return (t_warm - t) / span

    rs, gs, bs = [], [], []
    for t, r, g, b in anchors:
        x = x_of(t)
        rs.append((x, r, r))
        gs.append((x, g, g))
        bs.append((x, b, b))
    cdict = {"red": rs, "green": gs, "blue": bs}
    return LinearSegmentedColormap("dvorak_bd", cdict, N=512)


DVORAK_BD = _build_dvorak_bd()
DVORAK_BD_RANGE_K = (310.0, 183.0)  # (warm, cold) — what 0..1 maps to


# ---------------------------------------------------------------------------
# TAT Neon — site palette extension for IR
# ---------------------------------------------------------------------------
# Anchors picked to read on the existing dark background and reuse the
# cyan/magenta/hot-pink accent feel of triple-a-tropics.com.
def _build_tat_neon():
    anchors = [
        (303.0, 0.04, 0.06, 0.10),  # warm sea -> deep navy
        (273.0, 0.10, 0.18, 0.32),  # cool clouds -> blue
        (253.0, 0.10, 0.55, 0.75),  # mid clouds -> teal
        (233.0, 0.20, 0.85, 0.85),  # cyan
        (218.0, 0.50, 0.30, 0.95),  # violet
        (208.0, 0.95, 0.20, 0.85),  # magenta
        (200.0, 1.00, 0.40, 0.55),  # hot pink
        (192.0, 1.00, 0.85, 0.40),  # gold
        (183.0, 1.00, 1.00, 1.00),  # white-hot core
    ]
    t_warm, t_cold = anchors[0][0], anchors[-1][0]
    span = t_warm - t_cold
    rs, gs, bs = [], [], []
    for t, r, g, b in anchors:
        x = (t_warm - t) / span
        rs.append((x, r, r))
        gs.append((x, g, g))
        bs.append((x, b, b))
    cdict = {"red": rs, "green": gs, "blue": bs}
    return LinearSegmentedColormap("tat_neon", cdict, N=512)


TAT_NEON = _build_tat_neon()
TAT_NEON_RANGE_K = (303.0, 183.0)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
ENHANCEMENTS = {
    "tat_neon": {"cmap": TAT_NEON, "range_k": TAT_NEON_RANGE_K, "kind": "ir"},
    "dvorak_bd": {"cmap": DVORAK_BD, "range_k": DVORAK_BD_RANGE_K, "kind": "ir"},
    "grayscale": {"cmap": None, "range_k": None, "kind": "gray"},
}


def get_enhancement(name: str):
    if name not in ENHANCEMENTS:
        raise ValueError(f"unknown enhancement {name!r}")
    return ENHANCEMENTS[name]


def normalize_ir(data_k: np.ndarray, range_k: tuple[float, float]) -> np.ndarray:
    """Map IR brightness temperatures (Kelvin) -> 0..1 with warm=0, cold=1."""
    t_warm, t_cold = range_k
    x = (t_warm - data_k) / (t_warm - t_cold)
    return np.clip(x, 0.0, 1.0)


def normalize_visible(reflectance: np.ndarray) -> np.ndarray:
    """Sqrt-gamma stretch for visible reflectance (0..1)."""
    x = np.clip(reflectance, 0.0, 1.0)
    return np.sqrt(x)
