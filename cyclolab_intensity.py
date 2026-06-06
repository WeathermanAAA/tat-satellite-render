"""CycloLab INTENSITY CONE data layer (CYCLOLAB_DESIGN §8.6).

The per-basin published-intensity-error REGISTRY (versioned JSON blob,
cyclolab_intensity_errors.json - every number text-verified against the
downloaded source document; the never-invent contract) + the envelope
math shared by the in-app SVG's server-side mirror and the OG card
renderer.

HONESTY GUARD: a basin with no registry entry yields None - the caller
renders a labeled "no published statistics" panel, never a borrowed or
invented envelope. Missing taus interpolate linearly between published
neighbors (e.g. CP/WP have no 60 h column) and the disclosure says so.
Envelope construction is symmetric +/-MAE; the registry carries the
signed bias for AL/EP should the asymmetric variant ever be built, and
the disclosure would document it.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "cyclolab_intensity_errors.json"

_REGISTRY = None

KT_MIN, KT_MAX = 0.0, 200.0


def load_registry() -> dict:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return _REGISTRY


def basin_entry(basin: str) -> dict | None:
    """The registry entry for a CycloLab basin code (AL/EP/CP/WP), or
    None - the honesty-guard case the caller must label, never paper
    over."""
    return load_registry()["basins"].get((basin or "").upper())


def mae_at(entry: dict, tau_h: float) -> float:
    """MAE (kt) at lead time tau_h: published value when present, linear
    interpolation between published neighbors otherwise; tau 0 anchors
    at 0 (the analysis intensity is the known starting point); beyond
    the last published tau, clamp to the last value."""
    table = {float(k): float(v) for k, v in entry["mae_kt"].items()}
    taus = sorted(table)
    if tau_h <= 0:
        return 0.0
    if tau_h in table:
        return table[tau_h]
    lo, lo_v = 0.0, 0.0
    for t in taus:
        if t > tau_h:
            hi, hi_v = t, table[t]
            return lo_v + (hi_v - lo_v) * (tau_h - lo) / (hi - lo)
        lo, lo_v = t, table[t]
    return table[taus[-1]]


def envelope(points: list[dict], entry: dict) -> list[dict]:
    """Per-forecast-point envelope rows: tau_h, center (forecast VMAX),
    upper/lower = center +/- MAE(tau), clamped to [0, 200] kt. Points
    without an intensity are skipped (they cannot anchor a band)."""
    rows = []
    for p in points:
        kt = p.get("intensity_kt")
        if kt is None:
            continue
        tau = float(p.get("tau_h") or 0)
        m = mae_at(entry, tau)
        rows.append({
            "tau_h": tau,
            "center": float(kt),
            "upper": min(KT_MAX, float(kt) + m),
            "lower": max(KT_MIN, float(kt) - m),
            "mae": m,
        })
    return rows
