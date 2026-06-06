"""CycloLab intensity OG/share card (CYCLOLAB_DESIGN §8.6, second
placement): a server-rendered, TAT-branded PNG of the intensity cone -
the same registry + envelope math as the in-app SVG
(cyclolab_intensity.envelope, parity-pinned in tests), drawn with
matplotlib Agg at the 1200x630 OG aspect.

HONESTY GUARD: callers only invoke this with a real registry entry -
a basin without published statistics gets NO card (and no og:image
tag), never a borrowed envelope. The disclosure line is baked into the
image itself so the card can never travel without it.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ace_core import SSHS_COLORS  # noqa: E402
from cyclolab_intensity import envelope  # noqa: E402

BG = "#0b0e13"
PANEL = "#0a1019"
MUTED = "#8b95a5"
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "C1"), (83, 96, "C2"),
         (96, 113, "C3"), (113, 137, "C4"), (137, 999, "C5")]


def _cat_for_kt(kt: float) -> str:
    for lo, hi, cat in BANDS:
        if lo <= kt < hi:
            return cat
    return "C5"


def render_intensity_card(adv: dict, entry: dict, *,
                          storm_name: str) -> bytes:
    """The branded share card. Returns PNG bytes; raises on missing
    forecast intensity (callers treat any raise as 'no card')."""
    rows = envelope(adv.get("points") or [], entry)
    if len(rows) < 2:
        raise ValueError("not enough forecast intensity points for a card")
    taus = [r["tau_h"] for r in rows]
    kmax = max(80.0, max(r["upper"] for r in rows) + 12)

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.075, 0.16, 0.86, 0.66])
    ax.set_facecolor(PANEL)

    for lo, hi, cat in BANDS:
        if lo >= kmax:
            break
        ax.axhspan(lo, min(hi, kmax), color=SSHS_COLORS[cat], alpha=0.10,
                   lw=0)
        if hi <= kmax:
            ax.text(taus[-1] * 0.995, hi - (kmax * 0.012), cat,
                    color=SSHS_COLORS[cat], fontsize=9, ha="right",
                    va="top", alpha=0.85)

    ax.fill_between(taus, [r["lower"] for r in rows],
                    [r["upper"] for r in rows],
                    color="#ffffff", alpha=0.10, lw=0)
    ax.plot(taus, [r["upper"] for r in rows], color="#8cc8ff", lw=1.2,
            alpha=0.45)
    ax.plot(taus, [r["lower"] for r in rows], color="#8cc8ff", lw=1.2,
            alpha=0.45)
    ax.plot(taus, [r["center"] for r in rows], color="#ffffff", lw=2.6,
            solid_joinstyle="round")
    for r in rows:
        ax.plot([r["tau_h"]], [r["center"]], marker="o", ms=9,
                color=SSHS_COLORS[_cat_for_kt(r["center"])],
                markeredgecolor="#0b0e13", markeredgewidth=1.2)

    ax.set_xlim(0, taus[-1])
    ax.set_ylim(0, kmax)
    ax.set_xticks(taus)
    ax.set_xticklabels([f"+{int(t)}h" for t in taus], color=MUTED,
                       fontsize=9)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.set_ylabel("kt", color=MUTED, fontsize=10)
    for s in ax.spines.values():
        s.set_color("#232a36")

    fig.text(0.075, 0.92, storm_name.upper(), color="#ffffff",
             fontsize=22, fontweight="bold")
    fig.text(0.075, 0.875,
             f"Intensity forecast · advisory {adv.get('advisory')} "
             f"· TAT CycloLab", color=MUTED, fontsize=11)
    fig.text(0.075, 0.045,
             "Derived intensity range — not an official forecast "
             f"product · ±{entry.get('error_type', 'MAE')} "
             f"{entry.get('window', '')} ({entry['method_version']}) "
             "· triple-a-tropics.com",
             color=MUTED, fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()
