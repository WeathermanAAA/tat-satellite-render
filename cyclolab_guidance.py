#!/usr/bin/env python3
"""
cyclolab_guidance.py
--------------------
PURE parsers (no network, no R2) for the CycloLab guidance DATA layer:

  * parse_adeck(text)  -> guidance dict (model TRACKS + INTENSITY, one source for
    both) from the NHC public a-deck (aid_public/a{basin}{nn}{yyyy}.dat).
  * parse_ships(text)  -> ships dict (env diagnostics + RI + AHI) from a SHIPS
    _ships.txt.

Both are deliberately dependency-free + side-effect-free so they unit-test offline
against captured live bytes. The poller (cyclolab_guidance_poller.py) does the
fetch / decode / R2 write; it imports these.

Verified against live invest files (EP932026, 2026-06-15 18Z).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# --- curated aid sets (NOT the 70+ firehose). EARLY 'I' interpolated aids, time-
# aligned to the current init; consensus aids flagged. Detected at runtime - a code
# absent from a given a-deck (legacy HWRF/HMON dropped, a fresh invest carrying only
# statistical aids) is simply not in present_aids. ----------------------------------
TRACK_AIDS = ["AVNI", "HFAI", "HFBI", "HWFI", "HMNI", "CMCI", "NVGI", "EGRI", "CTCI", "TVCN", "HCCA"]
INTENSITY_AIDS = ["AVNI", "HFAI", "HFBI", "HWFI", "HMNI", "DSHP", "LGEM", "SHIP", "IVCN"]
CONSENSUS_AIDS = {"TVCN", "HCCA", "IVCN"}
CURATED_AIDS = list(dict.fromkeys(TRACK_AIDS + INTENSITY_AIDS))   # union, order preserved


# ===========================================================================
# A-DECK
# ===========================================================================
def _latlon(tok: str) -> Optional[float]:
    """ATCF lat/lon: tenths of a degree + hemisphere letter. '90N' -> 9.0,
    '1380W' -> -138.0 (S/W negative). None if malformed/empty."""
    tok = tok.strip().upper()
    if len(tok) < 2 or tok[-1] not in "NSEW":
        return None
    try:
        v = int(tok[:-1]) / 10.0
    except ValueError:
        return None
    return round(-v if tok[-1] in "SW" else v, 1)


def _int0(tok: str) -> Optional[int]:
    """ATCF int field; 0 (or blank/non-numeric) = MISSING -> None."""
    try:
        v = int(tok)
    except ValueError:
        return None
    return v or None


def init_to_iso(cycle: str) -> Optional[str]:
    """'2026061518' -> '2026-06-15T18:00:00Z'."""
    if not (cycle and len(cycle) == 10 and cycle.isdigit()):
        return None
    return f"{cycle[:4]}-{cycle[4:6]}-{cycle[6:8]}T{cycle[8:10]}:00:00Z"


def parse_adeck(text: str) -> dict:
    """Parse the public a-deck into model guidance for the CURRENT synoptic init.

    Keeps only the curated aids, time-aligned to the LATEST init present (early aids
    land first; re-polling picks up the rest). DEDUPES on (TECH, TAU) - the 34/50/64
    wind-radii rows repeat each fix and would otherwise triple-count points (the
    first row, RAD=34, is kept). vmax/mslp 0 = missing -> null."""
    rows: List[List[str]] = []
    for ln in text.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 11 and f[2].isdigit() and len(f[2]) == 10:
            rows.append(f)
    inits = [r[2] for r in rows]
    if not inits:
        return {"init_time": None, "init_cycle": None, "aids": {}, "consensus": [],
                "present_aids": [], "track_aids": [], "intensity_aids": []}
    latest = max(inits)
    aids: Dict[str, List[dict]] = {}
    seen: set = set()
    for r in rows:
        if r[2] != latest:
            continue
        tech = r[4]
        if tech not in CURATED_AIDS:
            continue
        try:
            tau = int(r[5])
        except ValueError:
            continue
        if tau < 0:                      # forecast only; CARQ's -24..0 analysis is not an aid
            continue
        key = (tech, tau)
        if key in seen:                  # wind-radii duplicate (RAD 50/64) -> skip
            continue
        seen.add(key)
        aids.setdefault(tech, []).append({
            "tau": tau, "lat": _latlon(r[6]), "lon": _latlon(r[7]),
            "vmax": _int0(r[8]), "mslp": _int0(r[9]),
        })
    for pts in aids.values():
        pts.sort(key=lambda p: p["tau"])
    present = [t for t in CURATED_AIDS if t in aids]
    return {
        "init_time": init_to_iso(latest),
        "init_cycle": latest,
        "aids": aids,
        "present_aids": present,
        "track_aids": [t for t in TRACK_AIDS if t in aids],
        "intensity_aids": [t for t in INTENSITY_AIDS if t in aids],
        "consensus": [t for t in present if t in CONSENSUS_AIDS],
    }


# ===========================================================================
# SHIPS  (_ships.txt: header, env time series, RI section, AHI trailer)
# ===========================================================================
_SENTINELS = {"XX.X", "XXX.X", "N/A", "NA", "LOST", "999.0", "999.0%", "9999.0"}


def _num(tok: str) -> Optional[float]:
    """SHIPS numeric token -> float, or None for a sentinel / non-numeric
    (xx.x, xxx.x, N/A, LOST, 999.0, the '0.'/'-0.' trailing-dot contributions)."""
    t = tok.strip().rstrip("%")
    if not t or t.upper() in _SENTINELS:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_ships(text: str) -> dict:
    """Parse a SHIPS _ships.txt into header + env_series (by tau) + RI section + AHI."""
    lines = text.splitlines()

    # --- header (the 4 starred lines) ---
    starred = [l.strip().strip("*").strip() for l in lines if l.strip().startswith("*")]
    header = {"raw": starred[:4]}
    if len(starred) >= 1:
        header["model"] = starred[0]
    if len(starred) >= 2:
        header["basin_year"] = starred[1]
    if len(starred) >= 3:
        header["data_flags"] = starred[2]
    if len(starred) >= 4:
        header["id_line"] = starred[3]
        m = re.search(r"\b([A-Z]{2}\d{6})\b\s+(\d\d/\d\d/\d\d)\s+(\d\d)\s*UTC", starred[3])
        if m:
            header["atcf_id"], header["date"], header["hour_utc"] = m.group(1), m.group(2), m.group(3)

    # --- env diagnostic time series (TIME (HR) line .. blank) ---
    taus: List[int] = []
    env_series: Dict[str, List[Optional[float]]] = {}
    storm_type: List[str] = []
    ti = next((i for i, l in enumerate(lines) if l.strip().startswith("TIME (HR)")), None)
    if ti is not None:
        taus = [int(x) for x in lines[ti].split()[2:] if x.lstrip("-").isdigit()]
        n = len(taus)
        for l in lines[ti + 1:]:
            s = l.strip()
            # the env block ends at the FORECAST TRACK / contributions section; a
            # BLANK line inside it (between the V-group and the SHEAR-group) is NOT
            # the end, so skip blanks rather than break on them.
            if s.startswith("FORECAST TRACK") or s.startswith("INDIVIDUAL CONTRIB"):
                break
            if not s:
                continue
            toks = l.split()
            if len(toks) <= n:
                continue
            label = " ".join(toks[:-n]).strip()
            vals = toks[-n:]
            if label.upper().startswith("STORM TYPE"):
                storm_type = vals
            else:
                env_series[label] = [_num(v) for v in vals]

    out: dict = {
        "available": True,
        "header": header,
        "taus": taus,
        "env_series": env_series,
        "storm_type": storm_type,
    }

    # --- RI section ---
    full = "\n".join(lines)
    m = re.search(r"PRELIM RI PROB[^:]*:\s*([\d.]+)", full)
    out["prelim_ri_prob"] = _num(m.group(1)) if m else None

    # SHIPS-RII predictor table: "<name> : value lo to hi scaled contribution"
    preds = []
    for l in lines:
        m = re.match(r"\s*(.+?)\s*:\s*(-?[\d.]+)\s+(-?[\d.]+)\s+to\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$", l)
        if m:
            preds.append({
                "predictor": m.group(1).strip(),
                "value": _num(m.group(2)), "range_lo": _num(m.group(3)),
                "range_hi": _num(m.group(4)), "scaled": _num(m.group(5)),
                "contribution": _num(m.group(6)),
            })
    out["ri_predictor_table"] = preds

    # threshold probabilities: "SHIPS Prob RI for 25kt/ 24hr ... = 13% is 1.0 times ... ( 12.5%)"
    thresh = []
    for l in lines:
        m = re.search(r"Prob RI for\s*(\d+kt)/\s*(\d+hr).*?=\s*(\d+)%\s*is\s*([\d.]+)\s*times.*?\(\s*([\d.]+)%\)", l)
        if m:
            thresh.append({"threshold": f"{m.group(1)}/{m.group(2)}", "prob_pct": _num(m.group(3)),
                           "ratio": _num(m.group(4)), "clim_pct": _num(m.group(5))})
    out["ri_threshold_probs"] = thresh

    # RI PROBABILITY MATRIX: header "RI (kt / h) | 20/12 | ..." then "<row>: % % ..."
    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    cols: List[str] = []
    mi = next((i for i, l in enumerate(lines) if "RI (kt" in l and "|" in l), None)
    if mi is not None:
        cols = [c.strip() for c in lines[mi].split("|")[1:] if c.strip()]
        for l in lines[mi + 1:]:
            if l.strip().startswith("---") or not l.strip():
                continue
            m = re.match(r"\s*([A-Za-z][\w/-]*)\s*:\s*(.+)$", l)
            if not m:
                if matrix:
                    break        # past the matrix block
                continue
            vals = [_num(v) for v in m.group(2).split()]
            if cols and len(vals) >= len(cols):
                matrix[m.group(1)] = dict(zip(cols, vals[:len(cols)]))
    out["ri_matrix"] = {"cols": cols, "rows": matrix}

    # --- AHI trailer: "AHI= N" value + the screening verdict line ---
    ahi_val = None
    verdict = None
    for i, l in enumerate(lines):
        m = re.search(r"AHI=\s*(-?\d+)", l)
        if m:
            ahi_val = int(m.group(1))
        if "ANNULAR" in l and "INDEX" in l:
            # the verdict is the NEXT '##'-bracketed line (NOT the AHI= line)
            for nxt in lines[i + 1:i + 4]:
                inner = nxt.strip().strip("#").strip()
                if inner and "AHI=" not in nxt:
                    verdict = inner
                    break
    out["ahi"] = {"value": ahi_val, "verdict": verdict}
    return out
