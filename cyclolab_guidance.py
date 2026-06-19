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

# --- WP (JTWC) curated aids -------------------------------------------------
# The NHC operational aid package (OFCL/GFS/ECMWF/HWRF/HMON/TVCN/SHIP) is NOT
# redistributed for the West Pacific; the only clean PUBLIC WPAC a-deck is the
# DTC ``adecks_open`` mirror, which carries the global ENSEMBLE-MEAN tracks +
# the deterministic global models. So WPAC guidance curates the named ensemble
# means + deterministic globals (clean, interpretable tracks) and deliberately
# DROPS the 30-member GEFS/CMC/NAVGEM spaghetti (AP##/CP##/NP##) - restrained,
# scientific, not a firework. Whatever of these is present in the a-deck is
# kept; a fresh TD that only the GEFS has touched shows just AEMN+AC00.
#   AEMN/CEMN/NEMN  GEFS / CMC / NAVGEM ensemble MEANS  (consensus-like)
#   CMC / UKM / NGX deterministic CMC / UKMET / NAVGEM-GX
#   AC00            GEFS control (a single clean track)
WP_TRACK_AIDS = ["AEMN", "CEMN", "NEMN", "CMC", "UKM", "NGX", "AC00"]
WP_INTENSITY_AIDS = ["AEMN", "CEMN", "NEMN", "CMC", "UKM", "NGX"]
WP_CONSENSUS_AIDS = {"AEMN", "CEMN", "NEMN"}   # the ensemble means stand in for TVCN/HCCA


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


# ===========================================================================
# NHC Tropical Weather Outlook (TWO) - invest FORMATION chances
# ===========================================================================
_TWO_REF = re.compile(r"\(([A-Z]{2})(9\d)\)")          # invest ref e.g. (EP93)
_TWO_FC = re.compile(
    r"Formation chance through\s+(\d+)\s*(hour|day)s?\b[.\s]*([A-Za-z]+)[.\s]*?(\d+)\s*percent",
    re.IGNORECASE)

# --- Active-Systems PTC narrative ------------------------------------------
# Once NHC begins issuing advisories on a system, it leaves the numbered-invest
# list ((AL90) disappears) and moves into the TWO's "Active Systems" narrative
# - which still carries a FORMATION chance for a Potential Tropical Cyclone (it
# is "potential", not yet a TC). _TWO_REF can no longer see it, so the genesis
# pill would FREEZE at the last invest-era odds. We read those too, keyed to the
# REAL designated sid (e.g. NHC_AL012026), derived AUTHORITATIVELY from the TCP
# AWIPS header (MIATCPAT1 -> basin AT->AL, storm 1). The storm NAME string is the
# link between the narrative block (which carries the chances) and the AWIPS
# header line (which carries the sid), so multiple simultaneous PTCs stay
# separated without a spelled-ordinal table.
_AWIPS_BASIN = {"AT": "AL", "EP": "EP", "CP": "CP"}    # AWIPS basin -> feed basin
# "Public Advisories on <NAME> are issued ... AWIPS header MIATCPAT1". MUST be
# anchored on "Public Advisories" (the TCP product line) - the bare "advisories
# on" would otherwise also match the narrative's "issuing advisories on" and the
# "Forecast/Advisories on" line (whose MIATCM header is not a TCP anyway).
_PTC_AWIPS = re.compile(
    r"Public\s+Advisories\s+on\s+(.+?)\s+(?:are|is)\s+issued.*?"
    r"AWIPS\s+header\s+\w{3}TCP([A-Z]{2})(\d{1,2})", re.I)
# "...is issuing advisories on <NAME>, located ..." (the narrative block head)
_PTC_NARR = re.compile(r"issuing\s+advisories\s+on\s+(.+?)\s*(?:,|\.|\bloc)", re.I)
# A numbered disturbance area ("1. Eastern...") or an invest ref ("(AL91)") marks
# the end of the Active-Systems narrative - a PTC's block must NOT bleed into a
# trailing disturbance's formation chances (which would overwrite the PTC's own
# odds, and would attach a spurious chance to a named storm that has none).
_AREA_BREAK = re.compile(r"(?:(?<=\s)\d{1,2}\.\s+[A-Z]|\([A-Z]{2}\d\d\))")


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _parse_active_ptc(body: str, year: int) -> Dict[str, dict]:
    """The Active-Systems PTC formation chances, keyed to the REAL designated
    sid. ``{}`` when the outlook has no Active-Systems PTC carrying a chance
    (the common case). Named storms in the narrative carry no formation chance,
    so they naturally yield no entry."""
    flat = re.sub(r"\s+", " ", body)
    # NAME -> (feed_basin, storm_number) from the TCP AWIPS header lines.
    awips: Dict[str, tuple] = {}
    for m in _PTC_AWIPS.finditer(flat):
        bcode = m.group(2).upper()
        if bcode in _AWIPS_BASIN:
            awips[_norm_name(m.group(1))] = (_AWIPS_BASIN[bcode], int(m.group(3)))
    if not awips:
        return {}
    out: Dict[str, dict] = {}
    narr = list(_PTC_NARR.finditer(flat))
    amp = flat.find("&&")                       # the narrative ends at "&&"
    narr_end = amp if amp != -1 else len(flat)
    for i, m in enumerate(narr):
        if m.start() >= narr_end:               # past the narrative (AWIPS block)
            continue
        name = _norm_name(m.group(1))
        if name not in awips:                   # named storm / unmatched -> skip
            continue
        start = m.end()
        end = narr[i + 1].start() if i + 1 < len(narr) else narr_end
        end = min(end, narr_end)
        # Cap the block at the first numbered-disturbance / invest-ref marker so
        # this system's chances never absorb a trailing disturbance's odds.
        brk = _AREA_BREAK.search(flat, start, end)
        if brk:
            end = brk.start()
        block = flat[start:end]
        p48 = p7 = None
        for fm in _TWO_FC.finditer(block):
            unit, pct = fm.group(2).lower(), int(fm.group(4))
            if unit.startswith("hour"):
                p48 = pct
            elif unit.startswith("day"):
                p7 = pct
        if p48 is None and p7 is None:          # named storm: no chance -> skip
            continue
        basin, num = awips[name]
        sid = f"NHC_{basin}{num:02d}{year}"
        out[sid] = {
            "sid": sid, "p48": p48, "p7": p7,
            "level48": formation_level(p48), "level7": formation_level(p7),
            "level": formation_level(max(x for x in (p48, p7) if x is not None)),
            "area": re.sub(r"\s+", " ", m.group(1)).strip() or None,
        }
    return out


def _two_body(xml_text: str) -> str:
    """The outlook text: the RSS CDATA with <br/> -> newlines (or the raw text
    if already de-XML'd)."""
    m = re.search(r"<!\[CDATA\[(.*?)\]\]>", xml_text, re.S)
    body = m.group(1) if m else xml_text
    return (body.replace("<br />", "\n").replace("<br/>", "\n")
                .replace("<br>", "\n"))


def formation_level(pct: Optional[int]) -> Optional[str]:
    """NHC genesis category from a formation percent - the canonical low / medium
    / high colour buckets: <= 30 'low' (yellow), 40-60 'medium' (orange), >= 70
    'high' (red). None when there is no chance."""
    if pct is None:
        return None
    if pct <= 30:
        return "low"
    if pct <= 60:
        return "medium"
    return "high"


def parse_two(xml_text: str, year: int) -> Dict[str, dict]:
    """Parse an NHC Tropical Weather Outlook (TWO{AT|EP|CP}) into per-INVEST
    formation chances, keyed by tracks-feed sid (``NHC_{BASIN}{NN}{YYYY}``).

    Each outlook 'area' that references an invest number - '(EP93)', '(AL90)' -
    yields its 48-hour and 7-day formation percentages + NHC level. Areas with no
    invest number (pre-genesis) are skipped (no sid to attach). Returns {} for an
    outlook with no numbered areas (the common quiet case)."""
    body = _two_body(xml_text)
    lines = body.splitlines()
    refs = []
    for li, ln in enumerate(lines):
        m = _TWO_REF.search(ln)
        if m:
            refs.append((li, m.group(1).upper(), m.group(2)))
    out: Dict[str, dict] = {}
    for k, (li, basin, nn) in enumerate(refs):
        end_li = refs[k + 1][0] if k + 1 < len(refs) else len(lines)
        block = "\n".join(lines[li:end_li])
        p48 = p7 = None
        for fm in _TWO_FC.finditer(block):
            unit, pct = fm.group(2).lower(), int(fm.group(4))
            if unit.startswith("hour"):
                p48 = pct
            elif unit.startswith("day"):
                p7 = pct
        if p48 is None and p7 is None:
            continue
        area = re.sub(r"\s*\([A-Z]{2}9\d\):?.*$", "", lines[li]).strip()
        sid = f"NHC_{basin}{nn}{year}"
        out[sid] = {
            "sid": sid, "p48": p48, "p7": p7,
            "level48": formation_level(p48), "level7": formation_level(p7),
            "level": formation_level(max(x for x in (p48, p7) if x is not None)),
            "area": area or None,
        }
    # Active-Systems PTC narrative -> REAL designated sid. A numbered ref (if one
    # somehow co-exists) is more specific and wins; otherwise the PTC entry adds.
    for sid, fc in _parse_active_ptc(body, year).items():
        out.setdefault(sid, fc)
    return out


def parse_adeck(text: str, basin: Optional[str] = None) -> dict:
    """Parse the public a-deck into model guidance for the CURRENT synoptic init.

    Keeps only the curated aids, time-aligned to the LATEST init present (early aids
    land first; re-polling picks up the rest). DEDUPES on (TECH, TAU) - the 34/50/64
    wind-radii rows repeat each fix and would otherwise triple-count points (the
    first row, RAD=34, is kept). vmax/mslp 0 = missing -> null.

    ``basin`` selects the curated aid set. ``WP`` (JTWC) uses the global-model /
    ensemble-mean set (WP_*), since the NHC operational aids never appear in the
    public WPAC a-deck; every other value (incl. None) keeps the NHC sets EXACTLY
    as before (the NHC path is byte-identical). For WP the latest init is the
    newest one that actually carries a curated MODEL aid - the DTC mirror posts a
    CARQ-only analysis record at the freshest synoptic while the ensemble lands a
    cycle behind, so ``max(init)`` alone would yield an empty (statistical-only)
    plot for a storm that in fact has guidance one cycle back."""
    is_wp = (basin or "").upper() == "WP"
    track_set = WP_TRACK_AIDS if is_wp else TRACK_AIDS
    intensity_set = WP_INTENSITY_AIDS if is_wp else INTENSITY_AIDS
    consensus_set = WP_CONSENSUS_AIDS if is_wp else CONSENSUS_AIDS
    curated = list(dict.fromkeys(track_set + intensity_set))   # union, order preserved

    rows: List[List[str]] = []
    for ln in text.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 11 and f[2].isdigit() and len(f[2]) == 10:
            rows.append(f)
    inits = [r[2] for r in rows]
    if not inits:
        return {"init_time": None, "init_cycle": None, "aids": {}, "consensus": [],
                "present_aids": [], "track_aids": [], "intensity_aids": []}
    if is_wp:
        # newest init that carries a forecast (tau>=0) curated aid, else newest
        # init at all (so init_cycle/init_time stay set -> graceful empty plot).
        aid_inits = [r[2] for r in rows if r[4] in curated
                     and r[5].lstrip("-").isdigit() and int(r[5]) >= 0]
        latest = max(aid_inits) if aid_inits else max(inits)
    else:
        latest = max(inits)
    aids: Dict[str, List[dict]] = {}
    seen: set = set()
    for r in rows:
        if r[2] != latest:
            continue
        tech = r[4]
        if tech not in curated:
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
    present = [t for t in curated if t in aids]
    return {
        "init_time": init_to_iso(latest),
        "init_cycle": latest,
        "aids": aids,
        "present_aids": present,
        "track_aids": [t for t in track_set if t in aids],
        "intensity_aids": [t for t in intensity_set if t in aids],
        "consensus": [t for t in present if t in consensus_set],
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
