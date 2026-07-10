"""JTWC tropical-cyclone warning-text parser (CYCLOLAB_DESIGN.md §8.4 sub-path).

JTWC publishes a per-storm TC Warning (``wp{NN}{YY}web.txt``) carrying a current
WARNING POSITION (tau 0) and a FORECASTS block (tau 12/24/36/48/60/72/96/120) --
the forecast track a WP derived cone is built from. There is no official cone and
no CurrentStorms entry; this text IS the product (knackwx is only the live
DESIGNATION feed). The companion Prognostic Reasoning (``...prog.txt``) is the
discussion/TCD equivalent and carries the SAME warning number for cross-product
verification.

``parse_jtwc_warning(text)`` ->
    {warning_number:int, issued_utc:iso, name:str, sid_hint:str,
     points:[{tau_h:int, valid_utc:iso, lat:float, lon:float, wind_kt:float|None}]}

Geometry conventions match derived_cone (which is antimeridian-safe): lat S ->
negative, lon W -> negative, signed -180..180. A FINAL WARNING (no FORECASTS)
yields points == [tau0] (derive_cone returns a full circle for a single point).
The warning number is read from THIS text (authoritative), never from knackwx.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Optional

UTC = dt.timezone.utc

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

# "WARNING NR 001" -- in web.txt SUBJ on one line; in prog.txt SUBJ it wraps
# ("...WARNING \nNR 001//"), so \s+ must span the newline.
_WARN_NR_RE = re.compile(r"WARNING\s+NR\s+0*(\d+)", re.I)
# "07W (SEVEN)" -- storm number+basin letter then the name in parens.
_NAME_RE = re.compile(r"\b(\d{1,2}[A-Z])\s*\(([^)]+)\)")
# Header issuance DDHHMM: "WTPN31 PGTW 182100".
_HEADER_DTG_RE = re.compile(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+(\d{6})\s*$", re.M)
# Month/year from a full DTG like "181351ZJUN2026".
_MONYEAR_RE = re.compile(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{4})")
# REMARKS short date "18JUN26" -> 2-digit year fallback.
_SHORTDATE_RE = re.compile(r"\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})\b")
# A position line: "181800Z --- NEAR 12.2N 145.1E"  (NEAR is optional).
_POS_RE = re.compile(
    r"(\d{2})(\d{2})(\d{2})Z\s*---\s*(?:NEAR\s+)?"
    r"(\d+(?:\.\d+)?)\s*([NS])\s+(\d+(?:\.\d+)?)\s*([EW])", re.I)
# A forecast header: "12 HRS, VALID AT:".
_FCHDR_RE = re.compile(r"(\d+)\s+HRS,\s+VALID\s+AT:", re.I)
_MAXWIND_RE = re.compile(r"MAX\s+SUSTAINED\s+WINDS\s*-\s*(\d+)\s*KT", re.I)


class JtwcParseError(Exception):
    """The warning text could not be parsed into a usable forecast."""


def _iso_z(t: dt.datetime) -> str:
    return t.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signed(val: str, hemi: str) -> float:
    v = float(val)
    return -v if hemi.upper() in ("S", "W") else v


def _month_year(text: str) -> tuple[int, int]:
    m = _MONYEAR_RE.search(text)
    if m:
        return _MONTHS[m.group(1).upper()], int(m.group(2))
    m = _SHORTDATE_RE.search(text)
    if m:
        return _MONTHS[m.group(1).upper()], 2000 + int(m.group(2))
    raise JtwcParseError("no month/year DTG found in warning text")


def _roll(dd: int, hh: int, mm: int, issued: dt.datetime) -> dt.datetime:
    """Combine a DDHHMM with the issuance month/year, rolling to the NEXT month
    (and year at Dec->Jan) when the day has wrapped past the issuance (forecast
    taus run up to +120 h, so a candidate >2 days BEFORE issuance wrapped)."""
    year, month = issued.year, issued.month
    for _ in range(2):
        try:
            cand = dt.datetime(year, month, dd, hh, mm, tzinfo=UTC)
        except ValueError:
            # e.g. day 31 in a 30-day month at a boundary -> try next month.
            month += 1
            if month > 12:
                month, year = 1, year + 1
            continue
        if cand < issued - dt.timedelta(days=2):
            month += 1
            if month > 12:
                month, year = 1, year + 1
            continue
        return cand
    return dt.datetime(year, month, dd, hh, mm, tzinfo=UTC)


def _wind_after(text: str, start: int) -> Optional[float]:
    m = _MAXWIND_RE.search(text, start, start + 400)
    return float(m.group(1)) if m else None


def parse_jtwc_warning(text: str) -> dict:
    if not text or not text.strip():
        raise JtwcParseError("empty warning text")

    wn = _WARN_NR_RE.search(text)
    if not wn:
        raise JtwcParseError("no 'WARNING NR' line")
    warning_number = int(wn.group(1))

    nm = _NAME_RE.search(text)
    sid_hint = nm.group(1).upper() if nm else None
    name = nm.group(2).strip().upper() if nm else None

    hdr = _HEADER_DTG_RE.search(text)
    if not hdr:
        raise JtwcParseError("no header DTG (WTPNxx PGTW DDHHMM)")
    month, year = _month_year(text)
    h_dd, h_hh, h_mm = (int(hdr.group(1)[0:2]), int(hdr.group(1)[2:4]),
                        int(hdr.group(1)[4:6]))
    issued = dt.datetime(year, month, h_dd, h_hh, h_mm, tzinfo=UTC)

    points: list[dict] = []

    # tau 0 from WARNING POSITION (the first position line after the marker).
    wp = re.search(r"WARNING\s+POSITION:", text, re.I)
    if wp:
        m0 = _POS_RE.search(text, wp.end())
        if m0:
            valid0 = _roll(int(m0.group(1)), int(m0.group(2)), int(m0.group(3)),
                           issued)
            points.append({
                "tau_h": 0,
                "valid_utc": _iso_z(valid0),
                "lat": _signed(m0.group(4), m0.group(5)),
                "lon": _signed(m0.group(6), m0.group(7)),
                "wind_kt": _wind_after(text, m0.end()),
            })

    # Forecast taus: each "NN HRS, VALID AT:" header followed by a position line.
    fc_start = text.find("FORECASTS:")
    scan_from = fc_start if fc_start >= 0 else 0
    for hm in _FCHDR_RE.finditer(text, scan_from):
        tau = int(hm.group(1))
        pm = _POS_RE.search(text, hm.end(), hm.end() + 120)
        if not pm:
            continue
        valid = _roll(int(pm.group(1)), int(pm.group(2)), int(pm.group(3)), issued)
        points.append({
            "tau_h": tau,
            "valid_utc": _iso_z(valid),
            "lat": _signed(pm.group(4), pm.group(5)),
            "lon": _signed(pm.group(6), pm.group(7)),
            "wind_kt": _wind_after(text, pm.end()),
        })

    if not points:
        raise JtwcParseError("no WARNING POSITION or FORECASTS positions parsed")

    # De-dup by tau (defensive) and order by tau.
    seen = {}
    for p in points:
        seen[p["tau_h"]] = p
    points = [seen[t] for t in sorted(seen)]

    return {
        "warning_number": warning_number,
        "issued_utc": _iso_z(issued),
        "name": name,
        "sid_hint": sid_hint,
        "points": points,
    }


def parse_warning_number(text: str) -> Optional[int]:
    """Just the warning number (the cross-product verification key). None if
    absent (an error page / outage interstitial -> stays unverified)."""
    m = _WARN_NR_RE.search(text or "")
    return int(m.group(1)) if m else None
