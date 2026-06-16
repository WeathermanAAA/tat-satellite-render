"""storm_ids - the ONE id-join for CycloLab (CYCLOLAB_DESIGN.md §3.3).

Three id dialects exist across the stack; everything derives from the
tracks-feed sid (``agency_BASINnnYYYY``):

    tracks sid     NHC_EP012026 / JTWC_WP062026
    -> atcf longid ep012026                       (basin.lower + nn + yyyy)
    -> hafs id     01e                            (nn + SUFFIX letter)
    -> nhc id      EP012026                       (CurrentStorms.json id)

BINDING (review fix): the HAFS/floater suffix letter is an EXPLICIT map,
never a slice of the basin code - ``AL -> "l"`` is the trap (a
first-letter slice yields "a"; the ATCF single-letter convention is
L=Atlantic) and ``CP -> "c"`` must hold. No Atlantic storm has run the
models pipeline this season, so tests/test_storm_ids.py carries the
mandatory AL case to keep this path correct before the first Atlantic
hurricane opens its lab.

A JS mirror of BASIN_SUFFIX ships inside the CycloLab shell template
(Stage 2) and is node-harness parity-tested against this module - the
ICON_* one-source rule. Change BOTH.
"""
from __future__ import annotations

from dataclasses import dataclass

# THE map. Explicit, exhaustive for the basins TAT serves. (ATCF also
# defines IO/SH letters; out of scope until those basins are onboarded.)
BASIN_SUFFIX: dict[str, str] = {"AL": "l", "EP": "e", "CP": "c", "WP": "w"}


@dataclass(frozen=True)
class StormIds:
    sid: str          # the tracks-feed sid, verbatim
    agency: str       # NHC / JTWC
    basin: str        # AL / EP / CP / WP
    number: int       # 1-49 designated, 90-99 invest
    year: int
    atcf_long: str    # ep012026 / ep932026
    hafs_id: str      # 01e  (EMPTY for invests - they never run the HAFS pipeline)
    nhc_id: str       # EP012026 / EP932026
    is_invest: bool = False   # 90-99: an invest AREA (grey / red-X subset page)


class InvestSidError(ValueError):
    """Retained for compatibility. Stage C made invests page-able (a grey /
    red-X SUBSET page), so parse_sid NO LONGER raises this - the only hard
    rejects are malformed sids, unmapped basins, and the 50-89 ATCF gap."""


def parse_sid(sid: str) -> StormIds:
    """Parse a tracks-feed sid into every dialect. Raises ValueError on a
    malformed sid or out-of-range storm number, KeyError on an unmapped basin
    (fail LOUD - a wrong suffix would silently 404 every model frame).

    Numbers: 1-49 = designated (full page); 90-99 = INVEST (``is_invest`` True;
    a SUBSET grey / red-X page - guidance + satellite + vitals, no cone /
    advisories / HAFS, so ``hafs_id`` is empty). 50-89 stay rejected (ATCF gap)."""
    try:
        agency, rest = sid.split("_", 1)
        basin, num_s, year_s = rest[:2], rest[2:4], rest[4:]
        number, year = int(num_s), int(year_s)
    except (ValueError, IndexError) as e:
        raise ValueError(f"malformed storm sid: {sid!r}") from e
    if basin not in BASIN_SUFFIX:
        raise KeyError(f"unmapped basin {basin!r} in sid {sid!r} "
                       f"(BASIN_SUFFIX has {sorted(BASIN_SUFFIX)})")
    is_invest = 90 <= number <= 99
    if not (2000 <= year <= 2100) or not (is_invest or 1 <= number <= 49):
        raise ValueError(f"implausible storm number/year in sid {sid!r}")
    return StormIds(
        sid=sid, agency=agency, basin=basin, number=number, year=year,
        atcf_long=f"{basin.lower()}{number:02d}{year}",
        hafs_id="" if is_invest else f"{number:02d}{BASIN_SUFFIX[basin]}",
        nhc_id=f"{basin}{number:02d}{year}",
        is_invest=is_invest,
    )
