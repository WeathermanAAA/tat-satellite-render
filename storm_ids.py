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
    number: int       # 1-49 designated (invests 90-99 rejected for V1)
    year: int
    atcf_long: str    # ep012026
    hafs_id: str      # 01e
    nhc_id: str       # EP012026


class InvestSidError(ValueError):
    """V1 scope: designated storms only - invests (90-99) get no page."""


def parse_sid(sid: str) -> StormIds:
    """Parse a tracks-feed sid into every dialect. Raises ValueError on a
    malformed sid, KeyError on an unmapped basin (fail LOUD - a wrong
    suffix would silently 404 every model frame), InvestSidError on an
    invest-range storm number (V1 designated-only guard)."""
    try:
        agency, rest = sid.split("_", 1)
        basin, num_s, year_s = rest[:2], rest[2:4], rest[4:]
        number, year = int(num_s), int(year_s)
    except (ValueError, IndexError) as e:
        raise ValueError(f"malformed storm sid: {sid!r}") from e
    if basin not in BASIN_SUFFIX:
        raise KeyError(f"unmapped basin {basin!r} in sid {sid!r} "
                       f"(BASIN_SUFFIX has {sorted(BASIN_SUFFIX)})")
    if number >= 90:
        raise InvestSidError(f"invest sid {sid!r} - V1 is designated-only")
    if not (1 <= number <= 49) or not (2000 <= year <= 2100):
        raise ValueError(f"implausible storm number/year in sid {sid!r}")
    return StormIds(
        sid=sid, agency=agency, basin=basin, number=number, year=year,
        atcf_long=f"{basin.lower()}{number:02d}{year}",
        hafs_id=f"{number:02d}{BASIN_SUFFIX[basin]}",
        nhc_id=f"{basin}{number:02d}{year}",
    )
