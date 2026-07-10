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

import re
from dataclasses import dataclass

# THE map. Explicit, exhaustive for the basins TAT serves. (ATCF also
# defines IO/SH letters; out of scope until those basins are onboarded.)
BASIN_SUFFIX: dict[str, str] = {"AL": "l", "EP": "e", "CP": "c", "WP": "w"}


# --- Designation vs real seasonal name (shared by BOTH live pollers) ---------
# NHC's spelled-out designation numbers ("ONE".."FIFTY-NINE") - the placeholder
# a depression/PTC carries before it is NAMED. Lock-step with
# cyclolab_shell._ptc_number_words()/isNamedTC (same rule; keep in sync).
def _build_number_words() -> frozenset:
    ones = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
            "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
            "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"]
    tens = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY"]
    s = set()
    for n in range(1, 60):
        s.add(ones[n] if n < 20
              else tens[n // 10] + (("-" + ones[n % 10]) if n % 10 else ""))
    return frozenset(s)


_NUMBER_WORD_DESIGNATIONS = _build_number_words()
_NAME_PLACEHOLDERS = frozenset({"", "INVEST", "NAMELESS", "UNNAMED"})
# CurrentStorms suffixes NHC's spelled designation with the basin letter for the
# eastern basins ("Four-E" EP, "Four-C" CP); AL/WP carry the bare word ("Four").
# A real seasonal name never contains a hyphen, so peeling a trailing
# "-<basin letter>" can never truncate a genuine name.
_DESIGNATION_BASIN_SUFFIXES = ("-E", "-C", "-L", "-W")
# "#04" / "04E" / "4E" numeric designation fallbacks (parse_bdeck's #NN,
# knackwx's <num><letter>) - a designation, not a real name.
_NUMERIC_DESIGNATION_RE = re.compile(r"^#?\d{1,2}[ELCWP]?$")


def is_real_storm_name(name) -> bool:
    """True iff ``name`` is a genuine seasonal storm name (DOUGLAS), NOT an
    unnamed depression's designation placeholder.

    Recognizes as NON-real: the blank/INVEST/UNNAMED/NAMELESS placeholders,
    NHC's spelled-ordinal designations in BOTH the b-deck form ("FOUR") and the
    CurrentStorms basin-suffixed form ("FOUR-E"), and the numeric "#04"/"04E"
    fallbacks. The live pollers use it so NHC CurrentStorms.json can never DEMOTE
    an already-real b-deck/feed name back to a designation when CurrentStorms
    lags a synoptic-time upgrade (bep042026.dat said DOUGLAS while CurrentStorms
    still said "Four-E", 2026-07-01)."""
    n = str(name or "").strip().upper()
    if not n or n in _NAME_PLACEHOLDERS or _NUMERIC_DESIGNATION_RE.match(n):
        return False
    core = n
    for suf in _DESIGNATION_BASIN_SUFFIXES:
        if core.endswith(suf) and len(core) > len(suf):
            core = core[:-len(suf)]
            break
    return core not in _NUMBER_WORD_DESIGNATIONS


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
