#!/usr/bin/env python3
"""Fixed mesoscale-sector configuration for the meso poller.

The meso poller (meso_poller.py) is a SECOND, fully isolated worker -- a sibling
of floater_poller.py. Where the floater poller follows *storms* (a moving 12 deg
crop centered on each active TC/invest), the meso poller follows the satellite
operators' own *mesoscale sectors*: the GOES ABI mesoscale floaters (CMIPM1 /
CMIPM2 -- the ~1000 km "M1"/"M2" boxes the NWS/NHC steer onto active weather)
and the Himawari AHI Target sector (the JMA-steered ~1000-2000 km Target box).

These five sectors are FIXED in this table -- the *list* of sectors never
changes. What DOES change, scan to scan, is each sector's geographic extent:
the operators move M1/M2/Target around. So the poller never hard-codes a bbox;
per scan it DISCOVERS the sector's current extent (see ``meso_poller`` ->
``discover_extent``) and renders the full band palette over that live box.

Each ``MesoSector`` row encodes only the *identity* of a sector and HOW TO
LOCATE its current scan on the public NOAA Open Data S3 buckets:

  * ``slug``        - R2 path + manifest key (e.g. "goes19-m1"). Never collides
                      with the floater poller's storm slugs (wp06 / ep90 / ...),
                      and the meso poller writes under the ``meso/`` prefix, so
                      the two workers are byte-isolated on R2.
  * ``label``       - UI label for the top manifest ("GOES-19 Mesoscale 1").
  * ``satellite``   - human satellite name ("GOES-19" / "GOES-18" /
                      "Himawari-9"), surfaced in the manifest + render title.
  * ``bucket``      - the NOAA Open Data bucket to list/read (anon S3):
                      noaa-goes19 / noaa-goes18 / noaa-himawari9.
  * ``family``      - "goes" or "himawari": picks the discovery + render path.
  * ``sector``      - the operator's sector id:
                        GOES: the ABI product short name CMIPM1 / CMIPM2.
                        Himawari: the AHI product family "Target".
  * ``locate_band`` - the generic channel used purely to FIND + read the
                      current scan's nav for extent discovery (clean_ir -- the
                      smallest, always-present IR band; never the visible
                      0.5 km band, to keep the extent probe cheap).

The sectors are basin/region-agnostic: the operators decide where M1/M2/Target
point, and the poller follows them wherever they go (subject to each sat's disk
-- handled by ``meso_poller``). Nothing here reads or writes tracks / ACE /
climatology data; meso sectors are a presentation-only satellite product.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class MesoSector:
    """One fixed mesoscale sector + how to locate its current scan.

    Identity only -- the live geographic extent is discovered per-scan by the
    poller (the operators steer these boxes around), never stored here.
    """
    slug: str            # R2 path / manifest key, e.g. "goes19-m1"
    label: str           # UI label, e.g. "GOES-19 Mesoscale 1"
    satellite: str       # human sat name, e.g. "GOES-19"
    bucket: str          # NOAA Open Data S3 bucket, e.g. "noaa-goes19"
    family: str          # "goes" | "himawari" -- selects discovery/render path
    sector: str          # operator sector id: "CMIPM1"/"CMIPM2"/"Target"
    locate_band: str     # generic channel used to find + read the scan nav


# The FIVE fixed sectors. GOES-East (GOES-19) + GOES-West (GOES-18) each run two
# independently-steered mesoscale floaters (M1, M2); Himawari-9 runs one steered
# Target sector. The list is intentionally exhaustive + immutable -- onboarding a
# new operator sector is a row here + nothing else (the poller is sector-agnostic).
MESO_SECTORS: tuple[MesoSector, ...] = (
    MesoSector(
        slug="goes19-m1",
        label="GOES-19 Mesoscale 1",
        satellite="GOES-19",
        bucket="noaa-goes19",
        family="goes",
        sector="CMIPM1",
        locate_band="clean_ir",
    ),
    MesoSector(
        slug="goes19-m2",
        label="GOES-19 Mesoscale 2",
        satellite="GOES-19",
        bucket="noaa-goes19",
        family="goes",
        sector="CMIPM2",
        locate_band="clean_ir",
    ),
    MesoSector(
        slug="goes18-m1",
        label="GOES-18 Mesoscale 1",
        satellite="GOES-18",
        bucket="noaa-goes18",
        family="goes",
        sector="CMIPM1",
        locate_band="clean_ir",
    ),
    MesoSector(
        slug="goes18-m2",
        label="GOES-18 Mesoscale 2",
        satellite="GOES-18",
        bucket="noaa-goes18",
        family="goes",
        sector="CMIPM2",
        locate_band="clean_ir",
    ),
    MesoSector(
        slug="himawari9-meso",
        label="Himawari-9 Target",
        satellite="Himawari-9",
        bucket="noaa-himawari9",
        family="himawari",
        sector="Target",
        locate_band="clean_ir",
    ),
)

MESO_SECTORS_BY_SLUG = {s.slug: s for s in MESO_SECTORS}
