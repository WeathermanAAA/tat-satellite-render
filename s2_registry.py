#!/usr/bin/env python3
"""Stage-2 PRODUCT REGISTRY for the satellite-ingest backbone.

STDLIB ONLY (+ the pure `s1_slots` logic module) -- no boto3 / xarray / numpy /
requests. This is the data-driven generalization of Stage 1's single-product
hardcodes: where `s1_slots.py` bakes ONE product (GOES-19 CMIPM2 clean-IR) into
module constants + a fixed regex + `is_s1_slot`, this module expresses **many**
products as a table of `ProductEntry` rows and provides the pure control logic
(key parsing, product routing, deterministic R2 keys, the completeness spec) that
a generalized ingest worker iterates over. It is ADDITIVE -- it does not touch
`s1_slots.py` / `s1_ingest.py`, so the running Stage-1 shadow + its tests are
unaffected (SATELLITE-REARCH §8: cut over per stage, never big-bang).

Scope of Stage-2 Phase 1 (this module): the **pure logic** only -- the substrate
that the §9.x logic tests exercise and that the Phase-2 ingest refactor will
consume. It deliberately does NOT add MCMIP support to the renderer
(`satellites.py` reads only per-band `CMI`; multiband `CMI_C01..C16` /  AHI L1b
multi-band assembly is Phase 3) and it does NOT re-subscribe the SNS topics
(Phase 4). See `S2-INGEST-DESIGN.md`.

THREE SUBSTRATES (how a slot's bands arrive), chosen per product:
  * "cmip"    -- GOES ABI L2 CMIP{M,C,F}: ONE band per NetCDF file (variable
                 `CMI`). A single-band product = 1 file; true color = its band
                 set accrues over N files. NATIVE resolution (C02 0.5 km) -- the
                 pixel-identity substrate for the by-construction products.
  * "mcmip"   -- GOES ABI L2 MCMIP{M,C,F}: ALL 16 bands in ONE file (variables
                 `CMI_C01..CMI_C16`), complete-on-arrival. NOAA pre-resamples
                 every band to 2 km, so it is pixel-lossy vs native-res true
                 color -- use it for NEW wide/tiled products with no prod
                 pixel-identity baseline (the §8-S2 full-disk map), never to
                 re-ingest an existing by-construction product.
  * "ahi_l1b" -- Himawari AHI L1b (Himawari has NO MCMIP/CMIP analogue): one
                 file per (band, segment). FLDK = ~10 segments/band; Target =
                 1 segment/band at 2.5-min sub-scans. Substrate for Himawari.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import re
from typing import Iterable, Optional

# Reuse the Stage-1 pure-logic primitives verbatim (both are stdlib-only and are
# the SSOT for these behaviors -- do not re-implement, so S1/S2 never diverge).
from s1_slots import (  # noqa: F401  (re-exported for consumers)
    STAMP_FMT,
    CompletenessGate,
    extract_object_key,
    scan_start_from_token,
)

UTC = dt.timezone.utc

# The composite gate-item token for a single-file multiband slot (MCMIP): the
# slot is complete the instant its one file lands (§3.2 "MCMIP = 1 file").
MCMIP_ITEM = "MCMIP"

# The tiled-product manifest scheme id (SSOT -- s2_pyramid reads it from here so
# the emitted manifest.scheme and any future variant never drift). A future
# EPSG:3857 reproject (§5.5) would add "webmercator-xyz".
TILE_SCHEME = "flat-native-xyz"

# Per-frame pyramid COMPLETION marker (written LAST, after every tile PUT
# succeeds): its presence == "this frame's whole pyramid is on R2". Idempotency
# heads THIS, not a tile, so a partial/interrupted emit (tiles present, marker
# absent) is re-rendered by the next run instead of being skipped as complete.
READY_MARKER = "_ready.json"


# ---------------------------------------------------------------------------
# Object-key parsing -> a generalized SatSlot (superset of s1_slots.Slot)
# ---------------------------------------------------------------------------
# GOES ABI L2 CMIP (per-band): OR_ABI-L2-CMIPM2-M6C13_G19_s20261692100572_e..._c...
#   sector = M1|M2|C|F  (meso sub-sector number, or CONUS/FullDisk)
_CMIP_RE = re.compile(
    r"OR_ABI-L2-CMIP(?P<sector>M[12]|C|F)-M(?P<mode>\d)C(?P<band>\d{2})"
    r"_G(?P<sat>\d{2})_s(?P<scan>\d{14})\d?"
)
# GOES ABI L2 MCMIP (multiband, NO C## band token):
#   OR_ABI-L2-MCMIPM2-M6_G19_s20260010000301_e..._c...
_MCMIP_RE = re.compile(
    r"OR_ABI-L2-MCMIP(?P<sector>M[12]|C|F)-M(?P<mode>\d)"
    r"_G(?P<sat>\d{2})_s(?P<scan>\d{14})\d?"
)
# Himawari AHI L1b: HS_H09_20260101_0000_B01_FLDK_R10_S0110.DAT[.bz2]
#   region = FLDK | R3NN  (Target 2.5-min sub-scan block); segment SkkLL (kk/LL).
_AHI_RE = re.compile(
    r"HS_H(?P<sat>\d{2})_(?P<date>\d{8})_(?P<time>\d{4})_B(?P<band>\d{2})"
    r"_(?P<region>FLDK|R\d{3})_R(?P<res>\d+)_S(?P<seg>\d{2})(?P<total>\d{2})"
)

# AHI Target sub-scan spacing: 4 sub-scans (R301..R304) per 10-min block = 2.5 min.
_AHI_SUBSCAN_S = 150


@dataclasses.dataclass(frozen=True)
class SatSlot:
    """One parsed satellite object: its family/substrate/satellite + the sector
    token, the (band, segment) it carries, and the canonical scan-start. `stamp`
    keys the R2 frame + the ledger; it is UNIQUE per renderable slot within a
    product (for AHI Target the 2.5-min sub-scan offset is folded into it)."""
    s3_key: str
    family: str                     # "goes" | "himawari"
    substrate: str                  # "cmip" | "mcmip" | "ahi_l1b"
    sat: str                        # zero-padded number: "19" | "18" | "09"
    sector_token: str               # "CMIPM2" | "MCMIPF" | "FLDK" | "R301" ...
    scan_start: dt.datetime
    band: Optional[int] = None      # ABI/AHI band; None for an MCMIP composite
    segment: Optional[int] = None   # AHI segment index (1-based); None for GOES
    total_segments: Optional[int] = None

    @property
    def stamp(self) -> str:
        return self.scan_start.strftime(STAMP_FMT)

    @property
    def slot_id(self) -> str:
        """Stable human-readable id for logging. Independent of the delivery
        filter (§3.1): parsed from the object key, not the SNS filter."""
        bits = [self.family, self.sat, self.sector_token]
        if self.band is not None:
            bits.append(f"C{self.band:02d}")
        if self.segment is not None:
            bits.append(f"S{self.segment:02d}/{self.total_segments:02d}")
        return "/".join(bits) + f"@{self.stamp}"


def _ahi_scan_start(date: str, time: str, region: str) -> dt.datetime:
    """AHI folder-time slot -> UTC. FLDK = the block time; Target (R3NN) adds the
    sub-scan offset so the 4 daily sub-scans key to distinct slots. The exact
    valid-time convention is refined in Phase 2; this is monotonic + unique."""
    base = dt.datetime.strptime(date + time, "%Y%m%d%H%M").replace(tzinfo=UTC)
    if region.startswith("R"):
        k = int(region[-1])                    # R301->1 .. R304->4
        base = base + dt.timedelta(seconds=(k - 1) * _AHI_SUBSCAN_S)
    return base


def parse_key(s3_key: str) -> Optional[SatSlot]:
    """Parse ANY supported NOAA object key (GOES CMIP, GOES MCMIP, Himawari AHI
    L1b) into a SatSlot, or None if it is not a shape we ingest. Matching the
    object key -- never the SNS delivery filter -- is the slot's identity (§3.1),
    so an over-broad/propagating filter can never mislabel a slot."""
    if not s3_key:
        return None
    name = s3_key.rsplit("/", 1)[-1]

    m = _CMIP_RE.search(name)
    if m:
        try:
            scan = scan_start_from_token(m.group("scan"))
        except (ValueError, IndexError):
            return None
        return SatSlot(
            s3_key=s3_key, family="goes", substrate="cmip", sat=m.group("sat"),
            sector_token="CMIP" + m.group("sector"), scan_start=scan,
            band=int(m.group("band")),
        )

    m = _MCMIP_RE.search(name)
    if m:
        try:
            scan = scan_start_from_token(m.group("scan"))
        except (ValueError, IndexError):
            return None
        return SatSlot(
            s3_key=s3_key, family="goes", substrate="mcmip", sat=m.group("sat"),
            sector_token="MCMIP" + m.group("sector"), scan_start=scan, band=None,
        )

    m = _AHI_RE.search(name)
    if m:
        try:
            scan = _ahi_scan_start(m.group("date"), m.group("time"), m.group("region"))
        except (ValueError, IndexError):
            return None
        return SatSlot(
            s3_key=s3_key, family="himawari", substrate="ahi_l1b",
            sat=m.group("sat"), sector_token=m.group("region"), scan_start=scan,
            band=int(m.group("band")),
            segment=int(m.group("seg")), total_segments=int(m.group("total")),
        )

    return None


# ---------------------------------------------------------------------------
# ProductEntry -- one row of the registry
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ProductEntry:
    """One ingested product: its NOAA source + slot-match predicate + bands +
    completeness spec + deterministic R2 output layout + /render dispatch. This
    is the per-product generalization of the scattered `S1_*` constants; a
    generalized worker iterates the REGISTRY and, per SQS/backfill object, marks
    every entry whose `claims(slot)` is True (one raw object can feed several
    products -- e.g. C13 feeds both the IR product and true color)."""

    # --- identity ---
    product_id: str                 # stable id, e.g. "goes19-meso2-ir"
    family: str                     # "goes" | "himawari"
    substrate: str                  # "cmip" | "mcmip" | "ahi_l1b"
    bucket: str                     # NOAA Open Data bucket, e.g. "noaa-goes19"
    sat_num: str                    # zero-padded sat number matched in the key
    # --- source location + SNS filter ---
    s3_prefix: str                  # NOAA listing/backfill prefix, e.g. "ABI-L2-CMIPM/"
    sns_filter_prefixes: tuple[str, ...]   # SNS MessageBody key-prefix policy
    accept_sectors: frozenset[str]  # sector tokens this product claims
    # --- bands / completeness ---
    channels: tuple[str, ...]       # generic channels rendered (satellites.GENERIC_CHANNELS keys)
    bands: tuple[int, ...]          # native band numbers required (empty for mcmip composite)
    ahi_segments: int = 1           # AHI segments/band required (FLDK ~10, Target 1)
    # --- R2 output layout (the NEW sat/ namespace) ---
    sat_key: str = ""               # e.g. "goes19"
    sector_key: str = ""            # e.g. "meso2" | "fd"
    band_key: str = ""              # e.g. "ir" | "truecolor"
    frame_ext: str = ".webp"
    content_type: str = "image/webp"
    prod_meso_slug: Optional[str] = None   # OLD-prod pixel-diff baseline slug ("goes19-m2"); None if no baseline
    # --- /render dispatch (satellites/render contract) ---
    render_channel: str = ""        # /render "channel", e.g. "clean_ir" | "true_color"
    render_enhancement: str = ""    # /render "enhancement", e.g. "rainbow_ir"
    render_product_hint: str = ""   # /render "product", e.g. "meso" | "conus" | "fd"
    render_sat_hint: str = ""       # /render "satellite", e.g. "GOES-East"
    # --- cadence (backfill tick + staleness window) ---
    cadence_s: int = 600
    # --- zoomable/tiled product (Stage-2 Phase 2a pyramid emitter) ---
    # tiled=True routes the slot through s2_pyramid (an XYZ tile pyramid + a
    # tiled latest_times.json) instead of the single-frame writer. floaters/meso
    # stay tiled=False single-frame sequences (SATELLITE-REARCH §4.2).
    tiled: bool = False
    tile_size: int = 512            # tile edge px (§4.2 "WebP, 512 px")
    pyramid_px: int = 4096          # target long-edge of the rendered raster -> maxzoom
    projection: str = "equirectangular"      # PlateCarree; flat-native pyramid (§4.1)
    sector_bbox: Optional[tuple] = None      # [W,S,E,N] fetch/render extent for a tiled product

    # -- routing ------------------------------------------------------------
    def claims(self, slot: Optional[SatSlot]) -> bool:
        """True if this product ingests `slot`. A raw object may be claimed by
        MORE THAN ONE entry (shared bands), so callers mark every matching
        entry's gate -- do not stop at the first."""
        if slot is None:
            return False
        if (slot.family != self.family or slot.substrate != self.substrate
                or slot.sat != self.sat_num):
            return False
        if slot.sector_token not in self.accept_sectors:
            return False
        if self.substrate == "mcmip":
            return True                     # composite: no per-band filter
        return slot.band in self.bands      # cmip / ahi_l1b: band must be wanted

    # -- completeness -------------------------------------------------------
    @property
    def required_items(self) -> frozenset:
        """The set CompletenessGate must see before a slot is renderable.
        mcmip -> {MCMIP_ITEM}; cmip -> band numbers; ahi_l1b -> (band, segment)
        pairs over all required segments."""
        if self.substrate == "mcmip":
            return frozenset({MCMIP_ITEM})
        if self.substrate == "ahi_l1b":
            return frozenset((b, s) for b in self.bands
                             for s in range(1, self.ahi_segments + 1))
        return frozenset(self.bands)

    def gate_key(self, slot: SatSlot) -> str:
        """Ledger/gate key for a slot within THIS product. `stamp` is unique per
        renderable slot (AHI sub-scan folded in), and the entry already fixes
        sat+sector, so the stamp alone keys the gate -- bands/segments of one
        scan accrue under it (the s1_ingest.py:548 NOTE, generalized)."""
        return slot.stamp

    def gate_item(self, slot: SatSlot):
        """What `slot` contributes to its gate: a composite token (mcmip), a
        (band, segment) pair (ahi_l1b), or a band number (cmip)."""
        if self.substrate == "mcmip":
            return MCMIP_ITEM
        if self.substrate == "ahi_l1b":
            return (slot.band, slot.segment)
        return slot.band

    def new_gate(self) -> CompletenessGate:
        return CompletenessGate(self.required_items)

    # -- R2 keys (deterministic; a redelivery/backfill resolves to the SAME key)
    @property
    def product_path(self) -> str:
        """The NEW namespace product path: sat/{sat_key}/{sector_key}/{band_key}."""
        return f"sat/{self.sat_key}/{self.sector_key}/{self.band_key}"

    def frame_key(self, prefix: str, stamp: str) -> str:
        return f"{prefix.strip('/')}/{self.product_path}/{stamp}{self.frame_ext}"

    def latest_times_key(self, prefix: str) -> str:
        return f"{prefix.strip('/')}/{self.product_path}/latest_times.json"

    def health_key(self, prefix: str) -> str:
        return f"{prefix.strip('/')}/{self.product_path}/health.json"

    def prod_frame_key(self, stamp: str) -> Optional[str]:
        """OLD-prod meso key for the same slot (the pixel-diff baseline), or None
        if this product has no prod baseline (a new product -- e.g. the FD map)."""
        if not self.prod_meso_slug:
            return None
        return f"meso/{self.prod_meso_slug}/{self.band_key}/{stamp}{self.frame_ext}"

    def stamp_from_frame_key(self, key: str) -> Optional[str]:
        """Recover {stamp} from one of THIS product's frame keys (tolerates
        .webp/.png), letting R2 itself be the ledger on cold start."""
        name = key.rsplit("/", 1)[-1]
        for ext in (self.frame_ext, ".webp", ".png"):
            if name.endswith(ext):
                stamp = name[: -len(ext)]
                try:
                    dt.datetime.strptime(stamp, STAMP_FMT)
                except ValueError:
                    return None
                return stamp
        return None

    def build_latest_times(self, stamps: Iterable[str], prefix: str,
                           as_of: dt.datetime) -> dict:
        """The §4.1 manifest-SSOT for this product (path template + sorted times
        + latest + as_of). Byte-compatible with s1_slots.build_latest_times."""
        times = sorted(set(stamps))
        return {
            "product": self.product_path,
            "path": f"{self.product_path}/{{t}}{self.frame_ext}",
            "tile": None,
            "times": times,
            "latest": times[-1] if times else None,
            "as_of": as_of.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": len(times),
        }

    # -- tiled/pyramid keys + manifest (Stage-2 Phase 2a) -------------------
    # The zoomable-product analogue of frame_key/latest_times/build_latest_times.
    # Deterministic keys: a redelivery/backfill resolves a (stamp,z,x,y) to the
    # SAME tile key, so dedup-by-existence holds exactly as for single frames.
    def tile_template(self) -> str:
        """Product-relative XYZ tile path template for the manifest `tile` key
        (the viewer joins the R2 prefix + substitutes {t}/{z}/{x}/{y})."""
        return f"{self.product_path}/{{t}}/{{z}}/{{x}}/{{y}}{self.frame_ext}"

    def tile_key(self, prefix: str, stamp: str, z: int, x: int, y: int) -> str:
        """Absolute R2 tile key: prefix/{product_path}/{stamp}/{z}/{x}/{y}.webp."""
        return (f"{prefix.strip('/')}/{self.product_path}"
                f"/{stamp}/{z}/{x}/{y}{self.frame_ext}")

    def tile_stamp_prefix(self, prefix: str, stamp: str) -> str:
        """List/delete prefix for one frame's whole pyramid (prune enumerates
        this and deletes ALL tiles + the marker -- a single-key prune would
        orphan tiles)."""
        base = f"{prefix.strip('/')}/{self.product_path}"
        return f"{base}/{stamp}/" if stamp else f"{base}/"

    def ready_key(self, prefix: str, stamp: str) -> str:
        """The per-frame completion marker key (written LAST). head() THIS for
        idempotency so a partial emit is never mistaken for a complete one."""
        return f"{prefix.strip('/')}/{self.product_path}/{stamp}/{READY_MARKER}"

    def stamp_from_ready_key(self, key: str) -> Optional[str]:
        """Recover {stamp} from a completion-marker key (None otherwise)."""
        suffix = "/" + READY_MARKER
        if not key.endswith(suffix):
            return None
        stamp = key[: -len(suffix)].rsplit("/", 1)[-1]
        try:
            dt.datetime.strptime(stamp, STAMP_FMT)
        except ValueError:
            return None
        return stamp

    def stamp_from_tile_key(self, key: str) -> Optional[str]:
        """Recover {stamp} from a tile key (cold-start dedup lets R2 be the
        ledger). None for latest_times.json/health.json or a malformed key."""
        if not key.endswith(self.frame_ext):
            return None
        parts = key[: -len(self.frame_ext)].split("/")
        if len(parts) < 4:
            return None
        stamp, z, x, y = parts[-4], parts[-3], parts[-2], parts[-1]
        if not (z.isdigit() and x.isdigit() and y.isdigit()):
            return None
        try:
            dt.datetime.strptime(stamp, STAMP_FMT)
        except ValueError:
            return None
        return stamp

    def build_tiled_latest_times(self, stamps: Iterable[str], *, bounds,
                                 image_px, maxzoom: int, as_of: dt.datetime,
                                 tile_size: int = 512, min_zoom: int = 0) -> dict:
        """The §4.1 SLIDER manifest, tiled variant (superset of the single-frame
        shape: keeps product/path/tile/times/latest/as_of/count with path=None +
        tile populated, adds scheme/projection/tile_size/minzoom/maxzoom/
        image_px/bounds). The viewer branches on `tile is not None` and derives
        the per-zoom grid from image_px+maxzoom -- it never lists the bucket."""
        times = sorted(set(stamps))
        return {
            "product": self.product_path,
            "path": None,                         # tiled: no single-frame path
            "tile": self.tile_template(),
            "scheme": TILE_SCHEME,                # vs a future "webmercator-xyz" (§5.5)
            "projection": self.projection,
            "tile_size": tile_size,
            "minzoom": min_zoom,
            "maxzoom": maxzoom,
            "image_px": [int(image_px[0]), int(image_px[1])] if image_px else None,
            "bounds": [float(b) for b in bounds] if bounds is not None else None,
            "times": times,
            "latest": times[-1] if times else None,
            "as_of": as_of.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": len(times),
        }

    def render_body(self, bbox, time_iso: str) -> dict:
        """The /render POST body for this product (the satellites/render
        contract: bbox,time,channel,enhancement,format,product,satellite)."""
        return {
            "bbox": bbox, "time": time_iso,
            "channel": self.render_channel, "enhancement": self.render_enhancement,
            "format": self.frame_ext.lstrip("."),
            "product": self.render_product_hint, "satellite": self.render_sat_hint,
        }


# ---------------------------------------------------------------------------
# THE REGISTRY
# ---------------------------------------------------------------------------
# Alignment sources of truth this table mirrors (must stay in sync):
#   * meso_sectors.MESO_SECTORS -- the 5 fixed sectors + slugs + buckets.
#   * meso_poller.BANDS         -- band key -> (channel, enhancement).
#   * satellites.GENERIC_CHANNELS / per-family generic_to_band -- band numbers.
#
# Stage 2 Phase 1 seeds the table with representative rows across all three
# substrates -- enough to prove the generalization + reproduce Stage 1 exactly.
# Phase 4 (re-subscribe) fills out the remaining sectors/bands.

# GOES-East (GOES-19) meso-2, clean-IR -- the STAGE-1 product, expressed as a
# registry row. It MUST reproduce s1_slots' outputs byte-for-byte (parity test).
_S1_ENTRY = ProductEntry(
    product_id="goes19-meso2-ir",
    family="goes", substrate="cmip", bucket="noaa-goes19", sat_num="19",
    s3_prefix="ABI-L2-CMIPM/",
    sns_filter_prefixes=("ABI-L2-CMIPM/",),
    accept_sectors=frozenset({"CMIPM2"}),
    channels=("clean_ir",), bands=(13,),
    sat_key="goes19", sector_key="meso2", band_key="ir",
    prod_meso_slug="goes19-m2",
    render_channel="clean_ir", render_enhancement="rainbow_ir",
    render_product_hint="meso", render_sat_hint="GOES-East",
    cadence_s=60,
)

REGISTRY: tuple[ProductEntry, ...] = (
    _S1_ENTRY,

    # --- GOES-19 meso-2 Dvorak-BD IR (same C13 object, second enhancement) ---
    ProductEntry(
        product_id="goes19-meso2-irbd",
        family="goes", substrate="cmip", bucket="noaa-goes19", sat_num="19",
        s3_prefix="ABI-L2-CMIPM/", sns_filter_prefixes=("ABI-L2-CMIPM/",),
        accept_sectors=frozenset({"CMIPM2"}),
        channels=("clean_ir",), bands=(13,),
        sat_key="goes19", sector_key="meso2", band_key="irbd",
        prod_meso_slug="goes19-m2",
        render_channel="clean_ir", render_enhancement="dvorak",
        render_product_hint="meso", render_sat_hint="GOES-East", cadence_s=60,
    ),

    # --- GOES-19 meso-2 TRUE COLOR via native-res CMIP (5-band completeness) ---
    # By-construction pixel-identity substrate: native 0.5 km C02, NOT MCMIP.
    ProductEntry(
        product_id="goes19-meso2-truecolor",
        family="goes", substrate="cmip", bucket="noaa-goes19", sat_num="19",
        s3_prefix="ABI-L2-CMIPM/", sns_filter_prefixes=("ABI-L2-CMIPM/",),
        accept_sectors=frozenset({"CMIPM2"}),
        # GOES true color: red=C02, blue=C01, veggie=C03 (green synth), clean-IR=C13.
        channels=("visible_red", "visible_blue", "veggie", "clean_ir"),
        bands=(2, 1, 3, 13),
        sat_key="goes19", sector_key="meso2", band_key="truecolor",
        prod_meso_slug="goes19-m2",
        render_channel="true_color", render_enhancement="tat_neon",
        render_product_hint="meso", render_sat_hint="GOES-East", cadence_s=60,
    ),

    # --- GOES-19 FULL-DISK via MCMIP (NEW wide product; the §8-S2 tile source) ---
    # MCMIP is correct here: one 2 km multiband file, complete-on-arrival, and the
    # z0-4 overview pyramid is coarser than 2 km so there is NO pixel-identity loss
    # that matters (no prod baseline -- reference-render gated per §6.3/§9).
    ProductEntry(
        product_id="goes19-fd-mcmip",
        family="goes", substrate="mcmip", bucket="noaa-goes19", sat_num="19",
        s3_prefix="ABI-L2-MCMIPF/", sns_filter_prefixes=("ABI-L2-MCMIPF/",),
        accept_sectors=frozenset({"MCMIPF"}),
        channels=("visible_red", "visible_blue", "veggie", "clean_ir"), bands=(),
        sat_key="goes19", sector_key="fd", band_key="truecolor",
        prod_meso_slug=None,   # additive: no prod frame to diff against
        render_channel="true_color", render_enhancement="tat_neon",
        render_product_hint="fd", render_sat_hint="GOES-East", cadence_s=600,
    ),

    # --- Himawari-9 Target (2.5-min meso) clean-IR via AHI L1b (NO MCMIP) ---
    # Tight filter: subscribe AHI-L1b-Target/ ONLY (1 segment/band), NOT the FLDK
    # firehose (10 segments x 16 bands). This is the Himawari cost/volume lever.
    ProductEntry(
        product_id="himawari9-target-ir",
        family="himawari", substrate="ahi_l1b", bucket="noaa-himawari9", sat_num="09",
        s3_prefix="AHI-L1b-Target/",
        sns_filter_prefixes=("AHI-L1b-Target/",),
        accept_sectors=frozenset({"R301", "R302", "R303", "R304"}),
        channels=("clean_ir",), bands=(13,), ahi_segments=1,
        sat_key="himawari9", sector_key="meso", band_key="ir",
        prod_meso_slug="himawari9-meso",
        render_channel="clean_ir", render_enhancement="rainbow_ir",
        render_product_hint="meso", render_sat_hint="Himawari-Pacific",
        cadence_s=150,
    ),

    # --- GOES-19 CONUS clean-IR TILE PYRAMID (Stage-2 Phase 2a zoomable) ---
    # NEW zoomable product: native-res C13 CONUS cut into a 512 px flat-native
    # XYZ WebP pyramid (s2_pyramid). No prod baseline -> reference-render gated
    # (§6.3/§9), so pixel-identity is against the FROZEN colortable, not a prod
    # frame. Single-band clean-IR => renderable with xarray+matplotlib (no satpy).
    ProductEntry(
        product_id="goes19-conus-ir",
        family="goes", substrate="cmip", bucket="noaa-goes19", sat_num="19",
        s3_prefix="ABI-L2-CMIPC/", sns_filter_prefixes=("ABI-L2-CMIPC/",),
        accept_sectors=frozenset({"CMIPC"}),
        channels=("clean_ir",), bands=(13,),
        sat_key="goes19", sector_key="conus", band_key="ir",
        prod_meso_slug=None,
        render_channel="clean_ir", render_enhancement="rainbow_ir",
        render_product_hint="conus", render_sat_hint="GOES-East",
        cadence_s=300,
        tiled=True, tile_size=512, pyramid_px=4096,
        # bbox fully inside the Mode-6 CONUS footprint (-135,14,-55,50) so the
        # fetch picker resolves CMIPC (not full disk). [W,S,E,N].
        sector_bbox=(-125.0, 15.0, -66.0, 49.0),
    ),

    # --- GOES-19 FULL-DISK clean-IR TILE PYRAMID (same code, bigger source) ---
    ProductEntry(
        product_id="goes19-fd-ir",
        family="goes", substrate="cmip", bucket="noaa-goes19", sat_num="19",
        s3_prefix="ABI-L2-CMIPF/", sns_filter_prefixes=("ABI-L2-CMIPF/",),
        accept_sectors=frozenset({"CMIPF"}),
        channels=("clean_ir",), bands=(13,),
        sat_key="goes19", sector_key="fd", band_key="ir",
        prod_meso_slug=None,
        render_channel="clean_ir", render_enhancement="rainbow_ir",
        render_product_hint="fd", render_sat_hint="GOES-East",
        cadence_s=600,
        tiled=True, tile_size=512, pyramid_px=8192,
        sector_bbox=(-156.0, -60.0, 6.0, 60.0),
    ),
)

REGISTRY_BY_ID = {e.product_id: e for e in REGISTRY}


def matching_entries(slot: Optional[SatSlot]) -> list[ProductEntry]:
    """Every product that ingests `slot` (a raw object can feed several -- e.g. a
    C13 file feeds the IR, IR-Dvorak, and true-color products). Empty if none."""
    if slot is None:
        return []
    return [e for e in REGISTRY if e.claims(slot)]
