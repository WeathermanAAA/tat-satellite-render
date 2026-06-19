#!/usr/bin/env python3
"""S1 multi-satellite source registry (STAGE A).

Generalises the GREEN-gated GOES-19 S1 pattern to two more satellites WITHOUT
reinventing the worker: GOES-18 (ABI, a config twin of GOES-19) and Himawari-9
(AHI, the 07W satellite -- a different key format + SEGMENTED scans).

A ``SatSource`` carries everything source-specific (NOAA bucket + SNS topic, the
SQS queue/DLQ names + the body-path filter prefix the IaC subscribes, the R2
shadow product path, how to PARSE an object key into a Slot, what makes a slot
COMPLETE, the /render body, and how to discover the render extent). The worker
(s1_ingest) selects one source by the ``S1_SOURCE`` env and is otherwise
unchanged -- the never-miss spine (SQS + watermark/backfill + idempotency + the
CompletenessGate + cold-start + DLQ + watchdog) is reused verbatim.

PER-SOURCE ISOLATION: each source is a SEPARATE worker process (its own queue),
so a new-sat parse/render failure can NEVER stale GOES-19 S1, the meso poller,
the floater, or ACE/tracks. AHI's multi-SEGMENT completeness (the southern-
segment lesson: never publish a half-uploaded FLDK scan -- wait for all 10
segments) is the CompletenessGate's required-set, exactly as the gate was
designed for (s1_slots docstring).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import re
from typing import Optional

import s1_slots as S

UTC = dt.timezone.utc

# NOAA NODD SNS topics (account 123901341784, us-east-1; SQS protocol allowed).
_NODD = "arn:aws:sns:us-east-1:123901341784"

# AHI Himawari Standard Data filename, e.g.
#   HS_H09_20260619_1850_B13_FLDK_R20_S0110.DAT.bz2
#                ^date    ^time ^band      ^res ^seg^total
_AHI_RE = re.compile(
    r"HS_H(?P<sat>\d{2})_(?P<date>\d{8})_(?P<time>\d{4})_B(?P<band>\d{2})"
    r"_FLDK_R\d{2}_S(?P<seg>\d{2})(?P<total>\d{2})\.DAT")


@dataclasses.dataclass(frozen=True)
class SatSource:
    key: str                 # "goes19" | "goes18" | "himawari9"
    family: str              # "abi" | "ahi"
    bucket: str              # NOAA NODD bucket
    topic_arn: str           # NOAA SNS topic ARN
    queue_name: str          # our SQS main queue
    dlq_name: str            # our SQS DLQ
    filter_prefix: str       # SNS body-path prefix + the NOAA listing root
    product_path: str        # R2 shadow product path: sat/{sat}/{sector}/{band}
    band: int                # native band we ingest (clean-IR = 13)
    required_segments: int   # 1 for ABI (single object); 10 for AHI FLDK
    render_channel: str      # /render channel
    render_enhancement: str  # /render enhancement
    render_sat_hint: str     # /render satellite hint (the picker family)
    render_product_hint: Optional[str]  # "meso" (ABI) | None (AHI FLDK -> picker)
    extent_mode: str         # "discover" (read the object's geo extent) | "fixed"
    fixed_bbox: Optional[tuple] = None   # AHI FLDK: [lon_w, lat_s, lon_e, lat_n]
    abi_sat_num: str = ""    # "19" | "18" (ABI key G## field)
    abi_sector_token: str = ""  # "CMIPM2" (ABI sector we accept)


SOURCES = {
    # GOES-19: byte-equivalent to the existing hardcoded S1 product (the GREEN
    # baseline) -- product_path/render params/extent all match s1_slots exactly.
    "goes19": SatSource(
        key="goes19", family="abi", bucket="noaa-goes19",
        topic_arn=f"{_NODD}:NewGOES19Object",
        queue_name="tat-sat-goes19-cmip", dlq_name="tat-sat-goes19-cmip-dlq",
        filter_prefix="ABI-L2-CMIPM/", product_path="sat/goes19/meso2/ir",
        band=13, required_segments=1,
        render_channel="clean_ir", render_enhancement="rainbow_ir",
        render_sat_hint="GOES-East", render_product_hint="meso",
        extent_mode="discover", abi_sat_num="19", abi_sector_token="CMIPM2"),
    # GOES-18 (GOES-West): the ABI config twin -- same key format, different
    # bucket/sat/topic/queue + GOES-West render hint.
    "goes18": SatSource(
        key="goes18", family="abi", bucket="noaa-goes18",
        topic_arn=f"{_NODD}:NewGOES18Object",
        queue_name="tat-sat-goes18-cmip", dlq_name="tat-sat-goes18-cmip-dlq",
        filter_prefix="ABI-L2-CMIPM/", product_path="sat/goes18/meso2/ir",
        band=13, required_segments=1,
        render_channel="clean_ir", render_enhancement="rainbow_ir",
        render_sat_hint="GOES-West", render_product_hint="meso",
        extent_mode="discover", abi_sat_num="18", abi_sector_token="CMIPM2"),
    # Himawari-9 (07W's satellite): AHI FLDK clean-IR, a SEGMENTED scan (10
    # segments) over a fixed WPAC extent that always covers the WPAC basin.
    "himawari9": SatSource(
        key="himawari9", family="ahi", bucket="noaa-himawari9",
        topic_arn=f"{_NODD}:NewHimawariNineObject",
        queue_name="tat-sat-himawari9-fldk", dlq_name="tat-sat-himawari9-fldk-dlq",
        filter_prefix="AHI-L1b-FLDK/", product_path="sat/himawari9/fldk/ir",
        band=13, required_segments=10,
        render_channel="clean_ir", render_enhancement="rainbow_ir",
        render_sat_hint="Himawari-Pacific", render_product_hint=None,
        extent_mode="fixed", fixed_bbox=(115.0, -5.0, 165.0, 35.0)),
}


def get_source(key: str) -> SatSource:
    if key not in SOURCES:
        raise SystemExit(f"unknown S1_SOURCE {key!r}; known: {sorted(SOURCES)}")
    return SOURCES[key]


# ---------------------------------------------------------------------------
# AHI key parsing -> a Slot (segment-aware). ABI reuses s1_slots.parse_goes_key.
# ---------------------------------------------------------------------------
def parse_ahi_key(s3_key: str) -> Optional[S.Slot]:
    """Parse a Himawari AHI FLDK HSD object key into a (segment-bearing) Slot, or
    None if not a parseable FLDK scan. scan_start = the filename's HHMM (FLDK is
    a 10-min slot); the segment number drives the completeness gate."""
    if not s3_key:
        return None
    name = s3_key.rsplit("/", 1)[-1]
    m = _AHI_RE.search(name)
    if not m:
        return None
    try:
        scan = dt.datetime.strptime(m.group("date") + m.group("time"),
                                    "%Y%m%d%H%M").replace(tzinfo=UTC)
    except ValueError:
        return None
    return S.Slot(s3_key=s3_key, sector_token="FLDK", band=int(m.group("band")),
                  sat="H" + m.group("sat"), scan_start=scan,
                  segment=int(m.group("seg")))


def parse(source: SatSource, s3_key: str) -> Optional[S.Slot]:
    """Parse an object key into a Slot for ``source`` (dispatch on family)."""
    if source.family == "ahi":
        return parse_ahi_key(s3_key)
    return S.parse_goes_key(s3_key)   # ABI


def is_ours(source: SatSource, slot: Optional[S.Slot]) -> bool:
    """Is this slot the source's product? Parsed from the key, INDEPENDENT of the
    SNS filter (an over-broad/propagating filter never mislabels a slot)."""
    if slot is None:
        return False
    if source.family == "ahi":
        return (slot.sat == f"H{source.abi_sat_num}" if source.abi_sat_num
                else slot.sat.startswith("H")) and \
            slot.sector_token == "FLDK" and slot.band == source.band
    return (slot.sector_token == source.abi_sector_token
            and slot.band == source.band
            and slot.sat == source.abi_sat_num)


# ---------------------------------------------------------------------------
# Completeness gate wiring (the CompletenessGate is reused verbatim; we only
# choose its required-set + per-slot key/item).
# ---------------------------------------------------------------------------
def gate_required(source: SatSource) -> frozenset:
    """The set a slot must accrue to be complete. ABI: the single native band.
    AHI FLDK: all N segments (the southern-segment lesson)."""
    if source.family == "ahi":
        return frozenset(range(1, source.required_segments + 1))
    return frozenset({source.band})


def gate_key(slot: S.Slot) -> str:
    """One scan = one completeness slot, keyed by its scan stamp."""
    return slot.stamp


def gate_item(source: SatSource, slot: S.Slot):
    """What this object contributes to its slot's completeness: ABI -> the band
    (a single C13 object completes it); AHI -> the segment number."""
    return slot.segment if source.family == "ahi" else slot.band


def slot_label(source: SatSource, slot: S.Slot) -> str:
    """Human/log id, source-aware (NOT the GOES-only Slot.slot_id)."""
    seg = f"S{slot.segment:02d}" if slot.segment is not None else ""
    return f"{source.key}/{slot.sector_token.lower()}/B{slot.band:02d}{seg}@{slot.stamp}"


# ---------------------------------------------------------------------------
# /render body (FROZEN params, source-specific) -- byte-identical to the meso/
# floater render for the same slot.
# ---------------------------------------------------------------------------
def render_body(source: SatSource, bbox, time_iso: str) -> dict:
    body = {
        "bbox": list(bbox), "time": time_iso,
        "channel": source.render_channel,
        "enhancement": source.render_enhancement,
        "format": "webp",
        "satellite": source.render_sat_hint,
    }
    if source.render_product_hint is not None:
        body["product"] = source.render_product_hint
    return body


# ---------------------------------------------------------------------------
# NOAA ground-truth listing prefixes (pure path logic; s1_ingest does the I/O).
# ---------------------------------------------------------------------------
def noaa_prefixes(source: SatSource, now: dt.datetime, lookback_min: int) -> list[str]:
    """The NOAA bucket prefixes to ListObjectsV2 to cover the lookback window.
    ABI: hour partitions (ABI-L2-CMIPM/Y/DOY/HH/). AHI FLDK: 10-min slot
    partitions (AHI-L1b-FLDK/Y/MM/DD/HHMM/)."""
    out: list[str] = []
    if source.family == "ahi":
        # one prefix per 10-min slot across the window (+ a slot of slack)
        n = max(1, lookback_min // 10 + 2)
        base = now.replace(second=0, microsecond=0)
        base = base - dt.timedelta(minutes=base.minute % 10)
        for i in range(n):
            t = base - dt.timedelta(minutes=10 * i)
            out.append(f"{source.filter_prefix}{t.year}/{t.month:02d}/"
                       f"{t.day:02d}/{t.hour:02d}{t.minute:02d}/")
        return out
    # ABI: hour partitions
    hours = max(1, lookback_min // 60 + 2)
    for h in range(hours):
        t = now - dt.timedelta(hours=h)
        out.append(f"{source.filter_prefix}{t.year}/{t.strftime('%j')}/{t.hour:02d}/")
    return out


def complete_scans(source: SatSource, keys) -> dict:
    """Group object keys into COMPLETE slots for ``source`` -> {stamp: Slot}.
    ABI: each accepted object is already a complete slot. AHI: a stamp is
    complete only when all required segments are present (never a half-scan)."""
    by_stamp_segs: dict[str, set] = {}
    rep: dict[str, S.Slot] = {}
    for k in keys:
        slot = parse(source, k)
        if not is_ours(source, slot):
            continue
        by_stamp_segs.setdefault(slot.stamp, set()).add(gate_item(source, slot))
        rep[slot.stamp] = slot
    required = gate_required(source)
    return {st: rep[st] for st, items in by_stamp_segs.items()
            if required.issubset(items)}
