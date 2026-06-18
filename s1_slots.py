#!/usr/bin/env python3
"""Pure slot logic for the S1 satellite-ingest backbone (STAGE 1).

STDLIB ONLY -- no boto3 / xarray / numpy / requests. The never-miss CONTROL
logic (SNS-envelope unwrap, NOAA key parsing, the CMIPM2/C13 product filter, the
completeness gate, the slot ledger, deterministic R2 key derivation, the
latest_times.json SSOT) is exercised by unit tests with no AWS/render deps
(SATELLITE-REARCH §9.x "logic testing beyond pixels"). All I/O orchestration --
SQS, R2, /render, extent discovery -- lives in s1_ingest.py.

S1 SCOPE: ONE product -- GOES-19 Mesoscale 2 (operator sector CMIPM2), clean-IR
(ABI band 13), enhancement rainbow_ir: the "ir" hot band of the goes19-m2 meso
sector (meso_sectors.MESO_SECTORS / meso_poller.BANDS). Byte-identical to prod
by construction: the shadow render reuses the FROZEN (fetch -> render_png ->
transcode_frame) path via the same /render service the box's meso lane calls.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
from typing import Iterable, Optional

UTC = dt.timezone.utc

# ---------------------------------------------------------------------------
# S1 product identity (one product) + the shadow R2 namespace (§4.1 scheme)
# ---------------------------------------------------------------------------
S1_BUCKET = "noaa-goes19"
S1_SAT_KEY = "goes19"            # R2 path segment
S1_SECTOR_KEY = "meso2"          # R2 path segment (operator sector CMIPM2)
S1_BAND_KEY = "ir"               # R2 path segment (meso BANDS "ir")
S1_ABI_SECTOR_TOKEN = "CMIPM2"   # the filename sector token we accept
S1_NATIVE_BAND = 13              # ABI band 13, clean-IR longwave window 10.3um
S1_RENDER_CHANNEL = "clean_ir"   # /render generic channel
S1_RENDER_ENHANCEMENT = "rainbow_ir"
S1_RENDER_PRODUCT_HINT = "meso"  # forces _pick_meso (CMIPM) in find_file
S1_RENDER_SAT_HINT = "GOES-East"

# Required bands for a slot to be "complete". Clean-IR meso = ONE band (C13);
# the gate is generic so true-color (S3) can require its 5 bands and an MCMIP
# composite can require a single item -- see CompletenessGate.
S1_REQUIRED_BANDS = frozenset({S1_NATIVE_BAND})

# Shadow product path: sat/goes19/meso2/ir  (frames + latest_times.json under it)
S1_PRODUCT_PATH = f"sat/{S1_SAT_KEY}/{S1_SECTOR_KEY}/{S1_BAND_KEY}"
# Prod (current meso poller) frame path for the SAME slot -- the pixel-diff baseline.
S1_PROD_PRODUCT_PATH = "meso/goes19-m2/ir"

STAMP_FMT = "%Y%m%dT%H%M%SZ"

# OR_ABI-L2-CMIPM2-M6C13_G19_s20261692100572_e..._c....nc
#            ^sector  ^mode^band ^sat ^scan-start (14 digits + tenths)
_KEY_RE = re.compile(
    r"OR_ABI-L2-(?P<prod>CMIPM(?P<sector_n>[12]))-M(?P<mode>\d)C(?P<band>\d{2})"
    r"_G(?P<sat>\d{2})_s(?P<scan>\d{14})\d?"
)


# ---------------------------------------------------------------------------
# SNS-envelope unwrap (defense in depth: the worker handles BOTH raw S3-event
# delivery and the SNS envelope -- RawMessageDelivery has a propagation window,
# and could be flipped; the worker never trusts a single delivery shape).
# ---------------------------------------------------------------------------
def extract_object_key(message_body) -> Optional[str]:
    """Pull Records[0].s3.object.key from an SQS message body that is EITHER a
    raw S3-event JSON OR an SNS-enveloped notification (Type=Notification with
    the S3 event in the "Message" string). Returns None if no key is present."""
    body = message_body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            return None
    if not isinstance(body, dict):
        return None
    # SNS envelope -> unwrap the inner S3 event.
    if "Records" not in body and isinstance(body.get("Message"), str):
        try:
            body = json.loads(body["Message"])
        except (ValueError, TypeError):
            return None
    try:
        key = body["Records"][0]["s3"]["object"]["key"]
    except (KeyError, IndexError, TypeError):
        return None
    return key if isinstance(key, str) else None


# ---------------------------------------------------------------------------
# NOAA GOES object-key parsing -> a Slot
# ---------------------------------------------------------------------------
def scan_start_from_token(s_token: str) -> dt.datetime:
    """Parse a GOES ``s`` scan-start token (YYYYDOYHHMMSS[t]) to UTC, dropping
    tenths -- BYTE-FOR-BYTE the same arithmetic as satellites._parse_scan_start,
    so a shadow slot stamp equals prod's X-Scan-Time stamp for the same scan."""
    year = int(s_token[0:4])
    doy = int(s_token[4:7])
    hh = int(s_token[7:9])
    mm = int(s_token[9:11])
    ss = int(s_token[11:13])
    return dt.datetime(year, 1, 1, hh, mm, ss, tzinfo=UTC) + dt.timedelta(days=doy - 1)


@dataclasses.dataclass(frozen=True)
class Slot:
    """One parsed GOES CMIPM object: its product/sector/band/satellite + the
    canonical s-slot scan-start. ``stamp`` keys the R2 frame + the ledger."""
    s3_key: str
    sector_token: str       # "CMIPM1" | "CMIPM2"
    band: int               # ABI band number (e.g. 13)
    sat: str                # "19"
    scan_start: dt.datetime

    @property
    def stamp(self) -> str:
        return self.scan_start.strftime(STAMP_FMT)

    @property
    def slot_id(self) -> str:
        # Stable, human-readable id for logging + the (product,sat,channel,s-slot)
        # ledger key -- independent of the delivery filter (§3.1).
        return f"{S1_SAT_KEY}/{self.sector_token.lower()}/C{self.band:02d}@{self.stamp}"


def parse_goes_key(s3_key: str) -> Optional[Slot]:
    """Parse a NOAA GOES ABI CMIPM object key into a Slot, or None if the key is
    not a parseable CMIPM scan (any other product is silently not-ours)."""
    if not s3_key:
        return None
    name = s3_key.rsplit("/", 1)[-1]
    m = _KEY_RE.search(name)
    if not m:
        return None
    try:
        scan = scan_start_from_token(m.group("scan"))
    except (ValueError, IndexError):
        return None
    return Slot(
        s3_key=s3_key,
        sector_token=m.group("prod"),          # CMIPM1 / CMIPM2
        band=int(m.group("band")),
        sat=m.group("sat"),
        scan_start=scan,
    )


def is_s1_slot(slot: Optional[Slot]) -> bool:
    """Is this the S1 product (GOES-19 CMIPM2, band 13)? The worker applies this
    REGARDLESS of the SNS delivery filter -- the slot identity is parsed from the
    object key, so an over-broad or propagating filter never mislabels a slot
    (SATELLITE-REARCH §3.1)."""
    return (slot is not None
            and slot.sector_token == S1_ABI_SECTOR_TOKEN
            and slot.band == S1_NATIVE_BAND
            and slot.sat == "19")


# ---------------------------------------------------------------------------
# Deterministic R2 keys (idempotency: a re-delivered event or a racing backfill
# resolves to the SAME key -> overwrite/skip is a no-op; §3.3)
# ---------------------------------------------------------------------------
def shadow_frame_key(prefix: str, stamp: str) -> str:
    """shadow/sat/goes19/meso2/ir/{stamp}.webp"""
    return f"{prefix.strip('/')}/{S1_PRODUCT_PATH}/{stamp}.webp"


def latest_times_key(prefix: str) -> str:
    """shadow/sat/goes19/meso2/ir/latest_times.json (the §4.1 SSOT)."""
    return f"{prefix.strip('/')}/{S1_PRODUCT_PATH}/latest_times.json"


def health_key(prefix: str) -> str:
    return f"{prefix.strip('/')}/sat/{S1_SAT_KEY}/{S1_SECTOR_KEY}/{S1_BAND_KEY}/health.json"


def prod_frame_key(stamp: str) -> str:
    """meso/goes19-m2/ir/{stamp}.webp -- the prod meso poller's key for the SAME
    slot (X-Scan-Time stamp == our s-slot stamp), used by the pixel-diff."""
    return f"{S1_PROD_PRODUCT_PATH}/{stamp}.webp"


def stamp_from_frame_key(key: str) -> Optional[str]:
    """Recover the {stamp} from a shadow/prod frame key (the inverse of
    shadow_frame_key), tolerating .webp / .png. Lets R2 itself be the ledger."""
    name = key.rsplit("/", 1)[-1]
    for ext in (".webp", ".png"):
        if name.endswith(ext):
            stamp = name[: -len(ext)]
            # Validate it round-trips through the stamp format.
            try:
                dt.datetime.strptime(stamp, STAMP_FMT)
            except ValueError:
                return None
            return stamp
    return None


def build_latest_times(stamps: Iterable[str], prefix: str,
                       as_of: dt.datetime) -> dict:
    """The §4.1 manifest-SSOT for a single-frame product: a path template + the
    sorted time list + latest + as_of. The viewer derives every URL from the
    template; no per-frame key list, no bucket listing."""
    times = sorted(set(stamps))
    return {
        "product": S1_PRODUCT_PATH,
        "path": f"{S1_PRODUCT_PATH}/{{t}}.webp",
        "tile": None,
        "times": times,
        "latest": times[-1] if times else None,
        "as_of": as_of.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(times),
    }


# ---------------------------------------------------------------------------
# Completeness gate + slot ledger (§3.2)
# ---------------------------------------------------------------------------
class CompletenessGate:
    """Accumulates landed items per slot; a slot is renderable once its required
    set is satisfied. Generic so S1 (clean-IR meso = {13}), true-color (S3 =
    its band set), and MCMIP (a single composite item) all reuse it. In-memory
    (a cache, not truth -- §3.6: R2 is truth; the ledger reseeds from R2 on cold
    start). The renderer's degenerate-NaN guard stays the last line of defence
    against a present-but-corrupt band (§3.2)."""

    def __init__(self, required: Iterable):
        self.required = frozenset(required)
        self._slots: dict[str, set] = {}
        self._done: set[str] = set()

    def mark(self, slot_key: str, item) -> bool:
        """Record that ``item`` (a band number, or a composite token) landed for
        ``slot_key``. Returns True the FIRST time the slot becomes complete (the
        edge that should schedule a render), False otherwise (incomplete, or
        already-complete = idempotent no re-fire)."""
        present = self._slots.setdefault(slot_key, set())
        present.add(item)
        if slot_key in self._done:
            return False
        if self.required.issubset(present):
            self._done.add(slot_key)
            return True
        return False

    def is_complete(self, slot_key: str) -> bool:
        return slot_key in self._done

    def missing(self, slot_key: str) -> set:
        return set(self.required) - self._slots.get(slot_key, set())

    def seed_complete(self, slot_key: str) -> None:
        """Mark a slot already-complete (cold-start seeding from R2 reality, or
        marking a slot we already published) so it never re-fires."""
        self._slots.setdefault(slot_key, set()).update(self.required)
        self._done.add(slot_key)

    def forget(self, slot_key: str) -> None:
        """Drop a slot from the ledger (retention/window trimming)."""
        self._slots.pop(slot_key, None)
        self._done.discard(slot_key)
