#!/usr/bin/env python3
"""S1 satellite-ingest worker -- event-driven, never-miss (STAGE 1).

ONE product: GOES-19 Mesoscale 2 clean-IR (the goes19-m2 "ir" band). SHADOW
ONLY -- writes the R2 ``shadow/`` prefix; no viewer change, no prod cutover. The
renderer is FROZEN: this worker calls the SAME /render service the box's meso
lane calls, with the SAME body params, so a shadow frame is byte-identical to
the prod meso frame for the same slot by construction (SATELLITE-REARCH §2/§5.3).

The never-miss backbone (SATELLITE-REARCH §3):
  * SQS events (primary)   -- NOAA NewGOES19Object -> SQS long-poll. Parse the
                              object key into a Slot INDEPENDENT of the SNS
                              filter; keep only CMIPM2/C13. Envelope-aware.
  * watermark + backfill   -- a ListObjectsV2 reconcile of the recent NOAA
                              prefix, NEWEST-FIRST, enqueues any complete slot
                              SQS never delivered (the §3.3 fallback authority).
  * idempotency            -- deterministic R2 keys + "skip if present" make a
                              re-delivered event / racing backfill a no-op.
  * completeness gate       -- clean-IR meso = 1 band (C13) -> complete on the
                              single object; the renderer's degenerate-NaN guard
                              stays the last line of defence.
  * cold start             -- seed the watermark + ledger from R2 REALITY (§3.6).
  * DLQ                    -- SQS redrive maxReceiveCount=5 (set by the IaC); a
                              poison slot lands in the DLQ instead of wedging the
                              queue. The worker surfaces DLQ depth + receive-count
                              climb in health, and re-detects DLQ'd slots via
                              backfill on recovery.
  * delete-after-PUT       -- the SQS message is deleted ONLY after the R2 PUT
                              succeeds (or the slot is acked as no-data/off-sat).
  * isolation + heartbeat  -- per-source SourceHealth (poller_framework) + an
                              always-emitted health snapshot + latest_times.json
                              as_of staleness (the cheapest §3.5 detector).
  * liveness               -- a watchdog self-exits on a stalled cycle so
                              systemd Restart=always (compose restart:always)
                              recovers it (the HAFS lesson).

Every drop / backfill / DLQ-bound message is log()-ed with its slot id, so
"we covered everything" is auditable, never false-green (§3.3).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import http.server
import json
import logging
import os
import threading
import time
from typing import Callable, Optional

import boto3
import requests
from botocore.config import Config as BotoConfig

import s1_slots as S
import s1_sources as SRC
from poller_framework import FAILING, FRESH, SourceHealth, process_mem_mb

UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Config (env-driven; safe defaults; mirrors the meso poller's knob style)
# ---------------------------------------------------------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    return (_env(name, "1" if default else "0") or "").strip().lower() not in (
        "0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


S1_ENABLED = _env_bool("S1_ENABLED", True)

# Which satellite source this worker ingests (STAGE A multi-sat): goes19 (the
# GREEN baseline), goes18, or himawari9. One worker == one source == one queue,
# so a new-sat failure can never stale another source (per-source isolation).
S1_SOURCE = _env("S1_SOURCE", "goes19")
SOURCE = SRC.get_source(S1_SOURCE)

# SQS (the worker's own AWS creds -- tat-sat-ingest -- come from the env via the
# boto3 default chain: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / region).
S1_QUEUE_URL = _env("S1_QUEUE_URL")
S1_DLQ_URL = _env("S1_DLQ_URL")          # optional; only for DLQ-depth in health
AWS_REGION = _env("AWS_DEFAULT_REGION") or _env("AWS_REGION") or "us-east-1"
SQS_WAIT_S = _env_int("S1_SQS_WAIT_S", 20)         # long-poll seconds
SQS_BATCH = _env_int("S1_SQS_BATCH", 10)
RECEIVE_COUNT_WARN = _env_int("S1_RECEIVE_COUNT_WARN", 3)  # log slot nearing DLQ

# R2 (same bucket as the meso/floater workers; SHADOW prefix only).
R2_ENDPOINT = _env("R2_ENDPOINT")
R2_BUCKET = _env("R2_BUCKET", "triple-a-tropics-media")
R2_ACCESS_KEY_ID = _env("R2_ACCESS_KEY_ID") or _env("AWS_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = _env("R2_SECRET_ACCESS_KEY") or _env("AWS_SECRET_ACCESS_KEY")
R2_PREFIX = (_env("S1_R2_PREFIX", "shadow") or "shadow").strip("/")

# Render service (dedicated s1-render container, byte-identical to the box's meso
# render -- same image; isolated so S1 never contends with the meso hot/cold lanes).
RENDER_URL = (_env("S1_RENDER_URL", "http://s1-render:8080") or "").rstrip("/") + "/render"
RENDER_TIMEOUT_S = _env_float("S1_RENDER_TIMEOUT_S", 60.0)
RENDER_MAX_RETRIES = _env_int("S1_RENDER_MAX_RETRIES", 3)

# Backfill (the never-miss fallback authority).
BACKFILL_INTERVAL_S = _env_float("S1_BACKFILL_INTERVAL_S", 60.0)
BACKFILL_LOOKBACK_MIN = _env_int("S1_BACKFILL_LOOKBACK_MIN", 120)  # current+prev hour-ish

# Shadow retention: keep a multi-day rolling window for the never-miss audit,
# prune beyond it so R2 stays bounded.
RETAIN_H = _env_float("S1_RETAIN_H", 72.0)

# Liveness watchdog: self-exit if a cycle makes no progress for this long (the
# HAFS lesson -- a wedged render must restart, not hang silently). Must exceed
# the SQS long-poll wait + a generous render.
WATCHDOG_S = _env_float("S1_WATCHDOG_S", 600.0)

# Health classification + tiny health HTTP server (a compose healthcheck curls it).
STALE_AFTER_S = _env_float("S1_STALE_AFTER_S", 600.0)
FAIL_THRESHOLD = _env_int("S1_FAIL_THRESHOLD", 3)
HEALTH_PORT = _env_int("S1_HEALTH_PORT", 8091)
HEALTH_FILE = _env("S1_HEALTH_FILE", "/tmp/s1_health.json")

CACHE_FRAME = "public, max-age=31536000, immutable"
CACHE_MANIFEST = "max-age=30"

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("s1")


def utcnow() -> dt.datetime:
    return dt.datetime.now(UTC)


def iso_z(d: dt.datetime) -> str:
    return d.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# R2 client (verbatim shape from meso_poller.R2 -- the proven S3-to-R2 wrapper)
# ---------------------------------------------------------------------------
class R2:
    def __init__(self) -> None:
        self.s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
        )

    def put_bytes(self, key: str, data: bytes, content_type: str, cache: str) -> bool:
        try:
            self.s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data,
                               ContentType=content_type, CacheControl=cache)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("R2 put %s failed: %s", key, e)
            return False

    def put_json(self, key: str, obj: dict, cache: str) -> bool:
        return self.put_bytes(key, json.dumps(obj, separators=(",", ":")).encode(),
                              "application/json", cache)

    def head(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=R2_BUCKET, Key=key)
            return True
        except Exception:  # noqa: BLE001 - NoSuchKey / 404 -> False
            return False

    def delete(self, keys) -> None:
        keys = [k for k in keys if k]
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            try:
                self.s3.delete_objects(Bucket=R2_BUCKET,
                                       Delete={"Objects": [{"Key": k} for k in batch]})
            except Exception as e:  # noqa: BLE001
                log.warning("R2 delete batch failed: %s", e)

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(
                Bucket=R2_BUCKET, Prefix=prefix):
            keys.extend(o["Key"] for o in page.get("Contents", []))
        return keys


# ---------------------------------------------------------------------------
# SQS wrapper
# ---------------------------------------------------------------------------
class SQS:
    def __init__(self, queue_url: str) -> None:
        self.url = queue_url
        self.c = boto3.client("sqs", region_name=AWS_REGION,
                              config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}))

    def receive(self, max_n: int, wait_s: int) -> list[dict]:
        r = self.c.receive_message(
            QueueUrl=self.url, MaxNumberOfMessages=max_n, WaitTimeSeconds=wait_s,
            AttributeNames=["ApproximateReceiveCount"])
        return r.get("Messages", [])

    def delete(self, receipt_handle: str) -> None:
        self.c.delete_message(QueueUrl=self.url, ReceiptHandle=receipt_handle)

    def visible_count(self, url: Optional[str] = None) -> Optional[int]:
        try:
            a = self.c.get_queue_attributes(
                QueueUrl=url or self.url,
                AttributeNames=["ApproximateNumberOfMessages"])
            return int(a["Attributes"]["ApproximateNumberOfMessages"])
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Render call (envelope of meso_poller.call_render, targeting a SPECIFIC slot)
# ---------------------------------------------------------------------------
class RenderError(Exception):
    pass


class RenderSkip(Exception):
    """422 coverage/off-disk/night -- expected, not an error (ack as no-data)."""


def _header(headers: dict, name: str) -> Optional[str]:
    ln = name.lower()
    return next((v for k, v in headers.items() if k.lower() == ln), None)


def call_render_slot(session: requests.Session, source: SRC.SatSource,
                     bbox: list[float], time_iso: str,
                     url: str = RENDER_URL) -> tuple[bytes, dict]:
    """POST the FROZEN /render exactly as the meso/floater lane does, but for a
    SPECIFIC slot time (not "latest"), with the source's render params (channel
    clean_ir, enhancement rainbow_ir, format webp + the source's satellite hint /
    product). Byte-identical to the prod frame for the slot. Raises RenderSkip on
    422, RenderError after retries."""
    body = SRC.render_body(source, bbox, time_iso)
    last: Optional[Exception] = None
    for attempt in range(RENDER_MAX_RETRIES):
        try:
            r = session.post(url, json=body, timeout=RENDER_TIMEOUT_S)
            if r.status_code == 422:
                raise RenderSkip(r.text[:200])
            if r.status_code == 429:
                time.sleep(6 + attempt * 4)
                last = RenderError("429 rate limited")
                continue
            r.raise_for_status()
            return r.content, dict(r.headers)
        except RenderSkip:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep((2 ** attempt) + 0.3)
    raise RenderError(str(last))


# ---------------------------------------------------------------------------
# Real extent reader + ground-truth lister (lazy heavy imports; injectable so
# the worker's never-miss logic is unit-tested with stubs -- no xarray/s3fs).
# ---------------------------------------------------------------------------
def read_object_extent(s3_key: str, bucket: str = S.S1_BUCKET) -> list[float]:
    """Read a CMIPM object's live geographic extent -> [lon_w, lat_s, lon_e,
    lat_n] (rounded 3dp, lon normalized), the SAME bbox the meso poller derives
    (meso_poller.discover_goes_extent / satellites._goes_meso_extent_from_ds), so
    the render crop is identical. Opens the object directly by key (no listing),
    attrs-only (h5netcdf, no array load). ABI sources only (the GOES meso extent
    is per-scan discovered); AHI FLDK uses a fixed extent, not this reader."""
    import xarray as xr
    import satellites
    fs = satellites._get_fs()
    with fs.open(f"{bucket}/{s3_key}" if not s3_key.startswith(bucket)
                 else s3_key, mode="rb") as f:
        ds = xr.open_dataset(f, decode_cf=False, engine="h5netcdf")
        try:
            lon_w, lat_s, lon_e, lat_n = satellites._goes_meso_extent_from_ds(ds)
        finally:
            ds.close()

    def norm_lon(lon: float) -> float:
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
        return lon
    return [round(norm_lon(lon_w), 3), round(lat_s, 3),
            round(norm_lon(lon_e), 3), round(lat_n, 3)]


def list_ground_truth(source: SRC.SatSource, lookback_min: int) -> list[S.Slot]:
    """List the recent NOAA prefixes for ``source`` and return its COMPLETE slots
    over the lookback window, NEWEST-FIRST (§3.3 H). For AHI this groups segments
    and returns only fully-segmented scans (never a half-scan -- the
    southern-segment lesson). Anonymous LIST on the NODD bucket is free; uses the
    SAME s3fs handle/tuning the render service uses."""
    import satellites
    fs = satellites._get_fs()
    now = utcnow()
    keys: list[str] = []
    for prefix in SRC.noaa_prefixes(source, now, lookback_min):
        try:
            listing = fs.ls(f"{source.bucket}/{prefix}")
        except (FileNotFoundError, OSError):
            continue
        for full in listing:
            # s3fs keys are "bucket/Key"; strip the bucket to match SNS keys.
            keys.append(full.split("/", 1)[1] if "/" in full else full)
    complete = SRC.complete_scans(source, keys)   # {stamp: Slot}, all-segments
    cutoff = now - dt.timedelta(minutes=lookback_min)
    out = [s for s in complete.values() if s.scan_start >= cutoff]
    out.sort(key=lambda s: s.scan_start, reverse=True)  # newest-first
    return out


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Stats:
    received: int = 0
    rendered: int = 0
    duplicates: int = 0
    not_ours: int = 0
    no_data: int = 0
    backfilled: int = 0
    render_fail: int = 0
    put_fail: int = 0


class S1Ingest:
    """Event-driven, never-miss ingest of ONE product to R2 /shadow/.

    All external I/O is injected so the never-miss control logic is unit-tested
    offline (moto SQS/S3 + stub render/extent): ``r2`` (an R2-like), ``sqs`` (an
    SQS-like), ``render_fn(bbox, time_iso) -> (bytes, headers)``, ``extent_fn(
    s3_key) -> bbox``, ``ground_truth_fn(lookback_min) -> [Slot]``, ``clock``."""

    def __init__(self, *, r2, sqs, render_fn: Callable, extent_fn: Callable,
                 ground_truth_fn: Callable, prefix: str = R2_PREFIX,
                 clock: Callable[[], dt.datetime] = utcnow,
                 source: SRC.SatSource = SOURCE) -> None:
        self.r2 = r2
        self.sqs = sqs
        self.render_fn = render_fn
        self.extent_fn = extent_fn
        self.ground_truth_fn = ground_truth_fn
        self.prefix = prefix
        self.source = source
        self._now = clock
        # The completeness gate's required-set is the source's (ABI: the single
        # native band; AHI FLDK: all 10 segments -- the southern-segment lesson).
        self.gate = S.CompletenessGate(SRC.gate_required(source))
        # The ledger of published slots (a cache; R2 is truth -- reseeded on cold
        # start). Maps stamp -> sha256 of the published frame.
        self.published: dict[str, str] = {}
        self.watermark: Optional[dt.datetime] = None  # newest published scan_start
        self.stats = Stats()
        self.health = {
            "sqs": SourceHealth(name="sqs"),
            "backfill": SourceHealth(name="backfill"),
        }
        self._last_progress = time.monotonic()
        self._dlq_visible: Optional[int] = None

    # -- cold start (§3.6) ----------------------------------------------------
    def cold_start(self) -> None:
        """Seed the ledger + watermark from R2 REALITY: list the shadow frames
        and recover their stamps. On an empty bucket the watermark stays unset
        and the first backfill seeds it from ground truth."""
        prefix = f"{self.prefix}/{self.source.product_path}/"
        try:
            keys = self.r2.list_keys(prefix)
        except Exception as e:  # noqa: BLE001
            log.warning("cold-start R2 list failed (%s); starting empty", e)
            keys = []
        n = 0
        for k in keys:
            stamp = S.stamp_from_frame_key(k)
            if stamp is None:
                continue
            self.published.setdefault(stamp, "")
            self.gate.seed_complete(stamp)
            n += 1
            scan = dt.datetime.strptime(stamp, S.STAMP_FMT).replace(tzinfo=UTC)
            if self.watermark is None or scan > self.watermark:
                self.watermark = scan
        log.info("cold start: seeded %d published slots from R2 (%s); watermark=%s",
                 n, prefix, iso_z(self.watermark) if self.watermark else "(none)")

    # -- the render path for one COMPLETE slot --------------------------------
    def process_slot(self, slot: S.Slot, source: str,
                     write_manifest: bool = True) -> str:
        """Render a complete S1 slot via the frozen path and publish to R2.
        Returns one of: rendered | duplicate | no-data. Raises on render/PUT
        failure (the caller must NOT ack so SQS redelivers -> DLQ after N)."""
        stamp = slot.stamp
        key = S.shadow_frame_key(self.prefix, stamp, self.source.product_path)
        # Stale-slot guard: a slot older than the retained window would be pruned
        # immediately anyway, so ack it without rendering. Bounds a deploy-day
        # thundering herd if the queue accreted a long backlog while the box was
        # down (the SQS 14-day retention holds events; we only catch up the last
        # RETAIN_H, which is the never-miss window -- older is intentionally gone).
        if slot.scan_start < self._now() - dt.timedelta(hours=RETAIN_H):
            self.stats.duplicates += 1   # counted as covered (acked, not rendered)
            log.info("stale slot %s older than %.0fh -- acked, not rendered",
                     SRC.slot_label(self.source, slot), RETAIN_H)
            return "stale"
        # Idempotency: already published (in-ledger) OR already in R2 -> no-op.
        if stamp in self.published or self.r2.head(key):
            self.published.setdefault(stamp, "")
            self.gate.seed_complete(stamp)
            self.stats.duplicates += 1
            return "duplicate"

        bbox = self.extent_fn(slot.s3_key)
        try:
            png, headers = self.render_fn(bbox, slot.scan_start.isoformat())
        except RenderSkip as e:
            # Off-disk / coverage / night -> a legitimate no-frame slot. Mark it
            # covered + acked so it does not wedge or DLQ; log with slot id so the
            # never-miss audit counts it as "frame OR logged no-data" (§3.3).
            self.gate.seed_complete(stamp)
            self.published.setdefault(stamp, "SKIP")
            self.stats.no_data += 1
            log.info("no-data slot %s (%s): %s",
                     SRC.slot_label(self.source, slot), source, e)
            return "no-data"

        # Verify the render resolved the SLOT WE ASKED FOR. The render service's
        # s3fs listing cache (30 s) can briefly miss a just-published object and
        # resolve a neighbour; X-Scan-Time != our stamp means "not ready" -> do
        # NOT ack, let SQS redeliver after the visibility timeout (s3fs refreshes).
        scan_hdr = _header(headers, "X-Scan-Time")
        if not scan_hdr:
            # Fail CLOSED: without the header we cannot confirm which scan the
            # render resolved, so we must not publish a possibly-wrong-slot frame.
            # Retry (->DLQ after N). The meso-branch render always emits it.
            raise RenderError("render response missing X-Scan-Time; cannot verify "
                              "slot -- will retry")
        try:
            got = dt.datetime.fromisoformat(scan_hdr.replace("Z", "+00:00"))
            got_stamp = got.astimezone(UTC).strftime(S.STAMP_FMT)
        except (ValueError, TypeError):
            raise RenderError(f"unparseable X-Scan-Time {scan_hdr!r}; will retry")
        if got_stamp != stamp:
            raise RenderError(
                f"render resolved {got_stamp} != requested {stamp} "
                f"(s3fs listing not yet current); will retry")

        digest = hashlib.sha256(png).hexdigest()
        if not self.r2.put_bytes(key, png, "image/webp", CACHE_FRAME):
            self.stats.put_fail += 1
            raise RenderError(f"R2 PUT failed for {key}")

        # PUT succeeded -> the slot is published. Update ledger + watermark + SSOT.
        self.published[stamp] = digest
        self.gate.seed_complete(stamp)
        scan = slot.scan_start
        if self.watermark is None or scan > self.watermark:
            self.watermark = scan
        self.stats.rendered += 1
        if write_manifest:
            self._write_latest_times()
        log.info("published %s (%d B, %s, src=%s)", key, len(png), digest[:10], source)
        return "rendered"

    def _write_latest_times(self) -> None:
        pp = self.source.product_path
        lt = S.build_latest_times(self.published.keys(), self.prefix, self._now(), pp)
        self.r2.put_json(S.latest_times_key(self.prefix, pp), lt, CACHE_MANIFEST)

    # -- SQS consume (primary path) ------------------------------------------
    def consume_once(self, wait_s: int = SQS_WAIT_S) -> int:
        """Receive a batch, process each message, delete ONLY after success.
        Returns the number of messages handled. Per-message isolation: one bad
        message never blocks the others or the heartbeat."""
        h = self.health["sqs"]
        h.last_attempt_utc = self._now()
        h.total_polls += 1
        try:
            msgs = self.sqs.receive(SQS_BATCH, wait_s)
        except Exception as e:  # noqa: BLE001
            h.consecutive_failures += 1
            h.total_failures += 1
            h.last_error = f"receive: {type(e).__name__}: {e}"
            log.warning("sqs receive failed: %s", h.last_error)
            return 0
        h.last_success_utc = self._now()
        h.consecutive_failures = 0
        h.last_error = None
        # A successful poll IS liveness, even when empty -- otherwise a
        # legitimately quiet firehose (upstream maintenance) would trip the
        # watchdog and restart-loop the worker (NOTE-3).
        self._last_progress = time.monotonic()
        for m in msgs:
            self.stats.received += 1
            try:
                self._handle_message(m)
            except Exception as e:  # noqa: BLE001 - per-message isolation: one
                # malformed message must never drop the rest of the batch
                # unacked. Not deleted -> redelivers -> DLQ after maxReceiveCount.
                self.health["sqs"].last_error = f"handle: {type(e).__name__}: {e}"
                log.warning("message handling error (left for redelivery): %s", e)
        return len(msgs)

    def _handle_message(self, m: dict) -> None:
        body = m.get("Body", "")
        handle = m.get("ReceiptHandle")
        recv_count = int(m.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
        key = S.extract_object_key(body)
        slot = SRC.parse(self.source, key) if key else None

        # Not our product (parsed-key filter, INDEPENDENT of the SNS filter):
        # ack it (it is not ours and never will be) and move on.
        if not SRC.is_ours(self.source, slot):
            self.stats.not_ours += 1
            if handle:
                self._safe_delete(handle)
            log.debug("drop non-%s message key=%s", self.source.key, key)
            return

        # Surface a slot climbing toward the DLQ BEFORE it gets there (§3.5).
        if recv_count >= RECEIVE_COUNT_WARN:
            log.warning("slot %s nearing DLQ (ApproximateReceiveCount=%d)",
                        SRC.slot_label(self.source, slot), recv_count)

        # Completeness gate, keyed by slot.stamp (the SAME key process_slot/
        # cold_start/prune use). ABI: 1 band -> the C13 object completes the slot
        # immediately. AHI FLDK: 10 segments -> complete ONLY when the last
        # segment lands (the southern-segment lesson -- never a half-scan). An
        # incomplete segment is acked (it landed); a restart gap or an SNS-dropped
        # segment is re-detected by the backfill, which renders a scan only once
        # all its segments are on NOAA.
        self.gate.mark(SRC.gate_key(slot), SRC.gate_item(self.source, slot))
        if not self.gate.is_complete(SRC.gate_key(slot)):
            if handle:
                self._safe_delete(handle)
            return
        try:
            outcome = self.process_slot(slot, source="sqs")
        except RenderSkip:
            outcome = "no-data"   # already handled inside process_slot
        except Exception as e:  # noqa: BLE001 - render/PUT failure
            self.stats.render_fail += 1
            self.health["sqs"].last_error = f"process: {type(e).__name__}: {e}"
            log.warning("process slot %s FAILED (recv=%d, will redeliver->DLQ@5): %s",
                        SRC.slot_label(self.source, slot), recv_count, e)
            return  # do NOT delete -> SQS redelivers -> DLQ after maxReceiveCount
        # rendered / duplicate / no-data / stale -> delete (PUT done or N/A).
        if handle:
            self._safe_delete(handle)
        if outcome in ("rendered", "duplicate", "no-data", "stale"):
            self.health["sqs"].last_change_utc = self._now()

    def _safe_delete(self, handle: str) -> None:
        try:
            self.sqs.delete(handle)
        except Exception as e:  # noqa: BLE001
            log.warning("sqs delete failed (msg will redeliver): %s", e)

    # -- backfill (the never-miss fallback authority, §3.3) -------------------
    def backfill_once(self) -> int:
        """ListObjectsV2-reconcile the recent NOAA prefix, NEWEST-FIRST, and
        render any complete slot newer than the watermark that SQS never
        delivered. Returns the number of slots backfilled. A list failure
        preserves the watermark (never blanks state) and flags the source."""
        h = self.health["backfill"]
        h.last_attempt_utc = self._now()
        h.total_polls += 1
        try:
            slots = self.ground_truth_fn(BACKFILL_LOOKBACK_MIN)  # newest-first
        except Exception as e:  # noqa: BLE001
            h.consecutive_failures += 1
            h.total_failures += 1
            h.last_error = f"list: {type(e).__name__}: {e}"
            log.warning("backfill list failed: %s", h.last_error)
            return 0
        h.last_success_utc = self._now()
        h.consecutive_failures = 0
        h.last_error = None
        n = 0
        newest = self.watermark
        for slot in slots:  # newest-first
            if slot.stamp in self.published:
                continue
            try:
                # Defer the manifest write -- a cold-start burst would otherwise
                # re-PUT latest_times once per slot (NOTE-4); write it once below.
                outcome = self.process_slot(slot, source="backfill",
                                            write_manifest=False)
            except RenderSkip:
                outcome = "no-data"
            except Exception as e:  # noqa: BLE001
                self.stats.render_fail += 1
                log.warning("backfill render %s FAILED: %s",
                            SRC.slot_label(self.source, slot), e)
                continue
            if outcome == "rendered":
                self.stats.backfilled += 1
                n += 1
                log.info("backfilled slot %s (SQS never delivered)",
                         SRC.slot_label(self.source, slot))
            if newest is None or slot.scan_start > newest:
                newest = slot.scan_start
        if newest is not None:
            self.watermark = newest
            h.last_valid_time = newest
        if n:
            self._write_latest_times()   # once per burst (NOTE-4)
            self._last_progress = time.monotonic()
        return n

    # -- retention prune ------------------------------------------------------
    def prune_once(self) -> int:
        """Drop shadow frames older than RETAIN_H + trim the ledger + rebuild
        latest_times. Keeps the rolling audit window bounded."""
        cutoff = self._now() - dt.timedelta(hours=RETAIN_H)
        dead_stamps = [s for s in self.published
                       if dt.datetime.strptime(s, S.STAMP_FMT).replace(tzinfo=UTC) < cutoff]
        if not dead_stamps:
            return 0
        self.r2.delete([S.shadow_frame_key(self.prefix, s, self.source.product_path)
                        for s in dead_stamps])
        for s in dead_stamps:
            self.published.pop(s, None)
            self.gate.forget(s)
        self._write_latest_times()
        log.info("pruned %d shadow frames older than %.0fh", len(dead_stamps), RETAIN_H)
        return len(dead_stamps)

    # -- health (§3.5) --------------------------------------------------------
    def refresh_dlq_depth(self) -> None:
        if S1_DLQ_URL:
            self._dlq_visible = self.sqs.visible_count(S1_DLQ_URL)

    def health_snapshot(self) -> dict:
        now = self._now()
        sources = {n: hh.snapshot(now, STALE_AFTER_S, FAIL_THRESHOLD)
                   for n, hh in self.health.items()}
        healthy = all(s["state"] == FRESH for s in sources.values())
        return {
            "poller": "s1-ingest",
            "enabled": S1_ENABLED,
            "source": self.source.key,
            "product": self.source.product_path,
            "prefix": self.prefix,
            "generated_utc": iso_z(now),
            "stale_after_s": STALE_AFTER_S,
            "healthy": healthy,
            "watermark": iso_z(self.watermark) if self.watermark else None,
            "published_slots": len(self.published),
            "dlq_visible": self._dlq_visible,
            "stats": dataclasses.asdict(self.stats),
            "process": process_mem_mb(),
            "sources": sources,
        }


# ---------------------------------------------------------------------------
# Health HTTP server (compose healthcheck curls GET /health) + liveness watchdog
# ---------------------------------------------------------------------------
_HEALTH_STATE: dict = {"snapshot": {"poller": "s1-ingest", "healthy": None,
                                    "enabled": S1_ENABLED}}


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") not in ("/health", ""):
            self.send_response(404); self.end_headers(); return
        snap = _HEALTH_STATE.get("snapshot") or {}
        ok = snap.get("healthy") is not False
        body = json.dumps(snap, separators=(",", ":")).encode()
        self.send_response(200 if ok else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


def _start_health_server() -> None:
    try:
        srv = http.server.ThreadingHTTPServer(("0.0.0.0", HEALTH_PORT), _HealthHandler)
    except OSError as e:
        log.warning("health server bind :%d failed: %s", HEALTH_PORT, e)
        return
    threading.Thread(target=srv.serve_forever, name="s1-health", daemon=True).start()
    log.info("health server on :%d (GET /health)", HEALTH_PORT)


def _emit_health(worker: S1Ingest) -> None:
    try:
        worker.refresh_dlq_depth()
    except Exception:  # noqa: BLE001
        pass
    snap = worker.health_snapshot()
    _HEALTH_STATE["snapshot"] = snap
    try:
        with open(HEALTH_FILE, "w", encoding="utf-8") as f:
            json.dump(snap, f, separators=(",", ":"))
    except OSError as e:
        log.warning("health file write failed: %s", e)
    try:
        worker.r2.put_json(
            S.health_key(worker.prefix, worker.source.product_path),
            snap, CACHE_MANIFEST)
    except Exception as e:  # noqa: BLE001
        log.warning("health R2 write failed: %s", e)


def _watchdog(worker: S1Ingest) -> None:
    """Self-exit if a cycle makes no progress for WATCHDOG_S (a wedged render /
    hung HTTP). compose restart:always + systemd Restart=always recover it -- a
    process restart, the HAFS lesson. R2 + the SQS queue survive, so cold-start +
    backfill self-heal any gap on restart."""
    while True:
        time.sleep(min(WATCHDOG_S / 3.0, 60.0))
        idle = time.monotonic() - worker._last_progress
        if idle > WATCHDOG_S:
            log.error("WATCHDOG: no progress for %.0fs (>%.0fs) -- self-exiting "
                      "for restart", idle, WATCHDOG_S)
            os._exit(1)


def _build_worker() -> S1Ingest:
    session = requests.Session()
    session.headers["User-Agent"] = f"tat-s1-ingest/1.0 ({SOURCE.key})"
    src = SOURCE
    # Extent: ABI discovers each scan's geo extent from the object; AHI FLDK uses
    # a fixed WPAC extent (the full disk doesn't move).
    if src.extent_mode == "fixed":
        fixed = list(src.fixed_bbox)
        extent_fn = lambda s3_key: list(fixed)            # noqa: E731
    else:
        extent_fn = lambda s3_key: read_object_extent(s3_key, src.bucket)  # noqa: E731
    return S1Ingest(
        r2=R2(),
        sqs=SQS(S1_QUEUE_URL),
        render_fn=lambda bbox, time_iso: call_render_slot(session, src, bbox, time_iso),
        extent_fn=extent_fn,
        ground_truth_fn=lambda lookback: list_ground_truth(src, lookback),
        source=src,
    )


def run() -> None:
    if not S1_ENABLED:
        log.warning("S1_ENABLED=false -- idle, writing nothing. Heartbeat only.")
        while True:
            _HEALTH_STATE["snapshot"] = {"poller": "s1-ingest", "enabled": False,
                                         "healthy": None}
            time.sleep(60)
    missing = [n for n in ("S1_QUEUE_URL", "R2_ENDPOINT") if not _env(n)]
    if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        missing.append("R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY")
    if missing:
        log.error("missing required env: %s", ", ".join(missing))
        raise SystemExit(1)

    worker = _build_worker()
    log.info("s1 ingest starting | source=%s | queue=%s | render=%s | "
             "bucket=%s prefix=%s | product=%s | completeness=%d", SOURCE.key,
             S1_QUEUE_URL, RENDER_URL, SOURCE.bucket, R2_PREFIX,
             SOURCE.product_path, SOURCE.required_segments)
    worker.cold_start()
    threading.Thread(target=_watchdog, args=(worker,), name="s1-watchdog",
                     daemon=True).start()
    last_backfill = 0.0
    last_prune = 0.0
    last_health = 0.0
    while True:
        try:
            worker.consume_once()              # SQS long-poll (primary)
            mono = time.monotonic()
            if mono - last_backfill >= BACKFILL_INTERVAL_S:
                worker.backfill_once()          # fallback authority
                last_backfill = mono
            if mono - last_prune >= 1800.0:
                worker.prune_once()
                last_prune = mono
            if mono - last_health >= 30.0:
                _emit_health(worker)
                last_health = mono
        except Exception as e:  # noqa: BLE001 - the loop must never die
            log.exception("loop error (continuing): %s", e)
            time.sleep(5)


if __name__ == "__main__":
    _start_health_server()
    run()
