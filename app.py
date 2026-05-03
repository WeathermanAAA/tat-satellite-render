"""FastAPI service: GOES-16 custom-zoom render endpoint.

POST /render -> image/png
GET  /health -> {status, goes_bucket_reachable, cache_size}

Rate limit: 10 req/min/IP on /render (cache hits exempt).
Concurrency: max 2 simultaneous renders via asyncio.Semaphore.
CORS: origins from ALLOWED_ORIGINS env var.

NOTE: do not enable `from __future__ import annotations` here — slowapi's
decorator + FastAPI body-param inference + pydantic 2 forward-ref resolution
will fail if RenderRequest is a stringified annotation.
"""

import asyncio
import gc
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from cache import RenderCache
from goes import (
    GOES_BUCKET_OVERRIDE,
    GOES_DISK_BBOX,
    PRIMARY_LIVE_BUCKET,
    bucket_reachable,
    fetch_data,
    pick_buckets_for_time,
    resolve_request,
)
from render import render_png


# ---------------------------------------------------------------------------
# Logging — JSON-ish single-line format
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
log = logging.getLogger("tat-satellite")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://localhost:5500,http://127.0.0.1:8000",
    ).split(",")
    if o.strip()
]
MAX_CONCURRENT_RENDERS = int(os.getenv("MAX_CONCURRENT_RENDERS", "2"))
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
LATEST_CACHE_TTL = float(os.getenv("LATEST_CACHE_TTL", "300"))  # 5 min

render_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RENDERS)
cache = RenderCache(max_entries=200, max_bytes=100 * 1024 * 1024)


# ---------------------------------------------------------------------------
# slowapi keyed on real client IP (X-Forwarded-For first hop)
# ---------------------------------------------------------------------------
def real_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=real_ip)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "startup origins=%s max_concurrent=%d rate=%s primary=%s override=%s",
        ALLOWED_ORIGINS,
        MAX_CONCURRENT_RENDERS,
        RATE_LIMIT,
        PRIMARY_LIVE_BUCKET,
        GOES_BUCKET_OVERRIDE or "(none, time-based picker active)",
    )
    yield
    log.info("shutdown")


app = FastAPI(title="tat-satellite-render", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Cache", "X-Render-Ms", "X-Product", "X-Bucket"],
    allow_credentials=False,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RenderRequest(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    time: str = "latest"
    channel: int
    enhancement: str = "tat_neon"

    @field_validator("bbox")
    @classmethod
    def _v_bbox(cls, v):
        lon_min, lat_min, lon_max, lat_max = v
        if not (-180 <= lon_min < lon_max <= 180):
            raise ValueError("invalid longitude range")
        if not (-90 <= lat_min < lat_max <= 90):
            raise ValueError("invalid latitude range")
        if (lon_max - lon_min) > 30 or (lat_max - lat_min) > 30:
            raise ValueError("bbox max size 30 degrees per axis")
        gd_lon_min, gd_lat_min, gd_lon_max, gd_lat_max = GOES_DISK_BBOX
        # Must overlap GOES-16 disk
        if (
            lon_max < gd_lon_min
            or lon_min > gd_lon_max
            or lat_max < gd_lat_min
            or lat_min > gd_lat_max
        ):
            raise ValueError("bbox does not overlap GOES-16 visible disk")
        return v

    @field_validator("channel")
    @classmethod
    def _v_channel(cls, v):
        if v not in (2, 8, 13):
            raise ValueError("channel must be 2 (visible), 8 (upper WV), or 13 (clean IR)")
        return v

    @field_validator("enhancement")
    @classmethod
    def _v_enh(cls, v):
        if v not in ("tat_neon", "dvorak_bd", "grayscale"):
            raise ValueError("enhancement must be tat_neon | dvorak_bd | grayscale")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    reachable = await bucket_reachable()
    return {
        "status": "ok",
        # Live/primary bucket — what 'latest' renders use, what /health probes.
        "goes_bucket": GOES_BUCKET_OVERRIDE or PRIMARY_LIVE_BUCKET,
        "goes_bucket_reachable": reachable,
        # Buckets currently in rotation for "latest" — gives ops a quick view
        # of whether time-based picking is active vs forced-override.
        "buckets_for_latest": pick_buckets_for_time("latest"),
        "override_active": bool(GOES_BUCKET_OVERRIDE),
        "cache_entries": len(cache),
        "cache_bytes": cache.size_bytes,
    }


def _request_key(body: RenderRequest, snapped_iso: str, bucket: str) -> str:
    # Bucket included so an algorithm tweak (e.g. moving the goes19/goes16
    # boundary date) cleanly invalidates entries that would now resolve
    # differently. Same scan_start in goes19 and goes16 still produces
    # different rendered output (different sat-label, possibly different
    # calibration), so they must cache separately.
    raw = f"{body.bbox}|{snapped_iso}|{body.channel}|{body.enhancement}|{bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()


@app.post("/render")
@limiter.limit(RATE_LIMIT)
async def render(request: Request, body: RenderRequest = Body(...)):
    t0 = time.perf_counter()
    is_latest = body.time == "latest"

    # Resolve which file we'll use; this gives us the snapped scan time which
    # is part of the cache key. For "latest" we don't snap to file precision
    # because that would produce a unique key every minute — instead we round
    # down to LATEST_CACHE_TTL buckets so concurrent "latest" renders share.
    try:
        resolved = await resolve_request(body.bbox, body.channel, body.time)
    except Exception as e:
        log.exception("resolve failed: %s", e)
        raise HTTPException(status_code=502, detail=f"could not resolve GOES file: {e}") from e

    if is_latest:
        bucket_id = int(time.time() // LATEST_CACHE_TTL)
        snapped = f"latest@{bucket_id}@{resolved.scan_start.isoformat()}"
    else:
        snapped = resolved.scan_start.isoformat()

    cache_key = _request_key(body, snapped, resolved.bucket)
    cached = cache.get(cache_key)
    if cached is not None:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return Response(
            content=cached,
            media_type="image/png",
            headers={
                "X-Cache": "HIT",
                "X-Render-Ms": str(elapsed_ms),
                "X-Product": resolved.product,
                "X-Bucket": resolved.bucket,
            },
        )

    async with render_semaphore:
        # Re-check cache inside the semaphore in case a concurrent request
        # already produced this same image.
        cached = cache.get(cache_key)
        if cached is not None:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return Response(
                content=cached,
                media_type="image/png",
                headers={
                    "X-Cache": "HIT",
                    "X-Render-Ms": str(elapsed_ms),
                    "X-Product": resolved.product,
                    "X-Bucket": resolved.bucket,
                },
            )

        try:
            data = await fetch_data(resolved, body.bbox, body.channel)
            png_bytes = await asyncio.get_event_loop().run_in_executor(
                None,
                render_png,
                data,
                body.bbox,
                body.channel,
                resolved.scan_start.strftime("%Y-%m-%d %H:%M"),
                body.enhancement,
            )
        except Exception as e:
            log.exception("render failed: %s", e)
            raise HTTPException(status_code=500, detail=f"render failed: {e}") from e
        finally:
            gc.collect()

    ttl: Optional[float] = LATEST_CACHE_TTL if is_latest else None
    cache.put(cache_key, png_bytes, ttl_seconds=ttl)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "render ok key=%s bucket=%s product=%s ch=%d enh=%s ms=%d bytes=%d",
        cache_key[:10],
        resolved.bucket,
        resolved.product,
        body.channel,
        body.enhancement,
        elapsed_ms,
        len(png_bytes),
    )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Cache": "MISS",
            "X-Render-Ms": str(elapsed_ms),
            "X-Product": resolved.product,
            "X-Bucket": resolved.bucket,
        },
    )
