"""FastAPI service: satellite-agnostic custom-zoom render endpoint.

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
import math
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional, Union
from urllib.parse import urlparse

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from cache import RenderCache
from poller_framework import process_mem_mb
from render import (render_png, render_backdrop_webp, transcode_frame,
                    encode_webp, state_lines_status)
from satellites import (
    ALL_SATELLITES,
    CoverageError,
    GENERIC_CHANNELS,
    GOES_BUCKET_OVERRIDE,
    GOES_EAST,
    PRIMARY_LIVE_BUCKET,
    Satellite,
    UnsupportedTimeError,
    bucket_reachable,
    goes_band_to_generic,
    parse_request_time,
    pick_buckets_for_time,
    pick_satellite,
)


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
# format=webp loop frames: the 1320 px render Lanczos-downscaled to this width
# (1056 = the 525 CSS px player box at 2x Retina, and exactly 0.8 x 1320) and
# encoded lossy-WebP at this quality (q90 verified visually transparent on the
# IR/IRBD enhancements -- the banding/chroma worst cases).
WEBP_FRAME_WIDTH = int(os.getenv("WEBP_FRAME_WIDTH", "1056"))
WEBP_QUALITY = int(os.getenv("WEBP_QUALITY", "90"))

render_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RENDERS)
cache = RenderCache(max_entries=200, max_bytes=100 * 1024 * 1024)

# ---------------------------------------------------------------------------
# Server-side loop export (POST /export) -- ffmpeg-encoded mp4 (primary) + a
# global-palette gif fallback. Bounded + rate-limited; the webp LOOP poller path
# and live /render are untouched (this is export-only).
# ---------------------------------------------------------------------------
EXPORT_MAX_FRAMES = int(os.getenv("EXPORT_MAX_FRAMES", "300"))   # hard cap on frames
EXPORT_MAX_DIM = int(os.getenv("EXPORT_MAX_DIM", "2048"))        # reject a frame longer than this
EXPORT_MAX_CONCURRENT = int(os.getenv("EXPORT_MAX_CONCURRENT", "1"))  # ffmpeg is CPU-heavy
EXPORT_DWELL_FRAMES = float(os.getenv("EXPORT_DWELL_FRAMES", "6"))    # hold last frame Nx (mirrors SAT_LAST_FRAME_DWELL)
EXPORT_FETCH_TIMEOUT_S = float(os.getenv("EXPORT_FETCH_TIMEOUT_S", "20"))
EXPORT_MAX_FRAME_BYTES = int(os.getenv("EXPORT_MAX_FRAME_BYTES", str(25 * 1024 * 1024)))  # per-frame wire cap
EXPORT_FFMPEG_TIMEOUT_S = float(os.getenv("EXPORT_FFMPEG_TIMEOUT_S", "150"))
# SSRF guard: only fetch frame URLs from these hosts (the public CDN the loops
# are served from). Comma-separated env override.
EXPORT_ALLOWED_HOSTS = {
    h.strip().lower() for h in os.getenv(
        "EXPORT_ALLOWED_HOSTS", "cdn.triple-a-tropics.com").split(",") if h.strip()
}
export_semaphore = asyncio.Semaphore(EXPORT_MAX_CONCURRENT)


def _ffmpeg_available() -> bool:
    """True if the ffmpeg binary is on PATH + runnable. Checked on boot + gated
    on each /export so a missing binary returns a clean 503 (frontend falls back
    to its client-side gif.js) rather than a 500."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10, check=True)
        return True
    except Exception:  # noqa: BLE001
        return False


FFMPEG_AVAILABLE = _ffmpeg_available()


# ---------------------------------------------------------------------------
# Pixel budget — replaces the old hard 30° edge cap. Any bbox is accepted; we
# auto-downsample at fetch/render time to keep output ≤ PIXEL_BUDGET.
# Native instrument resolution is read off the GENERIC_CHANNELS table so
# future sats with different native resolutions (e.g. AHI 0.5/1.0/2.0 km
# bands) get correct downsample factors automatically.
# Budget 16M px ≈ 4000×4000 — covers the figsize headroom and matches the
# 12 in × 12 in @ 110 dpi figure (~1320×1320 px) with comfortable Nyquist.
# Rate limit (10/min/IP) remains the real safety net against repeated
# full-disk pulls; this just keeps a single render bounded.
# ---------------------------------------------------------------------------
PIXEL_BUDGET = 16_000_000
DEG_TO_KM = 111.0  # flat conversion; geos pixel stretch at high lat partially cancels lon shrink

# Custom-zoom (draw-a-box) output RESOLUTION TIERS. Each tier is the pixel budget
# the downsample factor targets, so the output dimension scales with it (a bbox is
# never upscaled — a small zoom already under the budget renders native). "low"
# additionally re-encodes the already-small render as lossy WebP for the smallest /
# fastest download; "default" and "high" stay lossless PNG so text + coastlines stay
# crisp. The webp LOOP path (format=webp, the floater/meso pollers) is unaffected —
# it always uses the full PIXEL_BUDGET + the fixed WEBP_FRAME_WIDTH frame.
QUALITY_BUDGETS = {
    "low": 250_000,         # coarse satellite DATA — decimates hard, fastest
    "default": 2_250_000,   # balanced (the page default) — UNCHANGED
    "high": PIXEL_BUDGET,   # ~native data — genuinely sharper than default
}
DEFAULT_QUALITY = "default"
# The tier knob is the OUTPUT DPI (figsize is held at 12in in render_png, so
# layout proportions + font sizes scale uniformly and ALL chrome renders crisp
# at the tier dpi -- never bitmap-resized). default 110 == today (byte-identical);
# the webp LOOP path always renders at 110 (then transcodes to WEBP_FRAME_WIDTH).
TIER_DPI = {
    "low": 70,        # ~840 px  — small + fast, coarse imagery, CRISP chrome
    "default": 110,   # ~1320 px — the page default (unchanged)
    "high": 200,      # ~2400 px — sharper, slower (expected)
}
DEFAULT_DPI = 110
# "low" re-encodes its already-small native render as lossy WebP for the smallest
# download. NO downscale (that is what used to pixelate the chrome) -- the figure
# is rendered small at the low dpi, then encoded WebP at its native size.
LOWRES_WEBP_QUALITY = 88    # lossy-WebP quality for the "low" tier (native size)


def _native_km_per_pixel_generic(generic_channel: str) -> float:
    spec = GENERIC_CHANNELS.get(generic_channel)
    if spec is None:
        return 2.0
    return float(spec["native_km"])


def _native_km_per_pixel(channel) -> float:
    """Accepts either a generic channel name or a numeric (legacy) GOES band."""
    if isinstance(channel, int):
        generic = goes_band_to_generic(channel)
        return _native_km_per_pixel_generic(generic) if generic else 2.0
    return _native_km_per_pixel_generic(channel)


def compute_downsample_factor(bbox: list[float], channel, budget: int = PIXEL_BUDGET) -> int:
    """Integer factor N such that requested_pixels / N**2 <= ``budget``.

    Returns 1 when the request already fits. Caller passes N to render_png
    which strides the cmi/lats/lons arrays by [::N, ::N] before pcolormesh.
    ``budget`` defaults to the full PIXEL_BUDGET; the custom-zoom resolution
    tiers pass a smaller budget (QUALITY_BUDGETS) to cap the output dimension.
    """
    lon_w_deg = bbox[2] - bbox[0]
    lat_h_deg = bbox[3] - bbox[1]
    km_per_px = _native_km_per_pixel(channel)
    px_w = (lon_w_deg * DEG_TO_KM) / km_per_px
    px_h = (lat_h_deg * DEG_TO_KM) / km_per_px
    requested = px_w * px_h
    if requested <= budget:
        return 1
    return math.ceil(math.sqrt(requested / budget))


def pick_tier_dpi(fmt: str, quality: str) -> int:
    """Output DPI for a render. The webp LOOP path (pollers) always renders at the
    default 110 (then transcodes to WEBP_FRAME_WIDTH), so loop frames are
    unaffected by the tier; the custom-zoom png path honors the tier dpi. Unknown
    tiers fall back to the default."""
    return DEFAULT_DPI if fmt == "webp" else TIER_DPI.get(quality, DEFAULT_DPI)


def resolve_quality(fmt: str, quality: str) -> tuple[int, str]:
    """Map (output format, resolution tier) -> (pixel_budget, output_format).

    The webp LOOP path (format=="webp", the floater/meso pollers) is unchanged:
    full budget + a webp frame. The png/custom-zoom path honors the tier — low/
    default/high budgets, with "low" re-encoded as lossy WebP for the smallest,
    fastest download. Unknown tiers fall back to the default.
    """
    if fmt == "webp":
        return PIXEL_BUDGET, "webp"
    q = quality if quality in QUALITY_BUDGETS else DEFAULT_QUALITY
    return QUALITY_BUDGETS[q], ("webp" if q == "low" else "png")


# ---------------------------------------------------------------------------
# Channel normalization
# ---------------------------------------------------------------------------
# Accepted enhancements. rainbow_ir is the TAT default; wv_tat is the
# water-vapor table; ir_gray is the standard grayscale ("grayscale" is kept
# as a back-compat alias used by the floater poller + legacy share-links).
_ENHANCEMENTS = (
    "rainbow_ir", "dvorak", "tat_neon", "wv_tat", "ir_gray",
    "grayscale", "dvorak_bd",   # hidden back-compat aliases
)


def normalize_channel(raw) -> tuple[str, bool]:
    """Coerce a /render channel input to (generic_name, was_numeric).

    Accepts:
      - str generic name ("clean_ir")              -> ("clean_ir", False)
      - int numeric band (13)                      -> ("clean_ir", True)
      - str numeric ("13")                         -> ("clean_ir", True)

    Raises ValueError on unknown input.
    """
    if isinstance(raw, bool):  # bool is an int subclass — guard explicitly
        raise ValueError("channel must be a generic name (e.g. clean_ir) or a numeric band")
    # true_color is a multi-band PRODUCT, not a single channel; it routes to
    # the RGB compositor in /render rather than the single-band fetch path.
    if raw == "true_color":
        return "true_color", False
    if isinstance(raw, int):
        generic = goes_band_to_generic(raw)
        if generic is None:
            raise ValueError(f"unknown numeric channel: {raw}")
        return generic, True
    if isinstance(raw, str):
        try:
            i = int(raw)
        except ValueError:
            if raw not in GENERIC_CHANNELS:
                raise ValueError(f"unknown channel: {raw!r}")
            return raw, False
        generic = goes_band_to_generic(i)
        if generic is None:
            raise ValueError(f"unknown numeric channel: {i}")
        return generic, True
    raise ValueError("channel must be a string or integer")


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
        "startup origins=%s max_concurrent=%d rate=%s primary=%s override=%s satellites=%s",
        ALLOWED_ORIGINS,
        MAX_CONCURRENT_RENDERS,
        RATE_LIMIT,
        PRIMARY_LIVE_BUCKET,
        GOES_BUCKET_OVERRIDE or "(none, time-based picker active)",
        [s.family for s in ALL_SATELLITES],
    )
    if FFMPEG_AVAILABLE:
        try:
            ver = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                 text=True, timeout=10).stdout.splitlines()[0]
        except Exception:  # noqa: BLE001
            ver = "ffmpeg (version probe failed)"
        log.info("export: ffmpeg OK -> %s", ver)
    else:
        log.warning("export: ffmpeg NOT available -- POST /export will 503 "
                    "(frontend falls back to client-side gif.js)")
    yield
    log.info("shutdown")


app = FastAPI(title="tat-satellite-render", version="0.2.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=[
        "X-Cache",
        "X-Render-Ms",
        "X-Product",
        "X-Bucket",
        "X-Downsample",
        "X-Satellite",
        "X-Satellite-Family",
        "X-Sensor",
        "X-Sub-Sat-Lon",
        "X-Generic-Channel",
        "X-Native-Band",
        "X-Deprecated-Channel-API",
    ],
    allow_credentials=False,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class StormInfo(BaseModel):
    """Optional storm context for the rendered title strip.

    Supplied by the floater poller so the rendered PNG itself carries the
    storm's identity + current intensity (a color-coded badge top-left).
    All fields optional so legacy /satellite/ draw-a-box requests still
    render the plain title.
    """
    name: str = Field(..., max_length=40)
    basin: Optional[str] = Field(default=None, max_length=8)     # "WP", "AL", "EP"
    nature: Optional[str] = Field(default=None, max_length=8)    # "TS", "TD", "HU", "EX", ...
    wind_kt: Optional[float] = Field(default=None, ge=0, le=250)
    pressure_mb: Optional[float] = Field(default=None, ge=850, le=1050)


class RenderRequest(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    time: str = "latest"
    # Accept either generic name (preferred) or numeric (back-compat). The
    # field validator only checks structural validity here; the /render
    # endpoint computes the (generic, was_numeric) pair via normalize_channel.
    channel: Union[int, str]
    enhancement: str = "rainbow_ir"
    # Optional storm context. When supplied, render.py draws a color-coded
    # intensity badge (name · category · wind · pressure) on the title strip.
    storm: Optional[StormInfo] = None
    # Output codec. "png" (default) is the full-resolution lossless render --
    # the draw-a-box panel and every legacy caller keep getting exactly what
    # they got. "webp" is the loop-frame path used by the floater + meso
    # pollers: WEBP_FRAME_WIDTH px, lossy WebP (see transcode_frame).
    format: str = "png"
    # Custom-zoom RESOLUTION tier (png path only) -- the tier is the OUTPUT DPI at
    # a fixed 12in figure: "low" (~840px, lossy WebP), "default" (~1320px PNG),
    # "high" (~2400px PNG). Ignored on the webp loop path. Unknown values coerce to
    # "default" so legacy/odd callers stay safe.
    quality: str = "default"
    # Map-overlay toggles (custom-zoom). Default True = the existing look (bold
    # coastlines + political borders, labeled lat/lon gridlines); set False for
    # clean imagery with no overlay. Legacy callers + the pollers omit them ->
    # default True -> unchanged.
    coastlines: bool = True   # coastlines + political borders
    gridlines: bool = True    # labeled lat/lon graticule
    # PART 4 - clean Clean-IR BACKDROP for the ASCAT viewer: a bare GRAYSCALE
    # raster (no chrome) georeferenced to bbox, returned as WebP. Overrides the
    # normal render path; the enhancement is forced to grayscale. Additive +
    # opt-in: every existing caller omits it -> False -> unchanged.
    backdrop: bool = False

    @field_validator("quality")
    @classmethod
    def _v_quality(cls, v):
        v = (v or "default").strip().lower()
        return v if v in ("low", "default", "high") else "default"

    @field_validator("bbox")
    @classmethod
    def _v_bbox(cls, v):
        # Edge cap removed — pixel budget (PIXEL_BUDGET / compute_downsample_factor)
        # handles oversized bboxes by auto-downsampling rather than rejecting.
        # Disk-overlap check moved to pick_satellite() — out-of-coverage bboxes
        # surface as a 422 CoverageError that names the missing region/satellite.
        lon_min, lat_min, lon_max, lat_max = v
        if not (-180 <= lon_min < lon_max <= 180):
            raise ValueError("invalid longitude range")
        if not (-90 <= lat_min < lat_max <= 90):
            raise ValueError("invalid latitude range")
        return v

    @field_validator("channel")
    @classmethod
    def _v_channel(cls, v):
        try:
            normalize_channel(v)
        except ValueError as e:
            raise ValueError(str(e)) from None
        return v

    @field_validator("enhancement")
    @classmethod
    def _v_enh(cls, v):
        if v not in _ENHANCEMENTS:
            raise ValueError(
                "enhancement must be one of: " + " | ".join(_ENHANCEMENTS)
            )
        return v

    @field_validator("format")
    @classmethod
    def _v_format(cls, v):
        if v not in ("png", "webp"):
            raise ValueError("format must be png or webp")
        return v


class ExportRequest(BaseModel):
    """A server-side loop export. ``frames`` are the loop's already-rendered
    R2/CDN image URLs in play order (oldest->newest); the endpoint fetches them,
    honors ``skip`` (keep every (skip+1)th), and encodes a smooth mp4 (primary)
    or a global-palette gif. Bounded: <= EXPORT_MAX_FRAMES, each <= EXPORT_MAX_DIM."""
    frames: list[str] = Field(..., min_length=2, max_length=EXPORT_MAX_FRAMES)
    fps: int = Field(default=10, ge=1, le=30)
    skip: int = Field(default=0, ge=0, le=20)
    interpolate: bool = False
    target_fps: int = Field(default=0, ge=0, le=60)   # 0 -> 2x fps when interpolate
    format: str = "mp4"

    @field_validator("format")
    @classmethod
    def _v_export_format(cls, v):
        if v not in ("mp4", "gif"):
            raise ValueError("format must be mp4 or gif")
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
        "satellites": [
            {"family": s.family, "sensor": s.sensor} for s in ALL_SATELLITES
        ],
        "cache_entries": len(cache),
        "cache_bytes": cache.size_bytes,
        # Parent residency (MB), same shape as the pollers' heartbeat
        # "process" block (pf.process_mem_mb is stdlib-/proc-only). The
        # VPS-sizing audit had no measured number for the web service -
        # Railway bills RAM-minutes, so every service self-reports.
        "process": process_mem_mb(),
    }


def _request_key(body: RenderRequest, generic_channel: str, snapped_iso: str, bucket: str,
                 quality: str = DEFAULT_QUALITY) -> str:
    # Bucket included so an algorithm tweak (e.g. moving the goes19/goes16
    # boundary date) cleanly invalidates entries that would now resolve
    # differently. Same scan_start in goes19 and goes16 still produces
    # different rendered output (different sat-label, possibly different
    # calibration), so they must cache separately.
    #
    # Storm metadata (name + intensity) is included so the cache invalidates
    # when the storm's wind/pressure update at a new fix -- otherwise a
    # cached frame from an earlier fix could be served with stale intensity
    # burned into the title strip. None when no storm context is supplied
    # (legacy draw-a-box requests share cache as before).
    if body.storm is not None:
        storm_part = (
            f"|storm={body.storm.name},{body.storm.nature or ''},"
            f"{body.storm.wind_kt or ''},{body.storm.pressure_mb or ''}"
        )
    else:
        storm_part = ""
    # Output format joins the key ONLY when non-default so every pre-existing
    # png key stays byte-identical (cache continuity across the deploy); png
    # and webp variants of the same scan cache as distinct entries.
    fmt_part = "" if body.format == "png" else f"|fmt={body.format}"
    # Resolution tier keys the custom-zoom (png) path so the three tiers cache
    # separately; the webp LOOP path (poller frames) is unaffected -> frame-cache
    # continuity. Coastline / gridline toggles join the key ONLY when turned OFF,
    # so every default-on render keeps its existing key (cache continuity).
    q_part = "" if body.format == "webp" else f"|q={quality}"
    overlay_part = ""
    if not body.coastlines:
        overlay_part += "|nocoast"
    if not body.gridlines:
        overlay_part += "|nogrid"
    # PART 4: the bare grayscale backdrop is a DISTINCT artifact from the chromed
    # render at the same bbox/channel -> its own cache slot (never clobbers it).
    backdrop_part = "|bd" if body.backdrop else ""
    raw = (f"{body.bbox}|{snapped_iso}|{generic_channel}|{body.enhancement}"
           f"|{bucket}{storm_part}{fmt_part}{q_part}{overlay_part}{backdrop_part}")
    return hashlib.sha256(raw.encode()).hexdigest()


@app.post("/render")
@limiter.limit(RATE_LIMIT)
async def render(request: Request, body: RenderRequest = Body(...)):
    t0 = time.perf_counter()
    is_latest = body.time == "latest"

    # Channel input may be a generic name (preferred) or numeric (back-compat).
    # Validators already accepted it; here we resolve the pair and surface a
    # deprecation header on the response when numeric was used.
    generic_channel, channel_was_numeric = normalize_channel(body.channel)
    is_true_color = generic_channel == "true_color"

    # Pixel-budget downsample is deterministic from bbox+channel so the
    # cache key (already keyed on bbox+channel) implicitly covers it. We
    # surface the factor on every response (HIT and MISS) so the frontend
    # can show the "auto-downsampled Nx" badge. True color is gated by its
    # 0.5 km red band, so size it off visible_red.
    # Resolution tier -> pixel budget + output format. The webp LOOP path
    # (pollers) keeps the full budget + webp frame; the custom-zoom png path
    # honors low/default/high (low re-encodes to lossy WebP).
    budget, out_format = resolve_quality(body.format, body.quality)
    downsample = compute_downsample_factor(
        body.bbox, "visible_red" if is_true_color else generic_channel, budget
    )

    # Pick the satellite that can see this bbox at this time. CoverageError
    # surfaces as 422 with a message that names the right satellite for the
    # region (e.g. Himawari for Western Pacific).
    parsed_time, _ = parse_request_time(body.time)
    try:
        satellite: Satellite = pick_satellite(body.bbox, parsed_time)
    except (CoverageError, UnsupportedTimeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Resolve which file we'll use; this gives us the snapped scan time which
    # is part of the cache key. For "latest" we don't snap to file precision
    # because that would produce a unique key every minute — instead we round
    # down to LATEST_CACHE_TTL buckets so concurrent "latest" renders share.
    # True color resolves on its red band (pins product + scan time for all
    # RGB bands); the composite fetch reuses that resolved file.
    nearest_to_target = not is_latest
    resolve_channel = "visible_red" if is_true_color else generic_channel
    try:
        resolved = await satellite.find_file(
            parsed_time, resolve_channel, body.bbox, nearest_to_target
        )
    except Exception as e:
        log.exception("resolve failed: %s", e)
        raise HTTPException(status_code=502, detail=f"could not resolve satellite file: {e}") from e

    if is_latest:
        bucket_id = int(time.time() // LATEST_CACHE_TTL)
        snapped = f"latest@{bucket_id}@{resolved.scan_start.isoformat()}"
    else:
        snapped = resolved.scan_start.isoformat()

    cache_key = _request_key(body, generic_channel, snapped, resolved.bucket, body.quality)
    native_band = (
        satellite.truecolor_bands["red"] if is_true_color
        else satellite.generic_to_band[generic_channel]
    )

    def _response(content: bytes, cache_state: str, ms: int) -> Response:
        headers = {
            "X-Cache": cache_state,
            "X-Render-Ms": str(ms),
            "X-Product": resolved.product,
            "X-Bucket": resolved.bucket,
            "X-Downsample": str(downsample),
            "X-Satellite": resolved.sat_name,
            "X-Satellite-Family": satellite.family,
            "X-Sensor": satellite.sensor,
            "X-Sub-Sat-Lon": f"{resolved.sub_sat_lon:g}",
            "X-Generic-Channel": generic_channel,
            "X-Native-Band": str(native_band),
            # Resolution tier the frontend echoes in the result meta (and uses,
            # with the media type, to pick the .png/.webp download extension).
            "X-Quality": body.quality if body.format != "webp" else "loop",
            # Runtime diagnostic: did the vendored admin_1 state-line layer load
            # on this host? (verifies the deploy + the geojson availability)
            "X-State-Lines": state_lines_status(),
        }
        if channel_was_numeric:
            headers["X-Deprecated-Channel-API"] = "numeric"
        media_type = ("image/webp" if (out_format == "webp" or body.backdrop)
                      else "image/png")
        return Response(content=content, media_type=media_type, headers=headers)

    cached = cache.get(cache_key)
    if cached is not None:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return _response(cached, "HIT", elapsed_ms)

    async with render_semaphore:
        # Re-check cache inside the semaphore in case a concurrent request
        # already produced this same image.
        cached = cache.get(cache_key)
        if cached is not None:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return _response(cached, "HIT", elapsed_ms)

        # Output DPI is the tier knob (custom-zoom png path); the webp LOOP path
        # always renders at the default 110 dpi (then transcodes to the fixed
        # frame width), so poller frames are unaffected.
        tier_dpi = pick_tier_dpi(body.format, body.quality)
        try:
            if is_true_color:
                # Stride the bands BEFORE the per-pixel recipe (Rayleigh) inside
                # the fetch -- byte-identical to striding after, but far cheaper
                # (what makes low fast). render_png must then NOT stride again.
                data = await satellite.fetch_true_color(body.bbox, resolved, downsample)
                render_downsample = 1
            else:
                data = await satellite.fetch(resolved, body.bbox, generic_channel)
                render_downsample = downsample
            def _render_job() -> bytes:
                if body.backdrop:
                    # PART 4: bare grayscale Clean-IR backdrop for the ASCAT
                    # viewer (already WebP; skip the chromed render + transcode).
                    # Enhancement is forced to a gray palette regardless of input.
                    bd_enh = (body.enhancement
                              if body.enhancement in ("grayscale", "ir_gray")
                              else "grayscale")
                    return render_backdrop_webp(
                        data, body.bbox, enhancement=bd_enh,
                        downsample=render_downsample, dpi=tier_dpi,
                    )
                out = render_png(
                    data,
                    body.bbox,
                    native_band,
                    resolved.scan_start.strftime("%Y-%m-%d %H:%M"),
                    body.enhancement,
                    render_downsample,
                    storm=body.storm.model_dump() if body.storm is not None else None,
                    coastlines=body.coastlines,
                    gridlines=body.gridlines,
                    dpi=tier_dpi,
                )
                # webp LOOP frames (floater/meso pollers): downscale the 1320 px
                # render to the fixed frame width -- UNCHANGED. The custom-zoom
                # "low" tier instead re-encodes its already-small native render as
                # lossy WebP with NO downscale, so the chrome stays crisp.
                if body.format == "webp":
                    out = transcode_frame(out, WEBP_FRAME_WIDTH, WEBP_QUALITY)
                elif out_format == "webp":   # quality == "low"
                    out = encode_webp(out, LOWRES_WEBP_QUALITY)
                return out

            frame_bytes = await asyncio.get_event_loop().run_in_executor(
                None, _render_job
            )
        except Exception as e:
            log.exception("render failed: %s", e)
            raise HTTPException(status_code=500, detail=f"render failed: {e}") from e
        finally:
            gc.collect()

    ttl: Optional[float] = LATEST_CACHE_TTL if is_latest else None
    cache.put(cache_key, frame_bytes, ttl_seconds=ttl)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "render ok key=%s sat=%s bucket=%s product=%s gen=%s band=%d enh=%s downsample=%d fmt=%s ms=%d bytes=%d",
        cache_key[:10],
        resolved.sat_name,
        resolved.bucket,
        resolved.product,
        generic_channel,
        native_band,
        body.enhancement,
        downsample,
        body.format,
        elapsed_ms,
        len(frame_bytes),
    )
    return _response(frame_bytes, "MISS", elapsed_ms)


# ---------------------------------------------------------------------------
# Loop export (POST /export): ffmpeg mp4 (primary) + global-palette gif.
# Export-only -- the webp LOOP poller path + live /render are untouched.
# ---------------------------------------------------------------------------
def _export_vf(dwell_s: float, interpolate: bool, target_fps: int,
               for_mp4: bool) -> str:
    """The ffmpeg -vf chain shared by the mp4 + gif encoders (so the gif palette
    is generated from the SAME frames it's applied to). Order: optional motion
    interpolation -> last-frame dwell -> even-dim pad (mp4 only, for yuv420p)."""
    parts = []
    if interpolate:
        parts.append(
            f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir")
    # dwell is always present (>=0.1s) so the chain is never empty -> no leading
    # comma when spliced before palettegen/paletteuse.
    parts.append(f"tpad=stop_mode=clone:stop_duration={dwell_s:.3f}")
    if for_mp4:
        parts.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    return ",".join(parts)


def _run_ffmpeg(args: list, timeout: float) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-600:] or "ffmpeg failed"
        raise RuntimeError(f"ffmpeg: {tail}")


def _encode_export(tmp: str, fmt: str, fps: int, interpolate: bool,
                   target_fps: int) -> tuple[bytes, str]:
    """Encode the numbered PNGs in ``tmp`` (f_%05d.png) to mp4 or gif. Returns
    (bytes, media_type)."""
    pat = os.path.join(tmp, "f_%05d.png")
    dwell_s = min(1.5, max(0.1, EXPORT_DWELL_FRAMES / float(fps)))
    if fmt == "mp4":
        out = os.path.join(tmp, "out.mp4")
        vf = _export_vf(dwell_s, interpolate, target_fps, for_mp4=True)
        # libx264 -crf 18 yuv420p = broadly-playable, near-lossless, small.
        _run_ffmpeg(["-framerate", str(fps), "-i", pat, "-vf", vf,
                     "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", out], EXPORT_FFMPEG_TIMEOUT_S)
        return _read(out), "video/mp4"
    # gif: TWO-PASS GLOBAL palette so ALL frames share ONE palette (kills the
    # per-frame NeuQuant shimmer of the old client gif.js path).
    chain = _export_vf(dwell_s, interpolate, target_fps, for_mp4=False)
    palette = os.path.join(tmp, "palette.png")
    out = os.path.join(tmp, "out.gif")
    _run_ffmpeg(["-framerate", str(fps), "-i", pat, "-vf",
                 f"{chain},palettegen=stats_mode=full", palette],
                EXPORT_FFMPEG_TIMEOUT_S)
    _run_ffmpeg(["-framerate", str(fps), "-i", pat, "-i", palette, "-lavfi",
                 f"{chain}[x];[x][1:v]paletteuse=dither=sierra2_4a", out],
                EXPORT_FFMPEG_TIMEOUT_S)
    return _read(out), "image/gif"


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _fetch_frames_to_dir(urls: list, tmp: str) -> int:
    """Fetch each frame URL (host-allowlisted: SSRF guard) and normalize it to a
    same-size RGB PNG numbered f_00000.png.. so ffmpeg's %05d input + the gif
    palette stay uniform. Raises HTTPException on a disallowed host / unreadable
    or oversized frame. Returns the count written."""
    import io as _io

    import requests
    from PIL import Image

    ref = None
    n = 0
    for i, url in enumerate(urls):
        host = (urlparse(url).hostname or "").lower()
        if host not in EXPORT_ALLOWED_HOSTS:
            raise HTTPException(status_code=400,
                                detail=f"frame host not allowed: {host or '?'}")
        try:
            # allow_redirects=False: the host allowlist is checked on the URL we
            # send, so a 3xx to an off-allowlist (internal) host must NOT be
            # followed -- that would be an SSRF bypass. stream + a wire-byte cap
            # bound memory; the lazy header dims are checked BEFORE the full RGB
            # decode so a decompression bomb can't blow up RAM.
            r = requests.get(url, timeout=EXPORT_FETCH_TIMEOUT_S,
                             allow_redirects=False, stream=True)
            if 300 <= r.status_code < 400:
                raise HTTPException(status_code=502,
                                    detail=f"frame {i} redirected (not allowed)")
            r.raise_for_status()
            buf = bytearray()
            for chunk in r.iter_content(65536):
                buf.extend(chunk)
                if len(buf) > EXPORT_MAX_FRAME_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"frame {i} exceeds {EXPORT_MAX_FRAME_BYTES} bytes")
            im = Image.open(_io.BytesIO(bytes(buf)))   # lazy: dims from header
            if max(im.size) > EXPORT_MAX_DIM:
                raise HTTPException(
                    status_code=413,
                    detail=f"frame {i} {im.size[0]}x{im.size[1]} exceeds "
                           f"{EXPORT_MAX_DIM}px")
            im = im.convert("RGB")                     # decode now (dims bounded)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502,
                                detail=f"could not fetch frame {i}: {e}") from e
        if ref is None:
            ref = im.size
        elif im.size != ref:
            im = im.resize(ref, Image.LANCZOS)   # keep the ffmpeg input uniform
        im.save(os.path.join(tmp, f"f_{n:05d}.png"))
        n += 1
    return n


@app.post("/export")
@limiter.limit(RATE_LIMIT)
async def export(request: Request, body: ExportRequest = Body(...)):
    if not FFMPEG_AVAILABLE:
        raise HTTPException(status_code=503,
                            detail="export encoder (ffmpeg) unavailable")
    step = body.skip + 1
    urls = body.frames[::step]                       # honor skip: every (skip+1)th
    if len(urls) < 2:
        raise HTTPException(status_code=400, detail="need >= 2 frames after skip")
    if len(urls) > EXPORT_MAX_FRAMES:                # belt-and-suspenders bound
        urls = urls[-EXPORT_MAX_FRAMES:]
    if body.interpolate:
        target_fps = min(60, max(body.fps, body.target_fps or body.fps * 2))
    else:
        target_fps = body.fps                        # unused (no minterpolate)
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="export-")
    try:
        async with export_semaphore:
            def _job():
                _fetch_frames_to_dir(urls, tmp)
                return _encode_export(tmp, body.format, body.fps,
                                      body.interpolate, target_fps)
            data, media_type = await asyncio.get_event_loop().run_in_executor(
                None, _job)
    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="export encode timed out")
    except Exception as e:  # noqa: BLE001
        log.exception("export failed: %s", e)
        raise HTTPException(status_code=500, detail=f"export failed: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        gc.collect()
    ms = int((time.perf_counter() - t0) * 1000)
    ext = "mp4" if body.format == "mp4" else "gif"
    log.info("export ok fmt=%s frames=%d fps=%d interp=%s ms=%d bytes=%d",
             body.format, len(urls), body.fps, body.interpolate, ms, len(data))
    return Response(content=data, media_type=media_type, headers={
        "X-Export-Ms": str(ms),
        "X-Export-Frames": str(len(urls)),
        "Content-Disposition": f'attachment; filename="tat_loop.{ext}"',
    })
