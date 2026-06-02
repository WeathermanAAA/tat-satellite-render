#!/usr/bin/env python3
"""
hafs_render_poller.py
---------------------
HAFS model-plot render WORKER - the 4th Railway service (after web /render, the
floater worker, and the intensity poller). One PollerEngine with ONE HAFS Source
on poller_framework (the intensity-poller pattern - NOT the hand-rolled floater).

On a NEW complete HAFS cycle the Source's process():
  1. runs the UNCHANGED ``hafs_render`` render (``python -m
     hafs_render.generate_hafs_plots --cycle ...``) in a SUBPROCESS under a hard
     wall-clock WATCHDOG, into a temp out-dir - byte-identical to update-hafs.yml
     (same package, same registry, ALL products fully pre-rendered),
  2. uploads the frames + manifest to R2 with the cron's 3-pass order (PNGs
     no-delete -> manifest -> prune --delete scoped to *.png),
  3. writes a progress heartbeat throughout so a long (or wedged) render is
     OBSERVABLE, not silent.

Change-gated: ``change_key`` is the newest COMPLETE cycle id, so the expensive
render fires ONLY on a genuinely new cycle (the spine's cheap-when-nothing-new
guarantee). The render is the same code the cron runs, imported as the pinned
``hafs-render`` git package - one source of truth.

SHADOW MODE (Stage 2): ``HAFS_R2_PREFIX`` defaults to ``shadow/models/hafs`` so
this worker writes to a SHADOW prefix while the Actions cron remains the sole
LIVE writer at ``models/hafs``. The cutover (Stage 3) flips ``HAFS_R2_PREFIX`` to
``models/hafs`` and gates the cron's render off - reversible.

Anti-freeze (the floater-freeze lesson): the spine emits its health heartbeat
only between cycles, so a 30-60 min render would otherwise look frozen. Two
guards: (1) ``STALE_AFTER_S`` is set WELL above the watchdog so the health
watcher never false-alarms mid-render; (2) a ProgressHeartbeat thread writes
``{prefix}/render_progress.json`` every ~30 s during the render. And because the
spine has NO process() timeout, the SUBPROCESS watchdog (kill the render's
process group on timeout) is what guarantees a wedged cycle self-aborts and is
retried next poll, instead of hanging the worker forever.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import poller_framework as pf

log = logging.getLogger("hafs-render-poller")


# ---------------------------------------------------------------------------
# Config (env-driven, safe defaults). Shadow-first.
# ---------------------------------------------------------------------------
def _env(n: str, d: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(n)
    return v if v not in (None, "") else d


R2_ENDPOINT = _env("R2_ENDPOINT")
R2_BUCKET = _env("R2_BUCKET", "triple-a-tropics-media")
R2_ACCESS_KEY_ID = _env("R2_ACCESS_KEY_ID") or _env("AWS_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = _env("R2_SECRET_ACCESS_KEY") or _env("AWS_SECRET_ACCESS_KEY")

# SHADOW by default. Cutover flips this to "models/hafs" (the live cron keys).
HAFS_R2_PREFIX = _env("HAFS_R2_PREFIX", "shadow/models/hafs").strip("/")

# Render scope - identical defaults to update-hafs.yml (all models/domains/products).
HAFS_MODELS = _env("HAFS_MODELS", "hafsa,hafsb")
HAFS_DOMAINS = _env("HAFS_DOMAINS", "storm.atm,parent.atm")
HAFS_PRODUCTS = _env("HAFS_PRODUCTS")           # None -> generator default (all 11)

# Railway Pro: 8 vCPU for this service -> 8 render workers (2x the 4-core runner).
HAFS_JOBS = int(_env("HAFS_JOBS", "8"))

# INGEST runs at a LOWER width than render. Each ingest decodes a large
# multi-field GRIB (the parent.atm / hafsb domains are the heaviest) -> the stage
# is memory-bound, and 8 concurrent heavy decodes OOM'd the pool on this
# memory-tighter host (BrokenProcessPool), dropping exactly the heavy
# parent.atm/hafsb frames. 4-wide ingest fits in memory; render stays 8-wide
# (CPU-bound, reads small cached fields). The generator's halving backoff is the
# safety net if a cycle is heavy enough to still OOM at 4.
HAFS_INGEST_JOBS = int(_env("HAFS_INGEST_JOBS", "4"))

# Concurrent R2 PNG uploads (Pass 1 only). The render is CPU-bound; the upload is
# I/O/latency-bound (each put_object is a network round-trip), so a thread pool of
# ~16 cuts a 1738-frame upload from ~20 min (sequential) to ~1-2 min - matching
# the cron's parallel `aws s3 sync`. Only Pass 1 is parallel; the manifest write
# (Pass 2) and prune (Pass 3) stay strictly AFTER it (no 404 window).
HAFS_UPLOAD_WORKERS = int(_env("HAFS_UPLOAD_WORKERS", "16"))

# Watchdog: hard wall-clock cap on one cycle's render (mirrors the cron's
# timeout-minutes: 120). A wedged render self-aborts; the cycle is retried.
RENDER_TIMEOUT_S = int(_env("RENDER_TIMEOUT_S", str(120 * 60)))

# Poll cadence + staleness. STALE_AFTER_S must exceed RENDER_TIMEOUT_S so the
# health watcher never false-alarms while a legitimate long render is running.
POLL_INTERVAL_S = float(_env("POLL_INTERVAL_S", "120"))
STALE_AFTER_S = float(_env("STALE_AFTER_S", str(RENDER_TIMEOUT_S + 30 * 60)))
PROGRESS_HEARTBEAT_S = float(_env("PROGRESS_HEARTBEAT_S", "30"))

# Which model the cycle is resolved from (the cron uses models[0] = hafsa).
RESOLVE_MODEL = _env("HAFS_RESOLVE_MODEL", "hafsa")

CACHE_DIR = _env("HERBIE_DATA", "/tmp/herbie_data")

CC_FRAME = "public, max-age=300"      # match the cron's frame Cache-Control
CC_MANIFEST = "public, max-age=300"   # match the cron's manifest Cache-Control
CC_HEALTH = "public, max-age=30"


# ---------------------------------------------------------------------------
# R2 client - a faithful copy of the floater poller's R2 class
# (floater_poller.py:559-602): put_bytes + put_json + get_json + batched delete
# with 1000-key pagination. The repo convention is per-worker duplication (the
# floater has class R2, the intensity poller has class R2Sink); there is no
# shared helper module. This is the put+list+batched-delete client the cron's
# 3-pass ``aws s3 sync --delete`` prune needs.
# ---------------------------------------------------------------------------
class R2:
    def __init__(self) -> None:
        import boto3
        from botocore.config import Config as BotoConfig
        self.bucket = R2_BUCKET
        self.s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
        )

    def put_bytes(self, key: str, data: bytes, content_type: str, cache: str) -> bool:
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data,
                               ContentType=content_type, CacheControl=cache)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("R2 put %s failed: %s", key, e)
            return False

    def put_json(self, key: str, obj: dict, cache: str) -> bool:
        body = json.dumps(obj, separators=(",", ":")).encode()
        return self.put_bytes(key, body, "application/json", cache)

    def list_keys(self, prefix: str) -> list[str]:
        """All object keys under ``prefix`` (paginated)."""
        keys: list[str] = []
        token: Optional[str] = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kw["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                if not token:
                    break
            else:
                break
        return keys

    def delete(self, keys: Iterable[str]) -> None:
        keys = [k for k in keys if k]
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            try:
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": k} for k in batch]},
                )
            except Exception as e:  # noqa: BLE001
                log.warning("R2 delete batch failed: %s", e)


# ---------------------------------------------------------------------------
# Render in a subprocess under a hard wall-clock watchdog.
# ---------------------------------------------------------------------------
class RenderError(RuntimeError):
    """Render subprocess failed or was killed - hold the signature, retry."""


def run_render_subprocess(cycle: str, out_dir: str, *,
                          jobs: int = HAFS_JOBS,
                          ingest_jobs: int = HAFS_INGEST_JOBS,
                          models: str = HAFS_MODELS,
                          domains: str = HAFS_DOMAINS,
                          products: Optional[str] = HAFS_PRODUCTS,
                          timeout_s: int = RENDER_TIMEOUT_S,
                          save_dir: str = CACHE_DIR) -> dict:
    """Render one cycle with the UNCHANGED hafs_render package (byte-identical to
    the cron) in a subprocess, killed if it exceeds ``timeout_s``.

    The subprocess runs ``python -m hafs_render.generate_hafs_plots`` - the exact
    code path the cron's shim calls - so output is byte-identical (modulo
    cross-host raster noise). It is started in its OWN process group so the
    watchdog kills the whole render tree (the internal ProcessPoolExecutor
    workers too), never leaving orphans. Raises RenderError on timeout or a
    non-zero exit (e.g. the generator's total-failure exit 1)."""
    cmd = [sys.executable, "-m", "hafs_render.generate_hafs_plots",
           "--cycle", cycle, "--out-dir", out_dir, "--jobs", str(jobs),
           "--ingest-jobs", str(ingest_jobs),
           "--models", models, "--domains", domains, "--save-dir", save_dir]
    if products:
        cmd += ["--products", products]
    env = dict(os.environ, HERBIE_DATA=save_dir)
    log.info("render start: cycle=%s jobs=%d ingest_jobs=%d -> %s (timeout %ds)",
             cycle, jobs, ingest_jobs, out_dir, timeout_s)
    t0 = time.time()
    # Capture combined stdout+stderr to a file (NOT a PIPE - a chatty render would
    # deadlock on a full pipe buffer). On failure the tail is embedded in the
    # RenderError so it reaches R2 (render_progress.json), making a remote failure
    # self-diagnosing without Railway log access.
    log_path = os.path.join(tempfile.gettempdir(), f"hafs_render_{cycle}.log")
    logf = open(log_path, "wb")
    # start_new_session=True -> new process group; killpg on timeout takes the
    # whole render tree down (the spine has no process() timeout; this is it).
    proc = subprocess.Popen(cmd, start_new_session=True, env=env,
                            stdout=logf, stderr=subprocess.STDOUT)
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.error("render WATCHDOG fired after %ds - killing cycle %s",
                  timeout_s, cycle)
        _kill_tree(proc)
        logf.close()
        raise RenderError(f"render timeout after {timeout_s}s (cycle {cycle})\n"
                          + _tail(log_path))
    finally:
        if not logf.closed:
            logf.close()
    if rc != 0:
        # generator exits 1 on TOTAL failure (storms found, 0 rendered) so the
        # destructive prune never runs and the prior cycle stays live.
        raise RenderError(f"render exit {rc} (cycle {cycle})\n" + _tail(log_path))
    secs = time.time() - t0
    log.info("render ok: cycle=%s in %.0fs", cycle, secs)
    # Parse the captured log into a per-pair coverage/health summary so the cycle
    # SELF-REPORTS on success too (not just on failure): ingest/render ok-failed,
    # the pairs that were skipped-incomplete, and the failed-ingest error types -
    # which is how a partial-coverage cycle (e.g. dropped hafsb pairs) is visible.
    summary = _parse_render_log(log_path)
    summary["render_seconds"] = round(secs, 1)
    return summary


def _tail(path: str, nbytes: int = 6000) -> str:
    """Last ``nbytes`` of a log file (the render subprocess's stdout+stderr) for
    embedding in a RenderError so the real failure is observable on R2."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            data = f.read().decode("utf-8", "replace")
        return ("...\n" + data) if size > nbytes else data
    except Exception as e:  # noqa: BLE001
        return f"(could not read render log: {e})"


def _parse_render_log(log_path: str) -> dict:
    """Parse the render subprocess's stdout+stderr into a coverage/health summary.

    Extracts the generator's own log lines: the plan, the ingest ok/failed count,
    the render ok/failed count, every pair it SKIPPED (incomplete / no-data), and
    the per-frame ingest failures WITH their error type - aggregated into
    ``ingest_error_counts`` so the root cause is obvious (e.g.
    ``{"BrokenProcessPool ...": 100}`` => OOM, vs a fetch/timeout error). Never
    raises; returns whatever it could parse."""
    summary: dict = {"planned": {}, "ingest": {}, "render": {},
                     "skipped_pairs": [], "failed_ingest": [], "failed_render": []}
    try:
        with open(log_path, "r", errors="replace") as f:
            text = f.read()
    except Exception as e:  # noqa: BLE001
        summary["parse_error"] = str(e)
        return summary
    m = re.search(r"planned (\d+) ingest frame\(s\) \+ (\d+) render task\(s\) "
                  r"across (\d+) storm", text)
    if m:
        summary["planned"] = {"ingest_frames": int(m[1]),
                              "render_tasks": int(m[2]), "storms": int(m[3])}
    m = re.search(r"ingested (\d+)/(\d+) frame\(s\) ok \((\d+) failed\)", text)
    if m:
        summary["ingest"] = {"ok": int(m[1]), "total": int(m[2]), "failed": int(m[3])}
    m = re.search(r"rendered (\d+) ok, (\d+) failed", text)
    if m:
        summary["render"] = {"ok": int(m[1]), "failed": int(m[2])}
    for mm in re.finditer(r"skip (\S+) (\S+) (\S+), "
                          r"(incomplete \(max f\d+ < f\d+\)|no frames published[^\n]*)",
                          text):
        summary["skipped_pairs"].append(
            {"model": mm[1], "storm": mm[2], "domain": mm[3], "reason": mm[4].strip()})
    err_counts: dict = {}
    for mm in re.finditer(r"ingest failed: (\S+) (\S+) (\S+) f(\d+) - (.+)", text):
        if len(summary["failed_ingest"]) < 40:
            summary["failed_ingest"].append(
                {"model": mm[1], "storm": mm[2], "domain": mm[3],
                 "fxx": int(mm[4]), "error": mm[5].strip()[:200]})
        et = mm[5].strip().split(":")[0][:80]
        err_counts[et] = err_counts.get(et, 0) + 1
    for mm in re.finditer(r"render failed: (\S+) (\S+) (\S+) (\S+) f(\d+) - (.+)", text):
        if len(summary["failed_render"]) < 40:
            summary["failed_render"].append(
                {"model": mm[1], "storm": mm[2], "domain": mm[3],
                 "product": mm[4], "fxx": int(mm[5]), "error": mm[6].strip()[:200]})
    if err_counts:
        summary["ingest_error_counts"] = err_counts
    return summary


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("kill_tree failed: %s", e)
    try:
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# 3-pass R2 upload (replicates the cron's aws s3 sync ordering exactly).
# ---------------------------------------------------------------------------
def upload_cycle(r2: R2, out_dir: str, prefix: str) -> dict:
    """Upload a rendered out-dir to R2 under ``prefix`` in the cron's 3 passes:
      1. PNG frames (no delete) - image/png, max-age=300.
      2. manifest.json - application/json (now references only present frames).
      3. prune: delete *.png under ``prefix`` NOT in this render (batched).
    Returns a small summary dict. Mirrors update-hafs.yml's ordering so there is
    no 404 window and an off-season empty manifest correctly clears the prefix."""
    out = Path(out_dir)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise RenderError(f"no manifest at {manifest_path} - nothing to publish")

    pngs = sorted(p for p in out.rglob("*.png"))

    # Pass 1: PNG frames, no delete - uploaded CONCURRENTLY (boto3 clients are
    # thread-safe). This is a hard BARRIER: the ThreadPoolExecutor block exits
    # only when every put has returned, so the manifest (Pass 2) is never written
    # while a frame it references is still unpushed. All-or-nothing like the cron:
    # if any frame fails (after botocore's own retries), raise so NO manifest and
    # NO prune run - the cycle is held and retried, never published with 404s.
    def _put(p):
        rel = p.relative_to(out).as_posix()
        key = f"{prefix}/{rel}"
        try:
            ok = r2.put_bytes(key, p.read_bytes(), "image/png", CC_FRAME)
        except Exception:  # noqa: BLE001
            ok = False
        return key, ok

    fresh_png_keys = set()
    failures = 0
    with cf.ThreadPoolExecutor(max_workers=HAFS_UPLOAD_WORKERS) as ex:
        for key, ok in ex.map(_put, pngs):   # consuming all results = the barrier
            if ok:
                fresh_png_keys.add(key)
            else:
                failures += 1
    if failures:
        raise RenderError(
            f"{failures}/{len(pngs)} frame uploads failed under {prefix} - "
            "holding cycle (manifest NOT written, prune NOT run)")

    # Pass 2: manifest - written ONLY after Pass 1's barrier, so it references
    # only frames that are all present on R2.
    manifest = json.loads(manifest_path.read_text())
    r2.put_json(f"{prefix}/manifest.json", manifest, CC_MANIFEST)

    # Pass 3: prune - DUAL-WRITER SAFE. While the cron co-writes models/hafs, it
    # renders frames for the SAME cycle including pairs this worker skipped (e.g.
    # hafsb). The keys carry no cycle segment, so deleting every *.png not in THIS
    # render would delete the cron's CURRENT-cycle frames -> live coverage
    # flicker. So prune at the STORM level: keep every frame of a storm present in
    # the current render (incl. the co-writer's pairs), delete only frames of
    # storms no longer rendered at all (retired storms / old cycles). Off-season
    # (storms=[]) -> current_storms empty -> all frames pruned (correct).
    current_storms = {s.get("id") for s in manifest.get("storms", [])}
    existing = r2.list_keys(prefix + "/")
    orphans = []
    for k in existing:
        if not k.endswith(".png"):
            continue
        parts = k[len(prefix) + 1:].split("/")   # model/storm/domain/product/fNNN.png
        storm = parts[1] if len(parts) >= 2 else None
        if storm not in current_storms:          # a storm no longer rendered at all
            orphans.append(k)
    if orphans:
        r2.delete(orphans)

    # Per-pair coverage (model/storm/domain -> fxx count) for the cycle summary.
    coverage = []
    for s in manifest.get("storms", []):
        for mdl, doms in s.get("frames", {}).items():
            for dom, prods in doms.items():
                nf = len(next(iter(prods.values()))) if prods else 0
                coverage.append({"model": mdl, "storm": s.get("id"),
                                 "domain": dom, "products": len(prods), "fxx": nf})
    log.info("uploaded %d frame(s) + manifest, pruned %d retired-storm orphan(s) "
             "under %s", len(pngs), len(orphans), prefix)
    return {"frames": len(pngs), "pruned": len(orphans), "prefix": prefix,
            "storms": sorted(s for s in current_storms if s), "coverage": coverage}


# ---------------------------------------------------------------------------
# Progress heartbeat - written THROUGHOUT a long render so it is observable.
# ---------------------------------------------------------------------------
class ProgressHeartbeat:
    """Daemon thread that writes ``{prefix}/render_progress.json`` every
    ``interval`` seconds while a render runs (the spine's health heartbeat only
    fires between cycles, so a multi-minute render would otherwise look frozen).
    Best-effort: a failed write never disturbs the render."""

    def __init__(self, r2: R2, prefix: str, cycle: str, *,
                 interval: float = PROGRESS_HEARTBEAT_S,
                 clock: Callable[[], dt.datetime] = pf.utcnow):
        self.r2 = r2
        self.key = f"{prefix}/render_progress.json"
        self.cycle = cycle
        self.interval = interval
        self.clock = clock
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._t0 = time.time()

    def _emit(self, status: str, error: Optional[str] = None) -> None:
        payload = {
            "status": status,
            "cycle": self.cycle,
            "started_utc": pf.iso_z(self._started_dt),
            "updated_utc": pf.iso_z(self.clock()),
            "elapsed_s": round(time.time() - self._t0, 1),
        }
        if error:
            # The render subprocess's failure tail (embedded in RenderError) so a
            # remote failure is fully diagnosable from R2, no Railway log needed.
            payload["error"] = error[-6000:]
        self.r2.put_json(self.key, payload, CC_HEALTH)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._emit("rendering")
            except Exception as e:  # noqa: BLE001
                log.debug("progress heartbeat write failed: %s", e)

    def __enter__(self) -> "ProgressHeartbeat":
        self._started_dt = self.clock()
        self._t0 = time.time()
        try:
            self._emit("rendering")
        except Exception:  # noqa: BLE001
            pass
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2)
        try:
            err = f"{exc_type.__name__}: {exc}" if exc_type else None
            self._emit("failed" if exc_type else "done", error=err)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Cycle resolution (lazy import of hafs_render so the module imports - and the
# offline tests run - without the GRIB stack installed).
# ---------------------------------------------------------------------------
def resolve_latest_complete_cycle(model: str = RESOLVE_MODEL) -> Optional[str]:
    """The newest COMPLETE cycle id 'YYYYMMDDHH' (reuses the cron's resolver), or
    None off-season."""
    from hafs_render.generate_hafs_plots import resolve_latest_cycle  # lazy
    import requests
    r = resolve_latest_cycle(model, session=requests.Session())
    if r is None:
        return None
    date, hh = r
    return f"{date}{hh}"


# ---------------------------------------------------------------------------
# The HAFS Source (one Source; HMON later = a 2nd Source on the same engine).
# ---------------------------------------------------------------------------
def make_hafs_source(r2: R2, *, prefix: str = HAFS_R2_PREFIX,
                     cycle_resolver: Optional[Callable[[], Optional[str]]] = None,
                     render_fn: Optional[Callable[[str, str], None]] = None,
                     uploader: Optional[Callable[[R2, str, str], dict]] = None,
                     out_dir_factory: Optional[Callable[[str], str]] = None,
                     diagnoser: Optional[Callable[[str], str]] = None,
                     clock: Callable[[], dt.datetime] = pf.utcnow) -> pf.Source:
    """Build the HAFS Source. ``cycle_resolver`` / ``render_fn`` / ``uploader`` /
    ``out_dir_factory`` / ``diagnoser`` are injectable so the offline tests
    exercise the change-gate, watchdog-abort, and prune WITHOUT a real render,
    R2, or network diagnostic."""
    resolve = cycle_resolver or resolve_latest_complete_cycle
    render = render_fn or run_render_subprocess
    upload = uploader or upload_cycle
    diagnose = diagnoser or diagnose_ingest

    def fetch():
        # Cheap: an S3 listing to find the newest complete cycle. None off-season
        # is a legitimate 'nothing now' (no retry burns), not an error.
        return {"cycle": resolve()}

    def change_key(data):
        # New data iff a brand-new COMPLETE cycle appeared. A still-latest cycle
        # takes the spine's cheap path and is NOT re-rendered.
        return data["cycle"]

    def valid_time(data):
        c = data["cycle"]
        if not c:
            return None
        try:
            return dt.datetime.strptime(c, "%Y%m%d%H")
        except ValueError:
            return None

    def process(ctx: pf.ProcessContext):
        cycle = ctx.data["cycle"]
        if not cycle:
            # Off-season: nothing to render. (change_key None never reaches here
            # twice - the first None is a change from 'never'; subsequent Nones
            # are UNCHANGED. Publishing an empty manifest is the cron's job; the
            # worker simply renders nothing.)
            log.info("no complete cycle - nothing to render")
            return
        if out_dir_factory is not None:
            out_dir = out_dir_factory(cycle)
            _render_and_upload(r2, prefix, cycle, out_dir, render, upload,
                               diagnose, clock)
        else:
            with tempfile.TemporaryDirectory(prefix=f"hafs_{cycle}_") as td:
                out_dir = str(Path(td) / "hafs")
                _render_and_upload(r2, prefix, cycle, out_dir, render, upload,
                                   diagnose, clock)

    return pf.Source(name="hafs", fetch=fetch, change_key=change_key,
                     process=process, valid_time=valid_time)


def _render_and_upload(r2, prefix, cycle, out_dir, render, upload, diagnose,
                       clock) -> None:
    # Progress heartbeat wraps the WHOLE render+upload so a hang at any stage is
    # observable. The render raises RenderError on timeout / total-failure -> the
    # spine holds the signature and retries next poll (NO upload on failure, so
    # the destructive prune never runs and the prior shadow/live frames stay).
    with ProgressHeartbeat(r2, prefix, cycle, clock=clock):
        try:
            summary = render(cycle, out_dir) or {}   # parsed coverage/health summary
        except RenderError as e:
            # The generator's process pool swallows per-frame tracebacks (logs
            # only the exception type). On failure, run ONE un-swallowed ingest to
            # capture the real traceback to R2 - a remote decode failure is then
            # fully diagnosable without Railway log access.
            diag = diagnose(cycle)
            try:
                r2.put_json(f"{prefix}/render_error.json", {
                    "cycle": cycle, "utc": pf.iso_z(clock()),
                    "error": str(e)[-4000:], "ingest_traceback": diag[-8000:],
                }, CC_HEALTH)
            except Exception:  # noqa: BLE001
                pass
            raise
        upcov = upload(r2, out_dir, prefix) or {}
        # SUCCESS self-report: persist the coverage/health summary EVERY cycle
        # (not just on failure) so partial coverage - dropped pairs, the
        # ingest-error types - is visible on R2 without Railway-log access. Best
        # effort: a summary-write failure never fails an otherwise-good publish.
        try:
            r2.put_json(f"{prefix}/render_summary.json", {
                "cycle": cycle,
                "generated_utc": pf.iso_z(clock()),
                "frames": upcov.get("frames"),
                "storms": upcov.get("storms"),
                "pruned": upcov.get("pruned"),
                "coverage": upcov.get("coverage"),
                **{k: summary[k] for k in
                   ("render_seconds", "planned", "ingest", "render",
                    "skipped_pairs", "failed_ingest", "failed_render",
                    "ingest_error_counts") if k in summary},
            }, CC_HEALTH)
        except Exception:  # noqa: BLE001
            pass


def diagnose_ingest(cycle: str) -> str:
    """Run ONE hafs_render ingest in a subprocess WITHOUT the generator's
    exception swallowing, so the real traceback (e.g. the cfgrib/eccodes
    AssertionError location) is captured. Tries the simplest frame likely present
    (hafsa 06w storm.atm f000); returns combined stdout+stderr."""
    save_dir = os.path.join(tempfile.gettempdir(), "hafs_diag_cache")
    # Build the snippet with f-strings only (NO % operator - the embedded
    # strptime '%Y%m%d%H' would break str.__mod__). Whole body guarded so a
    # diagnostic bug can never mask or replace the real render failure.
    try:
        date, hh = cycle[:8], cycle[8:]
        code = "\n".join([
            "import datetime as dt, traceback",
            "from pathlib import Path",
            "from hafs_render import hafs_cache as fc",
            "from hafs_render.generate_hafs_plots import list_storms",
            "import requests",
            f"cy=dt.datetime.strptime({cycle!r},'%Y%m%d%H')",
            "try:",
            f"    storms=list_storms('hafsa',{date!r},{hh!r},session=requests.Session())",
            "    print('storms',storms)",
            "    st=storms[0] if storms else '06w'",
            f"    fc.ingest_frame('hafsa',st,'storm.atm',cy,0,Path({save_dir!r})/'diag.nc',{save_dir!r},want_refl=True,want_pwat=True,want_upper=True,sat_parms=(58,53),remove_grib=True)",
            "    print('DIAG_INGEST_OK')",
            "except Exception:",
            "    traceback.print_exc()",
        ])
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=240,
                           env=dict(os.environ, HERBIE_DATA=save_dir))
        return (r.stdout + "\n" + r.stderr)[-8000:]
    except Exception as e:  # noqa: BLE001
        return f"(diagnose_ingest failed: {type(e).__name__}: {e})"


# ---------------------------------------------------------------------------
# Engine + Railway entrypoint
# ---------------------------------------------------------------------------
class _HealthSink(pf.Sink):
    """Adapts the R2 client to the Sink protocol so the engine's health
    heartbeat lands at ``{prefix}/poller_health.json`` (the intensity poller's
    sink_heartbeat pattern)."""

    def __init__(self, r2: R2, prefix: str):
        self.r2 = r2
        self.prefix = prefix

    def write(self, key: str, payload: dict) -> None:
        self.r2.put_json(f"{self.prefix}/{key}", payload, CC_HEALTH)


def build_engine(r2: R2, *, prefix: str = HAFS_R2_PREFIX,
                 interval_s: float = POLL_INTERVAL_S,
                 clock: Callable[[], dt.datetime] = pf.utcnow,
                 sleep: Callable[[float], None] = time.sleep,
                 **source_kwargs) -> pf.PollerEngine:
    health = _HealthSink(r2, prefix)
    source = make_hafs_source(r2, prefix=prefix, clock=clock, **source_kwargs)
    return pf.PollerEngine(
        [source], name="hafs-render-poller", interval_s=interval_s,
        stale_after_s=STALE_AFTER_S, sink=health,
        heartbeat=pf.sink_heartbeat(health, "poller_health.json"),
        clock=clock, sleep=sleep)


def main() -> None:  # pragma: no cover - Railway worker entrypoint
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s")
    missing = []
    if not R2_ENDPOINT:
        missing.append("R2_ENDPOINT")
    if not R2_ACCESS_KEY_ID:
        missing.append("R2_ACCESS_KEY_ID/AWS_ACCESS_KEY_ID")
    if not R2_SECRET_ACCESS_KEY:
        missing.append("R2_SECRET_ACCESS_KEY/AWS_SECRET_ACCESS_KEY")
    if missing:
        raise SystemExit("hafs_render_poller: missing required env: "
                         + ", ".join(missing))
    r2 = R2()
    eng = build_engine(r2)
    log.info("hafs render poller starting | prefix=%s | jobs=%d | interval=%gs "
             "| watchdog=%ds | stale_after=%gs",
             HAFS_R2_PREFIX, HAFS_JOBS, POLL_INTERVAL_S, RENDER_TIMEOUT_S,
             STALE_AFTER_S)
    eng.run_forever()


if __name__ == "__main__":
    main()
