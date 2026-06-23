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

Change-gated: ``change_key`` is the newest COMPLETE cycle id PLUS the set of
(model, storm, domain) pairs complete upstream, so the expensive render fires
on a genuinely new cycle AND - the INTRA-CYCLE CATCH-UP - when a pair that was
absent or still uploading at render time (a late HAFS-B run, a late storm)
reaches its terminal frame while the cycle is still current. A brand-new cycle
takes the full-render path exactly as before; a late pair takes an INCREMENTAL
path that renders ONLY the newly-complete missing pairs (one filtered
``--models/--storm/--domains`` subprocess per (model, storm) group, same
HAFS_JOBS/HAFS_INGEST_JOBS), merges them ADDITIVELY into the live manifest +
render_summary, and NEVER prunes. Completed pairs are never re-rendered; the
picker gains a late Amanda/HAFS-B as soon as its data exists upstream, not a
cycle later. The render is the same code the cron runs, imported as the pinned
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

# Render-pool width. 8 OOM-KILLED the whole worker on a heavy multi-storm cycle
# (the 2026-06-17 06Z wedge: container OOM at 154/258 frames, then a
# Railway-gives-up crash-loop). CAPPED at HAFS_JOBS_MAX=4 so a STALE Railway env
# (the dashboard still sets HAFS_JOBS=8) cannot push it back into the OOM regime -
# the worker is self-protecting, not at the mercy of a leftover env var. Each
# render worker holds rendered arrays, but the memory-bound INGEST width below
# dominates the peak. To allow a higher width after a memory bump, raise
# HAFS_JOBS_MAX deliberately (a new env), not HAFS_JOBS.
HAFS_JOBS_MAX = int(_env("HAFS_JOBS_MAX", "4"))
HAFS_JOBS = min(int(_env("HAFS_JOBS", "4")), HAFS_JOBS_MAX)

# INGEST runs at a LOWER width than render: each ingest decodes a large
# multi-field GRIB (parent.atm / hafsb are the heaviest), so this stage is the
# memory-bound one and SETS the peak tree RSS. 8 OOM'd the pool (BrokenProcessPool)
# and even 4 OOM-killed the WHOLE worker on the 2026-06-17 06Z multi-storm cycle ->
# lowered to 2 so at most two concurrent heavy decodes are resident. The
# generator's halving backoff is the in-process safety net; the lower width + the
# liveness self-kill + the external ens_watchdog cover the cgroup OOM that took the
# whole worker down. Tune against render_summary mem.peak_tree_rss_mb. Capped the
# same way (HAFS_INGEST_JOBS_MAX) so a stale env can't re-open the OOM here either.
HAFS_INGEST_JOBS_MAX = int(_env("HAFS_INGEST_JOBS_MAX", "2"))
HAFS_INGEST_JOBS = min(int(_env("HAFS_INGEST_JOBS", "2")), HAFS_INGEST_JOBS_MAX)

# INTRA-CYCLE CATCH-UP. While the rendered cycle is still the newest, each poll
# re-checks upstream for (model, storm, domain) pairs that were skipped at
# render time (absent or still uploading - "incomplete (max fNNN < f126)") and
# renders the ones that have since reached their terminal frame, publishing
# them additively. Default ON; set HAFS_CATCHUP=false to restore pure
# cycle-id gating (instant rollback, no redeploy semantics change otherwise).
HAFS_CATCHUP = (_env("HAFS_CATCHUP", "true") or "").strip().lower() not in (
    "false", "0", "no")

# PROGRESSIVE FRAME-LOAD: render forecast hours AS THEY POST upstream instead
# of waiting for a pair's terminal f126 - the frame-granular generalization of
# the pair-level catch-up (which stays in-tree as the rollback:
# HAFS_PROGRESSIVE=false reverts to the classic complete-pair source above).
# Frames publish ADDITIVELY per poll batch under CYCLE-SCOPED keys
# ({prefix}/{cycle}/...) with a cycles[]-bearing manifest; the legacy manifest
# fields keep describing the newest COMPLETE cycle with its prefix baked into
# path_template, so an old frontend keeps working through deploy skew.
HAFS_PROGRESSIVE = (_env("HAFS_PROGRESSIVE", "true") or "").strip().lower() not in (
    "false", "0", "no")

# The expected forecast grid (mirrors the generator's TERMINAL_FXX / step) -
# drives frames_expected, completion detection, and the frontend's tick grid.
FXX_END = int(_env("HAFS_FXX_END", "126"))
FXX_STEP = int(_env("HAFS_FXX_STEP", "3"))

# Progressive render granularity: how many forecast-hour FRAMES one render
# subprocess covers before the manifest is re-published. HAFS posts a whole
# cycle's hours upstream in one BATCH, so a single poll's delta is the entire
# storm x 43 frames; rendering+publishing that as ONE subprocess only surfaces
# the frames at the very end ("empty until F126"). Rendering ASCENDING in small
# chunks and publishing after each makes the building cycle populate hour-by-hour
# so the user can watch the run build and scrub completed hours mid-run. Smaller
# = smoother build + lower per-subprocess peak memory, but more subprocess
# startups (each re-imports herbie/matplotlib); 4 frames (=12 fcst h) balances
# the two. Set 0/negative to disable chunking (render the whole delta at once).
PROGRESSIVE_FXX_CHUNK = int(_env("HAFS_PROGRESSIVE_FXX_CHUNK", "4"))

# Flat (pre-cycle-scoped) legacy PNG keys are deleted by the completion prune
# once they are older than this - long enough that no live browser session
# still holds a manifest that references them.
FLAT_KEY_TTL_S = int(_env("HAFS_FLAT_KEY_TTL_S", str(24 * 3600)))

# A pair whose catch-up render keeps failing is abandoned after this many
# attempts (visible in render_summary's skipped_pairs) so a permanently-broken
# upstream pair can't burn a render attempt every poll until the next cycle.
HAFS_CATCHUP_MAX_ATTEMPTS = int(_env("HAFS_CATCHUP_MAX_ATTEMPTS", "3"))

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

# Upstream-quiescence gate: when the newest AVAILABLE upstream cycle (from a
# CLEAN listing) is older than this, NOAA has stopped running HAFS for the
# storm (dissipated) or the season — the manifest must EMPTY itself instead
# of serving a dead storm's frozen last run as current (03E/Cristina sat on
# /models/ for ~2 days after dissipation: quiet upstream meant change_key
# never changed and nothing ever rewrote the manifest). HAFS publishes every
# ~6 h while a storm is tasked, so 30 h = five missed cycles. A listing
# FAILURE never reaches this path — resolve_active_cycle raises on a total
# outage so the spine keeps the last-known-good manifest (never mass-hide
# on an error).
HAFS_QUIET_AFTER_H = float(_env("HAFS_QUIET_AFTER_H", "30"))

# Memory-telemetry sample cadence: while a render subprocess runs, the parent
# samples the whole render tree's RSS (every process in the child's process
# group, via /proc) at this interval and records the peak. The Railway bill is
# RAM-minutes, ~76% of it in these render windows - this is the instrument
# that says where the GB-minutes go (and sizes the VPS: peak tree RSS during a
# heavy cycle is the "does it fit in 24GB" number). Pure observation: ~10
# /proc reads every 2s, no effect on render behavior.
MEM_SAMPLE_S = float(_env("MEM_SAMPLE_S", "2.0"))

# Liveness self-kill: a daemon watchdog exits the process NON-ZERO if no progress
# signal (a poll heartbeat or a render mem-sample) is seen for this long, turning a
# FROZEN worker into a clean exit that Railway's restartPolicy (ON_FAILURE) can
# actually cycle. A hang never trips ON_FAILURE on its own - the 2026-06-17 30 h
# wedge was exactly this: the worker stopped emitting both heartbeats and Railway
# never restarted it. Set WELL above any healthy gap (a render marks progress every
# ~MEM_SAMPLE_S=2 s, a poll every ~POLL_INTERVAL_S=120 s), so 1200 s of TOTAL
# silence is unambiguously wedged - ~90x faster than the wedge, zero false-positive
# room. The external ens_watchdog is the out-of-box backstop; this is the in-box one.
LIVENESS_TIMEOUT_S = float(_env("HAFS_LIVENESS_TIMEOUT_S", "1200"))
LIVENESS_CHECK_S = float(_env("HAFS_LIVENESS_CHECK_S", "60"))

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

    def copy(self, src_key: str, dst_key: str) -> bool:
        """Server-side object copy (no download) - the one-time v1->v2 key
        migration uses it to move the newest complete cycle's frames under
        their cycle-scoped prefix."""
        try:
            self.s3.copy_object(Bucket=self.bucket, Key=dst_key,
                                CopySource={"Bucket": self.bucket,
                                            "Key": src_key})
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("R2 copy %s -> %s failed: %s", src_key, dst_key, e)
            return False

    def get_json(self, key: str) -> Optional[dict]:
        """Fetch + parse a JSON object, or None on any failure (missing key,
        bad JSON). Used to bootstrap the progressive ledger from the live
        manifest after a worker restart."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except Exception as e:  # noqa: BLE001 - bootstrap is best-effort
            log.info("R2 get %s: %s (cold start)", key, e)
            return None

    def list_objects(self, prefix: str) -> list:
        """[(key, last_modified_datetime)] under ``prefix`` (paginated) - the
        LastModified-aware variant list_keys lacks, for the age-gated flat-key
        prune."""
        out: list = []
        token: Optional[str] = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kw["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                out.append((obj["Key"], obj.get("LastModified")))
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                if not token:
                    break
            else:
                break
        return out

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


def _proc_tree_rss_mb(pgid: int) -> tuple:
    """One memory sample of the render tree: (total_rss_mb, largest_proc_mb,
    nprocs) summed over every live process whose process GROUP is ``pgid``
    (run_render_subprocess starts the child with start_new_session=True, so
    the generator AND all its ProcessPoolExecutor workers share the child's
    pgid - the same property the watchdog's killpg relies on). Reads
    /proc/<pid>/stat directly (field 5 = pgrp, field 24 = rss pages); a pid
    vanishing mid-walk is skipped, never raised. Returns zeros on non-Linux."""
    page_mb = 4096 / (1024.0 * 1024.0)
    try:
        page_mb = os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except (ValueError, OSError, AttributeError):
        pass
    total = 0.0
    largest = 0.0
    n = 0
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return 0.0, 0.0, 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                stat = f.read().decode("ascii", "replace")
            # comm (field 2) may contain spaces/parens: split after the LAST
            # ')' so the numeric fields index stably from there.
            rest = stat[stat.rindex(")") + 2:].split()
            if int(rest[2]) != pgid:          # pgrp = overall field 5
                continue
            rss_mb = int(rest[21]) * page_mb  # rss = overall field 24 (pages)
            total += rss_mb
            largest = max(largest, rss_mb)
            n += 1
        except (OSError, ValueError, IndexError):
            continue
    return total, largest, n


def run_render_subprocess(cycle: str, out_dir: str, *,
                          jobs: int = HAFS_JOBS,
                          ingest_jobs: int = HAFS_INGEST_JOBS,
                          models: str = HAFS_MODELS,
                          domains: str = HAFS_DOMAINS,
                          storm: Optional[str] = None,
                          only_fxx: Optional[str] = None,
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
    non-zero exit (e.g. the generator's total-failure exit 1).

    ``storm`` (plus narrowed ``models``/``domains``) is the catch-up's scoping:
    the generator's filters render exactly one (model, storm) group's late
    pairs. Same jobs/ingest_jobs defaults as the full render, so an incremental
    render can't OOM where the full one fits."""
    cmd = [sys.executable, "-m", "hafs_render.generate_hafs_plots",
           "--cycle", cycle, "--out-dir", out_dir, "--jobs", str(jobs),
           "--ingest-jobs", str(ingest_jobs),
           "--models", models, "--domains", domains, "--save-dir", save_dir]
    if storm:
        cmd += ["--storm", storm]
    if only_fxx:
        # Progressive subset (hafs-render >= 0.4.0): render exactly these
        # hours, bypassing the per-pair terminal gate.
        cmd += ["--only-fxx", only_fxx]
    if products:
        cmd += ["--products", products]
    env = dict(os.environ, HERBIE_DATA=save_dir)
    log.info("render start: cycle=%s%s jobs=%d ingest_jobs=%d -> %s (timeout %ds)",
             cycle, f" storm={storm} models={models} domains={domains}"
             if storm else "", jobs, ingest_jobs, out_dir, timeout_s)
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
    # Memory telemetry: while waiting on the child, sample the whole render
    # tree's RSS (the child + its pool workers, by process group) and keep the
    # peak. Same hard wall-clock watchdog as before - the deadline is checked
    # against t0, so sampling never extends the timeout. Pure observation.
    mem = {"peak_tree_rss_mb": 0.0, "peak_proc_rss_mb": 0.0,
           "peak_procs": 0, "samples": 0}
    try:
        while True:
            try:
                rc = proc.wait(timeout=max(0.1, MEM_SAMPLE_S))
                break
            except subprocess.TimeoutExpired:
                if time.time() - t0 >= timeout_s:
                    raise
                tot, big, n = _proc_tree_rss_mb(proc.pid)
                mem["samples"] += 1
                _mark_progress()   # a render mem-sample = the worker is alive
                if tot > mem["peak_tree_rss_mb"]:
                    mem["peak_tree_rss_mb"] = round(tot, 1)
                    mem["peak_procs"] = n
                if big > mem["peak_proc_rss_mb"]:
                    mem["peak_proc_rss_mb"] = round(big, 1)
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
    log.info("render ok: cycle=%s in %.0fs - mem peak %.0f MB across %d proc(s), "
             "largest %.0f MB (%d samples)",
             cycle, secs, mem["peak_tree_rss_mb"], mem["peak_procs"],
             mem["peak_proc_rss_mb"], mem["samples"])
    # Parse the captured log into a per-pair coverage/health summary so the cycle
    # SELF-REPORTS on success too (not just on failure): ingest/render ok-failed,
    # the pairs that were skipped-incomplete, and the failed-ingest error types -
    # which is how a partial-coverage cycle (e.g. dropped hafsb pairs) is visible.
    summary = _parse_render_log(log_path)
    summary["render_seconds"] = round(secs, 1)
    summary["mem"] = mem
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
# 3-pass R2 upload (replicates the cron's aws s3 sync ordering exactly), plus
# the ADDITIVE catch-up upload (PNGs + merged manifest, no prune).
# ---------------------------------------------------------------------------
# domain slug (manifest / R2 path segment) -> raw S3 filename token. A static
# mirror of hafs_render.generate_hafs_plots.DOMAINS - NOT imported here, since
# that import pulls the whole GRIB/matplotlib stack and this map is needed by
# the lightweight manifest bookkeeping the offline tests exercise.
RAW_DOMAIN_BY_SLUG = {"storm": "storm.atm", "parent": "parent.atm"}


def _upload_pngs(r2: R2, out: Path, prefix: str) -> set:
    """Pass-1 concurrent PNG upload, shared by the full 3-pass upload and the
    additive catch-up upload.

    PNG frames are uploaded CONCURRENTLY (boto3 clients are thread-safe). This
    is a hard BARRIER: the ThreadPoolExecutor block exits only when every put
    has returned, so a manifest is never written while a frame it references is
    still unpushed. All-or-nothing like the cron: if any frame fails (after
    botocore's own retries), raise so NO manifest and NO prune run - the cycle
    is held and retried, never published with 404s."""
    pngs = sorted(p for p in out.rglob("*.png"))

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
    return fresh_png_keys


def _manifest_coverage(manifest: dict) -> list:
    """Per-pair coverage (model/storm/domain-slug -> product + fxx counts) of a
    manifest, for the cycle's render_summary."""
    coverage = []
    for s in manifest.get("storms", []) or []:
        for mdl, doms in (s.get("frames") or {}).items():
            for dom, prods in (doms or {}).items():
                nf = len(next(iter(prods.values()))) if prods else 0
                coverage.append({"model": mdl, "storm": s.get("id"),
                                 "domain": dom, "products": len(prods), "fxx": nf})
    return coverage


def manifest_pairs(manifest: dict) -> set:
    """The (model, storm, RAW domain) pairs with at least one rendered frame in
    a manifest - the worker's 'already rendered' ledger for the catch-up."""
    pairs = set()
    for s in manifest.get("storms", []) or []:
        for mdl, doms in (s.get("frames") or {}).items():
            for dom_slug, prods in (doms or {}).items():
                if any((prods or {}).values()):
                    pairs.add((mdl, s.get("id"),
                               RAW_DOMAIN_BY_SLUG.get(dom_slug, dom_slug)))
    return pairs


def manifest_frames(manifest: dict) -> set:
    """The (model, storm, RAW domain, fxx) FRAMES with a rendered PNG in a
    manifest (union across products) - the progressive ledger's granularity."""
    frames = set()
    for s in manifest.get("storms", []) or []:
        for mdl, doms in (s.get("frames") or {}).items():
            for dom_slug, prods in (doms or {}).items():
                raw = RAW_DOMAIN_BY_SLUG.get(dom_slug, dom_slug)
                for fxx_list in (prods or {}).values():
                    for f in fxx_list or []:
                        frames.add((mdl, s.get("id"), raw, int(f)))
    return frames


def merge_manifest(base: dict, incr: dict) -> dict:
    """ADDITIVE deep-merge of a catch-up render's manifest into the cycle's
    published manifest: per-storm frames union at the product-fxx level, new
    storms appended (storms stay sorted by id), generated_at/cycle refreshed
    from the increment. Header arrays (models/domains/products/path_template)
    stay the BASE's - the base is the full render (full scope); the increment
    is a filtered run whose headers list only its filter."""
    out = json.loads(json.dumps(base))   # deep copy; manifests are pure JSON
    if incr.get("generated_at"):
        out["generated_at"] = incr["generated_at"]
    if incr.get("cycle"):
        out["cycle"] = incr["cycle"]     # heals a prior empty/off-season base
    out.setdefault("storms", [])
    by_id = {s.get("id"): s for s in out["storms"]}
    for s in incr.get("storms", []) or []:
        cur = by_id.get(s.get("id"))
        if cur is None:                  # a storm that appeared late upstream
            cur = json.loads(json.dumps(s))
            cur["frames"] = {}
            out["storms"].append(cur)
        tgt = cur.setdefault("frames", {})
        for mdl, doms in (s.get("frames") or {}).items():
            for dom_slug, prods in (doms or {}).items():
                for prod, fxx in (prods or {}).items():
                    have = (tgt.setdefault(mdl, {}).setdefault(dom_slug, {})
                               .setdefault(prod, []))
                    tgt[mdl][dom_slug][prod] = sorted(set(have) | set(fxx))
    out["storms"].sort(key=lambda s: s.get("id") or "")
    return out


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

    # Pass 1: PNG frames, no delete (all-or-nothing barrier - see _upload_pngs).
    fresh_png_keys = _upload_pngs(r2, out, prefix)

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
    coverage = _manifest_coverage(manifest)
    log.info("uploaded %d frame(s) + manifest, pruned %d retired-storm orphan(s) "
             "under %s", len(fresh_png_keys), len(orphans), prefix)
    return {"frames": len(fresh_png_keys), "pruned": len(orphans), "prefix": prefix,
            "storms": sorted(s for s in current_storms if s), "coverage": coverage}


def upload_catchup(r2: R2, out_dir: str, prefix: str, base_manifest: dict) -> dict:
    """ADDITIVE upload for an intra-cycle catch-up render: the group's PNGs
    (same all-or-nothing barrier as the full upload), then the increment's
    manifest MERGED into ``base_manifest``. NEVER prunes - a catch-up only adds
    late pairs to a cycle already live, so deletion has no business here (the
    dual-writer-safe semantics are preserved by construction).

    Returns ``{"frames", "storms", "coverage", "pairs", "merged_manifest"}``
    where ``pairs`` is the increment's rendered (model, storm, raw-domain)
    triples and ``merged_manifest`` is what now lives at the manifest key."""
    out = Path(out_dir)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise RenderError(f"no manifest at {manifest_path} - nothing to publish")
    incr = json.loads(manifest_path.read_text())

    # Pass 1: the late pairs' PNGs (barrier; raise -> no manifest write, the
    # spine holds the signature and the group is retried next poll).
    fresh = _upload_pngs(r2, out, prefix)

    # Pass 2: the merged manifest - the picker gains the late pairs atomically,
    # with every referenced frame already on R2.
    merged = merge_manifest(base_manifest, incr)
    r2.put_json(f"{prefix}/manifest.json", merged, CC_MANIFEST)

    pairs = sorted(manifest_pairs(incr))
    log.info("catch-up uploaded %d frame(s) for %s + merged manifest under %s",
             len(fresh), " ".join("/".join(p) for p in pairs) or "(none)", prefix)
    return {"frames": len(fresh),
            "storms": sorted({s.get("id") for s in incr.get("storms", []) or []}),
            "coverage": _manifest_coverage(incr), "pairs": pairs,
            "merged_manifest": merged}


def upload_progressive_batch(r2: R2, out_dir: str, prefix: str,
                             cycle: str) -> dict:
    """ADDITIVE upload of one progressive batch under the CYCLE-SCOPED prefix
    ``{prefix}/{cycle}/...``. PNGs only (same all-or-nothing barrier); the
    manifest is composed + written by the caller AFTER this returns, so a
    listed frame's PNG is always already on R2. NEVER prunes.

    Returns ``{"frames", "incr", "new_frames", "coverage", "storms"}`` where
    ``incr`` is the subprocess's manifest and ``new_frames`` its
    (model, storm, raw_domain, fxx) set."""
    out = Path(out_dir)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise RenderError(f"no manifest at {manifest_path} - nothing to publish")
    incr = json.loads(manifest_path.read_text())
    fresh = _upload_pngs(r2, out, f"{prefix}/{cycle}")
    new_frames = manifest_frames(incr)
    log.info("progressive batch uploaded %d PNG(s) (%d model frame(s)) under "
             "%s/%s", len(fresh), len(new_frames), prefix, cycle)
    return {"frames": len(fresh), "incr": incr, "new_frames": new_frames,
            "coverage": _manifest_coverage(incr),
            "storms": sorted({s.get("id") for s in incr.get("storms", []) or []})}


def _manifest_headers() -> dict:
    """The canonical products/models/domains header arrays, derived from the
    SAME registry the generator's _manifest_skeleton uses (the worker pins the
    hafs-render package, so there is one source of header truth), scoped by
    the worker's HAFS_MODELS / HAFS_DOMAINS / HAFS_PRODUCTS env exactly like
    the render subprocess. Lazy + cached: pulls the GRIB stack."""
    global _HDR_CACHE
    if _HDR_CACHE is not None:
        return _HDR_CACHE
    from hafs_render.generate_hafs_plots import (  # lazy
        DEFAULT_PRODUCTS, DOMAINS, MODEL_LABEL, PRODUCTS)
    models = [m.strip() for m in HAFS_MODELS.split(",") if m.strip()]
    domains = [d.strip() for d in HAFS_DOMAINS.split(",") if d.strip()]
    products = ([p.strip() for p in HAFS_PRODUCTS.split(",") if p.strip()]
                if HAFS_PRODUCTS else list(DEFAULT_PRODUCTS))
    _HDR_CACHE = {
        "product": PRODUCTS[products[0]],
        "products": [PRODUCTS[p] for p in products],
        "models": [{"slug": m, "label": MODEL_LABEL[m]} for m in models],
        "domains": [{"slug": DOMAINS[d][0], "label": DOMAINS[d][1], "raw": d}
                    for d in domains],
        "n_products": len(products),
    }
    return _HDR_CACHE


_HDR_CACHE: Optional[dict] = None


def compose_manifest_v2(entries: list, headers: dict, *,
                        now_iso: str, fxx_step: Optional[int] = None,
                        fxx_end: Optional[int] = None,
                        fxx_pad: int = 3) -> dict:
    """The published manifest: NEW cycles[]-bearing fields for the progressive
    frontend + LEGACY single-cycle fields for an old frontend (zero-blink
    deploy skew). ``entries`` is newest-first cycle entries (at most 2).

    Legacy fields always describe the newest COMPLETE cycle - its prefix is
    BAKED INTO path_template as a literal, so old JS (which substitutes only
    {model}/{storm}/{domain}/{product}/{fxx}) resolves into the cycle-scoped
    keys unchanged. With no complete cycle yet, legacy falls back to the
    newest entry that has frames (a fresh deploy mid-build-out)."""
    fxx_step = FXX_STEP if fxx_step is None else fxx_step
    fxx_end = FXX_END if fxx_end is None else fxx_end
    legacy = next((e for e in entries if not e.get("in_progress")), None)
    if legacy is None:
        legacy = next((e for e in entries if e.get("storms")), None)
    out = {
        "generated_at": now_iso,
        "product": headers["product"],
        "products": headers["products"],
        "models": headers["models"],
        "domains": headers["domains"],
        "fxx_step": fxx_step,
        "fxx_pad": fxx_pad,
        "fxx_end": fxx_end,
        "path_template_cycles":
            "{cycle}/{model}/{storm}/{domain}/{product}/f{fxx}.png",
        "cycles": entries,
        # legacy single-cycle view (old frontend):
        "cycle": legacy["cycle"] if legacy else None,
        "storms": legacy["storms"] if legacy else [],
        "path_template": (
            f"{legacy['cycle']}/{{model}}/{{storm}}/{{domain}}/{{product}}/f{{fxx}}.png"
            if legacy else
            "{model}/{storm}/{domain}/{product}/f{fxx}.png"),
    }
    return out


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


def list_complete_pairs(cycle: str, *, models: str = HAFS_MODELS,
                        domains: str = HAFS_DOMAINS) -> tuple:
    """Every (model, storm, raw_domain) pair whose run is COMPLETE upstream for
    ``cycle`` - i.e. the pair's OWN terminal f126 GRIB exists, which is exactly
    the generator's per-pair accept gate. Storm enumeration mirrors
    build_cycle: the storm set comes from models[0]'s listing. One cheap
    exact-key list call per pair (a handful per poll).

    This is the catch-up's upstream eye: the pairs here that are NOT yet in the
    published manifest are precisely the late arrivals worth a render."""
    from hafs_render.generate_hafs_plots import (  # lazy - pulls the GRIB stack
        MODEL_TOKEN, TERMINAL_FXX, _s3_list, list_storms)
    import requests
    sess = requests.Session()
    date, hh = cycle[:8], cycle[8:]
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    domain_list = [d.strip() for d in domains.split(",") if d.strip()]
    pairs = []
    for storm in list_storms(model_list[0], date, hh, session=sess):
        for model in model_list:
            tok = MODEL_TOKEN[model]
            for domain in domain_list:
                key = (f"{tok}/{date}/{hh}/{storm}.{date}{hh}.{tok}."
                       f"{domain}.f{TERMINAL_FXX:03d}.grb2")
                keys, _ = _s3_list(key, session=sess)
                if key in keys:
                    pairs.append((model, storm, domain))
    return tuple(sorted(pairs))


def resolve_active_cycle(models: str = HAFS_MODELS) -> Optional[str]:
    """The newest cycle id with ANY upstream presence across the configured
    models - a cycle dir exists as soon as its first artifact (storm_info,
    ~1.3 h before f000) posts, which is the pre-announce beacon. Progressive
    mode switches to a new cycle here instead of waiting for completeness.

    OUTAGE != QUIESCENCE: if EVERY model's listing fails, raise (the spine
    treats it as a transient fetch failure and keeps last-known-good). A
    silent ``None`` here would be indistinguishable from a genuinely empty
    bucket and would let the quiescence gate blank the live manifest during
    an NOAA/network blip."""
    from hafs_render.generate_hafs_plots import list_dates, list_hours  # lazy
    import requests
    sess = requests.Session()
    best: Optional[str] = None
    listed_ok = 0
    for model in [m.strip() for m in models.split(",") if m.strip()]:
        try:
            dates = list_dates(model, session=sess)
        except Exception as e:  # noqa: BLE001 - one model's listing failing
            log.warning("resolve_active_cycle: %s listing failed: %s", model, e)
            continue
        listed_ok += 1
        for date in reversed(dates[-4:]):
            hours = list_hours(model, date, session=sess)
            if hours:
                cand = f"{date}{hours[-1]}"
                if best is None or cand > best:
                    best = cand
                break
    if listed_ok == 0:
        raise pf.TransientFetchError(
            "resolve_active_cycle: every model listing failed - treating as "
            "an outage (last-known-good manifest kept)")
    return best


def cycle_age_hours(cycle: str, now: dt.datetime) -> Optional[float]:
    """Age of a ``YYYYMMDDHH`` cycle id at ``now`` (naive or aware), or None
    when the id doesn't parse (never let a malformed dir trip the gate)."""
    try:
        t = dt.datetime.strptime(cycle, "%Y%m%d%H")
    except (ValueError, TypeError):
        return None
    if now.tzinfo is not None:
        now = now.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return (now - t).total_seconds() / 3600.0


_FXX_GRB_RE = re.compile(r"\.f(\d{3})\.grb2$")
_FXX_IDX_RE = re.compile(r"\.f(\d{3})\.grb2\.idx$")


def _needs_sat() -> bool:
    """True when the effective product set includes a simulated-satellite
    product (the default set does), so a frame is only 'posted' once its .sat
    sibling is up too - the union ingest hard-requires it."""
    if not HAFS_PRODUCTS:
        return True
    prods = {p.strip() for p in HAFS_PRODUCTS.split(",")}
    return bool(prods & {"clean_ir", "water_vapor"})


def list_posted_frames(cycle: str, *, models: str = HAFS_MODELS,
                       domains: str = HAFS_DOMAINS) -> tuple:
    """Every (model, storm, raw_domain, fxx) RENDERABLE upstream for ``cycle``:
    the .atm grb2+idx exist AND (when sim-sat products are in scope) the
    sibling .sat grb2+idx for the same fxx exist - both verified empirically
    to post essentially together, but a frame must never plan before its
    inputs are complete (the union ingest hard-raises on a missing .sat).

    One listing per (model, storm, family): 2 models x S storms x 4 families
    ~ 8-16 cheap list calls per poll. Storms come from each model's OWN
    listing (a model's pair exists iff that model has files)."""
    from hafs_render.generate_hafs_plots import (  # lazy
        MODEL_TOKEN, _s3_list, list_storms)
    import requests
    sess = requests.Session()
    date, hh = cycle[:8], cycle[8:]
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    domain_list = [d.strip() for d in domains.split(",") if d.strip()]
    need_sat = _needs_sat()
    frames = []
    for model in model_list:
        tok = MODEL_TOKEN[model]
        try:
            storms = list_storms(model, date, hh, session=sess)
        except Exception as e:  # noqa: BLE001
            log.warning("list_posted_frames: %s storm listing failed: %s",
                        model, e)
            continue
        for storm in storms:
            fam_ready: dict = {}
            fams = set()
            for domain in domain_list:
                fams.add(domain)                       # e.g. storm.atm
                if need_sat:
                    fams.add(domain.split(".")[0] + ".sat")
            for fam in sorted(fams):
                prefix = (f"{tok}/{date}/{hh}/{storm}.{date}{hh}.{tok}"
                          f".{fam}.f")
                keys, _ = _s3_list(prefix, session=sess)
                grb = {int(m.group(1)) for k in keys
                       if (m := _FXX_GRB_RE.search(k))}
                idx = {int(m.group(1)) for k in keys
                       if (m := _FXX_IDX_RE.search(k))}
                fam_ready[fam] = grb & idx
            for domain in domain_list:
                ready = fam_ready.get(domain, set())
                if need_sat:
                    ready = ready & fam_ready.get(
                        domain.split(".")[0] + ".sat", set())
                for fxx in sorted(ready):
                    if fxx % FXX_STEP == 0 and fxx <= FXX_END:
                        frames.append((model, storm, domain, fxx))
    return tuple(sorted(frames))


def _group_missing(missing) -> dict:
    """Group missing (model, storm, raw_domain) pairs by (model, storm) so each
    catch-up subprocess renders EXACTLY its pairs: the generator's
    --models/--storm/--domains filters are a cross-product, so a per-(model,
    storm) invocation carrying just that group's domains can never re-render a
    completed pair."""
    groups: dict = {}
    for model, storm, domain in sorted(missing):
        groups.setdefault((model, storm), []).append(domain)
    return groups


def _set_skip_reason(summary: dict, pair: tuple, reason: str) -> None:
    """Update (or add) one pair's entry in render_summary's skipped_pairs."""
    sp = summary.setdefault("skipped_pairs", [])
    for e in sp:
        if (e.get("model"), e.get("storm"), e.get("domain")) == pair:
            e["reason"] = reason
            return
    sp.append({"model": pair[0], "storm": pair[1], "domain": pair[2],
               "reason": reason})


def _fold_mem_peak(summary: dict, mem: Optional[dict], label: str,
                   clock: Callable[[], dt.datetime]) -> None:
    """Track the cycle-level memory high-water mark across batches: the single
    worst render-tree peak seen this cycle, with which batch owned it. THIS is
    the VPS-sizing number ("does the worker fit in N GB") - per-batch detail
    stays in the frame_batches/catchups audit entries."""
    if not mem or not mem.get("peak_tree_rss_mb"):
        return
    cur = summary.get("mem_peak") or {}
    if mem["peak_tree_rss_mb"] > (cur.get("tree_rss_mb") or 0):
        summary["mem_peak"] = {
            "tree_rss_mb": mem.get("peak_tree_rss_mb"),
            "procs": mem.get("peak_procs"),
            "largest_proc_rss_mb": mem.get("peak_proc_rss_mb"),
            "batch": label,
            "utc": pf.iso_z(clock()),
        }


def _fold_catchup_summary(summary: dict, upcov: dict, new_pairs: list,
                          clock: Callable[[], dt.datetime]) -> None:
    """Fold one catch-up group's result into the cycle's render_summary payload
    ADDITIVELY: bump the frame total, append/replace per-pair coverage, drop
    the now-rendered pairs from skipped_pairs, accumulate the run counters, and
    append a ``catchups`` audit entry so late arrivals are observable."""
    summary["frames"] = (summary.get("frames") or 0) + (upcov.get("frames") or 0)
    summary["storms"] = sorted(set(summary.get("storms") or [])
                               | set(upcov.get("storms") or []))

    def ckey(c):
        return (c.get("model"), c.get("storm"), c.get("domain"))

    new_cov = upcov.get("coverage") or []
    new_keys = {ckey(c) for c in new_cov}
    summary["coverage"] = ([c for c in (summary.get("coverage") or [])
                            if ckey(c) not in new_keys] + new_cov)
    rendered = set(new_pairs)
    summary["skipped_pairs"] = [
        e for e in (summary.get("skipped_pairs") or [])
        if (e.get("model"), e.get("storm"), e.get("domain")) not in rendered]
    for key in ("ingest", "render"):
        inc = upcov.get(key) or {}
        if inc:
            tot = summary.setdefault(key, {})
            for f in ("ok", "total", "failed"):
                if f in inc:
                    tot[f] = (tot.get(f) or 0) + (inc.get(f) or 0)
    mem = upcov.get("mem") or {}
    summary.setdefault("catchups", []).append({
        "utc": pf.iso_z(clock()),
        "pairs": ["/".join(p) for p in new_pairs],
        "frames": upcov.get("frames"),
        "render_seconds": upcov.get("render_seconds"),
        "tree_rss_mb": mem.get("peak_tree_rss_mb"),
        "largest_proc_rss_mb": mem.get("peak_proc_rss_mb"),
        "procs": mem.get("peak_procs"),
    })
    _fold_mem_peak(summary, mem,
                   "catchup " + ",".join("/".join(p) for p in new_pairs[:3]),
                   clock)


# ---------------------------------------------------------------------------
# The HAFS Source (one Source; HMON later = a 2nd Source on the same engine).
# ---------------------------------------------------------------------------
def make_hafs_source(r2: R2, *, prefix: str = HAFS_R2_PREFIX,
                     cycle_resolver: Optional[Callable[[], Optional[str]]] = None,
                     render_fn: Optional[Callable[..., dict]] = None,
                     uploader: Optional[Callable[[R2, str, str], dict]] = None,
                     catchup_uploader: Optional[Callable[..., dict]] = None,
                     complete_pairs_fn: Optional[Callable[[str], tuple]] = None,
                     catchup_max_attempts: int = HAFS_CATCHUP_MAX_ATTEMPTS,
                     out_dir_factory: Optional[Callable[[str], str]] = None,
                     diagnoser: Optional[Callable[[str], str]] = None,
                     clock: Callable[[], dt.datetime] = pf.utcnow) -> pf.Source:
    """Build the HAFS Source. ``cycle_resolver`` / ``render_fn`` / ``uploader`` /
    ``catchup_uploader`` / ``complete_pairs_fn`` / ``out_dir_factory`` /
    ``diagnoser`` are injectable so the offline tests exercise the change-gate,
    watchdog-abort, prune, and intra-cycle catch-up WITHOUT a real render, R2,
    or network."""
    resolve = cycle_resolver or resolve_latest_complete_cycle
    render = render_fn or run_render_subprocess
    upload = uploader or upload_cycle
    cu_upload = catchup_uploader or upload_catchup
    diagnose = diagnoser or diagnose_ingest
    pairs_fn = complete_pairs_fn or (list_complete_pairs if HAFS_CATCHUP
                                     else (lambda cycle: ()))

    # The catch-up ledger for the CURRENT cycle (in-memory on purpose: a worker
    # restart re-renders the current cycle in full anyway, which rebuilds it).
    # ``rendered`` is derived from OUTPUT manifests (ground truth), so pairs the
    # full render dropped (failed ingest, mid-upload skip) stay 'missing' and
    # the catch-up loop heals them too, not just upstream late arrivals.
    state = {"cycle": None, "manifest": None, "summary": None,
             "rendered": set(), "attempts": {}, "given_up": set()}

    def fetch():
        # Cheap: S3 listings to find the newest complete cycle + which of its
        # (model, storm, domain) pairs are complete upstream. None off-season
        # is a legitimate 'nothing now' (no retry burns), not an error.
        cycle = resolve()
        if not cycle:
            return {"cycle": None, "complete_pairs": ()}
        return {"cycle": cycle, "complete_pairs": tuple(pairs_fn(cycle))}

    def change_key(data):
        # New data iff (a) a brand-new COMPLETE cycle appeared, or (b) the same
        # cycle gained newly-complete upstream pairs - the intra-cycle catch-up
        # trigger. A still-latest cycle with no upstream movement takes the
        # spine's cheap path and is NOT re-rendered.
        if not data["cycle"]:
            return None
        return (data["cycle"], data["complete_pairs"])

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
        if cycle != state["cycle"]:
            _full_cycle(cycle)
            # GATE-REOPEN GUARD: a pair complete upstream BEFORE this fetch but
            # dropped by the full render (OOM stragglers, a listing flake - the
            # generator exits 0 as long as ANY frame rendered) leaves no
            # upstream change to flip the signature, so without this it would
            # be stranded until the next cycle. Raise AFTER publishing: the
            # spine holds the signature and the next poll routes the leftovers
            # through the catch-up path (bounded by the attempts cap).
            leftover = set(ctx.data["complete_pairs"]) - state["rendered"]
            if leftover:
                raise RenderError(
                    "cycle published but %d upstream-complete pair(s) produced "
                    "no frames: %s - holding signature so catch-up retries"
                    % (len(leftover), sorted(leftover)))
        else:
            _catchup_cycle(cycle, set(ctx.data["complete_pairs"]))

    def _full_cycle(cycle):
        def run(out_dir):
            manifest, summary = _render_and_upload(
                r2, prefix, cycle, out_dir, render, upload, diagnose, clock)
            state.update(cycle=cycle, manifest=manifest, summary=summary,
                         rendered=manifest_pairs(manifest or {}),
                         attempts={}, given_up=set())
        if out_dir_factory is not None:
            run(out_dir_factory(cycle))
        else:
            with tempfile.TemporaryDirectory(prefix=f"hafs_{cycle}_") as td:
                run(str(Path(td) / "hafs"))

    def _catchup_cycle(cycle, complete):
        # INTRA-CYCLE CATCH-UP: the cycle we already rendered gained newly
        # complete upstream pairs (a late HAFS-B, a late storm). Render ONLY the
        # missing ones, publish additively, never prune, never re-render a
        # completed pair. A group failure raises -> the spine holds the
        # signature and ONLY the still-missing pairs are retried next poll
        # (earlier groups' progress is already in the ledger).
        missing = complete - state["rendered"] - state["given_up"]
        if not missing:
            return    # upstream moved but nothing actionable; signature advances
        if state["manifest"] is None:
            # Unreachable while cycle == state["cycle"] implies a successful
            # full render - but never catch-up onto an unknown base.
            log.warning("catch-up skipped: no base manifest for cycle %s", cycle)
            return
        summary = state["summary"] or {}

        def write_summary():
            summary["generated_utc"] = pf.iso_z(clock())
            state["summary"] = summary
            try:
                r2.put_json(f"{prefix}/render_summary.json", summary, CC_HEALTH)
            except Exception:  # noqa: BLE001 - best-effort, like the full path
                pass

        for (model, storm), doms in sorted(_group_missing(missing).items()):
            runnable = []
            for d in doms:
                p = (model, storm, d)
                state["attempts"][p] = state["attempts"].get(p, 0) + 1
                if state["attempts"][p] > catchup_max_attempts:
                    state["given_up"].add(p)
                    _set_skip_reason(summary, p, "catch-up abandoned after "
                                     f"{catchup_max_attempts} failed attempts")
                    log.warning("catch-up giving up on %s %s %s after %d attempts",
                                model, storm, d, catchup_max_attempts)
                else:
                    runnable.append(d)
            if not runnable:
                write_summary()
                continue
            log.info("catch-up: late pair(s) complete upstream - rendering "
                     "%s %s %s", model, storm, ",".join(runnable))
            with ProgressHeartbeat(r2, prefix, f"{cycle} catchup {model}/{storm}",
                                   clock=clock):
                upcov = _run_catchup_group(cycle, model, storm, runnable)
            new_pairs = [tuple(p) for p in upcov.get("pairs", [])]
            state["rendered"] |= set(new_pairs)
            state["manifest"] = upcov.get("merged_manifest") or state["manifest"]
            _fold_catchup_summary(summary, upcov, new_pairs, clock)
            write_summary()
            # GATE-REOPEN GUARD (same as the full path): the subprocess can
            # exit 0 with one of its pairs producing zero frames (the exit-1
            # gate is whole-run n_ok==0, not per-pair). Upstream won't change
            # for an already-complete pair, so returning normally here would
            # advance the signature and strand it. Raise AFTER publishing the
            # pairs that did land - the spine holds the signature, the next
            # poll retries ONLY the dropped pair(s), and the attempts cap
            # (already incremented above) bounds a persistent failure into an
            # audited give-up.
            dropped = [(model, storm, d) for d in runnable
                       if (model, storm, d) not in state["rendered"]]
            if dropped:
                raise RenderError(
                    "catch-up subprocess exited 0 but produced no frames for "
                    f"{dropped} - holding signature for retry "
                    f"(attempt {max(state['attempts'][p] for p in dropped)}"
                    f"/{catchup_max_attempts})")

    def _run_catchup_group(cycle, model, storm, domains_g):
        def run(out_dir):
            rsum = render(cycle, out_dir, models=model,
                          domains=",".join(domains_g), storm=storm) or {}
            upcov = cu_upload(r2, out_dir, prefix, state["manifest"]) or {}
            upcov.setdefault("render_seconds", rsum.get("render_seconds"))
            upcov.setdefault("ingest", rsum.get("ingest"))
            upcov.setdefault("render", rsum.get("render"))
            upcov.setdefault("mem", rsum.get("mem"))
            return upcov
        if out_dir_factory is not None:
            return run(out_dir_factory(f"{cycle}-catchup-{model}-{storm}"))
        with tempfile.TemporaryDirectory(prefix=f"hafs_cu_{cycle}_") as td:
            return run(str(Path(td) / "hafs"))

    return pf.Source(name="hafs", fetch=fetch, change_key=change_key,
                     process=process, valid_time=valid_time)


# ---------------------------------------------------------------------------
# The PROGRESSIVE HAFS Source (HAFS_PROGRESSIVE=true): frame-granular - render
# forecast hours AS THEY POST, publish additively per poll batch under
# cycle-scoped keys, complete + prune when every posted pair reaches f126.
# The pair-level source above stays in-tree as the rollback.
# ---------------------------------------------------------------------------
def _fold_batch_summary(summary: dict, upres: dict, batch_label: str,
                        clock: Callable[[], dt.datetime]) -> None:
    """Fold one progressive batch into the cycle's render_summary payload:
    cumulative counters + a capped frame_batches audit trail."""
    summary["frames"] = (summary.get("frames") or 0) + (upres.get("frames") or 0)
    summary["storms"] = sorted(set(summary.get("storms") or [])
                               | set(upres.get("storms") or []))
    for key in ("ingest", "render"):
        inc = upres.get(key) or {}
        if inc:
            tot = summary.setdefault(key, {})
            for f in ("ok", "total", "failed"):
                if f in inc:
                    tot[f] = (tot.get(f) or 0) + (inc.get(f) or 0)
    if upres.get("render_seconds"):
        summary["render_seconds"] = round(
            (summary.get("render_seconds") or 0) + upres["render_seconds"], 1)
    batches = summary.setdefault("frame_batches", [])
    nf = sorted(upres.get("new_frames") or ())
    span = (f"f{nf[0][3]:03d}-f{nf[-1][3]:03d}" if nf else "(none)")
    mem = upres.get("mem") or {}
    batches.append({"utc": pf.iso_z(clock()), "batch": batch_label,
                    "model_frames": len(nf), "fxx_span": span,
                    "frames": upres.get("frames"),
                    "render_seconds": upres.get("render_seconds"),
                    # Render-tree memory peak for THIS batch's subprocess
                    # (parent samples the child's process group; see
                    # _proc_tree_rss_mb). Sums to the Railway RAM-minutes.
                    "tree_rss_mb": mem.get("peak_tree_rss_mb"),
                    "largest_proc_rss_mb": mem.get("peak_proc_rss_mb"),
                    "procs": mem.get("peak_procs")})
    del batches[:-60]   # cap the audit trail
    _fold_mem_peak(summary, mem, batch_label, clock)


def make_progressive_source(r2: R2, *, prefix: str = HAFS_R2_PREFIX,
                            cycle_resolver: Optional[Callable[[], Optional[str]]] = None,
                            posted_frames_fn: Optional[Callable[[str], tuple]] = None,
                            render_fn: Optional[Callable[..., dict]] = None,
                            batch_uploader: Optional[Callable[..., dict]] = None,
                            headers_fn: Optional[Callable[[], dict]] = None,
                            catchup_max_attempts: int = HAFS_CATCHUP_MAX_ATTEMPTS,
                            out_dir_factory: Optional[Callable[[str], str]] = None,
                            clock: Callable[[], dt.datetime] = pf.utcnow) -> pf.Source:
    """Build the frame-granular progressive Source. All collaborators are
    injectable so the offline tests exercise the ledger, batching, completion,
    prune, bootstrap, and zero-blink manifest WITHOUT a render, R2, or
    network."""
    resolve = cycle_resolver or resolve_active_cycle
    posted_fn = posted_frames_fn or list_posted_frames
    render = render_fn or run_render_subprocess
    upload_batch = batch_uploader or upload_progressive_batch
    headers = headers_fn or _manifest_headers

    # The frame ledger for the ACTIVE cycle + the retained previous-complete
    # entry (the frontend's flip-back target). In-memory, but bootstrapped
    # from the live manifest on restart so a deploy neither re-renders the
    # whole active cycle nor loses the flip-back entry.
    state = {"cycle": None, "entry": None, "prev_entry": None, "summary": None,
             "rendered": set(), "attempts": {}, "given_up": set(),
             "complete": False, "bootstrapped": False}

    def _grid_n() -> int:
        return FXX_END // FXX_STEP + 1

    def _entries() -> list:
        out = []
        if state["entry"] is not None:
            out.append(state["entry"])
        if (state["prev_entry"] is not None
                and (not out or state["prev_entry"]["cycle"] != out[0]["cycle"])):
            out.append(state["prev_entry"])
        return out

    def _publish_manifest() -> None:
        man = compose_manifest_v2(_entries(), headers(),
                                  now_iso=pf.iso_z(clock()))
        r2.put_json(f"{prefix}/manifest.json", man, CC_MANIFEST)

    def _write_summary() -> None:
        s = state["summary"]
        if s is None:
            return
        s["generated_utc"] = pf.iso_z(clock())
        s["frames_done"] = len(state["rendered"])
        s["complete"] = state["complete"]
        try:
            r2.put_json(f"{prefix}/render_summary.json", s, CC_HEALTH)
        except Exception:  # noqa: BLE001 - best-effort
            pass
        # ALSO under a cycle-scoped key: the flat key above is reset when the
        # next cycle goes active, which would erase the previous cycle's
        # frame_batches/mem audit ~6h after the fact. The cycle-scoped copy
        # makes per-cycle telemetry durable (read 18z/00z/06z side-by-side
        # the next morning for the VPS-sizing summary). The completion prune
        # never touches it (it deletes .png keys only), so these accumulate -
        # 4/day at ~1-2 KB each, negligible.
        cyc = s.get("cycle")
        if cyc:
            try:
                r2.put_json(f"{prefix}/{cyc}/render_summary.json", s, CC_HEALTH)
            except Exception:  # noqa: BLE001 - best-effort
                pass

    def _bootstrap() -> None:
        """Recover the ledger + flip-back entry from the live manifest once
        per process lifetime. Best-effort: any failure = cold start."""
        state["bootstrapped"] = True
        man = r2.get_json(f"{prefix}/manifest.json")
        if not man:
            return
        if not man.get("cycles"):
            # LIVE v1 manifest (the pre-progressive worker's): its complete
            # cycle's frames live at FLAT keys. Migrate ONCE - server-side
            # copy each listed frame under the cycle-scoped prefix, then
            # carry the cycle as the prev (flip-back/legacy) entry. Without
            # this, the first v2 publish would either show old frontends an
            # empty manifest or bake a path prefix whose keys don't exist.
            cyc = man.get("cycle")
            storms = man.get("storms") or []
            if cyc and storms:
                pad = int(man.get("fxx_pad") or 3)
                n = _migrate_flat_cycle(cyc, storms, pad)
                state["prev_entry"] = {
                    "cycle": cyc, "in_progress": False,
                    "frames_done": len(manifest_frames(man)),
                    "frames_expected": len(manifest_frames(man)),
                    "storms": storms,
                }
                log.info("v1 manifest bootstrap: migrated %d frame(s) of "
                         "cycle %s to cycle-scoped keys", n, cyc)
            return
        entries = man["cycles"]
        newest = entries[0]
        prev = next((e for e in entries[1:] if not e.get("in_progress")), None)
        state["prev_entry"] = prev
        if newest.get("in_progress"):
            state.update(cycle=newest["cycle"], entry=newest,
                         complete=False,
                         rendered=manifest_frames(newest))
            state["summary"] = {"cycle": newest["cycle"], "progressive": True,
                                "bootstrapped": True}
        else:
            # Newest is complete: adopt it as the CURRENT cycle state, not
            # just the flip-back target. Upstream keeps resolving this same
            # cycle until the next one posts, and process() compares against
            # state["cycle"] - leaving it None here made an IDLE-WINDOW
            # restart _begin_cycle() the already-finished cycle: the live
            # manifest was replaced by an empty pre-announce (the complete
            # cycle DELISTED from /models/) and the whole cycle re-rendered
            # from a cold ledger (observed on the 2026-06-05 18:43Z deploy,
            # 12Z complete + 18z not yet posting). With the entry + rendered
            # ledger adopted, the same-cycle re-resolve is the no-op this
            # branch always intended, and the next REAL cycle archives this
            # one via _begin_cycle's complete-entry branch exactly as if the
            # process had never restarted.
            state.update(cycle=newest["cycle"], entry=newest, complete=True,
                         rendered=manifest_frames(newest))
            state["summary"] = {"cycle": newest["cycle"], "progressive": True,
                                "bootstrapped": True}
        log.info("bootstrapped from live manifest: active=%s (%d frames), "
                 "prev=%s", state["cycle"],
                 len(state["rendered"]),
                 state["prev_entry"] and state["prev_entry"]["cycle"])

    def _migrate_flat_cycle(cyc, storms, pad) -> int:
        """Server-side copy of every frame a v1 manifest lists from its FLAT
        key to the cycle-scoped key. All-or-nothing is NOT required: a copy
        failure just leaves that frame missing under the new prefix (the
        frontend's availability grid greys it); copies are idempotent."""
        tasks = []
        for s in storms:
            sid = s.get("id")
            for mdl, doms in (s.get("frames") or {}).items():
                for dom_slug, prods in (doms or {}).items():
                    for prod, fxx_list in (prods or {}).items():
                        for f in fxx_list or []:
                            rel = (f"{mdl}/{sid}/{dom_slug}/{prod}/"
                                   f"f{int(f):0{pad}d}.png")
                            tasks.append((f"{prefix}/{rel}",
                                          f"{prefix}/{cyc}/{rel}"))
        done = 0
        with cf.ThreadPoolExecutor(max_workers=HAFS_UPLOAD_WORKERS) as ex:
            for ok in ex.map(lambda p_: r2.copy(p_[0], p_[1]), tasks):
                done += 1 if ok else 0
        return done

    def fetch():
        if not state["bootstrapped"]:
            try:
                _bootstrap()
            except Exception as e:  # noqa: BLE001 - never block polling
                log.warning("bootstrap failed (cold start): %s", e)
                state["bootstrapped"] = True
        cycle = resolve()
        if not cycle:
            # Clean listings with no cycle dirs in the lookback at all:
            # deep off-season — same quiescence handling as a stale cycle.
            return {"cycle": None, "posted": (), "quiet": True,
                    "stale_cycle": None}
        age_h = cycle_age_hours(cycle, clock())
        if age_h is not None and age_h > HAFS_QUIET_AFTER_H:
            # Genuine quiescence: the newest run NOAA ever published is old
            # news (storm dissipated / season over). Signal publish-empty.
            # Listing failures never get here — resolve() raises on outage.
            return {"cycle": None, "posted": (), "quiet": True,
                    "stale_cycle": cycle}
        return {"cycle": cycle, "posted": tuple(posted_fn(cycle))}

    def change_key(data):
        # Fires on a NEW cycle dir (pre-announce beacon) and on every newly
        # posted upstream frame - the progressive trigger. Quiet upstream =
        # the spine's cheap path.
        if data.get("quiet"):
            # Stable while quiet -> process() publishes the empty manifest
            # ONCE per quiet episode; re-keys when upstream resumes (or a
            # different stale cycle id appears).
            return ("quiet", data.get("stale_cycle"))
        if not data["cycle"]:
            return None
        return (data["cycle"], data["posted"])

    def valid_time(data):
        c = data["cycle"]
        if not c:
            return None
        try:
            return dt.datetime.strptime(c, "%Y%m%d%H")
        except ValueError:
            return None

    def process(ctx: pf.ProcessContext):
        if ctx.data.get("quiet"):
            _go_quiet(ctx.data.get("stale_cycle"))
            return
        cycle = ctx.data["cycle"]
        posted = set(ctx.data["posted"])
        if not cycle:
            log.info("no active cycle - nothing to render")
            return
        if cycle != state["cycle"]:
            _begin_cycle(cycle)
        _render_delta(cycle, posted)
        _update_completion(posted)

    def _go_quiet(stale_cycle) -> None:
        """Genuine upstream quiescence (clean listing; newest cycle older
        than HAFS_QUIET_AFTER_H, or no cycles at all): the storm is over.
        Publish the EMPTY manifest the frontend self-hides on — a dead
        storm's frozen last run must never sit on /models/ looking current.
        In-memory entries reset so a restart's bootstrap (which reads the
        live manifest) agrees. Durable per-cycle render summaries are left
        untouched. PNG frames are left in place (invisible once the
        manifest is empty; the next active cycle's completion prune
        retires them)."""
        was_empty = state["entry"] is None and state["prev_entry"] is None
        state.update(cycle=None, entry=None, prev_entry=None,
                     complete=False, rendered=set(), attempts={},
                     given_up=set())
        _publish_manifest()
        log.info("upstream quiet (newest cycle %s, older than %.0fh) - "
                 "published EMPTY manifest%s", stale_cycle or "(none)",
                 HAFS_QUIET_AFTER_H,
                 " (was already empty)" if was_empty else "")

    def _begin_cycle(cycle):
        if state["entry"] is not None and state["complete"]:
            # Archive the completed cycle as the flip-back target. A
            # superseded INCOMPLETE cycle is dropped (its prefix is pruned at
            # the next completion); the older complete entry stays.
            state["prev_entry"] = state["entry"]
        state.update(cycle=cycle, complete=False, rendered=set(),
                     attempts={}, given_up=set())
        state["entry"] = {"cycle": cycle, "in_progress": True,
                          "frames_done": 0, "frames_expected": None,
                          "started_utc": pf.iso_z(clock()), "storms": []}
        state["summary"] = {"cycle": cycle, "progressive": True}
        log.info("new active cycle %s - pre-announcing", cycle)
        _publish_manifest()
        _write_summary()

    def _render_delta(cycle, posted):
        missing = posted - state["rendered"] - state["given_up"]
        if not missing:
            return
        groups: dict = {}
        for (m, s, d, f) in sorted(missing):
            groups.setdefault((m, s), {}).setdefault(d, set()).add(f)
        group_errors = []
        # One subprocess per (model, storm, IDENTICAL-fxx-set of domains): the
        # generator's --domains x --only-fxx is a cross-product, so domains
        # with ASYMMETRIC missing sets must not share an invocation or the
        # overlap would re-render already-completed frames. Domains post near
        # lock-step upstream, so the common case stays one invocation.
        for (model, storm), doms in sorted(groups.items()):
            runnable_by_set: dict = {}
            for d, fxxs in sorted(doms.items()):
                keep = set()
                for f in sorted(fxxs):
                    key = (model, storm, d, f)
                    state["attempts"][key] = state["attempts"].get(key, 0) + 1
                    if state["attempts"][key] > catchup_max_attempts:
                        state["given_up"].add(key)
                        _set_skip_reason(
                            state["summary"], (model, storm, d),
                            f"frame f{f:03d} abandoned after "
                            f"{catchup_max_attempts} failed attempts")
                        log.warning("progressive giving up on %s %s %s f%03d",
                                    model, storm, d, f)
                    else:
                        keep.add(f)
                if keep:
                    runnable_by_set.setdefault(
                        tuple(sorted(keep)), []).append(d)
            if not runnable_by_set:
                _write_summary()
                continue
            # EVERY subgroup executes EVERY poll (errors collected, raised
            # at the end): the attempts counter above was bumped for the whole
            # group, so an early subgroup's failure must not strand the later
            # subgroups - otherwise their frames burn attempts without ever
            # running and get falsely abandoned (review-confirmed critical).
            errors = []
            for fxx_set, set_doms in sorted(runnable_by_set.items()):
                # Render the missing fxx ASCENDING in small chunks, publishing
                # the manifest after EACH so the building cycle populates
                # hour-by-hour (the frontend re-polls and merges). Without this
                # the whole storm-delta renders in one subprocess and only
                # appears at F126. PROGRESSIVE_FXX_CHUNK<=0 -> one chunk (legacy).
                fxx_ascending = sorted(fxx_set)
                step = PROGRESSIVE_FXX_CHUNK if PROGRESSIVE_FXX_CHUNK > 0 else len(fxx_ascending)
                for ci in range(0, len(fxx_ascending), max(step, 1)):
                    chunk = fxx_ascending[ci:ci + step]
                    runnable = [(model, storm, d, f) for d in set_doms
                                for f in chunk]
                    log.info("progressive batch: %s %s %s fxx=%s",
                             model, storm, ",".join(set_doms),
                             ",".join(str(f) for f in chunk))
                    try:
                        with ProgressHeartbeat(
                                r2, prefix,
                                f"{cycle} progressive {model}/{storm}",
                                clock=clock):
                            upres = _run_batch(cycle, model, storm, set_doms,
                                               set(chunk))
                    except RenderError as e:
                        errors.append(str(e))
                        continue
                    state["rendered"] |= set(upres.get("new_frames") or ())
                    _merge_entry(upres.get("incr") or {})
                    _refresh_counts(posted)
                    _fold_batch_summary(state["summary"], upres,
                                        f"{model}/{storm}", clock)
                    # Manifest AFTER the PNG barrier (inside _run_batch) - a
                    # listed frame's PNG is already on R2.
                    _publish_manifest()
                    _write_summary()
                    # GATE-REOPEN GUARD (frame level): the subprocess exits 0 as
                    # long as anything rendered; a planned frame that produced
                    # nothing must hold the signature so it retries (bounded by
                    # the attempts cap).
                    dropped = [k for k in runnable if k not in state["rendered"]]
                    if dropped:
                        errors.append(
                            "batch exited 0 but produced no frames for "
                            f"{dropped[:6]}{'...' if len(dropped) > 6 else ''}")
            if errors:
                group_errors.append(
                    f"{model}/{storm}: " + "; ".join(errors)[:600])
        if group_errors:
            raise RenderError(
                f"{len(group_errors)} progressive group failure(s) - holding "
                "signature for retry: " + " | ".join(group_errors)[:1500])

    def _run_batch(cycle, model, storm, doms, fxxs):
        only = ",".join(str(f) for f in sorted(fxxs))
        def run(out_dir):
            rsum = render(cycle, out_dir, models=model,
                          domains=",".join(doms), storm=storm,
                          only_fxx=only) or {}
            upres = upload_batch(r2, out_dir, prefix, cycle) or {}
            upres.setdefault("render_seconds", rsum.get("render_seconds"))
            upres.setdefault("ingest", rsum.get("ingest"))
            upres.setdefault("render", rsum.get("render"))
            upres.setdefault("mem", rsum.get("mem"))
            return upres
        if out_dir_factory is not None:
            return run(out_dir_factory(f"{cycle}-prog-{model}-{storm}"))
        with tempfile.TemporaryDirectory(prefix=f"hafs_pg_{cycle}_") as td:
            return run(str(Path(td) / "hafs"))

    def _merge_entry(incr):
        entry = state["entry"]
        merged = merge_manifest({"cycle": entry["cycle"],
                                 "storms": entry["storms"]}, incr)
        entry["storms"] = merged["storms"]
        state["summary"]["coverage"] = _manifest_coverage(
            {"storms": entry["storms"]})

    def _refresh_counts(posted):
        entry = state["entry"]
        if entry is None:
            return
        pairs = {(m, s, d) for (m, s, d, f) in (posted | state["rendered"])}
        entry["frames_done"] = len(state["rendered"])
        entry["frames_expected"] = (len(pairs) * _grid_n()) if pairs else None

    def _update_completion(posted):
        entry = state["entry"]
        if entry is None:
            return
        _refresh_counts(posted)
        missing_left = posted - state["rendered"] - state["given_up"]
        posted_pairs = {(m, s, d) for (m, s, d, f) in posted}
        # A pair's terminal counts as RESOLVED when rendered OR abandoned
        # after the attempts cap - completion-with-gaps is honest (the gaps
        # stay greyed in the UI and audited in the summary), while requiring
        # a render would wedge the cycle in_progress forever on one
        # persistently-broken frame (review-confirmed).
        resolved = state["rendered"] | state["given_up"]
        have_terminal = bool(posted_pairs) and all(
            (m, s, d, FXX_END) in resolved for (m, s, d) in posted_pairs)
        complete = (have_terminal and not missing_left
                    and bool(state["rendered"]))
        if complete and not state["complete"]:
            state["complete"] = True
            entry["in_progress"] = False
            log.info("cycle %s COMPLETE (%d model frames) - publishing + prune",
                     entry["cycle"], len(state["rendered"]))
            _publish_manifest()      # legacy fields flip to this cycle
            _write_summary()
            try:
                _prune_retired()
            except Exception as e:  # noqa: BLE001 - prune must never wedge
                log.warning("completion prune failed (retrying next "
                            "completion): %s", e)
        elif not complete and state["complete"]:
            # A late pair re-opened the cycle (e.g. hafsb posting after we
            # completed) - honest flip back; renders resume next polls.
            state["complete"] = False
            entry["in_progress"] = True
            _publish_manifest()
            _write_summary()

    def _prune_retired():
        """Completion prune: delete cycle prefixes not in the kept entries +
        flat (pre-cycle-scoped) legacy PNGs older than FLAT_KEY_TTL_S. The
        manifest key itself and health/summary JSONs are never touched."""
        keep = {e["cycle"] for e in _entries()}
        now = clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt.timezone.utc)
        doomed = []
        for key, lm in r2.list_objects(prefix + "/"):
            if not key.endswith(".png"):
                continue
            rel = key[len(prefix) + 1:]
            m = re.match(r"^(\d{10})/", rel)
            if m:
                if m.group(1) not in keep:
                    doomed.append(key)
            else:
                # flat legacy key: age-gated so a long-lived browser session
                # holding a pre-transition manifest doesn't 404 mid-scrub
                if lm is not None and lm.tzinfo is None:
                    lm = lm.replace(tzinfo=dt.timezone.utc)
                if lm is not None and (now - lm).total_seconds() > FLAT_KEY_TTL_S:
                    doomed.append(key)
        if doomed:
            log.info("completion prune: deleting %d retired PNG(s)", len(doomed))
            r2.delete(doomed)

    return pf.Source(name="hafs", fetch=fetch, change_key=change_key,
                     process=process, valid_time=valid_time)


def _render_and_upload(r2, prefix, cycle, out_dir, render, upload, diagnose,
                       clock) -> tuple:
    """Full-cycle render + 3-pass upload. Returns ``(manifest, summary_payload)``
    so the Source can seed its catch-up ledger (what rendered, what was skipped)
    from the cycle's ground truth."""
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
        payload = {
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
        }
        try:
            r2.put_json(f"{prefix}/render_summary.json", payload, CC_HEALTH)
        except Exception:  # noqa: BLE001
            pass
        try:
            manifest = json.loads((Path(out_dir) / "manifest.json").read_text())
        except Exception:  # noqa: BLE001 - a real upload already validated it
            manifest = None
        return manifest, payload


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
# ---------------------------------------------------------------------------
# Liveness self-kill - the in-box recovery the external ens_watchdog cannot do
# from outside Railway: turn a FROZEN worker into a non-zero exit so ON_FAILURE
# restarts it (a hang is not an exit, so ON_FAILURE never fires on its own).
# ---------------------------------------------------------------------------
_LAST_PROGRESS = [0.0]   # monotonic clock of the last progress signal


def _mark_progress() -> None:
    """Record that the worker is alive - called on every poll heartbeat and every
    render memory-sample (~2 s), so the only thing that stalls it is a real freeze."""
    _LAST_PROGRESS[0] = time.monotonic()


def _liveness_stale(last: float, now: float, timeout_s: float) -> bool:
    """Pure decision (unit-testable): has progress been silent longer than timeout?"""
    return (now - last) > timeout_s


class LivenessWatchdog(threading.Thread):
    """Daemon that self-exits the process (os._exit, non-zero) when no progress
    signal has been seen for ``timeout_s`` - converting a frozen worker (which
    Railway's ON_FAILURE can NOT restart, because a hang is not an exit) into a
    clean non-zero exit it CAN cycle. Belt-and-suspenders with the external
    ens_watchdog (which re-renders via Actions independent of this box)."""

    def __init__(self, timeout_s: float = LIVENESS_TIMEOUT_S,
                 check_s: float = LIVENESS_CHECK_S,
                 on_stall: Callable[[], None] = lambda: os._exit(1)):
        super().__init__(daemon=True, name="hafs-liveness")
        self.timeout_s = timeout_s
        self.check_s = check_s
        self.on_stall = on_stall

    def run(self) -> None:   # pragma: no cover - thread loop
        while True:
            time.sleep(self.check_s)
            if _liveness_stale(_LAST_PROGRESS[0], time.monotonic(), self.timeout_s):
                log.critical("liveness: no progress for >%ds - self-exiting "
                             "NON-ZERO so Railway ON_FAILURE restarts the worker",
                             int(self.timeout_s))
                self.on_stall()
                return


class _HealthSink(pf.Sink):
    """Adapts the R2 client to the Sink protocol so the engine's health
    heartbeat lands at ``{prefix}/poller_health.json`` (the intensity poller's
    sink_heartbeat pattern)."""

    def __init__(self, r2: R2, prefix: str):
        self.r2 = r2
        self.prefix = prefix

    def write(self, key: str, payload: dict) -> None:
        _mark_progress()   # a poll heartbeat = the worker is alive
        self.r2.put_json(f"{self.prefix}/{key}", payload, CC_HEALTH)


def build_engine(r2: R2, *, prefix: str = HAFS_R2_PREFIX,
                 interval_s: float = POLL_INTERVAL_S,
                 progressive: Optional[bool] = None,
                 clock: Callable[[], dt.datetime] = pf.utcnow,
                 sleep: Callable[[float], None] = time.sleep,
                 **source_kwargs) -> pf.PollerEngine:
    """``progressive`` selects the frame-granular source (default: the
    HAFS_PROGRESSIVE env flag); False = the classic complete-pair source -
    the in-tree rollback path."""
    if progressive is None:
        progressive = HAFS_PROGRESSIVE
    health = _HealthSink(r2, prefix)
    maker = make_progressive_source if progressive else make_hafs_source
    source = maker(r2, prefix=prefix, clock=clock, **source_kwargs)
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
    _mark_progress()                 # arm the liveness clock before the first poll
    LivenessWatchdog().start()
    # One-shot observability artifact so the EFFECTIVE config (after env overrides)
    # is verifiable on R2 without Railway log access - confirms the lowered jobs
    # actually took effect on this deploy.
    try:
        r2.put_json(f"{HAFS_R2_PREFIX}/worker_config.json", {
            "started_utc": pf.iso_z(pf.utcnow()),
            "jobs": HAFS_JOBS, "ingest_jobs": HAFS_INGEST_JOBS,
            "jobs_env": _env("HAFS_JOBS"), "jobs_max": HAFS_JOBS_MAX,
            "ingest_jobs_env": _env("HAFS_INGEST_JOBS"),
            "render_timeout_s": RENDER_TIMEOUT_S,
            "liveness_timeout_s": LIVENESS_TIMEOUT_S,
            "poll_interval_s": POLL_INTERVAL_S,
            "progressive": HAFS_PROGRESSIVE,
        }, CC_HEALTH)
    except Exception:  # noqa: BLE001
        pass
    log.info("hafs render poller starting | prefix=%s | jobs=%d ingest_jobs=%d "
             "| interval=%gs | watchdog=%ds | liveness=%ds | stale_after=%gs | mode=%s",
             HAFS_R2_PREFIX, HAFS_JOBS, HAFS_INGEST_JOBS, POLL_INTERVAL_S,
             RENDER_TIMEOUT_S, LIVENESS_TIMEOUT_S, STALE_AFTER_S,
             "progressive" if HAFS_PROGRESSIVE else "complete-pair")
    eng.run_forever()


if __name__ == "__main__":
    main()
