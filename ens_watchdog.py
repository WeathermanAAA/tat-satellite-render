#!/usr/bin/env python3
"""
ens_watchdog.py
---------------
Freshness WATCHDOG for the Ensemble Cyclone Centers ingest (the /models/
enscenters product). It is the ROBUST TRIGGER that replaces sole reliance on
GitHub's scheduled crons, which GitHub silently drops/throttles under load - the
recurring cause of a model getting stuck a cycle behind (e.g. ecens/gefs/fnv3/genc
stuck at 06z while ecaie reached 12z).

DESIGN (deliberately dumb + safe):
    Every ~20 min, fetch the PUBLIC enscenters manifest from R2 and, per model,
    compare its published ``latest`` cycle against the newest 6-hourly cycle that
    SHOULD be available by now (cycle time + a per-model delivery lag). If a model
    is behind, fire its GitHub Actions workflow via the API (workflow_dispatch with
    a BLANK cycle = the never-miss backfill, which advances). The heavy ingest stays
    on free Actions; the watchdog only TRIGGERS.

    The watchdog does NOT replicate each model's completeness gate. It pokes on a
    wall-clock heuristic; the never-miss run itself does the real completeness check
    and publishes only complete cycles (a too-early poke simply no-ops and retries
    next tick). So the watchdog is idempotent and cannot publish bad data. A
    per-model COOLDOWN prevents re-poking while a dispatched run is still in flight.

    workflow_dispatch is an API trigger, NOT a scheduled event, so it is not subject
    to the cron throttling/dropping that causes the staleness.

CONFIG (env, all optional except the token):
    ENS_WATCHDOG_GH_TOKEN   GitHub PAT (fine-grained or classic) with
                            actions:write on WeathermanAAA/Triple-A-Tropics. REQUIRED
                            - without it the watchdog logs and idles (never crashes
                            the host poller).
    ENS_WATCHDOG_INTERVAL_S poll cadence (default 1200 = 20 min)
    ENS_WATCHDOG_REPO       owner/repo (default WeathermanAAA/Triple-A-Tropics)
    ENS_WATCHDOG_DRYRUN     "1" -> decide + log but never dispatch (for staging)
    HAFS_WATCHDOG_LAG_H / _STUCK_BUILD_S / _COOLDOWN_S
                            HAFS-watcher knobs (defaults 8 h / 3600 s / 2400 s).

Run as its own tiny Railway service (railway.watchdog.json: ``python
ens_watchdog.py``) or import ``run_once`` into an existing always-on poller's loop.
Zero heavy deps - just ``requests``.

ALSO WATCHES HAFS: the /models/ HAFS plots are rendered by the tat-satellite-render
HAFS render worker (hafs_render_poller.py). When that worker wedges or OOM-crashes,
Railway's restartPolicy=ON_FAILURE cannot restart a frozen process and gives up
after maxRetries, so the manifest freezes a cycle behind with no recovery (the
~30 h staleness this watcher exists to break). Same pattern as the ensemble models:
``hafs_run_once`` compares the published HAFS cycle to the newest that should exist
and, if behind/stuck, fires ``update-hafs.yml`` (whose render is gated to ALWAYS run
on a workflow_dispatch - a render path independent of the wedged worker).
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import requests

UTC = dt.timezone.utc
log = logging.getLogger("ens_watchdog")

MANIFEST_URL = "https://cdn.triple-a-tropics.com/models/enscenters/manifest.json"

# Per-model: the workflow file to dispatch + the typical delivery LAG (hours after
# cycle time when that model's cycle is usually complete on its source). The lag is
# only a heuristic for WHEN to poke; correctness is the never-miss completeness gate.
# Lags are set slightly generous (poke a touch late rather than waste Actions early).
MODELS: Dict[str, Dict] = {
    "ecens": {"workflow": "update-enscenters.yml", "lag_h": 9},
    "ecaie": {"workflow": "update-aifs-ens.yml",   "lag_h": 6},
    "gefs":  {"workflow": "update-gefs.yml",        "lag_h": 8},
    "fnv3":  {"workflow": "update-fnv3.yml",        "lag_h": 8},
    "genc":  {"workflow": "update-genc.yml",        "lag_h": 8},
}
COOLDOWN_S = 2400.0   # 40 min: don't re-poke a model while its dispatched run is in flight


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


# --- HAFS render watchdog (same pattern; different manifest shape + workflow) ---
HAFS_MANIFEST_URL = "https://cdn.triple-a-tropics.com/models/hafs/manifest.json"
HAFS_WORKFLOW = "update-hafs.yml"
# HAFS runs 00/06/12/18Z, finishes ~6 h after synoptic (cron scheduled ~6 h 53 m
# after); lag 8 h = poke a touch late rather than mid-upload.
HAFS_LAG_H = float(_env("HAFS_WATCHDOG_LAG_H", "8"))
# A cycle legitimately renders for ~20-40 min; in_progress longer than this = a
# wedged build (a dead worker) -> recover regardless of the cycle id.
HAFS_STUCK_BUILD_S = float(_env("HAFS_WATCHDOG_STUCK_BUILD_S", "3600"))
HAFS_COOLDOWN_S = float(_env("HAFS_WATCHDOG_COOLDOWN_S", "2400"))
HAFS_KEY = "hafs"   # last_dispatch key (never collides with an enscenters slug)


# --------------------------------------------------------------------------
# Pure decision logic (no network -> unit-testable)
# --------------------------------------------------------------------------
def expected_latest_cycle(now: dt.datetime, lag_h: float) -> str:
    """Newest 6-hourly cycle (YYYYMMDDHH) that should be available by ``now`` given a
    delivery ``lag_h``: floor(now - lag_h) to the 00/06/12/18 boundary."""
    t = now - dt.timedelta(hours=lag_h)
    return t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")


def manifest_latest(manifest: dict, slug: str) -> Optional[str]:
    for m in (manifest or {}).get("models", []):
        if m.get("slug") == slug:
            return m.get("latest")
    return None


def decide_dispatches(manifest: dict, now: dt.datetime,
                      last_dispatch: Dict[str, dt.datetime],
                      models: Dict[str, Dict] = MODELS,
                      cooldown_s: float = COOLDOWN_S) -> List[Tuple[str, str]]:
    """Return [(slug, reason)] for each model that is BEHIND its expected cycle and
    not within its cooldown. YYYYMMDDHH strings compare lexicographically = by time.
    A model absent from the manifest is treated as behind (bootstrap)."""
    out: List[Tuple[str, str]] = []
    for slug, cfg in models.items():
        exp = expected_latest_cycle(now, cfg["lag_h"])
        latest = manifest_latest(manifest, slug)
        if latest is not None and str(latest) >= exp:
            continue                                   # current (or ahead) -> nothing to do
        last = last_dispatch.get(slug)
        if last is not None and (now - last).total_seconds() < cooldown_s:
            continue                                   # poked recently; let the run finish
        out.append((slug, f"latest={latest} behind expected={exp}"))
    return out


def _parse_z(s) -> Optional[dt.datetime]:
    """Parse an ISO 'YYYY-MM-DDThh:mm:ssZ' into an aware UTC datetime, else None."""
    if not s:
        return None
    try:
        return dt.datetime.strptime(str(s).replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")
    except (ValueError, TypeError):
        return None


def _hafs_newest_entry(manifest: dict) -> Tuple[Optional[str], dict]:
    """The newest cycle of the HAFS manifest: the worker's v2 ``cycles`` list if
    present (else the legacy top-level ``cycle``/``storms``). Returns
    (cycle_str, entry) where entry carries in_progress/started_utc/storms."""
    cycles = (manifest or {}).get("cycles") or []
    if cycles:
        newest = max(cycles, key=lambda c: str(c.get("cycle") or ""))
        return newest.get("cycle"), newest
    return ((manifest or {}).get("cycle"),
            {"in_progress": False, "storms": (manifest or {}).get("storms") or []})


def hafs_manifest_state(manifest: dict) -> Tuple[Optional[str], bool, Optional[dt.datetime], bool]:
    """(latest_cycle, in_progress, started, has_storms) for the newest HAFS cycle."""
    latest, e = _hafs_newest_entry(manifest)
    in_prog = bool(e.get("in_progress"))
    started = _parse_z(e.get("started_utc")) if in_prog else None
    return latest, in_prog, started, bool(e.get("storms"))


def decide_hafs_dispatch(manifest: dict, now: dt.datetime,
                         last_dispatch: Dict[str, dt.datetime], *,
                         lag_h: float = HAFS_LAG_H,
                         stuck_build_s: float = HAFS_STUCK_BUILD_S,
                         cooldown_s: float = HAFS_COOLDOWN_S) -> Optional[str]:
    """Reason string if the HAFS manifest should be re-rendered, else None. Fires when
    (a) the newest cycle is BEHIND the expected cycle AND storms are active (off-season
    has nothing to render -> stay quiet), or (b) the newest cycle is stuck in_progress
    longer than ``stuck_build_s`` (a wedged build = the dead-worker symptom). A
    cooldown prevents re-poking a dispatched run that's still in flight."""
    latest, in_prog, started, has_storms = hafs_manifest_state(manifest)
    exp = expected_latest_cycle(now, lag_h)
    behind = latest is None or str(latest) < exp
    stuck = bool(in_prog and started is not None
                 and (now - started).total_seconds() > stuck_build_s)
    if not (stuck or (behind and has_storms)):
        return None
    last = last_dispatch.get(HAFS_KEY)
    if last is not None and (now - last).total_seconds() < cooldown_s:
        return None
    if stuck:
        return f"newest cycle {latest} stuck in_progress since {started:%Y-%m-%dT%H:%MZ}"
    return f"latest={latest} behind expected={exp} (storms active)"


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def fetch_manifest(session: requests.Session, url: str = MANIFEST_URL, timeout: float = 20.0) -> dict:
    r = session.get(url, params={"t": int(time.time())}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def dispatch_workflow(session: requests.Session, token: str, repo: str, workflow: str,
                      *, ref: str = "main", timeout: float = 20.0) -> int:
    """Fire workflow_dispatch with a BLANK cycle (never-miss backfill). Returns the
    HTTP status (204 = accepted)."""
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    r = session.post(url, json={"ref": ref, "inputs": {}}, timeout=timeout, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return r.status_code


# --------------------------------------------------------------------------
# One tick + the loop
# --------------------------------------------------------------------------
def run_once(session: requests.Session, token: Optional[str], repo: str,
             last_dispatch: Dict[str, dt.datetime], *, now: Optional[dt.datetime] = None,
             dry_run: bool = False, dispatch: Optional[Callable] = None) -> List[Tuple[str, str]]:
    """One watchdog pass: fetch manifest, decide, dispatch the behind models. Mutates
    ``last_dispatch`` with the time of each fired dispatch. Returns what it kicked.
    Any error is logged + swallowed (the watchdog must never crash its host)."""
    now = now or dt.datetime.now(UTC)
    dispatch = dispatch or (lambda wf: dispatch_workflow(session, token, repo, wf))
    try:
        manifest = fetch_manifest(session)
    except Exception as e:  # noqa: BLE001
        log.warning("ens_watchdog: manifest fetch failed (%s); skipping tick", e)
        return []
    behind = decide_dispatches(manifest, now, last_dispatch)
    fired: List[Tuple[str, str]] = []
    for slug, reason in behind:
        wf = MODELS[slug]["workflow"]
        if dry_run or not token:
            log.info("ens_watchdog: WOULD dispatch %s (%s) [%s]", wf, reason,
                     "dry-run" if dry_run else "no token")
            continue
        try:
            code = dispatch(wf)
            if 200 <= code < 300:
                last_dispatch[slug] = now
                fired.append((slug, reason))
                log.warning("ens_watchdog: dispatched %s (%s) -> HTTP %s", wf, reason, code)
            else:
                log.warning("ens_watchdog: dispatch %s -> HTTP %s (not retried this tick)", wf, code)
        except Exception as e:  # noqa: BLE001
            log.warning("ens_watchdog: dispatch %s failed: %s", wf, e)
    if not behind:
        log.info("ens_watchdog: all models current")
    return fired


def hafs_run_once(session: requests.Session, token: Optional[str], repo: str,
                  last_dispatch: Dict[str, dt.datetime], *,
                  now: Optional[dt.datetime] = None, dry_run: bool = False,
                  dispatch: Optional[Callable] = None) -> Optional[Tuple[str, str]]:
    """One HAFS watchdog pass: fetch the HAFS manifest, decide, dispatch update-hafs.yml
    if behind/stuck. Mutates ``last_dispatch`` (key HAFS_KEY). Returns (HAFS_KEY, reason)
    if it fired, else None. Errors logged + swallowed (never crash the host poller)."""
    now = now or dt.datetime.now(UTC)
    dispatch = dispatch or (lambda wf: dispatch_workflow(session, token, repo, wf))
    try:
        manifest = fetch_manifest(session, HAFS_MANIFEST_URL)
    except Exception as e:  # noqa: BLE001
        log.warning("hafs_watchdog: manifest fetch failed (%s); skipping tick", e)
        return None
    reason = decide_hafs_dispatch(manifest, now, last_dispatch)
    if not reason:
        log.info("hafs_watchdog: HAFS current")
        return None
    if dry_run or not token:
        log.info("hafs_watchdog: WOULD dispatch %s (%s) [%s]", HAFS_WORKFLOW, reason,
                 "dry-run" if dry_run else "no token")
        return None
    try:
        code = dispatch(HAFS_WORKFLOW)
        if 200 <= code < 300:
            last_dispatch[HAFS_KEY] = now
            log.warning("hafs_watchdog: dispatched %s (%s) -> HTTP %s",
                        HAFS_WORKFLOW, reason, code)
            return (HAFS_KEY, reason)
        log.warning("hafs_watchdog: dispatch %s -> HTTP %s (not retried this tick)",
                    HAFS_WORKFLOW, code)
    except Exception as e:  # noqa: BLE001
        log.warning("hafs_watchdog: dispatch %s failed: %s", HAFS_WORKFLOW, e)
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    token = _env("ENS_WATCHDOG_GH_TOKEN")
    repo = _env("ENS_WATCHDOG_REPO", "WeathermanAAA/Triple-A-Tropics")
    interval = float(_env("ENS_WATCHDOG_INTERVAL_S", "1200"))
    dry_run = _env("ENS_WATCHDOG_DRYRUN", "0") == "1"
    if not token and not dry_run:
        log.warning("ens_watchdog: ENS_WATCHDOG_GH_TOKEN not set - watchdog will log "
                    "decisions but NOT dispatch. Set the PAT (actions:write) to arm it.")
    session = requests.Session()
    last_dispatch: Dict[str, dt.datetime] = {}
    log.info("ens_watchdog: armed (repo=%s interval=%ss dry_run=%s models=%s + hafs)",
             repo, interval, dry_run, ",".join(MODELS))
    # Explicit HAFS-watch confirmation in the logs: the recurring "behind on 00Z
    # HAFS" is auto-recovered by dispatching update-hafs.yml (which always renders
    # on a workflow_dispatch) when the render worker wedges or falls a cycle behind.
    log.info("ens_watchdog: HAFS watch armed -> dispatch %s when behind>lag=%.0fh OR "
             "in_progress>%.0fm (cooldown %.0fm)", HAFS_WORKFLOW, HAFS_LAG_H,
             HAFS_STUCK_BUILD_S / 60.0, HAFS_COOLDOWN_S / 60.0)
    while True:
        run_once(session, token, repo, last_dispatch, dry_run=dry_run)
        hafs_run_once(session, token, repo, last_dispatch, dry_run=dry_run)
        time.sleep(interval)


if __name__ == "__main__":
    main()
