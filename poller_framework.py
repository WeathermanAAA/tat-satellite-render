#!/usr/bin/env python3
"""
poller_framework.py
-------------------
A small, infra-agnostic poller framework: the reusable spine for every future
Triple-A-Tropics poller (streaming intensity now, models later). It generalizes
the hard-won lessons from the floater poller (floater_poller.py in the
tat-satellite-render repo) so each new poller gets them for free, and it bakes
in the one thing the floater poller lacked: a real, always-emitted health
heartbeat.

WHY THIS EXISTS
    The floater poller once froze silently for about half a day: one basin's
    tracks fetch timed out, an all-or-nothing guard preserved the whole stale
    manifest, and nothing alerted us. Two structural failures combined: (1) one
    source's failure was allowed to affect every source's state, and (2) there
    was no heartbeat an external watcher could alarm on. This framework designs
    both out:
      * PER-SOURCE ISOLATION. Each source is polled inside its own guard and
        keeps its own last-known-good. One source failing can never freeze,
        corrupt, or stale another source's state. There is no code path where a
        single failure preserves everything wholesale.
      * ALWAYS-ON HEARTBEAT. Every poll cycle ends by emitting a health snapshot
        (per-source fresh / stale / failing + seconds since each last
        succeeded), even when every source failed. Staleness is therefore always
        DETECTABLE by an external watcher, never silent.

WHAT IT IS NOT
    This module polls no real source and deploys nowhere. It has zero
    third-party dependencies (no requests, no boto3) so it imports and tests
    without a network. The caller injects the fetch sources, the change-detect
    and process callbacks, and the output sink; the framework owns the loop,
    the isolation, the retries, the change gating, the health, and the
    freshness stamping. Wiring it to real b-decks / R2 is a later piece.

PUBLIC API (see each symbol's docstring for the full contract)
    Timestamps / freshness stamping (format matches ace_core and the feeds):
        utcnow() iso_z() parse_iso()
        staleness_seconds() staleness_minutes() freshness_stamp()
    Resilient fetch:
        FetchPolicy  TransientFetchError  PermanentFetchError
        compute_backoff()  resilient_fetch()
    Output sinks (infra-agnostic; caller injects one):
        Sink (protocol)  DictSink  FileSink  sink_heartbeat()
    Sources and per-cycle results:
        Source  ProcessContext  SourceResult  SourceHealth
    The engine:
        PollerEngine
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Hashable, Optional, Sequence

UTC = dt.timezone.utc

log = logging.getLogger("poller")


# ---------------------------------------------------------------------------
# Timestamps / freshness stamping
#
# These mirror ace_core's iso_z / now_iso_z / staleness_minutes so a poller's
# outputs carry the SAME timestamp shape the rest of the site already emits
# (generated_utc + the data's own valid-time + derived staleness). Kept local
# (not imported from ace_core) so the framework stays a standalone, dependency
# free library that can move to another repo unchanged.
# ---------------------------------------------------------------------------

def utcnow() -> dt.datetime:
    """Timezone-aware current UTC time."""
    return dt.datetime.now(UTC)


def _as_utc(t: Optional[dt.datetime]) -> Optional[dt.datetime]:
    """Coerce a datetime to timezone-aware UTC. Naive datetimes are assumed to
    already be UTC (matching ace_core's convention)."""
    if t is None:
        return None
    if t.tzinfo is None:
        return t.replace(tzinfo=UTC)
    return t.astimezone(UTC)


def iso_z(t: Optional[dt.datetime]) -> Optional[str]:
    """ISO8601 with a trailing Z (seconds precision), or None. Strings pass
    through unchanged so a caller can hand in an already-formatted timestamp."""
    if t is None:
        return None
    if isinstance(t, str):
        return t
    return _as_utc(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: Optional[str]) -> Optional[dt.datetime]:
    """Parse an ISO8601 timestamp (trailing Z accepted) into tz-aware UTC, or
    None if it is empty / unparseable. Never raises."""
    if not s:
        return None
    if isinstance(s, dt.datetime):
        return _as_utc(s)
    try:
        d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return _as_utc(d)


def staleness_seconds(valid: Optional[dt.datetime],
                      now: Optional[dt.datetime] = None) -> Optional[float]:
    """Seconds between a valid-time and now (UTC), or None if no valid-time."""
    valid = _as_utc(valid)
    if valid is None:
        return None
    now = _as_utc(now) or utcnow()
    return (now - valid).total_seconds()


def staleness_minutes(valid: Optional[dt.datetime],
                      now: Optional[dt.datetime] = None) -> Optional[int]:
    """Whole minutes between a valid-time and now (UTC), or None. Matches
    ace_core.staleness_minutes so feed and poller agree."""
    secs = staleness_seconds(valid, now)
    return None if secs is None else int(secs // 60)


def freshness_stamp(latest_valid: Optional[dt.datetime],
                    now: Optional[dt.datetime] = None,
                    generated: Optional[dt.datetime] = None) -> dict:
    """The standard freshness block to stamp onto a poller's outputs:
        generated_utc         - when this output was produced (ISO8601 Z)
        latest_fix_valid_utc  - valid-time of the newest datum in the output
        staleness_minutes     - whole minutes between the two (matches the feeds)
        staleness_seconds     - finer-grained gap for sub-minute pollers
    ``generated`` defaults to ``now`` (or real now). Same keys ace_core/the
    tracks + ACE feeds already emit, plus a seconds field for fast pollers."""
    now = _as_utc(now) or utcnow()
    generated = _as_utc(generated) or now
    return {
        "generated_utc": iso_z(generated),
        "latest_fix_valid_utc": iso_z(latest_valid),
        "staleness_minutes": staleness_minutes(latest_valid, now),
        "staleness_seconds": staleness_seconds(latest_valid, now),
    }


# ---------------------------------------------------------------------------
# Resilient fetch
#
# One request helper that distinguishes a TRANSIENT failure (timeout, 5xx,
# connection reset -> retry with exponential backoff) from a GENUINE ABSENCE
# (the source legitimately has nothing; the caller's fetch returns a value such
# as None or [] and we do NOT retry) and from a PERMANENT, non-retryable error
# (a 400-class bug -> stop immediately, do not waste the retry budget).
# ---------------------------------------------------------------------------

class TransientFetchError(Exception):
    """A retryable failure (timeout / 5xx / connection reset). Raising this (or
    any generic Exception) from a fetch callback triggers retry-with-backoff."""


class PermanentFetchError(Exception):
    """A non-retryable failure (e.g. a malformed request / 4xx that retrying
    cannot fix). resilient_fetch re-raises this immediately without retrying, so
    the retry budget is not burned on a hopeless call."""


@dataclasses.dataclass(frozen=True)
class FetchPolicy:
    """Per-source request policy. ``connect_timeout_s`` / ``read_timeout_s`` map
    to a requests-style tuple timeout (fast connect, generous read) so a
    sluggish origin gets time to answer instead of being mistaken for a failure
    (the exact bug that silently stopped one basin refreshing). ``max_retries``
    is the number of retries AFTER the first attempt; total attempts are
    1 + max_retries. Backoff for retry ``a`` (1-based) is
    ``min(backoff_max_s, backoff_base_s * 2**(a-1)) + uniform(0, jitter_s)`` -
    base 2.0 gives the floater poller's 2s / 4s / 8s schedule."""
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 45.0
    max_retries: int = 3
    backoff_base_s: float = 2.0
    backoff_max_s: float = 60.0
    jitter_s: float = 0.5

    @property
    def timeout(self) -> tuple[float, float]:
        """(connect, read) tuple for a requests-style call."""
        return (self.connect_timeout_s, self.read_timeout_s)


def compute_backoff(policy: FetchPolicy, attempt: int,
                    rng: Optional[random.Random] = None) -> float:
    """Backoff (seconds) before retry ``attempt`` (1-based): exponential on
    backoff_base_s, capped at backoff_max_s, plus uniform [0, jitter_s) jitter.
    Pure function of its inputs (rng injectable) so tests are deterministic."""
    rng = rng or random
    raw = policy.backoff_base_s * (2 ** (attempt - 1))
    capped = min(policy.backoff_max_s, raw)
    jitter = rng.uniform(0, policy.jitter_s) if policy.jitter_s else 0.0
    return capped + jitter


def resilient_fetch(fn: Callable[[], Any],
                    policy: Optional[FetchPolicy] = None,
                    *,
                    sleep: Callable[[float], None] = time.sleep,
                    rng: Optional[random.Random] = None,
                    on_retry: Optional[Callable[[int, int, Exception, float], None]] = None
                    ) -> Any:
    """Call ``fn`` with retry-and-backoff and return its result.

    Contract:
      * SUCCESS (incl. genuine absence): ``fn`` returns a value -> returned
        as-is, no retry. Model "the source has nothing right now" as a normal
        return value (None / [] / a sentinel), NOT an exception.
      * TRANSIENT failure: ``fn`` raises TransientFetchError or any generic
        Exception -> retried up to ``policy.max_retries`` times with
        compute_backoff() sleeps in between. If still failing, the LAST
        exception is re-raised.
      * PERMANENT failure: ``fn`` raises PermanentFetchError -> re-raised
        immediately, no retries (do not burn the budget on a hopeless call).

    ``sleep`` and ``rng`` are injectable so tests run instantly and
    deterministically; ``on_retry(attempt, total, exc, backoff)`` is an optional
    observation hook (used by the engine to log/flag retries)."""
    policy = policy or FetchPolicy()
    total = policy.max_retries + 1
    last_exc: Optional[Exception] = None
    for attempt in range(1, total + 1):
        try:
            return fn()
        except PermanentFetchError:
            raise
        except Exception as e:  # noqa: BLE001 - transient; retry per policy
            last_exc = e
            if attempt < total:
                backoff = compute_backoff(policy, attempt, rng)
                if on_retry is not None:
                    on_retry(attempt, total, e, backoff)
                sleep(backoff)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Output sinks (infra-agnostic)
#
# A Sink is anything with ``write(key, payload_dict)``. The framework never
# knows whether that lands in R2, on disk, in a dict, or on a queue - the caller
# injects one. Two reference impls plus a heartbeat adapter are provided.
# ---------------------------------------------------------------------------

class Sink:
    """Output sink protocol: ``write(key, payload)`` persists a JSON-able dict
    under a string key. Subclass or duck-type. The framework calls this for the
    health heartbeat (via sink_heartbeat) and the caller's process callback uses
    it for its own outputs, so the same poller can target R2, disk, etc."""

    def write(self, key: str, payload: dict) -> None:  # pragma: no cover
        raise NotImplementedError


class DictSink(Sink):
    """In-memory sink. Last payload per key is kept in ``.store``; the full
    ordered history per key is in ``.history``. Ideal for tests and dry runs."""

    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self.history: dict[str, list[dict]] = {}

    def write(self, key: str, payload: dict) -> None:
        self.store[key] = payload
        self.history.setdefault(key, []).append(payload)


class FileSink(Sink):
    """Writes each payload as a JSON file under ``root_dir`` (key is the
    relative path; parent dirs created as needed). Atomic via temp-then-rename
    so a watcher never reads a half-written heartbeat."""

    def __init__(self, root_dir: str | os.PathLike) -> None:
        self.root = Path(root_dir)

    def write(self, key: str, payload: dict) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":"), default=str),
                       encoding="utf-8")
        tmp.replace(path)


def sink_heartbeat(sink: Sink, key: str) -> Callable[[dict], None]:
    """Adapt a Sink into a heartbeat callable that writes each health snapshot
    under a fixed ``key`` (e.g. 'intensity/health.json')."""
    def _emit(snapshot: dict) -> None:
        sink.write(key, snapshot)
    return _emit


# ---------------------------------------------------------------------------
# Sources and per-cycle context / results
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProcessContext:
    """Everything the caller's ``process`` callback needs for one CHANGED
    source. ``data`` is the freshly fetched payload; ``signature`` is its
    change-token; ``previous_signature`` is the last one processed (None on the
    first run); ``valid_time`` is the data's own newest valid-time; ``freshness``
    is the ready-to-stamp block from freshness_stamp(valid_time, now); ``sink``
    is the injected output sink (may be None if the caller closes over its own).
    """
    name: str
    data: Any
    signature: Hashable
    previous_signature: Optional[Hashable]
    valid_time: Optional[dt.datetime]
    now: dt.datetime
    freshness: dict
    sink: Optional[Sink]


@dataclasses.dataclass(frozen=True)
class Source:
    """One named, independently-polled source.

    fetch()        -> the source's current data. Return a value on success
                      (including a value that means "nothing right now" - that is
                      a successful poll, not a failure). Raise TransientFetchError
                      / any Exception to signal a retryable failure, or
                      PermanentFetchError for a non-retryable one.
    change_key(data) -> a hashable token answering "has this source produced new
                      data?". The framework stores the last token and only runs
                      ``process`` when it changes (the cheap-when-nothing-new
                      guarantee). Generalizes the floater poller's per-frame
                      sha256: return a content hash, a latest-fix ISO string, an
                      ETag, an advisory number - anything that changes iff the
                      data did.
    process(ctx)   -> do the expensive work for new data (recompute, stamp,
                      write to ctx.sink). Only called on a real change. May raise;
                      a raise is recorded as a process failure and the signature
                      is NOT advanced, so the same new data is retried next cycle.
    valid_time(data) -> optional; the data's own newest valid-time, used for
                      freshness stamping and data-lag visibility.
    policy         -> optional per-source FetchPolicy (else the engine default).
    """
    name: str
    fetch: Callable[[], Any]
    change_key: Callable[[Any], Hashable]
    process: Callable[[ProcessContext], None]
    valid_time: Optional[Callable[[Any], Optional[dt.datetime]]] = None
    policy: Optional[FetchPolicy] = None


# Per-cycle outcome for one source.
FETCH_FAILED = "fetch_failed"
PROCESS_FAILED = "process_failed"
UNCHANGED = "unchanged"
CHANGED = "changed"


@dataclasses.dataclass
class SourceResult:
    """Outcome of polling one source this cycle: ``status`` is one of
    CHANGED / UNCHANGED / FETCH_FAILED / PROCESS_FAILED; ``error`` is set on the
    failure statuses; ``signature`` is set on CHANGED / UNCHANGED."""
    name: str
    status: str
    error: Optional[str] = None
    signature: Optional[Hashable] = None

    @property
    def ok(self) -> bool:
        """True when the source was polled successfully this cycle (data either
        changed and processed, or was unchanged) - i.e. NOT a failure."""
        return self.status in (CHANGED, UNCHANGED)


# Health states.
FRESH = "fresh"
STALE = "stale"
FAILING = "failing"
NEVER = "never"


@dataclasses.dataclass
class SourceHealth:
    """Rolling health for one source. ``last_success_utc`` (last successful
    FETCH) is the freshness clock the heartbeat exposes - if it stops advancing,
    the source has frozen, and an external watcher can alarm. ``last_change_utc``
    is when data last actually changed (process ran). Counters and ``last_error``
    give a watcher the why."""
    name: str
    last_attempt_utc: Optional[dt.datetime] = None
    last_success_utc: Optional[dt.datetime] = None
    last_change_utc: Optional[dt.datetime] = None
    last_valid_time: Optional[dt.datetime] = None
    consecutive_failures: int = 0
    total_failures: int = 0
    total_polls: int = 0
    last_error: Optional[str] = None
    last_signature: Optional[Hashable] = None

    def classify(self, now: dt.datetime, stale_after_s: float,
                 fail_threshold: int) -> str:
        """fresh / stale / failing / never.
          never   - has never fetched successfully (and not yet enough failures
                    to call it failing).
          failing - consecutive_failures has reached fail_threshold (takes
                    precedence: an actively-erroring source is the loudest
                    signal).
          stale   - last successful fetch is older than stale_after_s (the
                    freeze signal: polling silently stopped making progress).
          fresh   - fetched successfully within stale_after_s.
        Note 'stale' here is FETCH staleness (the freeze condition). DATA lag
        (a source that answers promptly but with old data) is surfaced
        separately via freshness_stamp on the outputs / last_valid_time."""
        if self.consecutive_failures >= fail_threshold:
            return FAILING
        if self.last_success_utc is None:
            return NEVER
        age = (now - self.last_success_utc).total_seconds()
        if age > stale_after_s:
            return STALE
        return FRESH

    def snapshot(self, now: dt.datetime, stale_after_s: float,
                 fail_threshold: int) -> dict:
        """A JSON-able view of this source's health for the heartbeat."""
        return {
            "state": self.classify(now, stale_after_s, fail_threshold),
            "last_attempt_utc": iso_z(self.last_attempt_utc),
            "last_success_utc": iso_z(self.last_success_utc),
            "last_change_utc": iso_z(self.last_change_utc),
            "latest_fix_valid_utc": iso_z(self.last_valid_time),
            "seconds_since_success": (
                None if self.last_success_utc is None
                else round((now - self.last_success_utc).total_seconds(), 1)
            ),
            "data_staleness_seconds": staleness_seconds(self.last_valid_time, now),
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_polls": self.total_polls,
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class PollerEngine:
    """Polls a set of named Sources on an interval with per-source isolation,
    change-gated processing, and an always-emitted health heartbeat.

    Anti-freeze guarantees (the lessons the floater poller paid for):
      1. ISOLATION. poll_source wraps each source in its own guard; an exception
         in one source's fetch / change_key / process can never escape to skip
         another source or the heartbeat. A failed source keeps its own
         last-known-good signature and is flagged - it does not touch any other
         source's state. There is no wholesale 'preserve everything' path.
      2. DETECTABILITY. poll_once ALWAYS emits a health snapshot at the end, even
         if every source failed. last_success_utc per source is the freeze clock;
         an external watcher alarms when it stops advancing or a state flips to
         stale / failing.
      3. RESILIENCE. The fetch uses resilient_fetch (per-source timeout + backoff
         retries). run_forever never dies: any unexpected error is logged and the
         loop continues after emitting health.

    Injection points (all optional, all defaulted) keep it infra-agnostic and
    fully testable offline: ``sink`` (output target), ``heartbeat`` (a callable
    receiving each snapshot - use sink_heartbeat() to point it at any Sink),
    ``clock`` (now-provider), ``sleep``, ``rng``. Config (interval, timeouts,
    retries, staleness threshold, fail threshold) is by parameter, env-overridable
    by the caller - nothing cloud-specific is baked in.
    """

    def __init__(self,
                 sources: Sequence[Source],
                 *,
                 name: str = "poller",
                 interval_s: float = 60.0,
                 stale_after_s: float = 600.0,
                 fail_threshold: int = 3,
                 policy: Optional[FetchPolicy] = None,
                 sink: Optional[Sink] = None,
                 heartbeat: Optional[Callable[[dict], None]] = None,
                 clock: Callable[[], dt.datetime] = utcnow,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: Optional[random.Random] = None) -> None:
        self.sources = list(sources)
        self.name = name
        self.interval_s = interval_s
        self.stale_after_s = stale_after_s
        self.fail_threshold = fail_threshold
        self.policy = policy or FetchPolicy()
        self.sink = sink
        self.heartbeat = heartbeat
        self._now = clock
        self._sleep = sleep
        self._rng = rng or random
        self._health: dict[str, SourceHealth] = {
            s.name: SourceHealth(name=s.name) for s in self.sources
        }
        self._last_poll_utc: Optional[dt.datetime] = None

    # -- accessors -------------------------------------------------------

    def health(self, name: str) -> SourceHealth:
        """The live SourceHealth for one source (mutated in place each cycle)."""
        return self._health[name]

    # -- one source (fully isolated) -------------------------------------

    def poll_source(self, source: Source) -> SourceResult:
        """Poll exactly one source. NEVER raises and NEVER touches any other
        source's state. Returns this source's SourceResult and updates only this
        source's SourceHealth."""
        h = self._health[source.name]
        now = self._now()
        h.last_attempt_utc = now
        h.total_polls += 1

        # 1. FETCH (with retry/backoff). A total failure leaves the source at its
        #    last-known-good signature - no process, no state churn, just flagged.
        try:
            data = resilient_fetch(
                source.fetch,
                source.policy or self.policy,
                sleep=self._sleep,
                rng=self._rng,
                on_retry=lambda a, t, e, b: log.warning(
                    "%s/%s fetch attempt %d/%d failed (%s); retrying in %.1fs",
                    self.name, source.name, a, t, e, b),
            )
        except Exception as e:  # noqa: BLE001 - isolation boundary
            h.consecutive_failures += 1
            h.total_failures += 1
            h.last_error = _format_err(e)
            log.warning("%s/%s fetch FAILED after retries: %s",
                        self.name, source.name, h.last_error)
            return SourceResult(source.name, FETCH_FAILED, error=h.last_error)

        # Fetch succeeded (this includes a genuine 'nothing right now').
        h.last_success_utc = now

        # Optional data valid-time (kept if the source does not expose one).
        if source.valid_time is not None:
            try:
                vt = _as_utc(source.valid_time(data))
            except Exception:  # noqa: BLE001 - valid-time is best-effort
                vt = None
            if vt is not None:
                h.last_valid_time = vt
        vt = h.last_valid_time

        # 2. CHANGE DETECTION. Compute the change token; skip the expensive
        #    process entirely when it matches the last processed one.
        try:
            sig = source.change_key(data)
        except Exception as e:  # noqa: BLE001
            h.consecutive_failures += 1
            h.total_failures += 1
            h.last_error = "change_key: " + _format_err(e)
            return SourceResult(source.name, PROCESS_FAILED, error=h.last_error)

        if h.last_change_utc is not None and sig == h.last_signature:
            # Nothing new: cheap path. The successful fetch already refreshed
            # last_success_utc, so the source reads 'fresh' without doing work.
            h.consecutive_failures = 0
            h.last_error = None
            return SourceResult(source.name, UNCHANGED, signature=sig)

        # 3. PROCESS (new data). On failure, do NOT advance the signature so the
        #    same new data is retried next cycle.
        ctx = ProcessContext(
            name=source.name,
            data=data,
            signature=sig,
            previous_signature=h.last_signature,
            valid_time=vt,
            now=now,
            freshness=freshness_stamp(vt, now),
            sink=self.sink,
        )
        try:
            source.process(ctx)
        except Exception as e:  # noqa: BLE001
            h.consecutive_failures += 1
            h.total_failures += 1
            h.last_error = "process: " + _format_err(e)
            log.warning("%s/%s process FAILED (signature held for retry): %s",
                        self.name, source.name, h.last_error)
            return SourceResult(source.name, PROCESS_FAILED, error=h.last_error)

        h.last_signature = sig
        h.last_change_utc = now
        h.consecutive_failures = 0
        h.last_error = None
        return SourceResult(source.name, CHANGED, signature=sig)

    # -- one cycle -------------------------------------------------------

    def poll_once(self) -> dict[str, SourceResult]:
        """Poll every source once, each isolated, then ALWAYS emit health.
        Returns {source_name: SourceResult}. A failure in any one source affects
        neither the others' polling nor the heartbeat."""
        results: dict[str, SourceResult] = {}
        for source in self.sources:
            try:
                results[source.name] = self.poll_source(source)
            except Exception as e:  # noqa: BLE001 - belt and suspenders
                # poll_source is already guarded; this only fires on a framework
                # bug, and even then one source cannot take down the cycle.
                err = _format_err(e)
                h = self._health.get(source.name)
                if h is not None:
                    h.consecutive_failures += 1
                    h.total_failures += 1
                    h.last_error = err
                results[source.name] = SourceResult(source.name, FETCH_FAILED, error=err)
        self._last_poll_utc = self._now()
        self.emit_health()
        return results

    # -- health ----------------------------------------------------------

    def health_snapshot(self, now: Optional[dt.datetime] = None) -> dict:
        """A JSON-able health snapshot: per-source state + counters, plus an
        overall ``healthy`` flag (False if ANY source is stale / failing /
        never). This is the object an external watcher reads to alarm."""
        now = now or self._now()
        sources: dict[str, dict] = {}
        healthy = True
        worst = FRESH
        for src in self.sources:
            snap = self._health[src.name].snapshot(
                now, self.stale_after_s, self.fail_threshold)
            sources[src.name] = snap
            if snap["state"] != FRESH:
                healthy = False
            worst = _worse(worst, snap["state"])
        return {
            "poller": self.name,
            "generated_utc": iso_z(now),
            "last_poll_utc": iso_z(self._last_poll_utc),
            "interval_s": self.interval_s,
            "stale_after_s": self.stale_after_s,
            "fail_threshold": self.fail_threshold,
            "healthy": healthy,
            "worst_state": worst,
            "sources": sources,
        }

    def emit_health(self) -> dict:
        """Build and emit the health snapshot via the heartbeat callback (if
        configured). Always returns the snapshot so a caller / test can inspect
        it. Emission failures are swallowed (a broken heartbeat sink must never
        crash the loop) but logged."""
        snap = self.health_snapshot()
        if self.heartbeat is not None:
            try:
                self.heartbeat(snap)
            except Exception as e:  # noqa: BLE001
                log.error("%s heartbeat emit failed: %s", self.name, _format_err(e))
        return snap

    # -- main loop -------------------------------------------------------

    def run_forever(self, interval_s: Optional[float] = None, *,
                    max_cycles: Optional[int] = None) -> int:
        """Poll forever (or for ``max_cycles`` cycles, which tests use to run the
        loop deterministically). The loop never dies: an unexpected error is
        logged, health is still emitted, and it sleeps then continues. Returns
        the number of cycles run."""
        interval = self.interval_s if interval_s is None else interval_s
        cycles = 0
        log.info("%s starting | sources=%s | interval=%.0fs | stale_after=%.0fs",
                 self.name, ",".join(s.name for s in self.sources),
                 interval, self.stale_after_s)
        while True:
            try:
                self.poll_once()
            except Exception as e:  # noqa: BLE001 - the loop must never die
                log.exception("%s poll_once error (continuing): %s",
                              self.name, e)
                try:
                    self.emit_health()
                except Exception:  # noqa: BLE001
                    pass
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return cycles
            self._sleep(interval)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STATE_ORDER = {FRESH: 0, STALE: 1, NEVER: 2, FAILING: 3}


def _worse(a: str, b: str) -> str:
    """Return the more-alarming of two health states."""
    return a if _STATE_ORDER.get(a, 0) >= _STATE_ORDER.get(b, 0) else b


def _format_err(e: BaseException) -> str:
    """Compact 'TypeName: message' for logs / health, never a full traceback in
    the snapshot (the traceback goes to the debug log)."""
    log.debug("error detail:\n%s", "".join(
        traceback.format_exception(type(e), e, e.__traceback__)))
    msg = str(e).strip()
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


if __name__ == "__main__":  # pragma: no cover - tiny smoke demo, no network
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # A two-source demo driven by an in-memory counter, proving the loop,
    # change detection, and heartbeat without touching a network.
    _state = {"a": 0, "b": 0}

    def _make(nm: str) -> Source:
        return Source(
            name=nm,
            fetch=lambda nm=nm: {"v": _state[nm]},
            change_key=lambda data: data["v"],
            process=lambda ctx: log.info("processed %s -> %s", ctx.name, ctx.data),
            valid_time=lambda data: utcnow(),
        )

    demo_sink = DictSink()
    engine = PollerEngine(
        [_make("a"), _make("b")],
        name="demo",
        interval_s=0.0,
        heartbeat=sink_heartbeat(demo_sink, "demo/health.json"),
    )
    _state["a"] = 1  # only 'a' changes
    engine.run_forever(max_cycles=2)
    print(json.dumps(demo_sink.store["demo/health.json"], indent=2))
