#!/usr/bin/env python3
"""
Unit tests for poller_framework. No network, no third-party deps, deterministic
(injected clock / sleep / rng). Run:

    python -m unittest discover -s tests -v
    # or
    python tests/test_poller_framework.py

The four contract assertions the framework must satisfy, each with a test:
  (a) a persistently-failing source does NOT freeze or stale the healthy ones
  (b) change detection skips work when nothing changed, runs it when it did
  (c) the health snapshot reports fresh / stale / failing / never per source
  (d) retries + backoff fire exactly as configured
plus transient-then-recover, the always-on heartbeat, genuine-absence vs
transient, process-failure handling, and freshness-stamp shape.
"""

from __future__ import annotations

import datetime as dt
import os
import random
import sys
import tempfile
import unittest

# Make the repo root importable when run directly (python tests/test_*.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poller_framework as pf  # noqa: E402

UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Deterministic test doubles
# ---------------------------------------------------------------------------

class FakeClock:
    """A controllable UTC clock. ``advance(seconds)`` moves it forward; calling
    the instance returns the current time (so it drops into PollerEngine(clock=)).
    """

    def __init__(self, start: dt.datetime | None = None) -> None:
        self.t = start or dt.datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)

    def __call__(self) -> dt.datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + dt.timedelta(seconds=seconds)


class RecordingSleep:
    """A sleep() stand-in that records every duration it was asked to sleep,
    and optionally advances a FakeClock so 'time passes' during retries."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self.clock = clock

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.clock is not None:
            self.clock.advance(seconds)


def _no_jitter_policy(**kw) -> pf.FetchPolicy:
    kw.setdefault("jitter_s", 0.0)
    return pf.FetchPolicy(**kw)


def _zero_rng() -> random.Random:
    # Seeded so any jitter is reproducible (we mostly use jitter_s=0 anyway).
    return random.Random(0)


# ---------------------------------------------------------------------------
# (d) Resilient fetch: retries + backoff, absence, permanent
# ---------------------------------------------------------------------------

class TestResilientFetch(unittest.TestCase):

    def test_backoff_schedule_matches_config(self):
        """base 2.0 -> 2s, 4s, 8s; jitter 0 for determinism."""
        p = _no_jitter_policy(backoff_base_s=2.0, max_retries=3)
        self.assertEqual(pf.compute_backoff(p, 1), 2.0)
        self.assertEqual(pf.compute_backoff(p, 2), 4.0)
        self.assertEqual(pf.compute_backoff(p, 3), 8.0)

    def test_backoff_is_capped(self):
        p = _no_jitter_policy(backoff_base_s=2.0, backoff_max_s=5.0, max_retries=5)
        self.assertEqual(pf.compute_backoff(p, 1), 2.0)
        self.assertEqual(pf.compute_backoff(p, 2), 4.0)
        self.assertEqual(pf.compute_backoff(p, 3), 5.0)  # 8 capped to 5
        self.assertEqual(pf.compute_backoff(p, 9), 5.0)

    def test_transient_then_recover_retries_then_returns(self):
        """Fails transiently twice, then succeeds: exactly 2 backoff sleeps with
        the configured 2s, 4s schedule, and the success value is returned."""
        sleep = RecordingSleep()
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] <= 2:
                raise pf.TransientFetchError(f"boom {calls['n']}")
            return "recovered"

        out = pf.resilient_fetch(fn, _no_jitter_policy(backoff_base_s=2.0, max_retries=3),
                                 sleep=sleep, rng=_zero_rng())
        self.assertEqual(out, "recovered")
        self.assertEqual(calls["n"], 3)            # 2 failures + 1 success
        self.assertEqual(sleep.calls, [2.0, 4.0])  # backoff fired exactly twice

    def test_persistent_failure_exhausts_then_raises_last(self):
        """All attempts fail: 1 + max_retries attempts, max_retries sleeps, and
        the LAST exception propagates."""
        sleep = RecordingSleep()
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise pf.TransientFetchError(f"fail {calls['n']}")

        with self.assertRaises(pf.TransientFetchError) as cm:
            pf.resilient_fetch(fn, _no_jitter_policy(backoff_base_s=2.0, max_retries=3),
                               sleep=sleep, rng=_zero_rng())
        self.assertEqual(str(cm.exception), "fail 4")   # 4th (last) attempt
        self.assertEqual(calls["n"], 4)                 # 1 + 3 retries
        self.assertEqual(sleep.calls, [2.0, 4.0, 8.0])  # 3 backoffs

    def test_genuine_absence_is_not_retried(self):
        """A successful fetch that returns 'nothing' (None / []) is NOT a failure
        and is returned immediately with no sleeps."""
        sleep = RecordingSleep()
        out = pf.resilient_fetch(lambda: None,
                                 _no_jitter_policy(max_retries=3), sleep=sleep)
        self.assertIsNone(out)
        self.assertEqual(sleep.calls, [])

    def test_permanent_error_not_retried(self):
        """PermanentFetchError stops immediately: no retries, no sleeps."""
        sleep = RecordingSleep()
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise pf.PermanentFetchError("400 bad request")

        with self.assertRaises(pf.PermanentFetchError):
            pf.resilient_fetch(fn, _no_jitter_policy(max_retries=5), sleep=sleep)
        self.assertEqual(calls["n"], 1)     # tried once, never again
        self.assertEqual(sleep.calls, [])

    def test_on_retry_hook_observes_each_retry(self):
        seen = []
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] <= 2:
                raise pf.TransientFetchError("x")
            return 1

        pf.resilient_fetch(
            fn, _no_jitter_policy(backoff_base_s=2.0, max_retries=3),
            sleep=RecordingSleep(),
            on_retry=lambda a, t, e, b: seen.append((a, t, b)))
        self.assertEqual(seen, [(1, 4, 2.0), (2, 4, 4.0)])


# ---------------------------------------------------------------------------
# (c) Health classification: fresh / stale / failing / never
# ---------------------------------------------------------------------------

class TestHealthClassification(unittest.TestCase):

    def setUp(self):
        self.now = dt.datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)

    def test_never_when_no_success_yet(self):
        h = pf.SourceHealth(name="x")
        self.assertEqual(h.classify(self.now, stale_after_s=600, fail_threshold=3),
                         pf.NEVER)

    def test_fresh_when_recent_success(self):
        h = pf.SourceHealth(name="x",
                            last_success_utc=self.now - dt.timedelta(seconds=30))
        self.assertEqual(h.classify(self.now, 600, 3), pf.FRESH)

    def test_stale_when_success_too_old(self):
        h = pf.SourceHealth(name="x",
                            last_success_utc=self.now - dt.timedelta(seconds=900))
        self.assertEqual(h.classify(self.now, 600, 3), pf.STALE)

    def test_failing_takes_precedence_over_stale(self):
        """Enough consecutive failures => failing, even if a (stale-aged) success
        exists. Failing is the loudest signal."""
        h = pf.SourceHealth(name="x",
                            last_success_utc=self.now - dt.timedelta(seconds=900),
                            consecutive_failures=3)
        self.assertEqual(h.classify(self.now, 600, 3), pf.FAILING)

    def test_failing_when_never_succeeded_but_many_failures(self):
        h = pf.SourceHealth(name="x", consecutive_failures=5)
        self.assertEqual(h.classify(self.now, 600, 3), pf.FAILING)


# ---------------------------------------------------------------------------
# (b) Change detection at the engine level
# ---------------------------------------------------------------------------

class TestChangeDetection(unittest.TestCase):

    def test_process_runs_only_on_change(self):
        """Data changes on cycles 1 and 3 only; process must run on 1 (initial),
        skip 2, run on 3, skip 4."""
        clock = FakeClock()
        values = iter([10, 10, 20, 20])   # cycle 1..4
        processed: list = []

        src = pf.Source(
            name="s",
            fetch=lambda: {"v": next(values)},
            change_key=lambda d: d["v"],
            process=lambda ctx: processed.append((ctx.now, ctx.data["v"])),
        )
        eng = pf.PollerEngine([src], clock=clock, sleep=lambda s: None)

        statuses = []
        for _ in range(4):
            res = eng.poll_once()["s"]
            statuses.append(res.status)
            clock.advance(60)

        self.assertEqual(statuses,
                         [pf.CHANGED, pf.UNCHANGED, pf.CHANGED, pf.UNCHANGED])
        self.assertEqual([v for _, v in processed], [10, 20])  # ran twice only

    def test_unchanged_is_still_fresh(self):
        """An unchanged-but-successful poll keeps the source fresh (it fetched
        fine); it just does no work."""
        clock = FakeClock()
        src = pf.Source(name="s", fetch=lambda: {"v": 1},
                        change_key=lambda d: d["v"], process=lambda ctx: None)
        eng = pf.PollerEngine([src], clock=clock, sleep=lambda s: None,
                              stale_after_s=600)
        eng.poll_once()
        clock.advance(60)
        res = eng.poll_once()["s"]
        self.assertEqual(res.status, pf.UNCHANGED)
        self.assertEqual(eng.health_snapshot()["sources"]["s"]["state"], pf.FRESH)

    def test_restamp_processes_every_cycle_but_last_change_honest(self):
        """restamp=True: process runs EVERY successful cycle (re-stamp), but
        last_change_utc advances ONLY on a real change_key change, and the result
        status stays UNCHANGED on a pure re-stamp (the data did not change)."""
        clock = FakeClock()
        processed: list = []
        src = pf.Source(
            name="s",
            fetch=lambda: {"v": 1},            # data never changes
            change_key=lambda d: d["v"],
            process=lambda ctx: processed.append(ctx.now),
            restamp=True,
        )
        eng = pf.PollerEngine([src], clock=clock, sleep=lambda s: None)
        statuses = []
        for _ in range(3):
            statuses.append(eng.poll_once()["s"].status)
            clock.advance(60)
        self.assertEqual(statuses, [pf.CHANGED, pf.UNCHANGED, pf.UNCHANGED])
        self.assertEqual(len(processed), 3)            # process ran EVERY cycle (re-stamp)
        self.assertEqual(eng.health("s").last_change_utc, processed[0])  # honest: first change only


# ---------------------------------------------------------------------------
# (a) Per-source isolation: a failing source never freezes the healthy ones
# ---------------------------------------------------------------------------

class TestPerSourceIsolation(unittest.TestCase):

    def test_failing_source_does_not_freeze_or_stale_healthy_sources(self):
        clock = FakeClock()
        sleep = RecordingSleep()           # retries don't advance real time
        counter = {"n": 0}
        good_changes = []
        steady_runs = []

        def good_fetch():
            counter["n"] += 1
            return {"v": counter["n"]}      # changes every cycle

        good = pf.Source(name="good", fetch=good_fetch,
                         change_key=lambda d: d["v"],
                         process=lambda ctx: good_changes.append(ctx.data["v"]))
        steady = pf.Source(name="steady", fetch=lambda: {"v": 1},
                           change_key=lambda d: d["v"],
                           process=lambda ctx: steady_runs.append(ctx.now))

        def bad_fetch():
            raise pf.TransientFetchError("down")

        bad = pf.Source(name="bad", fetch=bad_fetch,
                        change_key=lambda d: d, process=lambda ctx: None,
                        policy=_no_jitter_policy(max_retries=2, backoff_base_s=1.0))

        hb: list[dict] = []
        eng = pf.PollerEngine(
            [good, steady, bad],
            name="iso", clock=clock, sleep=sleep, rng=_zero_rng(),
            stale_after_s=600, fail_threshold=3,
            heartbeat=hb.append,
        )

        CYCLES = 5
        for _ in range(CYCLES):
            results = eng.poll_once()
            # Healthy sources succeed regardless of the failing one.
            self.assertEqual(results["good"].status, pf.CHANGED)
            self.assertIn(results["steady"].status, (pf.CHANGED, pf.UNCHANGED))
            self.assertEqual(results["bad"].status, pf.FETCH_FAILED)
            clock.advance(60)

        snap = eng.health_snapshot()
        # bad is loudly flagged; good + steady are fresh and NOT frozen.
        self.assertEqual(snap["sources"]["bad"]["state"], pf.FAILING)
        self.assertEqual(snap["sources"]["good"]["state"], pf.FRESH)
        self.assertEqual(snap["sources"]["steady"]["state"], pf.FRESH)
        self.assertFalse(snap["healthy"])           # overall unhealthy
        self.assertEqual(snap["worst_state"], pf.FAILING)

        # The healthy sources kept making progress every cycle (no freeze): good
        # processed all 5 changes, steady kept a recent success each cycle.
        self.assertEqual(good_changes, [1, 2, 3, 4, 5])
        good_h = eng.health("good")
        self.assertEqual(good_h.total_polls, CYCLES)
        self.assertEqual(good_h.consecutive_failures, 0)
        # last_success advanced to the final cycle's time (not stuck in the past).
        self.assertEqual(eng.health("steady").last_success_utc,
                         clock() - dt.timedelta(seconds=60))

        # The heartbeat was emitted on EVERY cycle - staleness is detectable.
        self.assertEqual(len(hb), CYCLES)
        self.assertTrue(all("sources" in s for s in hb))

    def test_heartbeat_emitted_even_when_all_sources_fail(self):
        """The exact freeze scenario: every source is down. The framework must
        STILL emit a heartbeat (flagged unhealthy) so a watcher can alarm,
        instead of silently preserving stale state."""
        clock = FakeClock()
        def boom():
            raise pf.TransientFetchError("all down")
        srcs = [pf.Source(name=n, fetch=boom, change_key=lambda d: d,
                          process=lambda ctx: None,
                          policy=_no_jitter_policy(max_retries=0))
                for n in ("a", "b")]
        hb: list[dict] = []
        eng = pf.PollerEngine(srcs, clock=clock, sleep=lambda s: None,
                              fail_threshold=1, heartbeat=hb.append)
        snap = eng.poll_once()
        self.assertTrue(all(r.status == pf.FETCH_FAILED for r in snap.values()))
        self.assertEqual(len(hb), 1)                 # heartbeat still fired
        self.assertFalse(hb[0]["healthy"])
        self.assertEqual(hb[0]["sources"]["a"]["state"], pf.FAILING)
        self.assertIsNotNone(hb[0]["generated_utc"])  # advancing timestamp

    def test_transient_source_recovers_and_clears_failing(self):
        """A source that fails for two cycles then recovers must flip back to
        fresh and reset its consecutive-failure count - without any effect on the
        always-healthy neighbor."""
        clock = FakeClock()
        state = {"fail": True}

        def flaky():
            if state["fail"]:
                raise pf.TransientFetchError("temporarily down")
            return {"v": clock().isoformat()}

        flaky_src = pf.Source(name="flaky", fetch=flaky,
                              change_key=lambda d: d["v"], process=lambda ctx: None,
                              policy=_no_jitter_policy(max_retries=0))
        steady = pf.Source(name="steady", fetch=lambda: {"v": 1},
                           change_key=lambda d: d["v"], process=lambda ctx: None)
        eng = pf.PollerEngine([flaky_src, steady], clock=clock,
                              sleep=lambda s: None, fail_threshold=2)

        for _ in range(2):                      # two failing cycles
            eng.poll_once()
            clock.advance(60)
        self.assertEqual(eng.health_snapshot()["sources"]["flaky"]["state"],
                         pf.FAILING)
        self.assertEqual(eng.health_snapshot()["sources"]["steady"]["state"],
                         pf.FRESH)

        state["fail"] = False                   # source comes back
        eng.poll_once()
        snap = eng.health_snapshot()
        self.assertEqual(snap["sources"]["flaky"]["state"], pf.FRESH)
        self.assertEqual(eng.health("flaky").consecutive_failures, 0)
        self.assertTrue(snap["healthy"])


# ---------------------------------------------------------------------------
# Process-failure handling: signature held so new data is retried
# ---------------------------------------------------------------------------

class TestProcessFailure(unittest.TestCase):

    def test_process_failure_holds_signature_and_retries(self):
        clock = FakeClock()
        attempts = {"n": 0}

        def process(ctx):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("sink write failed")
            # second time it succeeds

        src = pf.Source(name="s", fetch=lambda: {"v": 7},
                        change_key=lambda d: d["v"], process=process)
        eng = pf.PollerEngine([src], clock=clock, sleep=lambda s: None,
                              fail_threshold=5)

        r1 = eng.poll_once()["s"]
        self.assertEqual(r1.status, pf.PROCESS_FAILED)
        # Signature was NOT advanced, so the same (still-new) data is reprocessed.
        clock.advance(60)
        r2 = eng.poll_once()["s"]
        self.assertEqual(r2.status, pf.CHANGED)
        self.assertEqual(attempts["n"], 2)
        # Now it is settled: unchanged on the next cycle.
        clock.advance(60)
        self.assertEqual(eng.poll_once()["s"].status, pf.UNCHANGED)


# ---------------------------------------------------------------------------
# Freshness stamping + sinks + loop
# ---------------------------------------------------------------------------

class TestFreshnessAndSinks(unittest.TestCase):

    def test_freshness_stamp_shape_matches_feeds(self):
        now = dt.datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        valid = now - dt.timedelta(minutes=178)
        stamp = pf.freshness_stamp(valid, now=now)
        self.assertEqual(set(stamp), {
            "generated_utc", "latest_fix_valid_utc",
            "staleness_minutes", "staleness_seconds"})
        self.assertEqual(stamp["generated_utc"], "2026-06-01T12:00:00Z")
        self.assertEqual(stamp["latest_fix_valid_utc"], "2026-06-01T09:02:00Z")
        self.assertEqual(stamp["staleness_minutes"], 178)
        self.assertEqual(stamp["staleness_seconds"], 178 * 60)

    def test_iso_z_and_parse_iso_roundtrip(self):
        s = "2026-06-01T12:00:00Z"
        self.assertEqual(pf.iso_z(pf.parse_iso(s)), s)
        self.assertIsNone(pf.parse_iso(""))
        self.assertIsNone(pf.parse_iso(None))
        # naive datetime is treated as UTC
        naive = dt.datetime(2026, 6, 1, 12, 0, 0)
        self.assertEqual(pf.iso_z(naive), "2026-06-01T12:00:00Z")

    def test_process_writes_stamped_output_to_sink(self):
        clock = FakeClock(dt.datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC))
        sink = pf.DictSink()

        def process(ctx):
            ctx.sink.write(f"intensity/{ctx.name}.json",
                           {"value": ctx.data["v"], **ctx.freshness})

        src = pf.Source(
            name="wp", fetch=lambda: {"v": 42, "t": clock()},
            change_key=lambda d: d["v"], process=process,
            valid_time=lambda d: d["t"])
        eng = pf.PollerEngine([src], clock=clock, sleep=lambda s: None, sink=sink)
        eng.poll_once()
        out = sink.store["intensity/wp.json"]
        self.assertEqual(out["value"], 42)
        self.assertEqual(out["generated_utc"], "2026-06-01T12:00:00Z")
        self.assertEqual(out["latest_fix_valid_utc"], "2026-06-01T12:00:00Z")
        self.assertEqual(out["staleness_seconds"], 0.0)

    def test_file_sink_writes_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            sink = pf.FileSink(d)
            sink.write("sub/health.json", {"ok": True})
            import json
            path = os.path.join(d, "sub", "health.json")
            self.assertTrue(os.path.exists(path))
            self.assertEqual(json.load(open(path))["ok"], True)
            # no leftover temp file
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_sink_heartbeat_writes_snapshot_under_key(self):
        clock = FakeClock()
        sink = pf.DictSink()
        src = pf.Source(name="s", fetch=lambda: {"v": 1},
                        change_key=lambda d: d["v"], process=lambda ctx: None)
        eng = pf.PollerEngine([src], clock=clock, sleep=lambda s: None,
                              heartbeat=pf.sink_heartbeat(sink, "p/health.json"))
        eng.poll_once()
        self.assertIn("p/health.json", sink.store)
        self.assertIn("sources", sink.store["p/health.json"])

    def test_broken_heartbeat_never_crashes_cycle(self):
        clock = FakeClock()
        def boom_hb(_snap):
            raise RuntimeError("watcher sink down")
        src = pf.Source(name="s", fetch=lambda: {"v": 1},
                        change_key=lambda d: d["v"], process=lambda ctx: None)
        eng = pf.PollerEngine([src], clock=clock, sleep=lambda s: None,
                              heartbeat=boom_hb)
        # Must not raise even though the heartbeat sink throws.
        res = eng.poll_once()
        self.assertEqual(res["s"].status, pf.CHANGED)


class TestRunForever(unittest.TestCase):

    def test_run_forever_runs_exactly_max_cycles_and_sleeps_between(self):
        clock = FakeClock()
        sleep = RecordingSleep()
        src = pf.Source(name="s", fetch=lambda: {"v": 1},
                        change_key=lambda d: d["v"], process=lambda ctx: None)
        eng = pf.PollerEngine([src], clock=clock, sleep=sleep, interval_s=30)
        cycles = eng.run_forever(max_cycles=3)
        self.assertEqual(cycles, 3)
        # Sleeps between cycles only (after the last cycle it returns): 2 sleeps.
        self.assertEqual(sleep.calls, [30, 30])

    def test_loop_survives_a_source_that_raises_unexpectedly(self):
        """Even a non-standard error path (e.g. change_key blowing up) is caught
        and the loop keeps running and emitting health."""
        clock = FakeClock()
        def bad_change_key(_d):
            raise ValueError("weird")
        src = pf.Source(name="s", fetch=lambda: {"v": 1},
                        change_key=bad_change_key, process=lambda ctx: None)
        hb: list[dict] = []
        eng = pf.PollerEngine([src], clock=clock, sleep=lambda s: None,
                              heartbeat=hb.append, fail_threshold=1)
        eng.run_forever(max_cycles=2)
        self.assertEqual(len(hb), 2)
        self.assertEqual(hb[-1]["sources"]["s"]["state"], pf.FAILING)


class TestParentHeartbeatMem(unittest.TestCase):
    """process_mem_mb + the heartbeat's process block (ported from the
    worker branch so EVERY pf-based poller reports parent residency -
    the VPS-sizing telemetry needs per-service RSS, and the Railway
    bill is RAM-minutes)."""

    def test_process_mem_mb_reads_self(self):
        m = pf.process_mem_mb()
        self.assertGreater(m.get("rss_mb", 0), 1.0)
        self.assertGreaterEqual(m.get("peak_rss_mb", 0), m.get("rss_mb", 0))

    def test_health_snapshot_carries_process_mem(self):
        eng = pf.PollerEngine(name="t", sources=[], sink=pf.DictSink(),
                              interval_s=1, stale_after_s=10)
        snap = eng.health_snapshot()
        self.assertIn("process", snap)
        self.assertGreater(snap["process"].get("rss_mb", 0), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
