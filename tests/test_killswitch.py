#!/usr/bin/env python3
"""Tests for the box-side R2 write kill switch (PHASE0 spec, 2026-08-25).

Three layers, all hermetic (no network, no boto client, no docker):

  * tat_killswitch module rules, driven through injected env / fetcher / clocks:
    env off, the fleet/ allowlist, disabled URL, armed-doc blocks, alert-doc
    allows, stale worker_ts allows, fetch error keeps the last good document
    then fails open, unparseable allows, the import guard, log throttling, a
    thread-safety smoke;
  * one integration test per guarded writer (s1_ingest.R2, meso_poller.R2,
    floater_poller.R2, hafs_render_poller.R2, intensity_poller.R2Sink,
    cyclolab_guidance_poller._R2) with a recording stub in place of the boto
    client: puts are skipped when blocked, happen when allowed, deletes are
    never touched, fleet/* always lands, and s1_ingest's ops ledger never
    counts a blocked put;
  * scripts/heartbeat.sh run for real against a sandbox checkout with a stub
    curl + stub s1_ingest on PATH: the mirror block writes fleet/breaker.json
    only after the heartbeat landed and only for a valid v1 document, and
    writes NOTHING on a failed or garbage fetch; plus scripts/fleet.sh's
    WRITE/NEVER container filters pinned against the spec's inventory, and
    scripts/box_writes.sh end to end on a throwaway compose project (env
    lands, guard probed in the container, a guard-less image reported
    ABSENT and failed);
  * s2_pyramid_emit's pass-level early exit: a blocked R2 pass returns
    before any LIST or fetch.

Run: python -m unittest tests.test_killswitch
"""
import datetime as dt
import io
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tat_killswitch as K  # noqa: E402

UTC = dt.timezone.utc
STATE_URL = "https://cdn.example.invalid/fleet/breaker.json"


def _read(path):
    with open(path) as f:
        return f.read()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class Clock:
    """Wall + monotonic clocks that only move when a test says so."""
    def __init__(self, wall=1_800_000_000.0, mono=1000.0):
        self.t_wall, self.t_mono = wall, mono

    def wall(self):
        return self.t_wall

    def mono(self):
        return self.t_mono

    def tick(self, s):
        self.t_wall += s
        self.t_mono += s

    def iso(self, offset_s=0.0):
        return dt.datetime.fromtimestamp(self.t_wall + offset_s, UTC).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z")


class Fetcher:
    """Stub for the breaker-document fetch: returns ``doc`` (or calls it),
    or raises ``exc``. Counts calls so poll cadence can be asserted."""
    def __init__(self, doc=None, exc=None):
        self.doc, self.exc, self.calls = doc, exc, 0

    def __call__(self):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.doc() if callable(self.doc) else self.doc


def status_doc(clock, mode="armed", writes_enabled=False, age_s=0.0, **extra):
    d = {"v": 1, "mode": mode, "writes_enabled": writes_enabled,
         "tripped_at": None if writes_enabled else clock.iso(-300),
         "trip_reason": None if writes_enabled else "rate_1h=200000",
         "worker_ts": clock.iso(-age_s)}
    d.update(extra)
    return d


def make(env=None, fetch=None, clock=None):
    clock = clock or Clock()
    env = {"TAT_BREAKER_STATE_URL": STATE_URL, **(env or {})}
    return K._Switch(env=env, fetch=fetch, wall=clock.wall, mono=clock.mono), clock


class LogCapture:
    """Context manager: every line tat.killswitch emits, via the logger."""
    def __enter__(self):
        self.records = []
        self.h = logging.Handler()
        self.h.emit = lambda r: self.records.append(r.getMessage())
        K._log.addHandler(self.h)
        return self

    def __exit__(self, *a):
        K._log.removeHandler(self.h)

    @property
    def lines(self):
        return list(self.records)


# --------------------------------------------------------------------------- #
# module rules
# --------------------------------------------------------------------------- #
class EnvRuleTest(unittest.TestCase):
    def test_off_values_block_any_case_and_whitespace(self):
        for v in ("0", "false", "FALSE", "off", "Off", "no", " NO "):
            with self.subTest(v=v):
                f = Fetcher(doc={})
                sw, _ = make(env={"TAT_R2_WRITES": v}, fetch=f)
                self.assertFalse(sw.writes_allowed("shadow/sat/x.webp"))
                self.assertEqual(sw.reason(), f"env TAT_R2_WRITES={v}")
                self.assertEqual(f.calls, 0, "env off must short-circuit the fetch")

    def test_on_or_unset_allows(self):
        for env in ({"TAT_R2_WRITES": "1"}, {"TAT_R2_WRITES": "true"},
                    {"TAT_R2_WRITES": ""}, {}):
            with self.subTest(env=env):
                sw, _ = make(env={**env, "TAT_BREAKER_STATE_URL": ""})
                self.assertTrue(sw.writes_allowed("shadow/sat/x.webp"))
                self.assertIsNone(sw.reason())

    def test_env_is_read_once_at_import(self):
        with mock.patch.dict(os.environ, {"TAT_R2_WRITES": "0",
                                          "TAT_BREAKER_STATE_URL": ""}):
            sw = K._reset_for_tests()          # env=None -> os.environ, once
            self.assertFalse(K.writes_allowed("shadow/x"))
            os.environ["TAT_R2_WRITES"] = "1"
            self.assertFalse(K.writes_allowed("shadow/x"),
                             "a live env change must NOT flip the switch; a restart does")
            self.assertIs(K._SW, sw)
        K._reset_for_tests(env={"TAT_BREAKER_STATE_URL": ""})


class AllowlistTest(unittest.TestCase):
    FLEET_KEYS = ("fleet/box1.json", "fleet/index.json", "fleet/breaker.json")

    def test_fleet_prefix_beats_env(self):
        sw, _ = make(env={"TAT_R2_WRITES": "0"})
        for k in self.FLEET_KEYS:
            self.assertTrue(sw.writes_allowed(k), k)
        self.assertFalse(sw.writes_allowed("fleetx/notallowed.json"))
        self.assertFalse(sw.writes_allowed("shadow/fleet/x.json"))

    def test_fleet_prefix_beats_armed_doc(self):
        clock = Clock()
        f = Fetcher(doc=lambda: status_doc(clock))
        sw, _ = make(fetch=f, clock=clock)
        self.assertFalse(sw.writes_allowed("shadow/sat/x.webp"))
        for k in self.FLEET_KEYS:
            self.assertTrue(sw.writes_allowed(k), k)

    def test_allowlisted_keys_never_fetch_or_count(self):
        f = Fetcher(doc={})
        sw, _ = make(fetch=f)
        for k in self.FLEET_KEYS:
            sw.writes_allowed(k)
        self.assertEqual(f.calls, 0)
        self.assertEqual(sw.state()["dropped_total"], 0)

    def test_empty_and_non_string_keys_are_judged_not_allowlisted(self):
        sw, _ = make(env={"TAT_R2_WRITES": "0"})
        self.assertFalse(sw.writes_allowed())
        self.assertFalse(sw.writes_allowed(None))


class BreakerDocTest(unittest.TestCase):
    def test_disabled_url_never_fetches(self):
        f = Fetcher(doc={})
        sw, _ = make(env={"TAT_BREAKER_STATE_URL": ""}, fetch=f)
        for _ in range(5):
            self.assertTrue(sw.writes_allowed("shadow/x"))
        self.assertEqual(f.calls, 0)
        self.assertIsNone(sw.state()["state_url"])

    def test_armed_and_tripped_blocks(self):
        clock = Clock()
        sw, _ = make(fetch=Fetcher(doc=status_doc(clock)), clock=clock)
        self.assertFalse(sw.writes_allowed("shadow/sat/x.webp"))
        self.assertIn("armed", sw.reason())
        self.assertIn("rate_1h=200000", sw.reason())

    def test_alert_mode_with_writes_enabled_never_blocks(self):
        clock = Clock()
        # a would-have-tripped alert doc (episode active, count up) still says
        # writes_enabled:true -- the Worker never sets it false by itself in
        # alert mode -- and must not stop writes: alert mode is advisory
        sw, _ = make(fetch=Fetcher(doc=status_doc(
            clock, mode="alert", writes_enabled=True, would_have_tripped_count=3,
            episode={"active": True, "started_at": clock.iso(-600),
                     "peak_rate_1h": 200000, "issue_url": "https://x/1"})), clock=clock)
        self.assertTrue(sw.writes_allowed("shadow/x"))
        self.assertIsNone(sw.reason())

    def test_manual_trip_blocks_in_any_mode(self):
        # writes_enabled is the Worker's ONE gate: an operator's POST /trip
        # sets it false in any mode ("works in any mode" in the contract),
        # and fleet.sh/RUNBOOK promise a fleet-wide stop for it -- so the
        # boxes must honour it in alert mode too (adversarial review finding)
        clock = Clock()
        for mode in ("alert", "armed"):
            with self.subTest(mode=mode):
                sw, _ = make(fetch=Fetcher(doc=status_doc(
                    clock, mode=mode, writes_enabled=False, trip_reason="manual")),
                    clock=clock)
                self.assertFalse(sw.writes_allowed("shadow/x"))
                self.assertIn("writes_enabled=false", sw.reason())
                self.assertIn(f"mode={mode}", sw.reason())
                self.assertIn("manual", sw.reason())

    def test_disarm_after_trip_keeps_blocking_until_reset(self):
        # /disarm drops mode to alert but "does NOT re-enable writes"; /reset
        # is "THE ONLY path back". A box that also required mode == armed
        # would resume on disarm -- a second, undocumented path back.
        clock = Clock()
        doc = {"d": status_doc(clock, mode="armed", writes_enabled=False)}
        f = Fetcher(doc=lambda: doc["d"])
        sw, _ = make(fetch=f, clock=clock)
        self.assertFalse(sw.writes_allowed("shadow/x"))
        doc["d"] = status_doc(clock, mode="alert", writes_enabled=False)   # disarm
        clock.tick(61)
        self.assertFalse(sw.writes_allowed("shadow/x"), "disarm must not resume writes")
        doc["d"] = status_doc(clock, mode="alert", writes_enabled=True)    # reset
        clock.tick(61)
        self.assertTrue(sw.writes_allowed("shadow/x"))

    def test_armed_with_writes_enabled_allows(self):
        clock = Clock()
        sw, _ = make(fetch=Fetcher(doc=status_doc(clock, writes_enabled=True)), clock=clock)
        self.assertTrue(sw.writes_allowed("shadow/x"))

    def test_stale_worker_ts_allows(self):
        clock = Clock()
        for age in (16 * 60, 3600, -16 * 60):
            with self.subTest(age=age):
                sw, _ = make(fetch=Fetcher(doc=status_doc(clock, age_s=age)), clock=clock)
                self.assertTrue(sw.writes_allowed("shadow/x"))
        # and just inside the window it still bites
        sw, _ = make(fetch=Fetcher(doc=status_doc(clock, age_s=14 * 60)), clock=clock)
        self.assertFalse(sw.writes_allowed("shadow/x"))

    def test_doc_goes_stale_in_cache_allows(self):
        # the cached doc stops being refreshed (fetch errors) and its
        # worker_ts ages past 15 min -> allowed even though the last GOOD
        # document says armed
        clock = Clock()
        f = Fetcher(doc=status_doc(clock))
        sw, _ = make(fetch=f, clock=clock)
        self.assertFalse(sw.writes_allowed("shadow/x"))
        f.exc = OSError("cdn down")
        clock.tick(14 * 60)
        self.assertFalse(sw.writes_allowed("shadow/x"), "last good doc stands < 15 min")
        clock.tick(2 * 60)
        self.assertTrue(sw.writes_allowed("shadow/x"), "past 15 min: fail open")

    def test_fetch_error_keeps_last_good_then_allows(self):
        clock = Clock()
        # the doc's worker_ts refreshes with the clock so only the FETCH age
        # governs here, isolating the last-good rule from the stale rule
        f = Fetcher(doc=lambda: status_doc(clock))
        sw, _ = make(fetch=f, clock=clock)
        self.assertFalse(sw.writes_allowed("shadow/x"))
        f.exc = OSError("cdn down")
        for _ in range(14):
            clock.tick(60)
            self.assertFalse(sw.writes_allowed("shadow/x"))
        self.assertGreaterEqual(sw.state()["fetch_errors"], 14)
        clock.tick(120)
        self.assertTrue(sw.writes_allowed("shadow/x"))
        self.assertIsNone(sw.reason())

    def test_first_fetch_error_allows_and_never_raises(self):
        for exc in (OSError("no route"), TimeoutError("5 s"), ValueError("bad json"),
                    RuntimeError("anything")):
            with self.subTest(exc=exc):
                sw, _ = make(fetch=Fetcher(exc=exc))
                self.assertTrue(sw.writes_allowed("shadow/x"))
                st = sw.state()
                self.assertEqual(st["fetch_errors"], 1)
                self.assertIn(type(exc).__name__, st["last_error"])

    def test_unparseable_allows(self):
        clock = Clock()
        bad = [
            [],                                              # not an object
            "armed",
            {},                                              # no fields
            {"v": 1, "mode": "armed", "writes_enabled": False},          # no worker_ts
            {"v": 1, "mode": "armed", "writes_enabled": False, "worker_ts": "yesterday"},
            {"v": 1, "mode": "armed", "writes_enabled": "false", "worker_ts": clock.iso()},
            {"v": 1, "mode": 7, "writes_enabled": False, "worker_ts": clock.iso()},
            {"v": 2, "mode": "armed", "writes_enabled": False, "worker_ts": clock.iso()},
            {"mode": "armed", "writes_enabled": False, "worker_ts": clock.iso()},  # no v
            {"v": 1, "mode": "armed", "writes_enabled": None, "worker_ts": clock.iso()},
        ]
        for doc in bad:
            with self.subTest(doc=doc):
                sw, _ = make(fetch=Fetcher(doc=doc), clock=clock)
                self.assertTrue(sw.writes_allowed("shadow/x"))

    def test_worker_ts_forms(self):
        # JS toISOString (fractional + Z), bare Z, explicit offset, naive
        clock = Clock()
        base = dt.datetime.fromtimestamp(clock.t_wall, UTC)
        for ts in (base.strftime("%Y-%m-%dT%H:%M:%S.123Z"),
                   base.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   base.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                   base.strftime("%Y-%m-%dT%H:%M:%S")):
            with self.subTest(ts=ts):
                sw, _ = make(fetch=Fetcher(doc=status_doc(clock, worker_ts=ts)), clock=clock)
                self.assertFalse(sw.writes_allowed("shadow/x"))

    def test_poll_cadence_once_per_window(self):
        clock = Clock()
        f = Fetcher(doc=lambda: status_doc(clock, writes_enabled=True))
        sw, _ = make(fetch=f, clock=clock)
        for _ in range(100):
            sw.writes_allowed("shadow/x")
            clock.tick(0.5)                                   # 50 s total
        self.assertEqual(f.calls, 1)
        clock.tick(10)
        sw.writes_allowed("shadow/x")
        self.assertEqual(f.calls, 2)

    def test_custom_poll_interval(self):
        clock = Clock()
        f = Fetcher(doc=lambda: status_doc(clock, writes_enabled=True))
        sw, _ = make(env={"TAT_BREAKER_POLL_S": "5"}, fetch=f, clock=clock)
        for _ in range(4):
            sw.writes_allowed("shadow/x")
            clock.tick(5)
        self.assertEqual(f.calls, 4)

    def test_reset_document_unblocks_on_next_poll(self):
        clock = Clock()
        doc = {"d": status_doc(clock)}
        f = Fetcher(doc=lambda: doc["d"])
        sw, _ = make(fetch=f, clock=clock)
        self.assertFalse(sw.writes_allowed("shadow/x"))
        doc["d"] = status_doc(clock, writes_enabled=True)
        self.assertFalse(sw.writes_allowed("shadow/x"), "cached until the poll window")
        clock.tick(61)
        with LogCapture() as cap:
            self.assertTrue(sw.writes_allowed("shadow/x"))
        self.assertTrue(any("ENABLED again" in l for l in cap.lines), cap.lines)

    def test_state_is_side_effect_free(self):
        clock = Clock()
        f = Fetcher(doc=status_doc(clock))
        sw, _ = make(fetch=f, clock=clock)
        for _ in range(3):
            st = sw.state()
        self.assertEqual(f.calls, 0, "state() must never fetch (health probes)")
        self.assertTrue(st["writes_allowed"])
        self.assertFalse(st["doc_present"])
        sw.writes_allowed("shadow/x")
        st = sw.state()
        for k in ("writes_allowed", "reason", "env_TAT_R2_WRITES", "state_url",
                  "poll_s", "doc_present", "doc_fetched_age_s", "doc_mode",
                  "doc_writes_enabled", "doc_worker_ts", "doc_worker_age_s",
                  "fetch_ok", "fetch_errors", "last_error", "blocked",
                  "blocked_for_s", "dropped_total"):
            self.assertIn(k, st)
        self.assertFalse(st["writes_allowed"])
        self.assertEqual(st["doc_mode"], "armed")
        self.assertIs(st["doc_writes_enabled"], False)
        self.assertEqual(st["dropped_total"], 1)
        self.assertEqual(f.calls, 1)


class LoggingTest(unittest.TestCase):
    def test_transitions_once_and_blocked_notes_throttled(self):
        clock = Clock()
        sw, _ = make(env={"TAT_R2_WRITES": "0"}, clock=clock)
        with LogCapture() as cap:
            for _ in range(100):                                # 50 s
                sw.writes_allowed("shadow/x")
                clock.tick(0.5)
            self.assertEqual(len(cap.lines), 1, cap.lines)
            self.assertIn("R2 writes DISABLED (env TAT_R2_WRITES=0)", cap.lines[0])
            clock.tick(10)                                      # >= 60 s since the note
            sw.writes_allowed("shadow/x")
            self.assertEqual(len(cap.lines), 2, cap.lines)
            self.assertRegex(cap.lines[1], r"dropped 100 put\(s\) since last note")
            for _ in range(10):                                 # still inside the window
                sw.writes_allowed("shadow/x")
            self.assertEqual(len(cap.lines), 2)
        self.assertEqual(sw.state()["dropped_total"], 111)

    def test_stale_note_once_per_ten_minutes(self):
        clock = Clock()
        sw, _ = make(fetch=Fetcher(doc=status_doc(clock, age_s=3600)), clock=clock)
        with LogCapture() as cap:
            for _ in range(50):
                sw.writes_allowed("shadow/x")
                clock.tick(6)                                   # 5 min
            self.assertEqual(sum("stale/unparseable" in l for l in cap.lines), 1, cap.lines)
            clock.tick(6 * 60)
            sw.writes_allowed("shadow/x")
            self.assertEqual(sum("stale/unparseable" in l for l in cap.lines), 2, cap.lines)

    def test_internal_error_fails_open_and_logs_once(self):
        sw, _ = make(env={"TAT_R2_WRITES": "0"})
        with mock.patch.object(sw, "_decide", side_effect=RuntimeError("boom")):
            with LogCapture() as cap:
                self.assertTrue(sw.writes_allowed("shadow/x"))
                self.assertTrue(sw.writes_allowed("shadow/x"))
                self.assertIsNone(sw.reason())
                self.assertTrue(sw.state()["writes_allowed"])
        self.assertEqual(len(cap.lines), 1, cap.lines)
        self.assertIn("internal error", cap.lines[0])

    def test_print_fallback_without_logging_config(self):
        sw, _ = make(env={"TAT_R2_WRITES": "0"})
        err = io.StringIO()
        with mock.patch.object(K._log, "hasHandlers", return_value=False), \
             mock.patch.object(sys, "stderr", err):
            sw.writes_allowed("shadow/x")
        self.assertIn("[killswitch] R2 writes DISABLED", err.getvalue())

    def test_logger_name(self):
        self.assertEqual(K._log.name, "tat.killswitch")


class ThreadSafetyTest(unittest.TestCase):
    def test_smoke(self):
        import time
        flip = {"n": 0}

        def doc():
            flip["n"] += 1
            time.sleep(0.002)                                   # a slow CDN
            return status_doc(clock, writes_enabled=(flip["n"] % 2 == 0))

        clock = Clock()
        f = Fetcher(doc=doc)
        # real monotonic clock with a 10 ms poll so several refreshes race
        sw = K._Switch(env={"TAT_BREAKER_STATE_URL": STATE_URL, "TAT_BREAKER_POLL_S": "1"},
                       fetch=f, wall=lambda: clock.t_wall, mono=time.monotonic)
        sw._poll_s = 0.01
        errors, results = [], []

        def worker():
            try:
                for i in range(300):
                    r = sw.writes_allowed(f"shadow/sat/{i}.webp")
                    results.append(r)
                    if i % 50 == 0:
                        sw.state()
                        sw.reason()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        ts = [threading.Thread(target=worker) for _ in range(12)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(30)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3600)
        self.assertTrue(all(r in (True, False) for r in results))
        self.assertGreaterEqual(f.calls, 1)
        self.assertFalse(sw._fetching, "the in-flight flag must always be released")


class ModuleSurfaceTest(unittest.TestCase):
    def test_public_api_and_mirror_header(self):
        for name in ("writes_allowed", "state", "reason"):
            self.assertTrue(callable(getattr(K, name)))
        self.assertEqual(K.ALLOW_PREFIXES, ("fleet/",))
        src = _read(os.path.join(ROOT, "tat_killswitch.py"))
        self.assertIn("MIRROR: identical copy lives in", src.splitlines()[3])
        self.assertEqual(K.DEFAULT_STATE_URL,
                         "https://cdn.triple-a-tropics.com/fleet/breaker.json")

    def test_stdlib_only(self):
        # some images have no requests; the switch must import on bare python
        src = _read(os.path.join(ROOT, "tat_killswitch.py"))
        for bad in ("import requests", "import boto3", "import yaml"):
            self.assertNotIn(bad, src)


# --------------------------------------------------------------------------- #
# writer integrations
# --------------------------------------------------------------------------- #
class _RecorderS3:
    """The only boto surface the writers touch. Records instead of sending."""
    def __init__(self):
        self.puts, self.deletes = [], []

    def put_object(self, **kw):
        self.puts.append(kw["Key"])

    def delete_objects(self, **kw):
        self.deletes.append([o["Key"] for o in kw["Delete"]["Objects"]])


def _blocked():
    return K._reset_for_tests(env={"TAT_R2_WRITES": "0", "TAT_BREAKER_STATE_URL": ""})


def _allowed():
    return K._reset_for_tests(env={"TAT_R2_WRITES": "1", "TAT_BREAKER_STATE_URL": ""})


class WriterIntegrationBase(unittest.TestCase):
    def tearDown(self):
        K._reset_for_tests(env={"TAT_BREAKER_STATE_URL": ""})


class S1IngestR2Test(WriterIntegrationBase):
    @classmethod
    def setUpClass(cls):
        import s1_ingest
        cls.W = s1_ingest

    def _r2(self):
        r2 = object.__new__(self.W.R2)        # skip __init__ (no boto client)
        r2.s3 = _RecorderS3()
        r2.ops = {}
        r2._ops_lock = threading.Lock()
        return r2

    def test_guard_bound(self):
        self.assertIs(self.W.tat_killswitch, K)

    def test_blocked_put_is_not_a_put(self):
        _blocked()
        r2 = self._r2()
        self.assertFalse(r2.put_bytes("shadow/sat/goes19/x.webp", b"x", "image/webp", "max-age=1"))
        self.assertFalse(r2.put_json("shadow/sat/goes19/manifest.json", {"a": 1}, "max-age=30"))
        self.assertEqual(r2.s3.puts, [])
        self.assertNotIn("put", r2.ops, "a blocked put must not count in the ops ledger")

    def test_fleet_keys_always_land_and_count(self):
        # heartbeat.sh's own PUT (fleet/<box>.json), the roster (fleet/index.json)
        # and the breaker mirror (fleet/breaker.json) all ride this method
        _blocked()
        r2 = self._r2()
        for k in ("fleet/box1.json", "fleet/index.json", "fleet/breaker.json"):
            self.assertTrue(r2.put_bytes(k, b"{}", "application/json", "max-age=60"), k)
        self.assertEqual(r2.s3.puts, ["fleet/box1.json", "fleet/index.json", "fleet/breaker.json"])
        self.assertEqual(r2.ops.get("put"), 3)

    def test_deletes_unaffected(self):
        _blocked()
        r2 = self._r2()
        r2.delete(["shadow/sat/a", "shadow/sat/b", ""])
        self.assertEqual(r2.s3.deletes, [["shadow/sat/a", "shadow/sat/b"]])
        self.assertEqual(r2.ops.get("delete_req"), 1)

    def test_allowed_writes_and_counts(self):
        _allowed()
        r2 = self._r2()
        self.assertTrue(r2.put_bytes("shadow/sat/goes19/x.webp", b"x", "image/webp", "max-age=1"))
        self.assertTrue(r2.put_json("shadow/sat/goes19/m.json", {}, "max-age=30"))
        self.assertEqual(len(r2.s3.puts), 2)
        self.assertEqual(r2.ops.get("put"), 2)

    def test_import_guard_none_means_allowed(self):
        _blocked()
        r2 = self._r2()
        with mock.patch.object(self.W, "tat_killswitch", None):
            self.assertTrue(r2.put_bytes("shadow/sat/x.webp", b"x", "image/webp", "max-age=1"))
        self.assertEqual(r2.s3.puts, ["shadow/sat/x.webp"])

    def test_s2_emit_store_is_the_guarded_class(self):
        # every s2 emit lane writes through s2_pyramid_emit's store, which is
        # s1_ingest.R2 -- so guarding that one method covers the whole ring
        import s2_pyramid_emit as E

        class Marker(self.W.R2):
            def __init__(self):
                pass
        with mock.patch.object(self.W, "R2_ENDPOINT", "https://r2.example"), \
             mock.patch.object(self.W, "R2", Marker):
            store, is_r2 = E._make_store("r2")
        self.assertIsInstance(store, Marker)
        self.assertTrue(is_r2)


class MesoPollerR2Test(WriterIntegrationBase):
    @classmethod
    def setUpClass(cls):
        import meso_poller
        cls.M = meso_poller

    def _r2(self):
        r2 = object.__new__(self.M.R2)
        r2.s3 = _RecorderS3()
        return r2

    def test_guard_bound(self):
        self.assertIs(self.M.tat_killswitch, K)

    def test_blocked_and_allowed(self):
        _blocked()
        r2 = self._r2()
        self.assertFalse(r2.put_bytes("meso/goes19/m1/x.webp", b"x", "image/webp", "max-age=1"))
        self.assertFalse(r2.put_json("meso/goes19/manifest.json", {}, "max-age=30"))
        self.assertTrue(r2.put_bytes("fleet/box1.json", b"{}", "application/json", "max-age=60"))
        r2.delete(["meso/goes19/old.webp"])
        self.assertEqual(r2.s3.puts, ["fleet/box1.json"])
        self.assertEqual(r2.s3.deletes, [["meso/goes19/old.webp"]])
        _allowed()
        self.assertTrue(r2.put_bytes("meso/goes19/m1/x.webp", b"x", "image/webp", "max-age=1"))
        self.assertEqual(r2.s3.puts[-1], "meso/goes19/m1/x.webp")


class FloaterPollerR2Test(WriterIntegrationBase):
    @classmethod
    def setUpClass(cls):
        import floater_poller
        cls.F = floater_poller

    def _r2(self):
        r2 = object.__new__(self.F.R2)
        r2.s3 = _RecorderS3()
        return r2

    def test_guard_bound(self):
        self.assertIs(self.F.tat_killswitch, K)

    def test_blocked_and_allowed(self):
        _blocked()
        r2 = self._r2()
        self.assertFalse(r2.put_bytes("floaters/al05/ir/x.webp", b"x", "image/webp", "max-age=1"))
        self.assertFalse(r2.put_json("floaters/index.json", {}, "max-age=30"))
        r2.delete(["floaters/al05/ir/old.webp"])
        self.assertEqual(r2.s3.puts, [])
        self.assertEqual(r2.s3.deletes, [["floaters/al05/ir/old.webp"]])
        _allowed()
        self.assertTrue(r2.put_json("floaters/index.json", {}, "max-age=30"))
        self.assertEqual(r2.s3.puts, ["floaters/index.json"])


class HafsRenderPollerR2Test(WriterIntegrationBase):
    @classmethod
    def setUpClass(cls):
        import hafs_render_poller
        cls.H = hafs_render_poller

    def _r2(self):
        r2 = object.__new__(self.H.R2)
        r2.s3 = _RecorderS3()
        r2.bucket = "b"
        return r2

    def test_guard_bound(self):
        self.assertIs(self.H.tat_killswitch, K)

    def test_blocked_and_allowed(self):
        _blocked()
        r2 = self._r2()
        self.assertFalse(r2.put_bytes("models/hafs/hfsa/06w/f000.png", b"x", "image/png", "max-age=1"))
        self.assertFalse(r2.put_json("models/hafs/manifest.json", {}, "max-age=30"))
        r2.delete(["models/hafs/hfsa/05w/f000.png"])
        self.assertEqual(r2.s3.puts, [])
        self.assertEqual(r2.s3.deletes, [["models/hafs/hfsa/05w/f000.png"]])
        _allowed()
        self.assertTrue(r2.put_bytes("models/hafs/hfsa/06w/f000.png", b"x", "image/png", "max-age=1"))
        self.assertEqual(r2.s3.puts, ["models/hafs/hfsa/06w/f000.png"])


class IntensityPollerSinkTest(WriterIntegrationBase):
    @classmethod
    def setUpClass(cls):
        import intensity_poller
        cls.I = intensity_poller

    def _sink(self):
        s = object.__new__(self.I.R2Sink)
        s.s3 = _RecorderS3()
        s.bucket = "b"
        return s

    def test_guard_bound(self):
        self.assertIs(self.I.tat_killswitch, K)

    def test_blocked_raises_and_allowed_writes(self):
        # A blocked PUT RAISES (R2Sink's documented raise-on-failure contract)
        # so the engine holds the signature -- never a silent no-op, which
        # would mark the feed as written while R2 keeps the pre-trip bytes.
        _blocked()
        s = self._sink()
        calls = (lambda: s.write("feeds/al_tracks_data.json", {"a": 1}),
                 lambda: s.write_html("cyclolab/al052026.html", "<html>"),
                 lambda: s.write_png("cyclolab/og/al052026.png", b"\x89PNG"))
        for call in calls:
            with self.assertRaises(self.I.WritesDisabled) as cm:
                call()
            self.assertIsInstance(cm.exception, RuntimeError)
            self.assertIn("kill switch", str(cm.exception))
            self.assertIn("TAT_R2_WRITES=0", str(cm.exception))
        self.assertEqual(s.s3.puts, [])
        # the framework's own health heartbeat rides write() too; it is a
        # feeds/ key, so it is gated like everything else (fleet/ is not)
        s.write("fleet/anything.json", {})
        self.assertEqual(s.s3.puts, ["fleet/anything.json"])
        _allowed()
        s.write("feeds/al_tracks_data.json", {"a": 1})
        s.write_html("cyclolab/al052026.html", "<html>")
        s.write_png("cyclolab/og/al052026.png", b"\x89PNG")
        self.assertEqual(s.s3.puts[1:], ["feeds/al_tracks_data.json",
                                         "cyclolab/al052026.html",
                                         "cyclolab/og/al052026.png"])

    def test_engine_holds_signature_while_blocked_and_reemits_after_reset(self):
        # The operational consequence, end to end through poller_framework:
        # feeds that change during a block are NOT marked written; on the
        # first cycle after writes resume every held feed lands, with no
        # upstream change and no container recreate. The health heartbeat
        # rides the same sink and must never kill the loop.
        import poller_framework as pf
        s = self._sink()
        data = {"v": 1}

        def process(ctx):
            ctx.sink.write("feeds/al_tracks_data.json", {"v": ctx.data["v"]})
            ctx.sink.write("feeds/al_ace_data.json", {"v": ctx.data["v"]})

        src = pf.Source(name="al", fetch=lambda: dict(data),
                        change_key=lambda d: d["v"], process=process)
        eng = pf.PollerEngine([src], sink=s, sleep=lambda x: None,
                              heartbeat=pf.sink_heartbeat(s, "feeds/poller_health.json"))
        _blocked()
        for _ in range(3):
            res = eng.poll_once()["al"]
            self.assertEqual(res.status, pf.PROCESS_FAILED)
            self.assertIn("kill switch", res.error)
        self.assertEqual(s.s3.puts, [], "nothing lands while blocked, health included")
        self.assertIsNone(eng.health("al").last_signature, "signature held, not advanced")
        _allowed()                                   # breaker reset / writes on
        res = eng.poll_once()["al"]
        self.assertEqual(res.status, pf.CHANGED, "same upstream data, re-emitted")
        self.assertEqual(s.s3.puts, ["feeds/al_tracks_data.json", "feeds/al_ace_data.json",
                                     "feeds/poller_health.json"])
        self.assertEqual(eng.health("al").last_signature, 1)
        self.assertEqual(eng.health("al").consecutive_failures, 0)


class GuidancePollerR2Test(WriterIntegrationBase):
    @classmethod
    def setUpClass(cls):
        import cyclolab_guidance_poller
        cls.G = cyclolab_guidance_poller

    def _r2(self):
        r2 = object.__new__(self.G._R2)
        r2.s3 = _RecorderS3()
        r2.bucket = "b"
        return r2

    def test_guard_bound(self):
        self.assertIs(self.G.tat_killswitch, K)

    def test_blocked_and_allowed(self):
        _blocked()
        r2 = self._r2()
        self.assertFalse(r2.put_json("cyclolab/guidance/al052026.json", {}))
        self.assertEqual(r2.s3.puts, [])
        _allowed()
        self.assertTrue(r2.put_json("cyclolab/guidance/al052026.json", {}))
        self.assertEqual(r2.s3.puts, ["cyclolab/guidance/al052026.json"])


class S2EmitPassGuardTest(WriterIntegrationBase):
    """A blocked emit pass must exit BEFORE its coverage LISTs and fetches
    (rc 0, one log line), not treadmill through every uncovered slot; a
    local store and an allowed switch go on exactly as before."""
    @classmethod
    def setUpClass(cls):
        import s2_pyramid_emit
        cls.E = s2_pyramid_emit
        cls.entry = next(e for e in cls.E.R.REGISTRY if e.tiled and e.sector_key == "conus")

    def _run(self, store_is_r2, step=True):
        E = self.E
        sentinel = RuntimeError("reached the listing/emit path")
        argv = ["--product", self.entry.product_id, "--store", "r2" if store_is_r2 else "local:/tmp/nope",
                "--prefix", "shadow"]
        if step:
            argv += ["--step", "10", "--backfill", "90"]
        out = io.StringIO()
        with mock.patch.object(E, "_make_store", return_value=(object(), store_is_r2)), \
             mock.patch.object(E, "_covered_times", side_effect=sentinel), \
             mock.patch.object(E, "emit_one", side_effect=sentinel), \
             mock.patch.object(E.P, "complete_stamps", side_effect=sentinel), \
             mock.patch.object(sys, "stdout", out):
            try:
                rc = E.main(argv)
            except RuntimeError as e:
                rc = e
        return rc, out.getvalue()

    def test_guard_bound(self):
        self.assertIs(self.E.tat_killswitch, K)

    def test_blocked_r2_pass_exits_clean_before_any_list_or_fetch(self):
        _blocked()
        for step in (True, False):
            with self.subTest(step=step):
                rc, out = self._run(store_is_r2=True, step=step)
                self.assertEqual(rc, 0, out)
                self.assertIn("[emit] R2 writes disabled by kill switch", out)
                self.assertIn("TAT_R2_WRITES=0", out)
                self.assertIn("skipping this pass", out)

    def test_blocked_but_local_store_is_not_gated(self):
        _blocked()
        rc, out = self._run(store_is_r2=False)
        self.assertIsInstance(rc, RuntimeError, "a local store must proceed to the emit path")
        self.assertNotIn("[emit] R2 writes disabled", out)

    def test_allowed_r2_pass_proceeds(self):
        _allowed()
        rc, out = self._run(store_is_r2=True)
        self.assertIsInstance(rc, RuntimeError)
        self.assertNotIn("[emit] R2 writes disabled", out)

    def test_import_guard_none_proceeds(self):
        _blocked()
        with mock.patch.object(self.E, "tat_killswitch", None):
            rc, out = self._run(store_is_r2=True)
        self.assertIsInstance(rc, RuntimeError)

    def test_probe_key_is_under_the_prefix_and_never_allowlisted(self):
        # the pass-level check judges the lane's own prefix, so a lane whose
        # prefix were ever fleet/ would (correctly) be allowlisted -- and no
        # emit lane's prefix is
        seen = []
        _blocked()
        with mock.patch.object(K, "writes_allowed", side_effect=lambda k: (seen.append(k), False)[1]):
            self._run(store_is_r2=True)
        self.assertEqual(seen, ["shadow/"])


# --------------------------------------------------------------------------- #
# scripts: heartbeat.sh mirror block (real bash, stub curl + stub s1_ingest)
# --------------------------------------------------------------------------- #
HEARTBEAT = os.path.join(ROOT, "scripts", "heartbeat.sh")
FLEET_SH = os.path.join(ROOT, "scripts", "fleet.sh")
BOX_WRITES = os.path.join(ROOT, "scripts", "box_writes.sh")

STUB_S1_INGEST = '''
import json, os
class R2:
    """Records every put_bytes as one JSON line; optionally fails the box's own
    heartbeat key so the mirror-only-after-success rule can be exercised."""
    def put_bytes(self, key, data, content_type, cache):
        if os.environ.get("HB_TEST_FAIL_KEY") == key:
            return False
        with open(os.environ["HB_TEST_REC"], "a") as f:
            f.write(json.dumps({"key": key, "data": data.decode(), "ct": content_type,
                                "cache": cache}) + "\\n")
        return True
'''

STUB_CURL = '''#!/usr/bin/env bash
# stub curl: STUB_CURL_MODE=ok|garbage|fail ; records the URL it was asked for
printf '%s\\n' "$*" >> "$STUB_CURL_LOG"
case "${STUB_CURL_MODE:-ok}" in
  ok)      printf '%s' "$STUB_CURL_BODY" ;;
  garbage) printf 'not json at all' ;;
  fail)    echo "curl: (22) The requested URL returned error: 503" >&2; exit 22 ;;
esac
'''

STUB_DOCKER = '''#!/usr/bin/env bash
exit 0
'''


def _worker_status(**over):
    now = dt.datetime.now(UTC)
    d = {"v": 1, "mode": "alert", "writes_enabled": True, "tripped_at": None,
         "trip_reason": None,
         "episode": {"active": False, "started_at": None, "peak_rate_1h": 0, "issue_url": None},
         "would_have_tripped_count": 0, "last_would_trip_at": None,
         "last_tick": now.isoformat(), "last_ok_tick": now.isoformat(),
         "heartbeat_age_s": 12, "tick_interval_s": 300, "rate_1h": 28000,
         "pace_15m": 30000, "warn_hourly": 80000, "trip_hourly": 150000,
         "verdict": "ok", "consecutive_over": 0, "consecutive_errors": 0,
         "last_error": None, "gaps_detected": 0,
         "worker_ts": (now - dt.timedelta(seconds=90)).strftime("%Y-%m-%dT%H:%M:%S.000Z")}
    d.update(over)
    return d


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class HeartbeatMirrorTest(unittest.TestCase):
    def _run(self, mode, body=None, fail_key=None):
        td = tempfile.mkdtemp(prefix="hb-")
        self.addCleanup(shutil.rmtree, td, ignore_errors=True)
        tsr = os.path.join(td, "tsr")
        bindir = os.path.join(td, "bin")
        os.makedirs(tsr)
        os.makedirs(bindir)
        rec = os.path.join(td, "puts.jsonl")
        curl_log = os.path.join(td, "curl.log")
        with open(os.path.join(tsr, "s1_ingest.py"), "w") as f:
            f.write(STUB_S1_INGEST)
        shutil.copy(os.path.join(ROOT, "fleet.yml"), os.path.join(tsr, "fleet.yml"))
        for name, src in (("curl", STUB_CURL), ("docker", STUB_DOCKER)):
            p = os.path.join(bindir, name)
            with open(p, "w") as f:
                f.write(src)
            os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
        env = dict(os.environ)
        env.update({
            "PATH": bindir + os.pathsep + env.get("PATH", ""),
            "TAT_TSR_DIR": tsr, "TAT_BOX_NAME": "boxtest",
            "HB_TEST_REC": rec, "STUB_CURL_MODE": mode, "STUB_CURL_LOG": curl_log,
            "STUB_CURL_BODY": json.dumps(body if body is not None else _worker_status()),
        })
        if fail_key:
            env["HB_TEST_FAIL_KEY"] = fail_key
        else:
            env.pop("HB_TEST_FAIL_KEY", None)
        p = subprocess.run(["bash", HEARTBEAT], env=env, capture_output=True,
                           text=True, timeout=120)
        puts = []
        if os.path.exists(rec):
            puts = [json.loads(l) for l in _read(rec).splitlines() if l.strip()]
        curl_calls = _read(curl_log) if os.path.exists(curl_log) else ""
        return p, puts, curl_calls

    def _by_key(self, puts):
        return {x["key"]: x for x in puts}

    def test_valid_status_is_mirrored_after_the_heartbeat(self):
        p, puts, curl_calls = self._run("ok")
        self.assertEqual(p.returncode, 0, p.stderr)
        keys = [x["key"] for x in puts]
        self.assertEqual(keys, ["fleet/index.json", "fleet/boxtest.json", "fleet/breaker.json"],
                         "roster, then the heartbeat, then (only then) the mirror")
        by = self._by_key(puts)
        hb = json.loads(by["fleet/boxtest.json"]["data"])
        self.assertEqual(hb["box"], "boxtest")
        self.assertEqual(hb["breaker_mode"], "alert")
        self.assertIsInstance(hb["breaker_age_s"], int)
        self.assertTrue(80 <= hb["breaker_age_s"] <= 200, hb["breaker_age_s"])
        m = by["fleet/breaker.json"]
        self.assertEqual(m["ct"], "application/json")
        self.assertEqual(m["cache"], "max-age=30")
        doc = json.loads(m["data"])
        self.assertEqual(doc["v"], 1)
        self.assertEqual(doc["mirrored_by"], "boxtest")
        self.assertRegex(doc["mirrored_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(doc["mode"], "alert")
        self.assertEqual(doc["rate_1h"], 28000)
        # exactly one Worker request per heartbeat run, to the status URL
        self.assertEqual(curl_calls.count("r2-breaker/status"), 1, curl_calls)
        self.assertIn("-m 5", curl_calls)

    def test_armed_tripped_status_mirrors_verbatim(self):
        body = _worker_status(mode="armed", writes_enabled=False,
                              trip_reason="rate_1h=200000")
        p, puts, _ = self._run("ok", body=body)
        self.assertEqual(p.returncode, 0, p.stderr)
        by = self._by_key(puts)
        doc = json.loads(by["fleet/breaker.json"]["data"])
        self.assertEqual(doc["mode"], "armed")
        self.assertIs(doc["writes_enabled"], False)
        self.assertEqual(json.loads(by["fleet/boxtest.json"]["data"])["breaker_mode"], "armed")

    def test_curl_failure_writes_no_mirror_and_nulls_the_age(self):
        p, puts, _ = self._run("fail")
        self.assertEqual(p.returncode, 0, p.stderr)
        keys = [x["key"] for x in puts]
        self.assertIn("fleet/boxtest.json", keys)
        self.assertNotIn("fleet/breaker.json", keys, "never clobber the last good mirror")
        hb = json.loads(self._by_key(puts)["fleet/boxtest.json"]["data"])
        self.assertIsNone(hb["breaker_age_s"])
        self.assertIsNone(hb["breaker_mode"])

    def test_garbage_body_writes_no_mirror(self):
        p, puts, _ = self._run("garbage")
        self.assertEqual(p.returncode, 0, p.stderr)
        keys = [x["key"] for x in puts]
        self.assertIn("fleet/boxtest.json", keys)
        self.assertNotIn("fleet/breaker.json", keys)
        hb = json.loads(self._by_key(puts)["fleet/boxtest.json"]["data"])
        self.assertIsNone(hb["breaker_age_s"])

    def test_wrong_contract_version_writes_no_mirror(self):
        for body in (_worker_status(v=2), {"hello": "world"}, [1, 2, 3]):
            with self.subTest(body=body):
                p, puts, _ = self._run("ok", body=body)
                self.assertEqual(p.returncode, 0, p.stderr)
                keys = [x["key"] for x in puts]
                self.assertIn("fleet/boxtest.json", keys)
                self.assertNotIn("fleet/breaker.json", keys)

    def test_failed_heartbeat_put_means_no_mirror_and_nonzero_exit(self):
        p, puts, _ = self._run("ok", fail_key="fleet/boxtest.json")
        self.assertNotEqual(p.returncode, 0)
        keys = [x["key"] for x in puts]
        self.assertNotIn("fleet/boxtest.json", keys)
        self.assertNotIn("fleet/breaker.json", keys, "mirror only AFTER the heartbeat landed")

    def test_heartbeat_json_stays_valid(self):
        # the script validates its own JSON before shipping; make sure the new
        # fields keep it valid in both the numeric and the null form
        for mode in ("ok", "fail"):
            p, puts, _ = self._run(mode)
            self.assertNotIn("refusing to publish malformed JSON", p.stderr)
            hb = json.loads(self._by_key(puts)["fleet/boxtest.json"]["data"])
            self.assertIn("breaker_age_s", hb)
            self.assertIn("breaker_mode", hb)

    def test_source_has_the_mirror_block(self):
        src = _read(HEARTBEAT)
        for needle in ("r2-breaker/status", "fleet/breaker.json", "mirrored_at",
                       "mirrored_by", "breaker_age_s", "breaker_mode",
                       "max-age=30", "curl -fsS -m 5"):
            self.assertIn(needle, src, needle)
        self.assertEqual(subprocess.run(["bash", "-n", HEARTBEAT]).returncode, 0)


# --------------------------------------------------------------------------- #
# scripts: fleet.sh writes/breaker verbs (no ssh: parse + the container filters)
# --------------------------------------------------------------------------- #
# the inventory from the spec (docker ps names on box1 + box2)
WRITE_CONTAINERS = [
    "tat-s2-conus-emit-cron-1", "tat-s2-conus-fast2-emit-cron-1",
    "tat-s2-g19fd-emit-cron-1", "tat-s2-g19fd-leads-emit-cron-1",
    "tat-s2-conus-fast-emit-cron-1", "tat-s2-geo-emit-cron-1", "tat-s2-mtg-emit-cron-1",
    "tat-s2-gk2a-emit-cron-1", "tat-s2-hwwpac-emit-cron-1", "tat-s2-hwfd-emit-cron-1",
    "tat-s2-hwfd-leads-emit-cron-1",
    "tat-render-floater-poller-1", "tat-render-intensity-poller-1",
    "tat-render-guidance-poller-1",
    "tat-s1-s1-ingest-1", "tat-s1-s1-ingest-himawari9-1", "tat-s1-s1-ingest-goes18-1",
    "tat-satellite-render-meso-poller-1",
] + [f"tat-overlays-{n}-poller-1" for n in
     ("mrms", "metar", "sfc", "nhc", "uhr", "ascat", "recon", "tpw", "salinity", "hy2", "sar")]

NEVER_CONTAINERS = [
    "tat-s2-prune-emit-cron-1",
    "tat-render-render-1", "tat-render-caddy-1", "tat-render-centerfix-plot-1",
    "tat-render-objfix-collector-1", "tat-render-ens-watchdog-1",
    "tat-s1-s1-render-1",
    "tat-satellite-render-meso-render-1", "tat-satellite-render-meso-render-cold-1",
    "tat-radar-render",
]


@unittest.skipUnless(shutil.which("bash") and shutil.which("grep"), "needs bash+grep")
class FleetShTest(unittest.TestCase):
    def _regexes(self):
        src = _read(FLEET_SH)
        w = re.search(r"^WRITE_RE='([^']+)'$", src, re.M)
        n = re.search(r"^NEVER_RE='([^']+)'$", src, re.M)
        self.assertTrue(w and n, "WRITE_RE / NEVER_RE must be single-quoted one-liners")
        return w.group(1), n.group(1)

    def _filter(self, names):
        """Run the names through the exact grep pipeline `writes` uses on a box."""
        write_re, never_re = self._regexes()
        p = subprocess.run(
            ["bash", "-c", "grep -E \"$1\" | grep -vE \"$2\" || true", "_", write_re, never_re],
            input="\n".join(names) + "\n", capture_output=True, text=True)
        return [l for l in p.stdout.splitlines() if l]

    def test_parses(self):
        self.assertEqual(subprocess.run(["bash", "-n", FLEET_SH]).returncode, 0)

    def test_write_lanes_all_selected_never_list_never(self):
        selected = self._filter(WRITE_CONTAINERS + NEVER_CONTAINERS)
        self.assertEqual(sorted(selected), sorted(WRITE_CONTAINERS))
        for n in NEVER_CONTAINERS:
            self.assertNotIn(n, selected, n)

    def test_never_list_is_excluded_even_if_the_write_regex_widens(self):
        # the NEVER gate must hold on its own
        _, never_re = self._regexes()
        p = subprocess.run(["bash", "-c", "grep -vE \"$1\" || true", "_", never_re],
                           input="\n".join(NEVER_CONTAINERS) + "\n",
                           capture_output=True, text=True)
        self.assertEqual(p.stdout.strip(), "")

    def test_usage_lists_the_new_verbs(self):
        p = subprocess.run(["bash", FLEET_SH], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("writes off|on", p.stdout)
        self.assertIn("breaker status|arm|disarm|trip|reset", p.stdout)
        for verb, args in (("writes", ["sideways"]), ("breaker", ["bogus"])):
            q = subprocess.run(["bash", FLEET_SH, verb] + args, capture_output=True, text=True)
            self.assertEqual(q.returncode, 2, q.stdout + q.stderr)

    def test_writes_pipes_the_box_script_same_image_no_deps(self):
        src = _read(FLEET_SH)
        body = src[src.index("cmd_writes()"):src.index("cmd_breaker()")]
        self.assertIn("bash -s\" < \"$REPO_DIR/scripts/box_writes.sh\"", body)
        self.assertIn("VAL=$val", body)
        self.assertNotIn(" on \"$host\"", body, "ssh -n would swallow the piped script")
        bw = _read(BOX_WRITES)
        self.assertEqual(subprocess.run(["bash", "-n", BOX_WRITES]).returncode, 0)
        code = bw[bw.index("set -uo pipefail"):]           # the header prose explains why not
        self.assertIn("up -d --no-build --no-deps --force-recreate", bw)
        self.assertIn("com.docker.compose.project.config_files", bw)
        self.assertIn("com.docker.compose.project.working_dir", bw)
        self.assertNotIn("docker restart", code, "restart keeps the old env (compose bakes env_file at create)")
        for forbidden in ("compose build", "compose pull", "docker rm", "compose down", "docker pull"):
            self.assertNotIn(forbidden, code, forbidden)
        # the verification is what makes "writes off" an honest report
        self.assertIn('grep -qx "TAT_R2_WRITES=$VAL"', bw)
        self.assertIn('DEFAULT_ENV="${DEFAULT_ENV:-/root/tsr-s2/.env}"', bw)

    def test_box_writes_probes_the_guard_inside_the_container(self):
        # env in the container is necessary, not sufficient: the writers
        # treat a missing tat_killswitch as ALLOWED, so the script must run
        # the module in the container and fail the box on ABSENT
        bw = _read(BOX_WRITES)
        code = bw[bw.index("set -uo pipefail"):]
        self.assertIn("guard_probe()", code)
        self.assertIn("docker exec", code)
        self.assertIn('import sys, tat_killswitch as k; sys.exit(0 if not k.writes_allowed("probe/box_writes") else 1)', code)
        self.assertIn('import sys, tat_killswitch as k; sys.exit(0 if k.state().get("writes_allowed") else 1)', code)
        self.assertIn('OVERLAYS_TAT_DIR="${OVERLAYS_TAT_DIR:-/work/tat}"', code)
        self.assertIn('tat-overlays-*) wd=(-w "$OVERLAYS_TAT_DIR")', code)
        self.assertIn("guard: present", code)
        self.assertIn("guard: ABSENT", code)
        # ABSENT must fail the box (rc=1 on the same line), never pass quietly
        absent_line = next(l for l in code.splitlines() if "guard: ABSENT" in l)
        self.assertIn("rc=1", absent_line)
        # the probe result gates the "verified" line: env grep alone is not enough
        self.assertIn('if guard_probe "$c"; then', code)
        # the probe key is not allowlisted (a fleet/ key would answer allowed)
        self.assertFalse("probe/box_writes".startswith(K.ALLOW_PREFIXES))

    def test_breaker_admin_header_and_key_sources(self):
        src = _read(FLEET_SH)
        self.assertIn("X-Admin-Key", src)
        self.assertIn("TAT_BREAKER_ADMIN_KEY", src)
        self.assertIn("/root/tsr-s2/.env", src[src.index("cmd_breaker()"):])
        self.assertIn("r2-breaker", src)


# --------------------------------------------------------------------------- #
# scripts/box_writes.sh end to end against a throwaway compose project
# (the docker half of `fleet.sh writes`; skipped without a daemon)
# --------------------------------------------------------------------------- #
def _docker_ok():
    try:
        return subprocess.run(["docker", "compose", "version"], capture_output=True,
                              timeout=20).returncode == 0 and \
            subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


COMPOSE_YML = """services:
  emit-cron:
    image: python:3-alpine
    command: ["sh", "-c", "while true; do sleep 5; done"]
    env_file: .env
    working_dir: /app
    volumes: ["%s:/app/tat_killswitch.py:ro"]
    profiles: ["cron"]
""" % os.path.join(ROOT, "tat_killswitch.py")

# same lane shape, but the image has NO tat_killswitch.py: the env lands and
# the process ignores it -- `writes off` must say so and fail
COMPOSE_YML_ABSENT = """services:
  emit-cron:
    image: python:3-alpine
    command: ["sh", "-c", "while true; do sleep 5; done"]
    env_file: .env
    working_dir: /app
    profiles: ["cron"]
"""


@unittest.skipUnless(_docker_ok(), "needs a docker daemon + compose plugin")
class BoxWritesDockerTest(unittest.TestCase):
    """PROVES the mechanism the spec got wrong: the value must land in the
    running process. A write lane and a NEVER lane (prune) share one compose
    file; `writes off` must recreate the first with TAT_R2_WRITES=0 and leave
    the second untouched (same container id, old env)."""
    def _compose(self, proj, *args, yml=None, timeout=240):
        return subprocess.run(["docker", "compose", "--profile", "cron", "-p", proj, "-f", yml or self.yml]
                              + list(args), capture_output=True, text=True, timeout=timeout)

    def _teardown(self, proj, yml=None):
        # `down` needs the profile or it leaves the profiled service behind;
        # rm -f is the belt to that brace so a failed test never strands one
        self._compose(proj, "down", "--remove-orphans", "-t", "1", yml=yml)
        subprocess.run(["docker", "rm", "-f", f"{proj}-emit-cron-1"], capture_output=True)

    def _project_dir(self, yml_text):
        td = tempfile.mkdtemp(prefix="bw-")
        self.addCleanup(shutil.rmtree, td, ignore_errors=True)
        yml = os.path.join(td, "docker-compose.yml")
        with open(yml, "w") as f:
            f.write(yml_text)
        with open(os.path.join(td, ".env"), "w") as f:
            f.write("TAT_R2_WRITES=1\nOTHER=keep\n")
        return td, yml

    def setUp(self):
        tag = f"{os.getpid()}{int(dt.datetime.now().timestamp()) % 100000}"
        self.WRITE = f"tat-s2-kstest{tag}"            # matches the WRITE shape
        self.NEVER = f"tat-s2-prune-kstest{tag}"      # matches NEVER_RE
        self.ABSENT = f"tat-s2-ksabsent{tag}"         # WRITE shape, no guard in the image
        self.td, self.yml = self._project_dir(COMPOSE_YML)
        self.td_absent, self.yml_absent = self._project_dir(COMPOSE_YML_ABSENT)
        self.default_env = os.path.join(self.td, "default.env")
        with open(self.default_env, "w") as f:
            f.write("R2_BUCKET=x\n")
        for proj, yml in ((self.WRITE, self.yml), (self.NEVER, self.yml),
                          (self.ABSENT, self.yml_absent)):
            self.addCleanup(self._teardown, proj, yml)
            r = self._compose(proj, "up", "-d", "emit-cron", yml=yml)
            if r.returncode != 0:
                self.skipTest(f"cannot start the throwaway project (pull blocked?): {r.stderr[-300:]}")

    def _inspect(self, name, fmt):
        r = subprocess.run(["docker", "inspect", "-f", fmt, name], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def _val(self, name):
        env = self._inspect(name, "{{range .Config.Env}}{{println .}}{{end}}").splitlines()
        vals = [l.split("=", 1)[1] for l in env if l.startswith("TAT_R2_WRITES=")]
        return vals[0] if vals else None

    def _run(self, val, write_re=None):
        never_re = re.search(r"^NEVER_RE='([^']+)'$", _read(FLEET_SH), re.M).group(1)
        env = dict(os.environ, VAL=str(val), NEVER_RE=never_re, DEFAULT_ENV=self.default_env,
                   WRITE_RE=write_re or f"^({self.WRITE}|{self.NEVER})-emit-cron-1$")
        return subprocess.run(["bash", BOX_WRITES], env=env, capture_output=True, text=True, timeout=300)

    def test_off_recreates_write_lane_only_and_verifies(self):
        w, n = self.WRITE + "-emit-cron-1", self.NEVER + "-emit-cron-1"
        self.assertEqual(self._val(w), "1")
        w_id0, n_id0 = self._inspect(w, "{{.Id}}"), self._inspect(n, "{{.Id}}")
        r = self._run(0)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f"recreated {w}  (TAT_R2_WRITES=0 verified in the container; guard: present)", r.stdout)
        self.assertNotIn(n, r.stdout, "the NEVER lane must not even be mentioned")
        self.assertEqual(self._val(w), "0", "the process must see the new value")
        self.assertNotEqual(self._inspect(w, "{{.Id}}"), w_id0, "a new container, not a restart")
        self.assertEqual(self._inspect(n, "{{.Id}}"), n_id0, "prune untouched")
        self.assertEqual(self._val(n), "1")
        self.assertEqual(self._inspect(w, "{{.State.Status}}"), "running")
        # every env file the lane loads, plus the fleet default, carry the value
        # and nothing else in them was disturbed
        self.assertIn("TAT_R2_WRITES=0\n", _read(os.path.join(self.td, ".env")))
        self.assertIn("OTHER=keep\n", _read(os.path.join(self.td, ".env")))
        self.assertIn("TAT_R2_WRITES=0\n", _read(self.default_env))
        self.assertIn("R2_BUCKET=x\n", _read(self.default_env))
        # and back on
        r = self._run(1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f"recreated {w}  (TAT_R2_WRITES=1 verified in the container; guard: present)", r.stdout)
        self.assertEqual(self._val(w), "1")
        self.assertEqual(_read(os.path.join(self.td, ".env")).count("TAT_R2_WRITES="), 1)

    def test_off_on_a_lane_without_the_guard_is_reported_absent_and_fails(self):
        # the success-on-empty trap: the env lands (docker inspect shows it)
        # but the image has no tat_killswitch.py, so the process keeps
        # writing. The script must say ABSENT and exit non-zero, never
        # "verified".
        a = self.ABSENT + "-emit-cron-1"
        r = self._run(0, write_re=f"^{self.ABSENT}-emit-cron-1$")
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self._val(a), "0", "the env still lands; that is the trap")
        self.assertIn(f"recreated {a}  (TAT_R2_WRITES=0 in its env; guard: ABSENT", r.stdout)
        self.assertIn("STILL WRITES", r.stdout)
        self.assertNotIn("guard: present", r.stdout)
        # and `writes on` reports the same defect (so it is found outside an incident)
        r = self._run(1, write_re=f"^{self.ABSENT}-emit-cron-1$")
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("guard: ABSENT", r.stdout)

    def test_no_matching_container_is_a_clean_noop(self):
        never_re = re.search(r"^NEVER_RE='([^']+)'$", _read(FLEET_SH), re.M).group(1)
        env = dict(os.environ, VAL="0", NEVER_RE=never_re, DEFAULT_ENV=self.default_env,
                   WRITE_RE="^tat-s2-does-not-exist-emit-cron-1$")
        r = subprocess.run(["bash", BOX_WRITES], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("no write containers running here", r.stdout)
        self.assertNotIn("TAT_R2_WRITES", _read(self.default_env),
                         "nothing to recreate means nothing to write")


if __name__ == "__main__":
    unittest.main()
