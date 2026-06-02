#!/usr/bin/env python3
"""Offline tests for hafs_render_poller - no network, no real render, no R2.

Mirrors the poller_framework / intensity_poller test style: inject a fake cycle
resolver, a fake render_fn, and a fake R2, drive the engine with poll_once /
run_forever(max_cycles=), and assert the change-gate, the watchdog-abort retry
(signature held), the off-season no-op, and the 3-pass upload+prune.

Run: python tests/test_hafs_render_poller.py -v
"""
import datetime as dt
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poller_framework as pf
import hafs_render_poller as hp


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self, start=dt.datetime(2026, 6, 2, 12, 0, 0)):
        self.t = start

    def __call__(self):
        self.t += dt.timedelta(seconds=1)
        return self.t


class FakeR2:
    """Records puts/deletes; list_keys reflects current state. No network."""
    def __init__(self, preload=None):
        self.store = dict(preload or {})       # key -> bytes
        self.deleted = []

    def put_bytes(self, key, data, content_type, cache):
        self.store[key] = data
        return True

    def put_json(self, key, obj, cache):
        self.store[key] = json.dumps(obj).encode()
        return True

    def list_keys(self, prefix):
        return [k for k in self.store if k.startswith(prefix)]

    def delete(self, keys):
        for k in keys:
            self.deleted.append(k)
            self.store.pop(k, None)


def _write_cycle_out(out_dir, cycle, products=("mslp_wind",), fxx=(0, 3)):
    """Create a minimal rendered out-dir (manifest + dummy PNGs)."""
    out = Path(out_dir)
    for p in products:
        for f in fxx:
            png = out / "hafsa" / "06w" / "storm" / p / f"f{f:03d}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            png.write_bytes(b"\x89PNG\r\n" + cycle.encode() + p.encode() + bytes([f]))
    (out / "manifest.json").write_text(json.dumps({"cycle": cycle, "storms": ["06w"]}))


# --------------------------------------------------------------------------- #
# Change-gate
# --------------------------------------------------------------------------- #
class TestChangeGate(unittest.TestCase):
    def _engine(self, cycles, calls, tmp):
        seq = iter(cycles)
        last = {"c": None}

        def resolver():
            try:
                last["c"] = next(seq)
            except StopIteration:
                pass
            return last["c"]

        def render(cycle, out_dir):
            calls.append(cycle)
            _write_cycle_out(out_dir, cycle)

        r2 = FakeR2()
        eng = hp.build_engine(
            r2, prefix="shadow/models/hafs", interval_s=1.0,
            clock=FakeClock(), sleep=lambda s: None,
            cycle_resolver=resolver, render_fn=render,
            out_dir_factory=lambda c: str(Path(tmp) / c))
        return eng, r2

    def test_renders_only_on_new_cycle(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            # same cycle twice, then a new one, then same new one again
            eng, r2 = self._engine(
                ["2026060206", "2026060206", "2026060212", "2026060212"],
                calls, tmp)
            eng.run_forever(max_cycles=4)
        # render fires only on the two DISTINCT cycles
        self.assertEqual(calls, ["2026060206", "2026060212"])

    def test_offseason_no_render(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            eng, r2 = self._engine([None, None], calls, tmp)
            eng.run_forever(max_cycles=2)
        self.assertEqual(calls, [])
        # health heartbeat still emitted (anti-freeze) even with nothing to do
        self.assertIn("shadow/models/hafs/poller_health.json", r2.store)


# --------------------------------------------------------------------------- #
# Watchdog abort -> process failure -> signature held -> retried
# --------------------------------------------------------------------------- #
class TestWatchdogRetry(unittest.TestCase):
    def test_render_failure_holds_signature_and_retries(self):
        calls = []

        def resolver():
            return "2026060206"   # same cycle every poll

        def render(cycle, out_dir):
            calls.append(cycle)
            # Simulate the watchdog firing (or a total-failure exit): raise.
            raise hp.RenderError("watchdog timeout (simulated)")

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = hp.build_engine(
                r2, prefix="shadow/models/hafs", interval_s=1.0,
                clock=FakeClock(), sleep=lambda s: None,
                cycle_resolver=resolver, render_fn=render,
                out_dir_factory=lambda c: str(Path(tmp) / c))
            eng.run_forever(max_cycles=3)
        # A failed render does NOT advance the change signature, so the SAME
        # cycle is retried every poll (3 attempts), never wedged, never skipped.
        self.assertEqual(calls, ["2026060206"] * 3)
        # health still emitted each cycle despite the failures
        self.assertIn("shadow/models/hafs/poller_health.json", r2.store)
        snap = json.loads(r2.store["shadow/models/hafs/poller_health.json"])
        self.assertFalse(snap["healthy"])   # the failing source shows unhealthy

    def test_no_upload_on_render_failure(self):
        """A render failure must NOT trigger the destructive 3-pass prune."""
        uploads = []

        def render(cycle, out_dir):
            raise hp.RenderError("boom")

        def uploader(r2, out_dir, prefix):
            uploads.append(prefix)
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2(preload={"shadow/models/hafs/hafsa/06w/storm/mslp_wind/f000.png": b"old"})
            eng = hp.build_engine(
                r2, prefix="shadow/models/hafs", interval_s=1.0,
                clock=FakeClock(), sleep=lambda s: None,
                cycle_resolver=lambda: "2026060206", render_fn=render,
                uploader=uploader, out_dir_factory=lambda c: str(Path(tmp) / c))
            eng.run_forever(max_cycles=2)
        self.assertEqual(uploads, [])          # never uploaded
        self.assertEqual(r2.deleted, [])       # prior frames never pruned


# --------------------------------------------------------------------------- #
# 3-pass upload + prune
# --------------------------------------------------------------------------- #
class TestUploadPrune(unittest.TestCase):
    def test_three_pass_upload_and_prune(self):
        prefix = "shadow/models/hafs"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            _write_cycle_out(out, "2026060206",
                             products=("mslp_wind", "refl"), fxx=(0, 3))
            # Pre-existing orphan PNG (a storm no longer rendered) + the manifest.
            r2 = FakeR2(preload={
                f"{prefix}/hafsa/99x/storm/mslp_wind/f000.png": b"orphan",
                f"{prefix}/manifest.json": b"{}",
            })
            summary = hp.upload_cycle(r2, str(out), prefix)

        # 4 fresh frames uploaded (2 products x 2 fxx) + manifest present
        self.assertEqual(summary["frames"], 4)
        self.assertIn(f"{prefix}/hafsa/06w/storm/mslp_wind/f000.png", r2.store)
        self.assertIn(f"{prefix}/hafsa/06w/storm/refl/f003.png", r2.store)
        self.assertIn(f"{prefix}/manifest.json", r2.store)
        # the orphan was pruned (scoped to *.png), the manifest was NOT
        self.assertEqual(summary["pruned"], 1)
        self.assertIn(f"{prefix}/hafsa/99x/storm/mslp_wind/f000.png", r2.deleted)
        self.assertIn(f"{prefix}/manifest.json", r2.store)

    def test_upload_requires_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "empty"
            out.mkdir()
            with self.assertRaises(hp.RenderError):
                hp.upload_cycle(FakeR2(), str(out), "shadow/models/hafs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
