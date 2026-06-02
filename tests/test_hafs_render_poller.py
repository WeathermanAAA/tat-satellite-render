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


def _write_cycle_out(out_dir, cycle, products=("mslp_wind",), fxx=(0, 3),
                     storm="06w", model="hafsa", dom_slug="storm"):
    """Create a minimal rendered out-dir (manifest + dummy PNGs) for one pair.

    The manifest matches the generator's real schema: storms is a list of dicts
    {"id", "name", "frames": {model: {dom_slug: {product: [fxx]}}}} - the
    storm-level prune + per-pair coverage in upload_cycle parse exactly this."""
    out = Path(out_dir)
    pair_frames = {}
    for p in products:
        for f in fxx:
            png = out / model / storm / dom_slug / p / f"f{f:03d}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            png.write_bytes(b"\x89PNG\r\n" + cycle.encode() + p.encode() + bytes([f]))
        pair_frames[p] = list(fxx)
    (out / "manifest.json").write_text(json.dumps(
        {"cycle": cycle,
         "storms": [{"id": storm, "name": storm.upper(),
                     "frames": {model: {dom_slug: pair_frames}}}]}))


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
                diagnoser=lambda c: "(test-diag, no network)",
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
                uploader=uploader, diagnoser=lambda c: "(test-diag, no network)",
                out_dir_factory=lambda c: str(Path(tmp) / c))
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

    def test_upload_failure_is_atomic_no_manifest_no_prune(self):
        """If any PNG put fails (Pass 1), raise BEFORE writing the manifest or
        pruning - so the manifest never references an unpushed frame (no 404
        window) and a stale prior manifest is left intact."""
        prefix = "shadow/models/hafs"

        class FlakyR2(FakeR2):
            def put_bytes(self, key, data, content_type, cache):
                if key.endswith("refl/f003.png"):   # one frame fails to upload
                    return False
                return super().put_bytes(key, data, content_type, cache)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            _write_cycle_out(out, "2026060206",
                             products=("mslp_wind", "refl"), fxx=(0, 3))
            r2 = FlakyR2(preload={
                f"{prefix}/hafsa/99x/storm/mslp_wind/f000.png": b"orphan",
                f"{prefix}/manifest.json": b'{"old":true}',
            })
            with self.assertRaises(hp.RenderError):
                hp.upload_cycle(r2, str(out), prefix)
        # manifest NOT overwritten (prior stays), nothing pruned (no 404 window)
        self.assertEqual(r2.store[f"{prefix}/manifest.json"], b'{"old":true}')
        self.assertEqual(r2.deleted, [])

    def test_upload_requires_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "empty"
            out.mkdir()
            with self.assertRaises(hp.RenderError):
                hp.upload_cycle(FakeR2(), str(out), "shadow/models/hafs")

    def test_dual_writer_keeps_co_writer_current_storm_frames(self):
        """DUAL-WRITER GUARDRAIL: during the cron+worker co-write window the worker
        may skip a heavy pair (e.g. hafsb/parent) that the cron rendered for the
        SAME current storm. The prune must KEEP those co-writer frames (storm is
        live) and delete ONLY frames of storms no longer rendered at all."""
        prefix = "models/hafs"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            # Worker renders only the light hafsa/storm pair for current storm 06w.
            _write_cycle_out(out, "2026060206", products=("mslp_wind",), fxx=(0, 3),
                             storm="06w", model="hafsa", dom_slug="storm")
            r2 = FakeR2(preload={
                # cron co-wrote a pair the worker SKIPPED, same current storm 06w:
                f"{prefix}/hafsb/06w/parent/mslp_wind/f000.png": b"cron",
                f"{prefix}/hafsb/06w/parent/mslp_wind/f126.png": b"cron",
                # a storm no longer rendered at all (retired / prior cycle):
                f"{prefix}/hafsa/55x/storm/mslp_wind/f000.png": b"retired",
                f"{prefix}/manifest.json": b"{}",
            })
            summary = hp.upload_cycle(r2, str(out), prefix)

        # the co-writer's CURRENT-storm frames are KEPT (storm 06w is live)
        self.assertIn(f"{prefix}/hafsb/06w/parent/mslp_wind/f000.png", r2.store)
        self.assertIn(f"{prefix}/hafsb/06w/parent/mslp_wind/f126.png", r2.store)
        self.assertNotIn(f"{prefix}/hafsb/06w/parent/mslp_wind/f000.png", r2.deleted)
        # only the RETIRED storm's frame was pruned
        self.assertEqual(summary["pruned"], 1)
        self.assertIn(f"{prefix}/hafsa/55x/storm/mslp_wind/f000.png", r2.deleted)
        self.assertEqual(summary["storms"], ["06w"])
        # per-pair coverage is reported for the worker's rendered pair
        self.assertIn({"model": "hafsa", "storm": "06w", "domain": "storm",
                       "products": 1, "fxx": 2}, summary["coverage"])

    def test_offseason_prunes_all_when_no_storms(self):
        """Off-season (manifest storms=[]) -> current_storms empty -> every *.png
        under the prefix is a retired orphan and is pruned (R2 correctly cleared),
        but the manifest itself is never deleted."""
        prefix = "models/hafs"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "manifest.json").write_text(json.dumps({"cycle": None, "storms": []}))
            r2 = FakeR2(preload={
                f"{prefix}/hafsa/06w/storm/mslp_wind/f000.png": b"stale",
                f"{prefix}/manifest.json": b"{}",
            })
            summary = hp.upload_cycle(r2, str(out), prefix)
        self.assertEqual(summary["pruned"], 1)
        self.assertIn(f"{prefix}/hafsa/06w/storm/mslp_wind/f000.png", r2.deleted)
        self.assertIn(f"{prefix}/manifest.json", r2.store)   # manifest never pruned


# --------------------------------------------------------------------------- #
# Render-log parsing -> root-cause summary (drives render_summary.json)
# --------------------------------------------------------------------------- #
class TestParseRenderLog(unittest.TestCase):
    SAMPLE = (
        "INFO cycle 20260602 06Z - storms: ['06w', '90e']\n"
        "INFO planned 1734 ingest frame(s) + 19074 render task(s) across 2 storm(s) - 8 worker(s)\n"
        "INFO skip hafsb 90e parent.atm, incomplete (max f029 < f126)\n"
        "INFO skip hafsb 90e storm.atm, no frames published this cycle\n"
        "WARNING ingest failed: hafsa 90e parent.atm f030 - BrokenProcessPool (unrecoverable after retries)\n"
        "WARNING ingest failed: hafsa 90e parent.atm f033 - BrokenProcessPool (unrecoverable after retries)\n"
        "WARNING ingest failed: hafsa 06w parent.atm f120 - HTTPError: 503 Server Error: Service Unavailable\n"
        "INFO ingested 1490/1734 frame(s) ok (244 failed) in 1801s\n"
        "WARNING render failed: hafsa 90e storm.atm clean_ir f000 - ValueError: bad palette\n"
        "INFO rendered 16380 ok, 0 failed in 210s\n"
    )

    def test_parses_counts_skips_and_error_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "render.log")
            with open(log_path, "w") as f:
                f.write(self.SAMPLE)
            s = hp._parse_render_log(log_path)

        self.assertEqual(s["planned"], {"ingest_frames": 1734, "render_tasks": 19074,
                                        "storms": 2})
        self.assertEqual(s["ingest"], {"ok": 1490, "total": 1734, "failed": 244})
        self.assertEqual(s["render"], {"ok": 16380, "failed": 0})
        # both skipped hafsb/90e pairs captured with their reasons
        reasons = {(p["model"], p["storm"], p["domain"]): p["reason"]
                   for p in s["skipped_pairs"]}
        self.assertEqual(reasons[("hafsb", "90e", "parent.atm")],
                         "incomplete (max f029 < f126)")
        self.assertEqual(reasons[("hafsb", "90e", "storm.atm")],
                         "no frames published this cycle")
        # OOM is made obvious: BrokenProcessPool aggregated to a count of 2
        self.assertEqual(s["ingest_error_counts"]["BrokenProcessPool (unrecoverable after retries)"], 2)
        self.assertEqual(s["ingest_error_counts"]["HTTPError"], 1)
        self.assertEqual(len(s["failed_render"]), 1)
        self.assertEqual(s["failed_render"][0]["product"], "clean_ir")

    def test_empty_log_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "render.log")
            open(log_path, "w").close()
            s = hp._parse_render_log(log_path)
        self.assertEqual(s["skipped_pairs"], [])
        self.assertEqual(s["planned"], {})


# --------------------------------------------------------------------------- #
# Success path writes render_summary.json (self-report on success, not just fail)
# --------------------------------------------------------------------------- #
class TestSuccessSummary(unittest.TestCase):
    def test_render_summary_written_on_success(self):
        prefix = "models/hafs"

        def resolver():
            return "2026060206"

        def render(cycle, out_dir):
            _write_cycle_out(out_dir, cycle)
            # mimic run_render_subprocess's parsed-summary return value
            return {"render_seconds": 1980.0, "planned": {"storms": 2},
                    "ingest": {"ok": 1490, "total": 1734, "failed": 244},
                    "skipped_pairs": [{"model": "hafsb", "storm": "90e",
                                       "domain": "parent.atm",
                                       "reason": "incomplete (max f029 < f126)"}],
                    "ingest_error_counts": {"BrokenProcessPool (unrecoverable after retries)": 200}}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = hp.build_engine(
                r2, prefix=prefix, interval_s=1.0,
                clock=FakeClock(), sleep=lambda s: None,
                cycle_resolver=resolver, render_fn=render,
                out_dir_factory=lambda c: str(Path(tmp) / c))
            eng.run_forever(max_cycles=1)

        self.assertIn(f"{prefix}/render_summary.json", r2.store)
        snap = json.loads(r2.store[f"{prefix}/render_summary.json"])
        self.assertEqual(snap["cycle"], "2026060206")
        self.assertEqual(snap["storms"], ["06w"])           # from upload coverage
        self.assertEqual(snap["ingest"]["failed"], 244)     # from render summary
        self.assertEqual(snap["render_seconds"], 1980.0)
        self.assertEqual(snap["ingest_error_counts"]
                         ["BrokenProcessPool (unrecoverable after retries)"], 200)
        self.assertTrue(snap["coverage"])                   # per-pair coverage present


if __name__ == "__main__":
    unittest.main(verbosity=2)
