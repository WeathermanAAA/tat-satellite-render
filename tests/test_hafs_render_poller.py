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
    """Records puts/deletes; list_keys reflects current state. No network.
    ``mtimes`` backs the LastModified-aware list_objects (flat-key prune)."""
    def __init__(self, preload=None, mtimes=None):
        self.store = dict(preload or {})       # key -> bytes
        self.mtimes = dict(mtimes or {})       # key -> tz-aware datetime
        self.deleted = []

    def put_bytes(self, key, data, content_type, cache):
        self.store[key] = data
        self.mtimes.setdefault(
            key, dt.datetime(2026, 6, 4, 12, 0, tzinfo=dt.timezone.utc))
        return True

    def put_json(self, key, obj, cache):
        self.store[key] = json.dumps(obj).encode()
        return True

    def get_json(self, key):
        if key not in self.store:
            return None
        try:
            return json.loads(self.store[key])
        except Exception:  # noqa: BLE001
            return None

    def list_keys(self, prefix):
        return [k for k in self.store if k.startswith(prefix)]

    def list_objects(self, prefix):
        return [(k, self.mtimes.get(
                    k, dt.datetime(2026, 6, 4, 12, 0, tzinfo=dt.timezone.utc)))
                for k in self.store if k.startswith(prefix)]

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
            r2, progressive=False, prefix="shadow/models/hafs", interval_s=1.0,
            clock=FakeClock(), sleep=lambda s: None,
            cycle_resolver=resolver, render_fn=render,
            complete_pairs_fn=lambda c: (),   # no upstream movement: pure cycle gate
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
                r2, progressive=False, prefix="shadow/models/hafs", interval_s=1.0,
                clock=FakeClock(), sleep=lambda s: None,
                cycle_resolver=resolver, render_fn=render,
                complete_pairs_fn=lambda c: (),
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
                r2, progressive=False, prefix="shadow/models/hafs", interval_s=1.0,
                clock=FakeClock(), sleep=lambda s: None,
                cycle_resolver=lambda: "2026060206", render_fn=render,
                complete_pairs_fn=lambda c: (),
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
                r2, progressive=False, prefix=prefix, interval_s=1.0,
                clock=FakeClock(), sleep=lambda s: None,
                cycle_resolver=resolver, render_fn=render,
                complete_pairs_fn=lambda c: (),
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


# --------------------------------------------------------------------------- #
# Intra-cycle catch-up: late pairs/storms render incrementally, additively
# --------------------------------------------------------------------------- #
def _write_out_multi(out_dir, cycle, spec):
    """Rendered out-dir for arbitrary coverage. ``spec`` is
    {storm: {model: {dom_slug: {product: [fxx]}}}} -> dummy PNGs + a manifest
    in the generator's real schema (storms sorted by id)."""
    out = Path(out_dir)
    storms = []
    for storm, models in spec.items():
        frames = {}
        for model, doms in models.items():
            for dom_slug, prods in doms.items():
                for prod, fxxs in prods.items():
                    for f in fxxs:
                        png = out / model / storm / dom_slug / prod / f"f{f:03d}.png"
                        png.parent.mkdir(parents=True, exist_ok=True)
                        png.write_bytes(b"\x89PNG" + cycle.encode() + model.encode()
                                        + storm.encode() + prod.encode() + bytes([f]))
                frames.setdefault(model, {})[dom_slug] = {
                    p: list(v) for p, v in prods.items()}
        storms.append({"id": storm, "name": storm.upper(), "frames": frames})
    storms.sort(key=lambda s: s["id"])
    (out / "manifest.json").write_text(json.dumps(
        {"generated_at": "2026-06-04T12:00:00Z", "cycle": cycle,
         "storms": storms}))


class TestIntraCycleCatchup(unittest.TestCase):
    PREFIX = "models/hafs"

    def _engine(self, r2, tmp, resolver, pairs_fn, render, **kw):
        return hp.build_engine(
            r2, progressive=False, prefix=self.PREFIX, interval_s=1.0,
            clock=FakeClock(), sleep=lambda s: None,
            cycle_resolver=resolver, complete_pairs_fn=pairs_fn,
            render_fn=render,
            diagnoser=lambda c: "(test-diag, no network)",
            out_dir_factory=lambda c: str(Path(tmp) / c.replace("/", "_")),
            **kw)

    def test_late_pair_triggers_incremental_catchup(self):
        """The reported gap: hafsb/01e was at f123 when the cycle rendered ->
        skipped. When its f126 lands upstream, the catch-up must render EXACTLY
        the hafsb pairs, publish them additively (no deletes), merge the
        manifest, and clear them from skipped_pairs."""
        calls = []
        polls = {"n": 0}
        BASE = (("hafsa", "01e", "parent.atm"), ("hafsa", "01e", "storm.atm"))
        LATE = tuple(sorted(BASE + (("hafsb", "01e", "parent.atm"),
                                    ("hafsb", "01e", "storm.atm"))))

        def pairs_fn(cycle):
            polls["n"] += 1
            return BASE if polls["n"] == 1 else LATE

        def render(cycle, out_dir, **kw):
            calls.append((cycle, kw))
            if not kw:    # full render: only hafsa was complete at render time
                _write_out_multi(out_dir, cycle, {"01e": {"hafsa": {
                    "storm": {"mslp_wind": [0, 3]},
                    "parent": {"mslp_wind": [0, 3]}}}})
                return {"render_seconds": 100.0,
                        "ingest": {"ok": 86, "total": 86, "failed": 0},
                        "render": {"ok": 946, "failed": 0},
                        "skipped_pairs": [
                            {"model": "hafsb", "storm": "01e",
                             "domain": "storm.atm",
                             "reason": "incomplete (max f123 < f126)"},
                            {"model": "hafsb", "storm": "01e",
                             "domain": "parent.atm",
                             "reason": "incomplete (max f123 < f126)"}]}
            # catch-up render: exactly the late hafsb pairs
            _write_out_multi(out_dir, cycle, {"01e": {"hafsb": {
                "storm": {"mslp_wind": [0, 3]},
                "parent": {"mslp_wind": [0, 3]}}}})
            return {"render_seconds": 50.0,
                    "ingest": {"ok": 86, "total": 86, "failed": 0},
                    "render": {"ok": 946, "failed": 0}, "skipped_pairs": []}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060406", pairs_fn, render)
            eng.run_forever(max_cycles=3)   # full, catch-up, unchanged

        # one full render (no filters) + ONE catch-up with exact filters
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ("2026060406", {}))
        self.assertEqual(calls[1][0], "2026060406")
        self.assertEqual(calls[1][1], {"models": "hafsb", "storm": "01e",
                                       "domains": "parent.atm,storm.atm"})
        # additive: both models' PNGs live under the prefix; NOTHING deleted
        self.assertIn(f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f000.png", r2.store)
        self.assertIn(f"{self.PREFIX}/hafsb/01e/parent/mslp_wind/f003.png", r2.store)
        self.assertEqual(r2.deleted, [])
        # merged manifest carries BOTH models for the storm
        man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
        frames = man["storms"][0]["frames"]
        self.assertEqual(sorted(frames), ["hafsa", "hafsb"])
        self.assertEqual(frames["hafsb"]["parent"]["mslp_wind"], [0, 3])
        self.assertEqual(man["cycle"], "2026060406")
        # render_summary: totals accumulated, skipped cleared, catch-up audited
        snap = json.loads(r2.store[f"{self.PREFIX}/render_summary.json"])
        self.assertEqual(snap["frames"], 8)
        self.assertEqual(snap["skipped_pairs"], [])
        self.assertEqual(len(snap["catchups"]), 1)
        self.assertEqual(sorted(snap["catchups"][0]["pairs"]),
                         ["hafsb/01e/parent.atm", "hafsb/01e/storm.atm"])
        self.assertEqual(snap["ingest"], {"ok": 172, "total": 172, "failed": 0})
        self.assertEqual(snap["render"]["ok"], 1892)
        # coverage gained the hafsb pairs without losing the hafsa ones
        cov = {(c["model"], c["domain"]) for c in snap["coverage"]}
        self.assertEqual(cov, {("hafsa", "storm"), ("hafsa", "parent"),
                               ("hafsb", "storm"), ("hafsb", "parent")})

    def test_catchup_noop_when_full_render_already_covered(self):
        """Pairs that completed between fetch and the full render's own listing
        are already in the manifest - the next signature change must NOT
        re-render them (never re-render completed pairs)."""
        calls = []
        polls = {"n": 0}
        BASE = (("hafsa", "01e", "storm.atm"),)
        LATE = BASE + (("hafsb", "01e", "storm.atm"),)

        def pairs_fn(cycle):
            polls["n"] += 1
            return BASE if polls["n"] == 1 else LATE

        def render(cycle, out_dir, **kw):
            calls.append(kw)
            # full render covered hafsb too (its own listing saw it complete)
            _write_out_multi(out_dir, cycle, {"01e": {
                "hafsa": {"storm": {"mslp_wind": [0]}},
                "hafsb": {"storm": {"mslp_wind": [0]}}}})
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060406", pairs_fn, render)
            eng.run_forever(max_cycles=3)
        self.assertEqual(calls, [{}])   # the full render only - no catch-up

    def test_catchup_failure_holds_signature_and_retries(self):
        """A failed catch-up group must hold the spine signature (retry next
        poll) and leave the live manifest at its last good state."""
        calls = []
        polls = {"n": 0}
        fail_once = {"left": 1}
        BASE = (("hafsa", "01e", "storm.atm"),)
        LATE = BASE + (("hafsb", "01e", "storm.atm"),)

        def pairs_fn(cycle):
            polls["n"] += 1
            return BASE if polls["n"] == 1 else LATE

        def render(cycle, out_dir, **kw):
            calls.append(kw)
            if not kw:
                _write_out_multi(out_dir, cycle,
                                 {"01e": {"hafsa": {"storm": {"mslp_wind": [0]}}}})
                return {}
            if fail_once["left"]:
                fail_once["left"] -= 1
                raise hp.RenderError("catch-up boom")
            _write_out_multi(out_dir, cycle,
                             {"01e": {"hafsb": {"storm": {"mslp_wind": [0]}}}})
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060406", pairs_fn, render)
            eng.poll_once()    # full render
            eng.poll_once()    # catch-up attempt 1: fails
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(sorted(man["storms"][0]["frames"]), ["hafsa"])
            self.assertEqual(r2.deleted, [])
            eng.poll_once()    # catch-up attempt 2: succeeds
            eng.poll_once()    # unchanged

        self.assertEqual(calls, [
            {},
            {"models": "hafsb", "storm": "01e", "domains": "storm.atm"},
            {"models": "hafsb", "storm": "01e", "domains": "storm.atm"}])
        man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
        self.assertEqual(sorted(man["storms"][0]["frames"]), ["hafsa", "hafsb"])

    def test_catchup_gives_up_after_max_attempts(self):
        """A permanently-broken pair is abandoned after the cap (recorded in
        skipped_pairs) so it can't burn a render attempt every poll forever."""
        calls = []
        polls = {"n": 0}
        BASE = (("hafsa", "01e", "storm.atm"),)
        LATE = BASE + (("hafsb", "01e", "storm.atm"),)

        def pairs_fn(cycle):
            polls["n"] += 1
            return BASE if polls["n"] == 1 else LATE

        def render(cycle, out_dir, **kw):
            calls.append(kw)
            if not kw:
                _write_out_multi(out_dir, cycle,
                                 {"01e": {"hafsa": {"storm": {"mslp_wind": [0]}}}})
                return {}
            raise hp.RenderError("permanently broken pair")

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060406", pairs_fn, render,
                               catchup_max_attempts=2)
            eng.run_forever(max_cycles=6)

        # 1 full + exactly 2 catch-up attempts, then abandoned: no more renders
        self.assertEqual(len(calls), 3)
        snap = json.loads(r2.store[f"{self.PREFIX}/render_summary.json"])
        reasons = {(p["model"], p["storm"], p["domain"]): p["reason"]
                   for p in snap["skipped_pairs"]}
        self.assertIn("abandoned", reasons[("hafsb", "01e", "storm.atm")])

    def test_late_storm_appears_via_catchup(self):
        """A storm absent at render time (not just a pair) is picked up: it gets
        its own (model, storm) group and lands in the merged manifest."""
        calls = []
        polls = {"n": 0}
        BASE = (("hafsa", "06w", "storm.atm"),)
        LATE = BASE + (("hafsa", "01e", "storm.atm"),)

        def pairs_fn(cycle):
            polls["n"] += 1
            return BASE if polls["n"] == 1 else LATE

        def render(cycle, out_dir, **kw):
            calls.append(kw)
            if not kw:
                _write_out_multi(out_dir, cycle,
                                 {"06w": {"hafsa": {"storm": {"mslp_wind": [0]}}}})
                return {}
            _write_out_multi(out_dir, cycle,
                             {kw["storm"]: {kw["models"]:
                                            {"storm": {"mslp_wind": [0]}}}})
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060406", pairs_fn, render)
            eng.run_forever(max_cycles=3)

        self.assertEqual(calls, [{}, {"models": "hafsa", "storm": "01e",
                                      "domains": "storm.atm"}])
        man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
        self.assertEqual([s["id"] for s in man["storms"]], ["01e", "06w"])
        snap = json.loads(r2.store[f"{self.PREFIX}/render_summary.json"])
        self.assertEqual(snap["storms"], ["01e", "06w"])

    def test_partial_catchup_progress_preserved(self):
        """Two late groups; the second fails. The first group's publish must
        survive, and the retry must re-render ONLY the failed group."""
        calls = []
        polls = {"n": 0}
        fail = {"hafsb": 1}
        BASE = (("hafsa", "01e", "storm.atm"),)
        LATE = BASE + (("hafsa", "02e", "storm.atm"), ("hafsb", "01e", "storm.atm"))

        def pairs_fn(cycle):
            polls["n"] += 1
            return BASE if polls["n"] == 1 else LATE

        def render(cycle, out_dir, **kw):
            calls.append(kw)
            if not kw:
                _write_out_multi(out_dir, cycle,
                                 {"01e": {"hafsa": {"storm": {"mslp_wind": [0]}}}})
                return {}
            if kw["models"] == "hafsb" and fail["hafsb"]:
                fail["hafsb"] -= 1
                raise hp.RenderError("late hafsb flake")
            _write_out_multi(out_dir, cycle,
                             {kw["storm"]: {kw["models"]:
                                            {"storm": {"mslp_wind": [0]}}}})
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060406", pairs_fn, render)
            eng.run_forever(max_cycles=4)   # full; cu(02e ok, hafsb fail); retry; idle

        self.assertEqual(calls, [
            {},
            {"models": "hafsa", "storm": "02e", "domains": "storm.atm"},
            {"models": "hafsb", "storm": "01e", "domains": "storm.atm"},
            {"models": "hafsb", "storm": "01e", "domains": "storm.atm"}])
        man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
        self.assertEqual([s["id"] for s in man["storms"]], ["01e", "02e"])
        self.assertEqual(sorted(man["storms"][0]["frames"]), ["hafsa", "hafsb"])

    def test_new_cycle_resets_catchup_ledger(self):
        """A new cycle supersedes any pending catch-up: full render fires and
        the attempts/given-up ledger starts fresh."""
        calls = []
        cycles = iter(["2026060406", "2026060406", "2026060412"])
        last = {"c": None}

        def resolver():
            try:
                last["c"] = next(cycles)
            except StopIteration:
                pass
            return last["c"]

        polls = {"n": 0}

        def pairs_fn(cycle):
            polls["n"] += 1
            if cycle == "2026060412":
                return (("hafsa", "01e", "storm.atm"),)
            return ((("hafsa", "01e", "storm.atm"),) if polls["n"] == 1 else
                    (("hafsa", "01e", "storm.atm"), ("hafsb", "01e", "storm.atm")))

        def render(cycle, out_dir, **kw):
            calls.append((cycle, kw))
            if kw:
                raise hp.RenderError("late pair still failing")
            _write_out_multi(out_dir, cycle,
                             {"01e": {"hafsa": {"storm": {"mslp_wind": [0]}}}})
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, resolver, pairs_fn, render)
            eng.run_forever(max_cycles=3)   # full c1; cu fail; full c2

        self.assertEqual(calls[0], ("2026060406", {}))
        self.assertEqual(calls[1][1], {"models": "hafsb", "storm": "01e",
                                       "domains": "storm.atm"})
        self.assertEqual(calls[2], ("2026060412", {}))   # full render, no filter
        man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
        self.assertEqual(man["cycle"], "2026060412")

    def test_upload_catchup_atomic_and_never_prunes(self):
        """upload_catchup: all-or-nothing PNG pass before the manifest write,
        and NO pass-3 prune even with stale keys under the prefix."""
        prefix = self.PREFIX
        base = {"generated_at": "2026-06-04T12:00:00Z", "cycle": "2026060406",
                "storms": [{"id": "01e", "name": "01E", "frames":
                            {"hafsa": {"storm": {"mslp_wind": [0]}}}}]}

        class FlakyR2(FakeR2):
            def put_bytes(self, key, data, content_type, cache):
                if key.endswith("f003.png"):
                    return False
                return super().put_bytes(key, data, content_type, cache)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            _write_out_multi(out, "2026060406",
                             {"01e": {"hafsb": {"storm": {"mslp_wind": [0, 3]}}}})
            r2 = FlakyR2(preload={f"{prefix}/manifest.json": b'{"old":true}',
                                  f"{prefix}/hafsa/99x/storm/mslp_wind/f000.png":
                                  b"stale"})
            with self.assertRaises(hp.RenderError):
                hp.upload_catchup(r2, str(out), prefix, base)
            # manifest untouched, nothing deleted
            self.assertEqual(r2.store[f"{prefix}/manifest.json"], b'{"old":true}')
            self.assertEqual(r2.deleted, [])

            # clean R2: success path merges + never deletes the stale key
            r2 = FakeR2(preload={f"{prefix}/hafsa/99x/storm/mslp_wind/f000.png":
                                 b"stale"})
            res = hp.upload_catchup(r2, str(out), prefix, base)
            self.assertEqual(res["frames"], 2)
            self.assertEqual(res["pairs"], [("hafsb", "01e", "storm.atm")])
            self.assertEqual(r2.deleted, [])
            self.assertIn(f"{prefix}/hafsa/99x/storm/mslp_wind/f000.png", r2.store)
            man = json.loads(r2.store[f"{prefix}/manifest.json"])
            self.assertEqual(sorted(man["storms"][0]["frames"]),
                             ["hafsa", "hafsb"])

    def test_merge_manifest_and_pairs_helpers(self):
        base = {"generated_at": "old", "cycle": None, "storms": []}
        incr = {"generated_at": "new", "cycle": "2026060406",
                "storms": [{"id": "01e", "name": "01E", "frames":
                            {"hafsb": {"storm": {"mslp_wind": [3, 0]}}}}]}
        merged = hp.merge_manifest(base, incr)
        # an off-season/empty base is healed by the increment
        self.assertEqual(merged["cycle"], "2026060406")
        self.assertEqual(merged["generated_at"], "new")
        self.assertEqual(merged["storms"][0]["frames"]["hafsb"]["storm"]
                         ["mslp_wind"], [0, 3])
        # fxx union at the product level, base entries preserved
        base2 = {"cycle": "2026060406", "storms": [
            {"id": "01e", "frames": {"hafsb": {"storm": {"mslp_wind": [0]}}}}]}
        incr2 = {"cycle": "2026060406", "storms": [
            {"id": "01e", "frames": {"hafsb": {"storm": {"mslp_wind": [3],
                                                         "refl": [0]}}}}]}
        m2 = hp.merge_manifest(base2, incr2)
        self.assertEqual(m2["storms"][0]["frames"]["hafsb"]["storm"],
                         {"mslp_wind": [0, 3], "refl": [0]})
        # manifest_pairs maps slugs back to raw domains
        self.assertEqual(hp.manifest_pairs(m2),
                         {("hafsb", "01e", "storm.atm")})
        # _group_missing groups by (model, storm) with sorted domains
        groups = hp._group_missing({("hafsb", "01e", "storm.atm"),
                                    ("hafsb", "01e", "parent.atm"),
                                    ("hafsa", "02e", "storm.atm")})
        self.assertEqual(groups, {("hafsa", "02e"): ["storm.atm"],
                                  ("hafsb", "01e"): ["parent.atm", "storm.atm"]})

    def test_catchup_exit0_drop_retries_then_abandons(self):
        """REVIEW FINDING (confirmed): a catch-up group whose subprocess exits 0
        while one of its pairs produced ZERO frames (whole-run exit gate, not
        per-pair) must NOT advance the signature - the dropped pair retries
        next poll and a persistent failure becomes an audited give-up."""
        calls = []
        polls = {"n": 0}
        BASE = (("hafsa", "01e", "storm.atm"),)
        LATE = BASE + (("hafsb", "01e", "parent.atm"),
                       ("hafsb", "01e", "storm.atm"))

        def pairs_fn(cycle):
            polls["n"] += 1
            return BASE if polls["n"] == 1 else LATE

        def render(cycle, out_dir, **kw):
            calls.append(kw)
            if not kw:
                _write_out_multi(out_dir, cycle,
                                 {"01e": {"hafsa": {"storm": {"mslp_wind": [0]}}}})
                return {}
            if kw["domains"] == "parent.atm,storm.atm":
                # exit-0 PARTIAL: storm.atm rendered, parent.atm dropped (OOM)
                _write_out_multi(out_dir, cycle,
                                 {"01e": {"hafsb": {"storm": {"mslp_wind": [0]}}}})
                return {}
            # single-pair retry: the real generator exits 1 (n_ok==0) -> raises
            raise hp.RenderError("render exit 1 (all frames failed)")

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060406", pairs_fn, render,
                               catchup_max_attempts=2)
            res1 = eng.poll_once()["hafs"]            # full render: ok
            self.assertEqual(res1.status, pf.CHANGED)
            res2 = eng.poll_once()["hafs"]            # partial group: published
            self.assertEqual(res2.status, pf.PROCESS_FAILED)   # ...but HELD
            # the successful domain IS live despite the held signature
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(sorted(man["storms"][0]["frames"]["hafsb"]),
                             ["storm"])
            res3 = eng.poll_once()["hafs"]            # retry parent alone: fails
            self.assertEqual(res3.status, pf.PROCESS_FAILED)
            res4 = eng.poll_once()["hafs"]            # cap hit: audited give-up
            self.assertEqual(res4.status, pf.CHANGED)
            res5 = eng.poll_once()["hafs"]            # quiet
            self.assertEqual(res5.status, pf.UNCHANGED)

        self.assertEqual(calls, [
            {},
            {"models": "hafsb", "storm": "01e",
             "domains": "parent.atm,storm.atm"},
            {"models": "hafsb", "storm": "01e", "domains": "parent.atm"}])
        snap = json.loads(r2.store[f"{self.PREFIX}/render_summary.json"])
        reasons = {(p["model"], p["storm"], p["domain"]): p["reason"]
                   for p in snap["skipped_pairs"]}
        self.assertIn("abandoned",
                      reasons[("hafsb", "01e", "parent.atm")])
        self.assertNotIn(("hafsb", "01e", "storm.atm"), reasons)
        self.assertEqual(snap["catchups"][0]["pairs"], ["hafsb/01e/storm.atm"])

    def test_full_render_drop_self_heals_via_catchup(self):
        """REVIEW FINDING (confirmed): a pair complete upstream BEFORE the fetch
        but dropped by the full render (exit 0 - another pair kept n_ok>0) has
        no upstream change to reopen the gate. The gate-reopen guard must hold
        the signature so the next poll catch-up renders it."""
        calls = []
        PAIRS = (("hafsa", "01e", "storm.atm"), ("hafsb", "01e", "storm.atm"))

        def render(cycle, out_dir, **kw):
            calls.append(kw)
            if not kw:
                # full render DROPPED hafsb (OOM analog) but exited 0
                _write_out_multi(out_dir, cycle,
                                 {"01e": {"hafsa": {"storm": {"mslp_wind": [0]}}}})
                return {}
            _write_out_multi(out_dir, cycle,
                             {"01e": {"hafsb": {"storm": {"mslp_wind": [0]}}}})
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060406",
                               lambda c: PAIRS, render)
            res1 = eng.poll_once()["hafs"]   # publishes hafsa, HOLDS signature
            self.assertEqual(res1.status, pf.PROCESS_FAILED)
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(sorted(man["storms"][0]["frames"]), ["hafsa"])
            res2 = eng.poll_once()["hafs"]   # catch-up heals the dropped pair
            self.assertEqual(res2.status, pf.CHANGED)
            res3 = eng.poll_once()["hafs"]
            self.assertEqual(res3.status, pf.UNCHANGED)

        self.assertEqual(calls, [{}, {"models": "hafsb", "storm": "01e",
                                      "domains": "storm.atm"}])
        man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
        self.assertEqual(sorted(man["storms"][0]["frames"]),
                         ["hafsa", "hafsb"])

    def test_list_complete_pairs_stubbed(self):
        """Pure-logic check of the upstream completeness probe with a stubbed
        hafs_render module (the real one pulls the GRIB stack)."""
        import types
        present = {
            "hfsa/20260604/06/01e.2026060406.hfsa.storm.atm.f126.grb2",
            "hfsa/20260604/06/01e.2026060406.hfsa.parent.atm.f126.grb2",
            "hfsb/20260604/06/01e.2026060406.hfsb.storm.atm.f126.grb2",
        }
        stub = types.ModuleType("hafs_render.generate_hafs_plots")
        stub.MODEL_TOKEN = {"hafsa": "hfsa", "hafsb": "hfsb"}
        stub.TERMINAL_FXX = 126
        stub._s3_list = (lambda prefix, delimiter=None, session=None:
                         ([k for k in present if k == prefix], []))
        stub.list_storms = lambda model, date, hh, session=None: ["01e"]
        pkg = types.ModuleType("hafs_render")
        pkg.generate_hafs_plots = stub
        saved = {n: sys.modules.get(n)
                 for n in ("hafs_render", "hafs_render.generate_hafs_plots")}
        sys.modules["hafs_render"] = pkg
        sys.modules["hafs_render.generate_hafs_plots"] = stub
        try:
            pairs = hp.list_complete_pairs("2026060406")
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m
        self.assertEqual(pairs, (("hafsa", "01e", "parent.atm"),
                                 ("hafsa", "01e", "storm.atm"),
                                 ("hafsb", "01e", "storm.atm")))


# --------------------------------------------------------------------------- #
# PROGRESSIVE frame-load: render fhrs as they post, cycle-scoped keys,
# manifest v2 (cycles[] + zero-blink legacy), completion + prune, bootstrap
# --------------------------------------------------------------------------- #
FAKE_HEADERS = {
    "product": {"slug": "mslp_wind", "label": "W", "short": "Wind"},
    "products": [{"slug": "mslp_wind", "label": "W", "short": "Wind"}],
    "models": [{"slug": "hafsa", "label": "HAFS-A"},
               {"slug": "hafsb", "label": "HAFS-B"}],
    "domains": [{"slug": "storm", "label": "S", "raw": "storm.atm"},
                {"slug": "parent", "label": "P", "raw": "parent.atm"}],
    "n_products": 1,
}


class TestProgressive(unittest.TestCase):
    """Frame-granular progressive source, with a tiny grid (FXX_END=6) so a
    full cycle is f000/f003/f006 per pair."""
    PREFIX = "models/hafs"

    def setUp(self):
        self._saved = (hp.FXX_END, hp.FXX_STEP)
        hp.FXX_END, hp.FXX_STEP = 6, 3

    def tearDown(self):
        hp.FXX_END, hp.FXX_STEP = self._saved

    def _engine(self, r2, tmp, resolver, posted_fn, render, **kw):
        return hp.build_engine(
            r2, progressive=True, prefix=self.PREFIX, interval_s=1.0,
            clock=FakeClock(), sleep=lambda s: None,
            cycle_resolver=resolver, posted_frames_fn=posted_fn,
            render_fn=render, headers_fn=lambda: FAKE_HEADERS,
            out_dir_factory=lambda c: str(Path(tmp) / c.replace("/", "_")),
            **kw)

    @staticmethod
    def _prog_render(plan):
        """A fake render honoring the progressive filters: writes exactly the
        frames asked for (models x storm x domains x only_fxx), recording each
        call's kwargs into ``plan`` (a list)."""
        def render(cycle, out_dir, **kw):
            plan.append(dict(kw))
            fxxs = [int(x) for x in kw["only_fxx"].split(",")]
            doms = kw["domains"].split(",")
            slug = {"storm.atm": "storm", "parent.atm": "parent"}
            spec = {kw["storm"]: {kw["models"]: {
                slug[d]: {"mslp_wind": list(fxxs)} for d in doms}}}
            _write_out_multi(out_dir, cycle, spec)
            return {"render_seconds": 5.0,
                    "ingest": {"ok": len(fxxs) * len(doms),
                               "total": len(fxxs) * len(doms), "failed": 0},
                    "render": {"ok": len(fxxs) * len(doms), "failed": 0}}
        return render

    def test_pre_announce_then_progressive_build_to_completion(self):
        calls = []
        polls = {"n": 0}
        F = lambda *fr: tuple(sorted(fr))
        seq = [
            (),                                                    # storm_info only
            F(("hafsa", "01e", "storm.atm", 0)),                   # f000 posts
            F(("hafsa", "01e", "storm.atm", 0),
              ("hafsa", "01e", "storm.atm", 3),
              ("hafsa", "01e", "parent.atm", 0)),                  # more posts
            F(("hafsa", "01e", "storm.atm", 0),
              ("hafsa", "01e", "storm.atm", 3),
              ("hafsa", "01e", "storm.atm", 6),
              ("hafsa", "01e", "parent.atm", 0),
              ("hafsa", "01e", "parent.atm", 3),
              ("hafsa", "01e", "parent.atm", 6)),                  # terminal
        ]

        def posted_fn(cycle):
            i = min(polls["n"], len(seq) - 1)
            polls["n"] += 1
            return seq[i]

        # previous complete cycle lives in the live manifest (bootstrap source)
        prev_man = {
            "cycles": [{"cycle": "2026060412", "in_progress": False,
                        "frames_done": 6, "frames_expected": 6,
                        "storms": [{"id": "01e", "name": "01E", "frames":
                                    {"hafsa": {"storm": {"mslp_wind": [0, 3, 6]}}}}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2(preload={
                f"{self.PREFIX}/manifest.json": json.dumps(prev_man).encode()})
            eng = self._engine(r2, tmp, lambda: "2026060418", posted_fn,
                               self._prog_render(calls))

            eng.poll_once()    # pre-announce (no frames posted yet)
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"][0]["cycle"], "2026060418")
            self.assertTrue(man["cycles"][0]["in_progress"])
            self.assertEqual(man["cycles"][0]["storms"], [])
            # legacy fields still serve the PREVIOUS complete cycle (zero-blink)
            self.assertEqual(man["cycle"], "2026060412")
            self.assertTrue(man["path_template"].startswith("2026060412/"))
            self.assertNotIn("{cycle}", man["path_template"])
            self.assertEqual(calls, [])

            eng.poll_once()    # f000 lands -> first batch
            self.assertEqual(calls[-1], {"models": "hafsa", "storm": "01e",
                                         "domains": "storm.atm",
                                         "only_fxx": "0"})
            self.assertIn(f"{self.PREFIX}/2026060418/hafsa/01e/storm/"
                          "mslp_wind/f000.png", r2.store)
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"][0]["storms"][0]["frames"]["hafsa"]
                             ["storm"]["mslp_wind"], [0])
            self.assertEqual(man["cycles"][0]["frames_done"], 1)
            self.assertEqual(man["cycle"], "2026060412")   # legacy unchanged

            eng.poll_once()    # delta: TWO exact batches (asymmetric sets)
            self.assertEqual(calls[-2], {"models": "hafsa", "storm": "01e",
                                         "domains": "parent.atm",
                                         "only_fxx": "0"})
            self.assertEqual(calls[-1], {"models": "hafsa", "storm": "01e",
                                         "domains": "storm.atm",
                                         "only_fxx": "3"})
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"][0]["frames_done"], 3)
            self.assertTrue(man["cycles"][0]["in_progress"])

            eng.poll_once()    # terminal frames -> COMPLETE
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertFalse(man["cycles"][0]["in_progress"])
            self.assertEqual(man["cycles"][0]["frames_done"], 6)
            # legacy fields FLIP to the newly-completed cycle
            self.assertEqual(man["cycle"], "2026060418")
            self.assertTrue(man["path_template"].startswith("2026060418/"))
            # flip-back target retained
            self.assertEqual([c["cycle"] for c in man["cycles"]],
                             ["2026060418", "2026060412"])
            # no frame ever re-rendered
            asked = [f for c in calls for f in c["only_fxx"].split(",")]
            self.assertEqual(len(asked), 6)

            eng.poll_once()    # quiet
            self.assertEqual(len(calls), 5)
        # summary audit
        snap = json.loads(r2.store[f"{self.PREFIX}/render_summary.json"])
        self.assertTrue(snap["progressive"])
        self.assertTrue(snap["complete"])
        self.assertEqual(snap["frames_done"], 6)
        self.assertEqual(len(snap["frame_batches"]), 5)

    def test_exit0_frame_drop_held_then_abandoned(self):
        calls = []

        def posted_fn(cycle):
            return (("hafsa", "01e", "storm.atm", 0),)

        def render(cycle, out_dir, **kw):
            calls.append(dict(kw))
            # exits 0 but writes a manifest with NO frames for the asked fxx
            _write_out_multi(out_dir, cycle, {"01e": {"hafsa": {}}})
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060418", posted_fn,
                               render, catchup_max_attempts=2)
            r1 = eng.poll_once()["hafs"]
            self.assertEqual(r1.status, pf.PROCESS_FAILED)   # held
            r2_ = eng.poll_once()["hafs"]
            self.assertEqual(r2_.status, pf.PROCESS_FAILED)  # retry, held
            r3 = eng.poll_once()["hafs"]                     # cap -> abandoned
            self.assertEqual(r3.status, pf.CHANGED)
            r4 = eng.poll_once()["hafs"]
            self.assertEqual(r4.status, pf.UNCHANGED)
        self.assertEqual(len(calls), 2)
        snap = json.loads(r2.store[f"{self.PREFIX}/render_summary.json"])
        reasons = [e["reason"] for e in snap["skipped_pairs"]]
        self.assertTrue(any("abandoned" in r for r in reasons))

    def test_render_failure_keeps_prior_batches(self):
        calls = []
        polls = {"n": 0}
        fail = {"left": 1}

        def posted_fn(cycle):
            polls["n"] += 1
            if polls["n"] == 1:
                return (("hafsa", "01e", "storm.atm", 0),)
            return (("hafsa", "01e", "storm.atm", 0),
                    ("hafsa", "01e", "storm.atm", 3))

        good = self._prog_render(calls)

        def render(cycle, out_dir, **kw):
            if kw["only_fxx"] == "3" and fail["left"]:
                fail["left"] -= 1
                calls.append(dict(kw))
                raise hp.RenderError("boom")
            return good(cycle, out_dir, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060418", posted_fn, render)
            eng.poll_once()                                  # f000 ok
            r = eng.poll_once()["hafs"]                      # f003 fails
            self.assertEqual(r.status, pf.PROCESS_FAILED)
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"][0]["storms"][0]["frames"]["hafsa"]
                             ["storm"]["mslp_wind"], [0])    # f000 survived
            eng.poll_once()                                  # f003 retried ok
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"][0]["storms"][0]["frames"]["hafsa"]
                             ["storm"]["mslp_wind"], [0, 3])
        self.assertEqual([c["only_fxx"] for c in calls], ["0", "3", "3"])

    def test_new_cycle_supersedes_partial_keeps_older_complete(self):
        calls = []
        cyc = {"v": "2026060418"}
        posted = {"2026060418": (("hafsa", "01e", "storm.atm", 0),),
                  "2026060500": (("hafsa", "01e", "storm.atm", 0),)}

        def posted_fn(cycle):
            return posted[cycle]

        prev_man = {"cycles": [
            {"cycle": "2026060412", "in_progress": False, "frames_done": 6,
             "frames_expected": 6, "storms": [
                 {"id": "01e", "name": "01E", "frames":
                  {"hafsa": {"storm": {"mslp_wind": [0, 3, 6]}}}}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2(preload={
                f"{self.PREFIX}/manifest.json": json.dumps(prev_man).encode()})
            eng = self._engine(r2, tmp, lambda: cyc["v"], posted_fn,
                               self._prog_render(calls))
            eng.poll_once()            # 18z partial (f000 only, never completes)
            cyc["v"] = "2026060500"
            eng.poll_once()            # 00z supersedes
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            # the PARTIAL 18z is dropped from cycles[]; the older COMPLETE 12z
            # stays as the flip-back target
            self.assertEqual([c["cycle"] for c in man["cycles"]],
                             ["2026060500", "2026060412"])
            self.assertEqual(man["cycle"], "2026060412")     # legacy intact

    def test_bootstrap_recovers_ledger_no_rerender(self):
        calls = []
        live = {"cycles": [
            {"cycle": "2026060418", "in_progress": True, "frames_done": 2,
             "frames_expected": 3, "started_utc": "x", "storms": [
                 {"id": "01e", "name": "01E", "frames":
                  {"hafsa": {"storm": {"mslp_wind": [0, 3]}}}}]},
            {"cycle": "2026060412", "in_progress": False, "frames_done": 6,
             "frames_expected": 6, "storms": [
                 {"id": "01e", "name": "01E", "frames":
                  {"hafsa": {"storm": {"mslp_wind": [0, 3, 6]}}}}]}]}

        def posted_fn(cycle):
            return (("hafsa", "01e", "storm.atm", 0),
                    ("hafsa", "01e", "storm.atm", 3),
                    ("hafsa", "01e", "storm.atm", 6))

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2(preload={
                f"{self.PREFIX}/manifest.json": json.dumps(live).encode()})
            eng = self._engine(r2, tmp, lambda: "2026060418", posted_fn,
                               self._prog_render(calls))
            eng.poll_once()
        # only the MISSING f006 rendered after restart; f000/f003 recovered
        self.assertEqual([c["only_fxx"] for c in calls], ["6"])
        man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
        self.assertFalse(man["cycles"][0]["in_progress"])    # completed too
        self.assertEqual([c["cycle"] for c in man["cycles"]],
                         ["2026060418", "2026060412"])

    def test_completion_prune_cycle_prefixes_and_aged_flat_keys(self):
        calls = []
        old_dt = dt.datetime(2026, 5, 25, 0, 0, tzinfo=dt.timezone.utc)
        fresh_dt = dt.datetime(2026, 6, 4, 11, 0, tzinfo=dt.timezone.utc)

        def posted_fn(cycle):
            return (("hafsa", "01e", "storm.atm", 0),
                    ("hafsa", "01e", "storm.atm", 3),
                    ("hafsa", "01e", "storm.atm", 6))

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2(preload={
                # retired cycle prefix (not in keep set)
                f"{self.PREFIX}/2026060400/hafsa/01e/storm/mslp_wind/f000.png": b"x",
                # flat legacy keys: one ancient, one fresh
                f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f000.png": b"x",
                f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f003.png": b"x",
            }, mtimes={
                f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f000.png": old_dt,
                f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f003.png": fresh_dt,
            })
            eng = self._engine(r2, tmp, lambda: "2026060418", posted_fn,
                               self._prog_render(calls))
            eng.poll_once()    # renders all + completes + prunes
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertFalse(man["cycles"][0]["in_progress"])
        # retired cycle prefix pruned; ancient flat key pruned; fresh flat kept
        self.assertIn(f"{self.PREFIX}/2026060400/hafsa/01e/storm/mslp_wind/f000.png",
                      r2.deleted)
        self.assertIn(f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f000.png",
                      r2.deleted)
        self.assertIn(f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f003.png",
                      r2.store)
        # the active cycle's own frames survive
        self.assertIn(f"{self.PREFIX}/2026060418/hafsa/01e/storm/mslp_wind/f000.png",
                      r2.store)

    def test_v1_manifest_bootstrap_migrates_flat_keys(self):
        """DEPLOY-TRANSITION: bootstrapping from the live v1 manifest must
        server-side-copy its complete cycle's flat frames under the
        cycle-scoped prefix and carry it as the prev/legacy entry - old
        frontends keep a working complete cycle through the first v2 publish."""
        calls = []
        v1 = {"generated_at": "x", "cycle": "2026060412", "fxx_pad": 3,
              "path_template": "{model}/{storm}/{domain}/{product}/f{fxx}.png",
              "storms": [{"id": "01e", "name": "01E", "frames":
                          {"hafsa": {"storm": {"mslp_wind": [0, 3, 6]}}}}]}

        class CopyR2(FakeR2):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.copies = []

            def copy(self, src, dst):
                self.copies.append((src, dst))
                if src in self.store:
                    self.store[dst] = self.store[src]
                    return True
                return False

        def posted_fn(cycle):
            return (("hafsa", "01e", "storm.atm", 0),)

        with tempfile.TemporaryDirectory() as tmp:
            r2 = CopyR2(preload={
                f"{self.PREFIX}/manifest.json": json.dumps(v1).encode(),
                f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f000.png": b"a",
                f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f003.png": b"b",
                f"{self.PREFIX}/hafsa/01e/storm/mslp_wind/f006.png": b"c",
            })
            eng = self._engine(r2, tmp, lambda: "2026060418", posted_fn,
                               self._prog_render(calls))
            eng.poll_once()
        # flat frames copied under the cycle prefix
        self.assertEqual(len(r2.copies), 3)
        self.assertIn(f"{self.PREFIX}/2026060412/hafsa/01e/storm/mslp_wind/"
                      "f000.png", r2.store)
        man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
        # 12z carried as the complete prev entry; legacy fields keep serving it
        self.assertEqual([c["cycle"] for c in man["cycles"]],
                         ["2026060418", "2026060412"])
        self.assertEqual(man["cycle"], "2026060412")
        self.assertTrue(man["path_template"].startswith("2026060412/"))
        self.assertFalse(man["cycles"][1]["in_progress"])

    def test_subgroup_failure_does_not_strand_siblings(self):
        """REVIEW FINDING (critical, confirmed): an early subgroup's failure
        must not strand later subgroups - every subgroup executes every poll,
        and no frame is abandoned without real attempts."""
        calls = []

        def posted_fn(cycle):
            # storm.atm{0} will fail forever; parent.atm{0,3} healthy ->
            # asymmetric sets -> two subgroups; storm's (0,) sorts first
            return (("hafsa", "01e", "parent.atm", 0),
                    ("hafsa", "01e", "parent.atm", 3),
                    ("hafsa", "01e", "storm.atm", 0))

        good = self._prog_render(calls)

        def render(cycle, out_dir, **kw):
            if kw["domains"] == "storm.atm":
                calls.append(dict(kw))
                raise hp.RenderError("storm.atm permanently broken")
            return good(cycle, out_dir, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060418", posted_fn,
                               render, catchup_max_attempts=2)
            r1 = eng.poll_once()["hafs"]
            # the failing subgroup held the signature, BUT the healthy
            # parent.atm subgroup rendered + published in the SAME poll
            self.assertEqual(r1.status, pf.PROCESS_FAILED)
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"][0]["storms"][0]["frames"]["hafsa"]
                             ["parent"]["mslp_wind"], [0, 3])
            eng.poll_once()     # storm.atm attempt 2: fails
            r3 = eng.poll_once()["hafs"]   # cap -> abandoned; clean
            self.assertEqual(r3.status, pf.CHANGED)
        # parent never re-rendered; storm tried exactly twice (real attempts)
        storm_calls = [c for c in calls if c.get("domains") == "storm.atm"]
        parent_calls = [c for c in calls if c.get("domains") == "parent.atm"]
        self.assertEqual(len(storm_calls), 2)
        self.assertEqual(len(parent_calls), 1)
        snap = json.loads(r2.store[f"{self.PREFIX}/render_summary.json"])
        reasons = [e["reason"] for e in snap["skipped_pairs"]]
        self.assertTrue(any("abandoned" in r for r in reasons))
        # parent frames were NEVER falsely abandoned
        self.assertFalse(any("parent" in e["domain"]
                             for e in snap["skipped_pairs"]))

    def test_terminal_giveup_completes_with_gaps(self):
        """REVIEW FINDING (major, confirmed): an abandoned TERMINAL frame must
        not wedge the cycle in_progress forever - completion-with-gaps."""
        calls = []

        def posted_fn(cycle):
            return (("hafsa", "01e", "storm.atm", 0),
                    ("hafsa", "01e", "storm.atm", 3),
                    ("hafsa", "01e", "storm.atm", 6))

        good = self._prog_render(calls)

        def render(cycle, out_dir, **kw):
            if "6" in kw["only_fxx"].split(","):
                # the batch renders everything EXCEPT the terminal frame
                calls.append(dict(kw))
                kept = [f for f in kw["only_fxx"].split(",") if f != "6"]
                if not kept:
                    _write_out_multi(out_dir, cycle, {"01e": {"hafsa": {}}})
                    return {}
                kw2 = dict(kw, only_fxx=",".join(kept))
                return good(cycle, out_dir, **kw2)
            return good(cycle, out_dir, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060418", posted_fn,
                               render, catchup_max_attempts=2)
            for _ in range(5):
                eng.poll_once()
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
        # f006 abandoned after the cap -> the cycle COMPLETES with the gap
        self.assertFalse(man["cycles"][0]["in_progress"])
        self.assertEqual(man["cycles"][0]["storms"][0]["frames"]["hafsa"]
                         ["storm"]["mslp_wind"], [0, 3])
        self.assertEqual(man["cycle"], "2026060418")     # legacy promoted
        snap = json.loads(r2.store[f"{self.PREFIX}/render_summary.json"])
        self.assertTrue(snap["complete"])
        self.assertTrue(any("abandoned" in e["reason"]
                            for e in snap["skipped_pairs"]))

    def test_compose_manifest_v2_zero_blink(self):
        entries = [
            {"cycle": "2026060418", "in_progress": True, "frames_done": 1,
             "frames_expected": 3, "storms": [{"id": "01e", "frames": {}}]},
            {"cycle": "2026060412", "in_progress": False, "frames_done": 6,
             "frames_expected": 6, "storms": [{"id": "01e", "frames": {}}]},
        ]
        man = hp.compose_manifest_v2(entries, FAKE_HEADERS, now_iso="now")
        self.assertEqual(man["cycle"], "2026060412")
        self.assertEqual(man["path_template"],
                         "2026060412/{model}/{storm}/{domain}/{product}/f{fxx}.png")
        self.assertEqual(man["path_template_cycles"],
                         "{cycle}/{model}/{storm}/{domain}/{product}/f{fxx}.png")
        self.assertEqual(man["fxx_end"], 6)    # patched grid
        # no complete cycle at all -> legacy falls to newest WITH frames
        man2 = hp.compose_manifest_v2([entries[0]], FAKE_HEADERS, now_iso="n")
        self.assertEqual(man2["cycle"], "2026060418")
        # nothing at all -> legacy template keeps the un-prefixed token form
        man3 = hp.compose_manifest_v2([], FAKE_HEADERS, now_iso="n")
        self.assertIsNone(man3["cycle"])
        self.assertEqual(man3["path_template"],
                         "{model}/{storm}/{domain}/{product}/f{fxx}.png")

    def test_list_posted_frames_gates_on_sat_and_idx(self):
        import types
        present = {
            # f000: atm grb2+idx AND sat grb2+idx -> ready
            "hfsa/20260604/18/01e.2026060418.hfsa.storm.atm.f000.grb2",
            "hfsa/20260604/18/01e.2026060418.hfsa.storm.atm.f000.grb2.idx",
            "hfsa/20260604/18/01e.2026060418.hfsa.storm.sat.f000.grb2",
            "hfsa/20260604/18/01e.2026060418.hfsa.storm.sat.f000.grb2.idx",
            # f003: atm complete but sat MISSING -> not ready
            "hfsa/20260604/18/01e.2026060418.hfsa.storm.atm.f003.grb2",
            "hfsa/20260604/18/01e.2026060418.hfsa.storm.atm.f003.grb2.idx",
            # f006: atm grb2 present but idx MISSING -> not ready
            "hfsa/20260604/18/01e.2026060418.hfsa.storm.atm.f006.grb2",
            "hfsa/20260604/18/01e.2026060418.hfsa.storm.sat.f006.grb2",
            "hfsa/20260604/18/01e.2026060418.hfsa.storm.sat.f006.grb2.idx",
        }
        stub = types.ModuleType("hafs_render.generate_hafs_plots")
        stub.MODEL_TOKEN = {"hafsa": "hfsa", "hafsb": "hfsb"}
        stub._s3_list = (lambda prefix, delimiter=None, session=None:
                         ([k for k in present if k.startswith(prefix)], []))
        stub.list_storms = lambda model, date, hh, session=None: (
            ["01e"] if model == "hafsa" else [])
        pkg = types.ModuleType("hafs_render")
        pkg.generate_hafs_plots = stub
        saved = {n: sys.modules.get(n)
                 for n in ("hafs_render", "hafs_render.generate_hafs_plots")}
        sys.modules["hafs_render"] = pkg
        sys.modules["hafs_render.generate_hafs_plots"] = stub
        try:
            frames = hp.list_posted_frames("2026060418",
                                           domains="storm.atm")
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m
        self.assertEqual(frames, (("hafsa", "01e", "storm.atm", 0),))



class TestQuiescenceGate(unittest.TestCase):
    """Upstream quiescence -> EMPTY manifest (the 03E/Cristina freeze fix).

    A dead storm's last run sat on /models/ for days because a quiet bucket
    never changed change_key and nothing rewrote the manifest. The gate:
    newest upstream cycle older than HAFS_QUIET_AFTER_H (from a CLEAN
    listing) -> publish the empty manifest the frontend self-hides on,
    exactly once per quiet episode. An outage (listing failure) must NEVER
    blank the manifest."""
    PREFIX = "models/hafs"

    def setUp(self):
        self._saved = (hp.FXX_END, hp.FXX_STEP)
        hp.FXX_END, hp.FXX_STEP = 6, 3

    def tearDown(self):
        hp.FXX_END, hp.FXX_STEP = self._saved

    # The frozen-manifest shape the live bug exhibited: one dead complete
    # cycle still advertised as current.
    DEAD_MAN = {
        "cycles": [{"cycle": "2026053100", "in_progress": False,
                    "frames_done": 6, "frames_expected": 6,
                    "storms": [{"id": "03e", "name": "03E", "frames":
                                {"hafsa": {"storm": {"mslp_wind": [0, 3, 6]}}}}]}],
    }

    def _engine(self, r2, tmp, resolver, posted_fn=lambda c: (), renders=None):
        return hp.build_engine(
            r2, progressive=True, prefix=self.PREFIX, interval_s=1.0,
            clock=FakeClock(), sleep=lambda s: None,
            cycle_resolver=resolver, posted_frames_fn=posted_fn,
            render_fn=renders or (lambda *a, **k: {}),
            headers_fn=lambda: FAKE_HEADERS,
            out_dir_factory=lambda c: str(Path(tmp) / c.replace("/", "_")))

    def test_stale_cycle_publishes_empty_manifest_once(self):
        # Clock starts 2026-06-02; newest upstream run is 2026-05-31 00z
        # (~60 h old) -> quiet. The dead manifest must be replaced by the
        # empty one, and ONLY once (change-gated) across repeated polls.
        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2(preload={
                f"{self.PREFIX}/manifest.json": json.dumps(self.DEAD_MAN).encode()})
            puts = []
            orig_put = r2.put_json
            def counting_put(key, obj, cache):
                if key.endswith("manifest.json"):
                    puts.append(json.loads(json.dumps(obj)))
                return orig_put(key, obj, cache)
            r2.put_json = counting_put

            eng = self._engine(r2, tmp, lambda: "2026053100")
            eng.poll_once()
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"], [])
            self.assertIsNone(man.get("cycle"))
            n_after_first = len(puts)
            self.assertGreaterEqual(n_after_first, 1)

            eng.poll_once()
            eng.poll_once()
            self.assertEqual(len(puts), n_after_first,
                             "quiet polls must not re-publish every cycle")

    def test_no_cycles_at_all_is_also_quiet(self):
        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2(preload={
                f"{self.PREFIX}/manifest.json": json.dumps(self.DEAD_MAN).encode()})
            eng = self._engine(r2, tmp, lambda: None)
            eng.poll_once()
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"], [])

    def test_outage_never_blanks_the_manifest(self):
        def boom():
            raise pf.TransientFetchError("listing outage")
        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2(preload={
                f"{self.PREFIX}/manifest.json": json.dumps(self.DEAD_MAN).encode()})
            eng = self._engine(r2, tmp, boom)
            eng.poll_once()
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"][0]["cycle"], "2026053100",
                             "an outage must keep last-known-good")

    def test_fresh_cycle_recovers_from_quiet(self):
        state = {"cycle": "2026053100"}
        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2(preload={
                f"{self.PREFIX}/manifest.json": json.dumps(self.DEAD_MAN).encode()})
            eng = self._engine(r2, tmp, lambda: state["cycle"])
            eng.poll_once()   # quiet -> empty
            self.assertEqual(json.loads(
                r2.store[f"{self.PREFIX}/manifest.json"])["cycles"], [])
            state["cycle"] = "2026060212"   # NOAA resumes (fresh vs FakeClock)
            eng.poll_once()   # pre-announce
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"][0]["cycle"], "2026060212")
            self.assertTrue(man["cycles"][0]["in_progress"])

    def test_fresh_cycle_within_window_not_quiet(self):
        # ~0 h old vs the 2026-06-02 FakeClock: the gate must not trip.
        with tempfile.TemporaryDirectory() as tmp:
            r2 = FakeR2()
            eng = self._engine(r2, tmp, lambda: "2026060212")
            eng.poll_once()
            man = json.loads(r2.store[f"{self.PREFIX}/manifest.json"])
            self.assertEqual(man["cycles"][0]["cycle"], "2026060212")

    def test_cycle_age_hours(self):
        naive = dt.datetime(2026, 6, 2, 12, 0)
        aware = naive.replace(tzinfo=dt.timezone.utc)
        self.assertAlmostEqual(hp.cycle_age_hours("2026060100", naive), 36.0)
        self.assertAlmostEqual(hp.cycle_age_hours("2026060100", aware), 36.0)
        self.assertIsNone(hp.cycle_age_hours("garbage", naive))

    def test_resolver_raises_when_every_listing_fails(self):
        import sys, types
        fake = types.SimpleNamespace(
            list_dates=lambda model, session=None: (_ for _ in ()).throw(
                RuntimeError("503")),
            list_hours=lambda model, date, session=None: [])
        saved = sys.modules.get("hafs_render.generate_hafs_plots")
        saved_pkg = sys.modules.get("hafs_render")
        sys.modules["hafs_render"] = types.SimpleNamespace(
            generate_hafs_plots=fake)
        sys.modules["hafs_render.generate_hafs_plots"] = fake
        try:
            with self.assertRaises(pf.TransientFetchError):
                hp.resolve_active_cycle("hfsa,hfsb")
        finally:
            for name, mod in (("hafs_render", saved_pkg),
                              ("hafs_render.generate_hafs_plots", saved)):
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod


if __name__ == "__main__":
    unittest.main(verbosity=2)
