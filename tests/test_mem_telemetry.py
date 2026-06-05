"""Memory-telemetry tests (the RAM-minutes instrument).

The Railway bill is dominated by RAM-minutes during HAFS render windows; the
telemetry must (a) measure the whole render TREE (the generator subprocess
plus its pool workers, by process group), (b) ride the existing
frame_batches[]/catchups audit + a cycle-level mem_peak rollup, and (c) expose
the PARENT poller's own residency on every heartbeat so idle holding is
distinguishable from render-time plateaus. These tests pin all three without a
real render: the full run_render_subprocess path runs against a FAKE
hafs_render package injected via PYTHONPATH that allocates a known block and
sleeps long enough to be sampled.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hafs_render_poller as hrp  # noqa: E402
import poller_framework as pf  # noqa: E402


class TestProcTreeSampler(unittest.TestCase):

    def test_sampler_sees_child_process_group(self):
        # A child in its OWN session (same start_new_session the render path
        # uses): the sampler keyed on the child's pgid must count it, and must
        # NOT count this test process (different group).
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(8)"],
            start_new_session=True)
        try:
            time.sleep(0.3)   # let it boot
            total, largest, n = hrp._proc_tree_rss_mb(child.pid)
            self.assertGreaterEqual(n, 1)
            self.assertGreater(total, 1.0)          # a python proc is >1 MB
            self.assertGreater(largest, 1.0)
            self.assertLessEqual(largest, total + 1e-6)
        finally:
            child.kill()
            child.wait()

    def test_sampler_empty_group_returns_zeros(self):
        # A pgid that cannot exist (pid 1's group is init's, but an absurd
        # number is safest): zeros, never an exception.
        total, largest, n = hrp._proc_tree_rss_mb(2 ** 22 + 12345)
        self.assertEqual((total, largest, n), (0.0, 0.0, 0))


class TestRenderSubprocessMem(unittest.TestCase):

    def test_run_render_subprocess_records_tree_peak(self):
        # Full-path: a fake hafs_render.generate_hafs_plots (injected via
        # PYTHONPATH - the render env inherits os.environ) allocates ~80 MB
        # and sleeps past one MEM_SAMPLE_S tick, so the sampler must record a
        # peak well above interpreter baseline AND the summary must carry it.
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "fake" / "hafs_render"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("")
            (pkg / "generate_hafs_plots.py").write_text(textwrap.dedent("""
                import sys, time
                if __name__ == "__main__":
                    block = bytearray(80 * 1024 * 1024)   # ~80 MB resident
                    block[::4096] = b"x" * len(block[::4096])  # touch pages
                    time.sleep(%f)
                    print("rendered 1 ok, 0 failed")
                    sys.exit(0)
            """ % (hrp.MEM_SAMPLE_S * 2 + 1.0)))
            old_pp = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = str(pkg.parent)
            try:
                summary = hrp.run_render_subprocess(
                    "2026060518", str(Path(td) / "out"),
                    timeout_s=60, save_dir=td)
            finally:
                if old_pp is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old_pp
        mem = summary.get("mem")
        self.assertIsNotNone(mem, "summary must carry the mem block")
        self.assertGreaterEqual(mem["samples"], 1)
        self.assertGreater(mem["peak_tree_rss_mb"], 60.0,
                           "80 MB allocation must dominate the tree peak")
        self.assertGreaterEqual(mem["peak_procs"], 1)
        self.assertGreaterEqual(mem["peak_tree_rss_mb"],
                                mem["peak_proc_rss_mb"] - 1e-6)
        # The pre-existing summary fields still work.
        self.assertEqual(summary["render"], {"ok": 1, "failed": 0})


class TestFoldMem(unittest.TestCase):

    def _clock(self):
        import datetime as dt
        return dt.datetime(2026, 6, 5, 18, 0, 0)

    def test_batch_entry_and_cycle_rollup(self):
        summary: dict = {}
        up1 = {"frames": 4, "new_frames": [("hafsa", "01e", "storm", 0)],
               "mem": {"peak_tree_rss_mb": 9000.0, "peak_proc_rss_mb": 2500.0,
                       "peak_procs": 7, "samples": 30}}
        hrp._fold_batch_summary(summary, up1, "hafsa/01e", self._clock)
        b = summary["frame_batches"][-1]
        self.assertEqual(b["tree_rss_mb"], 9000.0)
        self.assertEqual(b["largest_proc_rss_mb"], 2500.0)
        self.assertEqual(b["procs"], 7)
        self.assertEqual(summary["mem_peak"]["tree_rss_mb"], 9000.0)
        self.assertEqual(summary["mem_peak"]["batch"], "hafsa/01e")

        # A LOWER later batch must not displace the cycle peak...
        up2 = {"frames": 2, "new_frames": [("hafsb", "01e", "storm", 3)],
               "mem": {"peak_tree_rss_mb": 4000.0, "peak_proc_rss_mb": 1500.0,
                       "peak_procs": 4, "samples": 10}}
        hrp._fold_batch_summary(summary, up2, "hafsb/01e", self._clock)
        self.assertEqual(summary["mem_peak"]["tree_rss_mb"], 9000.0)
        # ...and a HIGHER one must.
        up3 = {"frames": 2, "new_frames": [("hafsb", "02w", "parent", 6)],
               "mem": {"peak_tree_rss_mb": 13000.0, "peak_proc_rss_mb": 3000.0,
                       "peak_procs": 9, "samples": 12}}
        hrp._fold_batch_summary(summary, up3, "hafsb/02w", self._clock)
        self.assertEqual(summary["mem_peak"]["tree_rss_mb"], 13000.0)
        self.assertEqual(summary["mem_peak"]["batch"], "hafsb/02w")

    def test_fold_without_mem_is_harmless(self):
        # Renders that predate the telemetry (or a failed sample) fold cleanly:
        # entry fields are None, no mem_peak appears.
        summary: dict = {}
        hrp._fold_batch_summary(summary, {"frames": 1, "new_frames": []},
                                "hafsa/01e", self._clock)
        b = summary["frame_batches"][-1]
        self.assertIsNone(b["tree_rss_mb"])
        self.assertNotIn("mem_peak", summary)

    def test_catchup_fold_carries_mem(self):
        summary: dict = {}
        upcov = {"frames": 6, "coverage": [],
                 "mem": {"peak_tree_rss_mb": 7000.0, "peak_proc_rss_mb": 2000.0,
                         "peak_procs": 5, "samples": 20}}
        hrp._fold_catchup_summary(summary, upcov,
                                  [("hafsb", "01e", "storm")], self._clock)
        c = summary["catchups"][-1]
        self.assertEqual(c["tree_rss_mb"], 7000.0)
        self.assertEqual(summary["mem_peak"]["tree_rss_mb"], 7000.0)


class TestParentHeartbeatMem(unittest.TestCase):

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
