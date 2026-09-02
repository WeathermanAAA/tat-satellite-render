"""Box-wide catch-up governor in s2_pyramid_emit (2026-09-02 box1 incident).

Pins the three overlap guards offline:
  SLOTS  - fcntl slot files: slot0 is reserved for leads, browse takes shared
           slots only, a held slot is not stolen, a lock dies with its process,
           a browse lane on a 1-slot box skips instead of blocking.
  GATE   - browse passes wait while steal > max, then skip; leads never sample.
  CAP    - S2_MAX_REBUILD_PER_PASS stops the product loop after K emits, newest
           first, and the skipped older slots are reported, not lost.
  STEAL  - the /proc/stat arithmetic (the 95 %-steal box reads as 95 %).

Run: python -m unittest tests.test_s2_governor
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_pyramid_emit as E  # noqa: E402

CPU1 = "cpu  1000 0 500 3000 10 0 5 700000 0 0\n"
CPU2 = "cpu  1096 0 524 3000 10 0 5 702280 0 0\n"     # 2400 jiffies, 2280 stolen


class Steal(unittest.TestCase):
    def test_capped_box_reads_95(self):
        self.assertAlmostEqual(E.steal_pct_from(CPU1, CPU2), 95.0, places=1)

    def test_unreadable_is_none(self):
        self.assertIsNone(E.steal_pct_from("garbage", CPU2))
        self.assertIsNone(E.steal_pct_from(CPU1, CPU1))          # no elapsed time
        self.assertIsNone(E.read_cpu_steal_pct(0, stat_path="/nonexistent/stat"))

    def test_read_from_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "stat")
            with open(p, "w") as f:
                f.write(CPU1)

            def sleeper(_):                      # the second read sees CPU2
                with open(p, "w") as f:
                    f.write(CPU2)
            self.assertAlmostEqual(E.read_cpu_steal_pct(1, stat_path=p, sleeper=sleeper), 95.0, places=1)


class Gate(unittest.TestCase):
    def test_leads_never_sample(self):
        sampler = mock.Mock(return_value=99.0)
        self.assertTrue(E.gate_wait("leads", 50, 60, 900, sampler=sampler, sleeper=lambda s: None, log=lambda m: None))
        sampler.assert_not_called()

    def test_off_when_max_is_zero(self):
        sampler = mock.Mock(return_value=99.0)
        self.assertTrue(E.gate_wait("browse", 0, 60, 900, sampler=sampler, sleeper=lambda s: None, log=lambda m: None))
        sampler.assert_not_called()

    def test_browse_waits_then_starts(self):
        sampler = mock.Mock(side_effect=[90.0, 80.0, 10.0])
        slept = []
        ok = E.gate_wait("browse", 50, 60, 900, sampler=sampler, sleeper=slept.append, log=lambda m: None)
        self.assertTrue(ok)
        self.assertEqual(slept, [60, 60])

    def test_browse_skips_after_max_wait(self):
        sampler = mock.Mock(return_value=95.0)
        slept, logs = [], []
        ok = E.gate_wait("browse", 50, 60, 120, sampler=sampler, sleeper=slept.append, log=logs.append)
        self.assertFalse(ok)
        self.assertEqual(slept, [60, 60])                # 0 -> 60 -> 120 >= max
        self.assertTrue(any("skipping this pass" in m for m in logs))

    def test_unreadable_sample_is_clear(self):
        self.assertTrue(E.gate_wait("browse", 50, 60, 900, sampler=lambda: None, sleeper=lambda s: None, log=lambda m: None))


class Slots(unittest.TestCase):
    def test_order(self):
        self.assertEqual(E.slot_order(2, "leads"), [0, 1])
        self.assertEqual(E.slot_order(2, "browse"), [1])
        self.assertEqual(E.slot_order(1, "leads"), [0])
        self.assertEqual(E.slot_order(1, "browse"), [])
        self.assertEqual(E.slot_order(0, "leads"), [])

    def test_leads_take_slot0_browse_never(self):
        with tempfile.TemporaryDirectory() as d:
            a = E.acquire_slot(d, 2, "leads", 1, sleeper=lambda s: None, log=lambda m: None)
            self.assertTrue(a.path.endswith("slot0"))
            b = E.acquire_slot(d, 2, "browse", 1, sleeper=lambda s: None, log=lambda m: None)
            self.assertTrue(b.path.endswith("slot1"))
            # both busy: a second leads waits for either, then gives up
            t = [0.0]
            def clock(): return t[0]
            def sleeper(s): t[0] += s
            c = E.acquire_slot(d, 2, "leads", 12, poll_s=5, sleeper=sleeper, clock=clock, log=lambda m: None)
            self.assertIsNone(c)
            b.release()
            c = E.acquire_slot(d, 2, "leads", 1, sleeper=lambda s: None, log=lambda m: None)
            self.assertTrue(c.path.endswith("slot1"))    # slot0 still held by `a`
            a.release(); c.release()

    def test_browse_on_one_slot_box_skips(self):
        with tempfile.TemporaryDirectory() as d:
            logs = []
            self.assertIsNone(E.acquire_slot(d, 1, "browse", 30, sleeper=lambda s: None, log=logs.append))
            self.assertTrue(any("no slot" in m for m in logs))

    def test_lock_dies_with_its_process(self):
        with tempfile.TemporaryDirectory() as d:
            holder = subprocess.Popen(
                [sys.executable, "-c",
                 "import fcntl,sys,time,os; p=os.path.join(sys.argv[1],'slot1'); f=open(p,'a+');"
                 " fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB); print('held', flush=True); time.sleep(30)", d],
                stdout=subprocess.PIPE, text=True)
            self.assertEqual(holder.stdout.readline().strip(), "held")
            try:
                self.assertIsNone(E.acquire_slot(d, 2, "browse", 0.3, poll_s=0.1, log=lambda m: None))
                holder.kill(); holder.wait()
                lease = E.acquire_slot(d, 2, "browse", 1, poll_s=0.1, log=lambda m: None)
                self.assertIsNotNone(lease)
                lease.release()
            finally:
                if holder.poll() is None:
                    holder.kill(); holder.wait()
                holder.stdout.close()


class Admit(unittest.TestCase):
    def tearDown(self):
        if E._GOV_LEASE is not None:
            E._GOV_LEASE.release()
            E._GOV_LEASE = None

    def test_env_off_is_noop(self):
        with mock.patch.dict(os.environ, {"S2_SLOTS": "", "S2_GATE_STEAL_MAX": ""}, clear=False):
            self.assertTrue(E.governor_admit(log=lambda m: None))
            self.assertIsNone(E._GOV_LEASE)

    def test_env_slots_and_gate(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {
                "S2_SLOTS": "2", "S2_SLOT_DIR": d, "S2_SLOT_CLASS": "leads",
                "S2_GATE_STEAL_MAX": "50", "S2_SLOT_WAIT_S": "1"}, clear=False):
            with mock.patch.object(E, "read_cpu_steal_pct", return_value=99.0):
                self.assertTrue(E.governor_admit(log=lambda m: None))   # leads exempt from gate
                self.assertTrue(E._GOV_LEASE.path.endswith("slot0"))
                E._GOV_LEASE.release(); E._GOV_LEASE = None
            os.environ["S2_SLOT_CLASS"] = "browse"
            os.environ["S2_GATE_SLEEP_S"] = "0"; os.environ["S2_GATE_MAX_WAIT_S"] = "0"
            with mock.patch.object(E, "read_cpu_steal_pct", return_value=99.0):
                self.assertFalse(E.governor_admit(log=lambda m: None))  # browse gated out

    def test_bad_class_treated_as_browse(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {
                "S2_SLOTS": "1", "S2_SLOT_DIR": d, "S2_SLOT_CLASS": "vip", "S2_SLOT_WAIT_S": "0.1"}, clear=False):
            self.assertFalse(E.governor_admit(log=lambda m: None))      # browse has no slot on a 1-slot box


class RebuildCap(unittest.TestCase):
    """End to end through main(): fake emit_one, empty coverage, local store."""

    def _run(self, cap, backfill_min=60):
        calls = []

        def fake_emit_one(entry, when, store, args, **kw):
            calls.append(when)
            return {"stamp": when.strftime("%Y%m%dT%H%M%SZ"), "manifest": {}}
        env = {"S2_MAX_REBUILD_PER_PASS": str(cap), "S2_SLOTS": "", "S2_GATE_STEAL_MAX": ""}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(E, "emit_one", fake_emit_one), \
             mock.patch.object(E, "_pass_coverage", lambda entries, *a, **k: [[] for _ in entries]), \
             mock.patch.object(E, "_probe_stale_pin", lambda *a, **k: None), \
             mock.patch.object(E, "_write_products_index", lambda *a, **k: None), \
             mock.patch.object(E, "_print_ops", lambda *a, **k: None):
            rc = E.main(["--product", "goes19-conus-ir", "--store", f"local:{d}",
                         "--step", "10", "--backfill", str(backfill_min)])
        return rc, calls

    def test_cap_limits_emits_newest_first(self):
        rc, calls = self._run(2)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[0], calls[1])            # newest first

    def test_zero_means_unlimited(self):
        rc, calls = self._run(0)
        self.assertEqual(rc, 0)
        self.assertGreaterEqual(len(calls), 6)             # the whole 60-min window at 10-min step


if __name__ == "__main__":
    unittest.main()
