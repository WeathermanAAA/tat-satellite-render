"""heartbeat.sh --sample-from: steal % and NIC KiB/s from two /proc snapshots.

Regression for the 2026-09-02 box1 incident: the fleet heartbeat published
load only, so 35 h under a host-side ingress shaper (eth0 pinned at exactly
1280 KiB/s) and 33+ h under Hostinger's CPU limiter (steal 94-96 %) both
rendered on /fleet/ as "load 16-49", indistinguishable from a busy box.

These tests drive the REAL script's arithmetic (the `--sample-from` hook runs
the same awk the timer uses) against canned /proc/stat + /proc/net/dev pairs,
so the percentages the page shows are pinned here, not re-implemented.

Run: python -m unittest tests.test_heartbeat_sample
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, os.pardir, "scripts", "heartbeat.sh")


def _stat(user, nice, system, idle, iowait, irq, softirq, steal):
    return (f"cpu  {user} {nice} {system} {idle} {iowait} {irq} {softirq} {steal} 0 0\n"
            "cpu0 1 2 3 4 5 6 7 8 0 0\n")


def _dev(rows):
    head = ("Inter-|   Receive                                                |  Transmit\n"
            " face |bytes    packets errs drop fifo frame compressed multicast|bytes    "
            "packets errs drop fifo colls carrier compressed\n")
    body = ""
    for name, rx, tx in rows:
        body += (f"{name:>6}: {rx} 100 0 0 0 0 0 0 {tx} 50 0 0 0 0 0 0\n")
    return head + body


def _run(stat1, stat2, dev1, dev2, iface="eth0", secs="3", missing=()):
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for name, content in (("s1", stat1), ("s2", stat2), ("d1", dev1), ("d2", dev2)):
            p = os.path.join(d, name)
            if name not in missing:
                with open(p, "w") as f:
                    f.write(content)
            paths.append(p)
        out = subprocess.run(["bash", SCRIPT, "--sample-from", *paths, iface, secs],
                             capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        frag = out.stdout.strip()
        return json.loads("{" + frag + "}")


class HeartbeatSample(unittest.TestCase):
    def test_capped_box_reads_as_steal_not_busy(self):
        # 3 s on 8 vCPU at 100 Hz = 2400 jiffies; 2280 stolen, 96 user, 24 sys.
        s1 = _stat(1000, 0, 500, 3000, 10, 0, 5, 700000)
        s2 = _stat(1096, 0, 524, 3000, 10, 0, 5, 702280)
        d1 = _dev([("lo", 10, 10), ("eth0", 1_000_000, 500_000)])
        d2 = _dev([("lo", 20, 20), ("eth0", 1_000_000 + 1280 * 1024 * 3, 500_000 + 100 * 1024 * 3)])
        r = _run(s1, s2, d1, d2)
        self.assertAlmostEqual(r["steal_pct"], 95.0, places=1)
        self.assertAlmostEqual(r["cpu_busy_pct"], 5.0, places=1)
        self.assertAlmostEqual(r["cpu_idle_pct"], 0.0, places=1)
        self.assertEqual(r["net_if"], "eth0")
        self.assertAlmostEqual(r["rx_kib_s"], 1280.0, places=1)   # the shaper constant
        self.assertAlmostEqual(r["tx_kib_s"], 100.0, places=1)
        self.assertEqual(r["sample_s"], 3)

    def test_free_running_box(self):
        s1 = _stat(100, 0, 50, 2000, 20, 0, 0, 3)
        s2 = _stat(1300, 0, 350, 2900, 40, 0, 10, 3)   # 1200 user, 300 sys, 900 idle, 20 iowait, 10 softirq, 0 steal
        d1 = _dev([("eth0", 0, 0)])
        d2 = _dev([("eth0", 6400 * 1024 * 3, 0)])
        r = _run(s1, s2, d1, d2)
        self.assertAlmostEqual(r["steal_pct"], 0.0, places=1)
        self.assertAlmostEqual(r["cpu_busy_pct"], 100 * 1510 / 2430, places=1)
        self.assertAlmostEqual(r["cpu_idle_pct"], 100 * 920 / 2430, places=1)
        self.assertAlmostEqual(r["rx_kib_s"], 6400.0, places=1)
        self.assertAlmostEqual(r["tx_kib_s"], 0.0, places=1)

    def test_unknown_iface_yields_null_net_but_valid_cpu(self):
        s1 = _stat(0, 0, 0, 100, 0, 0, 0, 0)
        s2 = _stat(50, 0, 0, 150, 0, 0, 0, 0)
        d = _dev([("eth0", 0, 0)])
        r = _run(s1, s2, d, d, iface="ens3")
        self.assertAlmostEqual(r["steal_pct"], 0.0, places=1)
        self.assertAlmostEqual(r["cpu_busy_pct"], 50.0, places=1)
        self.assertIsNone(r["net_if"])
        self.assertIsNone(r["rx_kib_s"])
        self.assertIsNone(r["tx_kib_s"])

    def test_empty_iface_and_missing_stat_yield_nulls_not_garbage(self):
        s = _stat(0, 0, 0, 100, 0, 0, 0, 0)
        d = _dev([("eth0", 0, 0)])
        r = _run(s, s, d, d, iface="", missing=("s2",))
        self.assertIsNone(r["steal_pct"])
        self.assertIsNone(r["cpu_busy_pct"])
        self.assertIsNone(r["cpu_idle_pct"])
        self.assertIsNone(r["net_if"])
        self.assertEqual(r["sample_s"], 3)

    def test_counter_wrap_or_reset_clamps_to_zero(self):
        s1 = _stat(500, 0, 0, 500, 0, 0, 0, 0)
        s2 = _stat(600, 0, 0, 600, 0, 0, 0, 0)
        d1 = _dev([("eth0", 9_000_000, 9_000_000)])
        d2 = _dev([("eth0", 1_000, 1_000)])   # counters reset
        r = _run(s1, s2, d1, d2)
        self.assertEqual(r["rx_kib_s"], 0.0)
        self.assertEqual(r["tx_kib_s"], 0.0)

    def test_zero_seconds_gives_null_rates(self):
        s = _stat(0, 0, 0, 100, 0, 0, 0, 0)
        d = _dev([("eth0", 0, 0)])
        r = _run(s, s, d, d, secs="0")
        self.assertIsNone(r["rx_kib_s"])
        self.assertIsNone(r["sample_s"])


if __name__ == "__main__":
    unittest.main()
