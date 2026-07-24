#!/usr/bin/env python3
"""--step/--backfill slot-backfill tests for s2_pyramid_emit: the slot grid,
per-entry coverage detection, skip-of-covered-slots, and the single-product
backfill loop's self-heal re-check. Pure logic + mocked emit -- no network."""
import datetime as dt
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2_pyramid_emit as E  # noqa: E402

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 7, 12, 3, 7, 42, tzinfo=UTC)


def _stamp(t):
    return t.strftime("%Y%m%dT%H%M%SZ")


class TestSlotGrid(unittest.TestCase):
    def test_slots_snap_to_step_grid_newest_first(self):
        # NEWEST first (the enscenters/§11-H lesson in _backfill_slots'
        # docstring): the current slot renders before old holes heal, so
        # freshness never waits on backfill depth
        slots = E._backfill_slots(10, 60, now=NOW)
        self.assertEqual(slots[0], dt.datetime(2026, 7, 12, 3, 0, tzinfo=UTC))
        self.assertEqual(slots[-1], dt.datetime(2026, 7, 12, 2, 10, tzinfo=UTC))
        self.assertTrue(all(s.minute % 10 == 0 and s.second == 0 for s in slots))
        self.assertEqual(slots, sorted(slots, reverse=True))
        self.assertEqual(len(slots), 6)

    def test_step_five_grid(self):
        # now 03:07:42, window 20m -> floor 02:47:42: 03:05..02:50 = 4 slots
        slots = E._backfill_slots(5, 20, now=NOW)
        self.assertEqual(slots[0], dt.datetime(2026, 7, 12, 3, 5, tzinfo=UTC))
        self.assertEqual(slots[-1], dt.datetime(2026, 7, 12, 2, 50, tzinfo=UTC))
        self.assertEqual(len(slots), 4)


class TestCoverage(unittest.TestCase):
    def test_slot_covered_within_half_step(self):
        tol = dt.timedelta(minutes=5)
        slot = dt.datetime(2026, 7, 12, 2, 50, tzinfo=UTC)
        near = [slot + dt.timedelta(seconds=77)]     # CONUS scan :51:17 vs :50
        far = [slot + dt.timedelta(minutes=6)]
        self.assertFalse(E._slot_missing(slot, [near], tol))
        self.assertTrue(E._slot_missing(slot, [far], tol))

    def test_suite_slot_open_until_every_entry_covered(self):
        tol = dt.timedelta(minutes=5)
        slot = dt.datetime(2026, 7, 12, 2, 50, tzinfo=UTC)
        a = [slot]                                    # product A has the frame
        b = []                                        # product B lags
        self.assertTrue(E._slot_missing(slot, [a, b], tol))
        self.assertFalse(E._slot_missing(slot, [a, [slot]], tol))

    def test_legacy_stamp_shapes_ignored(self):
        entry = mock.Mock()
        store = mock.Mock()
        with mock.patch.object(E.P, "complete_stamps",
                               return_value=[("garbage", 5),
                                             (_stamp(NOW), 5)]):
            out = E._covered_times([entry], store, "shadow")
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)


class TestBackfillLoop(unittest.TestCase):
    """Single-product --step path: emits ONLY missing slots, NEWEST first
    (freshness before hole-healing), re-checks coverage as frames land
    (adjacent slots resolving to the same scan dedup to one render),
    isolates per-slot failures."""

    def _run(self, covered_times, emit_effect=None, argv_extra=()):
        emitted = []

        def fake_emit(entry, when, store, args, band_cache=None):
            emitted.append(when)
            if emit_effect:
                emit_effect(when)
            return {"count": 1, "latest": _stamp(when), "minzoom": 0, "maxzoom": 5}

        cov = {"times": list(covered_times)}

        def fake_complete(entry, store, prefix):
            return [(_stamp(t), 5) for t in cov["times"]]

        entry = next(e for e in E.R.REGISTRY
                     if e.tiled and e.sector_key == "conus")
        fixed_now = NOW

        with mock.patch.object(E, "emit_one", side_effect=fake_emit), \
             mock.patch.object(E.P, "complete_stamps", side_effect=fake_complete), \
             mock.patch.object(E, "_write_products_index"), \
             mock.patch.object(E.dt, "datetime", wraps=dt.datetime) as md:
            md.now.return_value = fixed_now
            rc = E.main(["--product", entry.product_id,
                         "--store", "local:/tmp/nope",
                         "--step", "10", "--backfill", "30"]
                        + list(argv_extra))
        return rc, emitted, cov

    def test_emits_only_missing_slots_newest_first(self):
        # window (30m from 03:07:42): slots 03:00, 02:50, 02:40; only the
        # newest is covered -> the two older emit, newest of the holes first
        newest = dt.datetime(2026, 7, 12, 3, 0, tzinfo=UTC)
        rc, emitted, _ = self._run([newest])
        self.assertEqual(rc, 0)
        self.assertEqual(emitted, [dt.datetime(2026, 7, 12, 2, 50, tzinfo=UTC),
                                   dt.datetime(2026, 7, 12, 2, 40, tzinfo=UTC)])

    def test_fully_covered_window_emits_nothing(self):
        newest = dt.datetime(2026, 7, 12, 3, 0, tzinfo=UTC)
        covered = [newest - dt.timedelta(minutes=10 * i) for i in range(4)]
        rc, emitted, _ = self._run(covered)
        self.assertEqual(rc, 0)
        self.assertEqual(emitted, [])

    def test_recheck_skips_slot_covered_by_prior_emit(self):
        # the first emit lands a scan that ALSO covers the next slot (nearest-
        # scan collapse near native cadence): the loop must skip it, not
        # re-render -- the store re-check between slots is the self-heal.
        newest = dt.datetime(2026, 7, 12, 3, 0, tzinfo=UTC)
        cov_state = {"times": [newest]}   # 02:40 AND 02:50 both start missing
        emitted = []

        def fake_emit(entry, when, store, args, band_cache=None):
            emitted.append(when)
            # the rendered scan sits between the two missing slots (2:45)
            cov_state["times"].append(dt.datetime(2026, 7, 12, 2, 45, tzinfo=UTC))
            return {"count": 1}

        def fake_complete(entry, store, prefix):
            return [(_stamp(t), 5) for t in cov_state["times"]]

        entry = next(e for e in E.R.REGISTRY
                     if e.tiled and e.sector_key == "conus")
        with mock.patch.object(E, "emit_one", side_effect=fake_emit), \
             mock.patch.object(E.P, "complete_stamps", side_effect=fake_complete), \
             mock.patch.object(E, "_write_products_index"), \
             mock.patch.object(E.dt, "datetime", wraps=dt.datetime) as md:
            md.now.return_value = NOW
            rc = E.main(["--product", entry.product_id,
                         "--store", "local:/tmp/nope",
                         "--step", "10", "--backfill", "30"])
        # 2:45 covers BOTH 2:50 and 2:40 (tol 5m): exactly one render --
        # newest-first means the 02:50 hole renders and 02:40 self-heals
        self.assertEqual(rc, 0)
        self.assertEqual(emitted, [dt.datetime(2026, 7, 12, 2, 50, tzinfo=UTC)])

    def test_step_and_time_mutually_exclusive(self):
        entry = next(e for e in E.R.REGISTRY if e.tiled)
        with self.assertRaises(SystemExit):
            E.main(["--product", entry.product_id, "--store", "local:/x",
                    "--step", "10", "--time", "2026-07-12T02:00:00"])


if __name__ == "__main__":
    unittest.main()
