"""True-color rides the AHI Target sub-scan so all six Himawari meso bands hit
the ~2.5-min (150 s) cadence instead of leaving true-color on the 10-min FLDK.

Covers the two seams that make that safe:
  - _target_slot_sub: recovers (sub, 10-min slot) from a 'Target<sub>'
    resolution and returns None for FLDK. This is the loader-dispatch fork shared
    by the single-band fetch and the true-color compositor, so a Target red pairs
    with Target green/blue/veggie/IR (never an FLDK mix at a different scan time).
  - find_file: by default visible_red (which pins the composite's product)
    resolves to Target; MESO_TC_FLDK=true reverts JUST true-color to FLDK; every
    non-visible band always prefers Target.

Run from the repo root:  python -m unittest tests.test_meso_truecolor_target
"""

import asyncio
import datetime as dt
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satellites import HimawariPacificSatellite, ResolvedFile  # noqa: E402


def _rf(product, scan_start):
    return ResolvedFile(
        bucket="noaa-himawari9", s3_key="k", product=product,
        scan_start=scan_start, sat_name="Himawari-9", sub_sat_lon=140.7,
    )


class TargetSlotSub(unittest.TestCase):
    def test_target_recovers_sub_and_floored_slot(self):
        # R303 obs time 03:27:30Z (slot 03:20 + (3-1)*150 s) -> sub=3, slot 03:20.
        obs = dt.datetime(2026, 6, 20, 3, 27, 30, tzinfo=dt.timezone.utc)
        out = HimawariPacificSatellite._target_slot_sub(_rf("Target3", obs))
        self.assertIsNotNone(out)
        sub, slot = out
        self.assertEqual(sub, 3)
        self.assertEqual(
            slot, dt.datetime(2026, 6, 20, 3, 20, tzinfo=dt.timezone.utc))

    def test_fldk_returns_none(self):
        obs = dt.datetime(2026, 6, 20, 3, 20, tzinfo=dt.timezone.utc)
        self.assertIsNone(
            HimawariPacificSatellite._target_slot_sub(_rf("FLDK", obs)))

    def test_each_subscan_floors_to_its_slot(self):
        slot0 = dt.datetime(2026, 6, 20, 3, 20, tzinfo=dt.timezone.utc)
        for sub in (1, 2, 3, 4):
            obs = slot0 + dt.timedelta(seconds=(sub - 1) * 150)
            s, slot = HimawariPacificSatellite._target_slot_sub(
                _rf(f"Target{sub}", obs))
            self.assertEqual(s, sub)
            self.assertEqual(slot, slot0)


class FindFileTrueColorProduct(unittest.TestCase):
    def setUp(self):
        self.sat = HimawariPacificSatellite()
        self.when = dt.datetime(2026, 6, 20, 3, 25, tzinfo=dt.timezone.utc)
        self.bbox = [131.9, 9.5, 141.6, 19.0]

    def _find(self, channel):
        return asyncio.run(self.sat.find_file(
            self.when, channel, self.bbox, nearest_to_target=False,
            product_hint="meso"))

    def test_visible_red_uses_target_by_default(self):
        with mock.patch.object(self.sat, "_resolve_target_sync",
                               return_value=_rf("Target3", self.when)) as m:
            rf = self._find("visible_red")
        m.assert_called_once()
        self.assertEqual(rf.product, "Target3")

    def test_visible_red_reverts_to_fldk_when_MESO_TC_FLDK(self):
        with mock.patch.dict(os.environ, {"MESO_TC_FLDK": "true"}):
            with mock.patch.object(self.sat, "_resolve_target_sync") as m:
                rf = self._find("visible_red")
        m.assert_not_called()  # Target path skipped entirely
        self.assertEqual(rf.product, "FLDK")

    def test_non_visible_band_uses_target_even_with_TC_FLDK(self):
        with mock.patch.dict(os.environ, {"MESO_TC_FLDK": "true"}):
            with mock.patch.object(self.sat, "_resolve_target_sync",
                                   return_value=_rf("Target2", self.when)) as m:
                rf = self._find("clean_ir")
        m.assert_called_once()
        self.assertEqual(rf.product, "Target2")

    def test_falls_back_to_fldk_when_no_target_slot(self):
        with mock.patch.object(self.sat, "_resolve_target_sync",
                               return_value=None) as m:
            rf = self._find("clean_ir")
        m.assert_called_once()
        self.assertEqual(rf.product, "FLDK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
