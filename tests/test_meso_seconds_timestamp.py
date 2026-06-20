"""Meso plot timestamps carry SECONDS; floater / draw-a-box stay minute-only.

The Himawari Target sector publishes four sub-scans per 10-min slot
(:00 / :02:30 / :05 / :07:30). A minute-precision burned-in valid-time
collapses :02:30->:02 and :07:30->:07, so two adjacent sub-scans can read
the same HH:MM and the 2.5-min cadence is invisible on the plot. The render
endpoint stamps SECONDS only when the meso poller asks (product=="meso");
every other caller (storm floater, legacy draw-a-box) is byte-identical to
before, which is the floater-isolation guarantee.
"""
import datetime as dt
import unittest

from app import _valid_time_label


class ValidTimeLabelTests(unittest.TestCase):
    SCAN = dt.datetime(2026, 6, 20, 21, 37, 30, tzinfo=dt.timezone.utc)

    def test_meso_shows_seconds(self):
        # product=="meso" -> the meso poller path (GOES CMIPM + Himawari Target)
        self.assertEqual(
            _valid_time_label(self.SCAN, "meso"),
            "2026-06-20 21:37:30",
        )

    def test_meso_sub_scans_are_distinguishable(self):
        slot = dt.datetime(2026, 6, 20, 21, 30, 0, tzinfo=dt.timezone.utc)
        labels = [
            _valid_time_label(slot + dt.timedelta(minutes=2.5 * i), "meso")
            for i in range(4)
        ]
        # All four sub-scans of one 10-min slot read distinctly (the bug was
        # :02:30/:07:30 flattening onto the prior minute).
        self.assertEqual(len(set(labels)), 4)
        self.assertTrue(all(s.endswith((":00", ":30")) for s in labels))

    def test_floater_stays_minute_only(self):
        # Storm floater renders pass storm=... and NEVER product -> product is None.
        self.assertEqual(
            _valid_time_label(self.SCAN, None),
            "2026-06-20 21:37",
        )

    def test_draw_a_box_and_fldk_rollback_stay_minute_only(self):
        # Legacy /satellite/ draw-a-box (no product) and the Himawari FLDK
        # rollback (product reset to None) keep the prior minute precision.
        for product in (None, ""):
            self.assertEqual(
                _valid_time_label(self.SCAN, product),
                "2026-06-20 21:37",
            )


if __name__ == "__main__":
    unittest.main()
