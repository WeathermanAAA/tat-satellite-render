"""CycloLabPageWriter lifecycle (CYCLOLAB_DESIGN.md §3.4): birth /
refresh / debounced ENDED freeze, invest skip, best-effort contract."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyclolab_pages import CycloLabPageWriter, ENDED_DEBOUNCE_POLLS  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "cyclolab"
SYNTH = json.loads((FIX / "synth_storm.json").read_text())


class HtmlSink:
    def __init__(self):
        self.writes = []   # (key, html)

    def write_html(self, key, html, cache="public, max-age=30"):
        self.writes.append((key, html))


def feed(storms, basin="ep"):
    return {"basin": basin, "storms": storms}


class TestPageLifecycle(unittest.TestCase):

    def setUp(self):
        self.sink = HtmlSink()
        self.w = CycloLabPageWriter(self.sink, prefix="shadow/cyclolab")

    def test_birth_writes_live_page(self):
        self.w.update("ep", feed([SYNTH]))
        self.assertEqual(len(self.sink.writes), 1)
        key, html = self.sink.writes[0]
        self.assertEqual(key, "shadow/cyclolab/NHC_EP082026/index.html")
        self.assertIn('data-cat="C4"', html)
        self.assertNotIn("data-ended", html[html.index("<html"):html.index("<head")])
        # shadow pages hydrate the adv JSON from the absolute cdn URL
        self.assertIn(
            "https://cdn.triple-a-tropics.com/shadow/cyclolab/adv/"
            "NHC_EP082026.json", html)

    def test_no_rewrite_without_change(self):
        self.w.update("ep", feed([SYNTH]))
        self.w.update("ep", feed([SYNTH]))   # identical fix + cat
        self.assertEqual(len(self.sink.writes), 1)

    def test_refresh_on_new_fix_and_category(self):
        self.w.update("ep", feed([SYNTH]))
        bumped = {**SYNTH, "latest_fix_valid_utc": "2026-06-04T06:00:00Z",
                  "current_category": "C5"}
        self.w.update("ep", feed([bumped]))
        self.assertEqual(len(self.sink.writes), 2)
        self.assertIn('data-cat="C5"', self.sink.writes[1][1])

    def test_ended_after_debounce_only(self):
        self.w.update("ep", feed([SYNTH]))
        for i in range(ENDED_DEBOUNCE_POLLS - 1):
            self.w.update("ep", feed([]))    # storm gone, not yet frozen
        self.assertEqual(len(self.sink.writes), 1)
        self.w.update("ep", feed([]))        # debounce reached -> freeze
        self.assertEqual(len(self.sink.writes), 2)
        key, html = self.sink.writes[1]
        self.assertEqual(key, "shadow/cyclolab/NHC_EP082026/index.html")
        self.assertIn("data-ended", html[html.index("<html"):html.index("<head")])
        self.assertIn("THIS STORM HAS ENDED", html)
        # frozen once: further empty polls write nothing
        self.w.update("ep", feed([]))
        self.assertEqual(len(self.sink.writes), 2)

    def test_transient_absence_resets_debounce(self):
        self.w.update("ep", feed([SYNTH]))
        self.w.update("ep", feed([]))                       # blip
        self.w.update("ep", feed([SYNTH]))                  # back -> reset
        self.w.update("ep", feed([]))                       # 1 of 2 again
        self.assertEqual(len(self.sink.writes), 1)          # still live

    def test_rebirth_after_ended(self):
        self.w.update("ep", feed([SYNTH]))
        for _ in range(ENDED_DEBOUNCE_POLLS):
            self.w.update("ep", feed([]))
        self.w.update("ep", feed([SYNTH]))   # regenerated/redesignated
        self.assertEqual(len(self.sink.writes), 3)
        h3 = self.sink.writes[2][1]
        self.assertNotIn("data-ended", h3[h3.index("<html"):h3.index("<head")])

    def test_invests_and_inactive_skipped(self):
        invest = {**SYNTH, "sid": "NHC_EP902026"}
        inactive = {**SYNTH, "sid": "NHC_EP072026", "is_active": False}
        self.w.update("ep", feed([invest, inactive]))
        self.assertEqual(self.sink.writes, [])

    def test_other_basin_sweep_isolation(self):
        # An EP storm must not be aged out by a WP feed's update cycle.
        self.w.update("ep", feed([SYNTH]))
        for _ in range(ENDED_DEBOUNCE_POLLS + 1):
            self.w.update("wp", feed([], basin="wp"))
        self.assertEqual(len(self.sink.writes), 1)   # never frozen

    def test_best_effort_never_raises(self):
        class Boom:
            def write_html(self, *a, **k):
                raise RuntimeError("r2 down")
        w = CycloLabPageWriter(Boom(), prefix="shadow/cyclolab")
        w.update("ep", feed([SYNTH]))    # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
