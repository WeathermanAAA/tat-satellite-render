"""The intensity poller keeps the formation-chance pill FRESH.

A PTC's pill reads its spawning invest's formation.json (the NHC TWO odds live
under the invest, e.g. "(AL90)"). The guidance poller only refreshes invests
still in the ACTIVE feed, so once the 90L->01L handoff drops the invest its
formation.json freezes (observed: 60% stale while the live TWO read 70%). The
always-on intensity poller now refreshes formation.json for EVERY TWO-referenced
invest every poll, independent of the active feed.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import poller_framework as pf      # noqa: E402
import intensity_poller as ip      # noqa: E402

# A real-shaped NHC TWOAT with the Gulf area still tagged (AL90) at 70/70.
TWO_AL_70 = (
    "<rss><channel><item><description><![CDATA[000<br />TWOAT<br /><br />"
    "Tropical Weather Outlook<br /><br />"
    "Northwestern Gulf of America (AL90):<br />"
    "A broad area of low pressure is producing disorganized showers...<br />"
    "* Formation chance through 48 hours...high...70 percent.<br />"
    "* Formation chance through 7 days...high...70 percent.<br /><br />$$<br />"
    "]]></description></item></channel></rss>")


class _FakeResp:
    status_code = 200

    def __init__(self, text):
        self.text = text


class _FakeSession:
    def __init__(self, text):
        self._text = text
        self.calls = 0

    def get(self, url, **kw):
        self.calls += 1
        return _FakeResp(self._text)


class FormationRefreshTests(unittest.TestCase):
    def setUp(self):
        ip._two_cache.clear()           # the module-level TTL cache

    def _formation(self, sink, sid):
        p = sink.store.get(f"cyclolab/{sid}/formation.json")
        if isinstance(p, str):
            p = json.loads(p)
        return p

    def test_refreshes_formation_for_tw0_invest(self):
        sink = pf.DictSink()
        sess = _FakeSession(TWO_AL_70)
        ip._refresh_formation(sink, sess, "al", dt.datetime(2026, 6, 16, 17, 30))
        f = self._formation(sink, "NHC_AL902026")
        self.assertIsNotNone(f, "formation.json not written for the TWO invest")
        self.assertEqual(f["p48"], 70)
        self.assertEqual(f["p7"], 70)
        self.assertEqual(f["level"], "high")
        self.assertEqual(f["source"], "nhc-two")
        self.assertTrue(f["generated_at"].endswith("Z"))
        # the key matches what the PTC page reads (spawn_sid fallback).
        self.assertIn("cyclolab/NHC_AL902026/formation.json", sink.store)

    def test_independent_of_active_feed(self):
        # The whole point: it writes even though NO storm/invest list is passed
        # in (the dropped-from-feed case). The TWO reference alone drives it.
        sink = pf.DictSink()
        ip._refresh_formation(sink, _FakeSession(TWO_AL_70), "al",
                              dt.datetime(2026, 6, 16, 17, 30))
        self.assertEqual(self._formation(sink, "NHC_AL902026")["p7"], 70)

    def test_jtwc_basin_writes_nothing(self):
        # NHC TWO only (AL/EP/CP). A JTWC/WP basin must never fabricate odds.
        sink = pf.DictSink()
        sess = _FakeSession(TWO_AL_70)
        ip._refresh_formation(sink, sess, "wp", dt.datetime(2026, 6, 16, 17, 30))
        self.assertEqual(sink.store, {})
        self.assertEqual(sess.calls, 0, "must not even fetch the TWO for WP")

    def test_ttl_cache_avoids_refetch(self):
        sess = _FakeSession(TWO_AL_70)
        ip._refresh_formation(pf.DictSink(), sess, "al",
                              dt.datetime(2026, 6, 16, 17, 30))
        ip._refresh_formation(pf.DictSink(), sess, "al",
                              dt.datetime(2026, 6, 16, 17, 31))
        self.assertEqual(sess.calls, 1, "TWO should be TTL-cached, not refetched")

    def test_empty_two_writes_nothing_keeps_existing(self):
        # A transient/quiet TWO ({}) must not blank an existing pill.
        sink = pf.DictSink()
        ip._refresh_formation(sink, _FakeSession("<rss></rss>"), "al",
                              dt.datetime(2026, 6, 16, 17, 30))
        self.assertEqual(sink.store, {})


if __name__ == "__main__":
    unittest.main()
