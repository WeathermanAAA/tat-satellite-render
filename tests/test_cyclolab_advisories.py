"""CycloLab advisories Source - full offline poll cycles against the
REAL Amanda advisory-13 fixtures (CYCLOLAB_DESIGN.md §9 / Stage 1).

Covers: the happy path writing the §8.3 contract to the shadow prefix,
the advNum change-gate (no rewrite without a new advisory), the
issuance-regression guard (stale mirror rejection), the invest/V1-scope
filter, and the build_engine kill-switch wiring.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import poller_framework as pf  # noqa: E402
import cyclolab_advisories as ca  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "cyclolab"
CURRENT = (FIX / "CurrentStorms.json").read_text()
CONE = (FIX / "EP012026_013adv_CONE.kmz").read_bytes()
TRACK = (FIX / "EP012026_013adv_TRACK.kmz").read_bytes()


def _clock():
    return dt.datetime(2026, 6, 6, 1, 0, 0)


TCP_SHTML = (FIX / "tcp_amanda_current.shtml").read_text()


def make_engine(sink, *, current_text=CURRENT, kmz=None, tcp_text=TCP_SHTML,
                tcd_text="<pre>TCD DISCUSSION BODY</pre>", text_raises=None):
    kmz = kmz or {"CONE": CONE, "TRACK": TRACK}

    def fetch_text(url):
        # URL-routed: the CurrentStorms index, then the two text products
        # (None -> product missing; text_raises matches by substring).
        if text_raises and text_raises in url:
            raise RuntimeError("boom " + url)
        if "TCP" in url:
            return tcp_text
        if "TCD" in url:
            return tcd_text
        return current_text

    def fetch_bytes(url):
        for tag, body in kmz.items():
            if tag in url:
                return body
        return None

    src = ca.make_advisories_source(
        session=None, sink=sink, prefix="shadow/cyclolab",
        current_storms_url="fixture://CurrentStorms.json",
        policy=pf.FetchPolicy(),
        fetch_text=fetch_text, fetch_bytes=fetch_bytes, clock=_clock)
    return pf.PollerEngine([src], name="t", sink=sink, interval_s=1,
                           stale_after_s=60, clock=_clock)


class TestAdvisoriesSource(unittest.TestCase):

    def test_happy_path_writes_contract_to_shadow(self):
        sink = pf.DictSink()
        eng = make_engine(sink)
        eng.poll_once()
        key = "shadow/cyclolab/adv/NHC_EP012026.json"
        self.assertIn(key, sink.store)
        p = sink.store[key]
        if isinstance(p, str):
            p = json.loads(p)
        self.assertEqual(p["sid"], "NHC_EP012026")
        self.assertEqual(p["advisory"], 13)
        self.assertEqual(p["source"], "nhc")
        self.assertEqual(p["method"], "official-cone")
        self.assertGreaterEqual(len(p["cone"]), 1000)
        self.assertEqual(p["cone"][0], p["cone"][-1])  # closed ring
        self.assertEqual([pt["tau_h"] for pt in p["points"]],
                         [0, 12, 24, 36, 48, 60, 72, 96, 120])
        self.assertEqual(p["text"]["tcp_url"],
                         "https://www.nhc.noaa.gov/text/MIATCPEP1.shtml")
        self.assertEqual(p["provenance"]["parsed_utc"],
                         "2026-06-06T01:00:00Z")
        # JSON-serializable end to end
        json.dumps(p)

    def test_text_panels_ship_stripped_products(self):
        # §7.4: the payload carries the PLAIN product text for both
        # panels (the browser cannot fetch nhc.gov cross-origin).
        sink = pf.DictSink()
        make_engine(sink).poll_once()
        p = sink.store["shadow/cyclolab/adv/NHC_EP012026.json"]
        if isinstance(p, str):
            p = json.loads(p)
        self.assertIn("BULLETIN", p["text"]["tcp"])
        self.assertNotIn("<pre", p["text"]["tcp"])
        self.assertNotIn("<html", p["text"]["tcp"].lower())
        self.assertEqual(p["text"]["tcd"], "TCD DISCUSSION BODY")
        self.assertTrue(p["text"]["tcp_url"])  # urls still ride along

    def test_tcd_failure_never_blocks_tcp_cone_or_countdown(self):
        sink = pf.DictSink()
        make_engine(sink, text_raises="TCD").poll_once()
        p = sink.store["shadow/cyclolab/adv/NHC_EP012026.json"]
        if isinstance(p, str):
            p = json.loads(p)
        self.assertIn("tcp", p["text"])         # TCP still shipped
        self.assertNotIn("tcd", p["text"])      # TCD honestly absent
        self.assertGreaterEqual(len(p["cone"]), 1000)
        self.assertIn("next_advisory_utc", p)   # countdown still parsed

    def test_tcp_failure_blocks_neither_cone_nor_tcd(self):
        sink = pf.DictSink()
        make_engine(sink, text_raises="TCP").poll_once()
        p = sink.store["shadow/cyclolab/adv/NHC_EP012026.json"]
        if isinstance(p, str):
            p = json.loads(p)
        self.assertNotIn("tcp", p["text"])
        self.assertEqual(p["text"]["tcd"], "TCD DISCUSSION BODY")
        self.assertGreaterEqual(len(p["cone"]), 1000)
        self.assertNotIn("next_advisory_utc", p)  # countdown source gone

    def test_adv_gate_no_rewrite_on_same_advisory(self):
        sink = pf.DictSink()
        writes = []
        orig = sink.write
        sink.write = lambda k, v: (writes.append(k), orig(k, v))
        eng = make_engine(sink)
        eng.poll_once()
        eng.poll_once()   # same advNum -> change_key unchanged AND ledger gate
        adv_writes = [k for k in writes if "/adv/" in k]
        self.assertEqual(len(adv_writes), 1)

    def test_issuance_regression_rejected(self):
        # A NEW advisory number arrives but parses with an OLDER issuance
        # (the stale-mirror shape). The guard must keep the cached object
        # and record a process failure. One source, one mutable index.
        bumped = CURRENT.replace('"advNum": "013"', '"advNum": "014"')
        doctored = {
            "sid": "NHC_EP012026", "advisory": 14,
            "issued_utc": "2026-06-04T21:00:00Z",   # REGRESSED
            "source": "nhc", "method": "official-cone",
            "cone": [[0.0, 0.0]] * 4 + [[0.0, 0.0]],
            "points": [{"tau_h": 0}, {"tau_h": 12}],
            "text": None, "provenance": {},
        }
        src_sink = pf.DictSink()
        captured = {}

        def fetch_text(url):
            return captured.get("current", CURRENT)

        def fetch_bytes(url):
            return CONE if "CONE" in url else TRACK

        src = ca.make_advisories_source(
            session=None, sink=src_sink, prefix="shadow/cyclolab",
            current_storms_url="fixture://x", policy=pf.FetchPolicy(),
            fetch_text=fetch_text, fetch_bytes=fetch_bytes, clock=_clock)
        eng3 = pf.PollerEngine([src], name="t", sink=src_sink,
                               interval_s=1, stale_after_s=60, clock=_clock)
        eng3.poll_once()    # ledger now: adv 13 @ 06-05T21Z
        captured["current"] = bumped
        with mock.patch.object(ca, "build_advisory_json",
                               return_value=doctored):
            res = eng3.poll_once()
        # The regressed advisory must NOT replace the cached object...
        p = src_sink.store["shadow/cyclolab/adv/NHC_EP012026.json"]
        if isinstance(p, str):
            p = json.loads(p)
        self.assertEqual(p["advisory"], 13)
        # ...and the source must have recorded a process failure (the
        # engine holds the signature and retries).
        snap = eng3.health_snapshot()["sources"]["cyclolab-adv"]
        self.assertGreaterEqual(snap["consecutive_failures"], 1)

    def test_invests_and_unknown_basins_filtered(self):
        cs = json.loads(CURRENT)
        cs["activeStorms"].append(
            {**cs["activeStorms"][0], "id": "ep902026", "binNumber": "EP9"})
        cs["activeStorms"].append(
            {**cs["activeStorms"][0], "id": "io012026"})
        entries = ca._storm_entries(cs)
        self.assertEqual([e["sid"] for e in entries], ["NHC_EP012026"])

    def test_kill_switch_excludes_source(self):
        import intensity_poller as ip
        sink = pf.DictSink()
        eng_off = ip.build_engine(sink, session=mock.Mock(), cyclolab=False)
        self.assertNotIn("cyclolab-adv", [s.name for s in eng_off.sources])
        eng_on = ip.build_engine(sink, session=mock.Mock(), cyclolab=True)
        self.assertIn("cyclolab-adv", [s.name for s in eng_on.sources])


if __name__ == "__main__":
    unittest.main(verbosity=2)
