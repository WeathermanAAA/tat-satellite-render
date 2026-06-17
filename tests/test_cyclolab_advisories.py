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


# The CAPTURED fixture pair is itself out of sync: the TCP page had
# rolled to advisory 14 while CurrentStorms/cone were still at 13 - the
# exact rolling-URL hazard the §7.4 verification now rejects. The
# matched adv-13 variant drives the happy paths; the captured original
# drives the rolled-ahead rejection test.
TCP_SHTML_14 = (FIX / "tcp_amanda_current.shtml").read_text()
TCP_SHTML = (FIX / "tcp_amanda_adv13.shtml").read_text()
TCD_BODY = ("<pre>Tropical Storm Amanda Discussion Number  13\n\n"
            "TCD DISCUSSION BODY</pre>")


def make_engine(sink, *, current_text=CURRENT, kmz=None, tcp_text=TCP_SHTML,
                tcd_text=TCD_BODY, text_raises=None):
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
        self.assertIn("TCD DISCUSSION BODY", p["text"]["tcd"])
        self.assertIn("Discussion Number", p["text"]["tcd"])
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
        self.assertIn("TCD DISCUSSION BODY", p["text"]["tcd"])
        self.assertIn("Discussion Number", p["text"]["tcd"])
        self.assertGreaterEqual(len(p["cone"]), 1000)
        self.assertNotIn("next_advisory_utc", p)  # countdown source gone

    def test_og_card_written_when_sink_supports_png(self):
        class PngSink(pf.DictSink):
            def __init__(self):
                super().__init__()
                self.pngs = {}
            def write_png(self, key, data, cache=None):
                self.pngs[key] = data
        sink = PngSink()
        make_engine(sink).poll_once()
        key = "shadow/cyclolab/og/NHC_EP012026.png"
        self.assertIn(key, sink.pngs)
        self.assertEqual(sink.pngs[key][:4], b"\x89PNG")
        self.assertGreater(len(sink.pngs[key]), 10000)

    def test_og_card_skipped_gracefully_without_binary_sink(self):
        sink = pf.DictSink()           # no write_png
        make_engine(sink).poll_once()
        self.assertIn("shadow/cyclolab/adv/NHC_EP012026.json", sink.store)
        self.assertFalse([k for k in sink.store if k.endswith(".png")])

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


class TestTextHealAndVerification(unittest.TestCase):
    """Final-gate-3 #4 (the third strike): text products that fail or
    lag at advisory-processing time must HEAL on later polls instead of
    staying blank for the whole 3-6 h advisory cycle, and a rolling
    .shtml page serving the WRONG advisory's text must never attach."""

    def _engine_with_window(self, sink):
        """Engine whose text products 503 until the window 'opens' -
        the exact at-issuance mid-roll / transient-failure shape."""
        avail = {"on": False}

        def fetch_text(url):
            if "MIATC" in url:
                if not avail["on"]:
                    raise RuntimeError("503 mid-roll " + url)
                return TCP_SHTML if "TCP" in url else TCD_BODY
            return CURRENT

        def fetch_bytes(url):
            return CONE if "CONE" in url else (
                TRACK if "TRACK" in url else None)

        src = ca.make_advisories_source(
            session=None, sink=sink, prefix="shadow/cyclolab",
            current_storms_url="fixture://CurrentStorms.json",
            policy=pf.FetchPolicy(), fetch_text=fetch_text,
            fetch_bytes=fetch_bytes, clock=_clock)
        eng = pf.PollerEngine([src], name="t", sink=sink, interval_s=1,
                              stale_after_s=60, clock=_clock)
        return eng, avail

    def _payload(self, sink):
        p = sink.store["shadow/cyclolab/adv/NHC_EP012026.json"]
        return json.loads(p) if isinstance(p, str) else p

    def test_text_lag_heals_on_a_later_poll(self):
        sink = pf.DictSink()
        eng, avail = self._engine_with_window(sink)
        eng.poll_once()                       # adv ships text-less...
        p = self._payload(sink)
        self.assertNotIn("tcp", p["text"])
        self.assertNotIn("tcd", p["text"])
        self.assertGreaterEqual(len(p["cone"]), 1000)  # cone never blocked
        avail["on"] = True                    # window over: pages posted
        eng.poll_once()                       # ...and HEALS in place
        p = self._payload(sink)
        self.assertIn("BULLETIN", p["text"]["tcp"])
        self.assertIn("TCD DISCUSSION BODY", p["text"]["tcd"])
        self.assertEqual(p["provenance"]["text_healed_utc"],
                         "2026-06-06T01:00:00Z")
        # countdown healed along with the TCP
        self.assertIn("next_advisory_utc", p)

    def test_text_urls_absent_at_first_advisory_keep_heal_open(self):
        # NEGATIVE CONTROL (the PTC first-advisory blank-panel bug): at a
        # storm's FIRST advisory CurrentStorms can carry the cone/track KMZ a
        # poll or two BEFORE it populates the publicAdvisory/forecastDiscussion
        # URLs. TCP+TCD are ALWAYS-EXPECTED for an NHC designated storm, so the
        # heal debt must stay OPEN (text_done False) and the text must attach as
        # soon as the URLs appear. Under the OLD vacuous-True behavior the pulse
        # stopped and the panel stayed blank all cycle - and the poll-2 heal
        # assertion below fails. This test guards the regression.
        sink = pf.DictSink()
        full = json.loads(CURRENT)
        nourl = json.loads(CURRENT)
        st = nourl["activeStorms"][0]
        st["publicAdvisory"] = {"advNum": "013"}     # URLs not yet populated
        st["forecastDiscussion"] = {"advNum": "013"}
        state = {"current": json.dumps(nourl)}

        def fetch_text(url):
            if "TCP" in url:
                return TCP_SHTML
            if "TCD" in url:
                return TCD_BODY
            return state["current"]

        def fetch_bytes(url):
            return CONE if "CONE" in url else (
                TRACK if "TRACK" in url else None)

        src = ca.make_advisories_source(
            session=None, sink=sink, prefix="shadow/cyclolab",
            current_storms_url="fixture://CurrentStorms.json",
            policy=pf.FetchPolicy(), fetch_text=fetch_text,
            fetch_bytes=fetch_bytes, clock=_clock)
        eng = pf.PollerEngine([src], name="t", sink=sink, interval_s=1,
                              stale_after_s=60, clock=_clock)

        eng.poll_once()                          # adv ships; text URLs absent
        p = self._payload(sink)
        self.assertNotIn("tcp", p["text"])       # nothing attached yet...
        self.assertNotIn("tcd", p["text"])
        self.assertGreaterEqual(len(p["cone"]), 1000)   # ...cone never blocked

        state["current"] = json.dumps(full)      # NHC populates the text URLs
        eng.poll_once()                          # heal pulse must still fire
        p = self._payload(sink)
        self.assertIn("BULLETIN", p["text"]["tcp"])     # healed once URLs exist
        self.assertIn("TCD DISCUSSION BODY", p["text"]["tcd"])
        self.assertEqual(p["text"]["tcp_url"],
                         "https://www.nhc.noaa.gov/text/MIATCPEP1.shtml")

    def test_heal_settles_no_rewrites_after_complete(self):
        sink = pf.DictSink()
        eng, avail = self._engine_with_window(sink)
        writes = []
        orig = sink.write
        sink.write = lambda k, v: (writes.append(k), orig(k, v))
        eng.poll_once()
        avail["on"] = True
        eng.poll_once()    # heal write
        eng.poll_once()    # settled: gate is back, no further writes
        eng.poll_once()
        adv_writes = [k for k in writes if "/adv/" in k]
        self.assertEqual(len(adv_writes), 2)   # initial + one heal

    def test_rolled_ahead_text_never_attached(self):
        # The CAPTURED real pair: TCP page already at advisory 14 while
        # CurrentStorms/cone are at 13. Wrong-advisory text must never
        # ship; the panel placeholder is the honest state.
        sink = pf.DictSink()
        make_engine(sink, tcp_text=TCP_SHTML_14).poll_once()
        p = sink.store["shadow/cyclolab/adv/NHC_EP012026.json"]
        if isinstance(p, str):
            p = json.loads(p)
        self.assertNotIn("tcp", p["text"])     # rejected, not mis-attached
        self.assertIn("TCD DISCUSSION BODY", p["text"]["tcd"])

    def test_unverifiable_text_never_attached(self):
        # An outage interstitial / maintenance page has no Number line:
        # unverifiable text must never be attached as advisory text.
        sink = pf.DictSink()
        make_engine(sink,
                    tcp_text="<pre>scheduled maintenance</pre>").poll_once()
        p = sink.store["shadow/cyclolab/adv/NHC_EP012026.json"]
        if isinstance(p, str):
            p = json.loads(p)
        self.assertNotIn("tcp", p["text"])

    def test_intermediate_letter_passes_verification(self):
        # Intermediate public advisories ("Advisory Number 13A") belong
        # to the same advisory family - they verify against adv 13.
        sink = pf.DictSink()
        make_engine(sink, tcp_text=TCP_SHTML.replace(
            "Advisory Number  13", "Intermediate Advisory Number  13A"
        )).poll_once()
        p = sink.store["shadow/cyclolab/adv/NHC_EP012026.json"]
        if isinstance(p, str):
            p = json.loads(p)
        self.assertIn("BULLETIN", p["text"]["tcp"])

    def test_storm_leaving_index_closes_the_heal(self):
        # A storm that dissipates with text still owed can never heal -
        # the pulse must close so the source settles (no busy-looping
        # process calls forever).
        sink = pf.DictSink()
        eng, avail = self._engine_with_window(sink)
        eng.poll_once()                       # text-less, heal pending
        empty = json.dumps({"activeStorms": []})

        # swap the index to empty (storm gone) - reach into the source
        src = eng.sources[0]
        eng2, _ = self._engine_with_window(sink)  # unused; keep API shape
        writes = []
        orig = sink.write
        sink.write = lambda k, v: (writes.append(k), orig(k, v))
        with mock.patch.object(
                ca, "_storm_entries", return_value=[]):
            eng.poll_once()                   # storm left: debt closed
            eng.poll_once()                   # settled - no process churn
        self.assertEqual([k for k in writes if "/adv/" in k], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
