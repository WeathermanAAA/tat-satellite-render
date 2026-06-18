"""Tests for the JTWC/WP advisory sub-path (make_jtwc_advisories_source +
discovery), mirroring test_cyclolab_advisories + test_derived_cone. Drives the
full poll cycle with fixture text via injected fetch_text + a DictSink."""
import datetime as dt
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import poller_framework as pf  # noqa: E402
import cyclolab_advisories as C  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "cyclolab")
WEB = open(os.path.join(FIX, "wp0726web.txt")).read()
PROG = open(os.path.join(FIX, "wp0726prog.txt")).read()

KNACK = [
    {"atcf_id": "07W", "long_atcf_id": "wp072026", "storm_name": "SEVEN"},
    {"atcf_id": "93W", "long_atcf_id": "wp932026", "storm_name": "INVEST"},  # invest
    {"atcf_id": "05E", "long_atcf_id": "ep052026", "storm_name": "EUGENE"},  # non-WP
]
ADV = "shadow/cyclolab/adv/JTWC_WP072026.json"


def _ctx(src, data, sink, prev=None):
    return pf.ProcessContext(name="t", data=data, signature=src.change_key(data),
                             previous_signature=prev, valid_time=None,
                             now=dt.datetime.now(dt.timezone.utc), freshness={},
                             sink=sink)


def _src(sink, fetch_text):
    return C.make_jtwc_advisories_source(
        None, sink, prefix="shadow/cyclolab",
        knackwx_url="https://api.knackwx.com/atcf/v2",
        policy=pf.FetchPolicy(), fetch_text=fetch_text)


def _fetch_map(web=WEB, prog=PROG, knack=KNACK):
    def f(url):
        if "knackwx" in url:
            return json.dumps(knack)
        if url.endswith("wp0726web.txt"):
            return web
        if url.endswith("wp0726prog.txt"):
            return prog
        return None
    return f


class TestDiscovery(unittest.TestCase):
    def test_selects_designated_wp_only(self):
        got = C._jtwc_designated_storms(KNACK, 2026)
        sids = [s["sid"] for s in got]
        self.assertEqual(sids, ["JTWC_WP072026"])      # 93W invest + 05E EP excluded
        self.assertEqual(got[0]["nn"], 7)
        self.assertEqual(got[0]["web_url"],
                         "https://www.metoc.navy.mil/jtwc/products/wp0726web.txt")

    def test_nn_year_from_long_atcf_id(self):
        got = C._jtwc_designated_storms(
            [{"atcf_id": "13W", "long_atcf_id": "wp132025", "storm_name": "X"}], 2099)
        self.assertEqual((got[0]["nn"], got[0]["year"]), (13, 2025))   # not 2099


class TestFullPoll(unittest.TestCase):
    def test_cone_and_both_texts(self):
        sink = pf.DictSink()
        src = _src(sink, _fetch_map())
        data = src.fetch()
        src.process(_ctx(src, data, sink))
        p = sink.store[ADV]
        self.assertEqual(p["source"], "jtwc")
        self.assertEqual(p["advisory"], 1)
        self.assertEqual(p["issued_utc"], "2026-06-18T21:00:00Z")
        self.assertGreater(len(p["cone"]), 10)
        self.assertEqual(p["cone"][0], p["cone"][-1])      # closed
        self.assertEqual(len(p["points"]), 9)
        self.assertTrue(p["text"]["tcp"])                  # warning body
        self.assertTrue(p["text"]["tcd"])                  # prog (verified)
        self.assertIn("radii_method_version", p["provenance"])

    def test_change_gate_same_warning_no_rewrite(self):
        sink = pf.DictSink()
        src = _src(sink, _fetch_map())
        src.process(_ctx(src, src.fetch(), sink))
        n_after_first = len(sink.history[ADV])
        # poll again, same warning number -> heavy work skipped, no extra write
        src.process(_ctx(src, src.fetch(), sink))
        self.assertEqual(len(sink.history[ADV]), n_after_first)


class TestTextVerifyHeal(unittest.TestCase):
    def test_prog_wrong_warning_pending_then_heals(self):
        # prog at a DIFFERENT warning number -> tcd stays PENDING (not attached).
        wrong_prog = PROG.replace("NR 001", "NR 099")
        flips = {"prog": wrong_prog}
        def ft(url):
            if "knackwx" in url:
                return json.dumps(KNACK)
            if url.endswith("web.txt"):
                return WEB
            if url.endswith("prog.txt"):
                return flips["prog"]
            return None
        sink = pf.DictSink()
        src = _src(sink, ft)
        src.process(_ctx(src, src.fetch(), sink))
        self.assertTrue(sink.store[ADV]["text"]["tcp"])
        self.assertIsNone(sink.store[ADV]["text"].get("tcd"))   # mismatched -> pending
        # now the correct prog appears -> heal pulse attaches it
        flips["prog"] = PROG
        src.process(_ctx(src, src.fetch(), sink))
        self.assertTrue(sink.store[ADV]["text"]["tcd"])

    def test_missing_prog_never_blocks_cone(self):
        def ft(url):
            if "knackwx" in url:
                return json.dumps(KNACK)
            if url.endswith("web.txt"):
                return WEB
            return None                       # prog 404
        sink = pf.DictSink()
        src = _src(sink, ft)
        src.process(_ctx(src, src.fetch(), sink))
        self.assertGreater(len(sink.store[ADV]["cone"]), 10)    # cone present
        self.assertIsNone(sink.store[ADV]["text"].get("tcd"))   # tcd pending


class TestIssuanceRegression(unittest.TestCase):
    def test_older_warning_number_rejected_by_gate(self):
        sink = pf.DictSink()
        src = _src(sink, _fetch_map())
        # cache warning 5 first
        web5 = WEB.replace("WARNING NR 001", "WARNING NR 005")
        src.process(_ctx(src, _src_fetch(src, web5), sink))
        self.assertEqual(sink.store[ADV]["advisory"], 5)
        n = len(sink.history[ADV])
        # a stale mirror now serves warning 1 -> change-gate blocks it (no regress)
        src.process(_ctx(src, _src_fetch(src, WEB), sink))
        self.assertEqual(sink.store[ADV]["advisory"], 5)        # unchanged
        self.assertEqual(len(sink.history[ADV]), n)

    def test_issuance_regression_guard_on_new_number(self):
        # warning number UP but issuance DTG older (stale mirror) -> rejected.
        sink = pf.DictSink()
        src = _src(sink, _fetch_map())
        src.process(_ctx(src, _src_fetch(src, WEB), sink))      # wn1 @ 182100
        older_hdr = WEB.replace("WTPN31 PGTW 182100", "WTPN31 PGTW 170600") \
                       .replace("WARNING NR 001", "WARNING NR 002")
        # wn2 but issued 17th 0600 < cached 18th 2100 -> regression guard rejects
        try:
            src.process(_ctx(src, _src_fetch(src, older_hdr), sink))
        except RuntimeError:
            pass    # per-storm error surfaces after persisting others; expected
        self.assertEqual(sink.store[ADV]["advisory"], 1)        # not regressed


class TestIsolation(unittest.TestCase):
    def test_jtwc_failure_does_not_affect_nhc_in_same_engine_pass(self):
        sink = pf.DictSink()

        def nhc_process(ctx):
            ctx.sink.write("shadow/cyclolab/adv/NHC_AL012026.json", {"ok": True})
        nhc = pf.Source(name="cyclolab-adv", fetch=lambda: {"v": 1},
                        change_key=lambda d: d["v"], process=nhc_process)

        def boom(url):
            raise pf.TransientFetchError("knackwx down")    # JTWC discovery fails
        jtwc = _src(sink, boom)

        eng = pf.PollerEngine([nhc, jtwc], sink=sink, interval_s=0)
        results = eng.poll_once()
        self.assertTrue(results["cyclolab-adv"].ok)             # NHC succeeded
        self.assertFalse(results["cyclolab-adv-jtwc"].ok)       # JTWC failed
        self.assertIn("shadow/cyclolab/adv/NHC_AL012026.json", sink.store)  # NHC wrote

    def test_per_storm_parse_error_skips_only_that_storm(self):
        # a garbage web.txt for the storm -> skipped in fetch, no exception escapes
        def ft(url):
            if "knackwx" in url:
                return json.dumps(KNACK)
            if url.endswith("web.txt"):
                return "GARBAGE NOT A WARNING"
            return None
        sink = pf.DictSink()
        src = _src(sink, ft)
        data = src.fetch()                       # must not raise
        self.assertEqual(data["storms"], [])     # storm skipped
        src.process(_ctx(src, data, sink))       # no-op, no write
        self.assertNotIn(ADV, sink.store)


def _src_fetch(src, web):
    """Run the source's own fetch but force a specific web body (helper)."""
    # fetch() reads from the source's closed-over get_text; rebuild data by hand
    # using the same parse the source uses, to feed crafted warning bodies.
    import jtwc_warning as J
    parsed = J.parse_jtwc_warning(web)
    s = C._jtwc_designated_storms(KNACK, 2026)[0]
    s = dict(s)
    s["parsed"] = parsed
    s["warning_number"] = parsed["warning_number"]
    s["web_text"] = web
    return {"storms": [s]}


if __name__ == "__main__":
    unittest.main()
