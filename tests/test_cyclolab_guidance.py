#!/usr/bin/env python3
"""
Unit tests for the CycloLab guidance DATA layer (cyclolab_guidance parsers +
cyclolab_guidance_poller discovery/isolation). No network: the a-deck/SHIPS/feed are
injected. Real captured EP932026 (2026-06-15 18Z) files in tests/fixtures/ double as
format-drift guards. Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import gzip
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cyclolab_guidance as cg          # noqa: E402
import cyclolab_guidance_poller as gp   # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _read(name):
    with open(os.path.join(FIX, name)) as f:
        return f.read()


def _adeck(rows):
    return "\n".join(", ".join(str(x) for x in r) for r in rows)


# a synthetic a-deck row: BASIN,CY,INIT,TECHNUM,TECH,TAU,LAT,LON,VMAX,MSLP,TY,RAD
def _row(tech, tau, lat="120N", lon="800W", vmax=40, mslp=1000, init="2026061512", rad=34, ty="TS"):
    return ["AL", "90", init, "01", tech, tau, lat, lon, vmax, mslp, ty, rad]


class TestAdeck(unittest.TestCase):
    def test_dedupe_on_tech_tau(self):
        # 34/50/64 wind-radii rows for the SAME (TECH,TAU) -> ONE point
        a = _adeck([_row("AVNI", 0, rad=34), _row("AVNI", 0, rad=50), _row("AVNI", 0, rad=64),
                    _row("AVNI", 12, lat="130N")])
        g = cg.parse_adeck(a)
        self.assertEqual([p["tau"] for p in g["aids"]["AVNI"]], [0, 12])   # not [0,0,0,12]

    def test_tenths_latlon_and_hemisphere(self):
        g = cg.parse_adeck(_adeck([_row("AVNI", 0, lat="90N", lon="1380W"),
                                   _row("HWFI", 0, lat="155S", lon="0300E")]))
        self.assertEqual(g["aids"]["AVNI"][0]["lat"], 9.0)
        self.assertEqual(g["aids"]["AVNI"][0]["lon"], -138.0)   # W negative
        self.assertEqual(g["aids"]["HWFI"][0]["lat"], -15.5)    # S negative
        self.assertEqual(g["aids"]["HWFI"][0]["lon"], 30.0)     # E positive

    def test_vmax_mslp_zero_is_missing(self):
        g = cg.parse_adeck(_adeck([_row("AVNI", 0, vmax=0, mslp=0)]))
        self.assertIsNone(g["aids"]["AVNI"][0]["vmax"])
        self.assertIsNone(g["aids"]["AVNI"][0]["mslp"])

    def test_latest_init_only_and_curated_filter(self):
        a = _adeck([
            _row("AVNI", 0, init="2026061512"),     # latest
            _row("AVNI", 0, init="2026061506", lat="100N"),  # OLDER init -> excluded
            _row("AVNO", 0, init="2026061512"),     # non-curated -> excluded
            _row("CARQ", -12, init="2026061512"),   # analysis (neg tau / non-curated) -> excluded
        ])
        g = cg.parse_adeck(a)
        self.assertEqual(g["init_cycle"], "2026061512")
        self.assertEqual(list(g["aids"].keys()), ["AVNI"])
        self.assertEqual(g["aids"]["AVNI"][0]["lat"], 12.0)     # the latest-init fix, not 10.0

    def test_consensus_and_aid_split(self):
        g = cg.parse_adeck(_adeck([_row("TVCN", 0), _row("HCCA", 0), _row("DSHP", 0), _row("AVNI", 0)]))
        self.assertEqual(set(g["consensus"]), {"TVCN", "HCCA"})         # IVCN absent here
        self.assertIn("DSHP", g["intensity_aids"])
        self.assertNotIn("DSHP", g["track_aids"])                       # DSHP is intensity-only

    def test_fresh_invest_statistical_only(self):
        # a brand-new invest: only statistical intensity aids, no dynamical track aids
        g = cg.parse_adeck(_adeck([_row("SHIP", 0), _row("DSHP", 0), _row("LGEM", 0)]))
        self.assertEqual(set(g["present_aids"]), {"SHIP", "DSHP", "LGEM"})
        self.assertEqual(g["track_aids"], [])                          # no track guidance yet

    def test_real_fixture_shape(self):
        g = cg.parse_adeck(_read("aep932026_sample.dat"))
        self.assertEqual(g["init_cycle"], "2026061518")
        self.assertIn("AVNI", g["present_aids"])
        self.assertIn("TVCN", g["consensus"])
        self.assertTrue(all(len({p["tau"] for p in pts}) == len(pts) for pts in g["aids"].values()))


class TestShips(unittest.TestCase):
    def setUp(self):
        self.s = cg.parse_ships(_read("ep932026_ships_sample.txt"))

    def test_env_series_mapping_and_sentinels(self):
        self.assertEqual(self.s["taus"][:5], [0, 6, 12, 18, 24])
        self.assertEqual(self.s["env_series"]["SST (C)"][0], 28.9)
        self.assertEqual(self.s["env_series"]["SHEAR (KT)"][0], 8.0)
        lat = self.s["env_series"]["LAT (DEG N)"]
        self.assertEqual(lat[0], 9.2)
        self.assertIsNone(lat[-1])                                     # xx.x sentinel -> null
        self.assertEqual(self.s["storm_type"][0], "TROP")

    def test_ri_matrix_cols_rows(self):
        self.assertEqual(self.s["ri_matrix"]["cols"][0], "20/12")
        self.assertEqual(self.s["ri_matrix"]["cols"][-1], "65/72")
        self.assertIn("SHIPS-RII", self.s["ri_matrix"]["rows"])
        self.assertIn("Consensus", self.s["ri_matrix"]["rows"])
        self.assertEqual(self.s["ri_matrix"]["rows"]["SHIPS-RII"]["25/24"], 12.6)

    def test_ri_matrix_999_sentinel_null(self):
        synth = (
            "Matrix of RI probabilities\n"
            "------\n"
            "  RI (kt / h)  | 20/12 | 25/24 |65/72\n"
            "------\n"
            "   SHIPS-RII:   999.0%   12.6%    0.0%\n"
        )
        m = cg.parse_ships(synth)["ri_matrix"]
        self.assertIsNone(m["rows"]["SHIPS-RII"]["20/12"])             # 999.0% -> null
        self.assertEqual(m["rows"]["SHIPS-RII"]["25/24"], 12.6)

    def test_prelim_ri_and_predictor_table(self):
        self.assertEqual(self.s["prelim_ri_prob"], 0.4)
        self.assertTrue(self.s["ri_predictor_table"])
        self.assertEqual(self.s["ri_predictor_table"][0]["value"], 126.7)

    def test_ahi(self):
        self.assertEqual(self.s["ahi"]["value"], 0)
        self.assertIn("NOT ANNULAR", self.s["ahi"]["verdict"])


# ---- poller: discovery + isolation (injected I/O) ----
class _Resp:
    def __init__(self, *, content=b"", text="", payload=None, status=200):
        self.content, self.text, self._p, self.status_code = content, text, payload, status

    def json(self):
        return self._p


class _Session:
    """Routes .get by URL substring to a canned response."""
    def __init__(self, feed, adecks, ships):
        self.feed, self.adecks, self.ships = feed, adecks, ships

    def get(self, url, headers=None, timeout=None):
        if "global_storms" in url:
            return _Resp(payload=self.feed)
        if "aid_public" in url:
            for sub, raw in self.adecks.items():
                if sub in url:
                    return _Resp(content=gzip.compress(raw.encode()))
            return _Resp(status=404)
        if "stext" in url:
            for sub, txt in self.ships.items():
                if sub in url:
                    return _Resp(text=txt, status=200)
            return _Resp(text="<html>404</html>", status=404)
        return _Resp(status=404)


def _feed(*sids, kind="active_marker"):
    return {"features": [{"properties": {"kind": kind, "storm_id": s}} for s in sids]}


class TestPoller(unittest.TestCase):
    def test_sid_parts(self):
        self.assertEqual(gp.sid_parts("NHC_EP932026"), ("EP", "93", "2026"))
        self.assertEqual(gp.sid_parts("NHC_AL012026"), ("AL", "01", "2026"))
        self.assertIsNone(gp.sid_parts("JTWC_WP922026"))   # not NHC AL/EP/CP
        self.assertIsNone(gp.sid_parts("garbage"))

    def test_discover_filters_to_active_nhc(self):
        feed = {"features": [
            {"properties": {"kind": "active_marker", "storm_id": "NHC_EP932026"}},
            {"properties": {"kind": "active_marker", "storm_id": "JTWC_WP922026"}},   # non-NHC
            {"properties": {"kind": "observation", "storm_id": "NHC_AL012026"}},       # not active_marker
        ]}
        self.assertEqual(gp.discover_entities(feed), ["NHC_EP932026"])

    def test_run_once_writes_and_isolates(self):
        adeck = _adeck([_row("AVNI", 0), _row("AVNI", 12, lat="130N"), _row("DSHP", 0)])
        ships = _read("ep932026_ships_sample.txt")
        # EP93 has both a-deck + ships; AL90 a-deck present but ships missing (degrade)
        sess = _Session(_feed("NHC_EP932026", "NHC_AL902026"),
                        {"aep932026": adeck, "aal902026": adeck}, {"EP93": ships})
        written = {}
        st = gp.run_once(sess, lambda k, o: (written.__setitem__(k, o) or True))
        self.assertEqual(len(st), 2)
        # both entities wrote guidance + ships json (ships present for EP93, "unavailable" for AL90)
        self.assertIn(f"{gp.R2_PREFIX}/NHC_EP932026/guidance.json", written)
        self.assertIn(f"{gp.R2_PREFIX}/NHC_EP932026/ships.json", written)
        self.assertTrue(written[f"{gp.R2_PREFIX}/NHC_EP932026/ships.json"]["available"])
        self.assertFalse(written[f"{gp.R2_PREFIX}/NHC_AL902026/ships.json"]["available"])  # graceful

    def test_one_entity_failure_isolated(self):
        sess = _Session(_feed("NHC_EP932026", "NHC_EP912026"),
                        {"aep932026": _adeck([_row("AVNI", 0)])}, {})   # EP91 a-deck missing
        st = {s["sid"]: s for s in gp.run_once(sess, lambda k, o: True)}
        self.assertTrue(st["NHC_EP932026"]["ok"])                  # healthy one still processed
        self.assertFalse(st["NHC_EP912026"]["ok"])                 # missing-a-deck one degrades, no crash

    def test_feed_failure_swallowed(self):
        class Boom:
            def get(self, *a, **k):
                raise RuntimeError("net down")
        self.assertEqual(gp.run_once(Boom(), lambda k, o: True), [])   # heartbeat skip, no crash


if __name__ == "__main__":
    unittest.main()
