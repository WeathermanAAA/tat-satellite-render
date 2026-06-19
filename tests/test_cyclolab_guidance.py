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


# a WPAC DTC adecks_open row (global / ensemble model): BASIN=WP, CY=07, 143E
def _wp_row(tech, tau, lat="129N", lon="1431E", vmax=30, mslp=1005, init="2026061818"):
    return ["WP", "07", init, "01", tech, tau, lat, lon, vmax, mslp, "TS", 34]


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
        if "adecks_open" in url:        # DTC WPAC a-deck: plain .dat, NOT gzipped
            for sub, raw in self.adecks.items():
                if sub in url:
                    return _Resp(content=raw.encode())
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
        self.assertEqual(gp.sid_parts("JTWC_WP072026"), ("WP", "07", "2026"))  # JTWC WP now accepted
        self.assertEqual(gp.sid_parts("JTWC_WP922026"), ("WP", "92", "2026"))  # WP invest too
        self.assertIsNone(gp.sid_parts("JTWC_IO012026"))   # IO not yet a guidance basin
        self.assertIsNone(gp.sid_parts("garbage"))

    def test_discover_keeps_active_guidance_basins(self):
        feed = {"features": [
            {"properties": {"kind": "active_marker", "storm_id": "NHC_EP932026"}},
            {"properties": {"kind": "active_marker", "storm_id": "JTWC_WP072026"}},   # WP designated -> kept
            {"properties": {"kind": "active_marker", "storm_id": "JTWC_IO012026"}},   # non-guidance basin -> dropped
            {"properties": {"kind": "observation", "storm_id": "NHC_AL012026"}},       # not active_marker
        ]}
        self.assertEqual(gp.discover_entities(feed), ["NHC_EP932026", "JTWC_WP072026"])

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


class TestWPGuidance(unittest.TestCase):
    """JTWC/WP guidance: DTC adecks_open a-deck, WP curated aids, SHIPS stub,
    NHC isolation. NHC paths must be untouched."""

    def test_adeck_url_routes_wp_to_dtc(self):
        u = gp.adeck_url("WP", "07", "2026")
        self.assertIn("adecks_open", u)
        self.assertTrue(u.endswith("awp072026.dat"))   # plain .dat, not .gz
        self.assertIn("aid_public", gp.adeck_url("EP", "93", "2026"))   # NHC unchanged
        self.assertTrue(gp.adeck_url("EP", "93", "2026").endswith(".dat.gz"))

    def test_parse_adeck_wp_curated_set(self):
        # WP keeps ensemble means + global determ; DROPS the GEFS spaghetti (AP##)
        # and the NHC interpolated aids (AVNI) that never appear in a WPAC a-deck.
        a = _adeck([_wp_row("AEMN", 0), _wp_row("AEMN", 12, lat="135N"),
                    _wp_row("CEMN", 0), _wp_row("NGX", 0), _wp_row("AC00", 0),
                    _wp_row("AP05", 0), _wp_row("AVNI", 0)])
        g = cg.parse_adeck(a, basin="WP")
        self.assertEqual(set(g["present_aids"]), {"AEMN", "CEMN", "NGX", "AC00"})
        self.assertEqual(set(g["consensus"]), {"AEMN", "CEMN"})   # ensemble means
        self.assertNotIn("AP05", g["aids"])                       # spaghetti dropped
        self.assertNotIn("AVNI", g["aids"])                       # NHC aid absent

    def test_parse_adeck_wp_picks_latest_init_with_aids(self):
        # newest synoptic is a CARQ-only analysis (no model aids) -> guidance must
        # fall back to the prior init that actually carries the ensemble mean.
        a = _adeck([_wp_row("AEMN", 0, init="2026061818"),
                    _wp_row("CARQ", 0, init="2026061900")])   # CARQ non-curated, newest
        g = cg.parse_adeck(a, basin="WP")
        self.assertEqual(g["init_cycle"], "2026061818")
        self.assertIn("AEMN", g["present_aids"])

    def test_parse_adeck_nhc_byte_identical_with_basin_arg(self):
        a = _adeck([_row("AVNI", 0), _row("TVCN", 0), _row("AEMN", 0)])  # AEMN ignored for NHC
        self.assertEqual(cg.parse_adeck(a), cg.parse_adeck(a, basin="AL"))
        self.assertNotIn("AEMN", cg.parse_adeck(a)["aids"])

    def test_run_once_wp_guidance_and_ships_stub(self):
        wp = _adeck([_wp_row("AEMN", 0), _wp_row("AEMN", 12, lat="135N"), _wp_row("CEMN", 0)])
        sess = _Session(_feed("JTWC_WP072026"), {"awp072026": wp}, {})
        written = {}
        st = gp.run_once(sess, lambda k, o: (written.__setitem__(k, o) or True))
        self.assertEqual(len(st), 1)
        gj = written[f"{gp.R2_PREFIX}/JTWC_WP072026/guidance.json"]
        self.assertIn("AEMN", gj["present_aids"])
        self.assertEqual(gj["source"], "dtc-atcf-adecks_open")
        sj = written[f"{gp.R2_PREFIX}/JTWC_WP072026/ships.json"]
        self.assertFalse(sj["available"])
        self.assertEqual(sj["reason"], "SHIPS not published for WPAC")
        # NO formation.json for WP (no NHC TWO) and the run still reports ok.
        self.assertNotIn(f"{gp.R2_PREFIX}/JTWC_WP072026/formation.json", written)
        self.assertTrue(st[0]["ok"])

    def test_wp_failure_isolated_from_nhc(self):
        ep = _adeck([_row("AVNI", 0)])
        sess = _Session(_feed("NHC_EP932026", "JTWC_WP072026"),
                        {"aep932026": ep}, {})   # no awp072026 -> WP a-deck 404s
        st = {s["sid"]: s for s in gp.run_once(sess, lambda k, o: True)}
        self.assertTrue(st["NHC_EP932026"]["ok"])            # NHC fully processed
        self.assertFalse(st["JTWC_WP072026"]["ok"])          # WP degrades, no crash
        self.assertEqual(st["JTWC_WP072026"].get("reason"), "a-deck unavailable")


_TWO_EP = (
    "<rss><channel><item><description><![CDATA[000<br />TWOEP<br /><br />"
    "Tropical Weather Outlook<br /><br />"
    "Well East-Southeast of the Hawaiian Islands (EP93):<br />"
    "A broad area of low pressure...<br />"
    "* Formation chance through 48 hours...low...20 percent.<br />"
    "* Formation chance through 7 days...low...30 percent.<br /><br />$$<br />"
    "]]></description></item></channel></rss>")

# two numbered areas in one outlook (Atlantic), different levels.
_TWO_AT = (
    "Northwestern Gulf of America (AL90):\n"
    "A trough... long flooding text spanning several lines that must not\n"
    "break the block boundary before the chances.\n"
    "* Formation chance through 48 hours...medium...60 percent.\n"
    "* Formation chance through 7 days...medium...60 percent.\n\n"
    "Far Eastern Atlantic (AL91):\n"
    "A vigorous tropical wave...\n"
    "* Formation chance through 48 hours...high...70 percent.\n"
    "* Formation chance through 7 days...high...90 percent.\n$$")


class TestTWO(unittest.TestCase):
    def test_level_buckets_match_nhc(self):
        # <=30 low (yellow), 40-60 medium (orange), >=70 high (red)
        self.assertEqual([cg.formation_level(p) for p in (0, 30, 40, 60, 70, 100)],
                         ["low", "low", "medium", "medium", "high", "high"])
        self.assertIsNone(cg.formation_level(None))

    def test_parse_single_invest_area_from_rss(self):
        out = cg.parse_two(_TWO_EP, 2026)
        self.assertIn("NHC_EP932026", out)
        f = out["NHC_EP932026"]
        self.assertEqual((f["p48"], f["p7"]), (20, 30))
        self.assertEqual(f["level"], "low")                 # max(20,30)=30 -> low
        self.assertEqual(f["area"], "Well East-Southeast of the Hawaiian Islands")

    def test_parse_two_areas_distinct_levels(self):
        out = cg.parse_two(_TWO_AT, 2026)
        self.assertEqual(out["NHC_AL902026"]["p7"], 60)
        self.assertEqual(out["NHC_AL902026"]["level"], "medium")
        self.assertEqual(out["NHC_AL912026"]["p7"], 90)
        self.assertEqual(out["NHC_AL912026"]["level"], "high")   # red

    def test_no_numbered_area_is_empty(self):
        self.assertEqual(cg.parse_two("an area of low pressure, no invest yet", 2026), {})

    def test_active_systems_ptc_keyed_to_real_sid(self):
        # The LIVE TWO (2026-06-16 2336Z) AFTER advisories began: the system has
        # left the numbered-invest list and lives in the "Active Systems"
        # narrative as "Potential Tropical Cyclone One" @ 70/70, with the TCP
        # AWIPS header MIATCPAT1. parse_two must read those odds and key them to
        # the REAL designated sid NHC_AL012026 (AT->AL, storm 1) - so the pill
        # shows a live 70/70 instead of freezing at the invest-era value.
        raw = _read("cyclolab/twoat_active_ptc.xml")
        out = cg.parse_two(raw, 2026)
        self.assertIn("NHC_AL012026", out)
        f = out["NHC_AL012026"]
        self.assertEqual((f["p48"], f["p7"]), (70, 70))
        self.assertEqual(f["level"], "high")
        # no spurious invest entry (the (AL90) tag is gone from this TWO)
        self.assertNotIn("NHC_AL902026", out)

    def test_active_systems_block_does_not_bleed_into_disturbances(self):
        # A combined TWO: the Active-Systems PTC (70/70) FOLLOWED by a numbered
        # disturbance area (AL91, 20/40) BEFORE the &&. The PTC's block must NOT
        # absorb the disturbance's chances (last-wins would otherwise emit the
        # PTC at 20/40). The lone-PTC fixture masked this; this is the guard.
        combined = (
            "Active Systems:\nThe National Hurricane Center is issuing "
            "advisories on Potential Tropical Cyclone One, located over south "
            "Texas.\n* Formation chance through 48 hours...high...70 percent.\n"
            "* Formation chance through 7 days...high...70 percent.\n\n"
            "1. Eastern Tropical Atlantic (AL91):\nA tropical wave.\n"
            "* Formation chance through 48 hours...low...20 percent.\n"
            "* Formation chance through 7 days...medium...40 percent.\n\n&&\n"
            "Public Advisories on Potential Tropical Cyclone One are issued "
            "under WMO header WTNT31 KNHC and under AWIPS header MIATCPAT1.\n$$")
        out = cg.parse_two(combined, 2026)
        self.assertEqual((out["NHC_AL012026"]["p48"],
                          out["NHC_AL012026"]["p7"]), (70, 70))
        # the disturbance is still read by the numbered-invest path at its OWN odds
        self.assertEqual(out["NHC_AL912026"]["p7"], 40)

    def test_active_systems_named_storm_with_trailing_disturbance(self):
        # A NAMED storm in Active Systems FOLLOWED by a disturbance must still
        # yield NO formation pill (its block must not absorb the disturbance's
        # chance and attach a spurious genesis pill to an already-named storm).
        s = ("Active Systems:\nThe National Hurricane Center is issuing "
             "advisories on Hurricane Alberto, located over the Gulf.\n\n"
             "1. Eastern Atlantic (AL92):\nA wave.\n"
             "* Formation chance through 48 hours...high...80 percent.\n"
             "* Formation chance through 7 days...high...80 percent.\n\n&&\n"
             "Public Advisories on Hurricane Alberto are issued under WMO "
             "header WTNT34 KNHC and under AWIPS header MIATCPAT2.\n$$")
        out = cg.parse_two(s, 2026)
        self.assertNotIn("NHC_AL022026", out)

    def test_active_systems_named_storm_yields_no_formation(self):
        # A NAMED storm in the Active-Systems narrative carries NO formation
        # chance (it is already a TC) -> no pill entry, even though it has an
        # AWIPS header.
        named = (
            "Active Systems:\nThe National Hurricane Center is issuing "
            "advisories on Hurricane Alberto, located over the Gulf.\n\n&&\n"
            "Public Advisories on Hurricane Alberto are issued under WMO header "
            "WTNT34 KNHC and under AWIPS header MIATCPAT2.\n$$")
        self.assertEqual(cg.parse_two(named, 2026), {})


class TestFormationWrite(unittest.TestCase):
    def test_invest_gets_formation_json_named_does_not(self):
        writes = {}
        put = lambda k, o: writes.__setitem__(k, o) or True
        fmap = {"NHC_EP932026": {"sid": "NHC_EP932026", "p48": 20, "p7": 30, "level": "low"}}
        # invest -> formation.json written from the map
        gp.process_entity("NHC_EP932026", _StubSession(), put, fmap)
        self.assertIn("cyclolab/NHC_EP932026/formation.json", writes)
        self.assertEqual(writes["cyclolab/NHC_EP932026/formation.json"]["p7"], 30)
        # a designated storm gets NO formation.json
        writes.clear()
        gp.process_entity("NHC_EP012026", _StubSession(), put, fmap)
        self.assertNotIn("cyclolab/NHC_EP012026/formation.json", writes)


class _StubSession:
    """Minimal requests.Session stand-in: a-deck/SHIPS fetches return empty so
    process_entity reaches the formation branch without network."""
    def get(self, url, **kw):
        class _R:
            status_code = 404
            content = b""
            text = ""
        return _R()


if __name__ == "__main__":
    unittest.main()
