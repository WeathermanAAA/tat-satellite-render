#!/usr/bin/env python3
"""Offline proofs for the floater poller's invest discovery + handoff.

No network, no prod-R2: mocked sources exercise the per-basin invest
discovery and the storm-set assembly, asserting the guarantees the
2026-06-04 "no 91W floater" fix added:
  DISCOVERY   - WP invests (90-99) come from the knackwx ATCF API (the same
                source the site's tracks pipeline uses), labeled by the
                established invest convention (INVEST 91W / category INVEST /
                raw ATCF status nature).
  LIFECYCLE   - only invests with a fix inside ACTIVE_WINDOW_HOURS float; a
                stale entry (dissipated, or a mirror ghost) is dropped.
  HANDOFF     - a designated invest is dropped the cycle it is promoted,
                via knackwx transitioned_from (explicit) or co-location with
                a same-basin named storm (tracks feeds + CurrentStorms); the
                explicit signal is RECYCLE-SAFE (a reused invest number with
                fixes newer than the first-seen transitioned_from link floats,
                even while the promoted storm still reports the link).
  ISOLATION   - a missing/flaky invest source NEVER affects named-storm
                floaters, the other invest source, or other basins; failures
                preserve last-known-good (per-source).

Run:  python tests/test_floater_poller.py -v
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import floater_poller as fp          # noqa: E402


def _iso(d: dt.datetime) -> str:
    return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


NOW = fp.utcnow()
FRESH = _iso(NOW - dt.timedelta(hours=2))
STALE = _iso(NOW - dt.timedelta(hours=fp.ACTIVE_WINDOW_HOURS + 6))
CUTOFF = NOW - dt.timedelta(hours=fp.ACTIVE_WINDOW_HOURS)


def _kx_invest(atcf_id="91W", t=FRESH, lat=19.1, lon=118.8, winds=20,
               pressure=1005, nature="DB", **extra):
    """A knackwx ATCF API entry, shaped like the live payload."""
    num_year = f"{atcf_id[:-1]}2026"
    e = {
        "atcf_id": atcf_id,
        "long_atcf_id": f"wp{num_year}" if atcf_id.endswith("W") else f"xx{num_year}",
        "storm_name": "INVEST",
        "analysis_time": t,
        "latitude": lat,
        "longitude": lon,
        "cyclone_nature": nature,
        "winds": winds,
        "pressure": pressure,
        "transitioned_from": None,
        "origin_basin": atcf_id[-1],
    }
    e.update(extra)
    return e


def _kx_named(atcf_id="07W", name="GHOST", transitioned_from=None, t=FRESH):
    return {
        "atcf_id": atcf_id, "long_atcf_id": f"wp{atcf_id[:-1]}2026",
        "storm_name": name, "analysis_time": t, "latitude": 19.0,
        "longitude": 119.0, "cyclone_nature": "TS", "winds": 40,
        "pressure": 995, "transitioned_from": transitioned_from,
        "origin_basin": atcf_id[-1],
    }


def _session_with(payload) -> mock.Mock:
    """A requests.Session whose every GET returns ``payload`` (JSON-encoded)."""
    r = mock.Mock()
    r.status_code = 200
    r.text = json.dumps(payload)
    r.raise_for_status = mock.Mock()
    s = mock.Mock()
    s.get.return_value = r
    return s


def _feed_storm(sid="JTWC_WP072026", name="GHOST", lat=19.0, lon=119.0,
                wind=40.0, active=True, t=FRESH, nature="TS"):
    return {
        "sid": sid, "name": name, "is_active": active,
        "current_category": "TS", "peak_wind_kt": wind,
        "points": [{"t": t, "lat": lat, "lon": lon, "wind_kt": wind,
                    "pressure_mb": 995.0, "nature": nature}],
    }


# ---------------------------------------------------------------------------
# DISCOVERY + LIFECYCLE: _parse_knackwx_invest / fetch_wp_invests
# ---------------------------------------------------------------------------

class TestParseKnackwxInvest(unittest.TestCase):
    def test_fresh_wp_invest_parses_with_invest_convention(self):
        s = fp._parse_knackwx_invest(_kx_invest(), CUTOFF)
        self.assertIsNotNone(s)
        self.assertEqual(s.sid, "WP912026")
        self.assertEqual(s.slug, "wp91")
        self.assertEqual(s.name, "INVEST 91W")
        self.assertEqual(s.basin, "WP")
        self.assertEqual(s.category, "INVEST")    # frontend plain-text pill
        self.assertEqual(s.nature, "DB")          # raw ATCF status -> gray badge
        self.assertEqual((s.lat, s.lon), (19.1, 118.8))
        self.assertEqual(s.current_wind_kt, 20.0)
        self.assertEqual(s.current_pressure_mb, 1005.0)

    def test_stale_invest_is_dropped(self):
        # The mirror-ghost / dissipated-invest lifecycle: an entry whose latest
        # fix is older than ACTIVE_WINDOW_HOURS never floats.
        self.assertIsNone(fp._parse_knackwx_invest(_kx_invest(t=STALE), CUTOFF))

    def test_non_wp_and_named_entries_are_not_invests(self):
        self.assertIsNone(fp._parse_knackwx_invest(_kx_invest("90E"), CUTOFF))
        self.assertIsNone(fp._parse_knackwx_invest(_kx_named("06W"), CUTOFF))

    def test_malformed_fields_are_safe(self):
        self.assertIsNone(fp._parse_knackwx_invest(_kx_invest(lat=None), CUTOFF))
        self.assertIsNone(fp._parse_knackwx_invest(_kx_invest(t=""), CUTOFF))
        s = fp._parse_knackwx_invest(_kx_invest(winds=0, pressure=0), CUTOFF)
        self.assertIsNone(s.current_wind_kt)       # 0 -> omitted from badge
        self.assertIsNone(s.current_pressure_mb)

    def test_year_falls_back_to_fix_year(self):
        s = fp._parse_knackwx_invest(_kx_invest(long_atcf_id="garbage"), CUTOFF)
        self.assertEqual(s.sid, f"WP91{fp.parse_iso(FRESH).year}")

    def test_tropical_nature_is_coerced_to_neutral(self):
        # knackwx's cyclone_nature CAN read TS/TY on a still-numbered 9X system;
        # the invest convention is NO Saffir-Simpson pill, so anything outside
        # the neutral set becomes DB (render.py's gray badge path).
        for tropical in ("TS", "TD", "TY", "HU"):
            s = fp._parse_knackwx_invest(_kx_invest(nature=tropical, winds=35), CUTOFF)
            self.assertEqual(s.nature, "DB", f"{tropical} must coerce to DB")
        for neutral in ("LO", "WV", "DB"):
            s = fp._parse_knackwx_invest(_kx_invest(nature=neutral), CUTOFF)
            self.assertEqual(s.nature, neutral)


class TestFetchWpInvests(unittest.TestCase):
    def test_live_shape_payload(self):
        payload = [_kx_invest(), _kx_named("01E", "AMANDA", "90E"),
                   _kx_named("06W", "JANGMI", "99W")]
        out = fp.fetch_wp_invests(_session_with(payload))
        self.assertIsNotNone(out)
        invests, retired = out
        self.assertEqual([s.slug for s in invests], ["wp91"])
        self.assertEqual(retired, {"90E", "99W"})

    def test_fetch_failure_returns_none(self):
        with mock.patch.object(fp, "_fetch_text", return_value=None):
            self.assertIsNone(fp.fetch_wp_invests(mock.Mock()))

    def test_malformed_json_returns_none(self):
        r = mock.Mock(status_code=200, text="{not json", raise_for_status=mock.Mock())
        s = mock.Mock()
        s.get.return_value = r
        self.assertIsNone(fp.fetch_wp_invests(s))

    def test_non_list_payload_returns_none(self):
        self.assertIsNone(fp.fetch_wp_invests(_session_with({"oops": 1})))

    def test_one_bad_entry_never_sinks_the_source(self):
        payload = ["not-a-dict", {"atcf_id": 12345}, _kx_invest()]
        invests, retired = fp.fetch_wp_invests(_session_with(payload))
        self.assertEqual([s.slug for s in invests], ["wp91"])

    def test_disabled_is_quiescence_not_failure(self):
        with mock.patch.object(fp, "WP_INVESTS_ENABLED", False):
            self.assertEqual(fp.fetch_wp_invests(mock.Mock()), ([], set()))


# ---------------------------------------------------------------------------
# Named path: invest-range entries never ride it
# ---------------------------------------------------------------------------

class TestNamedPathInvestGuard(unittest.TestCase):
    def test_active_invest_range_sid_is_skipped(self):
        feed = {"storms": [_feed_storm(sid="JTWC_WP912026", name="91W"),
                           _feed_storm(sid="JTWC_WP072026", name="GHOST")]}
        with mock.patch.object(fp, "_fetch_tracks_json", return_value=feed):
            res = fp.fetch_active_storms(mock.Mock())
        for basin, storms in res.items():
            slugs = {s.slug for s in storms}
            self.assertNotIn("wp91", slugs,
                             f"invest-range sid floated via named path ({basin})")
            self.assertIn("wp07", slugs)


# ---------------------------------------------------------------------------
# HANDOFF + ISOLATION: refresh_storms assembly
# ---------------------------------------------------------------------------

class _FakeR2:
    def __init__(self):
        self.json_puts = {}

    def get_json(self, key):
        return None

    def put_json(self, key, obj, cache):
        self.json_puts[key] = obj
        return True

    def put_bytes(self, *a, **k):
        return True

    def delete(self, keys):
        pass


def _poller() -> fp.Poller:
    with mock.patch.object(fp, "R2", _FakeR2):
        p = fp.Poller()
    return p


WP91 = fp._parse_knackwx_invest(_kx_invest(), CUTOFF)


def _refresh(p, named=None, nhc_invests=None, wp=None, current=None):
    """Run refresh_storms with every source mocked."""
    named = named if named is not None else {"wp": [], "al": [], "ep": []}
    with mock.patch.object(fp, "fetch_active_storms", return_value=named), \
         mock.patch.object(fp, "fetch_active_invests", return_value=nhc_invests), \
         mock.patch.object(fp, "fetch_wp_invests", return_value=wp), \
         mock.patch.object(fp, "fetch_current_named", return_value=current):
        p.refresh_storms()


def _named_storm(slug="wp07", lat=30.0, lon=140.0, basin="WP", name="GHOST"):
    # Default position is deliberately FAR from WP91 (19.1N 118.8E) so tests
    # that just want a named storm don't accidentally trigger the co-location
    # handoff; the handoff tests pass close coordinates explicitly.
    return fp.Storm(sid=f"JTWC_{slug.upper()}2026", slug=slug, name=name,
                    basin=basin, lat=lat, lon=lon, category="TS",
                    intensity_kt=40.0, last_fix=FRESH, current_wind_kt=40.0,
                    current_pressure_mb=995.0, nature="TS")


class TestRefreshStorms(unittest.TestCase):
    def test_wp_invest_floats_alongside_named(self):
        p = _poller()
        ghost = _named_storm()
        _refresh(p, named={"wp": [ghost], "al": [], "ep": []},
                 nhc_invests=[], wp=([WP91], set()), current={})
        self.assertEqual(set(p.storms), {"wp07", "wp91"})
        top = p.r2.json_puts[fp.top_manifest_key()]
        by_slug = {s["slug"]: s for s in top["storms"]}
        self.assertEqual(by_slug["wp91"]["name"], "INVEST 91W")
        self.assertEqual(by_slug["wp91"]["category"], "INVEST")
        self.assertEqual(by_slug["wp91"]["nature"], "DB")

    def test_explicit_transitioned_from_handoff(self):
        # 91W upgraded: knackwx named entry says transitioned_from=91W -> the
        # invest floater is dropped even before co-location could fire.
        p = _poller()
        _refresh(p, nhc_invests=[], wp=([WP91], {"91W"}), current={})
        self.assertEqual(set(p.storms), set())

    def test_colocation_handoff_with_feed_named_wp_storm(self):
        # The 90E -> One-E pattern for WP: the successor appears in the wp
        # tracks feed (no CurrentStorms for JTWC) co-located with the invest.
        p = _poller()
        succ = _named_storm(slug="wp07", lat=19.3, lon=119.1)
        _refresh(p, named={"wp": [succ], "al": [], "ep": []},
                 nhc_invests=[], wp=([WP91], set()), current={})
        self.assertEqual(set(p.storms), {"wp07"})

    def test_distant_named_storm_keeps_the_invest(self):
        p = _poller()
        far = _named_storm(slug="wp07", lat=30.0, lon=140.0)
        _refresh(p, named={"wp": [far], "al": [], "ep": []},
                 nhc_invests=[], wp=([WP91], set()), current={})
        self.assertEqual(set(p.storms), {"wp07", "wp91"})

    def test_other_basin_named_storm_never_drops_the_invest(self):
        p = _poller()
        ep = _named_storm(slug="ep01", lat=19.1, lon=118.8, basin="EP",
                          name="AMANDA")
        _refresh(p, named={"wp": [], "al": [], "ep": [ep]},
                 nhc_invests=[], wp=([WP91], set()), current={})
        self.assertEqual(set(p.storms), {"ep01", "wp91"})

    def test_knackwx_failure_preserves_lkg_and_named(self):
        p = _poller()
        _refresh(p, nhc_invests=[], wp=([WP91], set()), current={})
        self.assertIn("wp91", p.storms)
        ghost = _named_storm()
        _refresh(p, named={"wp": [ghost], "al": [], "ep": []},
                 nhc_invests=[], wp=None, current={})        # knackwx down
        self.assertEqual(set(p.storms), {"wp07", "wp91"},
                         "knackwx outage must keep LKG invest + refresh named")

    def test_named_failure_keeps_wp_invests_refreshing(self):
        p = _poller()
        _refresh(p, named={"wp": None, "al": None, "ep": None},
                 nhc_invests=[], wp=([WP91], set()), current=None)
        self.assertEqual(set(p.storms), {"wp91"})

    def test_nhc_invests_isolated_from_knackwx(self):
        p = _poller()
        ep90 = fp.Storm(sid="EP902026", slug="ep90", name="INVEST 90E",
                        basin="EP", lat=9.4, lon=-125.7, category="INVEST",
                        intensity_kt=20.0, last_fix=FRESH,
                        current_wind_kt=20.0, current_pressure_mb=1009.0,
                        nature="DB")
        _refresh(p, nhc_invests=[ep90], wp=None, current={})  # knackwx down
        self.assertEqual(set(p.storms), {"ep90"})
        _refresh(p, nhc_invests=None, wp=([WP91], set()), current={})  # NHC down
        self.assertEqual(set(p.storms), {"ep90", "wp91"})

    def test_recycled_invest_number_floats_again(self):
        # JANGMI lingers in knackwx with transitioned_from=99W for its whole
        # life; when JTWC RECYCLES 99W for a brand-new invest, its fixes are
        # NEWER than the first-seen link, so it must float, not be censored.
        p = _poller()
        _refresh(p, nhc_invests=[], wp=([], {"99W"}), current={})  # link first seen
        newer = _iso(NOW + dt.timedelta(hours=6))
        wp99 = fp._parse_knackwx_invest(
            _kx_invest("99W", t=newer, lat=12.0, lon=150.0), CUTOFF)
        _refresh(p, nhc_invests=[], wp=([wp99], {"99W"}), current={})
        self.assertEqual(set(p.storms), {"wp99"})

    def test_frozen_lkg_retired_set_cannot_suppress_newer_invest(self):
        # knackwx dies AFTER recording 90E as retired: the frozen LKG set must
        # not censor a later recycled NHC 90E whose fixes are newer (the dead
        # WP source must never veto a live invest from another source).
        p = _poller()
        _refresh(p, nhc_invests=[], wp=([], {"90E"}), current={})  # knackwx healthy
        newer = _iso(NOW + dt.timedelta(hours=6))
        ep90 = fp.Storm(sid="EP902026", slug="ep90", name="INVEST 90E",
                        basin="EP", lat=9.4, lon=-125.7, category="INVEST",
                        intensity_kt=20.0, last_fix=newer,
                        current_wind_kt=20.0, current_pressure_mb=1009.0,
                        nature="DB")
        _refresh(p, nhc_invests=[ep90], wp=None, current={})       # knackwx dead
        self.assertEqual(set(p.storms), {"ep90"})

    def test_retired_set_drops_nhc_invest_too(self):
        # The explicit signal is basin-agnostic: 01E's transitioned_from=90E
        # retires the ep90 floater even if its NHC deck lingers fresh.
        p = _poller()
        ep90 = fp.Storm(sid="EP902026", slug="ep90", name="INVEST 90E",
                        basin="EP", lat=9.4, lon=-125.7, category="INVEST",
                        intensity_kt=20.0, last_fix=FRESH,
                        current_wind_kt=20.0, current_pressure_mb=1009.0,
                        nature="DB")
        _refresh(p, nhc_invests=[ep90], wp=([], {"90E"}), current={})
        self.assertEqual(set(p.storms), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
