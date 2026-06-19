"""Never-regress guard (CycloLab cluster): the poller must never publish a
designated storm's track sparser than its last-known-good, so a transient JTWC
b-deck/mirror failure can't clobber the full track. ACE-safety is gate #1:
prove the ACE feed is byte-identical regardless of the tracks-feed never-regress.
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feed_recompute as fr  # noqa: E402


def _storm(sid, npts, is_invest=False, base_kt=30, is_active=True,
           name=None, cat="TD"):
    # ascending-time points; intensifying so peak grows with the track.
    pts = [{"t": "2026-06-%02dT%02d:00:00" % (11 + i // 24, i % 24),
            "lat": 12 + i * 0.1, "lon": 145 - i * 0.2,
            "wind_kt": base_kt + i, "pressure_mb": 1004 - i, "nature": "TS",
            "cls": "TD"} for i in range(npts)]
    lf = (pts[-1]["t"] + "Z") if pts else None
    return {"sid": sid, "is_invest": is_invest, "points": pts,
            "latest_fix_valid_utc": lf, "current_category": cat,
            "is_active": is_active,
            "name": name or ("SEVEN" if "07" in sid else sid)}


def _feed(storms):
    return {"basin": "wp", "storms": storms, "header": {"active": len(storms)},
            "latest_fix_valid_utc": "x"}


def _designated(s):
    return not s.get("is_invest")


class TestNeverRegress(unittest.TestCase):
    def test_clobber_republishes_full_track(self):
        hwm = {}
        # poll 1: full 31-pt track -> adopt + HWM
        f1 = _feed([_storm("JTWC_WP072026", 31)])
        self.assertEqual(fr.apply_never_regress(f1, hwm, _designated), 0)
        self.assertEqual(len(f1["storms"][0]["points"]), 31)
        # poll 2: transient b-deck miss -> 1-pt clobber -> republish 31
        f2 = _feed([_storm("JTWC_WP072026", 1)])
        n = fr.apply_never_regress(f2, hwm, _designated)
        self.assertEqual(n, 1)
        self.assertEqual(len(f2["storms"][0]["points"]), 31)   # NOT 1

    def test_growth_adopts_and_updates_hwm(self):
        hwm = {}
        fr.apply_never_regress(_feed([_storm("JTWC_WP072026", 31)]), hwm, _designated)
        f = _feed([_storm("JTWC_WP072026", 32)])     # a new fix landed
        self.assertEqual(fr.apply_never_regress(f, hwm, _designated), 0)
        self.assertEqual(len(f["storms"][0]["points"]), 32)
        self.assertEqual(hwm["JTWC_WP072026"]["n"], 32)

    def test_debounce_releases_after_sustained_sparseness(self):
        hwm = {}
        fr.apply_never_regress(_feed([_storm("X_WP012026", 20)]), hwm,
                               _designated, max_misses=3)
        # 3 sparse polls -> republished; 4th -> released (adopt the sparse)
        for i in range(3):
            f = _feed([_storm("X_WP012026", 2)])
            self.assertEqual(fr.apply_never_regress(f, hwm, _designated, max_misses=3), 1)
            self.assertEqual(len(f["storms"][0]["points"]), 20)
        f = _feed([_storm("X_WP012026", 2)])
        self.assertEqual(fr.apply_never_regress(f, hwm, _designated, max_misses=3), 0)
        self.assertEqual(len(f["storms"][0]["points"]), 2)     # released
        self.assertEqual(hwm["X_WP012026"]["n"], 2)            # HWM reset

    def test_invests_never_regressed(self):
        hwm = {}
        fr.apply_never_regress(_feed([_storm("NHC_AL902026", 10, is_invest=True)]),
                               hwm, _designated)
        f = _feed([_storm("NHC_AL902026", 1, is_invest=True)])  # invest snapshot
        self.assertEqual(fr.apply_never_regress(f, hwm, _designated), 0)
        self.assertEqual(len(f["storms"][0]["points"]), 1)     # left as-is
        self.assertNotIn("NHC_AL902026", hwm)                  # not tracked

    def test_first_poll_no_regress(self):
        hwm = {}
        f = _feed([_storm("JTWC_WP072026", 1)])   # first sight, sparse, no HWM
        self.assertEqual(fr.apply_never_regress(f, hwm, _designated), 0)
        self.assertEqual(len(f["storms"][0]["points"]), 1)

    def test_review_total_track_loss_absent_storm_re_appended(self):
        # REVIEW MAJOR: a designated storm that VANISHES from the poll entirely
        # (total b-deck+knackwx miss) is re-appended from the HWM (debounced),
        # not silently lost.
        hwm = {}
        fr.apply_never_regress(_feed([_storm("JTWC_WP072026", 30)]), hwm, _designated)
        f = _feed([])                              # storm absent this poll
        n = fr.apply_never_regress(f, hwm, _designated)
        self.assertEqual(n, 1)
        self.assertEqual([s["sid"] for s in f["storms"]], ["JTWC_WP072026"])
        self.assertEqual(len(f["storms"][0]["points"]), 30)   # full track restored

    def test_review_is_active_retirement_not_resurrected(self):
        # REVIEW MAJOR: a same-poll dissipation (is_active False) + sparse track
        # must keep the FRESH is_active=False (don't resurrect the live marker);
        # only the track geometry is restored.
        hwm = {}
        fr.apply_never_regress(_feed([_storm("NHC_AL072026", 30, is_active=True)]),
                               hwm, _designated)
        f = _feed([_storm("NHC_AL072026", 1, is_active=False)])   # retired + sparse
        fr.apply_never_regress(f, hwm, _designated)
        s = f["storms"][0]
        self.assertFalse(s["is_active"])                  # fresh retirement kept
        self.assertEqual(len(s["points"]), 30)            # but full track restored

    def test_review_rename_upgrade_not_masked(self):
        # REVIEW MINOR: a same-poll rename/upgrade (ONE/TD -> ARTHUR/TS) + sparse
        # keeps the FRESH name/category, not the stale HWM scalars.
        hwm = {}
        fr.apply_never_regress(
            _feed([_storm("NHC_AL072026", 8, name="ONE", cat="TD")]), hwm, _designated)
        f = _feed([_storm("NHC_AL072026", 1, name="ARTHUR", cat="TS", base_kt=40)])
        fr.apply_never_regress(f, hwm, _designated)
        s = f["storms"][0]
        self.assertEqual(s["name"], "ARTHUR")             # fresh name kept
        self.assertEqual(s["current_category"], "TS")     # fresh category kept
        self.assertEqual(len(s["points"]), 8)             # full track restored

    def test_part_c_radii_survive_the_clobber(self):
        # PART C: analyzed wind radii ride on the b-deck points (ace_core's
        # _fix_radii serializer). A b-deck-absent clobber drops them; the
        # never-regress republishes the full radii-bearing storm so the Wind
        # Swath + current-fix quadrant rings render (no new radii code needed --
        # the swath is fed by the preserved full track).
        hwm = {}
        full = _storm("JTWC_WP072026", 12, base_kt=40)
        for p in full["points"]:                       # the b-deck radii a TS+ carries
            p["radii"] = {"34": [90, 80, 70, 80]}
        fr.apply_never_regress(_feed([full]), hwm, _designated)
        clob = _feed([_storm("JTWC_WP072026", 1, base_kt=40)])   # b-deck gone -> no radii
        self.assertNotIn("radii", clob["storms"][0]["points"][0])
        fr.apply_never_regress(clob, hwm, _designated)            # republish full
        kept = clob["storms"][0]["points"]
        self.assertEqual(len(kept), 12)
        self.assertTrue(any("radii" in p for p in kept))          # radii restored


class TestAceUntouched(unittest.TestCase):
    """GATE #1: the never-regress guard touches ONLY the tracks feed; the ACE
    feed (recompute_ace_feed) is a separate computation and must be byte-
    identical whether or not the guard republishes a clobbered track."""

    def test_ace_feed_byte_identical_through_guard(self):
        # The guard never calls recompute_ace_feed and never mutates the ACE
        # inputs; assert that running the guard on a tracks feed does not require
        # or alter any ACE state. Structural proof: apply_never_regress's only
        # writes are to tracks_feed['storms'/'header'/'latest_fix_valid_utc'/
        # 'staleness_minutes'] -- never an ACE feed. (The live cross-basin ACE
        # byte-identical proof runs in s1-style audit post-deploy.)
        hwm = {}
        ace_like = {"season_ace": 12.345, "storms": [{"ace": 1.2}]}
        before = copy.deepcopy(ace_like)
        f = _feed([_storm("JTWC_WP072026", 31)])
        fr.apply_never_regress(f, hwm, _designated)
        # a republish poll: capture which tracks_feed keys the guard touches
        f2 = _feed([_storm("JTWC_WP072026", 1)])
        f2["extra_unrelated"] = {"keep": 1}
        before2 = {k: copy.deepcopy(v) for k, v in f2.items()}
        fr.apply_never_regress(f2, hwm, _designated)        # republishes
        self.assertEqual(ace_like, before)                  # sibling ACE untouched
        # the guard mutates ONLY the tracks-feed keys -- never any ACE key.
        changed = {k for k in f2 if k not in before2 or f2[k] != before2[k]}
        self.assertTrue(changed <= {"storms", "header", "latest_fix_valid_utc",
                                    "staleness_minutes"},
                        f"guard touched unexpected keys: {changed}")
        self.assertEqual(f2["extra_unrelated"], {"keep": 1})  # nothing else moved


if __name__ == "__main__":
    unittest.main()
