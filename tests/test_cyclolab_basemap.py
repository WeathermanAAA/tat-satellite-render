"""Maps-pass tests: the ne_10m basemap bake (land + WHITE coastline,
distance-adaptive Douglas-Peucker, open-polyline clip) and the CycloLab
shell maps-pass restyle markers (gear cog, basemap canon CSS, blue-glass
cone + solid white centerline, white track connector)."""
import json
import math
import unittest
from pathlib import Path

import cyclolab_basemap as cb
import cyclolab_shell

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "cyclolab" / "synth_storm.json"
FEED_URL = "https://cdn.triple-a-tropics.com/feeds/ep_tracks_data.json"

# representative storms: near-land (dense coast), mid-ocean (sparse), and an
# antimeridian case (the contiguous-unwrap + shift-into-frame path).
CASES = [("near-land", 11.2, -87.5, "EP"),
         ("mid-ocean", 22.0, -135.0, "EP"),
         ("antimeridian", 18.0, 178.0, "WP")]


class TestBasemapBake(unittest.TestCase):
    def test_emits_coast_land_borders_keys(self):
        bm = cb.basemap_for(11.2, -87.5, "EP")
        for k in ("land", "coast", "borders"):
            self.assertIn(k, bm)
            self.assertIsInstance(bm[k], list)

    def test_coast_vertices_are_a_subset_of_land(self):
        # round-2 #2: the coast is DERIVED from the land-fill rings (minus the
        # window-edge segments), so every coast vertex is a land vertex - the
        # thick white coast can NEVER misalign with the fill into a sliver.
        for nm, lat, lon, basin in CASES:
            bm = cb.basemap_for(lat, lon, basin)
            land = {tuple(p) for r in bm["land"] for p in r}
            coast = {tuple(p) for r in bm["coast"] for p in r}
            self.assertTrue(coast <= land,
                            f"{nm}: {len(coast - land)} coast pts not in land")

    def test_no_sliver_islands(self):
        # round-2 #2: after the post-DP bbox/thin/area filters, no kept land
        # ring is a near-collinear sliver (a white dash on the ocean).
        bm = cb.basemap_for(11.2, -87.5, "EP")
        for r in bm["land"]:
            xs = [p[0] for p in r]
            ys = [p[1] for p in r]
            self.assertGreaterEqual(min(max(xs) - min(xs), max(ys) - min(ys)),
                                    cb.THIN_DEG - 1e-9, "thin sliver kept")
            area = abs(sum(r[k][0] * r[(k + 1) % len(r)][1] -
                           r[(k + 1) % len(r)][0] * r[k][1]
                           for k in range(len(r)))) * 0.5
            self.assertGreaterEqual(area, cb.AREA_MIN - 1e-9,
                                    "near-collinear sliver kept")

    def test_borders_and_states_are_clipped_to_land(self):
        # v2 #1: the country/state border polylines are clipped to the BAKED
        # land rings, so NO border vertex is ever out over open water (the old
        # window-only clip let a coast-following border dangle into the ocean).
        # Every border/state vertex must be inside land OR on the coast (a
        # clipped vertex sits exactly on a land edge).
        def near_land(pt, rings, tol=0.06):
            if cb._point_in_land(pt, rings):
                return True
            x, y = pt
            for r in rings:
                for i in range(len(r)):
                    ax, ay = r[i]
                    bx, by = r[(i + 1) % len(r)]
                    dx, dy = bx - ax, by - ay
                    d2 = dx * dx + dy * dy
                    t = 0.0 if d2 == 0 else max(0.0, min(
                        1.0, ((x - ax) * dx + (y - ay) * dy) / d2))
                    cx, cy = ax + t * dx, ay + t * dy
                    if (x - cx) ** 2 + (y - cy) ** 2 <= tol * tol:
                        return True
            return False

        bm = cb.basemap_for(28.6, -93.6, "AL")   # TX/LA coast - dense borders
        land = bm["land"]
        self.assertTrue(land, "near-land bake must have land to clip against")
        for key in ("borders", "states"):
            off = [v for line in bm[key] for v in line
                   if not near_land(v, land)]
            self.assertEqual(off, [], f"{key}: {len(off)} verts off water")

    def test_clip_line_to_land_keeps_only_inland_part(self):
        # a unit square of "land"; a horizontal line crossing it keeps only the
        # inside run, dropping the parts that exit into "water".
        sq = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
        runs = cb._clip_line_to_land([[-5, 5], [15, 5]], sq)
        self.assertEqual(len(runs), 1)
        (x0, _), (x1, _) = runs[0][0], runs[0][-1]
        self.assertAlmostEqual(min(x0, x1), 0.0, places=6)
        self.assertAlmostEqual(max(x0, x1), 10.0, places=6)

    def test_coast_lines_have_at_least_two_points(self):
        # coastlines are drawn as open polylines (M..L.., no trailing Z).
        # A mainland-coast segment is open; an ISLAND coast is a naturally
        # CLOSED LineString (first == last) - both are valid, the renderer
        # just never appends a Z, so a closed island line still draws as a
        # loop. The contract is only ">= 2 points".
        bm = cb.basemap_for(11.2, -87.5, "EP")
        self.assertGreater(len(bm["coast"]), 0, "near-land bake has coast")
        for line in bm["coast"]:
            self.assertGreaterEqual(len(line), 2)

    def test_all_geometry_inside_window(self):
        for nm, lat, lon, basin in CASES:
            bm = cb.basemap_for(lat, lon, basin)
            la0, la1, lo0, lo1 = bm["window"]
            for layer in ("land", "coast"):
                for ring in bm[layer]:
                    for x, y in ring:
                        self.assertTrue(lo0 - 1 <= x <= lo1 + 1,
                                        f"{nm} {layer} lon {x} out of window")
                        self.assertTrue(la0 - 1 <= y <= la1 + 1,
                                        f"{nm} {layer} lat {y} out of window")

    def test_bakes_are_bounded_in_size(self):
        # distance-adaptive DP must keep even a worst-case near-land bake
        # to a sane size (a regression here = the basemap ballooning the
        # per-storm page). Gulf-of-Mexico landfall is the densest case.
        bm = cb.basemap_for(27.0, -90.0, "AL")
        raw = json.dumps(bm, separators=(",", ":"))
        # v2 #2: the budget rose 200k -> 225k when TOL_NEAR went 0.022 -> 0.018
        # (~18% finer coast so coast-following borders sit on the shore). The
        # worst real Gulf bake (Apalachee Bay) is ~212 KB, so 225 KB both fits and
        # still catches a regression. A per-storm cone page loaded on demand.
        self.assertLess(len(raw), 225_000,
                        f"Gulf bake {len(raw)} bytes - DP simplify regressed")

    def test_antimeridian_does_not_crash_or_leak(self):
        bm = cb.basemap_for(18.0, 178.0, "WP")
        self.assertEqual(bm["ocean"], "PACIFIC OCEAN")
        # no land/coast band should span the whole window (the torn-ring bug)
        for ring in bm["land"] + bm["coast"]:
            xs = [p[0] for p in ring]
            self.assertLess(max(xs) - min(xs), 65,
                            "geometry spans >65 deg lon - antimeridian tear")


class TestPolylineClipAndSimplify(unittest.TestCase):
    def test_clip_polyline_splits_into_disjoint_runs(self):
        # a line that dips BELOW the box and comes back up (all within the
        # box's x-range) must split into two disjoint in-window runs.
        line = [(1, 5), (3, 5), (3, -3), (6, -3), (6, 5), (9, 5)]
        runs = cb._clip_polyline(line, 0.0, 0.0, 10.0, 10.0)
        self.assertGreaterEqual(len(runs), 2,
                                "in/out/in polyline must split into runs")
        for run in runs:
            for x, y in run:
                self.assertTrue(-1e-6 <= x <= 10 + 1e-6)
                self.assertTrue(-1e-6 <= y <= 10 + 1e-6)

    def test_clip_polyline_fully_outside_is_empty(self):
        self.assertEqual(cb._clip_polyline(
            [(20, 20), (30, 30)], 0.0, 0.0, 10.0, 10.0), [])

    def test_simplify_collapses_a_straight_run(self):
        line = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]  # collinear
        self.assertEqual(cb._simplify(line, 0.01), [(0, 0), (4, 0)])

    def test_simplify_keeps_a_real_corner(self):
        line = [(0, 0), (2, 0), (2, 2)]                  # a right angle
        out = cb._simplify(line, 0.01)
        self.assertIn((2, 0), out)                       # the corner stays

    def test_adaptive_tol_is_finer_near_centre(self):
        # a ring grazing the storm centre keeps more detail than the same
        # shape parked at the window edge.
        import random
        rng = random.Random(0)
        wiggle = [[math.cos(t / 30) * 0.02 + 0.01 * rng.random(),
                   math.sin(t / 30) * 0.02] for t in range(400)]
        near = cb._simplify([[c[0], c[1]] for c in wiggle],
                            cb.TOL_NEAR)
        far = cb._simplify([[c[0] + 28, c[1]] for c in wiggle],
                           cb.TOL_FAR)
        self.assertGreater(len(near), len(far))


class TestMapsPassRenderMarkers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        storm = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.html = cyclolab_shell.render_page(storm, feed_url=FEED_URL,
                                              loader="")

    def test_gear_is_a_real_cog(self):
        # the settings glyph is the straight-toothed cog (fill-rule evenodd
        # hollow centre), NOT the old rounded flower/blob.
        self.assertIn('fill-rule="evenodd" d="M9.81,4.62', self.html)
        self.assertNotIn("M12 8a4 4 0 100 8", self.html)

    def test_basemap_canon_css(self):
        # LIGHT-GRAY land, no land stroke; WHITE coastline; the inner frame
        # is killed (stroke:none).
        self.assertIn(".ac-land { fill: #a7b2c4; stroke: none; }", self.html)
        self.assertIn(".ac-coast { fill: none; stroke: #ffffff;", self.html)
        self.assertIn(".ac-frame { fill: none; stroke: none; }", self.html)

    def test_coast_is_drawn_in_all_furniture_sites(self):
        # all THREE basemap sites (guidance gBasemap, mapFurniture track+swath,
        # and the cone inline) emit the white coast polylines.
        self.assertEqual(self.html.count('<path class="ac-coast" d="'), 3)

    def test_cone_is_blue_glass_with_solid_white_centerline(self):
        self.assertIn('<linearGradient id="cone-glass"', self.html)
        self.assertIn('fill="url(#cone-glass)"', self.html)
        # the centerline is no longer dotted
        self.assertNotIn('stroke-dasharray="2 5"', self.html)

    def test_track_connector_is_white_with_casing(self):
        # the maps-pass white connector: a dark casing (0.5 alpha) under the
        # white .tp-track line - the casing alpha (0.5) is distinct from the
        # cone centerline casing (0.55), so it pins the track connector edit.
        self.assertIn('stroke="rgba(9,22,42,0.5)"', self.html)
        self.assertIn('<path class="tp-track" d="\' + dline +\n', self.html)

    def test_now_icon_is_enlarged(self):
        # legibility: NOW dominates; R3 #1 bumped the forecast glyph to 0.98.
        self.assertIn("var scale = (i === 0 ? 1.9 : 0.98)", self.html)

    def test_country_borders_drawn_in_all_furniture_sites(self):
        # thin SLATE internal country borders (ne_10m boundary_lines), drawn in
        # all THREE basemap sites: the guidance gBasemap, mapFurniture
        # (track+swath), and the cone inline. Phase-4 C: slate, not white -
        # furniture recedes behind the subject layer.
        self.assertEqual(self.html.count('<path class="ac-border" d="'), 3)
        self.assertIn(".ac-border { fill: none; stroke: rgba(71,85,105,0.92)",
                      self.html)

    def test_state_borders_drawn_in_all_furniture_sites(self):
        # ne_10m admin_1 state/province lines, dimmer than the country border,
        # drawn UNDER it in all three basemap sites (guidance + track/swath +
        # cone). A dim SLATE .ac-state stroke; the maps degrade gracefully if an
        # old baked basemap lacks the 'states' key.
        self.assertEqual(self.html.count('<path class="ac-state" d="'), 3)
        self.assertIn(".ac-state { fill: none; stroke: rgba(71,85,105,0.60)",
                      self.html)

    def test_lockup_is_a_top_left_html_overlay(self):
        # R3 #2: the lockup is an HTML overlay PINNED to the panel's top-left
        # corner (the SVG is meet-scaled+centered, so an in-SVG lockup floats
        # inset); the redundant "FORECAST CONE" head is gone (panel <h3> head).
        self.assertNotIn('">FORECAST CONE</text>', self.html)
        self.assertIn("<h3>Forecast cone</h3>", self.html)
        self.assertIn('<div class="adv-lockup" id="advcone-lockup"', self.html)
        self.assertIn(".adv-lockup { position: absolute; top: 12px; "
                      "left: 12px;", self.html)
        self.assertIn('id="advcone-lockup-name"', self.html)

    def test_graticule_has_casing_and_four_edge_labels(self):
        # R3 #3: a dark-casing/halo graticule (reads over light land) with
        # labels on all four edges, drawn top-most.
        self.assertIn(".ac-graticule .grat-cas {", self.html)
        self.assertIn(".ac-graticule .grat-lin {", self.html)
        self.assertIn('class="grat-cas"', self.html)
        self.assertIn('class="grat-lab"', self.html)

    def test_reveal_is_arc_length_smoothed(self):
        # round-2 #7: a Catmull-Rom spline densifies the track before the
        # growth clip, so the front glides smoothly through curves.
        self.assertIn("function catmullRom(", self.html)
        self.assertIn("tpExt = catmullRom(tpExt, PER_SEG)", self.html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
