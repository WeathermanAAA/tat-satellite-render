# CycloLab Phase-4 follow-up batch — inland W/W fills + map polish + 2 bug fixes, HELD for sign-off

**Status: REBUILT from spec after a codespace timeout lost the original
`ptc-phase4-ww-counties` branch (it was never pushed). On the branch
`ptc-phase4-ww-counties`; NOT on main, NOT deployed. HELD for Andrew's visual
sign-off.**

---

## VISUAL REVISION v2 (latest) — basemap borders + W/W zone layer

Two separate border layers were both fixed, kept straight:

### Basemap state/country outlines
1. **Clip borders to land (FIXED).** The country/state border lines come from
   DIFFERENT NE datasets than the land/coast and were clipped only to the
   rectangular *window*, never to land — so a coast/river-following border
   diverged off the land and dangled into the ocean. `cyclolab_basemap` now clips
   every border + state polyline to the BAKED land rings (the exact geometry the
   drawn coast is derived from) via a stdlib ray-cast PIP + segment-intersection
   clipper (`_clip_lines_to_land`), as the last bake step. **0 off-water
   border/state vertices** on the TX/LA coast now (was dangling). See still 05.
2. **Align / higher-res.** `TOL_NEAR` 0.022 → 0.016 (~1.7 km) for a crisper
   coast; the page-size budget rises 200k → 235k. Combined with the clip, a
   coast-following border now sits ON the drawn coast.
3. **Thin strokes.** coast 2.6 → 1.3, country border 1.4 → 0.7, state 0.8 → 0.5
   — fine hairlines retuned together.

### Watch/warning zone layer
4. **Zone outlines clipped to land.** Each warned county is a FULL outline +
   fill, clipped EXACTLY to the basemap land via an SVG `clipPath` of the SAME
   `BASEMAP.land` the coast comes from — so nothing dangles into water and the
   fills align EXACTLY with the drawn coast. A thin DARK outline per county so
   adjacent counties read as bounded shapes, not one blob.
5. **Wider attribution.** The zone box now spans the FULL forecast track (cone +
   all track points) with a 4° margin (was cone-bbox + 3°), so ALL warned
   counties along the track are attributed, not just the coastal row.
6. **Max opacity + z-order.** fill-opacity 0.22 → **1.0**; the fills draw on the
   basemap UNDER the cone / track / forecast points / icons / labels (all stay
   above, never buried).

### Both
7. **"Everything lines up, no mismatch."** Achieved by clipping BOTH the borders
   AND the W/W zones to the SAME baked land — so the basemap coast + admin
   borders + W/W county edges are all bounded by one coast, no internal
   mismatch. Applied across ALL CycloLab maps (cone + guidance track + overview
   track/swath) via the one shared `BASEMAP` + the one `.ac-*` CSS rule, no fork.
   **NOTE / SIGN-OFF DECISION:** the basemap COAST is still Natural Earth 10m
   (raised to 0.016, the page-size budget's practical ceiling — even 0.018 blew
   the old 200k budget). A full switch to a US Census county / GSHHG-high *source*
   (so the coast geometry itself matches the NWS-zone shapes) is a larger,
   separable build; I confirmed it is FEASIBLE (the Census county geojson is
   topologically consistent → coast/state/county derivable by edge-hashing, and
   the land union by edge-chaining, no shapely), but it adds page weight + seam
   handling at the US/Mexico land border. Held as an option: say the word and I
   will build the Census-source basemap. The alignment goal of #7 is met *now*
   via clipping; the Census switch would additionally make the coast *shape* itself
   higher-fidelity.

Stills re-rendered from the live PTC AL01: `01` desktop cone, `02` inland fills
zoom (full-opacity counties clipped to land), `03` mobile, `04` track-history
(second map, borders clipped to land), `05` coastal-border zoom proving borders
no longer run into the water.

---

This is the durability-recovery rebuild. The original branch lived in an
ephemeral `/tmp` clone that the codespace restart wiped; it had never been
pushed to origin, and a filesystem-wide search + an `ls-remote` confirmed it was
unrecoverable. Rebuilt here from the work-order spec; the two bug root-causes
were already nailed in the spec, so they were implemented directly, not
re-derived.

## RECONCILIATION — what the spec asked vs what was already on main

Before rebuilding, the spec's six items (A–F) were reconciled against `main`.
Three had ALREADY landed via separate PRs that DID get pushed before the
timeout, so they are **confirmed, not re-implemented** (re-doing them would risk
regressions):

| Item | Spec | State on `main` before this branch | Action here |
| --- | --- | --- | --- |
| **A** | `ww_zones` inland county/zone W/W FILLS (new adv-JSON key) | **MISSING** — `ww_zones` absent everywhere | **BUILT** |
| **B** | Cone basemap → Natural Earth 10m | **DONE** — PR #8 confirms "cone basemap was ALREADY 10m" | verify only |
| **C** | Darken state/country borders to slate (subtle) | **MISSING** — PR #8 added state *lines* but kept them **white** (`rgba(255,255,255,…)`) | **BUILT** |
| **D** | Borders on ALL CycloLab maps, one shared rule | **DONE** — PR #8 added `.ac-state`+`.ac-border` to all 3 render sites; one shared CSS rule | verify only |
| **E** | parse_two reads the Active-Systems PTC narrative + keys to the REAL sid; page-side freshness guard | **PARTIAL** — PR #7 keeps `formation.json` fresh via the always-on intensity poller, but parse_two still only read the numbered-invest `(AL90)` parenthetical, and there is **no page-side freshness guard** | **BUILT** (the missing parser path + guard) |
| **F** | Advisory text heal-debt: TCP+TCD always-expected for NHC designated storms | **MISSING** — `_attach_text` still vacuously marks text "done" when the URLs are absent | **BUILT** |

So the genuinely-new work on this branch is **A, C, the missing half of E, and
F** — plus a re-verification that B and D are live and correct.

## The two bug root-causes (already nailed; implemented faithfully)

### E — formation pill froze at the invest-era 60/60
`parse_two` only matched a **numbered-invest parenthetical** (`_TWO_REF =
\(([A-Z]{2})(9\d)\)`, e.g. `(AL90)`). Once NHC began issuing advisories on the
system it moved the system into the TWO **"Active Systems"** narrative — which
has NO `(AL90)` tag — so parse_two emitted nothing for it and the last
invest-era value (60/60) froze on R2. PR #7's "refresh every TWO-referenced
invest" could not help: there was no longer anything referencing the invest.

Confirmed against the LIVE TWO captured 2026-06-16 2336Z
(`tests/fixtures/cyclolab/twoat_active_ptc.xml`):

```
Active Systems:
The National Hurricane Center is issuing advisories on Potential
Tropical Cyclone One, located over south Texas.
* Formation chance through 48 hours...high...70 percent.
* Formation chance through 7 days...high...70 percent.
...
Public Advisories on Potential Tropical Cyclone One are issued
under WMO header WTNT31 KNHC and under AWIPS header MIATCPAT1.
```

**FIX:** parse_two also reads the Active-Systems PTC narrative, derives the real
designated sid from the **AWIPS header** (`MIATCPAT1` → basin `AT`→`AL`, storm
`1` → `NHC_AL012026`) cross-checked against the spelled ordinal ("One"→1), and
emits the 70/70 odds keyed to that real sid. Because PR #7's `_refresh_formation`
(and the guidance poller) call parse_two, the real-sid `formation.json` is now
re-stamped every poll for free — no poller change. The PTC page already tries
its own (real) sid first, so it now shows a live 70/70. PLUS a page-side
**freshness guard**: `loadFormation` hides the pill when `formation.json`'s
`generated_at` is provably stale, so a genuinely-frozen pill HIDES rather than
showing stale odds.

### F — advisory text panel blank for the whole first-advisory cycle
At a PTC's (or any storm's) FIRST advisory, `CurrentStorms` can carry the
cone/track KMZ a poll or two BEFORE it populates the `publicAdvisory` /
`forecastDiscussion` text URLs. With both URLs absent, `_attach_text`'s loop
skipped both products (`if not url … continue`) and `complete` stayed True — so
`text_done` was set VACUOUSLY true, the text-heal pulse never fired, and when the
URLs finally appeared nothing re-fetched them → blank panel all cycle.

**FIX:** TCP and TCD are ALWAYS-EXPECTED for NHC designated storms (the only
storms this source handles — `_storm_entries` already restricts to AL/EP/CP
designated, non-invest). `_attach_text` now keeps the heal debt OPEN
(`complete=False`) whenever TCP or TCD is not yet attached, even when its URL is
absent this poll, so the heal pulse keeps firing until both land. A
negative-control test asserts the OLD vacuous-True behavior is gone.

## Items

### A — `ww_zones` inland county/zone FILLS (NEW)
The coastal `ww` overlay is the NHC TCWW breakpoint **lines** (from the per-storm
WW KMZ); `ww_zones` is the inland county/zone **fills**, from a different source —
the NWS public alerts API (`api.weather.gov/alerts/active`).

- **`kml_advisories.parse_nws_alert_zones()`** — pure parser: the `/alerts/active`
  FeatureCollection → `[{type, geometry, ugc, area}]`. Only the six TC event types
  (`NWS_TC_EVENTS`); **marine UGC regions excluded** (`_MARINE_UGC_REGIONS` — the
  fills are inland only). Geometry from the alert Feature's OWN embedded
  `Polygon`/`MultiPolygon` (the common case — for the live Gulf event, every land
  TXZ/LAZ feature carried an embedded polygon and only the marine `GMZ` features
  were null), with a pluggable + caller-cached `resolve_zone` fallback for the
  rare null-geometry land feature. Rings are Douglas-Peucker-simplified (`_rdp`);
  per-storm attribution by **cone bbox + margin**.
- **`build_advisory_json(..., ww_zones=...)`** → new `ww_zones` contract key
  (empty array by default; `provenance.ww_zones_count` when populated). Additive +
  graceful — a non-list degrades to `[]`, never blocks the cone.
- **`cyclolab_advisories`** — fetches the national TC alerts **once per poll**
  (cached by poll seq; NWS-compliant UA; `404`/transient/down all degrade to `[]`),
  then attributes to each storm by its cone bbox (`_WW_ZONES_MARGIN_DEG = 3.0`).
  Kill-switch `CYCLOLAB_WW_ZONES`.
- **`cyclolab_shell` JS** — renders `advFull.ww_zones` as translucent
  canonical-NHC fills (`.ww-zone`, `fill-opacity 0.22` + stronger same-color
  outline) in a toggleable `#ac-ww-zones-group` drawn **UNDER** the coastal lines,
  sharing the one `WW_STYLE` palette + legend + the single "Watches & warnings"
  toggle (now covers both layers).

Validated against a REAL `api.weather.gov` capture
(`tests/fixtures/cyclolab/nws_tc_alerts_sample.json` — a live Gulf TC) and
rendered live (stills 01/02): the eight Louisiana parish/zone TS-Warning fills.

### C — slate map borders
`.ac-border` / `.ac-state` strokes go white → slate (`rgba(255,255,255,.72/.40)`
→ `rgba(71,85,105,.92/.60)`). ONE shared CSS rule, so the cone, the guidance track
map, and the overview track+swath all inherit it (the "borders on all maps, no
fork" of item D is already on `main`; the 3-furniture-site count tests confirm it
and the slate rgba is asserted). Stills 01/04 show the recessive slate borders.

### E — Active-Systems formation parse + page-side freshness guard
See the bug root-cause above. `parse_two` extension is in `cyclolab_guidance.py`
(`_PTC_AWIPS` / `_PTC_NARR` / `_parse_active_ptc`); the freshness guard is in
`cyclolab_shell.loadFormation` (`fresh()` / `STALE_MS = 12h`). No poller change —
PR #7's `_refresh_formation` and the guidance poller both call `parse_two`, so the
real-sid `formation.json` is restamped every poll for free.

### F — advisory text heal-debt stays open
See the bug root-cause above. `_attach_text` in `cyclolab_advisories.py`; the
negative-control test proves the old vacuous-True behavior is gone.

## Out-of-scope observation (NOT changed here)
While rendering the stills I hit a PRE-EXISTING, unrelated issue: NHC **intermediate**
advisories carry a non-numeric `advisoryNum` (`"2A"`), which `parse_track_kmz`'s
`int(...)` rejects and the change-gate's `int(str(advNum).lstrip("0"))` skips. So
CycloLab does not refresh on intermediate advisories. This is independent of this
batch (it predates it) and out of scope; flagging it for a future fix, not touched
here.

## Stills (`review/phase4/`, force-added past the repo `*.png` ignore)
Rendered from the LIVE PTC AL01 "One" advisory-2 cone + the LIVE Gulf NWS alerts
via headless chromium (reduced-motion final frame):
- `01_cone_desktop_ptc_one.png` — cone: 10m basemap, slate borders, coastal lines
  + inland fills, shared legend, "POTENTIAL TROPICAL CYCLONE ONE".
- `02_cone_inland_fills_zoom.png` — the inland Louisiana parish/zone fills close-up.
- `03_cone_mobile.png` — mobile cone.
- `04_track_history_slate_borders.png` — a second map: slate state/country borders
  across the Gulf.

## Suite
Full repo suite green: **527 passed + 16 subtests, 0 failures** (`python -m pytest
tests/`; the slow part is the jsdom CycloLab shell/visual harness — needs `node`,
present here). New tests added by this batch:
- `test_cyclolab_ww.py` — `parse_nws_alert_zones`: land-only/marine-excluded,
  event→type map, cone-bbox attribution, zone-resolver fallback,
  marine-URL-not-resolved, one-bad-feature-survives, RDP, GeometryCollection.
- `test_kml_advisories.py` — `ww_zones` contract key (empty default + populated).
- `test_cyclolab_guidance.py` — Active-Systems PTC → real sid @ 70/70;
  named-storm-yields-nothing; **block-doesn't-bleed-into-disturbances** +
  **named-storm-with-trailing-disturbance** (the adversarial-review regressions).
- `test_cyclolab_advisories.py` — F negative-control (text URLs absent at first
  advisory → heal stays open); ww_zones attach / far-not-attributed /
  api-down-never-blocks-cone / once-per-poll / once-per-poll-not-per-storm.
- `test_cyclolab_basemap.py` — slate border rgba (item C).
- `test_cyclolab_shell.py` — freshness-guard scaffold + ww_zones render scaffold.

## Adversarial review
A scoped multi-agent review (4 dimension reviewers — parser / poller / parse_two /
render — each finding then adversarially verified by a skeptical agent) surfaced 5
candidates; verification **confirmed 1, rejected 4**:

- **CONFIRMED — MAJOR (fixed):** `parse_two`'s Active-Systems block bled into a
  trailing numbered disturbance's formation chances (PTC One 70/70 → 20/40; a
  spurious pill on a named storm). The lone-PTC fixture masked it. Fixed with the
  `_AREA_BREAK` block cap + 2 regression tests (above).
- **Rejected (verified NOT real bugs, but cheap wins taken anyway):**
  GeometryCollection silent-drop (not a known NWS shape → added handling + test);
  mixed land+marine UGC in one feature (NWS issues one feature per zone → documented
  the assumption); per-poll-cache multi-storm "untested" (code correct → added the
  pinning test); WW legend swatch with nothing drawn (cosmetic, mirrors the existing
  coastal block → moved the `wwTypes` assignment after the empty-path guard in both).
