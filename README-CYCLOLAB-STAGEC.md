# CycloLab Stage C - invest pages (grey / red-X), HELD for sign-off

**Status: ART-DIRECTION REVIEW of the grey / red-X EXECUTION. On the branch
`cyclolab-invest-shells`; NOT on main, NOT deployed.** This unlocks the Stage-B
Guidance tab for the only systems active right now (invests).

## What this branch does
1. **Invests are page-able.** `storm_ids.parse_sid` now accepts invest sids
   (`{AL|EP|CP}{90-99}{YYYY}`) -> `is_invest=True`, empty `hafs_id` (invests never
   run HAFS). 50-89 stay rejected (ATCF gap). The page lifecycle
   (`cyclolab_pages`) births / refreshes / freezes invest pages through the SAME
   path as named storms - no lifecycle code change was needed once parse_sid
   accepts them. Recycle-safe: when an invest upgrades to a named storm it leaves
   the active set and its page freezes to the ENDED archive ("shared links never
   die"); the named sid is a different page. `cyclolab_advisories` now explicitly
   skips invests (no NHC advisory product).
2. **Grey / red-X identity (the art-gated treatment).** `render_page` detects an
   invest (`ids.is_invest`) and adds `data-invest` to `<html>`, which OVERRIDES the
   category vars to GREY (`--cat-ramp/--cat-accent/--cat-ink`) - so the banner,
   vitals, chips, nav, everything keyed on `--cat-*` reads grey with NO per-element
   edits. The spinning cyclone glyph is replaced by a GIANT RED X (`.invest-x`,
   CSS-toggled); the type word is "INVEST AREA"; the category chip + ACE + Next-
   advisory vitals rows are hidden. The spin glyph + category colors are untouched
   for named/active TCs.
3. **Subset sections.** Invests have no official NHC products, so the cone +
   advisories + HAFS-models sections AND their nav buttons are CSS-hidden
   (`html[data-invest] #sec-models/#sec-advisories/[data-sec=...]`). The invest page
   shows: **Overview** (grey vitals + SST hero + track), **Satellite** (invest
   floater), **Guidance** (Stage B - invests already have guidance.json + ships.json).
   Clean empty states where a section has no data (SST not yet rendered, wind-swath
   needs radii, etc.).

## Verified (routed-fixture Playwright on the real shell, EP93 invest)
- Grey banner + red X + "INVEST AREA / 93E"; grey vitals (ACE + Next-advisory
  hidden, position/last-fix populated).
- Nav subset to **Overview / Satellite / Guidance** (Models + Advisories hidden).
- Guidance tab renders the full Stage-B tracks / intensity / SHIPS.
- Ended-freeze state renders the grey/red-X archive ("THIS STORM HAS ENDED").
- Desktop + mobile. Named-storm pages UNCHANGED (no `data-invest`, `IS_INVEST=false`).
- Tests: `test_storm_ids` (invest parse), `test_cyclolab_pages` (invest page keys),
  `test_cyclolab_page_writer` (invest rebirth), `test_cyclolab_shell`
  (`test_invest_gets_grey_redx_subset_page`), `test_cyclolab_advisories` (skip).

## NHC formation-chance pill (invest banner)
The invest banner carries a colour-coded **formation-chance pill** - the genesis
odds, the headline invest metric - eagerly (not behind a tab). Source = the NHC
**Tropical Weather Outlook** (`https://www.nhc.noaa.gov/xml/TWO{AT|EP|CP}.xml`),
parsed by `cyclolab_guidance.parse_two` (matches the invest ref `(EP93)`/`(AL90)`,
pulls the 48-hour + 7-day chances). The guidance poller fetches the TWO once per
basin per poll and writes `cyclolab/{sid}/formation.json` for each invest; the shell
`loadFormation()` reads it. Colours are the canonical NHC low/medium/high buckets,
which match the spec exactly: **<=30% yellow, 40-60% orange, >=70% red** (10%-step
values never fall in the 31-39 / 61-69 gaps). Live: EP93 20% yellow, AL90 60% orange.
The pill shows both windows (7-day headline + 48h), coloured by the higher level;
hides cleanly when the TWO has no area for the invest.

## GO-LIVE companion (ship WITH the merge, after sign-off)
The map's invest-X markers are still **non-launchable** by design: making them link
to `/cyclolab/{sid}/` BEFORE the invest pages are live would create dead links. So
the one remaining wiring is a **main-repo** (`Triple-A-Tropics`) change to
`generate_tracks_plot.py`, to ship in the SAME release that merges this branch:
  * the popup "Open in CycloLab" button is gated `isActive && !isInvest` (~line 2420)
    -> allow invests;
  * wrap the invest-X marker (`.invest-x-marker`, Python `render_tracks_svg` +
    the byte-identical `LIVE_BASIN_JS` mirror, ~line 3223) in the same
    `<a href="/cyclolab/{sid}/" target="_blank">` the hurricane glyph uses;
  * keep `test_live_overlay_parity` + `test_invest_x_anchor` green (the X must stay
    centred on its fix).
Held here because it is cross-repo + parity-constrained + must not precede the pages.

## Render the review locally
```
python3 -c "import cyclolab_shell,json,urllib.request as u; \
  f=json.loads(u.urlopen(u.Request('https://cdn.triple-a-tropics.com/feeds/ep_tracks_data.json',headers={'User-Agent':'x'})).read()); \
  s=dict([x for x in f['storms'] if x['sid']=='NHC_EP932026'][0]); s['is_active']=True; \
  open('/tmp/invest.html','w').write(cyclolab_shell.render_page(s,feed_url='x'))"
# then screenshot /tmp/invest.html with the guidance/ships fetch routed to live R2.
```
Stills under `_review_out/stills/INVEST_*.png`.
