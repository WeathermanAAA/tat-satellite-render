# CycloLab Phase-4 follow-up batch — inland W/W fills + map polish + 2 bug fixes, HELD for sign-off

**Status: REBUILT from spec after a codespace timeout lost the original
`ptc-phase4-ww-counties` branch (it was never pushed). On the branch
`ptc-phase4-ww-counties`; NOT on main, NOT deployed. HELD for Andrew's visual
sign-off.**

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

## Items (full detail filled in as each lands)

- **A — ww_zones inland fills:** _(pending)_
- **C — slate borders:** _(pending)_
- **E — Active-Systems parse + freshness guard:** _(pending)_
- **F — heal-debt:** _(pending)_

## Suite

_(filled in at the end)_

## Adversarial review

_(filled in at the end)_

## Stills

_(rendered desktop + mobile; filed under `review/phase4/`)_
