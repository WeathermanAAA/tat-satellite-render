# CycloLab guidance Stage B - renderers (REVIEW, HELD)

**Status: ART-DIRECTION REVIEW. NOT deployed, NOT merged to the live flow.** This
branch holds the three guidance RENDERERS for sign-off. The contested track color
scale is rendered THREE ways (options board) for Andrew to pick - not self-chosen.

Three hand-rolled SVG renderers in the CycloLab house style, hydrating from the live
Stage-A JSON (`cyclolab/{sid}/guidance.json`, `ships.json`):

1. **Model Forecast Tracks** - `track_aids` spaghetti on the cone basemap (reused
   `cyclolab_basemap.basemap_for`), 5-deg graticule, forecast-hour labels at
   0/24/48/72/96/120 along the TVCN consensus, consensus aids drawn heavier + cased,
   PEAK-WIND colorbar key card, aid legend. Strands colored by each track's peak wind.
2. **Model Forecast Intensity** - each `intensity_aid`'s Vmax vs forecast hour; SSHWS
   category bands (the cone's thresholds); IVCN consensus = white heavy, hi-res
   (HFAI/HFBI/HWFI/HMNI) colored, statistical (DSHP/LGEM/SHIP) thin dashed, coarse
   global (AVNI) thin neutral.
3. **SHIPS Output Diagram** - env-diagnostic small-multiples (shear, SST, RH, POT INT,
   OHC, 200 DIV, storm speed, Vmax, theta-e dev), the RI probability matrix table, an
   annularity (AHI) + prelim-RI header. `available:false` -> a clean "SHIPS unavailable".

## Options board (Andrew's pick)
The track color scale, three ways, with an illustrative intensity ramp (30-150 kt) so
the scales are distinguishable even on a weak invest:
  * **A** cyclonicwx kt-rainbow
  * **B** TAT SSHWS category (`ace_core.SSHS_COLORS`, discrete 7)
  * **C** WIND_TIER blue->gold (the locked house wind palette; current default)

## Reuse, no forks
* basemap geometry  -> `cyclolab_basemap.basemap_for`
* category palette   -> `ace_core.SSHS_COLORS` / `sshs_class` (single source)
* intensity bands    -> the cone's SSHS thresholds [34,64,83,96,113,137]
* projection / graticule -> the cone's `fitProjection` / `graticule` math
* house tokens/font  -> mirrored from `cyclolab_shell` CSS (Metropolis, navy, tnum)

On go-live (after sign-off), the JS renderers port into `cyclolab_shell.render_page`
(client-JS hydration, same pattern as the cone) and the WIND_TIER / SSHS constants get
single-sourced rather than mirrored.

## Render the review locally
```
python cyclolab_guidance_review.py NHC_EP932026 93E     # -> _review_out/review_*.html
python cyclolab_guidance_review.py NHC_AL902026 90L
python make_mocks.py                                    # fresh-invest + SHIPS-off states
PLAYWRIGHT_BROWSERS_PATH=~/.cache/ms-playwright python shoot_review.py   # desktop+mobile stills
python -m unittest tests.test_cyclolab_guidance_review  # developed / fresh / unavailable
```

Files: `cyclolab_guidance_review.{py,js,css}`, `make_mocks.py`, `shoot_review.py`,
`tests/test_cyclolab_guidance_review.py`. Stills under `_review_out/stills/`.
