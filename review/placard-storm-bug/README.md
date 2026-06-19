# Cone forecast-point placard — storm-bug restyle · OPTIONS BOARD (HOLD)

TWEAK 1B item 2 — **ART-GATED**. Three "storm-bug" pill treatments for the cone
forecast-point placards, for sign-off. **Nothing here is deployed**: the live
cone placard markup (`cyclolab_shell.py` `renderAdvCone`) is untouched, and this
branch is held — not merged.

Board: `placard_board.png` (rendered by `generate_board.py`, faithful to the
production glyph `HURRICANE_PATH`, palette `ace_core.SSHS_COLORS`, pill glass
ramp `CAT_TOKENS`, Metropolis, tabular-nums, light ink, the `lab-spin` glyph).

| | Treatment | Idea |
|---|---|---|
| **A** | Broadcast chip | Category-glass pill (TV lower-third). Badge-left glyph + stacked time / wind. Boldest, most on-air. |
| **B** | Stat card | Dark glass card + SSHS accent rail/top-bar. Glyph + category eyebrow, big wind, muted time. Data-forward, restrained. |
| **C** | Minimal bug | Compact dark inline pill, SSHS-edged, accent glyph + one tight row. Smallest footprint — kindest to a crowded cone. |

All three: keep the **leaderless, collision-aware** placement (no connector
lines — a standing call, NOT restyled), reuse the **spinning category glyph**,
colour-code by **SSHWS** (canonical palette), keep ink **light**, show wind in
**mph** (production stays kt/mph-toggle aware via `windDisp`/`windUnitLabel`),
and label time from each point's **valid_utc** ("Wed 7 PM", UTC).

Once Andrew picks, the chosen treatment is wired into `renderAdvCone`'s placard
markup block (pill geometry `pw/ph` + the emitted `<g data-role="placard">`),
reusing the existing placement sweep + a new UTC weekday/hour formatter for
`p.valid_utc`. Regenerate: `python review/placard-storm-bug/generate_board.py`.
