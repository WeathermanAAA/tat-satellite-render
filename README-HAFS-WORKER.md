# HAFS render worker (`hafs_render_poller.py`)

The 4th Railway service. A persistent, poller-driven worker that moves the full
HAFS model-plot ingest+render off the GitHub Actions cron (`update-hafs.yml` in
the main repo) onto dedicated compute, while keeping **every product fully
pre-rendered** (no lazy, no frontend change).

## What it does

One `PollerEngine` + one HAFS `Source` on `poller_framework` (the **intensity
poller** pattern, not the hand-rolled floater). Each poll:

1. resolves the newest **complete** HAFS cycle (cheap S3 listing) plus which of
   its `(model, storm, domain)` pairs are complete upstream (one exact-key list
   per pair). `change_key` is the cycle id **+ that pair set**, so the
   expensive render fires only on a new cycle **or** when the current cycle
   gains a newly-complete pair (the **intra-cycle catch-up** trigger).
2. on a new cycle, runs `python -m hafs_render.generate_hafs_plots --cycle ...`
   in a **subprocess** under a hard wall-clock **watchdog** (kills the render's
   whole process group on timeout). This is the *same code* the cron runs (the
   pinned `hafs-render` package), so output is byte-identical modulo cross-host
   raster noise.
3. uploads frames + manifest to R2 in the cron's **3 passes** (PNGs no-delete →
   manifest → prune `--delete` scoped to `*.png`), via the floater `R2` client
   (`put_bytes`/`put_json`/`list`/batched `delete`).
4. **INTRA-CYCLE CATCH-UP**: while the rendered cycle is still the newest, a
   pair that was absent or still uploading at render time (a late HAFS-B run, a
   late storm - the generator logs it in `skipped_pairs`) is rendered as soon
   as its terminal `f126` lands upstream: one *filtered*
   `--models/--storm/--domains` subprocess per `(model, storm)` group (same
   `HAFS_JOBS`/`HAFS_INGEST_JOBS`, so an incremental render can't OOM where a
   full one fits), then an **additive** upload - the group's PNGs (all-or-
   nothing barrier) + the manifest **merged** into the live one. Catch-up
   **never prunes** and **never re-renders a completed pair**; a failing group
   holds the spine signature (only the still-missing pairs retry next poll) -
   including the **exit-0 drop** case (the generator's exit-1 gate is whole-run
   `n_ok==0`, not per-pair, so a group/cycle can exit 0 while one pair produced
   zero frames; a gate-reopen guard raises after publishing what landed, since
   an already-complete pair gives upstream no change to flip the signature) -
   and a persistently-failing pair is abandoned after
   `HAFS_CATCHUP_MAX_ATTEMPTS` (visible in
   `render_summary.json` `skipped_pairs`; each success appends a `catchups`
   audit entry). `HAFS_CATCHUP=false` restores pure cycle-id gating.
5. writes `{prefix}/render_progress.json` throughout (so a long/wedged render is
   observable) and `{prefix}/poller_health.json` between cycles.

Anti-freeze: the spine has no `process()` timeout, so the subprocess watchdog is
what guarantees a wedged cycle self-aborts and retries; `STALE_AFTER_S` is set
above the watchdog so the health watcher never false-alarms mid-render.

## Progressive frame-load (HAFS_PROGRESSIVE, default ON)

The frame-granular generalization of the catch-up: `change_key = (cycle,
posted-frame set)` where a frame is "posted" once its `.atm` AND `.sat`
grb2+idx exist upstream. Each poll renders only the NEW frames (one filtered
`--models/--storm/--domains/--only-fxx` subprocess per (model, storm,
identical-fxx-set) group - exact, never re-renders), uploads them under
CYCLE-SCOPED keys (`{prefix}/{cycle}/...`, immutable), then publishes a
cycles[]-bearing manifest whose LEGACY fields keep describing the newest
COMPLETE cycle with its prefix baked into `path_template` (an old frontend
keeps working through deploy skew). A new cycle dir (storm_info, ~1.3 h before
f000) pre-announces with an empty in-progress entry. Completion = every posted
pair rendered through f126 -> legacy fields flip, the completion prune deletes
retired cycle prefixes + aged-out flat legacy keys. Frame-level attempts cap +
gate-reopen guard as in the pair path; the ledger bootstraps from the live
manifest on restart (no full re-render after a deploy).
`HAFS_PROGRESSIVE=false` = the classic complete-pair source, untouched.

## Deploy (Stage 2 = SHADOW)

Prereq (main repo, one-time): the `hafs_render` package must be importable as the
pinned git dep. Push branch `hafs-render-package` and tag `hafs-render-v0.1.0`
(commit that adds `hafs_render/`), or temporarily point the `requirements.txt`
`hafs-render @ git+...` ref at the branch.

1. New Railway service in this project, **Config as Code** = `railway.hafs.json`
   (`startCommand: python hafs_render_poller.py`, no port, no healthcheck).
2. Plan: **Pro, 8 vCPU** for this service. Set `HAFS_JOBS=8`.
3. Env: R2 creds (`R2_ENDPOINT` + `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY`),
   `HAFS_R2_PREFIX=shadow/models/hafs` (the default — **shadow**, not live), and
   the optional knobs in `.env.example`.
4. The shared `requirements.txt` now pulls `hafs-render` (GRIB stack +
   matplotlib/numpy pins) into the image — all services share the build.

The worker writes to `shadow/models/hafs/...`. The Actions cron keeps writing the
live `models/hafs/...` keys. **Nothing the frontend reads changes.**

## Shadow-verify (the Stage 2 gate)

Over a couple of real cycles, confirm:
- shadow PNGs match the live cron PNGs within tolerance (`shadow/models/hafs/...`
  vs `models/hafs/...`, same cycle) — the diff is pure cross-host raster noise,
- the worker renders within budget at 8 vCPU (compare to the ~60 min cron),
- `render_progress.json` ticks during the render and `poller_health.json` is
  healthy between cycles,
- the 3-pass prune is correct (no stale shadow frames).

## Cutover (Stage 3, separate, reversible — NOT in Stage 2)

Flip `HAFS_R2_PREFIX=models/hafs` (worker writes the live keys) and gate the
cron's render off in the main repo (`RENDER_HAFS_ON_CRON=false`). Reversible: flip
back. The worker writes the same keys/content, so `/models/` never blinks.
