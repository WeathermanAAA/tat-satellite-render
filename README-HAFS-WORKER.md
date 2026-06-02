# HAFS render worker (`hafs_render_poller.py`)

The 4th Railway service. A persistent, poller-driven worker that moves the full
HAFS model-plot ingest+render off the GitHub Actions cron (`update-hafs.yml` in
the main repo) onto dedicated compute, while keeping **every product fully
pre-rendered** (no lazy, no frontend change).

## What it does

One `PollerEngine` + one HAFS `Source` on `poller_framework` (the **intensity
poller** pattern, not the hand-rolled floater). Each poll:

1. resolves the newest **complete** HAFS cycle (cheap S3 listing). `change_key`
   is that cycle id, so the expensive render fires **only on a new cycle**.
2. on a new cycle, runs `python -m hafs_render.generate_hafs_plots --cycle ...`
   in a **subprocess** under a hard wall-clock **watchdog** (kills the render's
   whole process group on timeout). This is the *same code* the cron runs (the
   pinned `hafs-render` package), so output is byte-identical modulo cross-host
   raster noise.
3. uploads frames + manifest to R2 in the cron's **3 passes** (PNGs no-delete →
   manifest → prune `--delete` scoped to `*.png`), via the floater `R2` client
   (`put_bytes`/`put_json`/`list`/batched `delete`).
4. writes `{prefix}/render_progress.json` throughout (so a long/wedged render is
   observable) and `{prefix}/poller_health.json` between cycles.

Anti-freeze: the spine has no `process()` timeout, so the subprocess watchdog is
what guarantees a wedged cycle self-aborts and retries; `STALE_AFTER_S` is set
above the watchdog so the health watcher never false-alarms mid-render.

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
