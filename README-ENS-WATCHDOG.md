# Ensemble Cyclone Centers — freshness watchdog (`ens_watchdog.py`)

Robust TRIGGER for the `/models/` enscenters ingest, replacing sole reliance on
GitHub's scheduled crons (which GitHub silently drops/throttles under load — the
recurring cause of a model getting stuck a cycle behind).

**What it does.** Every ~20 min it reads the public enscenters manifest from R2 and,
per model, compares the published `latest` cycle to the newest 6-hourly cycle that
*should* be available by now (cycle time + a per-model delivery lag). If a model is
behind, it fires that model's GitHub Actions workflow via the API
(`workflow_dispatch` with a **blank cycle** = the never-miss backfill, which
advances). Heavy ingest stays on free Actions; the watchdog only triggers.

It is deliberately dumb + safe: it pokes on a wall-clock heuristic and the
**never-miss run does the real completeness gate** (a too-early poke just no-ops and
retries). Idempotent, with a 40-min per-model cooldown so it never re-pokes a run
that's still in flight. `workflow_dispatch` is an API trigger, not a scheduled event,
so it isn't subject to the cron throttling that causes the staleness.

## Also watches HAFS (`hafs_run_once`)

The same service also watches the `/models/` **HAFS** plots, which are rendered by
the tat-satellite-render HAFS render worker (`hafs_render_poller.py`). When that
worker wedges or OOM-crashes, Railway's `restartPolicy=ON_FAILURE` cannot restart a
*frozen* process and gives up after `maxRetries`, so the HAFS manifest freezes a
cycle behind with **no recovery** (the ~30 h staleness on 2026-06-17 — worker died
~60% through the 06-17 06Z render and never came back). `hafs_run_once` reads the
public HAFS manifest each tick and fires `update-hafs.yml` when the newest cycle is
**behind** the expected cycle (with active storms — off-season is left quiet) **or**
is **stuck `in_progress`** longer than `HAFS_WATCHDOG_STUCK_BUILD_S` (a wedged build).
`update-hafs.yml`'s render is gated to **always run on a `workflow_dispatch`** even
when the scheduled cron is off (`RENDER_HAFS_ON_CRON=false` while the worker is
primary) — so recovery runs on free Actions, **independent of the wedged worker**,
and the completed legacy manifest clears any stuck "building" pill. Knobs:
`HAFS_WATCHDOG_LAG_H` (8), `HAFS_WATCHDOG_STUCK_BUILD_S` (3600),
`HAFS_WATCHDOG_COOLDOWN_S` (2400). It reuses the same `ENS_WATCHDOG_GH_TOKEN`.

## Deploy (Railway)

1. New Railway service in this repo using **`railway.watchdog.json`**
   (`startCommand: python ens_watchdog.py`).
2. Set the env var **`ENS_WATCHDOG_GH_TOKEN`** = a GitHub PAT (fine-grained:
   `actions: read+write` on `WeathermanAAA/Triple-A-Tropics`; or a classic token with
   `repo`+`workflow`). **Without it the watchdog runs but only logs decisions — it
   does not dispatch.** This is the one manual step.
3. Optional env: `ENS_WATCHDOG_INTERVAL_S` (default 1200), `ENS_WATCHDOG_REPO`
   (default `WeathermanAAA/Triple-A-Tropics`), `ENS_WATCHDOG_DRYRUN=1` (decide+log,
   never dispatch — good for a first staging run).

To co-host instead of a separate service, import `ens_watchdog.run_once(...)` into an
existing always-on poller's loop on a ~20-min cadence.

Tests: `python -m unittest tests.test_ens_watchdog` (offline; manifest + dispatch
injected).
