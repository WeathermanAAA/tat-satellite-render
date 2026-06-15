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
