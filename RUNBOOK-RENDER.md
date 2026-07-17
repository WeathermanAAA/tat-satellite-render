# RUNBOOK-RENDER — the ex-Railway stack on the box

Railway paused all 6 tsr services at the $150 compute cap (2026-07-11/12).
This stack replaces them on the Hostinger box (KVM 8, Ubuntu 24.04), reusing
the exact S2/meso box pattern: one compose project, shared `.env`, no host
pip, no cred paste. Files: `docker-compose.render.yml` + `Dockerfile.render`
+ `Caddyfile.render` (all on `main`).

What runs where after this migration:

| ex-Railway service | box service | notes |
| --- | --- | --- |
| web `/render`+`/export` (railway.json) | `render` | uvicorn on 8080, TLS via `caddy` |
| floater poller (railway.worker.json) | `floater-poller` | `RENDER_BASE_URL=http://render:8080` |
| intensity poller (railway.intensity.json) | `intensity-poller` | owns `global_storms.geojson` |
| guidance poller (railway.guidance.json) | `guidance-poller` | |
| ens watchdog (railway.watchdog.json) | `ens-watchdog` | GH dispatch needs `ENS_WATCHDOG_GH_TOKEN` |
| hafs worker (railway.hafs.json, branch `hafs-render-worker`) | `hafs-worker` | **profile-gated OFF** — ~23 GB peak; GH `update-hafs.yml` stays the HAFS renderer until a dedicated box exists |

---

## 0. Secrets — what the box `.env` already has vs. needs

Already in the box `.env` (shared with s1/meso/s2): `R2_ENDPOINT`,
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
(+ `EUMETSAT_CONSUMER_KEY/SECRET` if Q13 landed).

**NEW keys to add** (append to the same `.env`):

```bash
# GitHub PAT, actions:read+write on WeathermanAAA/Triple-A-Tropics ONLY --
# lets ens-watchdog fire the update-hafs/enscenters fallback dispatches.
# WITHOUT it the watchdog still runs but only logs (safe to defer).
ENS_WATCHDOG_GH_TOKEN=...

# OPTIONAL (Q12, already queued): GES DISC Earthdata for the MergIR archive
# tier of /render. Absent => MergIR requests fail cleanly, GridSat fallback
# stays live. Either the token OR user+pass form:
# EARTHDATA_TOKEN=...
#   -- or --
# EARTHDATA_USERNAME=...
# EARTHDATA_PASSWORD=...
```

## 1. DNS (one Cloudflare edit, before or after bring-up)

Add an `A` record `render.triple-a-tropics.com` → the box public IP,
**DNS-only (grey cloud)** — Caddy provisions Let's Encrypt via HTTP-01 and
an orange-cloud proxy breaks the challenge in this simple setup. Ports 80 +
443 must be open on the box firewall (`ufw allow 80,443/tcp` if ufw is on).

The frontend already points at `https://render.triple-a-tropics.com`
(landed together with this runbook), so custom zooms / Time Machine /
deep-archive objfix / mp4 export light up the moment TLS resolves.

## 2. Bring-up (the box session)

```bash
cd ~/tat-satellite-render        # or wherever the existing clone lives
git fetch origin && git checkout main && git pull

docker compose -p tat-render -f docker-compose.render.yml build
docker compose -p tat-render -f docker-compose.render.yml up -d
docker compose -p tat-render -f docker-compose.render.yml ps
```

`up -d` starts caddy + render + the three pollers + the watchdog. The
`hafs-worker` service does NOT start (profile-gated) — see §5.

## 3. Verify

```bash
# 1. API healthy (local, no TLS involved):
curl -fsS http://127.0.0.1:8210/health | head -c 400; echo

# 2. TLS edge (after DNS propagates; first hit may take ~30 s while Caddy
#    fetches the cert):
curl -fsS https://render.triple-a-tropics.com/health | head -c 400; echo

# 3. Pollers alive:
docker compose -p tat-render -f docker-compose.render.yml logs --tail 20 \
    floater-poller intensity-poller guidance-poller ens-watchdog

# 4. Floaters actually producing (fresh as_of within ~15 min):
curl -fsS https://cdn.triple-a-tropics.com/floaters/manifest.json | head -c 400; echo

# 5. Browser check: triple-a-tropics.com/satellite/ -> draw-a-box render;
#    /satellite/explorer/ -> Time Machine. Both were HARD DEAD on Railway.
```

## 4. Retire the GH-Actions floater stopgap (after §3 passes)

`.github/workflows/floater-worker.yml` in Triple-A-Tropics duplicates the
poller fleet every 30 min. It is idempotent alongside the box (per-frame
content-hash dedup), so there is NO rush — but once box `/render` + pollers
are verified, disable its `schedule:` block (keep `workflow_dispatch` for
emergencies), per the header comment in that file.

### 4a. 2026-07-12 floater-poller stall — root cause + the Q17 session

The first box bring-up's floater poller wedged ~05:42Z and sat "healthy"
for hours: a hung S3 fetch inside the global-mosaic disk subprocess, plus
the old `ProcessPoolExecutor` shape whose `with`-exit ran
`shutdown(wait=True)` — the timeout on `future.result()` fired, but the
context exit then blocked the MAIN LOOP forever behind the still-hung
child. A hang is invisible to `restart:` policies (they only see exits).

Fixed on main: the disk fetch is a spawned process that is **killed** at
`PER_DISK_TIMEOUT_S` (no wait-on-exit anywhere), and the poller carries a
**stall watchdog** (`FLOATER_WATCHDOG_STALL_S`, default 900 s) that
hard-exits on any future wedge so compose restarts it — self-healing by
construction, whatever the cause.

The Q17 box session is therefore a PULL + REBUILD, not just a restart:

```bash
cd ~/tat-satellite-render     # or wherever the render repo lives
git pull                      # picks up the mosaic-kill + watchdog fix
docker compose -p tat-render -f docker-compose.render.yml up -d --build \
  render floater-poller intensity-poller guidance-poller ens-watchdog
docker compose -p tat-render -f docker-compose.render.yml logs --tail 20 floater-poller
# expect: "stall watchdog armed: exit after 900s without progress"
```

Then watch `floaters/{slug}/manifest.json` `generated_utc` advance for
~an hour of box-only operation and retire the stopgap schedule per above.

## 5. HAFS worker — deliberately NOT auto-started

Peak RSS ~23 GB for a full HAFS-A+B cycle (2026-06 Railway telemetry; the
verdict then was "HAFS gets its own box"). On this 32 GB box it can run
capped at 24 GB **only** when you accept render/poller memory pressure
during active storms:

```bash
docker compose -p tat-render -f docker-compose.render.yml --profile hafs up -d hafs-worker
```

It builds from the pinned `hafs-render-worker` BRANCH (hafs-render v0.11.0);
main's requirements-hafs.txt carries the v0.12 repin still gated on the Q9
go/no-go. Until enabled, HAFS plots keep coming from the Triple-A-Tropics
`update-hafs.yml` GH workflow (6-hourly cron; ens-watchdog re-dispatches it
on staleness once its token is in `.env`).

## 6. Railway teardown

After a clean week on the box: delete the 6 Railway services / the project.
Nothing deploys from Railway anymore; the `railway.*.json` configs stay in
the repo as the migration record. (Railway's push-to-deploy hooks die with
the project — pushes to `main` / `hafs-render-worker` then only feed the box
and CI.)

## 7. Output watchdog — the product is the truth (2026-07-17)

The 2026-07-15..17 floaters/ freeze looked like a poller hang but was a
CONFIG failure: the shared box .env carries R2_PREFIX=meso (for the meso
stack), env_file inheritance silently pointed the render floater-poller
at meso/* from its first start, and floaters/* lost its last writer the
moment the GH stopgap schedule retired. Nothing crashed, nothing hung —
restart policies and the section-4a in-process stall watchdog are both
blind to a healthy process writing the WRONG keys (or to a wedged
sub-task while the rest of the loop progresses). The durable defense
watches the PRODUCT:

- /usr/local/bin/tat-floater-watchdog.sh (on the box): fetches
  floaters/manifest.json from the CDN cache-busted; if generated_utc is
  older than 30 min it restarts the floater-poller service via compose.
  A stamp file (/var/run/tat-floater-watchdog.last) limits restarts to
  one per 30 min so a genuine upstream outage (or a config failure a
  restart cannot fix, like this one) never restart-loops the container —
  the log then shows repeated stale findings, which IS the alert.
- systemd timer tat-floater-watchdog.timer (every 10 min, enabled):
  systemctl list-timers tat-floater-watchdog.timer to check,
  /var/log/tat-floater-watchdog.log for actions.

Pattern note: any box product with a manifest timestamp can get the same
treatment — copy the script, swap the URL + service name. The env-pin
that fixes this incident is in docker-compose.render.yml (floater-poller
environment: R2_PREFIX: floaters — environment beats env_file).

## Ops crib

```bash
P="docker compose -p tat-render -f docker-compose.render.yml"
$P ps                         # status + health
$P logs -f render             # follow the API
$P restart floater-poller     # bounce one worker
$P pull && $P build && $P up -d   # redeploy after a git pull
$P down                       # stop everything (caddy cert state survives in the volume)
```
