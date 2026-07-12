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

## Ops crib

```bash
P="docker compose -p tat-render -f docker-compose.render.yml"
$P ps                         # status + health
$P logs -f render             # follow the API
$P restart floater-poller     # bounce one worker
$P pull && $P build && $P up -d   # redeploy after a git pull
$P down                       # stop everything (caddy cert state survives in the volume)
```
