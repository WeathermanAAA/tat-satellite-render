# Meso-Sector Poller -- Runbook (one page)

A self-contained Docker stack that keeps R2 fresh with the operators' steered
**mesoscale sectors**: GOES-19 M1/M2, GOES-18 M1/M2, and the Himawari-9 Target
box. It is fully ISOLATED from the floater poller (separate process, separate
R2 prefix `meso/`, separate health). Two containers: `meso-render` (the /render
service on loopback) and `meso-poller` (discovers each sector's live extent and
renders the full band palette over it).

## 1. Clone the branch
```bash
git clone --branch meso https://github.com/WeathermanAAA/tat-satellite-render.git
cd tat-satellite-render
```

## 2. Create your .env from the example
```bash
cp .env.meso.example .env
```
Edit `.env` and set ONLY these (everything else has a safe default):
```
MESO_ENABLED=true
R2_ENDPOINT=https://<your-account-id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<your R2 access key id>
R2_SECRET_ACCESS_KEY=<your R2 secret access key>
R2_BUCKET=triple-a-tropics-media
```
`.env` is gitignored -- never commit it. Use an R2 API token with Object Read &
Write on that bucket.

## 3. Bring the stack up
```bash
docker compose -f docker-compose.meso.yml up -d --build
```
First build takes a few minutes (cartopy/scipy wheels + pyspectral LUT pre-warm).
The poller waits for `meso-render` to be healthy, then starts discovering.

## 4. Check health
Poller health (200 = healthy/booting/disabled, 503 = a sector stale/failing):
```bash
curl -s localhost:8090/health | python3 -m json.tool
```
Render service health:
```bash
curl -s localhost:8080/health | python3 -m json.tool
```
Container status (look for `healthy`):
```bash
docker compose -f docker-compose.meso.yml ps
```

## 5. Watch logs
```bash
docker compose -f docker-compose.meso.yml logs -f meso-poller
```
Expect lines like `discover goes19-m1 -> bbox=[...] scan=...` and
`uploaded meso/goes19-m1/ir/<ts>.png`.

## 6. Kill switch (stop writing, keep the container)
Set the switch and recreate the poller (the render service can stay up):
```bash
sed -i 's/^MESO_ENABLED=.*/MESO_ENABLED=false/' .env
docker compose -f docker-compose.meso.yml up -d
```
The poller idles and writes NOTHING to R2; `/health` reports `enabled:false`.
Re-enable by flipping it back to `true` and running the same `up -d`.

## 7. Stop / remove the stack
```bash
docker compose -f docker-compose.meso.yml down
```

## 8. Deploying an update (already-cloned box, on the `meso` branch)
```bash
cd <repo>                                    # the dir cloned with --branch meso
git pull                                     # fast-forward to the latest meso commit
docker compose -f docker-compose.meso.yml up -d --build
docker compose -f docker-compose.meso.yml ps        # both services -> healthy
```

## 9. Self-heal watchdog (never-stale)
`restart: always` only recovers a CRASH (process exit). A silent WEDGE -- the
loop frozen in a hung call, OR the render service stalled so no frames upload --
leaves the container "running" but producing nothing, and the 503 healthcheck can
only DETECT it (it reads a stored snapshot the frozen loop can no longer refresh).
A daemon thread in `meso_poller.py`, run independently of the main loop, watches
the per-sector last SUCCESSFUL frame-upload time against the current clock; when
EVERY producing sector has gone stale past its per-source threshold (GOES 600 s,
Himawari 900 s) it force-exits so `restart: always` brings it back clean. A sector
that has not produced since (re)start is ignored, so a restart while the render
service is still down never loops. Tunables + kill switch (`MESO_SELFHEAL`) live
in `.env.meso.example`.

Confirm it is armed after deploy (and at the REAL thresholds):
```bash
docker compose -f docker-compose.meso.yml logs meso-poller | grep "self-heal watchdog ON"
#   expect: ... goes>600s hima>900s grace=300s every=30s
```
Confirm cadence is HOLDING (hot IR ~60 s, Himawari ~2.5 min) via consecutive
frame gaps in the live manifest:
```bash
curl -s https://cdn.triple-a-tropics.com/meso/goes19-m1/manifest.json \
  | python3 -c "import sys,json,datetime as d;f=[x['t'] for x in json.load(sys.stdin)['bands']['ir']['frames'][-12:]];p=lambda s:d.datetime.fromisoformat(s.replace('Z','+00:00'));print('gaps(s):',[int((p(f[i])-p(f[i-1])).total_seconds()) for i in range(1,len(f))])"
```

### LIVE kill/recover proof (fast variant -- lower thresholds, induce, REVERT)
```bash
# 0) baseline: both healthy + watchdog armed at the REAL thresholds
docker compose -f docker-compose.meso.yml ps
docker compose -f docker-compose.meso.yml logs --since 5m meso-poller | grep "self-heal watchdog ON"
#    expect: ... goes>600s hima>900s grace=300s every=30s

# 1) TEMP: lower thresholds so a wedge trips in minutes (REVERTED in step 5)
cat >> .env <<'EOF'
# --- TEMP self-heal proof: DELETE these 4 lines in step 5 ---
SELFHEAL_STALE_GOES_S=120
SELFHEAL_STALE_HIMA_S=120
SELFHEAL_GRACE_S=60
SELFHEAL_CHECK_S=15
EOF
docker compose -f docker-compose.meso.yml up -d meso-poller
docker compose -f docker-compose.meso.yml logs --since 1m meso-poller | grep "self-heal watchdog ON"
#    expect TEST values now: ... goes>120s hima>120s grace=60s every=15s
sleep 90    # let it upload a few fresh frames so last_frame_utc is populated

# 2) induce the wedge: stop the render service -> frames stop uploading -> stale
docker compose -f docker-compose.meso.yml stop meso-render ; date -u

# 3) RECOVERY proof: stale detected -> force-exit(1) -> restart:always restarts it
sleep 210   # > 120s threshold + 60s grace + a 15s check
docker compose -f docker-compose.meso.yml logs --since 6m meso-poller | grep "SELF-HEAL"
#    expect: SELF-HEAL: meso poller WEDGED -- no fresh scan on ANY sector (...>120s...); force-exiting ...
docker compose -f docker-compose.meso.yml ps
#    expect: meso-poller "Up <a few> seconds" (it auto-restarted). It will sit
#    quietly (no loop) while meso-render is down -- that is the None-skip working.

# 4) restore the render service -> fresh frames resume with NO manual touch
docker compose -f docker-compose.meso.yml start meso-render ; sleep 90
docker compose -f docker-compose.meso.yml logs --since 2m meso-poller | grep -E "uploaded meso/.*/ir/"
#    expect: fresh "uploaded meso/<sector>/ir/..." lines = production resumed

# 5) REVERT (CRITICAL -- leaving the tiny values would false-trip in normal ops):
#    delete the 4 TEMP lines added in step 1 from .env, then recreate the poller
${EDITOR:-nano} .env        # remove the "TEMP self-heal proof" block
docker compose -f docker-compose.meso.yml up -d meso-poller

# 6) REVERT proof -- live thresholds back at 600/900 AND no TEMP overrides linger:
docker compose -f docker-compose.meso.yml logs --since 1m meso-poller | grep "self-heal watchdog ON"
#    MUST show: ... goes>600s hima>900s grace=300s every=30s
docker compose -f docker-compose.meso.yml exec meso-poller sh -lc 'env | grep SELFHEAL_ || echo "OK: no SELFHEAL_ overrides -> defaults 600/900"'
#    MUST show: OK: no SELFHEAL_ overrides -> defaults 600/900
curl -s localhost:8090/health | python3 -m json.tool | grep '"healthy"'   # -> true
```
Paste back: step 3 (the `SELF-HEAL` line + `ps` showing the restart), step 4 (the
fresh `uploaded …/ir/…` lines), and step 6 (the `goes>600s hima>900s` line + the
`OK: no SELFHEAL_ overrides` line) -- that proves BOTH the recovery and the revert.

## What it writes to R2 (prefix `meso/`)
- Frames:  `meso/{slug}/{band}/{YYYYMMDDTHHMMZ}.png` (immutable, 1-yr cache)
- Per-sector manifest: `meso/{slug}/manifest.json` (`max-age=30`)
- Top index: `meso/manifest.json` (`max-age=30`)
- Health:   `meso/health.json` (`max-age=30`)

Slugs: `goes19-m1`, `goes19-m2`, `goes18-m1`, `goes18-m2`, `himawari9-meso`.
Bands: `ir`, `irbd` (hot, 60 s), `wv_up`, `wv_low`, `truecolor`, `swir` (cold).

## Notes
- The render port (8080) and poller health port (8090) bind to `127.0.0.1` only
  -- not exposed to the public internet.
- A single sector's failure (S3 hiccup / unreadable scan) preserves its
  last-known-good extent + frames and flips only that sector to stale/failing;
  the other four sectors and the floater poller are untouched.
- No real credentials, SSH keys, or passwords live in this repo. All secrets go
  in your local, gitignored `.env`.
