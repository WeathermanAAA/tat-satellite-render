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
