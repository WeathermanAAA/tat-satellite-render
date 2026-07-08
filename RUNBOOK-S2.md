# RUNBOOK-S2 — shadow tile-pyramid emitter on the box (Docker)

Stage-2 explorer backend as a box service: cuts chrome-free WebP tile pyramids
(+ calibrated BT rasters + `products.json`) for the `/satellite/explorer/`
suite into `shadow/sat/goes19/**` on R2. Shadow-only — live `/satellite/` is
untouched. Runs on the **Hostinger box** (NOT Railway), on the same image base
and `.env` as the meso/S1 stacks — **no host pip, no cred paste, no
tat_palettes install**.

All commands from the repo dir on the box, on branch `s2-sat-ingest`:

```bash
git fetch origin && git checkout s2-sat-ingest && git pull
```

## 0) Build the image (once per code pull)

```bash
docker compose -p tat-s2 -f docker-compose.s2.yml build emit
```

## 1) One-time: lifecycle TTL (Q7 retention floor)

```bash
docker compose -p tat-s2 -f docker-compose.s2.yml run --rm lifecycle --days 10
```

Merge-safe: adds/updates only the `s2-shadow-sat-goes19-ttl` rule on
`shadow/sat/goes19/`; every other bucket rule is preserved. `--show` prints
rules without changing anything.

## 2) One-shot emit (the eyeball step)

```bash
# single product, native zoom (z6) + BT raster:
docker compose -p tat-s2 -f docker-compose.s2.yml run --rm emit \
    --product goes19-conus-ir --store r2 --prefix shadow

# or the whole 24-product imagery suite off ONE scan (z0-5, Q7-tiered):
docker compose -p tat-s2 -f docker-compose.s2.yml run --rm emit \
    --suite conus --store r2 --prefix shadow --max-zoom 5
```

Any registry product id works with `--product` (e.g. `goes19-conus-airmass`,
`goes19-conus-truecolor`, `goes19-conus-c08`). The emitter prints the manifest
CDN URL on success.

## 3) Cron loop (continuous, Q7 tiering baked in)

```bash
docker compose -p tat-s2 -f docker-compose.s2.yml --profile cron up -d emit-cron
docker compose -p tat-s2 -f docker-compose.s2.yml logs -f emit-cron   # watch
docker compose -p tat-s2 -f docker-compose.s2.yml --profile cron down # stop
```

Defaults: full `conus` suite, `--max-zoom 5`, every **900 s**. Override in
`.env`: `S2_CRON_SUITE`, `S2_CRON_MAX_ZOOM`, `S2_CRON_INTERVAL_S`,
`S2_CRON_PREFIX`. Duplicate scans are skipped via the per-frame ready marker.

**Zoom-tier rule (Q7):** the cron stays at `--max-zoom 5`; full-res z6 is an
on-demand `run --rm emit --product … --prefix shadow-z6` (no `--max-zoom`).
z6 and z5 frames must NOT share a product prefix: the manifest keeps ONE
geometry, so the emitter now **refuses** an emit that would drop existing
frames (override for deliberate migrations: `--allow-geometry-change`), and a
scan the cron already emitted **dedups on the ready marker** — an uncapped
re-emit of it on the same prefix writes nothing (the emitter says so).
The lifecycle TTL rule covers `shadow/sat/goes19/` as a whole, including a
`shadow-z6`-style sub-prefix only if it also starts with that path — for
one-off deep-zoom prefixes outside it, re-run the lifecycle service with the
matching `--prefix`.

**PUT budget** (R2 Class A, $4.50/M after the 1M free tier):
- suite z0–5 ≈ 950 PUTs/emit → 15-min cron ≈ **2.8 M/mo ≈ $8–13/mo**
- suite z0–5 @ 30 min ≈ 1.4 M/mo ≈ **~$2/mo**; single product @ 5 min ≈ 0.35 M/mo ≈ free tier
- storage is minor (TTL keeps ≈10 days; manifests list only the newest 90 frames)

## 4) Verify (anyone, no creds — the CDN serves shadow/)

```bash
curl -s https://cdn.triple-a-tropics.com/shadow/sat/goes19/conus/products.json | head -c 400
curl -s https://cdn.triple-a-tropics.com/shadow/sat/goes19/conus/ir/latest_times.json | head -c 400
```

Then open **https://triple-a-tropics.com/satellite/explorer/** (product picker
top-left; compare panes at `/satellite/explorer/compare.html`). Products the
box hasn't emitted yet show "no data yet" and appear as they land.
