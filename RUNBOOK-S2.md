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

## 1) Retention: the object-level prune (Q7 retention floor)

> Bucket lifecycle rules are OUT: the box R2 token lacks
> Get/PutBucketLifecycleConfiguration (the old `lifecycle` service
> AccessDenied'd, 2026-07-08) and token scopes stay frozen. `s2_prune.py`
> needs only ListObjectsV2 + DeleteObject + Get/PutObject — the scopes the
> token already has.

```bash
# dry-run report (what WOULD go; deletes nothing):
docker compose -p tat-s2 -f docker-compose.s2.yml run --rm prune
# delete now:
docker compose -p tat-s2 -f docker-compose.s2.yml run --rm prune --apply
# daily loop (start once, survives reboots):
docker compose -p tat-s2 -f docker-compose.s2.yml --profile cron up -d prune-cron
```

Deletes `shadow/sat/**` frames whose **stamp** is older than 14 days
(`S2_PRUNE_DAYS`), always keeping the newest 2 stamps per product
(`S2_PRUNE_KEEP_MIN`) so an emitter outage can never empty a product.
Ready markers are deleted first (an interrupted prune never leaves an
advertised half-frame); manifests that still list a pruned stamp are
rewritten (cron-emitted products self-heal anyway on the next tick).
Non-frame keys (`latest_times.json`, `products.json`, `health.json`)
are never touched, and prefixes outside `shadow/` are refused outright.

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
The prune's default `--prefix shadow/sat/` covers every sat product prefix;
a `shadow-z6/…`-style prefix sits OUTSIDE it — run the prune service once
with the matching `--prefix shadow-z6/…` (still must start with `shadow/`)
to age those out too.

**PUT budget** (R2 Class A, $4.50/M after the 1M free tier):
- suite z0–5 ≈ 950 PUTs/emit → 15-min cron ≈ **2.8 M/mo ≈ $8–13/mo**
- suite z0–5 @ 30 min ≈ 1.4 M/mo ≈ **~$2/mo**; single product @ 5 min ≈ 0.35 M/mo ≈ free tier
- storage is minor (the prune keeps ≈14 days; manifests list only the newest 90 frames)

## 4) Verify (anyone, no creds — the CDN serves shadow/)

```bash
curl -s https://cdn.triple-a-tropics.com/shadow/sat/goes19/conus/products.json | head -c 400
curl -s https://cdn.triple-a-tropics.com/shadow/sat/goes19/conus/ir/latest_times.json | head -c 400
```

Then open **https://triple-a-tropics.com/satellite/explorer/** (product picker
top-left; compare panes at `/satellite/explorer/compare.html`). Products the
box hasn't emitted yet show "no data yet" and appear as they land.

## Meteosat members (GEO-RING gap fill) — one-time setup

The global composite's Africa/Europe/Indian-Ocean wedge fills from two SEVIRI
services (`s2_meteosat.py`: Meteosat 0° `EO:EUM:DAT:MSG:HRSEVIRI` + IODC 45.5°E
`EO:EUM:DAT:MSG:HRSEVIRI-IODC`). Creds-gated: until set up, both members
degrade honestly and the wedge stays the labeled transparent gap.

1. Free account at user.eumetsat.int; consumer key/secret from
   https://api.eumetsat.int/api-key/.
2. On user.eumetsat.int accept the **"Meteosat Level 1 data with latency
   ≥ 1 hour"** licence (free, self-service; takes ~1 h to activate — log out
   and back in). This allows any-purpose use of DERIVED imagery with
   attribution; we never republish the .nat source data. Do NOT rely on the
   <1 h NRT licence (paid tier for service providers) — the fetcher enforces
   a ≥60 min delay (`EUMETSAT_DELAY_MIN`, default 60) to stay compliant.
3. Box `.env`: add `EUMETSAT_CONSUMER_KEY=...` and
   `EUMETSAT_CONSUMER_SECRET=...`.
4. Image deps: `pip install -r requirements-s2-geo.txt` (satpy for the
   seviri_l1b_native reader) — add to the s2 image build.
5. Attribution (licence requirement) rides the viewer's source line:
   "Contains EUMETSAT Meteosat data".

Per-member valid times ride `latest_times.json` as `members[]` — the ~1 h
Meteosat skew is surfaced, never hidden.

## Deploy topology discipline (2026-07-23)

Same rule as RUNBOOK-RENDER: box clone pulls from AND pushes to
origin/main only; no long-lived box-only branches. Rebuild images from
/root/tsr-s2 at main; the compose lane files are canonical on main.
