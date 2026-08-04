# RUNBOOK-S2 — shadow tile-pyramid emitter on the box (Docker)

Stage-2 explorer backend as a box service: cuts chrome-free WebP tile pyramids
(+ calibrated BT rasters + `products.json`) for the `/satellite/explorer/`
suite into `shadow/sat/goes19/**` on R2. Shadow-only — live `/satellite/` is
untouched. Runs on the **Hostinger box** (NOT Railway), on the same image base
and `.env` as the meso/S1 stacks — **no host pip, no cred paste, no
tat_palettes install**.

>> **MULTI-BOX:** this runbook covers ONE box's emitter. For which box runs
>> which lane, adding a box, secret propagation, the fleet git rule and the
>> health page, see **[RUNBOOK-FLEET.md](RUNBOOK-FLEET.md)** — `fleet.yml` is
>> the assignment map and `scripts/fleet.sh` the entry point. Lanes are
>> deployed from that map, not started by hand.

All commands from the repo dir on the box, on branch `main`:

```bash
git fetch origin main && git checkout main && git pull
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
rewritten (mainly for retired products -- active emitters append
incrementally with a prune-horizon filter and full-rebuild only on the
heal tick; see §3b).
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
- ⚠️ those per-lane numbers stopped describing the FLEET once 11 lanes ran at
  native cadence: 2026-08 measured ~2.1M tile PUTs/day **plus ~7M LIST
  pages/day** — listing, not writing, was 65% of a $445/16-days bill. See the
  ops-discipline rules below before adding any listing to a loop path.

## 3b) Ops discipline (2026-08-03 — the R2 Class A cost incident)

Every `list_objects_v2` **page** bills like a PUT ($4.50/M); HEAD/GET are
Class B ($0.36/M); DeleteObjects is free. The emitter now enforces:

- **Coverage listing is tail-bounded**: `_covered_times` passes
  `after=now-(backfill+2·step)` so `complete_stamps` probes only stamps that
  could still cover a slot (~6–30 LISTs/pass, was min(retention, 300)).
- **Slot re-checks are in-memory**: each successful `emit_one` returns its
  rendered stamp and the loop appends it to `covered` — the per-slot
  re-listing (~3M LISTs/day fleet-wide) is gone. Single writer per product
  (fleet.yml) is what makes this sound.
- **Manifests update incrementally**: one Class B GET + append + PUT. The
  full rebuild-from-listing runs only on cold start, geometry change (the
  refuse-to-clobber guard is unchanged), or the `S2_MANIFEST_HEAL_S` heal
  tick (default 6 h) which reconciles marker-present/manifest-absent
  orphans from interrupted runs. A duplicate already advertised writes
  nothing.
- **products.json is gated + throttled**: rebuilt only after a pass that
  rendered ≥1 frame, at most once per `S2_INDEX_MIN_S` (default 30 min) per
  sector; one-shot/manual emits bypass with force. `has_complete_frame`
  answers from a 72 h tail listing first, falling back to the full history
  so a paused product never drops out of the index.
- **Cross-pass state** (heal/throttle stamps) lives in
  `S2_STATE_DIR=/var/tmp/s2_state` inside the container — the shell loop
  re-execs python per pass but the container persists; a recreate just means
  one cold rebuild pass.
- **Measured, not projected**: `s1_ingest.R2` counts every billable request
  and each pass logs `[ops] … list_pages=N put=N …`. When judging any
  future change, read those lines from the lane logs — this incident's
  10× step was invisible in the code and obvious in the request counts.

## 3c) Container-publish flip (per lane, watchdog-guarded)

Every emit lane has a compose-mapped flip switch (`S2_CONTAINER_TILES_<LANE>`
in the box `.env`; unset = per-tile publishing). The canary order and the
verified procedure (conus-fast, 2026-08-03):

1. Baseline gate: every product of the lane fresh (< ~4x step) and zero
   recent `[FAIL]`s.
2. `sed -i 's/^S2_CONTAINER_TILES_<LANE>=.*/S2_CONTAINER_TILES_<LANE>=1/' .env`
   (append if absent) + force-recreate the lane.
3. Arm the watchdog BEFORE walking away — it rolls back on its own:
   `systemd-run --unit s2canary-<lane> --working-directory=/root/tsr-s2 \
      --property=Restart=on-failure --property=RestartSec=30 \
      --property=SuccessExitStatus="0 42" \
      --setenv=S2_CANARY_LANE_PROJ=tat-s2-<lane> \
      --setenv=S2_CANARY_LANE_COMPOSE=docker-compose.s2.<lane-file>.yml \
      --setenv=S2_CANARY_FLAG=S2_CONTAINER_TILES_<LANE> \
      --setenv=S2_CANARY_PRODUCTS="<sat/sector/product ...>" \
      --setenv=S2_CANARY_STALE_S=<7x lane step in seconds> \
      /root/tsr-s2/scripts/s2_canary_watchdog.sh`
4. Verify like the canary: `[ops] put=` collapses to ~6-8/frame while
   `[emit] wrote N tiles` is unchanged; `tiles.z{N}.json` + `t*.tar` on the
   CDN; a ranged read byte-matches the block slice; the explorer renders the
   newest frame.
5. Verdict in `/root/s2_canary_verdict_<proj>.txt` (PASS after 12 h).

Stale thresholds by lane step: 5 min -> 2100 s, 10 min -> 4200 s,
20 min -> 8400 s. The viewer needs no action in any direction: the manifest
`containers` hint is sticky, container frames stay readable after a
rollback, and legacy frames always read via the per-frame fallback.

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

## MTG FCI true color (mtgi1-fd) — activation runbook (2026-07-24)

Meteosat-12 (MTG-I1) FCI is the ring's FIFTH true-color sensor: registry
suite `mtgi1-fd` (truecolor/ir/irbd, the GK-2A model), collection
`EO:EUM:DAT:0662` (FDHSI L1C, ~800 MB zip of 41 chunked netCDFs per 10-min
cycle), ingest `s2_meteosat.fetch_fci_disk` behind a chunk-completeness gate
(a partial slot NEVER renders; the lane's backfill retries it).

State as of 2026-07-24: creds are on the box `.env` and the v1 token flow is
verified working (see the auth-migration note in `s2_meteosat.py` — we stay
on v1 client-credentials; the banner's "new method" is an interactive PKCE
flow unsuited to a headless box). Search/pinning works WITHOUT the licence;
downloads 403 with "GeneralLicense required to access this collection" until
the account accepts the EUMETSAT General Licence on user.eumetsat.int (one
click covers FCI + both SEVIRI services; activation lags up to 1 h). The
lane can run pre-licence: every tick fails honestly, publishes nothing, and
self-heals on the first post-licence tick.

Lane (dedicated — the base rotation's 4+ h cycle can't cover a 10-min
cadence + 1 h licence delay):

    docker compose -p tat-s2-mtg -f docker-compose.s2.yml \
      -f docker-compose.s2.mtg-lane.yml --profile cron up -d --no-build emit-cron

Seam validation at first post-licence data (mount a host dir for the PNG):

    docker compose -p tat-s2 -f docker-compose.s2.yml run --rm \
      -v /root:/host -e SEAM_OUT=/host \
      --entrypoint python emit validate_fci_seam.py

Attribution rides the viewer source line: "Contains EUMETSAT Meteosat data".
SEVIRI stays OUT of the true-color ring (no blue/green band) — BT members
only, per policy.

## Deploy topology discipline (2026-07-23)

Same rule as RUNBOOK-RENDER: box clone pulls from AND pushes to
origin/main only; no long-lived box-only branches. Rebuild images from
/root/tsr-s2 at main; the compose lane files are canonical on main.
