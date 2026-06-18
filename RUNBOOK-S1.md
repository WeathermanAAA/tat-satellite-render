# S1 Satellite-Ingest -- Runbook (one page)

Stage 1 of the satellite re-architecture (`SATELLITE-REARCH.md` §3 + §8): an
**event-driven, never-miss** ingest of **one** product -- **GOES-19 Mesoscale 2
clean-IR** (`goes19-m2` "ir") -- into R2 under the **`shadow/`** prefix. SHADOW
ONLY: no viewer change, no prod cutover. The renderer is FROZEN (the `s1-render`
container is byte-identical to the box's meso render), so a shadow frame is
pixel-identical to the prod meso frame for the same slot **by construction**.

Fully ISOLATED from the live meso stack: distinct image tag (`tat-s1`), distinct
compose project (`-p tat-s1`), distinct `shadow/` R2 prefix, distinct health
ports. Bringing S1 up/down never touches the running meso poller.

> AWS resources are ALREADY created (the SQS queue + DLQ + the NOAA SNS
> subscription -- `infra/s1_sqs_sns.sh`, run from the codespace). You only run
> the box stack below. You touch no codespace secrets; you type the AWS + R2
> creds into the box's `.env` directly (this repo + chat never see them).

## 1. Get the S1 branch (a SEPARATE checkout, so the meso stack is untouched)
```bash
cd ~                       # or wherever you keep deploys
git clone --branch s1-sat-ingest \
  https://github.com/WeathermanAAA/tat-satellite-render.git tat-sat-s1
cd tat-sat-s1
```
(If you already have a clone, use a fresh directory for S1 so the running meso
checkout/containers are not disturbed.)

## 2. Create your .env from the example
```bash
cp .env.s1.example .env
```
Edit `.env` and set ONLY these (everything else has a safe default):
```
S1_ENABLED=true
# --- AWS (SQS): the tat-sat-ingest IAM key. Region MUST be us-east-1. ---
AWS_ACCESS_KEY_ID=<tat-sat-ingest access key id>
AWS_SECRET_ACCESS_KEY=<tat-sat-ingest secret access key>
AWS_DEFAULT_REGION=us-east-1
S1_QUEUE_URL=https://queue.amazonaws.com/532918216657/tat-sat-goes19-cmip
S1_DLQ_URL=https://queue.amazonaws.com/532918216657/tat-sat-goes19-cmip-dlq
# --- R2 (shadow frames): the SAME bucket + token as the meso worker. ---
R2_ENDPOINT=https://<your-account-id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<your R2 access key id>
R2_SECRET_ACCESS_KEY=<your R2 secret access key>
R2_BUCKET=triple-a-tropics-media
S1_R2_PREFIX=shadow
```
`.env` is gitignored -- never commit it.

## 3. Bring the stack up
```bash
docker compose -p tat-s1 -f docker-compose.s1.yml up -d --build
```
First build takes a few minutes (cartopy/scipy wheels + pyspectral LUT pre-warm).
`s1-ingest` waits for `s1-render` to be healthy, then cold-starts (seeds its
watermark + ledger from R2 reality) and begins consuming SQS.

## 4. Check health
Worker health (200 = healthy/booting/disabled, 503 = a source stale/failing):
```bash
curl -s localhost:8091/health | python3 -m json.tool
```
Render service health:
```bash
curl -s localhost:8082/health | python3 -m json.tool
```
Container status (look for `healthy`):
```bash
docker compose -p tat-s1 -f docker-compose.s1.yml ps
```

## 5. Watch logs
```bash
docker compose -p tat-s1 -f docker-compose.s1.yml logs -f s1-ingest
```
Expect lines like `published shadow/sat/goes19/meso2/ir/<ts>.webp ...` (a slot
rendered + uploaded) and, occasionally, `backfilled slot ... (SQS never
delivered)` (the fallback authority catching a drop). Every drop / backfill /
DLQ-bound message is logged with its slot id.

## 6. Never-miss audit (run after the stack has been up >= a few hours)
Lists the NOAA CMIPM2-C13 ground truth over a window and compares it to the
published shadow frames -> **zero missed slots** is the §8 S1 gate. Runs on the
box (it reads R2 with your `.env` creds + NOAA anonymously):
```bash
docker compose -p tat-s1 -f docker-compose.s1.yml run --rm --no-deps \
  s1-ingest python s1_audit.py --hours 6
```

## 7. Shadow-vs-prod pixel diff (proves byte-identity to the live meso frame)
For each slot present in BOTH `shadow/sat/goes19/meso2/ir/` and the prod
`meso/goes19-m2/ir/`, decodes both and diffs pixels (strict-identity tier):
```bash
docker compose -p tat-s1 -f docker-compose.s1.yml run --rm --no-deps \
  s1-ingest python s1_pixeldiff.py --sample 30
```
**Byte-identity (the strict gate) requires s1-render and the box's meso-render to
be the SAME image.** They are if you rebuild the meso stack and this S1 stack
from the same current branch together. Otherwise an independently-built s1-render
leaves a ~0.2% lossy-WebP encode floor (identical framing/bbox/content, mean diff
0.19%, 99.96% of pixels within +/-8 levels -- measured) that vanishes on a shared
image. For zero-floor, set `S1_RENDER_URL` in `.env` to the live meso hot render.
A standalone by-construction proof (renders a slot via the frozen path + diffs
the live CDN frame) is also available:
```bash
docker compose -p tat-s1 -f docker-compose.s1.yml run --rm --no-deps \
  s1-ingest python s1_byconstruction_proof.py --render-url http://s1-render:8080
```

## 8. Kill switch (stop writing, keep the containers)
```bash
sed -i 's/^S1_ENABLED=.*/S1_ENABLED=false/' .env
docker compose -p tat-s1 -f docker-compose.s1.yml up -d
```
The worker idles and writes NOTHING; `/health` reports `enabled:false`.
Re-enable by flipping it back to `true` and running the same `up -d`.

## 9. Tear down
```bash
docker compose -p tat-s1 -f docker-compose.s1.yml down
```

## What it writes to R2 (prefix `shadow/`)
- Frames:        `shadow/sat/goes19/meso2/ir/{YYYYMMDDTHHMMSSZ}.webp` (immutable, 1-yr cache)
- SSOT manifest: `shadow/sat/goes19/meso2/ir/latest_times.json` (`max-age=30`)
- Health:        `shadow/sat/goes19/meso2/ir/health.json` (`max-age=30`)

It writes NOTHING under `meso/`, `floaters/`, or any prod key.

## Notes
- The render port (8082) and worker health port (8091) bind to `127.0.0.1` only.
- The SQS message is deleted ONLY after the R2 PUT succeeds; a render/PUT failure
  redelivers and, after `maxReceiveCount=5`, lands in the DLQ
  (`tat-sat-goes19-cmip-dlq`). A poison slot is re-detected by the backfill on
  recovery (drain the DLQ back to the source to retry).
- On a box rebuild: the SQS queue + R2 frames survive, so a fresh `up -d`
  cold-starts from R2 + backfills the down-window gap -- it self-heals.
- Liveness: a watchdog self-exits the worker if a cycle stalls > `S1_WATCHDOG_S`
  (default 600 s); `restart: always` brings it back (the HAFS lesson).
- No real credentials, SSH keys, or passwords live in this repo. All secrets go
  in your local, gitignored `.env`.
```
