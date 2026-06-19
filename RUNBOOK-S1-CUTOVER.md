# S1 shadow -> prod cutover -- Runbook (one page)

Promote the GREEN-gated **GOES-19 Mesoscale-2 clean-IR** ingest from shadow to
PROD: S1 becomes the prod producer of `meso/goes19-m2/ir/` and the box meso
poller retires its own goes19-m2/ir render lane. **No viewer change** (the
viewer keeps reading `meso/goes19-m2/manifest.json`, written only by the
poller); only the FRAME PRODUCER for that one band changes. Branch:
`s1-prod-cutover` (off `s1-sat-ingest`).

> Prereq: the S1 shadow stack is up and the **§8 gate is GREEN** (run from
> anywhere, no box access):
> ```
> ./s1_accept.sh --hours 6 --sample 30   # never-miss PASS + REAL pixel delta 0
> ```

## How it works (two env vars, both default OFF = no behavior change)
- **S1 (`s1-ingest`): `S1_PROD_WRITE=true`** -> each frame is ALSO PUT to
  `meso/goes19-m2/ir/{stamp}.webp` (byte-identical to shadow). The prod PUT is
  inside the never-miss gate: a failed prod PUT does not ack the SQS message ->
  redelivers -> idempotent retry. Shadow + its `latest_times.json` SSOT stay the
  continuously-audited 72 h reference.
- **Meso poller: `MESO_ADOPT_EXTERNAL=goes19-m2:ir`** -> the poller STOPS
  rendering that band and instead ADOPTS S1's R2 frames into its
  memory-authoritative manifest each hot cycle (~60 s). The poller stays the
  sole manifest writer + owns prod retention; reconcile is the slow backstop.

## Cutover (zero-gap, ordered so prod is never un-produced)
Run on the box; type creds into each stack's gitignored `.env`, never the repo.

**0. Deploy the code (default-OFF -> safe, zero behavior change).** Pull
`s1-prod-cutover` for BOTH stacks and rebuild; confirm nothing changed:
```bash
# S1 stack
cd ~/tat-sat-s1 && git fetch && git checkout s1-prod-cutover && git pull
docker compose -p tat-s1 -f docker-compose.s1.yml up -d --build
curl -s localhost:8091/health | python3 -m json.tool | grep prod_write   # false
# meso stack
cd ~/tat-sat && git fetch && git checkout s1-prod-cutover && git pull
docker compose -p tat-meso -f docker-compose.meso.yml up -d --build
curl -s localhost:8090/health | python3 -m json.tool                      # unchanged
```
Re-run `./s1_accept.sh` -> still GREEN (shadow unaffected).

**1. Turn ON S1 prod-write (additive; poller still renders ir -> harmless
identical overwrite).**
```bash
cd ~/tat-sat-s1
sed -i 's/^S1_PROD_WRITE=.*/S1_PROD_WRITE=true/' .env || echo 'S1_PROD_WRITE=true' >> .env
docker compose -p tat-s1 -f docker-compose.s1.yml up -d
curl -s localhost:8091/health | python3 -m json.tool | grep prod_write   # true
```
Verify (from anywhere) that prod frames are now landing AND match shadow:
```bash
python3 s1_pixeldiff.py --remote --sample 30   # REAL delta still 0 (prod==shadow)
# prod frames present + advancing at 60 s:
curl -s "https://cdn.triple-a-tropics.com/meso/goes19-m2/manifest.json" \
  | python3 -c "import sys,json;d=json.load(sys.stdin)['bands']['ir'];print('ir frames',len(d['frames']),'latest',d['latest'])"
```
Let it run a few minutes; confirm no DLQ growth (`dlq_visible` in /health).

**2. Turn ON poller adopt (retires the poller's ir render lane).**
```bash
cd ~/tat-sat
sed -i 's/^MESO_ADOPT_EXTERNAL=.*/MESO_ADOPT_EXTERNAL=goes19-m2:ir/' .env \
  || echo 'MESO_ADOPT_EXTERNAL=goes19-m2:ir' >> .env
docker compose -p tat-meso -f docker-compose.meso.yml up -d
docker compose -p tat-meso -f docker-compose.meso.yml logs --since=3m meso-poller \
  | grep -E '\[adopt\] goes19-m2/ir'      # adopting external frames
```
Now S1 is the sole producer; the poller adopts. The other goes19-m2 bands
(irbd/wv/truecolor/swir) and all other sectors are untouched.

## Gates (do not skip -- Andrew's process)
- **Coverage gate (never-miss):** `python3 s1_audit.py --remote --hours 6` ->
  zero missed. (Shadow is the audited reference; prod is byte-identical by the
  pixel-diff, so shadow never-miss == prod never-miss for this product.)
- **Pixel gate:** `python3 s1_pixeldiff.py --remote --sample 30` -> REAL delta 0.
- **Viewer gate:** load /satellite/ -> the GOES-19 Mesoscale-2 "Clean IR" band
  advances every ~60 s with no gaps (the meso manifest now sources ir from S1).

## Reliability proof (never-stale demonstrated)
1. **Kill the S1 worker:** `docker kill <tat-s1 s1-ingest>` -> `restart: always`
   brings it back; on boot it cold-starts (seeds ledger+watermark from R2) and
   backfill catches any down-window slot. Confirm the prod ir band has NO gap
   over the kill window (`s1_audit.py --remote`).
2. **Reboot the box:** `sudo reboot`. With docker `restart: always` + the docker
   daemon enabled on boot (`sudo systemctl enable docker`), both stacks return
   automatically and self-heal the gap. Confirm zero prod gap afterward.

## Rollback (instant, no gap)
Clear EITHER var and `up -d` the affected stack:
```bash
# poller resumes rendering ir:
cd ~/tat-sat && sed -i 's/^MESO_ADOPT_EXTERNAL=.*/MESO_ADOPT_EXTERNAL=/' .env \
  && docker compose -p tat-meso -f docker-compose.meso.yml up -d
# S1 stops prod-writing (shadow continues):
cd ~/tat-sat-s1 && sed -i 's/^S1_PROD_WRITE=.*/S1_PROD_WRITE=false/' .env \
  && docker compose -p tat-s1 -f docker-compose.s1.yml up -d
```
Roll back the poller FIRST (so it resumes rendering before S1 stops producing)
to keep the ir band gap-free during a rollback.

## Notes
- During the overlap (step 1 before step 2) both writers emit the SAME key with
  byte-identical bytes -> a no-op overwrite, not a race (proven: pixel-diff REAL
  delta 0.000000%).
- The poller owns `meso/` retention; S1 does NOT prune prod keys (S1 prune only
  touches the shadow prefix). One deleter, no fight.
- Keep S1 prod-write ON permanently after cutover; the shadow path stays the
  audit reference. Do not delete the shadow stack.
