# S1 STAGE A multi-sat -- GOES-18 + Himawari-9 ingest (one page)

Extends the GREEN GOES-19 S1 ingest to **GOES-18** (GOES-West ABI, a config twin)
and **Himawari-9** (AHI FLDK clean-IR, 07W's satellite -- SEGMENTED scans).
SHADOW ONLY (R2 `shadow/sat/{sat}/{sector}/ir/`); no viewer/floater/meso change.
Per-source isolation: each sat is its OWN worker + queue, so a new-sat failure
can never stale GOES-19 S1 / meso / the floater / ACE.

## AWS resources -- ALREADY CREATED (idempotent; re-run safe)
Created from the codespace with the tat-sat-ingest creds (acct 532918216657):
```
./infra/s1_create_source.sh goes18       # -> tat-sat-goes18-cmip (+ -dlq), sub NewGOES18Object,     filter ABI-L2-CMIPM/
./infra/s1_create_source.sh himawari9    # -> tat-sat-himawari9-fldk (+ -dlq), sub NewHimawariNineObject, filter AHI-L1b-FLDK/
```
Both subscribe with `FilterPolicyScope=MessageBody` (the INGEST-1 catch) and were
verified RECEIVING within a minute (queues accumulate until the box workers
drain them; SQS holds 14 days). The worker re-filters by PARSED key, so the
broad AHI-L1b-FLDK/ prefix (all bands) is trimmed to B13 by the worker.

## Box deploy (paste-back; the S1 stack is `-p tat-s1`)
On the box's S1 checkout (`~/tat-sat-s1` or a fresh clone of this branch):
```bash
cd ~/tat-sat-s1 && git fetch && git checkout s1-multisat-copyplots && git pull
# .env: add the two new queue URLs (S1_SOURCE is set per-service by compose):
grep -q S1_QUEUE_URL_GOES18 .env || cat >> .env <<'EOF'
S1_QUEUE_URL_GOES18=https://queue.amazonaws.com/532918216657/tat-sat-goes18-cmip
S1_QUEUE_URL_HIMAWARI9=https://queue.amazonaws.com/532918216657/tat-sat-himawari9-fldk
EOF
docker compose -p tat-s1 -f docker-compose.s1.yml up -d --build
docker compose -p tat-s1 -f docker-compose.s1.yml ps   # s1-render + 3 ingest, all healthy
curl -s localhost:8092/health | python3 -m json.tool | grep -E 'source|product|healthy'  # goes18
curl -s localhost:8093/health | python3 -m json.tool | grep -E 'source|product|healthy'  # himawari9
```
The new services share the one `s1-render`; GOES-19 (`s1-ingest`, port 8091) is
untouched. Watch logs for `published shadow/sat/goes18/...` and
`published shadow/sat/himawari9/fldk/ir/...` (+ occasional `backfilled ...`).

## Verify (from ANYWHERE -- no box access; run after the stack is up a while)
Never-miss vs NOAA ground truth, completeness-gated (AHI = all 10 segments, never
a half-scan); shipped set via the public CDN `latest_times.json`:
```bash
python3 s1_audit.py --remote --source goes18    --hours 6   # zero missed
python3 s1_audit.py --remote --source himawari9 --hours 6   # zero missed (10-seg complete only)
```
Pre-deploy these read PENDING (0 shadow frames yet) but already list the live
NOAA ground truth (~120 GOES-18 CMIPM2-C13/2h; ~12 Himawari FLDK complete
scans/2h), proving the ground-truth + AHI segment-grouping. Pixel-correctness:
spot-check a frame URL from each `latest_times.json` on the CDN.

## Reliability proof (never-stale; per source)
```bash
docker kill $(docker compose -p tat-s1 -f docker-compose.s1.yml ps -q s1-ingest-himawari9)
# restart:always brings it back; it cold-starts (seeds ledger+watermark from R2)
# and the backfill fills the down-window gap. Re-run the --source himawari9 audit
# -> zero missed over the kill window. Reboot the box -> docker restart:always +
# `systemctl enable docker` return all 4 containers; the gap self-heals.
```
Isolation check: killing the himawari9 worker leaves goes18 + goes19 + meso +
the floater + ACE untouched (separate processes/queues).

## Rollback / kill switch
Per source: `S1_ENABLED=false` in .env stops ALL S1 workers; or `docker compose
-p tat-s1 -f docker-compose.s1.yml stop s1-ingest-himawari9` stops just one. The
AWS resources persist (idempotent re-create); the queues hold the backlog.
