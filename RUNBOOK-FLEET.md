# RUNBOOK-FLEET — running Triple-A-Tropics on more than one box

Two boxes today, more coming (the models pipeline especially). These are the
conventions that keep box N+1 a *step* instead of a bespoke build, and that
keep "which box runs what" answerable from a file rather than from memory.

**Everything here is driven by two files in this repo:**

| file | is |
| --- | --- |
| `fleet.yml` | the inventory (boxes) **and** the assignment map (lane → box) |
| `scripts/fleet.sh` | the only command you need for multi-box operations |

Supporting: `scripts/provision_box.sh` (bare box → ready), `scripts/heartbeat.sh`
(liveness → R2), `scripts/box_name.py` (a box's fleet identity).

---

## The five rules

### 1. The assignment map is data, not tribal knowledge

`fleet.yml` says which box owns which lane. To move a lane you edit the map and
deploy — you never start a lane on a box by hand, because the next agent (or
you in a month) reads the map, not the container list.

```bash
scripts/fleet.sh lanes box2        # what box2 is SUPPOSED to run
scripts/fleet.sh status            # what every box IS running, + load/RAM/sha
```

If those two disagree, the box is wrong and `deploy` fixes it — deploy
**reconciles**, so it both starts what the map assigns and stops any lane the
map no longer assigns to that box. That is what makes "moving a lane is an
edit plus a deploy" true: without the stop half, a reassigned lane keeps
running on the old box too, and two emitters race the same product's manifest
geometry.

### 2. Every box pulls from `main` and pushes to `main`. Drift is an incident.

No box-only branches, ever. Work that exists only on a box is invisible to
every other box, to the site, and to the next agent — that is how a fleet
silently forks, and it is exactly the failure the 2026-07-23 tsr reconciliation
cleaned up. A box is expected to be **clean, on main, neither ahead nor
behind**.

```bash
scripts/fleet.sh drift             # the rule, enforced
# ok    box1 (c396aca) clean on main
# DRIFT box2 (a1b2c3d): 2 commit(s) NOT pushed to main;
# DRIFT box3: UNREACHABLE (ssh failed)
```

**Exit code is the gate**: `0` = every box clean, `1` = at least one drifted,
missing its repo, or unreachable. So `fleet.sh drift || alert` is meaningful.
A box that is DOWN counts as drift — it is a box whose state you cannot
vouch for — and one dead box never stops the others from being checked.

Treat any `DRIFT` line as an incident: land the work on `main` (relay it
through a box that has push rights if the box itself lacks them), then
`deploy`. Do **not** "fix" drift by resetting over it — `provision_box.sh`
deliberately refuses to `reset --hard` a dirty tree, because resetting over
unique state is how you destroy the evidence that the rule was broken.

Boxes may lack push credentials; that is fine and does not weaken the rule.
`provision_box.sh` generates a deploy key and prints it under
`QUEUED FOR ANDREW` when it has to fall back to an HTTPS (read-only) clone.
Until that key is added, author commits in the Codespace and relay:

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/tat_box' \
  git push ssh://root@<box>/root/tsr-s2 HEAD:refs/heads/codespace-inbox
ssh root@<box> 'cd /root/tsr-s2 && git merge --ff-only codespace-inbox && git push origin main'
```

### 3. Adding box N+1 is three commands

```bash
# 1. add it to fleet.yml (name, host, tier, role, lanes) and land that on main
# 2. build it out — idempotent, safe to re-run on an existing box
scripts/fleet.sh provision box3
# 3. give it the secrets, then start its assigned lanes
scripts/fleet.sh setenv box3 R2_ACCESS_KEY_ID=...      # once per key
scripts/fleet.sh deploy box3
```

`provision_box.sh` installs docker + git + the host python deps, clones the
repo pinned to `main`, creates the `.env` skeleton, builds `tat-s2:latest`, and
installs the heartbeat timer. **If a new box needs a step this script does not
do, add it to the script** — that is the whole guarantee. Box N+2 inherits
every fix box N+1 needed.

Prerequisite: the `tat-box-agent` public key in the new box's
`root@.../.ssh/authorized_keys`, and an entry in `fleet.yml`.

### 4. Secrets live only in each box's `.env`, and propagate through one command

Never in git, never in a script, never in a compose file. `fleet.yml`'s
`env_keys` is the required-key list, and provisioning reports any that are
missing rather than letting a lane fail at 03:00 in a cron log.

```bash
scripts/fleet.sh setenv all EUMETSAT_CONSUMER_SECRET=...   # every box
scripts/fleet.sh setenv box2 R2_BUCKET=...                 # one box
scripts/fleet.sh deploy all                                # restart to pick it up
```

`setenv` reports per box and **exits non-zero if any box was missed**, because
a half-applied credential — some boxes on the new secret, some on the old, no
signal — is the worst outcome available. If it says `INCOMPLETE`, fix the
failed box before deploying.

Rolling a credential fleet-wide is `setenv all` + `deploy all`. To seed a new
box from an existing one without the values ever printing:

```bash
ssh root@<old-box> 'grep -E "^(R2_|EUMETSAT_)" /root/tsr-s2/.env' \
  | ssh root@<new-box> 'cat >> /root/tsr-s2/.env'
```

### 5. Every box has a heartbeat, because a silent box looks like quiet weather

Each box publishes `fleet/<name>.json` to R2 every minute (systemd timer), and
`/fleet/` on the site renders it. The box asserts liveness **itself**,
independently of whether its lanes produced a frame — otherwise a dead box is
indistinguishable from a quiet basin, since both just mean "no new imagery".

The heartbeat runs on the **host**, not in a lane container, precisely so it
keeps reporting when every container is dead. It is installed
**unconditionally** at provision time, before secrets exist: it simply fails
until `setenv` lands them and then starts reporting on its own. (Gating it on
secrets meant a brand-new box was invisible on `/fleet/` exactly when you most
want to watch it.)

The page's roster is published too — every box writes `fleet/index.json` from
`fleet.yml` next to its own heartbeat — so **box3 appears on the health page as
soon as it is in the inventory**, with nothing to edit in the site repo.

The heartbeat carries: git sha + branch + dirty count (so drift shows up on the
health page too), load per core, memory, disk, **OOM kills since boot** (the
failure mode that actually bites an emit box), and which lanes are up vs
exited.

Consumers must judge freshness on the heartbeat's own `ts`. A stale object
means a dead box no matter how healthy its contents look — the page treats
>3 min as stale and >10 min as silent.

---

## The R2 write kill switch

R2 bills Class A operations (PUT/LIST) per million; the 2026-08-03 incident
ran ~445,000/h against a ~29,000/h norm. The defence has two halves, and the
boxes only ever see the second:

* **The r2-breaker Worker** (TAT `workers/r2-breaker.js`, cron every 5 min)
  reads the R2 analytics and publishes a status document. In `alert` mode it
  opens a GitHub issue when it *would have* tripped; in `armed` mode it also
  sets `writes_enabled:false`. It never resumes on its own: `reset` is the only
  way back, and it is manual.
* **`tat_killswitch.py`** (repo root, byte-identical copy in TAT) gates every
  R2 PUT on the boxes. Each writer asks `writes_allowed(key)` first. It reads
  the CDN mirror `fleet/breaker.json` (never the Worker -- free-plan request
  budget) at most once a minute per process, and blocks only on a fresh
  (`worker_ts` within 15 min) document that says `writes_enabled:false`.
  That flag is the Worker's one gate: it goes false only on an auto-trip
  while `armed` or on a manual `trip` (any mode, alert included), and only
  `reset` turns it back on -- `disarm` does not, and the boxes do not look
  at `mode` for the decision (gating on `armed` as well would make `disarm`
  a silent second path back to writing and a manual trip in alert mode a
  no-op). Anything else -- unreachable CDN, stale copy, missing fields, an
  exception -- **fails open**.
  Deletes are never gated, so prune keeps reducing storage. Keys under
  `fleet/` are always allowed, so the heartbeat, the roster and the mirror
  itself keep landing while writes are stopped: a darkened box must never
  look like a dead one.

The heartbeat is the mirror: every minute each box fetches `/r2-breaker/status`
once, stamps `breaker_age_s` + `breaker_mode` into its own `fleet/<box>.json`,
and (only after that PUT succeeds, only for a valid `v:1` document) copies the
status to `fleet/breaker.json`. A failed or garbage fetch writes nothing to the
mirror; the last good copy stands.

### The 30-second manual path

When you need writes stopped NOW and do not want to depend on the Worker:

```bash
scripts/fleet.sh writes off all        # or: writes off box2
# ... investigate, fix, then:
scripts/fleet.sh writes on all
```

`writes off` writes `TAT_R2_WRITES=0` into every `.env` the write lanes load
(box1 runs stacks out of more than one checkout, so that is more than
`/root/tsr-s2/.env`), then brings every **write** container present back up
from its own compose labels with the **same image** (`--no-build --no-deps
--force-recreate`; the box half is `scripts/box_writes.sh`) -- emit-cron lanes
(not prune), `s1-ingest*`, `meso-poller`, the floater/intensity/guidance
pollers, every overlays poller. The list comes from `docker ps` with name
filters, so a lane that is not running is skipped, not an error. It then
**verifies**, per container, that `TAT_R2_WRITES` is inside it AND that the
guard is really there (`guard: present`): it runs `tat_killswitch` with the
container's own python from the container's own cwd and makes it refuse a
probe key (or, for `on`, import and allow). Either failure exits non-zero:
success means the value is in the process and the process has something that
reads it, not that a command returned 0. It **never** touches anything that
serves: caddy, render, `meso-render*`, `s1-render`, `radar-render`,
centerfix/objfix, prune. Expect ~30-60 s per box.

### Guard presence: the switch is inert on an image that predates it

Every writer imports the switch as `try: import tat_killswitch except
Exception: tat_killswitch = None` and treats None as ALLOWED (the switch can
only ever stop writes, never break a lane). So a lane whose **image** was
built before `tat_killswitch.py` existed takes `TAT_R2_WRITES=0`, shows it in
`docker inspect`, and keeps writing. `writes off` reports that lane as
`guard: ABSENT` and fails the box; do not read the env value alone as a stop.
As of 2026-08-25 that is every lane on both boxes (all four image families
were checked on box1). The images that must be rebuilt, and from where:

* **s2 emit lanes** (`tat-s2-*-emit-cron-1`, both boxes): `scripts/fleet.sh
  deploy all` (pull main, rebuild `tat-s2`, recreate the lanes).
* **tat-render pollers** (`tat-render-{floater,intensity,guidance}-poller-1`,
  box1): `docker compose -p tat-render -f docker-compose.render.yml build
  && ... up -d` from `/root/tsr-s2` (RUNBOOK-RENDER.md).
* **meso-poller** (`tat-satellite-render-meso-poller-1`, box1): rebuild from
  its own checkout `/root/tat-satellite-render` (it is NOT `/root/tsr-s2`,
  and `fleet.sh deploy` never touches it) -- pull main there first.
* **s1 ingest** (`tat-s1-s1-ingest*-1`, box1): rebuild from `/root/tat-sat-s1`
  per RUNBOOK-S1-MULTISAT.md -- same caveat, a separate checkout.
* **tat-overlays pollers** (box1) need no rebuild: they hard-reset a sparse
  checkout of TAT main every tick and import the TAT copy from `/work/tat`,
  so the guard arrives on its own once TAT main carries
  `tat_killswitch.py`. Until then they too report `guard: ABSENT`.
* **hafs-worker** builds from the pinned `hafs-render-worker` branch; its
  guard lands only when that branch is updated.

After any rebuild, `scripts/fleet.sh writes on <box>` is a cheap way to see
`guard: present` on every lane without stopping anything.

### While writes are stopped

* **s2 emit lanes**: each pass exits immediately after building its store
  (`[emit] R2 writes disabled by kill switch (...) -- skipping this pass`,
  rc 0) -- no coverage LISTs, no NOAA fetches, no renders. Without that a
  blocked pass would re-fetch and re-render every uncovered slot in its
  window every tick, since nothing ever becomes covered.
* **s1 ingest, meso/floater/hafs pollers, guidance poller**: the PUT returns
  False and the lane retries on its own cadence; nothing is retained.
* **intensity poller** (`R2Sink`): a blocked PUT *raises*, so the engine
  holds every changed feed's signature (`process FAILED (signature held for
  retry)`, one line per source per cycle) and re-emits all of them --
  tracks/ACE feeds, the global geojson, cyclolab adv/formation/pages -- on
  the first cycle after writes resume. No recreate is needed after
  `breaker reset`. (A silent no-op would have marked those feeds as written
  and left R2 at the pre-trip bytes until the next real upstream change.)
* **heartbeats** (`fleet/*`) keep landing; deletes/prune keep running.

Why recreate and not `docker restart`: compose bakes `env_file` into the
container when it is created, so a restart keeps the old env (verified
2026-08-25), and `tat_killswitch` reads the env once at import. Recreate
with the same image is the smallest action that actually changes the value
the process sees. Because it re-reads the compose file on disk, keep boxes
clean on main (`fleet.sh drift`) so a recreate deploys nothing you did not
mean to.

### The breaker verbs

```bash
scripts/fleet.sh breaker status        # the status JSON (public)
scripts/fleet.sh breaker arm           # mode=armed: a trip now stops writes
scripts/fleet.sh breaker disarm        # mode=alert (does NOT re-enable writes: boxes gate on writes_enabled, not mode)
scripts/fleet.sh breaker trip          # writes_enabled=false in ANY mode (alert too), fleet-wide, ~2 min
scripts/fleet.sh breaker reset         # the ONLY way back to writing
```

The admin verbs send `X-Admin-Key` from `TAT_BREAKER_ADMIN_KEY` in your env,
or else from box1's `.env` (`fleet.sh setenv box1 TAT_BREAKER_ADMIN_KEY=...`).
`trip`/`reset` reach the boxes through the mirror: up to 1 min for the next
heartbeat plus up to 1 min for each process's poll. Verify with
`fleet.sh status`, the `/fleet/` page's breaker card, or a lane's log lines
(`[killswitch] R2 writes DISABLED (...)`, one per minute at most, and one
`ENABLED again` line when they resume).

---

## Capacity: how to decide what goes where

The currency is **cut CPU**, measured per product-slot, and the binding
constraint on an emit box is usually **RAM**, not cores.

* a 305-tile full-disk product-slot costs ~39 s and peaks ~4.5 GB
* a lane saturates ~1 core per `S2_CUT_WORKERS` while cutting
* an 8 vCPU / 31 GB box holds ~5 heavy lanes; eight tripped the *host* OOM
  killer on box1 (`constraint=CONSTRAINT_NONE, global_oom`)
* sustained rate for a serial lane: `products x slots_per_hour x 39 s` against
  3600 s. Over ~0.8x real time, the lane will not hold its grid.

Placement rules of thumb:

* **Region-locked work goes where its source is.** Nothing in the satellite
  emit path is: GK-2A reads `noaa-gk2a-pds`, MTG reads the EUMETSAT Data
  Store, the GEO composite reads the other members' output from R2. Those
  lanes are placement-free and are the first things to move when rebalancing.
* **Live-request work pins the box it serves.** box1 answers cyclolab requests,
  so its spare capacity is worth more than an emit-only box's — prefer moving
  emit lanes off box1 rather than adding to it.
* **Split a satellite's leads from its browse bands** when RAM allows: the five
  animated products in their own container get a ~3 min pass (latency tracks
  the scan) while the 22 browse bands sweep separately.

---

## Reference

```bash
scripts/fleet.sh status                 # per-box sha, load, RAM, lanes
scripts/fleet.sh drift                  # git rule, enforced
scripts/fleet.sh lanes <box>            # assignment map for one box
scripts/fleet.sh provision <box>        # bare box -> ready (idempotent)
scripts/fleet.sh deploy <box|all>       # pull main, rebuild, recreate lanes
scripts/fleet.sh setenv <box|all> K=V   # secret propagation
scripts/fleet.sh writes off|on [box|all]   # R2 write kill switch (env + recreate of write lanes)
scripts/fleet.sh breaker status|arm|disarm|trip|reset   # the r2-breaker Worker
```

Health page: `/fleet/` (unlinked, noindexed, robots-disallowed).
Per-lane cadence design and the grid arithmetic: the header comment of each
`docker-compose.s2.*-lane.yml`.
