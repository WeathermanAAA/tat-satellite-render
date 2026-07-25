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
soon as it is in the inventory**, with nothing to edit in the site repo. It carries: git sha + branch +
dirty count (so drift shows up on the health page too), load per core, memory,
disk, **OOM kills since boot** (the failure mode that actually bites an emit
box), and which lanes are up vs exited.

Consumers must judge freshness on the heartbeat's own `ts`. A stale object
means a dead box no matter how healthy its contents look — the page treats
>3 min as stale and >10 min as silent.

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
```

Health page: `/fleet/` (unlinked, noindexed, robots-disallowed).
Per-lane cadence design and the grid arithmetic: the header comment of each
`docker-compose.s2.*-lane.yml`.
