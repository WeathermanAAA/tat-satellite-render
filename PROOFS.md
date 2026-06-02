# Intensity poller - Stage A offline proofs

Stage A builds + proves the streaming intensity poller OFFLINE. **No Railway
deploy, no prod-R2 write, no cutover** - the 6 h cron still owns the live
`wp/al/ep_{ace,tracks}_data.json` feeds. This is a freshness change, not a
data-content change.

## Architecture (base / live split, sole-writer model)

- **Cron base-writer** (main repo, `build_feed_base.py`): writes the slow-moving
  archive to R2 `feeds/base/{basin}_{ace,tracks}_base.json` - the per-season ACE
  curves, climo bands, past `storms_by_year`, ranking backbone, and the
  current-year IBTrACS backbone. Additive (runs alongside the existing live-feed
  write); daily cadence is fine.
- **Intensity poller** (this repo, `intensity_poller.py` + `feed_recompute.py`):
  one `PollerEngine` (on `poller_framework`) with a `Source` per basin. Each
  cycle it reads its base from R2, fetches ONLY the fresh live b-decks via the
  SAME proxy chain the generators use, parses them with the FROZEN
  `ace_core.parse_bdeck`, merges onto the base's current-season canon with
  `ace_core.merge_named_sources`, and recomputes the live feed (current curve +
  ytd@doy + rank + storms[] + header + freshness) with `ace_core`'s shared
  assembly - ONLY when a new fix lands. ace_core is pinned at
  `ace-core-v0.1.0`, so ACE numbers cannot shift under the poller.

The poller owns ONLY the fresh slice. It never rebuilds the historical archive,
never touches climo or `/historical`, never alters ACE methodology, and adds
nothing to ACE/tracks/climo that was not already there.

## The four proofs

Self-contained offline proofs (synthetic fixtures, no network) live in
`tests/test_intensity_poller.py` (run `python tests/test_intensity_poller.py -v`,
8 tests). The full cron-parity proof needs the main-repo cron + a live b-deck
snapshot and was run once; its result is recorded below.

1. **SHAPE** - the recomputed `ace` / `tracks` feeds carry EXACTLY the live
   feed's key set (the keys the frontend reads). build_payload is the same
   assembler the cron uses, so the keys match by construction.

2. **CORRECTNESS** - run on the SAME fixes the cron sees, the poller reproduces
   the cron's feeds BYTE-FOR-BYTE (only the build-timestamp fields differ), and
   the cross-feed invariant holds exactly:
   `ace.current.latest_value == sum(per-storm ACE) == tracks.header.total_ace`.
   Captured run (WP, the captured live b-deck frame; 6 named storms incl.
   06W/JANGMI): poller == cron for both feeds (data fields identical), invariant
   = **51.228** three ways, matching the cron's 51.228.

3. **FRESHNESS** - the poller's `latest_fix_valid_utc` tracks the newest
   advisory. Against the real live WP b-decks, 06W/JANGMI's newest fix is
   reflected immediately; and the moment a fresh advisory lands the
   `staleness_minutes` is MINUTES (simulated 06Z fix at 06:05Z -> staleness 5),
   whereas the 6 h cron's feed would stay hours behind until its next run. The
   poller reflects a new fix within one poll (~60 s).

4. **ISOLATION** - one basin's b-deck fetch failing NEVER freezes or stales the
   others. Proven across two cycles: wp succeeds (cycle 1, publishes), then wp's
   fetch raises (cycle 2) - al/ep still refresh, wp's last-known-good feed +
   change-signature are preserved (no half-write), the health heartbeat still
   fires, and no exception escapes `poll_once`. Resilient fetch retries with
   backoff before giving up. This is the stale-pill / silent-freeze lesson
   designed out.

## Reproduce the cron-parity proof

```
# 1. build the base (main repo, needs the IBTrACS CSVs):
python build_feed_base.py            # writes feeds/base/{basin}_*_base.json

# 2. capture one live frame + run the cron with it injected (main repo)
# 3. run the poller recompute on (base + same live frame) and diff:
#    feed_recompute.recompute_ace_feed / recompute_tracks_feed
# -> identical to the cron (modulo generated_utc / staleness / updated).
```

## NOT in Stage A (Stage B, a separate prompt after review)

Railway deploy + R2 sink, the deliberate cron -> poller cutover, retiring the
git-commit path, and verifying live staleness drops. Also: invest tracks
(the cron's knackwx invest source) must be preserved at cutover so no content is
lost.
