#!/usr/bin/env python3
"""Stage-2 shadow pyramid emitter -- END-TO-END runner (Phase 2a/3).

Fetch frames for tiled ProductEntry rows, reproduce the FROZEN renderer's
chrome-free imagery (s2_imagery), cut a 512 px XYZ WebP pyramid + a slider
latest_times.json (s2_pyramid), and write it to a SHADOW R2 prefix --
registry-driven, additive, and NOT touching the running S1 shadow.

Phase 3 adds the multi-product imagery suite: any entry with a `recipe_id`
routes through the declarative recipe engine (s2_recipes + s2_imagery), and
`--suite` emits EVERY tiled product of a sector off ONE pinned scan with a
shared band cache (each ABI band is downloaded once per scan, not once per
product). A `products.json` index (the viewer picker's on-R2 SSOT) is
refreshed after every emit.

STORES (the injected R2 interface, key-for-key):
  * --store local:/path  -> FilesystemStore (Codespace verification, no creds)
  * --store r2           -> s1_ingest.R2 (needs R2_ENDPOINT/R2_ACCESS_KEY_ID/
                            R2_SECRET_ACCESS_KEY in env -- the Hostinger box, or
                            a Codespace with the shadow token exported)

Examples
--------
  # one product, local:
  python s2_pyramid_emit.py --product goes19-conus-ir --store local:/tmp/shadow

  # one product, real shadow R2 (the box):
  python s2_pyramid_emit.py --product goes19-conus-airmass --store r2 --prefix shadow

  # the WHOLE CONUS imagery suite (one scan, shared band fetches), cron-tiered:
  python s2_pyramid_emit.py --suite conus --store r2 --prefix shadow --max-zoom 5
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import sys
import traceback

import s2_imagery as I
import s2_pyramid as P
import s2_registry as R

UTC = dt.timezone.utc
CDN = "https://cdn.triple-a-tropics.com"


def _parse_stamp(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def _backfill_slots(step_min: int, backfill_min: int,
                    now: dt.datetime | None = None) -> list:
    """The step-minute slot grid covering the trailing backfill window,
    NEWEST first (the enscenters/§11-H lesson: oldest-first freezes 'latest'
    behind the backfill — a ~20-min geo slot render walking old→new left the
    world composite 75+ min stale, and a timeout kill lost the newest slot
    entirely). Newest-first makes freshness independent of backfill depth:
    the current slot lands first, older holes heal with whatever budget
    remains, and a killed run costs only old-slot work (ready-marker dedup
    resumes it next tick). Slots are grid times (:00/:10/:20 for step 10);
    the actual scan each slot renders is resolved nearest-to-slot by the
    fetchers. Emit order is safe to change: manifests rebuild from R2
    reality per emit and frames dedup by ready marker, so no consumer
    depends on slot arrival order."""
    now = now or dt.datetime.now(UTC)
    newest = now.replace(second=0, microsecond=0)
    newest -= dt.timedelta(minutes=newest.minute % step_min)
    slots, t = [], newest
    floor = now - dt.timedelta(minutes=backfill_min)
    while t >= floor:
        slots.append(t)
        t -= dt.timedelta(minutes=step_min)
    return slots


def _covered_times(entries, store, prefix: str) -> list:
    """Per entry: the sorted datetimes of its COMPLETE frames in the store."""
    out = []
    for e in entries:
        ts = []
        for s, _mz in P.complete_stamps(e, store, prefix):
            try:
                ts.append(_parse_stamp(s))
            except ValueError:
                pass                     # legacy/foreign stamp shape: ignore
        out.append(sorted(ts))
    return out


def _slot_missing(slot: dt.datetime, covered: list, tol: dt.timedelta) -> bool:
    """A slot is DONE only when EVERY entry has a complete frame within tol
    of it (per-product isolation: one lagging product keeps its slot open
    without forcing re-renders of the healthy ones -- dedup skips those)."""
    for ts in covered:
        if not any(abs(t - slot) <= tol for t in ts):
            return True
    return False


def _make_store(spec: str):
    if spec == "r2":
        import s1_ingest  # lazy: only import boto3/R2 when actually writing R2
        if not s1_ingest.R2_ENDPOINT:
            sys.exit("ERROR: --store r2 needs R2_ENDPOINT (+ R2 keys) in env. "
                     "On the Codespace use --store local:/path instead.")
        return s1_ingest.R2(), True
    if spec.startswith("local:"):
        return P.FilesystemStore(spec.split(":", 1)[1]), False
    sys.exit(f"ERROR: --store must be 'r2' or 'local:/path', got {spec!r}")


def emit_one(entry, when, store, args, band_cache=None) -> dict:
    """Fetch+render+tile+manifest for ONE tiled entry. Returns the manifest."""
    spec = P.PyramidSpec(tile_size=entry.tile_size, quality=args.quality)
    scheme = args.scheme or entry.pyramid_scheme

    print(f"[fetch] {entry.product_id}  time={when.isoformat() if when else 'latest'}  "
          f"bbox={entry.sector_bbox}  scheme={scheme}")
    if entry.sat_key == "geo":
        img = I.produce_global_composite(entry, time=when, nearest=True,
                                         band_cache=band_cache)
    elif entry.recipe_id:
        img = I.produce_recipe_imagery(entry, time=when, nearest=True,
                                       band_cache=band_cache)
    else:
        img = I.produce_imagery(entry, time=when, nearest=True)
    print(f"[imagery] {img.product} {img.s3_key.split('/')[-1]}  stamp={img.stamp}")
    print(f"[imagery] raster={img.rgba.shape[1]}x{img.rgba.shape[0]}  "
          f"bounds(W,S,E,N)={tuple(round(b, 3) for b in img.bounds)}")

    # Calibrated BT data raster beside the tiles (pixel/BT inspector, §6).
    bt_png = bt_desc = None
    if getattr(img, "bt_grid", None) is not None:
        import s2_bt
        bt_png = s2_bt.encode_bt_png(img.bt_grid)
        bt_desc = s2_bt.bt_descriptor(entry.product_path, img.bounds, img.bt_dims)
        print(f"[bt] calibrated BT raster {img.bt_dims[0]}x{img.bt_dims[1]}  "
              f"({len(bt_png)//1024} KB)")

    meta = P.emit_pyramid(entry, store, args.prefix, img.stamp, img.rgba,
                          img.bounds, spec, scheme=scheme, bt_png=bt_png,
                          max_zoom=args.max_zoom)
    if meta["outcome"] == "duplicate":
        print(f"[emit] duplicate -- {img.stamp} already present, skipped")
        if args.max_zoom is None:
            # The Q7 on-demand-z6 gotcha: a scan the z5 cron already emitted
            # dedups on the ready marker, so an uncapped re-emit writes NOTHING.
            print("[emit] NOTE: frames dedup by ready marker -- to re-cut this "
                  "scan at native zoom use a DIFFERENT --prefix (e.g. shadow-z6)")
    else:
        print(f"[emit] wrote {meta['n_tiles']} tiles  maxzoom={meta['maxzoom']}  "
              f"per-zoom={meta['tile_counts']}")

    # Rebuild the manifest from R2 reality so re-runs accumulate a real series.
    # The current frame's maxzoom is read from its OWN tiles in the store
    # (scheme-agnostic; works on a 'duplicate' too); times[] then keeps only
    # COMPLETE frames at that same maxzoom, so the viewer's grid derivation
    # never 404s. A frame at a different geometry is dropped + logged.
    image_px = [img.rgba.shape[1], img.rgba.shape[0]]
    frames = P.complete_stamps(entry, store, args.prefix)   # [(stamp, maxzoom)]
    maxzoom = dict(frames).get(img.stamp, meta.get("maxzoom") or 0)
    times = [s for s, mz in frames if mz == maxzoom]
    dropped = [s for s, mz in frames if mz != maxzoom]
    if img.stamp not in times:
        times.append(img.stamp)
    if args.keep and len(times) > args.keep:
        times = sorted(times)[-args.keep:]   # viewer window; R2 keeps the rest until TTL
    if dropped and not args.allow_geometry_change:
        # Refuse rather than clobber: an uncapped z6 emit against the z5 cron
        # prefix would rebuild the manifest around ONE frame and discard the
        # whole loop (until the next cron tick reversed it). RuntimeError (not
        # sys.exit) so suite mode's per-product isolation contains it.
        raise RuntimeError(
            f"{entry.product_id}: this emit (maxzoom={maxzoom}) would DROP "
            f"{len(dropped)} existing frame(s) at a different pyramid geometry "
            f"(e.g. {dropped[:3]}). For an on-demand deep-zoom cut use a "
            f"different --prefix; for a deliberate geometry migration pass "
            f"--allow-geometry-change.")
    if dropped:
        print(f"[manifest] WARNING: dropped {len(dropped)} frame(s) at a DIFFERENT "
              f"pyramid geometry (maxzoom != {maxzoom}); a scheme/pyramid_px change "
              f"needs a fresh prefix or a full re-emit. e.g. {dropped[:3]}")
    manifest = P.write_tiled_manifest(entry, store, args.prefix, times,
                                      img.bounds, image_px, maxzoom,
                                      dt.datetime.now(UTC), spec=spec, scheme=scheme,
                                      bt=bt_desc,
                                      members=getattr(img, "members", None))
    print(f"[done] {entry.product_id}: frames={manifest['count']} "
          f"latest={manifest['latest']} zoom={manifest['minzoom']}..{manifest['maxzoom']}")
    return manifest


def _write_products_index(store, prefix: str, sat_key: str, sector_key: str):
    idx = R.build_products_index(sat_key, sector_key, dt.datetime.now(UTC))
    key = R.products_index_key(prefix, sat_key, sector_key)
    store.put_json(key, idx, P.CACHE_MANIFEST)
    print(f"[index] {key}: {idx['count']} products")
    return key


def _pin_suite_scan(entries, when):
    """Resolve the scan anchor ONCE so every suite product renders the SAME
    scan (a 'latest' that drifts to a newer scan mid-suite would split the
    suite across two times). GOES anchors on the clean-IR file (fd sectors
    bypass find_file's CONUS-first pick); Himawari resolves the newest FLDK
    slot with a COMPLETE segment set for EVERY band any suite product needs
    (NOAA uploads segments over minutes -- pinning an incomplete slot would
    stitch short windows)."""
    t = when or dt.datetime.now(UTC)
    e0 = entries[0]
    bbox = list(e0.sector_bbox)

    if e0.sat_key == "geo":
        # the composite fetches each ring member's newest scan itself; the
        # pin is just a shared reference time for the suite tick
        return t.replace(second=0, microsecond=0)

    if e0.family == "himawari":
        from satellites import HIMAWARI_PACIFIC
        sat = HIMAWARI_PACIFIC
        need = sorted({b for e in entries for b in e.bands})
        if when is not None:
            return sat._snap_10min(t, True)
        base = t.replace(second=0, microsecond=0)
        floored = base.replace(minute=(base.minute // 10) * 10)
        slot = sat._first_available_fldk_slot_sync(floored, need)
        if slot is None:
            sys.exit(f"ERROR: no complete AHI FLDK slot for bands {need} "
                     f"near {t.isoformat()}")
        return slot

    if e0.family == "gk2a":
        import s2_gk2a
        need = sorted({b for e in entries for b in e.bands})
        names = [s2_gk2a.BAND_TOKENS[b] for b in need]
        slot = s2_gk2a.newest_complete_slot(names, time=when, nearest=True)
        if slot is None:
            sys.exit(f"ERROR: no complete GK-2A AMI FD slot for bands {need} "
                     f"near {t.isoformat()}")
        return slot

    if e0.family == "mtgi1":
        # OpenSearch needs no creds, so pinning works even while the
        # download is still licence-gated; the pin is the product's
        # sensing END (one FDHSI product carries every band).
        import s2_meteosat as MET
        slot = MET.newest_fci_slot(time=when)
        if slot is None:
            sys.exit(f"ERROR: no licence-compliant MTG FCI repeat cycle "
                     f"near {t.isoformat()}")
        return slot

    from satellites import GOESEastSatellite
    sat = GOESEastSatellite()
    if (e0.render_product_hint or "").lower() == "fd":
        resolved = asyncio.run(sat._pick_full_disk(e0.bucket, bbox, 13, t, True))
        if resolved is None:
            sys.exit(f"ERROR: no full-disk C13 scan near {t.isoformat()}")
    else:
        resolved = asyncio.run(sat.find_file(t, "clean_ir", bbox, nearest_to_target=True))
    return resolved.scan_start


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage-2 shadow pyramid emitter")
    ap.add_argument("--product", default=None,
                    help="tiled ProductEntry id, e.g. goes19-conus-ir | goes19-conus-airmass")
    ap.add_argument("--suite", default=None, metavar="SECTOR",
                    help="emit EVERY tiled product of a sector off one scan. "
                         "Bare sectors are goes19 ('conus', 'fd'); other "
                         "satellites qualify with the sat key "
                         "('himawari9-fd', 'himawari9-wpac')")
    ap.add_argument("--time", default="latest",
                    help="'latest' or an ISO UTC time (nearest scan is used)")
    ap.add_argument("--prefix", default="shadow", help="R2 prefix (default: shadow)")
    ap.add_argument("--store", default="local:/tmp/tat-shadow",
                    help="'r2' or 'local:/path' (default: local:/tmp/tat-shadow)")
    ap.add_argument("--pyramid-px", type=int, default=None,
                    help="override the raster long-edge (drives maxzoom; single-product only)")
    ap.add_argument("--quality", type=int, default=90,
                    help="WebP quality (default 90 -- rainbow_ir colortable edges, §4.2)")
    ap.add_argument("--scheme", default=None,
                    choices=["flat-native-xyz", "webmercator-xyz"],
                    help="tile scheme (default: the product's pyramid_scheme)")
    ap.add_argument("--max-zoom", type=int, default=None,
                    help="cap the pyramid maxzoom (cron uses 5 for z0-5; omit for native z6+ on-demand)")
    ap.add_argument("--keep", type=int, default=90,
                    help="manifest lists only the newest N frames (default 90, the "
                         "export window; older tiles stay until the R2 lifecycle TTL)")
    ap.add_argument("--step", type=int, default=None, metavar="MIN",
                    help="target loop cadence: instead of emitting ONLY the newest "
                         "scan, emit every missing MIN-minute slot in the trailing "
                         "--backfill window (oldest first). A slot already covered "
                         "by a complete frame within MIN/2 is skipped, so re-runs "
                         "are cheap and a slow/killed pass self-heals next tick. "
                         "Without --step the old newest-scan-only behavior holds.")
    ap.add_argument("--backfill", type=int, default=None, metavar="MIN",
                    help="lookback window for --step, minutes (default 6x step)")
    ap.add_argument("--allow-geometry-change", action="store_true",
                    help="permit an emit whose maxzoom differs from the prefix's "
                         "existing frames (drops them from the manifest); without "
                         "this the emit refuses instead of clobbering the loop")
    args = ap.parse_args(argv)

    if bool(args.product) == bool(args.suite):
        sys.exit("ERROR: pass exactly one of --product or --suite SECTOR")
    if args.step and args.time != "latest":
        sys.exit("ERROR: --step (slot backfill) and --time are mutually "
                 "exclusive -- a pinned time IS a single slot")
    if args.step and args.step <= 0:
        sys.exit("ERROR: --step must be a positive number of minutes")

    when = None
    if args.time != "latest":
        when = dt.datetime.fromisoformat(args.time)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)

    store, is_r2 = _make_store(args.store)

    if args.product:
        entry = R.REGISTRY_BY_ID.get(args.product)
        if entry is None:
            sys.exit(f"ERROR: unknown product {args.product!r}. Tiled products: "
                     + ", ".join(e.product_id for e in R.REGISTRY if e.tiled))
        if not entry.tiled:
            sys.exit(f"ERROR: {args.product} is not a tiled product (tiled=False)")
        if args.pyramid_px:
            entry = dataclasses.replace(entry, pyramid_px=args.pyramid_px)
        if args.step:
            slots = _backfill_slots(args.step, args.backfill or args.step * 6)
            tol = dt.timedelta(minutes=args.step / 2.0)
            covered = _covered_times([entry], store, args.prefix)
            todo = [s for s in slots if _slot_missing(s, covered, tol)]
            print(f"[backfill] {entry.product_id}: {len(todo)}/{len(slots)} "
                  f"slot(s) missing at step {args.step}m")
            n_ok = n_fail = 0
            for slot in todo:
                # re-check against frames written THIS run: two adjacent slots
                # can resolve to the same scan near the sat's native cadence
                covered = _covered_times([entry], store, args.prefix) if n_ok else covered
                if not _slot_missing(slot, covered, tol):
                    print(f"[backfill] {slot.isoformat()} now covered -- skip")
                    continue
                try:
                    emit_one(entry, slot, store, args)
                    n_ok += 1
                except Exception as e:   # noqa: BLE001 -- one bad slot must not
                    n_fail += 1          # kill the rest of the window
                    print(f"[FAIL] {entry.product_id} @ {slot.isoformat()}: {e}")
                    traceback.print_exc()
            if n_ok or not n_fail:
                _write_products_index(store, args.prefix, entry.sat_key,
                                      entry.sector_key)
            else:
                print("[index] SKIPPED (total failure): products.json left "
                      "untouched")
            print(f"\n=== BACKFILL SUMMARY === ok={n_ok} failed={n_fail} "
                  f"skipped={len(slots) - len(todo)}")
            return 1 if (n_fail and not n_ok) else 0
        manifest = emit_one(entry, when, store, args)
        _write_products_index(store, args.prefix, entry.sat_key, entry.sector_key)
        mkey = entry.latest_times_key(args.prefix)
        print("\n=== SHADOW PYRAMID WRITTEN ===")
        print(f" product      : {entry.product_id}  ({entry.product_path})")
        print(f" manifest key : {mkey}")
        if is_r2:
            print(f" manifest URL : {CDN}/{mkey}")
        return 0

    # ---- suite mode: every tiled product of the sector, ONE pinned scan ----
    # bare sector = goes19 (back-compat with the box cron); '<sat>-<sector>'
    # qualifies another satellite (himawari9-fd, himawari9-wpac).
    sat_keys = sorted({e.sat_key for e in R.REGISTRY if e.tiled})
    suite_sat, suite_sector = "goes19", args.suite
    for sk in sat_keys:
        if args.suite.startswith(sk + "-"):
            suite_sat, suite_sector = sk, args.suite[len(sk) + 1:]
            break
    entries = [e for e in R.REGISTRY
               if e.tiled and e.sat_key == suite_sat
               and e.sector_key == suite_sector]
    if not entries:
        sys.exit(f"ERROR: no tiled products for suite {args.suite!r}. Suites: "
                 + ", ".join(sorted({("%s-%s" % (e.sat_key, e.sector_key))
                                     if e.sat_key != "goes19" else e.sector_key
                                     for e in R.REGISTRY if e.tiled})))
    if args.step:
        slots = _backfill_slots(args.step, args.backfill or args.step * 6)
        tol = dt.timedelta(minutes=args.step / 2.0)
        covered = _covered_times(entries, store, args.prefix)
        todo = [s for s in slots if _slot_missing(s, covered, tol)]
        print(f"[backfill] suite {args.suite}: {len(todo)}/{len(slots)} "
              f"slot(s) missing at step {args.step}m")
    else:
        todo = [when]                     # legacy: one tick, newest (or --time)

    ok_all, failed_all = [], []
    for slot in todo:
        pinned = _pin_suite_scan(entries, slot)
        print(f"[suite] {args.suite}: {len(entries)} products @ scan {pinned.isoformat()}")
        band_cache: dict = {}             # per-scan: bands download once per slot
        for entry in entries:
            try:
                emit_one(entry, pinned, store, args, band_cache=band_cache)
                ok_all.append(entry.product_id)
            except Exception as e:   # noqa: BLE001  (per-product isolation: one bad
                # band/recipe never kills the rest of the suite; non-zero exit below)
                failed_all.append(entry.product_id)
                print(f"[FAIL] {entry.product_id}: {e}")
                traceback.print_exc()
    if ok_all or not failed_all:
        _write_products_index(store, args.prefix, suite_sat, suite_sector)
    else:
        # total failure: never advertise a product list with zero frames
        # behind it (a still-gated suite -- e.g. FCI pre-licence -- would
        # un-grey the explorer domain and 404 every manifest). The viewer's
        # availability probe stays honestly dark until an emit succeeds.
        print(f"[index] SKIPPED (total failure -- {len(failed_all)} products, "
              f"0 ok): products.json left untouched")

    print("\n=== SUITE EMIT SUMMARY ===")
    print(f" slots  : {len(todo)}")
    print(f" ok     : {len(ok_all)}  {sorted(set(ok_all))}")
    print(f" failed : {len(failed_all)}  {sorted(set(failed_all))}")
    if is_r2 and ok_all:
        print(f" index  : {CDN}/{R.products_index_key(args.prefix, suite_sat, suite_sector)}")
    if failed_all and not ok_all:
        return 1          # total failure aborts (mirror the HAFS total-failure rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
