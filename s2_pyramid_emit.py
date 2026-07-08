#!/usr/bin/env python3
"""Stage-2 shadow pyramid emitter -- END-TO-END runner (Phase 2a).

Fetch ONE clean-IR frame for a tiled ProductEntry, reproduce the FROZEN
renderer's chrome-free imagery (s2_imagery), cut a 512 px flat-native XYZ WebP
pyramid + a SLIDER latest_times.json (s2_pyramid), and write it to a SHADOW R2
prefix -- registry-driven, additive, and NOT touching the running S1 shadow.

STORES (the injected R2 interface, key-for-key):
  * --store local:/path  -> FilesystemStore (Codespace verification, no creds)
  * --store r2           -> s1_ingest.R2 (needs R2_ENDPOINT/R2_ACCESS_KEY_ID/
                            R2_SECRET_ACCESS_KEY in env -- the Hostinger box, or
                            a Codespace with the shadow token exported)

Examples
--------
  # Codespace, no creds -- write the pyramid to a local shadow tree:
  python s2_pyramid_emit.py --product goes19-conus-ir --store local:/tmp/shadow

  # Hostinger box (or creds exported) -- write the real shadow R2 prefix:
  python s2_pyramid_emit.py --product goes19-conus-ir --store r2 --prefix shadow
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import sys

import s2_imagery as I
import s2_pyramid as P
import s2_registry as R

UTC = dt.timezone.utc
CDN = "https://cdn.triple-a-tropics.com"


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage-2 shadow pyramid emitter (Phase 2a)")
    ap.add_argument("--product", required=True,
                    help="tiled ProductEntry id, e.g. goes19-conus-ir | goes19-fd-ir")
    ap.add_argument("--time", default="latest",
                    help="'latest' or an ISO UTC time (nearest scan is used)")
    ap.add_argument("--prefix", default="shadow", help="R2 prefix (default: shadow)")
    ap.add_argument("--store", default="local:/tmp/tat-shadow",
                    help="'r2' or 'local:/path' (default: local:/tmp/tat-shadow)")
    ap.add_argument("--pyramid-px", type=int, default=None,
                    help="override the raster long-edge (drives maxzoom)")
    ap.add_argument("--quality", type=int, default=90,
                    help="WebP quality (default 90 -- rainbow_ir colortable edges, §4.2)")
    ap.add_argument("--scheme", default=None,
                    choices=["flat-native-xyz", "webmercator-xyz"],
                    help="tile scheme (default: the product's pyramid_scheme)")
    ap.add_argument("--max-zoom", type=int, default=None,
                    help="cap the pyramid maxzoom (cron uses 5 for z0-5; omit for native z6+ on-demand)")
    args = ap.parse_args(argv)

    entry = R.REGISTRY_BY_ID.get(args.product)
    if entry is None:
        sys.exit(f"ERROR: unknown product {args.product!r}. Tiled products: "
                 + ", ".join(e.product_id for e in R.REGISTRY if e.tiled))
    if not entry.tiled:
        sys.exit(f"ERROR: {args.product} is not a tiled product (tiled=False)")
    if args.pyramid_px:
        entry = dataclasses.replace(entry, pyramid_px=args.pyramid_px)

    when = None
    if args.time != "latest":
        when = dt.datetime.fromisoformat(args.time)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)

    store, is_r2 = _make_store(args.store)
    spec = P.PyramidSpec(tile_size=entry.tile_size, quality=args.quality)
    scheme = args.scheme or entry.pyramid_scheme

    print(f"[fetch] {entry.product_id}  time={args.time}  bbox={entry.sector_bbox}  scheme={scheme}")
    img = I.produce_imagery(entry, time=when, nearest=True)
    print(f"[imagery] {img.product} {img.s3_key.split('/')[-1]}  stamp={img.stamp}")
    print(f"[imagery] raster={img.rgba.shape[1]}x{img.rgba.shape[0]}  "
          f"bounds(W,S,E,N)={tuple(round(b,3) for b in img.bounds)}")

    # Calibrated BT data raster beside the tiles (pixel/BT inspector, §6).
    bt_png = bt_desc = None
    if getattr(img, "bt_grid", None) is not None:
        import s2_bt
        bt_png = s2_bt.encode_bt_png(img.bt_grid)
        bt_desc = s2_bt.bt_descriptor(entry.product_path, img.bounds, img.bt_dims)
        print(f"[bt] calibrated BT raster {img.bt_dims[0]}x{img.bt_dims[1]}  ({len(bt_png)//1024} KB)")

    meta = P.emit_pyramid(entry, store, args.prefix, img.stamp, img.rgba,
                          img.bounds, spec, scheme=scheme, bt_png=bt_png,
                          max_zoom=args.max_zoom)
    if meta["outcome"] == "duplicate":
        print(f"[emit] duplicate -- {img.stamp} already present, skipped")
    else:
        print(f"[emit] wrote {meta['n_tiles']} tiles  maxzoom={meta['maxzoom']}  "
              f"per-zoom={meta['tile_counts']}")

    # Rebuild the manifest from R2 reality so re-runs accumulate a real series.
    # The current frame's maxzoom is read from its OWN tiles in the store
    # (scheme-agnostic -- flat-native and webmercator have different maxzoom
    # rules, and it works on a 'duplicate' too); times[] then keeps only COMPLETE
    # frames at that same maxzoom, so the viewer's grid derivation never 404s. A
    # frame at a different geometry (scheme/pyramid_px change) is dropped + logged.
    image_px = [img.rgba.shape[1], img.rgba.shape[0]]
    frames = P.complete_stamps(entry, store, args.prefix)   # [(stamp, maxzoom)]
    maxzoom = dict(frames).get(img.stamp, meta.get("maxzoom") or 0)
    times = [s for s, mz in frames if mz == maxzoom]
    dropped = [s for s, mz in frames if mz != maxzoom]
    if img.stamp not in times:
        times.append(img.stamp)
    if dropped:
        print(f"[manifest] WARNING: dropped {len(dropped)} frame(s) at a DIFFERENT "
              f"pyramid geometry (maxzoom != {maxzoom}); a scheme/pyramid_px change "
              f"needs a fresh prefix or a full re-emit. e.g. {dropped[:3]}")
    manifest = P.write_tiled_manifest(entry, store, args.prefix, times,
                                      img.bounds, image_px, maxzoom,
                                      dt.datetime.now(UTC), spec=spec, scheme=scheme,
                                      bt=bt_desc)

    mkey = entry.latest_times_key(args.prefix)
    sample = entry.tile_key(args.prefix, img.stamp, maxzoom, 0, 0)
    print("\n=== SHADOW PYRAMID WRITTEN ===")
    print(f" store        : {args.store}")
    print(f" product      : {entry.product_id}  ({entry.product_path})")
    print(f" manifest key : {mkey}")
    print(f" sample tile  : {sample}")
    print(f" frames       : {manifest['count']}  latest={manifest['latest']}")
    print(f" zoom         : {manifest['minzoom']}..{manifest['maxzoom']}  "
          f"tile={manifest['tile_size']}px  scheme={manifest['scheme']}")
    if is_r2:
        print(f" manifest URL : {CDN}/{mkey}")
        print(f" sample URL   : {CDN}/{sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
