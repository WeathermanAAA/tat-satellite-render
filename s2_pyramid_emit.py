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

    print(f"[fetch] {entry.product_id}  time={args.time}  bbox={entry.sector_bbox}")
    img = I.produce_imagery(entry, time=when, nearest=True)
    print(f"[imagery] {img.product} {img.s3_key.split('/')[-1]}  stamp={img.stamp}")
    print(f"[imagery] raster={img.rgba.shape[1]}x{img.rgba.shape[0]}  "
          f"bounds(W,S,E,N)={tuple(round(b,3) for b in img.bounds)}")

    meta = P.emit_pyramid(entry, store, args.prefix, img.stamp, img.rgba,
                          img.bounds, spec)
    if meta["outcome"] == "duplicate":
        print(f"[emit] duplicate -- {img.stamp} already present, skipped")
    else:
        print(f"[emit] wrote {meta['n_tiles']} tiles  maxzoom={meta['maxzoom']}  "
              f"per-zoom={meta['tile_counts']}")

    # Rebuild the manifest from R2 reality so re-runs accumulate a real series.
    # The current frame's geometry (from its raster, not the emit outcome) is the
    # advertised top-level; times[] includes ONLY COMPLETE frames cut at the SAME
    # maxzoom, so the viewer's per-frame grid derivation never 404s. A frame at a
    # different pyramid geometry (a pyramid_px change) is dropped + logged loudly.
    image_px = [img.rgba.shape[1], img.rgba.shape[0]]
    maxzoom = P.max_zoom_for(image_px[0], image_px[1], spec.tile_size)
    frames = P.complete_stamps(entry, store, args.prefix)   # [(stamp, maxzoom)]
    times = [s for s, mz in frames if mz == maxzoom]
    dropped = [s for s, mz in frames if mz != maxzoom]
    if img.stamp not in times:
        times.append(img.stamp)
    if dropped:
        print(f"[manifest] WARNING: dropped {len(dropped)} frame(s) at a DIFFERENT "
              f"pyramid geometry (maxzoom != {maxzoom}); a pyramid_px change needs a "
              f"fresh prefix or a full re-emit. e.g. {dropped[:3]}")
    manifest = P.write_tiled_manifest(entry, store, args.prefix, times,
                                      img.bounds, image_px, maxzoom,
                                      dt.datetime.now(UTC), spec=spec)

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
