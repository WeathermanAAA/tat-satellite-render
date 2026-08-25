#!/usr/bin/env python3
"""s2_coverage_equivalence.py -- the pre-canary gate for coverage-from-manifest
(2026-08-25, change C of the R2 cost audit).

Runs ON a box (read-only against R2, same env as the lanes: no PUT, no
DELETE, nothing written under STATE_DIR). For every product it computes the
--step loops' `covered` set BOTH ways with the SAME window:

  * manifest : s2_pyramid_emit._covered_from_manifest  (1 Class B GET)
  * listing  : s2_pyramid_emit._covered_times           (the pre-change path:
               1 tail listing + 1 LIST page per stamp in the window)

and prints per product |manifest|, |listing|, listing-manifest split into
ORPHANS (complete in R2 at the manifest's geometry, not advertised -- under
the new path they self-heal within one pass, so re-run after a lane pass to
confirm they clear) and STRAYS (complete at a FOREIGN geometry -- never
advertised by design; the lane memos them after one fetch and they clear only
when they age out of the window or are deleted), manifest-listing
(ADVERTISED-BUT-NOT-IN-R2: must be 0 or explained -- e.g. a frame deleted
under a live manifest), its own op counts, and one verdict line:

  EQUIVALENT   every manifest usable AND both differences empty
  DIFFERS      anything else (the per-product lines say what)

Boundary: each function derives its window start from its own now(); a stamp
landing in the seconds between the two calls would show as a spurious
difference, so the set comparison is made on the common inner window
[t0 - window + 120 s, now] (t0 = this product's start time); the raw sizes
are printed exactly as each function returned them. S2_COVER_FROM_MANIFEST is
forced to "1" in THIS process so the gate exercises the manifest path even
while the lanes are rolled back to the listing.

Usage: INSIDE the tat-s2 image (the box host python3 has no numpy --
provision_box.sh installs only python3-yaml/boto3 -- and this module imports
s2_pyramid_emit -> s2_imagery). The `emit` service carries env_file .env and
WORKDIR /app; build first so the image holds this script:

  cd /root/tsr-s2 \
    && docker compose -f docker-compose.s2.yml build -q emit-cron \
    && docker compose -f docker-compose.s2.yml run --rm --entrypoint python \
         emit scripts/s2_coverage_equivalence.py            # five himawari9-fd leads
  ... emit scripts/s2_coverage_equivalence.py --suite himawari9-fd   # every tiled product of a sector
  ... emit scripts/s2_coverage_equivalence.py --products himawari9-fd-ir,himawari9-fd-b08
  ... emit scripts/s2_coverage_equivalence.py --sat goes19 --sector conus --step 5 --backfill 60

(Building the image does not touch running lanes.) Window = --window-min, or
derived exactly like the lane (_cover_window_min: backfill + 2*step; default
--step 10 --backfill 60 -> 80 min, the hwfd-leads lane). --store local:/path
works for a Codespace dry run. Exit 0 on EQUIVALENT, 1 on DIFFERS, 2 on
usage/store errors.
"""
import argparse
import datetime as dt
import json
import os
import sys

try:
    import numpy  # noqa: F401  -- the emitter's import chain needs it
except ImportError:
    sys.exit("ERROR: numpy not importable -- run this INSIDE the tat-s2 image "
             "(see the module docstring): docker compose -f docker-compose.s2.yml "
             "run --rm --entrypoint python emit scripts/s2_coverage_equivalence.py")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["S2_COVER_FROM_MANIFEST"] = "1"          # the path under test

import s2_pyramid as P            # noqa: E402
import s2_pyramid_emit as E       # noqa: E402
import s2_registry as R           # noqa: E402

UTC = dt.timezone.utc
#: docker-compose.s2.hw-fd-leads.yml: `for prod in ir irbd sandwich airmass b08`
LEADS = ["himawari9-fd-ir", "himawari9-fd-irbd", "himawari9-fd-sandwich",
         "himawari9-fd-airmass", "himawari9-fd-b08"]
BOUNDARY_S = 120


class _Counting:
    """Op counters for a store that has none (local:); R2 already counts."""

    def __init__(self, inner):
        self._inner = inner
        self.ops = {}

    def _op(self, name):
        self.ops[name] = self.ops.get(name, 0) + 1

    def __getattr__(self, name):
        fn = getattr(self._inner, name)
        counted = {"get_bytes": "get", "head": "head", "list_keys": "list_pages",
                   "list_prefixes": "list_pages", "list_level": "list_pages",
                   "put_bytes": "put"}.get(name)
        if counted is None or not callable(fn):
            return fn

        def wrapped(*a, **kw):
            self._op(counted)
            return fn(*a, **kw)
        return wrapped


def _entries(args) -> list:
    if args.products:
        ids = [p.strip() for p in args.products.split(",") if p.strip()]
        out = []
        for pid in ids:
            e = R.REGISTRY_BY_ID.get(pid)
            if e is None or not e.tiled:
                sys.exit(f"ERROR: unknown or non-tiled product {pid!r}")
            out.append(e)
        return out
    if args.sat or args.sector:
        if not (args.sat and args.sector):
            sys.exit("ERROR: --sat and --sector go together")
        out = [e for e in R.REGISTRY if e.tiled and e.sat_key == args.sat
               and e.sector_key == args.sector]
        if not out:
            sys.exit(f"ERROR: no tiled products for {args.sat}/{args.sector}")
        return out
    suite = args.suite or "himawari9-fd-leads"
    if suite == "himawari9-fd-leads":
        return [R.REGISTRY_BY_ID[p] for p in LEADS]
    # the emitter's own suite resolution: bare sector = goes19, else <sat>-<sector>
    sat_keys = sorted({e.sat_key for e in R.REGISTRY if e.tiled})
    suite_sat, suite_sector = "goes19", suite
    for sk in sat_keys:
        if suite.startswith(sk + "-"):
            suite_sat, suite_sector = sk, suite[len(sk) + 1:]
            break
    out = [e for e in R.REGISTRY if e.tiled and e.sat_key == suite_sat
           and e.sector_key == suite_sector]
    if not out:
        sys.exit(f"ERROR: no tiled products for suite {suite!r}")
    return out


def _snap(store) -> dict:
    return dict(getattr(store, "ops", None) or {})


def _delta(before: dict, after: dict) -> dict:
    keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(keys)
            if after.get(k, 0) - before.get(k, 0)}


def _fmt_ops(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(d.items())) or "none"


def _fmt_stamps(ts) -> str:
    return ", ".join(t.strftime("%Y%m%dT%H%M%SZ") for t in sorted(ts)) or "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", default=None,
                    help="'himawari9-fd-leads' (default; the canary lane's five "
                         "products) or an emitter suite: 'conus', 'fd', "
                         "'himawari9-fd', 'gk2a-fd', ...")
    ap.add_argument("--products", default=None, help="comma-separated product ids")
    ap.add_argument("--sat", default=None)
    ap.add_argument("--sector", default=None)
    ap.add_argument("--prefix", default="shadow")
    ap.add_argument("--store", default="r2", help="'r2' (default) or 'local:/path'")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--backfill", type=int, default=60)
    ap.add_argument("--window-min", type=int, default=None,
                    help="override the lane-derived window (backfill + 2*step)")
    args = ap.parse_args(argv)

    entries = _entries(args)
    store, is_r2 = E._make_store(args.store)
    if not hasattr(store, "ops"):
        store = _Counting(store)
    lane_args = type("A", (), {"step": args.step, "backfill": args.backfill})()
    window = args.window_min or E._cover_window_min(lane_args)

    print(f"[equiv] {len(entries)} product(s)  prefix={args.prefix}  "
          f"window={window}m  store={'r2' if is_r2 else args.store}")
    verdict_ok = True
    tot_m, tot_l = {}, {}
    for e in entries:
        t0 = dt.datetime.now(UTC)
        b = _snap(store)
        man, reason = E._covered_from_manifest([e], store, args.prefix, window)
        ops_m = _delta(b, _snap(store))
        b = _snap(store)
        lst = E._covered_times([e], store, args.prefix, window_min=window)
        ops_l = _delta(b, _snap(store))
        for k, v in ops_m.items():
            tot_m[k] = tot_m.get(k, 0) + v
        for k, v in ops_l.items():
            tot_l[k] = tot_l.get(k, 0) + v

        inner = t0 - dt.timedelta(minutes=window) + dt.timedelta(seconds=BOUNDARY_S)
        L = {t for t in lst[0] if t >= inner}
        if man is None:
            verdict_ok = False
            print(f"[equiv] {e.product_id}: manifest UNUSABLE ({reason}) -- the lane "
                  f"would take the listing fallback  |listing|={len(lst[0])}  "
                  f"ops manifest[{_fmt_ops(ops_m)}] listing[{_fmt_ops(ops_l)}]")
            continue
        M = {t for t in man[0] if t >= inner}
        orphans = L - M
        ghosts = M - L
        strays = set()
        if orphans:
            # classify by geometry (only when something differs: one GET +
            # one windowed listing more): the manifest's maxzoom vs each
            # frame's own, exactly the test emit_one's stray guard makes
            b = _snap(store)
            after = (t0 - dt.timedelta(minutes=window)).strftime("%Y%m%dT%H%M%SZ")
            mz_of = dict(P.complete_stamps(e, store, args.prefix, limit=300,
                                           after=after))
            man_mz = json.loads(store.get_bytes(
                e.latest_times_key(args.prefix))).get("maxzoom")
            strays = {t for t in orphans
                      if mz_of.get(t.strftime("%Y%m%dT%H%M%SZ")) != man_mz}
            orphans = orphans - strays
            for k, v in _delta(b, _snap(store)).items():
                ops_l[k] = ops_l.get(k, 0) + v
                tot_l[k] = tot_l.get(k, 0) + v
        status = ("same" if not orphans and not ghosts and not strays
                  else "DIFFERS")
        if status == "DIFFERS":
            verdict_ok = False
        print(f"[equiv] {e.product_id}: |manifest|={len(man[0])} |listing|={len(lst[0])}  "
              f"listing-manifest={len(orphans) + len(strays)} "
              f"(orphans={len(orphans)} strays={len(strays)}) "
              f"manifest-listing={len(ghosts)}  "
              f"{status}  ops manifest[{_fmt_ops(ops_m)}] listing[{_fmt_ops(ops_l)}]")
        if orphans:
            print(f"[equiv]   orphans (complete in R2 at the manifest's geometry, "
                  f"not advertised; self-heal in one pass): {_fmt_stamps(orphans)}")
        if strays:
            print(f"[equiv]   strays (complete at a FOREIGN geometry; never "
                  f"advertised by design, the lane memos them after one fetch; "
                  f"clear by ageing out or deletion): {_fmt_stamps(strays)}")
        if ghosts:
            print(f"[equiv]   ADVERTISED BUT NOT COMPLETE IN R2 (must be explained): "
                  f"{_fmt_stamps(ghosts)}")
    print(f"[equiv] ops total: manifest path [{_fmt_ops(tot_m)}]  "
          f"listing path [{_fmt_ops(tot_l)}]  (compared on the common inner "
          f"window, boundary margin {BOUNDARY_S}s)")
    print("EQUIVALENT" if verdict_ok else "DIFFERS")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as e:
        if isinstance(e.code, str):          # argparse/_make_store error text
            print(e.code, file=sys.stderr)
            raise SystemExit(2)
        raise
