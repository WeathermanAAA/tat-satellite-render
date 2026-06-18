#!/usr/bin/env python3
"""By-construction pixel proof for S1 (run inside the PINNED render image).

Replicates the prod meso poller's EXACT clean-IR call for goes19-m2 -- discover
the sector extent the same way (meso_poller.discover_goes_extent), then POST the
SAME /render body the meso lane sends (time=latest, channel=clean_ir,
enhancement=rainbow_ir, format=webp, product=meso, satellite=GOES-East) -- and
diffs the result against the LIVE prod frame for the same scan on the CDN
(cdn.triple-a-tropics.com/meso/goes19-m2/ir/{stamp}.webp).

If the operator box was stable across prod's <=120 s discovery->render window
(the common case), prod used the same bbox and the frames are BYTE-IDENTICAL --
the strict-identity gate (§7.2/§9). This is the empirical complement to the
structural by-construction argument: same frozen code + same inputs -> same
decoded pixels.

Run (with the s1-render container's image, which bundles meso_poller + the
frozen renderer):
    docker run --rm --network host tat-s1:latest \
        python s1_byconstruction_proof.py --render-url http://localhost:8080
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request

import requests

import meso_poller as MP
import s1_pixeldiff as PD
import s1_slots as S

CDN = "https://cdn.triple-a-tropics.com"
GOES19_M2 = MP.MESO_SECTORS_BY_SLUG["goes19-m2"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-url", default="http://localhost:8080")
    ap.add_argument("--rounds", type=int, default=3,
                    help="distinct latest scans to test (waits for new scans)")
    a = ap.parse_args(argv)

    session = requests.Session()
    seen = set()
    results = []
    for _ in range(a.rounds * 4):
        if len(results) >= a.rounds:
            break
        # 1. Discover the sector extent EXACTLY as the prod meso poller does.
        try:
            ext = MP.discover_goes_extent(GOES19_M2)
        except Exception as e:  # noqa: BLE001
            print(f"discover failed: {e}"); continue
        stamp = ext.scan_start.strftime(S.STAMP_FMT)
        if stamp in seen:
            import time; time.sleep(20); continue
        seen.add(stamp)

        # 2. POST the SAME /render body the meso ir lane sends (time=latest).
        body = {"bbox": ext.bbox, "time": "latest", "channel": "clean_ir",
                "enhancement": "rainbow_ir", "format": "webp",
                "product": "meso", "satellite": "GOES-East"}
        r = session.post(a.render_url + "/render", json=body, timeout=120)
        if r.status_code != 200:
            print(f"  {stamp}: /render {r.status_code} {r.text[:120]}"); continue
        shadow_bytes = r.content
        xscan = next((v for k, v in r.headers.items()
                      if k.lower() == "x-scan-time"), None)
        rstamp = (xscan and __import__("datetime").datetime.fromisoformat(
            xscan.replace("Z", "+00:00")).strftime(S.STAMP_FMT)) or stamp

        # 3. Fetch the LIVE prod frame for the same scan from the CDN.
        prod_url = f"{CDN}/{S.prod_frame_key(rstamp)}"
        try:
            prod_bytes = urllib.request.urlopen(urllib.request.Request(
                prod_url, headers={"User-Agent": "tat-s1-proof/1.0"}), timeout=30).read()
        except Exception as e:  # noqa: BLE001
            print(f"  {rstamp}: prod frame not on CDN yet ({e}); will retry a "
                  f"later scan"); seen.discard(stamp);
            import time; time.sleep(20); continue

        # 4. Diff.
        d = PD.diff_frames(shadow_bytes, prod_bytes)
        results.append((rstamp, len(shadow_bytes), len(prod_bytes), d))
        tag = ("BYTE-IDENTICAL" if d["byte_equal"] else
               "pixel-identical" if d["pixel_equal"] else
               f"DIFFER max={d.get('max_abs_diff')} frac={d.get('diff_frac')}")
        print(f"  {rstamp}: shadow={len(shadow_bytes)}B prod={len(prod_bytes)}B "
              f"bbox={ext.bbox} -> {tag}")
        import time; time.sleep(20)

    if not results:
        print(">> no rounds completed (prod frame lag / discover failure)")
        return 1
    byte_id = sum(1 for _, _, _, d in results if d["byte_equal"])
    px_id = sum(1 for _, _, _, d in results if d["pixel_equal"])
    print(f"\n>> {byte_id}/{len(results)} byte-identical, "
          f"{px_id}/{len(results)} pixel-identical to the live prod frame")
    print(">> by-construction "
          + ("CONFIRMED" if px_id == len(results)
             else "MOSTLY (inspect differing -- likely operator-box-moved bbox)"))
    return 0 if px_id == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
