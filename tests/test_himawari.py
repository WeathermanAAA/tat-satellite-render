"""End-to-end Himawari smoke test against Mawar 2023.

Anonymous AWS public-bucket access (no creds), small 5°×5° bbox centered
on the Mawar storm position 2023-05-24 12:00 UTC, ``clean_ir`` channel.

This is the cheapest backend check that the full chain works:
  - segment listing on noaa-himawari9
  - bz2 decompress + HSD parse
  - segment-filter (single-segment download for a tight bbox)
  - calibration, projection, render

Run from the repo root:

    .venv/bin/python tests/test_himawari.py

Skips with exit 0 if no internet (offline CI).
"""

import asyncio
import datetime as dt
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_internet() -> bool:
    try:
        socket.create_connection(("s3.amazonaws.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


def test_mawar_2023_smoke():
    if not _has_internet():
        print("skip mawar_2023_smoke — no internet")
        return

    # Imports gated behind the internet check so the test file remains
    # importable in offline environments without service deps.
    import s3fs  # noqa: F401  — fail fast if not installed
    from satellites import HIMAWARI_PACIFIC, FetchResult

    bbox = [140.0, 12.0, 145.0, 17.0]  # 5° square centered on Mawar position
    when = dt.datetime(2023, 5, 24, 12, 0, tzinfo=dt.timezone.utc)

    sat = HIMAWARI_PACIFIC
    assert sat.can_see(bbox, when)

    async def go():
        resolved = await sat.find_file(when, "clean_ir", bbox, nearest_to_target=True)
        assert resolved.bucket == "noaa-himawari9", resolved
        assert resolved.sat_name == "Himawari-9", resolved
        assert resolved.product == "FLDK", resolved
        # scan_start should snap to the 12:00 slot exactly (target is 12:00 sharp)
        assert resolved.scan_start == when, resolved.scan_start

        t0 = time.time()
        result: FetchResult = await sat.fetch(resolved, bbox, "clean_ir")
        elapsed = time.time() - t0
        return result, elapsed

    result, elapsed = asyncio.get_event_loop().run_until_complete(go())

    # Sanity on the data we got back
    assert result.units == "K", result.units
    assert result.channel == 13
    assert result.generic_channel == "clean_ir"
    assert result.sat_name == "Himawari-9"
    assert result.cmi.ndim == 2 and result.cmi.size > 0
    # The bbox includes the eyewall; a Cat-5 storm has cloud tops well below 220K
    import numpy as np

    valid = result.cmi[np.isfinite(result.cmi)]
    assert valid.size > 1000, f"too few valid pixels: {valid.size}"
    coldest = float(np.nanmin(valid))
    assert coldest < 220.0, (
        f"Mawar 2023-05-24 12z eyewall should produce cloud tops < 220 K, got coldest={coldest:.1f} K"
    )

    print(
        f"ok mawar_2023_smoke ({elapsed:.1f}s, shape={result.cmi.shape}, "
        f"coldest BT={coldest:.1f}K, valid_frac={valid.size / result.cmi.size:.2f})"
    )

    assert elapsed < 60.0, f"mawar smoke fetch took {elapsed:.1f}s — perf regression"


def main():
    test_mawar_2023_smoke()
    print("\nall himawari tests passed")


if __name__ == "__main__":
    main()
