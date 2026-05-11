"""End-to-end GOES-West smoke test against Hurricane Hilary 2023.

Anonymous AWS public-bucket access (no creds), 20°×13° bbox over the EPac
covering the Hilary eye on 2023-08-19 12:00 UTC, ``clean_ir`` channel.

This is the cheapest backend check that the GOES-West chain works end to
end:
  - picker routes EPac bbox → GOES_WEST
  - resolve(2023-08-19) → GOES-18 (post G18_OPERATIONAL)
  - find_file → CMIPC ("PACUS") OR CMIPF — bbox extends east of -77° so
    the PACUS footprint check should reject it and fall back to CMIPF
  - download + project_to_latlon + sanity-check brightness temps

Run from the repo root:

    .venv/bin/python tests/test_goes_west.py

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


def test_hilary_2023_smoke():
    if not _has_internet():
        print("skip hilary_2023_smoke — no internet")
        return

    import s3fs  # noqa: F401  — fail fast if not installed
    from satellites import GOES_WEST, FetchResult, pick_satellite

    bbox = [-120.0, 12.0, -100.0, 25.0]  # covers Hilary eye ~17°N -110°W
    when = dt.datetime(2023, 8, 19, 12, 0, tzinfo=dt.timezone.utc)

    sat = pick_satellite(bbox, when)
    assert sat is GOES_WEST, f"Hilary EPac bbox should pick GOES_WEST, got {sat.family}"

    async def go():
        resolved = await sat.find_file(when, "clean_ir", bbox, nearest_to_target=True)
        # 2023-08-19 is post-G18_OPERATIONAL (2023-01-04) so we expect noaa-goes18.
        assert resolved.bucket == "noaa-goes18", resolved
        assert resolved.sat_name == "GOES-18", resolved
        # bbox extends to -100°W — well inside PACUS — so CMIPC should be picked.
        # (If PACUS_FOOTPRINT widens to exclude this in future, CMIPF is also valid.)
        assert resolved.product in ("CMIPC", "CMIPF"), resolved.product

        t0 = time.time()
        result: FetchResult = await sat.fetch(resolved, bbox, "clean_ir")
        elapsed = time.time() - t0
        return result, elapsed

    result, elapsed = asyncio.get_event_loop().run_until_complete(go())

    assert result.units == "K", result.units
    assert result.channel == 13
    assert result.generic_channel == "clean_ir"
    assert result.sat_name == "GOES-18"
    assert result.sub_sat_lon == -137.2, result.sub_sat_lon
    assert result.cmi.ndim == 2 and result.cmi.size > 0

    import numpy as np

    valid = result.cmi[np.isfinite(result.cmi)]
    assert valid.size > 1000, f"too few valid pixels: {valid.size}"
    coldest = float(np.nanmin(valid))
    # Hilary 2023-08-19 12z was a Cat-4 hurricane — eyewall cloud tops well under 220 K.
    assert coldest < 220.0, (
        f"Hilary 2023-08-19 12z eyewall should produce cloud tops < 220 K, "
        f"got coldest={coldest:.1f} K"
    )

    print(
        f"ok hilary_2023_smoke ({elapsed:.1f}s, shape={result.cmi.shape}, "
        f"product={result.product}, coldest BT={coldest:.1f}K, "
        f"valid_frac={valid.size / result.cmi.size:.2f})"
    )

    assert elapsed < 90.0, f"hilary smoke fetch took {elapsed:.1f}s — perf regression"


def main():
    test_hilary_2023_smoke()
    print("\nall goes-west tests passed")


if __name__ == "__main__":
    main()
