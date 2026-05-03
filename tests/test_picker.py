"""Standalone smoke test for the goes.py product/bucket pickers.

Run from the repo root:

    .venv/bin/python tests/test_picker.py

Exits non-zero on any failure. No pytest dependency — assertions only.
Add new cases here as the picker grows; this is the cheapest CI-style
guard against silent regressions in the routing logic.
"""

import asyncio
import datetime as dt
import os
import sys

# Make the repo root importable from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goes import (  # noqa: E402
    pick_buckets_for_time,
    _conus_footprint,
    _bbox_inside,
    CMIPC_MODE6_START,
    CONUS_FOOTPRINT_MODE3,
    CONUS_FOOTPRINT_MODE6,
)
from app import compute_downsample_factor, DEG_TO_KM, _native_km_per_pixel  # noqa: E402


def test_bucket_picker_boundaries():
    assert pick_buckets_for_time("latest") == ["noaa-goes19"]
    # 2017-09-05 (Irma) -> goes16 only (before 2018-08-01)
    assert pick_buckets_for_time("2017-09-05T17:47:00Z") == ["noaa-goes16"]
    # 2024-09-15 -> goes19 then goes16 (overlap window)
    assert pick_buckets_for_time("2024-09-15T12:00:00Z") == ["noaa-goes19", "noaa-goes16"]
    # 2025-04-04 (handover boundary) -> goes19 only
    assert pick_buckets_for_time("2025-04-04T00:00:00Z") == ["noaa-goes19"]
    # Way in the future -> goes19 only
    assert pick_buckets_for_time("2030-01-01T00:00:00Z") == ["noaa-goes19"]
    print("ok bucket_picker_boundaries")


def test_conus_footprint_dates():
    # Irma 2017 -> Mode 3 footprint, eastern edge -65W
    irma = dt.datetime(2017, 9, 5, 17, 47, tzinfo=dt.timezone.utc)
    fp = _conus_footprint(irma)
    assert fp == CONUS_FOOTPRINT_MODE3, fp
    assert fp[2] == -65.0  # east edge

    # 2020 -> Mode 6 footprint, eastern edge -55W
    later = dt.datetime(2020, 8, 1, 12, 0, tzinfo=dt.timezone.utc)
    fp = _conus_footprint(later)
    assert fp == CONUS_FOOTPRINT_MODE6, fp
    assert fp[2] == -55.0

    # Boundary itself is Mode 6 (>=)
    fp = _conus_footprint(CMIPC_MODE6_START)
    assert fp == CONUS_FOOTPRINT_MODE6
    print("ok conus_footprint_dates")


def test_irma_caribbean_bbox_rejected_for_cmipc():
    """The exact regression: Irma 2017-09-05 with a Caribbean bbox extending
    east of -65W must NOT pass the CONUS footprint check (forces CMIPF).
    """
    irma = dt.datetime(2017, 9, 5, 17, 47, tzinfo=dt.timezone.utc)
    fp = _conus_footprint(irma)

    # Bbox exactly as the Sept-2017 user complaint: extends to -60W
    irma_caribbean = [-78, 14, -60, 25]
    assert not _bbox_inside(irma_caribbean, fp), (
        "regression: Irma Caribbean bbox should be rejected by Mode 3 footprint"
    )

    # Tight US-mainland bbox should still pass
    us_mainland = [-100, 30, -80, 45]
    assert _bbox_inside(us_mainland, fp), (
        "us mainland bbox should be inside Mode 3 footprint"
    )
    print("ok irma_caribbean_bbox_rejected_for_cmipc")


def test_post_mode6_bbox_to_minus_60_accepted():
    """Same-shape Caribbean bbox extending to -60W IS inside Mode 6 footprint
    (eastern edge -55W), so post-2019-04-02 it would route to CMIPC. Encodes
    the date-awareness contract explicitly.
    """
    post = dt.datetime(2024, 9, 15, 12, 0, tzinfo=dt.timezone.utc)
    fp = _conus_footprint(post)
    bbox = [-78, 14, -60, 25]
    assert _bbox_inside(bbox, fp), "Mode 6 footprint should accept lon=-60"
    print("ok post_mode6_bbox_to_minus_60_accepted")


def test_pixel_budget_large_ir_bbox_accepted():
    """80°×80° IR bbox (channel 13, 2 km native) must be accepted and
    auto-downsample to a per-axis output ≤4000 px (i.e. ≤16M total)."""
    bbox = [-100.0, -10.0, -20.0, 70.0]  # 80×80
    factor = compute_downsample_factor(bbox, channel=13)
    raw_per_axis = (80.0 * DEG_TO_KM) / _native_km_per_pixel(13)  # 80*111/2 = 4440
    out_per_axis = raw_per_axis / factor
    assert factor >= 1, f"factor must be a positive int, got {factor}"
    assert out_per_axis <= 4000, (
        f"80° IR: output {out_per_axis:.0f} px/axis exceeds 4000 (factor={factor})"
    )
    # Whole-pixel-budget check: total output ≤ 16M pixels.
    assert (out_per_axis ** 2) <= 16_000_000, (
        f"80° IR: output {out_per_axis ** 2:.0f} px exceeds 16M budget"
    )
    print(f"ok pixel_budget_large_ir_bbox_accepted (factor={factor}, out={out_per_axis:.0f} px/axis)")


def test_pixel_budget_large_visible_bbox_accepted():
    """60°×60° visible bbox (channel 2, 0.5 km native) must be accepted and
    auto-downsample to a per-axis output ≤4000 px."""
    bbox = [-100.0, -10.0, -40.0, 50.0]  # 60×60
    factor = compute_downsample_factor(bbox, channel=2)
    raw_per_axis = (60.0 * DEG_TO_KM) / _native_km_per_pixel(2)  # 60*111/0.5 = 13320
    out_per_axis = raw_per_axis / factor
    assert factor > 1, (
        f"visible 60° must trigger downsampling, got factor={factor}"
    )
    assert out_per_axis <= 4000, (
        f"60° vis: output {out_per_axis:.0f} px/axis exceeds 4000 (factor={factor})"
    )
    assert (out_per_axis ** 2) <= 16_000_000, (
        f"60° vis: output {out_per_axis ** 2:.0f} px exceeds 16M budget"
    )
    print(f"ok pixel_budget_large_visible_bbox_accepted (factor={factor}, out={out_per_axis:.0f} px/axis)")


def main():
    test_bucket_picker_boundaries()
    test_conus_footprint_dates()
    test_irma_caribbean_bbox_rejected_for_cmipc()
    test_post_mode6_bbox_to_minus_60_accepted()
    test_pixel_budget_large_ir_bbox_accepted()
    test_pixel_budget_large_visible_bbox_accepted()
    print("\nall picker tests passed")


if __name__ == "__main__":
    main()
