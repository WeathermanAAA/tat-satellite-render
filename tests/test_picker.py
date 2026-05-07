"""Standalone smoke test for the satellites.py picker logic.

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

from satellites import (  # noqa: E402
    CMIPC_MODE6_START,
    CONUS_FOOTPRINT_MODE3,
    CONUS_FOOTPRINT_MODE6,
    CoverageError,
    GENERIC_CHANNELS,
    GOES_EAST,
    HIMAWARI_PACIFIC,
    _bbox_inside,
    _conus_footprint,
    antimeridian_safe_center_lon,
    goes_band_to_generic,
    pick_buckets_for_time,
    pick_satellite,
)
from app import (  # noqa: E402
    DEG_TO_KM,
    _native_km_per_pixel,
    compute_downsample_factor,
    normalize_channel,
)


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


def test_pixel_budget_uses_generic_channel_native_km():
    """Pixel-budget calc must read native_km from the generic-channel table
    so future sats (different native resolutions per channel) work without
    hard-coded GOES band tables in app.py.
    """
    bbox = [-100.0, -10.0, -20.0, 70.0]  # 80×80
    by_generic = compute_downsample_factor(bbox, channel="clean_ir")
    by_numeric = compute_downsample_factor(bbox, channel=13)
    assert by_generic == by_numeric, (
        f"clean_ir vs band 13 should produce identical downsample factor: "
        f"{by_generic} vs {by_numeric}"
    )

    by_vis_generic = compute_downsample_factor(bbox, channel="visible_red")
    by_vis_numeric = compute_downsample_factor(bbox, channel=2)
    assert by_vis_generic == by_vis_numeric, (
        f"visible_red vs band 2 should produce identical downsample factor"
    )
    # Visible at 0.5 km demands ~4× the downsample of clean_ir at 2.0 km.
    assert by_vis_generic > by_generic, (
        f"visible (0.5 km) should downsample more than clean_ir (2.0 km): "
        f"{by_vis_generic} vs {by_generic}"
    )
    print("ok pixel_budget_uses_generic_channel_native_km")


def test_pick_satellite_caribbean_2026_resolves_goes19():
    """Caribbean bbox at a 2026 timestamp picks GOES_EAST and resolves to GOES-19."""
    caribbean = [-78.0, 14.0, -60.0, 25.0]
    when = dt.datetime(2026, 5, 7, 12, 0, tzinfo=dt.timezone.utc)
    sat = pick_satellite(caribbean, when)
    assert sat is GOES_EAST, f"Caribbean 2026 should pick GOES_EAST, got {sat.family}"
    resolved = sat.resolve(when)
    assert resolved.name == "GOES-19", resolved
    assert resolved.bucket == "noaa-goes19", resolved
    assert resolved.sub_sat_lon == -75.2, resolved
    print("ok pick_satellite_caribbean_2026_resolves_goes19")


def test_pick_satellite_caribbean_2017_resolves_goes16():
    """Caribbean bbox at an Irma-era timestamp picks GOES_EAST and resolves to GOES-16."""
    caribbean = [-78.0, 14.0, -60.0, 25.0]
    when = dt.datetime(2017, 9, 5, 17, 47, tzinfo=dt.timezone.utc)
    sat = pick_satellite(caribbean, when)
    assert sat is GOES_EAST, f"Caribbean 2017 should pick GOES_EAST, got {sat.family}"
    resolved = sat.resolve(when)
    assert resolved.name == "GOES-16", resolved
    assert resolved.bucket == "noaa-goes16", resolved
    print("ok pick_satellite_caribbean_2017_resolves_goes16")


def test_pick_satellite_japan_2026_resolves_himawari9():
    """Japan bbox at 2026 picks HIMAWARI_PACIFIC and resolves to Himawari-9."""
    japan = [130.0, 20.0, 145.0, 40.0]
    when = dt.datetime(2026, 5, 7, 12, 0, tzinfo=dt.timezone.utc)
    sat = pick_satellite(japan, when)
    assert sat is HIMAWARI_PACIFIC, f"Japan 2026 should pick HIMAWARI_PACIFIC, got {sat.family}"
    resolved = sat.resolve(when)
    assert resolved.name == "Himawari-9", resolved
    assert resolved.bucket == "noaa-himawari9", resolved
    assert resolved.sub_sat_lon == 140.7, resolved
    print("ok pick_satellite_japan_2026_resolves_himawari9")


def test_pick_satellite_japan_2018_resolves_himawari8():
    """Japan bbox at 2018 picks HIMAWARI_PACIFIC and resolves to Himawari-8 (pre-handover)."""
    japan = [130.0, 20.0, 145.0, 40.0]
    when = dt.datetime(2018, 10, 24, 18, 0, tzinfo=dt.timezone.utc)
    sat = pick_satellite(japan, when)
    assert sat is HIMAWARI_PACIFIC, f"Japan 2018 should pick HIMAWARI_PACIFIC, got {sat.family}"
    resolved = sat.resolve(when)
    assert resolved.name == "Himawari-8", resolved
    assert resolved.bucket == "noaa-himawari8", resolved
    print("ok pick_satellite_japan_2018_resolves_himawari8")


def test_pick_satellite_hagibis_2019_resolves_himawari8():
    """Hagibis bbox (138-143°E lat 30-35°N) at 2019-10-12 03Z resolves to H-8."""
    hagibis = [138.0, 30.0, 143.0, 35.0]
    when = dt.datetime(2019, 10, 12, 3, 0, tzinfo=dt.timezone.utc)
    sat = pick_satellite(hagibis, when)
    assert sat is HIMAWARI_PACIFIC, f"Hagibis 2019 should pick HIMAWARI_PACIFIC, got {sat.family}"
    resolved = sat.resolve(when)
    assert resolved.name == "Himawari-8", resolved
    assert resolved.bucket == "noaa-himawari8", resolved
    print("ok pick_satellite_hagibis_2019_resolves_himawari8")


def test_pick_satellite_h9_handover_boundary():
    """H-9 promoted 2022-12-13 00:00 UTC; ≥ that instant resolves to H-9."""
    japan = [130.0, 20.0, 145.0, 40.0]
    just_before = dt.datetime(2022, 12, 12, 23, 59, tzinfo=dt.timezone.utc)
    just_after = dt.datetime(2022, 12, 13, 0, 0, tzinfo=dt.timezone.utc)
    assert pick_satellite(japan, just_before).resolve(just_before).name == "Himawari-8"
    assert pick_satellite(japan, just_after).resolve(just_after).name == "Himawari-9"
    print("ok pick_satellite_h9_handover_boundary")


def test_pick_satellite_african_bbox_raises_meteosat_coverage_error():
    """An African / European bbox (lon 0-30°E) is outside both GOES-East and Himawari coverage,
    and must raise CoverageError that names METEOSAT coming soon.
    """
    africa = [0.0, -10.0, 30.0, 20.0]
    when = dt.datetime(2026, 5, 7, 12, 0, tzinfo=dt.timezone.utc)
    try:
        pick_satellite(africa, when)
    except CoverageError as e:
        msg = str(e)
        assert "METEOSAT" in msg, f"CoverageError should mention METEOSAT: {msg!r}"
        assert "coming soon" in msg, f"CoverageError should mention coming soon: {msg!r}"
        print(f"ok pick_satellite_african_bbox_raises_meteosat_coverage_error")
        return
    raise AssertionError("expected CoverageError for African bbox, got no exception")


def test_himawari_can_see_caribbean_false():
    """Himawari MUST NOT claim coverage of a Caribbean bbox (140° away from sub-sat)."""
    caribbean = [-78.0, 14.0, -60.0, 25.0]
    when = dt.datetime(2026, 5, 7, 12, 0, tzinfo=dt.timezone.utc)
    assert HIMAWARI_PACIFIC.can_see(caribbean, when) is False, (
        "Himawari should not claim coverage of Caribbean — picker would route incorrectly"
    )
    print("ok himawari_can_see_caribbean_false")


def test_himawari_can_see_antimeridian_crossing():
    """Himawari should see a bbox that straddles the antimeridian, since its
    visible disk extends well past +180°."""
    crossing = [170.0, -20.0, -170.0, 0.0]   # 170°E to 190°E (= -170°E)
    when = dt.datetime(2026, 5, 7, 12, 0, tzinfo=dt.timezone.utc)
    assert HIMAWARI_PACIFIC.can_see(crossing, when), (
        "Himawari should see an antimeridian-crossing bbox near +180°"
    )
    sat = pick_satellite(crossing, when)
    assert sat is HIMAWARI_PACIFIC
    print("ok himawari_can_see_antimeridian_crossing")


def test_himawari_generic_to_band():
    """Himawari clean_ir → 13 (same as GOES); visible_red → 3 (DIFFERENT from GOES band 2)."""
    assert HIMAWARI_PACIFIC.generic_to_band["clean_ir"] == 13
    assert HIMAWARI_PACIFIC.generic_to_band["wv_upper"] == 8
    assert HIMAWARI_PACIFIC.generic_to_band["wv_lower"] == 10
    assert HIMAWARI_PACIFIC.generic_to_band["shortwave_ir"] == 7
    assert HIMAWARI_PACIFIC.generic_to_band["ir_window"] == 14
    assert HIMAWARI_PACIFIC.generic_to_band["visible_red"] == 3, (
        "AHI band 3 is the 0.64 µm visible — distinct from GOES ABI band 2"
    )
    # GENERIC_CHANNELS table must list both mappings for visible_red
    assert GENERIC_CHANNELS["visible_red"]["goes"] == 2
    assert GENERIC_CHANNELS["visible_red"]["ahi"] == 3
    print("ok himawari_generic_to_band")


def test_antimeridian_safe_center_lon():
    """Bboxes that cross ±180° must have center near the antimeridian, not 0."""
    # Crosses ±180°: from 170°E to 190°E (= -170°E). Center should be ±180°.
    crossing = [170.0, -30.0, -170.0, -20.0]
    c = antimeridian_safe_center_lon(crossing)
    assert abs(abs(c) - 180.0) < 0.01, f"crossing bbox center should be ±180°, got {c}"

    # Normal bbox: from -78° to -60°. Center should be -69°.
    normal = [-78.0, 14.0, -60.0, 25.0]
    assert abs(antimeridian_safe_center_lon(normal) - (-69.0)) < 0.01

    # Symmetric around 0: from -10° to 10°. Center should be 0°.
    sym = [-10.0, 0.0, 10.0, 5.0]
    assert abs(antimeridian_safe_center_lon(sym) - 0.0) < 0.01

    print("ok antimeridian_safe_center_lon")


def test_generic_channel_to_band():
    """generic_to_band on GOES_EAST + the GENERIC_CHANNELS table agree on band numbers."""
    assert GOES_EAST.generic_to_band["clean_ir"] == 13
    assert GOES_EAST.generic_to_band["wv_upper"] == 8
    assert GOES_EAST.generic_to_band["visible_red"] == 2
    assert GOES_EAST.generic_to_band["shortwave_ir"] == 7
    assert GOES_EAST.generic_to_band["wv_lower"] == 10
    assert GOES_EAST.generic_to_band["ir_window"] == 14

    # Inverse: numeric -> generic
    assert goes_band_to_generic(13) == "clean_ir"
    assert goes_band_to_generic(8) == "wv_upper"
    assert goes_band_to_generic(2) == "visible_red"
    assert goes_band_to_generic(99) is None  # unknown band

    # GENERIC_CHANNELS table mirrors generic_to_band
    for name, band in GOES_EAST.generic_to_band.items():
        assert GENERIC_CHANNELS[name]["goes"] == band

    print("ok generic_channel_to_band")


def test_normalize_channel_back_compat():
    """Numeric channel input must round-trip to a generic name AND flag was_numeric=True
    (so /render emits the X-Deprecated-Channel-API header)."""
    # int 13 -> ("clean_ir", True)
    name, was_numeric = normalize_channel(13)
    assert name == "clean_ir" and was_numeric, (name, was_numeric)

    # str "13" -> still legacy numeric, must flag was_numeric=True
    name, was_numeric = normalize_channel("13")
    assert name == "clean_ir" and was_numeric, (name, was_numeric)

    # Generic name pass-through, was_numeric=False
    name, was_numeric = normalize_channel("clean_ir")
    assert name == "clean_ir" and not was_numeric, (name, was_numeric)

    name, was_numeric = normalize_channel("wv_upper")
    assert name == "wv_upper" and not was_numeric

    # Unknown numeric raises ValueError
    try:
        normalize_channel(99)
    except ValueError:
        pass
    else:
        raise AssertionError("normalize_channel(99) should raise ValueError")

    # Unknown string raises ValueError
    try:
        normalize_channel("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("normalize_channel('nope') should raise ValueError")

    print("ok normalize_channel_back_compat")


def main():
    test_bucket_picker_boundaries()
    test_conus_footprint_dates()
    test_irma_caribbean_bbox_rejected_for_cmipc()
    test_post_mode6_bbox_to_minus_60_accepted()
    test_pixel_budget_large_ir_bbox_accepted()
    test_pixel_budget_large_visible_bbox_accepted()
    test_pixel_budget_uses_generic_channel_native_km()
    test_pick_satellite_caribbean_2026_resolves_goes19()
    test_pick_satellite_caribbean_2017_resolves_goes16()
    test_pick_satellite_japan_2026_resolves_himawari9()
    test_pick_satellite_japan_2018_resolves_himawari8()
    test_pick_satellite_hagibis_2019_resolves_himawari8()
    test_pick_satellite_h9_handover_boundary()
    test_pick_satellite_african_bbox_raises_meteosat_coverage_error()
    test_himawari_can_see_caribbean_false()
    test_himawari_can_see_antimeridian_crossing()
    test_himawari_generic_to_band()
    test_antimeridian_safe_center_lon()
    test_generic_channel_to_band()
    test_normalize_channel_back_compat()
    print("\nall picker tests passed")


if __name__ == "__main__":
    main()
