"""M2 policy registration and reversible default activation."""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

import app
import render_policy as rp
import satellites
from validation.truecolor import m2


RENDER_WITH_HELPERS = """
def _draw_rgb(ax, rgb, *, policy):
    _imshow_with_render_policy(ax, rgb, policy=policy)

def _draw_scalar(ax, scalar, *, policy):
    _imshow_with_render_policy(ax, scalar, cmap="gray", policy=policy)

def render_png(ax, rgb, scalar, *, policy):
    _draw_rgb(ax, rgb, policy=policy)
    _draw_scalar(ax, scalar, policy=policy)
"""


def test_m2_policy_is_registered_with_only_its_authorized_switches() -> None:
    policy = rp.resolve_render_policy(rp.M2_POLICY.algo_version)
    assert policy is rp.M2_POLICY
    assert policy.stage == "M2"
    assert policy.algo_version == "tat-truecolor-m2-reflectance-resample-v1"
    assert policy.resample_kernel == "hanning"
    assert policy.resample_in_reflectance is True
    assert policy.native_resolution_cap is False
    assert policy.ratio_sharpen is True
    assert policy.single_output_raster is False
    assert policy.freeze_sequence_geometry is False
    assert policy.blend_taper_deg == 95.0
    assert policy.gate_blend_by_day_survival is False
    assert policy.operational_integrity is False
    assert policy.glint_in_reflectance is False
    assert policy.quantizer == "production_floor"


def test_m2_is_the_registered_default_after_gate_closure() -> None:
    assert rp.resolve_render_policy() is rp.M2_POLICY
    assert rp.DEFAULT_POLICY_VERSION == rp.M2_POLICY.algo_version
    assert rp.ALGO_VERSION == rp.M2_POLICY.algo_version
    assert rp.M2_POLICY.algo_version in rp.POLICIES


def test_m2_binds_live_policy_forwarding_and_registered_rollbacks() -> None:
    binding = m2.m2_policy_rollback_binding()
    assert binding["passed"], binding["issues"]
    assert binding["live_dispatcher_call_count"] == 2
    assert binding["m1_literal_hanning_sites"] is True
    assert binding["m0_rollback_registered"] is True


@pytest.mark.parametrize(
    "mutant",
    (
        RENDER_WITH_HELPERS.replace(
            "_draw_rgb(ax, rgb, policy=policy)",
            "_draw_rgb(ax, rgb, policy=rp.M0_POLICY)",
        ),
        RENDER_WITH_HELPERS.replace(
            '_imshow_with_render_policy(ax, scalar, cmap="gray", policy=policy)',
            '_imshow_with_render_policy(ax, scalar, cmap="gray")',
        ),
        RENDER_WITH_HELPERS.replace(
            "_imshow_with_render_policy(ax, rgb, policy=policy)",
            "ax.imshow(rgb, interpolation=policy.resample_kernel)",
        ),
        RENDER_WITH_HELPERS.replace(
            "    _draw_scalar(ax, scalar, policy=policy)\n", ""
        )
        + """
def _dead_scalar_lookalike(ax, scalar, *, policy):
    _imshow_with_render_policy(ax, scalar, cmap="gray", policy=policy)
""",
    ),
)
def test_m2_policy_binding_kills_helper_bypass_and_dead_lookalikes(mutant) -> None:
    call_count, issues = m2.m2_render_policy_forwarding_issues(mutant)
    assert issues
    assert call_count != 2 or any("policy=policy" in issue for issue in issues)


def test_m2_changes_no_later_milestone_behavior() -> None:
    m1 = rp.M1_POLICY.canonical_dict()
    m2 = rp.M2_POLICY.canonical_dict()
    for changed in ("stage", "algo_version", "resample_in_reflectance"):
        m1.pop(changed)
        m2.pop(changed)
    assert m2 == m1


def test_m2_identity_and_fingerprint_cannot_be_reconstructed() -> None:
    lookalike = dataclasses.replace(rp.M2_POLICY)
    assert lookalike == rp.M2_POLICY
    assert lookalike is not rp.M2_POLICY
    assert lookalike.policy_sha256 == rp.M2_POLICY.policy_sha256
    with pytest.raises(RuntimeError, match="unregistered render policy object"):
        rp.require_registered_policy(lookalike)


def test_m2_has_a_distinct_cache_visible_algorithm_identity() -> None:
    versions = {
        rp.M0_POLICY.algo_version,
        rp.M1_POLICY.algo_version,
        rp.M2_POLICY.algo_version,
    }
    fingerprints = {
        rp.M0_POLICY.policy_sha256,
        rp.M1_POLICY.policy_sha256,
        rp.M2_POLICY.policy_sha256,
    }
    assert len(versions) == 3
    assert len(fingerprints) == 3
    assert tuple(rp.POLICIES.values()) == (
        rp.M0_POLICY,
        rp.M1_POLICY,
        rp.M2_POLICY,
    )


def test_m2_cache_identity_binds_resolved_product_and_source_object() -> None:
    body = app.RenderRequest(
        bbox=[-80.0, 14.0, -68.0, 26.0], channel="true_color"
    )
    arguments = (
        body,
        "true_color",
        "2025-10-27T15:00:00+00:00",
        "noaa-goes19",
    )
    with pytest.raises(RuntimeError, match="resolved product and source object"):
        app._request_key(*arguments, policy=rp.M2_POLICY)
    with pytest.raises(RuntimeError, match="exact physical source manifest"):
        app._request_key(
            *arguments,
            policy=rp.M2_POLICY,
            resolved_product="CMIPM1",
            resolved_s3_key="red-a.nc",
        )

    scan_start = dt.datetime(2025, 10, 27, 15, tzinfo=dt.timezone.utc)

    def manifest(*, product="CMIPM1", red="red-a.nc", blue="blue-a.nc"):
        return satellites.TrueColorPhysicalSourceManifest(
            sensor="abi",
            bucket="noaa-goes19",
            product=product,
            scan_start=scan_start,
            sat_name="GOES-19",
            resolved_anchor_s3_key=red,
            bbox=(-80.0, 14.0, -68.0, 26.0),
            sources=(
                satellites.TrueColorPhysicalSource("red", 2, (red,), (red,)),
                satellites.TrueColorPhysicalSource("blue", 1, (blue,), (blue,)),
                satellites.TrueColorPhysicalSource(
                    "veggie", 3, ("veggie-a.nc",), ("veggie-a.nc",)
                ),
                satellites.TrueColorPhysicalSource(
                    "ir", 13, ("ir-a.nc",), ("ir-a.nc",)
                ),
            ),
        )

    base = app._request_key(
        *arguments,
        policy=rp.M2_POLICY,
        resolved_product="CMIPM1",
        resolved_s3_key="red-a.nc",
        resolved_source_manifest=manifest(),
    )
    different_product = app._request_key(
        *arguments,
        policy=rp.M2_POLICY,
        resolved_product="CMIPC",
        resolved_s3_key="red-a.nc",
        resolved_source_manifest=manifest(product="CMIPC"),
    )
    different_object = app._request_key(
        *arguments,
        policy=rp.M2_POLICY,
        resolved_product="CMIPM1",
        resolved_s3_key="red-b.nc",
        resolved_source_manifest=manifest(red="red-b.nc"),
    )
    different_sibling = app._request_key(
        *arguments,
        policy=rp.M2_POLICY,
        resolved_product="CMIPM1",
        resolved_s3_key="red-a.nc",
        resolved_source_manifest=manifest(blue="blue-reprocessed.nc"),
    )
    assert len({base, different_product, different_object, different_sibling}) == 4
    reordered = dataclasses.replace(
        manifest(), sources=tuple(reversed(manifest().sources))
    )
    with pytest.raises(RuntimeError, match="unregistered or incomplete"):
        app._request_key(
            *arguments,
            policy=rp.M2_POLICY,
            resolved_product="CMIPM1",
            resolved_s3_key="red-a.nc",
            resolved_source_manifest=reordered,
        )

    # Extra resolved metadata must not alter an established rollback key.
    m1_plain = app._request_key(*arguments, policy=rp.M1_POLICY)
    m1_annotated = app._request_key(
        *arguments,
        policy=rp.M1_POLICY,
        resolved_product="CMIPC",
        resolved_s3_key="different-object.nc",
    )
    assert m1_plain == m1_annotated


def test_m2_ahi_cache_identity_binds_every_selected_segment() -> None:
    body = app.RenderRequest(
        bbox=[120.0, 10.0, 132.0, 22.0], channel="true_color"
    )
    scan_start = dt.datetime(2026, 6, 12, 12, tzinfo=dt.timezone.utc)
    anchor = "noaa-himawari9/AHI-L1b-FLDK/2026/06/12/1200/"

    def manifest(blue_tail: str):
        def path(band: int, tail: str) -> str:
            return f"{anchor}B{band:02d}-{tail}.DAT.bz2"

        return satellites.TrueColorPhysicalSourceManifest(
            sensor="ahi",
            bucket="noaa-himawari9",
            product="FLDK",
            scan_start=scan_start,
            sat_name="Himawari-9",
            resolved_anchor_s3_key=anchor,
            bbox=(120.0, 10.0, 132.0, 22.0),
            sources=(
                satellites.TrueColorPhysicalSource(
                    "red", 3, (path(3, "a"),), (path(3, "a"),)
                ),
                satellites.TrueColorPhysicalSource(
                    "green", 2, (path(2, "a"),), (path(2, "a"),)
                ),
                satellites.TrueColorPhysicalSource(
                    "blue",
                    1,
                    (path(1, blue_tail),),
                    (path(1, blue_tail),),
                ),
                satellites.TrueColorPhysicalSource(
                    "veggie", 4, (path(4, "a"),), (path(4, "a"),)
                ),
                satellites.TrueColorPhysicalSource(
                    "ir", 13, (path(13, "a"),), (path(13, "a"),)
                ),
            ),
        )

    arguments = (
        body,
        "true_color",
        scan_start.isoformat(),
        "noaa-himawari9",
    )
    original = app._request_key(
        *arguments,
        policy=rp.M2_POLICY,
        resolved_product="FLDK",
        resolved_s3_key=anchor,
        resolved_source_manifest=manifest("original"),
    )
    reprocessed_sibling = app._request_key(
        *arguments,
        policy=rp.M2_POLICY,
        resolved_product="FLDK",
        resolved_s3_key=anchor,
        resolved_source_manifest=manifest("reprocessed"),
    )
    assert original != reprocessed_sibling


def test_environment_can_select_exact_registered_m2_singleton(monkeypatch) -> None:
    monkeypatch.setenv(rp.POLICY_ENV, rp.M2_POLICY.algo_version)
    assert rp.resolve_render_policy() is rp.M2_POLICY
