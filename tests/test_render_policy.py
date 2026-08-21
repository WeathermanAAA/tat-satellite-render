import dataclasses

import pytest

import render_policy as rp


def test_m0_policy_is_an_exact_legacy_behavior_declaration():
    policy = rp.resolve_render_policy(rp.M0_POLICY.algo_version)
    assert policy is rp.M0_POLICY
    assert policy.stage == "M0"
    assert policy.algo_version == "tat-truecolor-m0-baseline-v1"
    assert policy.resample_kernel == "nearest"
    assert policy.resample_in_reflectance is False
    assert policy.native_resolution_cap is False
    assert policy.ratio_sharpen is True
    assert policy.single_output_raster is False
    assert policy.freeze_sequence_geometry is False
    assert policy.blend_taper_deg == 95.0
    assert policy.gate_blend_by_day_survival is False
    assert policy.operational_integrity is False
    assert policy.glint_in_reflectance is False
    assert policy.quantizer == "production_floor"


def test_m1_changes_only_the_explicit_kernel():
    policy = rp.resolve_render_policy(rp.M1_POLICY.algo_version)
    assert policy is rp.M1_POLICY
    assert policy.stage == "M1"
    assert policy.resample_kernel == "hanning"

    m0_behavior = rp.M0_POLICY.canonical_dict()
    m1_behavior = rp.M1_POLICY.canonical_dict()
    for identity_field in ("stage", "algo_version", "resample_kernel"):
        m0_behavior.pop(identity_field)
        m1_behavior.pop(identity_field)
    assert m1_behavior == m0_behavior


def test_policy_is_immutable_and_fingerprint_is_deterministic():
    with pytest.raises(dataclasses.FrozenInstanceError):
        rp.M0_POLICY.resample_kernel = "hanning"
    assert rp.M0_POLICY.policy_sha256 == rp.M0_POLICY.policy_sha256
    assert len(rp.M0_POLICY.policy_sha256) == 64


@pytest.mark.parametrize("kernel", ["auto", "antialiased", "bicubic", "lanczos"])
def test_automatic_and_forbidden_general_kernels_cannot_be_registered(kernel):
    with pytest.raises(ValueError):
        dataclasses.replace(rp.M0_POLICY, resample_kernel=kernel)


def test_algorithm_version_must_be_canonical_and_match_stage():
    with pytest.raises(ValueError):
        dataclasses.replace(rp.M0_POLICY, algo_version="latest")
    with pytest.raises(ValueError):
        dataclasses.replace(
            rp.M0_POLICY,
            stage="M1",
            algo_version="tat-truecolor-m0-baseline-v2",
        )


def test_unknown_policy_fails_loudly(monkeypatch):
    monkeypatch.setenv(rp.POLICY_ENV, "tat-truecolor-m7-not-implemented-v1")
    with pytest.raises(RuntimeError, match="unknown TAT_TRUECOLOR_RENDER_POLICY"):
        rp.resolve_render_policy()


def test_production_boundaries_reject_an_unregistered_policy_lookalike():
    lookalike = dataclasses.replace(rp.M0_POLICY)
    assert lookalike == rp.M0_POLICY
    assert lookalike is not rp.M0_POLICY
    with pytest.raises(RuntimeError, match="unregistered render policy object"):
        rp.require_registered_policy(lookalike)


def test_m8_glint_switch_exists_but_is_parked():
    assert all(not policy.glint_in_reflectance for policy in rp.POLICIES.values())
    with pytest.raises(ValueError, match="separately reviewed M8"):
        dataclasses.replace(rp.M0_POLICY, glint_in_reflectance=True)


def test_glint_in_reflectance_can_only_be_described_by_an_unregistered_m8_policy():
    future = dataclasses.replace(
        rp.M0_POLICY,
        stage="M8",
        algo_version="tat-truecolor-m8-glint-review-v1",
        glint_in_reflectance=True,
    )
    assert future.stage == "M8"
    assert future.glint_in_reflectance
    assert future.algo_version not in rp.POLICIES
