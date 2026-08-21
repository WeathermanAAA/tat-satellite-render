"""Mutation-sensitive M1 literal-kernel validation contracts."""

from __future__ import annotations

import pytest

import render_policy
from validation.truecolor import production_adapter
from validation.truecolor.m1 import (
    audit_production_policy,
    audit_static_policy_sources,
    trace_all_runtime_kernels,
)


DISPATCHER = """
def dispatch(ax, image, *, policy, **kwargs):
    if policy.resample_kernel == "nearest":
        return ax.imshow(image, interpolation="nearest", **kwargs)
    if policy.resample_kernel == "hanning":
        return ax.imshow(image, interpolation="hanning", **kwargs)
    raise RuntimeError
"""

RENDER = """
def render_png(ax, rgb, scalar, *, policy):
    _imshow_with_render_policy(ax, rgb, policy=policy)
    _imshow_with_render_policy(ax, scalar, cmap="gray", policy=policy)
"""

def _audit(dispatcher: str = DISPATCHER, render: str = RENDER):
    return audit_static_policy_sources(
        dispatcher,
        render,
        default_policy_is_m1=True,
        m0_rollback_registered=True,
    )


def test_historical_m1_auditor_refuses_to_rebless_the_m2_default() -> None:
    audit = audit_production_policy()
    assert not audit.passed
    assert audit.default_policy_is_m1 is False
    assert any("default render policy" in issue for issue in audit.issues)


def test_registered_m0_m1_runtime_kernel_contracts_remain_available() -> None:
    traces = trace_all_runtime_kernels()
    assert len(traces) == 4
    assert all(trace.passed for trace in traces)
    assert {
        (trace.policy_stage, trace.product_path, trace.interpolations)
        for trace in traces
    } == {
        ("M0", "rgb", ("nearest",)),
        ("M0", "scalar_regular_grid", ("nearest",)),
        ("M1", "rgb", ("hanning",)),
        ("M1", "scalar_regular_grid", ("hanning",)),
    }


@pytest.mark.parametrize(
    "mutant",
    [
        DISPATCHER.replace('interpolation="hanning"', 'interpolation="nearest"'),
        DISPATCHER.replace('interpolation="hanning"', 'interpolation="auto"'),
        DISPATCHER.replace('interpolation="hanning"', 'interpolation="antialiased"'),
        DISPATCHER.replace(
            'interpolation="hanning"',
            "interpolation=policy.resample_kernel",
        ),
        DISPATCHER.replace(', interpolation="hanning"', ""),
    ],
)
def test_nearest_automatic_dynamic_and_omitted_hanning_mutants_fail(mutant: str) -> None:
    assert not _audit(dispatcher=mutant).passed


@pytest.mark.parametrize(
    "mutant",
    [
        RENDER.replace("policy=policy", "policy=render_policy.M0_POLICY", 1),
        RENDER.replace(", policy=policy", "", 1),
        RENDER.replace(
            '    _imshow_with_render_policy(ax, scalar, cmap="gray", policy=policy)\n',
            "",
        ),
    ],
)
def test_each_rgb_and_scalar_site_must_forward_function_policy(mutant: str) -> None:
    audit = _audit(render=mutant)
    assert not audit.passed
    assert not audit.render_calls_forward_policy_parameter or audit.render_call_count != 2


def test_default_m1_and_registered_m0_are_both_mandatory() -> None:
    assert not audit_static_policy_sources(
        DISPATCHER,
        RENDER,
        default_policy_is_m1=False,
        m0_rollback_registered=True,
    ).passed
    assert not audit_static_policy_sources(
        DISPATCHER,
        RENDER,
        default_policy_is_m1=True,
        m0_rollback_registered=False,
    ).passed


def test_adapter_reports_effective_kernel_at_exactly_both_production_sites() -> None:
    m0 = production_adapter.imshow_calls(render_policy.M0_POLICY.algo_version)
    m1 = production_adapter.imshow_calls(render_policy.M1_POLICY.algo_version)
    assert [(call.branch, call.interpolation) for call in m0] == [
        ("rgb", "nearest"),
        ("scalar", "nearest"),
    ]
    assert [(call.branch, call.interpolation) for call in m1] == [
        ("rgb", "hanning"),
        ("scalar", "hanning"),
    ]
