"""M1's reversible, explicit Hanning render policy."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
from pathlib import Path

import pytest

import app
import render
from render_policy import M0_POLICY, M1_POLICY, M2_POLICY


REPO_ROOT = Path(__file__).resolve().parents[1]


class _AxesSpy:
    def __init__(self) -> None:
        self.interpolations: list[str] = []

    def imshow(self, image, **kwargs):
        self.interpolations.append(kwargs["interpolation"])
        return image


@pytest.mark.parametrize(
    ("policy", "expected"),
    [(M0_POLICY, "nearest"), (M1_POLICY, "hanning")],
)
def test_dispatcher_selects_literal_registered_kernel(policy, expected):
    axes = _AxesSpy()
    marker = object()
    assert (
        render._imshow_with_render_policy(axes, marker, policy=policy, zorder=1)
        is marker
    )
    assert axes.interpolations == [expected]


def test_dispatcher_fails_closed_for_non_imshow_future_kernel():
    fractional_area = M1_POLICY.__class__(
        **{
            **M1_POLICY.canonical_dict(),
            "resample_kernel": "fractional_area",
        }
    )
    with pytest.raises(RuntimeError, match="unregistered render policy object"):
        render._imshow_with_render_policy(
            _AxesSpy(), object(), policy=fractional_area
        )


def test_cache_and_raster_reject_policy_lookalikes():
    lookalike = dataclasses.replace(M0_POLICY)
    body = app.RenderRequest(
        bbox=[-99.0, 34.0, -82.0, 49.0], channel="clean_ir"
    )
    with pytest.raises(RuntimeError, match="unregistered render policy object"):
        app._request_key(
            body,
            "clean_ir",
            "2026-06-12T06:00:00",
            "noaa-goes19",
            policy=lookalike,
        )
    with pytest.raises(RuntimeError, match="unregistered render policy object"):
        render._imshow_with_render_policy(
            _AxesSpy(), object(), policy=lookalike
        )
    with pytest.raises(RuntimeError, match="unregistered render policy object"):
        render.render_png(
            None,
            [-99.0, 34.0, -82.0, 49.0],
            13,
            "2026-06-12 06:00",
            "clean_ir",
            policy=lookalike,
        )


def test_production_imshow_policy_is_literal_and_never_automatic():
    tree = ast.parse((REPO_ROOT / "render.py").read_text(encoding="utf-8"))
    interpolation_values: list[object] = []
    render_dispatch_calls = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "imshow":
            interpolation = next(
                (kw.value for kw in node.keywords if kw.arg == "interpolation"),
                None,
            )
            assert isinstance(interpolation, ast.Constant)
            interpolation_values.append(interpolation.value)
        if isinstance(node.func, ast.Name) and node.func.id == "_imshow_with_render_policy":
            policy_keyword = next(
                (kw for kw in node.keywords if kw.arg == "policy"),
                None,
            )
            assert policy_keyword is not None
            assert isinstance(policy_keyword.value, ast.Name)
            assert policy_keyword.value.id == "policy"
            render_dispatch_calls += 1

    # One literal per supported rollback branch, used by both production
    # regular-grid paths. A mutant that restores nearest at either render site,
    # or delegates to auto/antialiased, changes these exact assertions.
    assert interpolation_values == ["nearest", "hanning"]
    assert render_dispatch_calls == 2
    assert not {"auto", "antialiased"}.intersection(interpolation_values)


def test_render_png_requires_an_explicit_policy_keyword():
    policy_param = inspect.signature(render.render_png).parameters["policy"]
    assert policy_param.kind is inspect.Parameter.KEYWORD_ONLY
    assert policy_param.default is inspect.Parameter.empty


def _cache_raw(body: app.RenderRequest) -> str:
    return (
        f"{body.bbox}|2026-06-12T06:00:00|clean_ir|{body.enhancement}"
        "|noaa-goes19|q=default"
    )


def test_m0_cache_key_is_legacy_and_m1_opens_reversible_generation():
    body = app.RenderRequest(
        bbox=[-99.0, 34.0, -82.0, 49.0], channel="clean_ir"
    )
    raw = _cache_raw(body)
    legacy = hashlib.sha256(raw.encode()).hexdigest()

    m0 = app._request_key(
        body,
        "clean_ir",
        "2026-06-12T06:00:00",
        "noaa-goes19",
        policy=M0_POLICY,
    )
    m1 = app._request_key(
        body,
        "clean_ir",
        "2026-06-12T06:00:00",
        "noaa-goes19",
        policy=M1_POLICY,
    )

    assert m0 == legacy
    assert m1 == hashlib.sha256(
        (raw + f"|algo={M1_POLICY.algo_version}").encode()
    ).hexdigest()
    assert m1 != m0


def test_worker_resolves_one_request_policy_for_cache_render_and_provenance():
    assert app.PROCESS_RENDER_POLICY is M2_POLICY
    source = inspect.getsource(app)
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "request_policy"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    resolver = assignments[0].value
    assert isinstance(resolver, ast.Call)
    assert isinstance(resolver.func, ast.Name)
    assert resolver.func.id == "_effective_render_policy"
    process_keyword = next(
        item.value for item in resolver.keywords if item.arg == "process_policy"
    )
    assert isinstance(process_keyword, ast.Name)
    assert process_keyword.id == "PROCESS_RENDER_POLICY"

    # The same resolved singleton must bind every cache/render-facing policy
    # boundary. A mutant that reuses the process default for just one consumer
    # would let M1 and M2 share bytes under a lying identity.
    policy_targets = {
        "_request_key": [],
        "_fetch_true_color_for_policy": [],
        "_reported_point_stride": [],
        "render_png": [],
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else None
        )
        if target not in policy_targets:
            continue
        keyword = next(
            (item.value for item in node.keywords if item.arg == "policy"),
            None,
        )
        policy_targets[target].append(keyword)

    assert set(policy_targets) == {
        "_request_key",
        "_fetch_true_color_for_policy",
        "_reported_point_stride",
        "render_png",
    }
    for target, values in policy_targets.items():
        assert len(values) == 1, f"unexpected {target} call count"
        assert isinstance(values[0], ast.Name)
        assert values[0].id == "request_policy"

    algorithm_headers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "X-Algorithm-Version":
                algorithm_headers.append(value)
    assert len(algorithm_headers) == 1
    header_value = algorithm_headers[0]
    assert isinstance(header_value, ast.Attribute)
    assert header_value.attr == "algo_version"
    assert isinstance(header_value.value, ast.Name)
    assert header_value.value.id == "request_policy"

    # Health reports the configured process generation; response provenance
    # reports the request-scoped effective generation proven above.
    assert '"algorithm_version": PROCESS_RENDER_POLICY.algo_version' in source


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"process_policy": M1_POLICY, "is_true_color": True}, M1_POLICY),
        ({"is_true_color": True, "use_archive": False}, M2_POLICY),
        (
            {"is_true_color": True, "crosses_antimeridian": True},
            M1_POLICY,
        ),
        (
            {"is_true_color": True, "m2_truecolor_supported": False},
            M1_POLICY,
        ),
        ({"is_true_color": False, "use_archive": True}, M2_POLICY),
        (
            {
                "is_true_color": False,
                "use_archive": True,
                "m2_archive_scalar_supported": False,
            },
            M1_POLICY,
        ),
        ({"is_true_color": False, "use_archive": False}, M1_POLICY),
        (
            {
                "is_true_color": False,
                "use_archive": True,
                "crosses_antimeridian": True,
            },
            M1_POLICY,
        ),
        (
            {"is_true_color": False, "use_archive": True, "is_backdrop": True},
            M1_POLICY,
        ),
        (
            {"is_true_color": False, "use_archive": True, "output_format": "btpng"},
            M1_POLICY,
        ),
    ),
)
def test_request_policy_preserves_m1_on_paths_outside_m2_authority(
    changes, expected
):
    arguments = {
        "process_policy": M2_POLICY,
        "is_true_color": False,
        "use_archive": False,
        "crosses_antimeridian": False,
        "is_backdrop": False,
        "output_format": "png",
        "m2_truecolor_supported": True,
        "m2_archive_scalar_supported": True,
    }
    arguments.update(changes)
    assert app._effective_render_policy(**arguments) is expected

    lookalike = dataclasses.replace(arguments["process_policy"])
    with pytest.raises(RuntimeError, match="unregistered render policy object"):
        app._effective_render_policy(**(arguments | {"process_policy": lookalike}))
