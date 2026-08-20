from __future__ import annotations

import numpy as np
import pytest

from validation.truecolor.gates import (
    HttpFailureClass,
    classify_http_failure,
    gridline_boundary_probe,
    retry_permitted,
    sequence_dimensions_invariant,
    source_grid_invariant,
)
from validation.truecolor.metrics import (
    ObservationCandidate,
    RayleighContractError,
    assert_rayleigh_startup_ready,
    evaluate_changed_observation,
)


def _candidate(**changes: object) -> ObservationCandidate:
    values: dict[str, object] = {
        "required_bands": frozenset({"B01", "B02", "B03"}),
        "present_bands": frozenset({"B01", "B02", "B03"}),
        "fill_fraction": 0.75,
        "fill_floor": 0.05,
        "change_token": "new-slot",
        "previous_change_token": "old-slot",
        "http_status": 200,
        "all_black": False,
    }
    values.update(changes)
    return ObservationCandidate(**values)  # type: ignore[arg-type]


def test_http_200_all_black_gibs_response_does_not_advance_liveness() -> None:
    decision = evaluate_changed_observation(
        _candidate(fill_fraction=0.0, all_black=True, http_status=200)
    )
    assert not decision.qualifies
    assert "insufficient_usable_fill" in decision.reasons


def test_himawari_slot_missing_required_b03_is_rejected() -> None:
    decision = evaluate_changed_observation(
        _candidate(present_bands=frozenset({"B01", "B02"}))
    )
    assert not decision.qualifies
    assert "required_band_missing" in decision.reasons


def test_unwritable_pyspectral_directory_fails_before_render() -> None:
    factory_called = False

    def factory(_platform: str, _sensor: str) -> object:
        nonlocal factory_called
        factory_called = True
        return object()

    with pytest.raises(RayleighContractError, match="not writable"):
        assert_rayleigh_startup_ready(
            (("GOES-19", "abi"),), factory, data_directory_writable=False
        )
    assert not factory_called


@pytest.mark.parametrize("status", [401, 403])
def test_credential_failures_are_permanent_and_never_retried(status: int) -> None:
    assert classify_http_failure(status) is HttpFailureClass.PERMANENT_AUTH
    assert not retry_permitted(status)


def test_transient_server_failure_remains_retryable() -> None:
    assert classify_http_failure(503) is HttpFailureClass.TRANSIENT
    assert retry_permitted(503)


def test_non_square_drifting_sequence_is_detected_without_square_snap() -> None:
    drifting = ((1188, 713), (1188, 714), (1188, 713))
    assert not sequence_dimensions_invariant(drifting)
    assert sequence_dimensions_invariant(((1188, 713),) * 3)


def test_limb_crossing_dimension_expansion_is_detected() -> None:
    limb_crossing = ((1638, 1188), (1752, 1188), (1938, 1188))
    assert not sequence_dimensions_invariant(limb_crossing)


def test_one_row_abi_source_grid_wobble_is_detected() -> None:
    assert not source_grid_invariant(((1527, 1600), (1528, 1600), (1527, 1600)))
    assert source_grid_invariant(((1527, 1600),) * 3)


def test_twelve_degree_floating_boundary_detects_current_unquantized_policy() -> None:
    def current_policy(span: float) -> float:
        if span <= 2:
            return 0.5
        if span <= 5:
            return 1.0
        if span <= 12:
            return 2.0
        if span <= 25:
            return 5.0
        return 10.0

    probe = gridline_boundary_probe(current_policy)
    assert probe["span_nextafter"] == np.nextafter(12.0, np.inf)
    assert probe["step_exact"] == 2.0
    assert probe["step_nextafter"] == 5.0
    assert not probe["stable"]
