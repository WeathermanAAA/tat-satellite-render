"""Versioned true-colour migration policy.

The rendering repair is intentionally staged.  This module is the single
place where a stage's behaviour and cache-visible algorithm identity are
declared.  M0 is measurement-only and therefore reproduces the legacy
renderer exactly; later commits add one immutable policy at a time while
retaining earlier policies for rollback.

Do not use a policy name as an appearance claim.  A policy is registered only
after the implementation and its assigned hard gates exist.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from types import MappingProxyType
from typing import Mapping


POLICY_ENV = "TAT_TRUECOLOR_RENDER_POLICY"
_VERSION_RE = re.compile(r"^tat-truecolor-m[0-8]-[a-z0-9][a-z0-9-]*-v[1-9][0-9]*$")


@dataclasses.dataclass(frozen=True)
class RenderPolicy:
    """All independently reversible true-colour migration controls.

    Float reflectance remains the eventual spatial domain and M8 remains
    parked.  The M0 values below deliberately describe current production,
    including behavior which later gates will reject.
    """

    stage: str
    algo_version: str
    resample_kernel: str = "nearest"
    resample_in_reflectance: bool = False
    native_resolution_cap: bool = False
    ratio_sharpen: bool = True
    single_output_raster: bool = False
    freeze_sequence_geometry: bool = False
    blend_taper_deg: float = 95.0
    gate_blend_by_day_survival: bool = False
    operational_integrity: bool = False
    glint_in_reflectance: bool = False

    def __post_init__(self) -> None:
        stage = self.stage.upper()
        if stage not in {f"M{i}" for i in range(9)}:
            raise ValueError(f"invalid migration stage: {self.stage!r}")
        object.__setattr__(self, "stage", stage)
        if not _VERSION_RE.fullmatch(self.algo_version):
            raise ValueError(f"non-canonical algorithm version: {self.algo_version!r}")
        if f"-{stage.lower()}-" not in self.algo_version:
            raise ValueError("algorithm version does not name its migration stage")
        if self.resample_kernel in {"auto", "antialiased"}:
            raise ValueError("automatic Matplotlib interpolation is forbidden")
        if self.resample_kernel not in {"nearest", "hanning", "fractional_area"}:
            raise ValueError(f"unsupported render kernel: {self.resample_kernel!r}")
        if not 88.0 <= float(self.blend_taper_deg) <= 95.0:
            raise ValueError("terminator blend endpoint must remain within 88..95 degrees")
        if self.glint_in_reflectance and stage != "M8":
            raise ValueError(
                "glint-in-reflectance is reserved for the separately reviewed M8 stage"
            )

    @property
    def quantizer(self) -> str:
        """The declared delivery quantizer for the policy's raster path."""

        return "round_half_up" if self.single_output_raster else "production_floor"

    def canonical_dict(self) -> dict[str, object]:
        """Stable JSON-compatible representation used by evidence ledgers."""

        return dataclasses.asdict(self)

    @property
    def policy_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


M0_POLICY = RenderPolicy(
    stage="M0",
    algo_version="tat-truecolor-m0-baseline-v1",
)

M1_POLICY = RenderPolicy(
    stage="M1",
    algo_version="tat-truecolor-m1-hanning-v1",
    resample_kernel="hanning",
)


# MappingProxyType prevents a long-running worker (or a test mutation) from
# changing the stage registry in place.
POLICIES: Mapping[str, RenderPolicy] = MappingProxyType(
    {
        M0_POLICY.algo_version: M0_POLICY,
        M1_POLICY.algo_version: M1_POLICY,
    }
)
DEFAULT_POLICY_VERSION = M1_POLICY.algo_version
ALGO_VERSION = DEFAULT_POLICY_VERSION


def require_registered_policy(policy: RenderPolicy) -> RenderPolicy:
    """Reject policy lookalikes whose version does not name this exact object.

    Production cache and raster boundaries accept only the immutable singleton
    registered for an algorithm version.  Otherwise a reconstructed or
    modified policy could reuse a canonical version while producing different
    pixels (or addressing a second cache namespace), making provenance false.
    """

    if POLICIES.get(policy.algo_version) is not policy:
        raise RuntimeError(
            f"unregistered render policy object for {policy.algo_version!r}"
        )
    return policy


def resolve_render_policy(name: str | None = None) -> RenderPolicy:
    """Resolve an explicitly registered policy, defaulting to M1.

    Unknown versions fail at startup rather than silently falling back to a
    different algorithm whose cache/provenance identity would be false.
    """

    selected = name if name is not None else os.environ.get(POLICY_ENV, ALGO_VERSION)
    try:
        return POLICIES[selected]
    except KeyError as exc:
        known = ", ".join(sorted(POLICIES))
        raise RuntimeError(
            f"unknown {POLICY_ENV}={selected!r}; registered policies: {known}"
        ) from exc
