"""Exact physical-field resampling for the M2 imagery-layer boundary.

M2 performs the one authoritative imagery-layer spatial operation before
display encoding.  The operation here is deliberately narrow: it replays an
axis-aligned affine captured from Matplotlib's real ``AxesImage.make_image``
geometry with the literal Agg Hanning kernel.  Values and coverage are
filtered through the same transform using premultiplied coverage.

The final figure compositor can still clip an internal Agg raster at a canvas
edge.  ``RasterPlan`` therefore distinguishes the internal destination raster
from an optional, dynamically derived internal-to-delivered slice.  Neither a
nominal canvas size nor a version-specific one-pixel offset is sufficient.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from typing import TYPE_CHECKING, TypeAlias

import numpy as np


if TYPE_CHECKING:
    from matplotlib.backend_bases import RendererBase
    from matplotlib.image import AxesImage


Shape2D: TypeAlias = tuple[int, int]
Extent: TypeAlias = tuple[float, float, float, float]
Matrix3x3: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
InternalBounds: TypeAlias = tuple[int, int, int, int]

KERNEL = "hanning"
DOMAIN = "physical_float32"
DEFAULT_COVERAGE_THRESHOLD = 1.0e-6
PHYSICAL_DOMAIN_TRANSPORT = "coverage_bound_affine_unit_interval_v1"
_GEOMETRY_ATOL = 1.0e-7


class HanningBackendError(RuntimeError):
    """The private Agg Hanning contract is absent or has changed."""


@dataclasses.dataclass(frozen=True, slots=True)
class IdentityDrawTrace:
    """Proof record for an integer-positioned, one-to-one FigureImage draw."""

    source_shape: Shape2D
    output_shape: Shape2D
    device_bounds: Extent
    affine_matrix: Matrix3x3
    interpolation: str


def make_hanning_planning_image(axes, values: np.ndarray, **kwargs):
    """Create the disposable geometry probe with one literal kernel.

    Keeping this call beside the private-Agg boundary prevents M2 planning
    from adding a third apparent M0/M1 production dispatcher site.  Callers
    cannot weaken or dynamically replace the interpolation keyword.
    """

    if "interpolation" in kwargs:
        raise TypeError("planning-image interpolation is fixed to literal hanning")
    return axes.imshow(values, interpolation="hanning", **kwargs)


def _finite_tuple(values, *, length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _shape2d(values, *, name: str) -> Shape2D:
    result = tuple(int(value) for value in values)
    if len(result) != 2 or any(value <= 0 for value in result):
        raise ValueError(f"{name} must be a positive (height, width) pair")
    return result  # type: ignore[return-value]


def _matrix3x3(transform) -> Matrix3x3:
    raw = transform.get_matrix() if hasattr(transform, "get_matrix") else transform
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("affine transform must be a finite 3x3 matrix")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-12):
        raise ValueError("affine transform must have the homogeneous row (0, 0, 1)")
    if not np.allclose(matrix[[0, 1], [1, 0]], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("M2 raster transforms must be axis-aligned (no shear/rotation)")
    if matrix[0, 0] == 0.0 or matrix[1, 1] == 0.0:
        raise ValueError("M2 raster transforms must have non-zero x/y scale")
    return tuple(tuple(float(value) for value in row) for row in matrix)  # type: ignore[return-value]


def _fractional_phase(translation: float) -> float:
    """Return pixel-centre phase implied by an Agg-axis translation."""

    return float((0.5 - translation) % 1.0)


def _canvas_orientation(
    values: np.ndarray,
    *,
    flip_y: bool,
    flip_x: bool,
) -> np.ndarray:
    """Apply the orientation measured from the live renderer canvas."""

    result = np.asarray(values)
    if flip_y:
        result = np.flip(result, axis=0)
    if flip_x:
        result = np.flip(result, axis=1)
    return result


def _delivered_canvas_crop(renderer, device_bounds: Extent) -> np.ndarray:
    """Extract integer Agg device bounds from the live top-origin canvas."""

    if not hasattr(renderer, "buffer_rgba"):
        raise RuntimeError("renderer does not expose an RGBA canvas for binding")
    canvas = np.asarray(renderer.buffer_rgba())
    if canvas.ndim != 3 or canvas.shape[-1] != 4:
        raise RuntimeError(f"unexpected renderer canvas shape: {canvas.shape}")
    left, bottom, right, top = (int(round(value)) for value in device_bounds)
    canvas_height, canvas_width = canvas.shape[:2]
    if not (
        0 <= left < right <= canvas_width and 0 <= bottom < top <= canvas_height
    ):
        raise ValueError("delivered device bounds lie outside the renderer canvas")
    return canvas[canvas_height - top : canvas_height - bottom, left:right]


def _bind_internal_to_canvas(
    internal_rgba: np.ndarray,
    canvas_rgba: np.ndarray,
    *,
    boundary_exclusion_px: int = 1,
) -> tuple[InternalBounds, float, float, bool, bool]:
    """Find the unique orientation and internal slice by pixel identity."""

    raw_internal = np.asarray(internal_rgba)
    delivered = np.asarray(canvas_rgba)
    if raw_internal.ndim != 3 or raw_internal.shape[-1] != 4:
        raise ValueError("internal identity raster must be RGBA")
    if delivered.ndim != 3 or delivered.shape[-1] != 4:
        raise ValueError("delivered identity raster must be RGBA")
    excess_y = raw_internal.shape[0] - delivered.shape[0]
    excess_x = raw_internal.shape[1] - delivered.shape[1]
    if excess_y < 0 or excess_x < 0:
        raise RuntimeError("internal Agg raster is smaller than its canvas crop")
    if excess_y > 64 or excess_x > 64:
        raise RuntimeError(
            "internal/canvas geometry mismatch is too large for identity binding"
        )
    boundary = int(boundary_exclusion_px)
    if boundary < 0:
        raise ValueError("boundary exclusion must be non-negative")
    if 2 * boundary >= min(delivered.shape[:2]):
        boundary = 0

    candidates: list[tuple[InternalBounds, float, float, bool, bool]] = []
    for flip_y in (False, True):
        for flip_x in (False, True):
            internal = _canvas_orientation(
                raw_internal,
                flip_y=flip_y,
                flip_x=flip_x,
            )
            for row_start in range(excess_y + 1):
                for column_start in range(excess_x + 1):
                    bounds = (
                        row_start,
                        row_start + delivered.shape[0],
                        column_start,
                        column_start + delivered.shape[1],
                    )
                    candidate = internal[
                        bounds[0] : bounds[1], bounds[2] : bounds[3]
                    ]
                    # RGB is load-bearing. Alpha must also be opaque so a
                    # transparent coincidence cannot bind the geometry.
                    if not np.all(candidate[..., 3] == 255) or not np.all(
                        delivered[..., 3] == 255
                    ):
                        raise RuntimeError(
                            "canvas identity binding requires an opaque planning image"
                        )
                    equal = np.all(
                        candidate[..., :3] == delivered[..., :3], axis=-1
                    )
                    scored = (
                        equal[
                            boundary : equal.shape[0] - boundary,
                            boundary : equal.shape[1] - boundary,
                        ]
                        if boundary
                        else equal
                    )
                    candidates.append(
                        (
                            bounds,
                            float(equal.mean()),
                            float(scored.mean()),
                            flip_y,
                            flip_x,
                        )
                    )

    passing = [candidate for candidate in candidates if candidate[2] >= 0.998]
    if len(passing) != 1:
        raise RuntimeError(
            "internal-to-canvas mapping is not a unique >=99.8% identity: "
            + repr(
                [
                    (bounds, score, flip_y, flip_x)
                    for bounds, _, score, flip_y, flip_x in candidates
                ]
            )
        )
    return passing[0]


@dataclasses.dataclass(frozen=True, slots=True)
class RasterPlan:
    """Immutable replay contract for one exact Agg imagery raster.

    ``affine_matrix`` maps source-array pixel-edge coordinates into the local
    internal destination raster.  Extents use Matplotlib image order
    ``(left, right, bottom, top)``.  Device bounds use geometric
    ``(left, bottom, right, top)`` order.  The optional delivered bounds are
    ``(row_start, row_stop, column_start, column_stop)`` in the internal Agg
    raster and are never inferred from a library version.
    """

    source_height: int
    source_width: int
    destination_height: int
    destination_width: int
    source_extent: Extent
    view_extent: Extent
    device_bounds: Extent
    affine_matrix: Matrix3x3
    sampling_phase_yx: tuple[float, float]
    delivered_internal_bounds: InternalBounds | None = None
    delivered_device_bounds: Extent | None = None
    delivered_mapping_full_identity: float | None = None
    delivered_mapping_scored_identity: float | None = None
    internal_to_canvas_flip_y: bool | None = None
    internal_to_canvas_flip_x: bool | None = None
    kernel: str = KERNEL
    domain: str = DOMAIN

    def __post_init__(self) -> None:
        source_shape = _shape2d(self.source_shape, name="source_shape")
        destination_shape = _shape2d(
            self.destination_shape, name="destination_shape"
        )
        object.__setattr__(self, "source_height", source_shape[0])
        object.__setattr__(self, "source_width", source_shape[1])
        object.__setattr__(self, "destination_height", destination_shape[0])
        object.__setattr__(self, "destination_width", destination_shape[1])

        source_extent = _finite_tuple(
            self.source_extent, length=4, name="source_extent"
        )
        view_extent = _finite_tuple(self.view_extent, length=4, name="view_extent")
        device_bounds = _finite_tuple(
            self.device_bounds, length=4, name="device_bounds"
        )
        if source_extent[0] == source_extent[1] or source_extent[2] == source_extent[3]:
            raise ValueError("source_extent must have non-zero x/y span")
        if view_extent[0] == view_extent[1] or view_extent[2] == view_extent[3]:
            raise ValueError("view_extent must have non-zero x/y span")
        if not (
            device_bounds[2] > device_bounds[0]
            and device_bounds[3] > device_bounds[1]
        ):
            raise ValueError("device_bounds must have positive width and height")
        if not math.isclose(
            device_bounds[2] - device_bounds[0],
            destination_shape[1],
            rel_tol=0.0,
            abs_tol=_GEOMETRY_ATOL,
        ) or not math.isclose(
            device_bounds[3] - device_bounds[1],
            destination_shape[0],
            rel_tol=0.0,
            abs_tol=_GEOMETRY_ATOL,
        ):
            raise ValueError(
                "device bounds must describe the exact internal destination shape"
            )
        object.__setattr__(self, "source_extent", source_extent)
        object.__setattr__(self, "view_extent", view_extent)
        object.__setattr__(self, "device_bounds", device_bounds)
        object.__setattr__(self, "affine_matrix", _matrix3x3(self.affine_matrix))

        phase = _finite_tuple(
            self.sampling_phase_yx, length=2, name="sampling_phase_yx"
        )
        if any(not 0.0 <= value < 1.0 for value in phase):
            raise ValueError("sampling phases must be in [0, 1)")
        object.__setattr__(self, "sampling_phase_yx", phase)

        if self.delivered_internal_bounds is not None:
            bounds = tuple(int(value) for value in self.delivered_internal_bounds)
            if len(bounds) != 4:
                raise ValueError("delivered_internal_bounds must contain four values")
            row_start, row_stop, column_start, column_stop = bounds
            if not (
                0 <= row_start < row_stop <= destination_shape[0]
                and 0 <= column_start < column_stop <= destination_shape[1]
            ):
                raise ValueError(
                    "delivered_internal_bounds must select a non-empty destination slice"
                )
            object.__setattr__(self, "delivered_internal_bounds", bounds)

            for name, value in (
                ("internal_to_canvas_flip_y", self.internal_to_canvas_flip_y),
                ("internal_to_canvas_flip_x", self.internal_to_canvas_flip_x),
            ):
                if type(value) is not bool:
                    raise ValueError(
                        f"{name} must be an explicitly captured boolean when "
                        "delivered bounds are present"
                    )
        elif (
            self.internal_to_canvas_flip_y is not None
            or self.internal_to_canvas_flip_x is not None
        ):
            raise ValueError(
                "internal-to-canvas orientation requires delivered bounds"
            )

        if self.delivered_device_bounds is not None:
            delivered_device_bounds = _finite_tuple(
                self.delivered_device_bounds,
                length=4,
                name="delivered_device_bounds",
            )
            if self.delivered_internal_bounds is None:
                raise ValueError(
                    "delivered_device_bounds require delivered_internal_bounds"
                )
            left, bottom, right, top = delivered_device_bounds
            if not right > left or not top > bottom:
                raise ValueError("delivered_device_bounds must have positive area")
            if any(
                not math.isclose(value, round(value), rel_tol=0.0, abs_tol=1.0e-7)
                for value in delivered_device_bounds
            ):
                raise ValueError(
                    "delivered device bounds must lie on integer canvas pixels"
                )
            delivered_shape = self.delivered_shape
            assert delivered_shape is not None
            if not math.isclose(
                right - left,
                delivered_shape[1],
                rel_tol=0.0,
                abs_tol=_GEOMETRY_ATOL,
            ) or not math.isclose(
                top - bottom,
                delivered_shape[0],
                rel_tol=0.0,
                abs_tol=_GEOMETRY_ATOL,
            ):
                raise ValueError(
                    "delivered device bounds and internal slice shapes disagree"
                )
            object.__setattr__(
                self, "delivered_device_bounds", delivered_device_bounds
            )

        identities = (
            self.delivered_mapping_full_identity,
            self.delivered_mapping_scored_identity,
        )
        if any(value is not None for value in identities):
            if self.delivered_internal_bounds is None or any(
                value is None for value in identities
            ):
                raise ValueError(
                    "delivered identity metrics require a complete delivered binding"
                )
            full_identity, scored_identity = (float(value) for value in identities)
            if not 0.0 <= full_identity <= 1.0 or not 0.0 <= scored_identity <= 1.0:
                raise ValueError("delivered identity metrics must be in [0, 1]")
            object.__setattr__(
                self, "delivered_mapping_full_identity", full_identity
            )
            object.__setattr__(
                self, "delivered_mapping_scored_identity", scored_identity
            )

        if self.kernel != KERNEL:
            raise ValueError("M2 RasterPlan requires the literal 'hanning' kernel")
        if self.domain != DOMAIN:
            raise ValueError("M2 RasterPlan requires the physical-float32 domain")

    @classmethod
    def from_affine(
        cls,
        *,
        source_shape: Shape2D,
        destination_shape: Shape2D,
        transform,
        source_extent: Extent | None = None,
        view_extent: Extent | None = None,
        device_bounds: Extent | None = None,
        sampling_phase_yx: tuple[float, float] | None = None,
        delivered_internal_bounds: InternalBounds | None = None,
        delivered_device_bounds: Extent | None = None,
        delivered_mapping_full_identity: float | None = None,
        delivered_mapping_scored_identity: float | None = None,
        internal_to_canvas_flip_y: bool | None = None,
        internal_to_canvas_flip_x: bool | None = None,
    ) -> "RasterPlan":
        """Build a plan from a captured source-to-internal affine."""

        source_height, source_width = _shape2d(source_shape, name="source_shape")
        destination_height, destination_width = _shape2d(
            destination_shape, name="destination_shape"
        )
        matrix = _matrix3x3(transform)
        source_extent = source_extent or (
            0.0,
            float(source_width),
            0.0,
            float(source_height),
        )
        view_extent = view_extent or source_extent
        device_bounds = device_bounds or (
            0.0,
            0.0,
            float(destination_width),
            float(destination_height),
        )
        if sampling_phase_yx is None:
            sampling_phase_yx = (
                _fractional_phase(matrix[1][2]),
                _fractional_phase(matrix[0][2]),
            )
        return cls(
            source_height=source_height,
            source_width=source_width,
            destination_height=destination_height,
            destination_width=destination_width,
            source_extent=source_extent,
            view_extent=view_extent,
            device_bounds=device_bounds,
            affine_matrix=matrix,
            sampling_phase_yx=sampling_phase_yx,
            delivered_internal_bounds=delivered_internal_bounds,
            delivered_device_bounds=delivered_device_bounds,
            delivered_mapping_full_identity=delivered_mapping_full_identity,
            delivered_mapping_scored_identity=delivered_mapping_scored_identity,
            internal_to_canvas_flip_y=internal_to_canvas_flip_y,
            internal_to_canvas_flip_x=internal_to_canvas_flip_x,
        )

    @classmethod
    def from_axes_image(
        cls,
        image: "AxesImage",
        renderer: "RendererBase",
        *,
        delivered_device_bounds: Extent | None = None,
    ) -> "RasterPlan":
        """Capture the exact internal geometry produced by a real AxesImage.

        The caller must finish layout and draw the figure before invoking this
        method.  ``make_image`` is called once with the literal Hanning image;
        its returned shape, source-to-internal affine, and device origin are
        authoritative.  If final integer imagery bounds are known, pass them
        in Agg lower-left device coordinates to retain the derived internal
        slice as well.
        """

        if image.get_interpolation() != KERNEL:
            raise ValueError("axes-image planning requires interpolation='hanning'")
        source = image.get_array()
        if source is None or np.ndim(source) not in {2, 3}:
            raise ValueError("axes image must contain a two-dimensional raster")
        source_shape = (int(source.shape[0]), int(source.shape[1]))
        made, origin_x, origin_y, transform = image.make_image(
            renderer, magnification=1.0, unsampled=False
        )
        if made is None or transform is None:
            raise RuntimeError("axes image does not intersect its resolved clip bounds")
        if made.ndim != 3 or made.shape[-1] != 4:
            raise HanningBackendError(
                f"unexpected AxesImage.make_image result shape: {made.shape}"
            )
        destination_shape = (int(made.shape[0]), int(made.shape[1]))
        device_bounds = (
            float(origin_x),
            float(origin_y),
            float(origin_x + destination_shape[1]),
            float(origin_y + destination_shape[0]),
        )
        delivered_internal_bounds = None
        mapping_full_identity = None
        mapping_scored_identity = None
        flip_y = None
        flip_x = None
        if delivered_device_bounds is not None:
            normalized_device_bounds = _finite_tuple(
                delivered_device_bounds,
                length=4,
                name="delivered_device_bounds",
            )
            canvas_crop = _delivered_canvas_crop(renderer, normalized_device_bounds)
            (
                delivered_internal_bounds,
                mapping_full_identity,
                mapping_scored_identity,
                flip_y,
                flip_x,
            ) = _bind_internal_to_canvas(
                made,
                canvas_crop,
            )

        axes = image.axes
        source_extent = tuple(float(value) for value in image.get_extent())
        view_extent = (
            float(axes.get_xlim()[0]),
            float(axes.get_xlim()[1]),
            float(axes.get_ylim()[0]),
            float(axes.get_ylim()[1]),
        )
        return cls.from_affine(
            source_shape=source_shape,
            destination_shape=destination_shape,
            transform=transform,
            source_extent=source_extent,  # type: ignore[arg-type]
            view_extent=view_extent,
            device_bounds=device_bounds,
            delivered_internal_bounds=delivered_internal_bounds,
            delivered_device_bounds=delivered_device_bounds,
            delivered_mapping_full_identity=mapping_full_identity,
            delivered_mapping_scored_identity=mapping_scored_identity,
            internal_to_canvas_flip_y=flip_y,
            internal_to_canvas_flip_x=flip_x,
        )

    @classmethod
    def from_axes_image_delivered(
        cls,
        image: "AxesImage",
        renderer: "RendererBase",
        *,
        delivered_device_bounds: Extent,
    ) -> "RasterPlan":
        """Capture the source affine directly in delivered device pixels.

        ``AxesImage.make_image`` remains a required live backend probe, but its
        private intermediate raster is not stable across supported Matplotlib
        ABIs: the same resolved axes can become 1188 rows on one version and
        1189 rows plus a one-row canvas clip on another.  Scientific sampling
        must instead follow the public, post-layout image transform into the
        exact integer device rectangle that receives the final FigureImage.

        The returned plan therefore has one full internal/delivered raster and
        no inferred or version-specific clip.  A sampled affine check rejects
        projection warps or shear rather than approximating them.
        """

        if image.get_interpolation() != KERNEL:
            raise ValueError("axes-image planning requires interpolation='hanning'")
        source = image.get_array()
        if source is None or np.ndim(source) not in {2, 3}:
            raise ValueError("axes image must contain a two-dimensional raster")
        source_shape = (int(source.shape[0]), int(source.shape[1]))

        # Exercise the actual installed image backend after layout. Its private
        # shape/affine is deliberately not promoted to scientific geometry.
        made, _, _, private_transform = image.make_image(
            renderer, magnification=1.0, unsampled=False
        )
        if made is None or private_transform is None:
            raise RuntimeError("axes image does not intersect its resolved clip bounds")
        if made.ndim != 3 or made.shape[-1] != 4:
            raise HanningBackendError(
                f"unexpected AxesImage.make_image result shape: {made.shape}"
            )

        bounds = _finite_tuple(
            delivered_device_bounds,
            length=4,
            name="delivered_device_bounds",
        )
        if any(
            not math.isclose(value, round(value), rel_tol=0.0, abs_tol=1.0e-7)
            for value in bounds
        ):
            raise ValueError("delivered device bounds must lie on integer pixels")
        left_device, bottom_device, right_device, top_device = bounds
        destination_shape = (
            int(round(top_device - bottom_device)),
            int(round(right_device - left_device)),
        )
        _shape2d(destination_shape, name="destination_shape")

        source_height, source_width = source_shape
        left, right, bottom, top = (
            float(value) for value in image.get_extent()
        )
        origin = image.origin
        if origin not in {"upper", "lower"}:
            raise ValueError(f"unsupported AxesImage origin: {origin!r}")

        source_points = np.asarray(
            (
                (0.0, 0.0),
                (float(source_width), 0.0),
                (0.0, float(source_height)),
                (float(source_width), float(source_height)),
                (float(source_width) / 2.0, float(source_height) / 2.0),
            ),
            dtype=np.float64,
        )
        data_points = np.empty_like(source_points)
        data_points[:, 0] = left + (
            source_points[:, 0] * (right - left) / source_width
        )
        if origin == "upper":
            data_points[:, 1] = top + (
                source_points[:, 1] * (bottom - top) / source_height
            )
        else:
            data_points[:, 1] = bottom + (
                source_points[:, 1] * (top - bottom) / source_height
            )
        image_transform = image.get_transform()
        if not image_transform.is_affine:
            raise RuntimeError("M2 delivered image transform must be affine")
        display_points = np.asarray(
            image_transform.transform(data_points), dtype=np.float64
        )
        if display_points.shape != source_points.shape or not np.all(
            np.isfinite(display_points)
        ):
            raise RuntimeError("resolved AxesImage transform returned invalid points")
        local_points = display_points - np.asarray(
            (left_device, bottom_device), dtype=np.float64
        )

        x_scale = (local_points[1, 0] - local_points[0, 0]) / source_width
        x_shear = (local_points[2, 0] - local_points[0, 0]) / source_height
        y_shear = (local_points[1, 1] - local_points[0, 1]) / source_width
        y_scale = (local_points[2, 1] - local_points[0, 1]) / source_height
        matrix_array = np.asarray(
            (
                (x_scale, x_shear, local_points[0, 0]),
                (y_shear, y_scale, local_points[0, 1]),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        homogeneous = np.column_stack(
            (source_points, np.ones(source_points.shape[0], dtype=np.float64))
        )
        replayed = (matrix_array @ homogeneous.T).T[:, :2]
        maximum_residual = float(np.max(np.abs(replayed - local_points)))
        if maximum_residual > 1.0e-7:
            raise RuntimeError(
                "resolved AxesImage transform is not one axis-aligned affine: "
                f"max residual {maximum_residual!r}"
            )
        matrix = _matrix3x3(matrix_array)
        destination_height, destination_width = destination_shape
        source_rectangle = replayed[:4]
        if (
            float(np.max(source_rectangle[:, 0])) <= 0.0
            or float(np.min(source_rectangle[:, 0])) >= destination_width
            or float(np.max(source_rectangle[:, 1])) <= 0.0
            or float(np.min(source_rectangle[:, 1])) >= destination_height
        ):
            raise RuntimeError("resolved source rectangle misses delivered device bounds")
        axes = image.axes
        view_extent = (
            float(axes.get_xlim()[0]),
            float(axes.get_xlim()[1]),
            float(axes.get_ylim()[0]),
            float(axes.get_ylim()[1]),
        )
        return cls.from_affine(
            source_shape=source_shape,
            destination_shape=destination_shape,
            transform=matrix,
            source_extent=(left, right, bottom, top),
            view_extent=view_extent,
            device_bounds=bounds,
            delivered_internal_bounds=(
                0,
                destination_height,
                0,
                destination_width,
            ),
            delivered_device_bounds=bounds,
            # Agg device arrays are bottom-origin while the delivered canvas
            # is top-origin, independently of the source image's own origin.
            internal_to_canvas_flip_y=True,
            internal_to_canvas_flip_x=False,
        )

    @property
    def source_shape(self) -> Shape2D:
        return (self.source_height, self.source_width)

    @property
    def destination_shape(self) -> Shape2D:
        return (self.destination_height, self.destination_width)

    @property
    def x_transform(self) -> tuple[float, float]:
        return (self.affine_matrix[0][0], self.affine_matrix[0][2])

    @property
    def y_transform(self) -> tuple[float, float]:
        return (self.affine_matrix[1][1], self.affine_matrix[1][2])

    @property
    def x_source_pixels_per_destination(self) -> float:
        return self.source_width / self.destination_width

    @property
    def y_source_pixels_per_destination(self) -> float:
        return self.source_height / self.destination_height

    @property
    def delivered_shape(self) -> Shape2D | None:
        if self.delivered_internal_bounds is None:
            return None
        row_start, row_stop, column_start, column_stop = (
            self.delivered_internal_bounds
        )
        return (row_stop - row_start, column_stop - column_start)

    @property
    def is_identity_mapping(self) -> bool:
        """Whether this plan is exactly identity, including phase/orientation."""

        return self.source_shape == self.destination_shape and self.affine_matrix == (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    def delivered_values(self, values: np.ndarray) -> np.ndarray:
        """Return the exact top-origin canvas raster selected from Agg output.

        Resampling returns the complete internal Agg array.  The captured
        affine determines its canvas orientation; the verified bounds are
        defined only after that whole-array orientation.  Orientation must
        therefore precede slicing, especially for an asymmetric legacy
        one-row clip.  This is an index conversion, not another spatial sample.
        """

        if self.delivered_internal_bounds is None:
            raise RuntimeError("RasterPlan has no verified internal-to-delivered slice")
        array = np.asarray(values)
        if array.shape[:2] != self.destination_shape:
            raise ValueError(
                f"destination array shape {array.shape[:2]} does not match "
                f"plan {self.destination_shape}"
            )
        assert self.internal_to_canvas_flip_y is not None
        assert self.internal_to_canvas_flip_x is not None
        oriented = _canvas_orientation(
            array,
            flip_y=self.internal_to_canvas_flip_y,
            flip_x=self.internal_to_canvas_flip_x,
        )
        row_start, row_stop, column_start, column_stop = (
            self.delivered_internal_bounds
        )
        return oriented[row_start:row_stop, column_start:column_stop, ...]


_BACKEND_PROBE_LOCK = threading.Lock()
_BACKEND_PROBE_KEYS: set[tuple[str, int, int]] = set()


def _ensure_hanning_backend():
    """Fail closed if Matplotlib's private literal-Hanning ABI drifts."""

    import matplotlib
    from matplotlib import _image
    import matplotlib.image as mpl_image
    from matplotlib.transforms import Affine2D

    hanning = getattr(_image, "HANNING", None)
    mapping = getattr(mpl_image, "_interpd_", None)
    resample = getattr(_image, "resample", None)
    if hanning is None or not callable(resample):
        raise HanningBackendError("Matplotlib Agg Hanning resampler is unavailable")
    if not isinstance(mapping, dict) or mapping.get(KERNEL) != hanning:
        raise HanningBackendError(
            "Matplotlib's literal 'hanning' interpolation mapping has changed"
        )

    key = (str(matplotlib.__version__), id(resample), int(hanning))
    with _BACKEND_PROBE_LOCK:
        if key in _BACKEND_PROBE_KEYS:
            return _image, Affine2D
        constant = np.full((3, 5), np.float32(0.625), dtype=np.float32)
        observed = np.zeros((4, 7), dtype=np.float32)
        transform = Affine2D().scale(7.0 / 5.0, 4.0 / 3.0)
        try:
            resample(
                constant,
                observed,
                transform,
                hanning,
                True,
                1.0,
                True,
                4.0,
            )
        except Exception as exc:  # pragma: no cover - version-dependent failure
            raise HanningBackendError(
                "Matplotlib Agg Hanning resampler signature/behavior changed"
            ) from exc
        if observed.dtype != np.float32 or not np.allclose(
            observed, constant[0, 0], rtol=0.0, atol=2.0e-6
        ):
            raise HanningBackendError(
                "Matplotlib Agg Hanning resampler failed the float32 DC guard"
            )
        _BACKEND_PROBE_KEYS.add(key)
    return _image, Affine2D


def _hanning_resample(values: np.ndarray, plan: RasterPlan) -> np.ndarray:
    if plan.is_identity_mapping:
        return np.array(values, dtype=np.float32, copy=True, order="C")
    agg_image, affine_type = _ensure_hanning_backend()
    source = np.ascontiguousarray(values, dtype=np.float32)
    destination = np.zeros(plan.destination_shape, dtype=np.float32)
    transform = affine_type(np.asarray(plan.affine_matrix, dtype=np.float64))
    try:
        agg_image.resample(
            source,
            destination,
            transform,
            agg_image.HANNING,
            True,
            1.0,
            True,
            4.0,
        )
    except Exception as exc:  # pragma: no cover - version-dependent failure
        raise HanningBackendError("literal float32 Agg Hanning replay failed") from exc
    return destination


def _physical_field(values, plan: RasterPlan, *, name: str) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise TypeError(f"{name} must be a plain numpy.ndarray")
    array = values
    if array.dtype != np.float32:
        raise TypeError(f"{name} must be float32 physical data, got {array.dtype}")
    if array.shape != plan.source_shape:
        raise ValueError(
            f"{name} shape {array.shape} does not match plan source {plan.source_shape}"
        )
    return array


def _coverage(coverage, plan: RasterPlan) -> np.ndarray:
    if type(coverage) is not np.ndarray:
        raise TypeError("coverage must be a plain numpy.ndarray")
    array = coverage
    if array.dtype == np.bool_:
        array = array.astype(np.float32)
    elif array.dtype != np.float32:
        raise TypeError(f"coverage must be bool or float32, got {array.dtype}")
    if array.shape != plan.source_shape:
        raise ValueError(
            f"coverage shape {array.shape} does not match plan source {plan.source_shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("coverage must be finite")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("coverage must remain within [0, 1]")
    return array


def _threshold(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("coverage_threshold must be finite and in (0, 1]")
    return result


def _encode_covered_unit_interval(
    source: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.float32, np.float32]:
    """Encode one covered physical channel for Agg's image-value ABI.

    Matplotlib's low-level image resampler is a literal implementation of the
    required Hanning kernel, but some supported ABI versions clamp floating
    image samples to ``[0, 1]``.  Feeding Kelvin, angles, or signed corrected
    reflectance through that interface directly is therefore not a physical
    resample.

    For covered values ``V`` and coverage ``C``, this function supplies the
    affine coordinate ``E = (V - lo) / (hi - lo)``.  The caller filters
    ``E * C`` and ``C`` through the same Agg operator and decodes
    ``lo + (hi - lo) * H(E*C) / H(C)``.  By linearity this is exactly
    ``H(V*C) / H(C)``; it is a coordinate change, not value clipping or a
    fitted compensation.
    """

    values = np.asarray(source, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    if values.shape != mask.shape:
        raise ValueError("physical channel and coverage mask differ")
    if not np.any(mask):
        return np.zeros_like(values), np.float32(0.0), np.float32(0.0)
    covered = values[mask]
    if not np.all(np.isfinite(covered)):
        raise ValueError("covered physical channel is non-finite")
    lower = np.float32(np.min(covered))
    upper = np.float32(np.max(covered))
    span64 = float(upper) - float(lower)
    if not math.isfinite(span64) or span64 < 0.0:
        raise ValueError("physical channel has an unencodable finite range")
    span = np.float32(span64)
    if not np.isfinite(span):
        raise ValueError("physical channel range exceeds float32")

    encoded = np.zeros_like(values)
    if span > 0.0:
        np.subtract(values, lower, out=encoded, where=mask)
        np.divide(encoded, span, out=encoded, where=mask)
        # This clip applies only to the mathematically unit-interval transport
        # coordinate. The exact physical endpoints are restored after the
        # coverage-aware Hanning operation below.
        np.clip(encoded, np.float32(0.0), np.float32(1.0), out=encoded)
    return encoded, lower, span


def resample_fields_shared_coverage(
    fields: tuple[np.ndarray, ...],
    coverage: np.ndarray,
    plan: RasterPlan,
    *,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Resample physical scalar/RGB fields through one coverage transform.

    Every field must be either ``(H, W)`` or ``(H, W, C)`` float32 on the
    plan's source grid.  Computing the shared coverage exactly once is both a
    binding guarantee (the fields cannot drift to different support) and a
    material memory reduction for the high-resolution true-colour delivery.
    """

    if type(fields) is not tuple or not fields:
        raise TypeError("fields must be a non-empty exact tuple")
    source_coverage = _coverage(coverage, plan)
    threshold = _threshold(coverage_threshold)
    valid = source_coverage > 0.0
    sources: list[np.ndarray] = []
    for index, field in enumerate(fields):
        if type(field) is not np.ndarray:
            raise TypeError(f"fields[{index}] must be a plain numpy.ndarray")
        source = field
        if source.dtype != np.float32:
            raise TypeError(
                f"fields[{index}] must be float32 physical data, got "
                f"{source.dtype}"
            )
        if source.ndim not in (2, 3) or source.shape[:2] != plan.source_shape:
            raise ValueError(
                f"fields[{index}] shape {source.shape} does not match source "
                f"grid {plan.source_shape} with optional channels"
            )
        if source.ndim == 3 and source.shape[2] <= 0:
            raise ValueError(f"fields[{index}] must contain at least one channel")
        if not np.all(np.isfinite(source[valid])):
            raise ValueError(
                f"fields[{index}] must be finite wherever coverage is positive"
            )
        sources.append(source)

    destination_coverage = _hanning_resample(source_coverage, plan)
    destination_coverage[destination_coverage <= threshold] = np.float32(0.0)
    outputs: list[np.ndarray] = []
    for source in sources:
        trailing_shape = source.shape[2:]
        destination = np.full(
            (*plan.destination_shape, *trailing_shape),
            np.nan,
            dtype=np.float32,
        )
        channel_count = source.shape[2] if source.ndim == 3 else 1
        for channel_index in range(channel_count):
            source_channel = (
                source[..., channel_index] if source.ndim == 3 else source
            )
            encoded, lower, span = _encode_covered_unit_interval(
                source_channel, valid
            )
            destination_premultiplied = _hanning_resample(
                encoded * source_coverage, plan
            )
            destination_channel = (
                destination[..., channel_index]
                if destination.ndim == 3
                else destination
            )
            np.divide(
                destination_premultiplied,
                destination_coverage,
                out=destination_channel,
                where=destination_coverage > 0.0,
            )
            np.multiply(
                destination_channel,
                span,
                out=destination_channel,
                where=destination_coverage > 0.0,
            )
            np.add(
                destination_channel,
                lower,
                out=destination_channel,
                where=destination_coverage > 0.0,
            )
        outputs.append(destination)
    return tuple(outputs), destination_coverage


def resample_field(
    field: np.ndarray,
    coverage: np.ndarray,
    plan: RasterPlan,
    *,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage-aware Hanning resample of one float32 physical scalar field.

    Both returned arrays have the plan's complete internal Agg destination
    shape.  Call ``plan.delivered_values`` to obtain the verified top-origin
    canvas slice for a one-to-one FigureImage placement.
    """

    source = _physical_field(field, plan, name="field")
    destinations, destination_coverage = resample_fields_shared_coverage(
        (source,),
        coverage,
        plan,
        coverage_threshold=coverage_threshold,
    )
    return destinations[0], destination_coverage


def resample_rgb(
    rgb: np.ndarray,
    coverage: np.ndarray,
    plan: RasterPlan,
    *,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage-aware Hanning resample of float32 physical RGB reflectance.

    The returned RGB and coverage retain the full internal Agg raster for
    provenance.  ``plan.delivered_values`` performs the exact canvas slice and
    orientation conversion without filtering.
    """

    if type(rgb) is not np.ndarray:
        raise TypeError("rgb must be an exact plain numpy.ndarray")
    source = rgb
    if source.dtype != np.float32:
        raise TypeError(f"rgb must be float32 physical data, got {source.dtype}")
    if source.shape != (*plan.source_shape, 3):
        raise ValueError(
            f"rgb shape {source.shape} does not match plan source "
            f"{(*plan.source_shape, 3)}"
        )
    destinations, destination_coverage = resample_fields_shared_coverage(
        (source,),
        coverage,
        plan,
        coverage_threshold=coverage_threshold,
    )
    return destinations[0], destination_coverage


def delivered_view(values: np.ndarray, plan: RasterPlan) -> np.ndarray:
    """Select a dynamically bound internal-to-delivered raster view."""

    return plan.delivered_values(values)


def verify_identity_draw(
    image,
    renderer: "RendererBase",
    plan: RasterPlan,
) -> IdentityDrawTrace:
    """Prove that a final FigureImage draw has no spatial interpolation.

    The image must contain ``plan.delivered_values(...)``, use upper-origin
    nearest placement, and sit on the plan's integer device bounds.  Calling
    the real ``FigureImage.make_image`` proves output shape, placement, and the
    complete affine; dimensions alone are intentionally insufficient.
    """

    if plan.delivered_shape is None or plan.delivered_device_bounds is None:
        raise RuntimeError("identity draw requires verified delivered geometry")
    source = image.get_array()
    if source is None or tuple(source.shape[:2]) != plan.delivered_shape:
        raise ValueError(
            "FigureImage source does not match the plan's delivered raster shape"
        )
    if image.get_interpolation() != "nearest":
        raise ValueError("final one-to-one FigureImage must use literal nearest")
    if image.origin != "upper":
        raise ValueError("final one-to-one FigureImage must use origin='upper'")

    made, origin_x, origin_y, transform = image.make_image(
        renderer, magnification=1.0, unsampled=False
    )
    if made is None or transform is None:
        raise RuntimeError("final FigureImage does not intersect the canvas")
    output_shape = (int(made.shape[0]), int(made.shape[1]))
    if output_shape != plan.delivered_shape:
        raise RuntimeError(
            f"FigureImage changed raster dimensions: {plan.delivered_shape} "
            f"-> {output_shape}"
        )

    height, width = output_shape
    device_bounds = (
        float(origin_x),
        float(origin_y),
        float(origin_x + width),
        float(origin_y + height),
    )
    if not np.allclose(
        device_bounds,
        plan.delivered_device_bounds,
        rtol=0.0,
        atol=_GEOMETRY_ATOL,
    ):
        raise RuntimeError(
            "FigureImage placement differs from the captured delivered bounds"
        )

    matrix = _matrix3x3(transform)
    expected: Matrix3x3 = (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, float(height)),
        (0.0, 0.0, 1.0),
    )
    if not np.allclose(matrix, expected, rtol=0.0, atol=_GEOMETRY_ATOL):
        raise RuntimeError(
            "FigureImage draw is not the exact upper-origin one-to-one affine"
        )
    return IdentityDrawTrace(
        source_shape=plan.delivered_shape,
        output_shape=output_shape,
        device_bounds=device_bounds,
        affine_matrix=matrix,
        interpolation="nearest",
    )
