"""Matplotlib + cartopy render pipeline.

Takes a FetchResult (cropped CMI on the geos grid with companion lat/lon arrays),
projects to PlateCarree, applies the requested enhancement, and produces a
clean dark-themed PNG with title strip, a labeled right-side colorbar, and a
footer credit.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from matplotlib.colors import Normalize, to_rgb
from shapely.geometry import shape

from colormaps import get_enhancement, enhancement_norm, normalize_visible
from render_policy import RenderPolicy, require_registered_policy
from satellites import FetchResult

log = logging.getLogger("tat-satellite.render")

# admin_1 state/province boundary LINES, vendored as NE 10m geojson (the same
# asset the CycloLab basemap ships). Loaded locally instead of via cartopy's
# runtime downloader, whose admin_1 URL 404s on the deploy host (coastline +
# admin_0 borders download fine there; admin_1 does not).
# .resolve() so a relative __file__ (uvicorn import from CWD) still yields the
# absolute path next to render.py — mirrors cyclolab_basemap's proven pattern.
_STATE_LINES_PATH = Path(__file__).resolve().with_name("cyclolab_ne_10m_states.geojson")


@lru_cache(maxsize=1)
def _state_lines_feature() -> Optional[ShapelyFeature]:
    """admin_1 boundary LINES as a cartopy feature, read once from the vendored
    geojson (no network). Returns None if the asset is missing/empty."""
    if not _STATE_LINES_PATH.exists():
        return None
    gj = json.loads(_STATE_LINES_PATH.read_text(encoding="utf-8"))
    geoms = [shape(f["geometry"]) for f in gj.get("features", []) if f.get("geometry")]
    if not geoms:
        return None
    return ShapelyFeature(geoms, ccrs.PlateCarree())


def state_lines_status() -> str:
    """Diagnostic: did the vendored admin_1 layer load on this host? Surfaced
    via the X-State-Lines response header to verify the deploy at runtime."""
    p = _STATE_LINES_PATH
    try:
        return "loaded" if _state_lines_feature() is not None else f"absent(exists={p.exists()},p={p})"
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}:{e}"[:160]

DARK_BG = "#0a0d12"
GRID_COLOR = "#3a4252"
# Coast/border style (2026-07-12, Andrew's call — OVERRIDES the 2026-07-11
# cyan/white restyle): coastlines, borders, state lines and the halo are all
# BLACK. The halo under-stroke geometry is kept (it just reads as a slightly
# heavier black line), so a future restyle is a constants-only change again.
COAST_COLOR = "#000000"   # black — landmass outlines
BORDER_COLOR = "#000000"  # black — political borders
LINE_HALO = "#000000"     # black halo under both (thin: +~1.2 px)
TEXT_COLOR = "#e8eef5"
ACCENT_COLOR = "#79f0d6"
MUTED_COLOR = "#9199a4"


def _imshow_with_render_policy(
    ax,
    image,
    *,
    policy: RenderPolicy,
    **kwargs,
):
    """Draw a regular raster with the policy's explicit Matplotlib kernel.

    Keep both literals in this dispatcher.  Matplotlib's ``"auto"`` and
    ``"antialiased"`` modes can silently choose nearest-neighbour sampling at
    the resize ratios used by TAT, so the production call must never delegate
    this decision back to Matplotlib.  M0 remains available as an exact
    nearest-neighbour rollback; M1 is the explicit Hanning stopgap only.
    """

    require_registered_policy(policy)
    if policy.resample_kernel == "nearest":
        return ax.imshow(image, interpolation="nearest", **kwargs)
    if policy.resample_kernel == "hanning":
        return ax.imshow(image, interpolation="hanning", **kwargs)
    raise RuntimeError(
        f"render policy {policy.algo_version!r} cannot use "
        f"{policy.resample_kernel!r} through Matplotlib imshow"
    )


def _axes_device_pixel_bounds(fig, ax) -> tuple[int, int, int, int]:
    """Return the exact half-open canvas rows/columns covered by ``ax``.

    Agg samples at device-pixel centres.  Deriving the integer bounds from the
    *resolved* axes bbox (after Cartopy aspect/letterboxing) keeps fractional
    layout phase dynamic across Matplotlib versions and avoids the historical
    1189-versus-1188 hard-code.  Rows are measured from the top of the canvas;
    columns from the left.
    """

    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer)
    figure_height = int(round(float(fig.bbox.height)))
    figure_width = int(round(float(fig.bbox.width)))
    left = int(math.ceil(float(bbox.x0) - 0.5))
    right = int(math.ceil(float(bbox.x1) - 0.5))
    top = int(math.ceil(figure_height - float(bbox.y1) - 0.5))
    bottom = int(math.ceil(figure_height - float(bbox.y0) - 0.5))
    if not (0 <= top < bottom <= figure_height):
        raise RuntimeError(
            f"resolved axes rows {(top, bottom)} exceed canvas height "
            f"{figure_height}"
        )
    if not (0 <= left < right <= figure_width):
        raise RuntimeError(
            f"resolved axes columns {(left, right)} exceed canvas width "
            f"{figure_width}"
        )
    return top, bottom, left, right


def _place_device_raster(fig, ax, raster: np.ndarray, bounds) -> object:
    """Place an already-delivered RGB(A) raster onto the canvas exactly 1:1.

    ``Figure.figimage`` is intentionally used instead of ``Axes.imshow``:
    figimage performs no spatial resampling.  The GeoAxes patch becomes
    transparent while coastlines, borders and graticules remain above the
    figure image.  The figure's dark facecolor still supplies the same outer
    background and any fractional half-pixel edge.
    """

    top, bottom, left, right = (int(value) for value in bounds)
    image = np.asarray(raster)
    expected = (bottom - top, right - left)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"device raster must be HxWxRGB(A), got {image.shape}")
    if tuple(image.shape[:2]) != expected:
        raise ValueError(
            f"device raster shape {image.shape[:2]} does not match bounds "
            f"{expected}"
        )
    if not np.isfinite(image).all():
        raise ValueError("device raster must be finite before opaque delivery")
    figure_height = int(round(float(fig.bbox.height)))
    ax.patch.set_alpha(0.0)
    return fig.figimage(
        image,
        xo=left,
        yo=figure_height - bottom,
        origin="upper",
        interpolation="nearest",
        resize=False,
        zorder=-10,
    )


def _validate_m2_truecolor_source_geometry(data: FetchResult, bbox) -> None:
    """Bind a typed M2 payload to the exact regular request grid.

    Shape agreement alone is not geolocation.  The primary M2 authority is a
    north-to-south regular Plate Carrée grid, so every coordinate endpoint,
    metadata tag, and FetchResult alias is checked before raster planning.
    """

    import truecolor

    state = data.truecolor_physical
    if type(state) is not truecolor.TrueColorPhysicalState:
        raise RuntimeError("M2 true-color state has an unregistered type")
    expected_extent = tuple(float(value) for value in bbox)
    if state.extent is None or not np.allclose(
        np.asarray(state.extent, dtype=np.float64),
        np.asarray(expected_extent, dtype=np.float64),
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise RuntimeError("M2 physical-state extent is not bound to the request")
    if type(state.projection) is not str or state.projection != "PlateCarree":
        raise RuntimeError("M2 physical-state projection is not registered PlateCarree")
    metadata = state.grid_metadata
    if type(metadata) is not dict:
        raise RuntimeError("M2 physical-state grid metadata must be an exact dict")
    if state.sensor not in ("abi", "ahi"):
        raise RuntimeError("M2 primary renderer authority is limited to ABI/AHI")
    required_metadata = {
        "grid": "regular_latlon",
        "row_order": "north_to_south",
        "sensor": state.sensor,
        "point_stride_applied": False,
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise RuntimeError("M2 physical-state grid metadata is unregistered or inconsistent")

    shape = state.day_coverage.shape
    # ``np.asarray`` would silently strip a MaskedArray's mask and admit a
    # lookalike coordinate payload.  M2 binds the exact plain arrays supplied
    # by the fetcher before any coercion can erase type semantics.
    if type(data.lats) is not np.ndarray or type(data.lons) is not np.ndarray:
        raise RuntimeError("M2 FetchResult coordinates must be exact plain ndarrays")
    lats = data.lats
    lons = data.lons
    if (
        lats.dtype != np.float32
        or lons.dtype != np.float32
        or lats.shape != shape
        or lons.shape != shape
        or not np.all(np.isfinite(lats))
        or not np.all(np.isfinite(lons))
    ):
        raise RuntimeError("M2 FetchResult coordinates do not match the physical grid")
    if data.cos_sza is not state.cos_sza or data.geom_valid is not state.geom_valid:
        raise RuntimeError("M2 FetchResult geometry aliases are not bound to the typed state")
    expected_source_day_weight = truecolor.terminator_day_weight(
        state.cos_sza
    ).astype(np.float32, copy=False)
    if not np.array_equal(
        state.day_weight[state.geom_valid],
        expected_source_day_weight[state.geom_valid],
    ):
        raise RuntimeError(
            "M2 source day weight is not bound to the 88-to-95-degree taper"
        )
    if not np.array_equal(lats, np.broadcast_to(lats[:, :1], shape)):
        raise RuntimeError("M2 latitude grid is not separable regular latitude")
    if not np.array_equal(lons, np.broadcast_to(lons[:1, :], shape)):
        raise RuntimeError("M2 longitude grid is not separable regular longitude")

    lat_axis = lats[:, 0].astype(np.float64, copy=False)
    lon_axis = lons[0, :].astype(np.float64, copy=False)
    if lat_axis.size < 2 or lon_axis.size < 2:
        raise RuntimeError("M2 regular grid must have at least two pixels per axis")
    if not np.all(np.diff(lat_axis) < 0.0):
        raise RuntimeError("M2 latitude rows are not strictly north-to-south")
    if not np.all(np.diff(lon_axis) > 0.0):
        raise RuntimeError("M2 longitude columns are not strictly west-to-east")
    lon_min, lat_min, lon_max, lat_max = expected_extent
    expected_lon_axis = np.linspace(
        lon_min, lon_max, lon_axis.size
    ).astype(np.float32)
    expected_lat_axis = np.linspace(
        lat_max, lat_min, lat_axis.size
    ).astype(np.float32)
    if not np.array_equal(lons[0], expected_lon_axis) or not np.array_equal(
        lats[:, 0], expected_lat_axis
    ):
        raise RuntimeError("M2 regular-grid pixel centres are not request-bound linspace")
    endpoints = np.array(
        (lon_axis[0], lon_axis[-1], lat_axis[0], lat_axis[-1]),
        dtype=np.float64,
    )
    expected_endpoints = np.array(
        (lon_min, lon_max, lat_max, lat_min), dtype=np.float64
    )
    if not np.allclose(
        endpoints, expected_endpoints, rtol=0.0, atol=2.0e-5
    ):
        raise RuntimeError("M2 regular-grid endpoints do not match the request extent")


def _m2_truecolor_cell_edge_extent(data: FetchResult) -> list[float]:
    """Return image edges for already-validated regular centre coordinates."""

    lons = data.lons
    lats = data.lats
    if type(lons) is not np.ndarray or type(lats) is not np.ndarray:
        raise RuntimeError("M2 true-color edge extent requires plain coordinate arrays")
    lon_axis = lons[0, :].astype(np.float64, copy=False)
    lat_axis = lats[:, 0].astype(np.float64, copy=False)
    if lon_axis.size < 2 or lat_axis.size < 2:
        raise RuntimeError("M2 true-color edge extent requires two pixels per axis")
    dlon = (lon_axis[-1] - lon_axis[0]) / (lon_axis.size - 1)
    dlat = (lat_axis[-1] - lat_axis[0]) / (lat_axis.size - 1)
    if not (np.isfinite(dlon) and dlon > 0.0 and np.isfinite(dlat) and dlat < 0.0):
        raise RuntimeError("M2 true-color centre axes have unsupported orientation")
    return [
        float(lon_axis[0] - dlon / 2.0),
        float(lon_axis[-1] + dlon / 2.0),
        float(lat_axis[-1] + dlat / 2.0),
        float(lat_axis[0] - dlat / 2.0),
    ]


def _resample_truecolor_physical_state(state, plan):
    """Deliver every M2 field/coverage through one exact RasterPlan.

    The same Hanning geometry is replayed for every numerator and coverage.
    Auxiliary geometry is never inferred from encoded RGB, and no nearest
    mask resize or missing-to-black fill exists in this boundary.
    """

    import render_resample
    import truecolor

    if type(state) is not truecolor.TrueColorPhysicalState:
        raise TypeError("M2 renderer requires an exact TrueColorPhysicalState")

    day_sources = (
        state.day_rgb_reflectance,
        state.red_corrected,
        state.glint_deg,
        *((state.veggie_corrected,) if state.veggie_corrected is not None else ()),
    )
    day_field_names = [
        "day_rgb_reflectance",
        "red_corrected",
        "glint_deg",
        *(["veggie_corrected"] if state.veggie_corrected is not None else []),
    ]
    day_fields, day_coverage = render_resample.resample_fields_shared_coverage(
        day_sources, state.day_coverage, plan
    )
    day_rgb, red_corrected, glint_deg, *optional_day_fields = day_fields
    veggie_corrected = (
        optional_day_fields[0] if optional_day_fields else None
    )
    if state.ir_bt_kelvin is None:
        ir_bt = None
        ir_coverage = None
        source_union = state.day_coverage
    else:
        ir_bt, ir_coverage = render_resample.resample_field(
            state.ir_bt_kelvin, state.ir_coverage, plan
        )
        source_union = np.maximum(state.day_coverage, state.ir_coverage).astype(
            np.float32, copy=False
        )

    (cos_sza, day_weight), union_coverage = (
        render_resample.resample_fields_shared_coverage(
            (state.cos_sza, state.day_weight), source_union, plan
        )
    )
    day_weight = np.where(
        union_coverage > 0.0,
        np.clip(day_weight, 0.0, 1.0),
        np.float32(0.0),
    ).astype(np.float32, copy=False)
    # A constant value carried by geom_valid yields the exact filtered
    # geometric coverage without inventing a nearest-resized boolean mask.
    _, geom_coverage = render_resample.resample_field(
        np.ones(plan.source_shape, dtype=np.float32), state.geom_valid, plan
    )
    # ``resample_field`` has already canonicalised rejected Hanning tails to
    # exact zero using the same threshold as every modality.  Geometry is
    # therefore derived from the effective support, not from a second and
    # potentially divergent comparison.
    geom_valid = geom_coverage > 0.0

    # The day RGB and every day-only auxiliary were delivered in one shared
    # coverage call above; a future field cannot silently select another mask,
    # kernel, phase, or threshold.
    if state.ir_bt_kelvin is not None and ir_coverage is None:
        raise RuntimeError("M2 IR coverage was lost during delivery resampling")
    if np.any(day_coverage[~geom_valid] != 0.0):
        raise RuntimeError("M2 day coverage escaped delivered geometric support")
    if ir_coverage is not None and np.any(ir_coverage[~geom_valid] != 0.0):
        raise RuntimeError("M2 IR coverage escaped delivered geometric support")
    if np.any(union_coverage[~geom_valid] != 0.0):
        raise RuntimeError("M2 modality coverage escaped delivered geometry")

    metadata = {
        "source_geometry_binding": "request_regular_latlon_exact_float32_linspace_v1",
        "primary_sensor": state.sensor,
        "source_projection": state.projection,
        "source_declared_extent": state.extent,
        "source_grid_metadata": state.grid_metadata,
        "resample_kernel": plan.kernel,
        "resample_domain": plan.domain,
        "physical_domain_transport": render_resample.PHYSICAL_DOMAIN_TRANSPORT,
        "source_shape": plan.source_shape,
        "internal_destination_shape": plan.destination_shape,
        "delivered_shape": plan.delivered_shape,
        "source_extent": plan.source_extent,
        "view_extent": plan.view_extent,
        "device_bounds": plan.device_bounds,
        "delivered_device_bounds": plan.delivered_device_bounds,
        "delivered_internal_bounds": plan.delivered_internal_bounds,
        "internal_to_canvas_flip_y": plan.internal_to_canvas_flip_y,
        "internal_to_canvas_flip_x": plan.internal_to_canvas_flip_x,
        "affine_matrix": plan.affine_matrix,
        "sampling_phase_yx": plan.sampling_phase_yx,
        "coverage_threshold": render_resample.DEFAULT_COVERAGE_THRESHOLD,
        "shared_day_fields": day_field_names,
        "day_coverage_binding": "one_premultiplied_hanning_replay_shared_by_all_day_fields",
        "terminator_weight_binding": (
            "source_88_to_95_day_weight_resampled_with_modality_union_v1"
        ),
        "terminator_weight_policy": truecolor.M2_DAY_WEIGHT_POLICY_ID,
    }
    return dataclasses.replace(
        state,
        day_rgb_reflectance=day_rgb,
        day_coverage=day_coverage,
        ir_bt_kelvin=ir_bt,
        ir_coverage=ir_coverage,
        cos_sza=cos_sza,
        day_weight=day_weight,
        geom_valid=geom_valid,
        red_corrected=red_corrected,
        veggie_corrected=veggie_corrected,
        glint_deg=glint_deg,
        grid_metadata=metadata,
    )


def _m2_plan_axes_raster(
    fig,
    ax,
    source_shape: tuple[int, int],
    *,
    origin: str,
    extent,
    transform,
    policy: RenderPolicy,
):
    """Capture a real Agg plan with a disposable, value-neutral image."""

    import render_resample

    require_registered_policy(policy)
    if not policy.resample_in_reflectance:
        raise RuntimeError("M2 raster planning requires reflectance resampling")
    if policy.resample_kernel != "hanning":
        raise RuntimeError("M2 raster planning requires literal Hanning")
    # A deterministic opaque tag pattern exercises the live image backend and
    # canvas during planning. Scientific geometry is derived separately from
    # the resolved image-to-delivered-device transform below, so a private Agg
    # intermediate row cannot move source support across library versions.
    probe = np.empty((*source_shape, 3), dtype=np.uint8)
    # One-dimensional modular terms produce the same collision-resistant tag
    # as full ``np.indices`` grids while avoiding two source-sized uint32
    # allocations.  This matters at the bounded high-resolution M2 envelope.
    x_mod = (np.arange(source_shape[1], dtype=np.uint32) & 0xFF).astype(
        np.uint8, copy=False
    )
    y_mod = (np.arange(source_shape[0], dtype=np.uint32) & 0xFF).astype(
        np.uint8, copy=False
    )
    for channel, (x_factor, y_factor, offset) in enumerate(
        ((17, 31, 23), (47, 13, 71), (29, 61, 131))
    ):
        x_term = np.multiply(x_mod, np.uint8(x_factor), dtype=np.uint8)
        y_term = np.multiply(y_mod, np.uint8(y_factor), dtype=np.uint8)
        np.add(
            y_term[:, None],
            x_term[None, :],
            out=probe[..., channel],
        )
        np.add(
            probe[..., channel],
            np.uint8(offset),
            out=probe[..., channel],
        )
    image = render_resample.make_hanning_planning_image(
        ax,
        probe,
        origin=origin,
        extent=extent,
        transform=transform,
        zorder=1,
    )
    spine_visibility = {
        name: spine.get_visible() for name, spine in ax.spines.items()
    }
    for spine in ax.spines.values():
        spine.set_visible(False)
    try:
        # Cartopy's equal-aspect adjustment and fractional device bbox do not
        # become authoritative until a real renderer has completed layout.
        fig.canvas.draw()
        top, bottom, left, right = _axes_device_pixel_bounds(fig, ax)
        figure_height = int(round(float(fig.bbox.height)))
        delivered_device_bounds = (
            float(left),
            float(figure_height - bottom),
            float(right),
            float(figure_height - top),
        )
        plan = render_resample.RasterPlan.from_axes_image_delivered(
            image,
            fig.canvas.get_renderer(),
            delivered_device_bounds=delivered_device_bounds,
        )
    finally:
        image.remove()
        for name, visible in spine_visibility.items():
            ax.spines[name].set_visible(visible)
    expected = (bottom - top, right - left)
    if plan.delivered_shape != expected:
        raise RuntimeError(
            f"captured delivered shape {plan.delivered_shape} differs from "
            f"resolved device bounds {expected}"
        )
    return plan, (top, bottom, left, right)


def _m2_place_verified(fig, ax, raster, plan, bounds):
    """Place and independently prove a no-resample final imagery draw."""

    import render_resample

    artist = _place_device_raster(fig, ax, raster, bounds)
    render_resample.verify_identity_draw(
        artist,
        fig.canvas.get_renderer(),
        plan,
    )
    return artist


_M2_ARCHIVE_SCALAR_CHANNELS = {
    "GRIDSAT-B1": {"clean_ir": 1, "wv_upper": 2},
    "GridSat-GOES": {
        "visible_red": 1,
        "shortwave_ir": 2,
        "wv_upper": 3,
        "clean_ir": 4,
    },
}
_M2_VISIBLE_SOURCE_UNITS = frozenset({"1"})
_M2_KELVIN_SOURCE_UNITS = frozenset({"K"})
_M2_CELSIUS_SOURCE_UNITS = frozenset({"C", "celsius", "degC"})


def _validate_m2_scalar_source(data: FetchResult, requested_channel: int) -> None:
    """Fail closed at the bounded archive scalar/calibration boundary."""

    if type(data.cmi) is not np.ndarray or data.cmi.dtype != np.dtype(np.float32):
        raise RuntimeError("M2 scalar calibrated field must be an exact float32 ndarray")
    if data.cmi.ndim != 2:
        raise RuntimeError("M2 scalar calibrated field must be two-dimensional")
    if type(data.lats) is not np.ndarray or type(data.lons) is not np.ndarray:
        raise RuntimeError("M2 scalar coordinates must be exact plain ndarrays")
    if data.lats.shape != data.cmi.shape or data.lons.shape != data.cmi.shape:
        raise RuntimeError("M2 scalar field and coordinate shapes differ")
    if type(data.product) is not str or type(data.generic_channel) is not str:
        raise RuntimeError("M2 scalar archive identity must use exact string tags")
    supported = _M2_ARCHIVE_SCALAR_CHANNELS.get(data.product)
    if supported is None:
        raise RuntimeError(
            f"M2 scalar source family {data.product!r} is outside archive authority"
        )
    expected_channel = supported.get(data.generic_channel)
    if expected_channel is None:
        raise RuntimeError(
            f"M2 scalar channel {data.generic_channel!r} is unsupported for "
            f"{data.product}"
        )
    if data.channel != expected_channel or requested_channel != expected_channel:
        raise RuntimeError(
            "M2 scalar native channel is not bound to its archive family"
        )
    if type(data.units) is not str:
        raise RuntimeError("M2 scalar source units must be an explicit string")
    if data.generic_channel == "visible_red":
        if data.units not in _M2_VISIBLE_SOURCE_UNITS:
            raise RuntimeError("M2 scalar visible_red requires source units '1'")
    elif data.units not in (
        _M2_KELVIN_SOURCE_UNITS | _M2_CELSIUS_SOURCE_UNITS
    ):
        raise RuntimeError(
            "M2 scalar thermal/WV requires source units K, C, celsius, or degC"
        )


def _m2_scalar_rgba(
    physical_field: np.ndarray,
    coverage: np.ndarray,
    plan,
    *,
    is_visible: bool,
    source_units: str,
    cmap,
    norm,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a regular scalar field before unit mapping and its palette."""

    import render_resample

    if (
        type(physical_field) is not np.ndarray
        or physical_field.dtype != np.dtype(np.float32)
        or physical_field.ndim != 2
    ):
        raise RuntimeError("M2 scalar finalizer requires an exact float32 field")
    if is_visible:
        if source_units not in _M2_VISIBLE_SOURCE_UNITS:
            raise RuntimeError("M2 visible scalar finalizer requires source units '1'")
    elif source_units not in (
        _M2_KELVIN_SOURCE_UNITS | _M2_CELSIUS_SOURCE_UNITS
    ):
        raise RuntimeError("M2 thermal scalar finalizer received unsupported units")
    field = physical_field
    # Temperature offsets are linear, but normalize to the declared kelvin
    # domain before the spatial operation so the trace never calls Celsius a
    # different physical quantity.
    if not is_visible and source_units in _M2_CELSIUS_SOURCE_UNITS:
        field = (field + np.float32(273.15)).astype(np.float32, copy=False)
    delivered_field, delivered_coverage = render_resample.resample_field(
        field,
        coverage,
        plan,
    )
    delivered_field = plan.delivered_values(delivered_field)
    delivered_coverage = plan.delivered_values(delivered_coverage)
    if not np.all(np.isfinite(delivered_coverage)) or np.any(
        (delivered_coverage < 0.0) | (delivered_coverage > 1.0)
    ):
        raise RuntimeError("M2 delivered scalar coverage escaped [0, 1]")

    if is_visible:
        mapped_field = normalize_visible(delivered_field)
    else:
        mapped_field = delivered_field - np.float32(273.15)
    finite = np.isfinite(mapped_field)
    fill = float(norm.vmin if norm.vmin is not None else 0.0)
    safe = np.where(finite, mapped_field, fill)
    rgba = np.asarray(cmap(norm(safe)), dtype=np.float32)
    alpha = np.where(finite, delivered_coverage, 0.0).astype(np.float32)
    background = np.asarray(to_rgb(DARK_BG), dtype=np.float32)
    opaque_rgb = (
        rgba[..., :3] * alpha[..., None]
        + background[None, None, :] * (1.0 - alpha[..., None])
    ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(opaque_rgb)):
        raise RuntimeError("M2 scalar palette produced a non-finite device raster")
    return opaque_rgb, mapped_field


def _fill_coord_nan(a: np.ndarray) -> np.ndarray:
    """Nearest-finite fill for inverse-projection coordinate arrays.

    A meso sector steered onto the disk limb (e.g. GOES-18 M2 over the
    Bering Sea) has bbox corners past the geometric horizon — those pixels
    have no lat/lon. pcolormesh requires finite coords everywhere, so fill
    NaNs with the nearest finite value along each row (then column). The
    affected cells are masked in the data, and repeating a coordinate makes
    the phantom quads zero-area, so nothing fictional is drawn."""
    out = a.astype(np.float64, copy=True)
    for _ in range(2):          # pass 1: along rows; pass 2: along columns
        cols = np.arange(out.shape[1])
        rows = np.arange(out.shape[0])[:, None]
        # Forward fill: index of the last finite column so far (a leading
        # NaN run pulls col 0's value, which is NaN — the mirrored pass
        # below catches it).
        idx = np.where(np.isfinite(out), cols, 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = out[rows, idx]
        # Backward fill = forward fill on the mirror.
        rev = out[:, ::-1]
        idx = np.where(np.isfinite(rev), cols, 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = rev[rows, idx][:, ::-1]
        out = out.T
    return out


def _effective_extent(
    lats: np.ndarray,
    lons: np.ndarray,
    valid: "np.ndarray | None",
    bbox: list,
    lon_span_req: float,
) -> tuple[float, float, float, float]:
    """(lon_lo, lon_hi, lat_lo, lat_hi) of valid-data ∩ bbox, with longitudes
    UNWRAPPED from the bbox's west edge (monotonic even across ±180).

    A disk-limb meso sector (e.g. GOES-18 M2 over the Bering) carries a big
    off-disk corner in its lat/lon bounding box; framing the full request
    renders mostly-empty black margins. Cropping to the valid bounding box
    keeps the frame tight around real imagery while a genuine limb edge
    inside it still shows. Sectors whose data fills the request (every
    normal sector + storm floater) clip to exactly the request — unchanged
    by construction."""
    lon_min, lat_min, lon_max, lat_max = bbox
    fallback = (lon_min, lon_min + lon_span_req, lat_min, lat_max)
    if valid is None or not valid.any():
        return fallback
    vlat = lats[valid]
    vlon_uw = lon_min + ((lons[valid] - lon_min) % 360.0)
    lo = max(lon_min, float(vlon_uw.min()))
    hi = min(lon_min + lon_span_req, float(vlon_uw.max()))
    la = max(lat_min, float(vlat.min()))
    lb = min(lat_max, float(vlat.max()))
    if hi - lo < 0.5 or lb - la < 0.5:   # degenerate crop -> keep request
        return fallback
    return lo, hi, la, lb


def _gridline_xlocs(lon_min: float, lon_max_uw: float, step: float) -> np.ndarray:
    """Meridian locations for the graticule. ``lon_max_uw`` may exceed 180 (the
    unwrapped east edge of an antimeridian-crossing bbox) — locs are generated
    in unwrapped space so the sequence is continuous across the dateline, then
    wrapped into [-180, 180] for cartopy's Gridliner (which speaks wrapped
    PlateCarree longitudes). Non-crossing boxes reproduce the legacy arange."""
    locs = np.arange(np.floor(lon_min / step) * step, lon_max_uw + step, step)
    if lon_max_uw > 180.0:
        locs = np.unique(((locs + 180.0) % 360.0) - 180.0)
    return locs


def _gridline_step(span: float) -> float:
    """Pick a sane gridline interval (degrees) for the given bbox span."""
    if span <= 2:
        return 0.5
    if span <= 5:
        return 1.0
    if span <= 12:
        return 2.0
    if span <= 25:
        return 5.0
    return 10.0


def _map_geometry(bbox: list[float]):
    """Crossing-aware map geometry for a ``[W, S, E, N]`` bbox.

    Returns ``(projection, extent, lon_span, crossing)``:
      * ``projection`` — the AXES projection. PlateCarree(central_longitude=180)
        when the bbox crosses the antimeridian (normalized convention:
        lon_max < lon_min), so the extent is continuous in axes space; the
        plain PlateCarree otherwise (existing behavior, byte-identical).
      * ``extent`` — [lon_min, lon_max_unwrapped, lat_min, lat_max] to pass to
        ``set_extent(..., crs=ccrs.PlateCarree())``; cartopy transforms the
        unwrapped east edge into the crossing projection correctly.
      * ``lon_span`` — positive unwrapped span (drives aspect / coast scale /
        gridline step).
    Data plotting stays ``transform=ccrs.PlateCarree()`` in both cases: with a
    180-centered axes projection the transformed mesh is continuous across the
    dateline (the ±180 jump in wrapped source longitudes lands at the axes
    center, not at a seam), so pcolormesh needs no special handling.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    crossing = lon_max < lon_min
    lon_max_uw = lon_max + 360.0 if crossing else lon_max
    proj = ccrs.PlateCarree(central_longitude=180.0) if crossing else ccrs.PlateCarree()
    extent = [lon_min, lon_max_uw, lat_min, lat_max]
    return proj, extent, lon_max_uw - lon_min, crossing


def _unwrap_lons(lons, lon_min: float):
    """Make source longitudes CONTINUOUS across the dateline for a crossing
    bbox. Wrapped [-180, 180] lons (the AHI inverse projection wraps; GOES-West
    emits sub-sat-relative values) carry a ±360 jump at the dateline column —
    cartopy's pcolormesh wrap-detection then mangles the jump-spanning cells
    into misplaced horizontal smears (observed on the first live swpac
    backdrop), and _guard_mesh_coords' nanmean fill lands far outside the
    window. Everything west of the bbox's west edge is really the >180 side:
    shift it up by 360. PlateCarree transforms are periodic, so unwrapped
    values >180 plot exactly right."""
    lons = np.asarray(np.ma.filled(lons, np.nan), dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(lons < lon_min - 1e-9, lons + 360.0, lons)


def _coast_resolution(span_deg: float) -> str:
    """Natural Earth scale matched to bbox size.

    10m up to 90°: every meso sector (incl. the ~70°-wide GOES-18 M2 limb
    box, the view that exposed the blockiness) and every storm floater gets
    crisp coastlines — 50m at these zooms reads visibly blocky, especially
    at high latitude. The old jagged path-clipping concern applies to
    genuinely wide (near-disk) views, which still step down. The 10m
    geometry caches in cartopy after the first draw (~+6 s once, then ~0).
    """
    if span_deg < 90:
        return "10m"
    if span_deg < 180:
        return "50m"
    return "110m"


# Saffir-Simpson + tropical-status colors for the title-strip storm badge.
# Per Andrew's spec (2026-05-28): TD neon blue, TS lime, C1 yellow, C2 amber,
# C3 red, C4 pink, C5 magenta/purple. Tuned to read clearly against DARK_BG
# (each is used as a tinted/translucent bbox face for the text, not a fill).
_SS_COLORS: dict[str, str] = {
    "TD": "#3b82f6",   # tropical depression — neon blue
    "TS": "#84cc16",   # tropical storm — lime green
    "C1": "#fde047",   # cat 1 — yellow
    "C2": "#f59e0b",   # cat 2 — amber
    "C3": "#dc2626",   # cat 3 — red
    "C4": "#ec4899",   # cat 4 — pink
    "C5": "#a855f7",   # cat 5 — magenta / purple
    "EX": "#9199a4",   # extratropical / post-tropical / remnant low — gray
}


def _ss_category(nature: Optional[str], wind_kt: Optional[float]) -> str:
    """Map (nature, wind) -> short Saffir-Simpson-ish category label."""
    n = (nature or "").upper()
    # Non-tropical natures keep their own label (and the gray color).
    if n in ("EX", "PT", "DB", "WV", "LO", "SD", "SS"):
        return n or "EX"
    w = wind_kt or 0.0
    if w < 34:
        return "TD"
    if w < 64:
        return "TS"
    if w < 83:
        return "C1"
    if w < 96:
        return "C2"
    if w < 113:
        return "C3"
    if w < 137:
        return "C4"
    return "C5"


# Which generic channels carry a BRIGHTNESS TEMPERATURE (so a min/max readout
# in degrees is meaningful). Visible / true-colour pixels are reflectance and
# get no readout rather than a number in the wrong units.
_BT_BAND_TAGS: dict[str, str] = {
    "clean_ir": "IR", "ir_window": "IR",
    "wv_upper": "WV", "wv_lower": "WV",
    "shortwave_ir": "SWIR",
}


def _bt_band_tag(data: FetchResult, channel: int) -> Optional[str]:
    """Short band tag for the brightness-temperature readout, or None when the
    product's pixels are not a temperature.

    Deliberately coarse. The title already names the satellite, channel and
    wavelength; this only has to say WHICH PHYSICAL QUANTITY the min/max
    describe, because an IR-window extreme and a water-vapour-channel extreme
    are read completely differently -- an unlabelled "min/max" invites reading
    a WV frame's -60 C as a cloud-top temperature."""
    bucket = data.bucket or ""
    if bucket == "gesdisc-mergir":
        return "IR"
    if bucket.startswith("noaa-cdr-gridsat"):
        return "IR" if channel == 1 else "WV"
    if bucket == "ncei-gridsat-goes":
        return {2: "SWIR", 3: "WV", 4: "IR"}.get(channel)
    return _BT_BAND_TAGS.get(getattr(data, "generic_channel", "") or "")


def render_png(
    data: FetchResult,
    bbox: list[float],
    channel: int,
    time_str: str,
    enhancement: str,
    downsample: int = 1,
    storm: Optional[dict] = None,
    coastlines: bool = True,
    gridlines: bool = True,
    dpi: int = 110,
    *,
    policy: RenderPolicy,
) -> bytes:
    require_registered_policy(policy)
    # ``coastlines`` draws coastlines + political borders; ``gridlines`` draws the
    # labeled lat/lon graticule. Both default True (the standard look); the custom-
    # zoom page can switch either off for clean imagery. (The floater/meso loop
    # frames never pass them -> always on, unchanged.)
    # M0/M1 carry display RGB. M2 carries an explicitly tagged physical state;
    # accepting one through the other's boundary would make the policy/cache
    # identity lie about the numerical domain used for the delivered pixels.
    is_m2_truecolor = data.units == "truecolor_physical"
    is_rgb = data.units == "rgb" or is_m2_truecolor
    if is_m2_truecolor:
        if not policy.resample_in_reflectance:
            raise RuntimeError("physical true color requires the registered M2 policy")
        if data.truecolor_physical is None:
            raise RuntimeError("physical true color is missing its typed state")
        if data.cmi is not data.truecolor_physical.day_rgb_reflectance:
            raise RuntimeError("physical FetchResult cmi is not bound to its typed state")
    elif data.generic_channel == "true_color" and policy.resample_in_reflectance:
        raise RuntimeError("registered M2 policy rejects legacy encoded true-color RGB")
    is_visible = data.units == "1"
    enh = None if is_rgb else get_enhancement(enhancement)

    lon_min, lat_min, lon_max, lat_max = bbox
    m2_truecolor_edge_extent = None
    if is_m2_truecolor:
        _validate_m2_truecolor_source_geometry(data, bbox)
        m2_truecolor_edge_extent = _m2_truecolor_cell_edge_extent(data)
    # The bounded M2 authority covers typed true colour and non-crossing
    # regular scalar grids. Those paths retain the complete post-fetch
    # physical grid here for the final Agg delivery reduction below. Typed
    # true colour may already carry an explicitly traced upstream physical
    # Hanning source cap; this renderer does not claim one total spatial
    # operation. Curvilinear/crossing scalar paths remain the explicitly
    # scoped legacy fallback until M5.
    crosses = lon_max < lon_min
    m2_scalar_axes = None
    if policy.resample_in_reflectance and not is_rgb:
        _validate_m2_scalar_source(data, channel)
        if not crosses:
            m2_scalar_axes = _m2_regular_grid_axes(data.lons, data.lats)
        if m2_scalar_axes is None:
            raise RuntimeError(
                "registered M2 policy rejects curvilinear, warped, or "
                "antimeridian-crossing scalar grids; caller must select the "
                "registered legacy policy"
            )
    m2_regular_scalar = m2_scalar_axes is not None
    preserve_m2_physical_grid = is_m2_truecolor or m2_regular_scalar

    # Pixel-budget stride. App-layer (compute_downsample_factor) sets this
    # based on raw bbox×channel so output_pixels ≤ the tier budget. For SCALAR
    # products the stride happens HERE; for TRUE COLOR it is applied in the fetch
    # (satellites._stride_tc_grids, before the per-pixel recipe) and this is
    # called with downsample=1, so the RGB is never double-strided.
    # ``cmi[::d, ::d]`` strides the first two axes for both 2D and 3D arrays.
    cmi = data.cmi
    lats = data.lats
    lons = data.lons
    if downsample > 1 and not preserve_m2_physical_grid:
        log.info(
            "downsampled bbox by factor %d to stay within pixel budget "
            "(in shape %s -> out shape %s)",
            downsample,
            cmi.shape,
            cmi[::downsample, ::downsample].shape,
        )
        cmi = cmi[::downsample, ::downsample]
        lats = lats[::downsample, ::downsample]
        lons = lons[::downsample, ::downsample]

    # ---- Build the scalar plot field (skipped for true-color RGB) ----------
    # IR/WV  -> plot real brightness temperature in °C with the enhancement's
    #           cmap + a fresh Normalize over its °C domain. The colorbar ticks
    #           are then physical °C with no remap.
    # visible -> sqrt-stretched reflectance (0..1) in grayscale; colorbar shows
    #           reflectance %.
    plot_field = None
    plot_cmap = None
    plot_cnorm = None
    cbar_ticks = None
    cbar_ticklabels = None
    cbar_label = None
    bt_min_c = bt_max_c = None  # IR/WV only -> bottom-left min/max overlay
    if not is_rgb:
        if is_visible:
            # The M2 regular path carries raw reflectance through the spatial
            # operation; this early field exists only for validity/extents.
            # Legacy rendering retains its established pre-draw sqrt mapping.
            plot_field = np.ma.masked_invalid(
                np.asarray(cmi, dtype=np.float32)
                if m2_regular_scalar
                else normalize_visible(cmi)
            )
            plot_cmap = plt.get_cmap("gray")
            plot_cnorm = Normalize(vmin=0.0, vmax=1.0)
            cbar_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
            cbar_ticklabels = ["0", "25", "50", "75", "100"]
            cbar_label = "Reflectance (%)"
        else:
            # IR/WV: brightness temperature. Source is Kelvin unless tagged C.
            bt = cmi
            if data.units in ("C", "celsius", "degC"):
                bt = bt + 273.15
            bt_c = bt - 273.15
            # ---- DEGENERATE-FRAME GUARD (scalar IR/WV) --------------------
            # A partial satellite-segment fetch (missing AHI/ABI tiles -- the
            # same s3fs listings-cache race that bit truecolor) leaves most of
            # the bbox NaN, so the frame renders as a mostly-empty "strip" and
            # the poller would upload it -> glitchy loop. Mirror the truecolor
            # guard: raise so /render returns 500 and the poller retries 3x
            # then skips, and the next scan cycle (cache aged out) renders
            # cleanly. Threshold 55% NaN: a storm floater fully inside the disk
            # is ~0% NaN, a partial fetch is ~80%+, so this only trips on the
            # broken frames (off-disk on-demand boxes are already 422'd by the
            # satellite picker before reaching here).
            # Counted over GEOMETRICALLY VALID pixels only: a disk-limb meso
            # sector (e.g. GOES-18 M2 over the Bering) legitimately has a
            # big off-disk corner whose coords are NaN — that's geometry,
            # not a broken fetch, and must not permanently 500 the sector.
            valid_geom = np.isfinite(lats) & np.isfinite(lons)
            n_valid = float(valid_geom.sum())
            if n_valid == 0:
                raise RuntimeError("bbox has no on-disk pixels to render")
            nan_frac = float(
                (~np.isfinite(bt_c) & valid_geom).sum() / n_valid
            )
            if nan_frac > 0.55:
                log.warning(
                    "scalar IR/WV degenerate (nan=%.0f%%) -- bailing out so the "
                    "poller doesn't ship a partial-fetch frame", nan_frac * 100.0,
                )
                raise RuntimeError(
                    f"scalar render produced a mostly-NaN field "
                    f"(nan={nan_frac * 100:.0f}%) -- likely a partial satellite "
                    f"segment fetch; the next scan cycle will re-render"
                )
            # --------------------------------------------------------------
            if np.isfinite(bt_c).any():
                bt_min_c = float(np.nanmin(bt_c))
                bt_max_c = float(np.nanmax(bt_c))
            plot_field = np.ma.masked_invalid(
                bt if m2_regular_scalar else bt_c
            )
            plot_cmap = enh["cmap"]
            plot_cnorm = enhancement_norm(enhancement)  # fresh, not shared
            cbar_ticks = list(enh["ticks"])
            cbar_ticklabels = [str(t) for t in enh["ticks"]]
            cbar_label = enh.get("cbar_label", "Brightness Temperature (°C)")

    # Antimeridian crossing (lon_max < lon_min, e < w convention): draw in a
    # PlateCarree frame re-centered on the bbox so the ±180 seam falls on the
    # far side of the planet. Data keeps transform=PlateCarree(0) — cartopy
    # wraps it into the shifted frame continuously, so a GOES-18 meso sector
    # steered over the Bering Sea renders seamlessly.
    lon_span_req = (lon_max - lon_min) % 360.0 or 360.0

    # ---- per-pixel validity (degenerate guards + extent crop) -------------
    if is_rgb:
        # ---- DEGENERATE-RGB GUARD --------------------------------------
        # Truecolor pulls 5 input bands (R/G/B/veggie/clean-IR) and a
        # transient cache race on ANY of them can leave the composite
        # mostly-NaN -- nan_to_num at draw time would then paint those
        # regions pure black and the render would ship a 200 OK with a
        # black PNG. Detect this and raise so /render returns 500: the
        # poller's retry/skip path discards the frame instead of uploading
        # it, and the next 10-min scan cycle (with the s3fs listings
        # cache aged out) re-renders cleanly. Threshold: >50% NaN or
        # mean-of-valid-channel-max <0.04 (~almost-pure-black even where
        # not NaN) counts as degenerate.
        if is_m2_truecolor:
            import truecolor

            physical_state = data.truecolor_physical
            day_support = physical_state.day_coverage > truecolor.M2_COVERAGE_EPSILON
            if physical_state.ir_coverage is None:
                ir_support = np.zeros(day_support.shape, dtype=bool)
            else:
                ir_support = (
                    physical_state.ir_coverage > truecolor.M2_COVERAGE_EPSILON
                )
            finite_mask = day_support | ir_support
        else:
            finite_mask = np.isfinite(cmi).all(axis=-1)
        # NaN fraction counted over geometrically-valid (on-disk) pixels
        # only, when the fetch supplied the mask: a disk-limb sector's big
        # off-disk corner is geometry, not a broken fetch (mirrors the
        # scalar guard). The mask strides with the data above.
        geom = (
            data.truecolor_physical.geom_valid
            if is_m2_truecolor
            else data.geom_valid
        )
        if geom is not None and downsample > 1 and not preserve_m2_physical_grid:
            geom = geom[::downsample, ::downsample]
        if geom is not None:
            n_geom = float(geom.sum())
            if n_geom == 0:
                raise RuntimeError("bbox has no on-disk pixels to render")
            nan_frac = float((~finite_mask & geom).sum() / n_geom)
        else:
            nan_frac = float((~finite_mask).mean())
        if is_m2_truecolor and finite_mask.any():
            # Diagnostic only: distinguish a genuinely empty/black partial
            # fetch without building or spatially sampling a legacy RGB.
            diagnostic_max = np.zeros(finite_mask.shape, dtype=np.float32)
            if np.any(day_support):
                diagnostic_max[day_support] = np.nanmax(
                    physical_state.day_rgb_reflectance[day_support], axis=-1
                )
            if np.any(ir_support):
                night_gray = truecolor._clean_ir_gray(  # noqa: SLF001
                    physical_state.ir_bt_kelvin
                )
                diagnostic_max[ir_support] = np.maximum(
                    diagnostic_max[ir_support], night_gray[ir_support]
                )
            valid_max = float(diagnostic_max[finite_mask].mean())
        elif finite_mask.any():
            valid_max = float(
                np.clip(cmi[finite_mask], 0.0, 1.0).max(axis=-1).mean()
            )
        else:
            valid_max = 0.0
        if nan_frac > 0.50 or valid_max < 0.04:
            log.warning(
                "truecolor RGB degenerate (nan=%.0f%%, mean valid max=%.3f) "
                "-- bailing out so the poller doesn't ship a black frame",
                nan_frac * 100.0, valid_max,
            )
            raise RuntimeError(
                f"truecolor render produced degenerate RGB "
                f"(nan={nan_frac * 100:.0f}%, mean_valid_max={valid_max:.3f}) "
                f"-- likely a cache race on one of the input band listings; "
                f"the next scan cycle will re-render"
            )
        # ----------------------------------------------------------------
        valid_px = finite_mask & geom if geom is not None else finite_mask
    else:
        # Disk-limb sectors: off-disk pixels inverse-project to NaN lat/lon,
        # which pcolormesh hard-rejects. Mask those cells in the field and
        # nearest-fill the coords — the repeated coordinates collapse the
        # phantom quads to zero area, so only real data draws.
        coord_bad = ~(np.isfinite(lats) & np.isfinite(lons))
        valid_px = ~np.ma.getmaskarray(plot_field) & ~coord_bad
        if coord_bad.any():
            plot_field = np.ma.masked_where(coord_bad, plot_field)
            lats = _fill_coord_nan(lats)
            lons = _fill_coord_nan(lons)

    # ---- frame geometry: tighten the view to the valid-data extent --------
    eff_lo, eff_hi, eff_lat_lo, eff_lat_hi = _effective_extent(
        lats, lons, valid_px, bbox, lon_span_req
    )
    lon_span = eff_hi - eff_lo
    lat_span = eff_lat_hi - eff_lat_lo
    aspect = lon_span / max(lat_span, 1e-6)

    # Figure size: fixed 12 in wide, height by aspect. The OUTPUT RESOLUTION is
    # the per-tier ``dpi`` at savefig (default 110 -> ~1320 px; low ~70 -> ~840 px;
    # high ~200 -> ~2400 px). figsize is held constant so layout proportions +
    # font sizes scale uniformly and ALL chrome (vector text/lines) renders crisp
    # at the tier dpi -- never bitmap-resized. default dpi 110 == today (byte-
    # identical); the webp LOOP path always passes 110 (then transcodes to 1056).
    fig_w = 12.0
    fig_h = max(4.0, fig_w / max(aspect, 0.3))
    # M2 captures and replays exact device-pixel geometry before savefig.  The
    # figure must therefore be created at the requested delivery DPI; planning
    # at Matplotlib's default DPI and changing DPI only in savefig would scale
    # the canvas after the authoritative raster had already been placed.
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=DARK_BG)

    # Layout: title strip on top (~6%), main map fills the rest with a small
    # bottom margin for gridline labels. A labeled vertical colorbar sits in a
    # reserved right margin for every scalar (non-RGB) product; true color has
    # no colorbar so the map uses the full width.
    title_h = 0.06
    bottom_pad = 0.04  # leaves room for x-axis gridline labels
    map_h = 1.0 - title_h - bottom_pad
    show_cbar = not is_rgb
    map_w = 0.84 if show_cbar else 0.92

    # The map frame is the EFFECTIVE (valid-data) extent. Legacy imagery keeps
    # its historical request-as-outer-edge extent. M2 true colour instead uses
    # the validated centre-derived cell edges while this view continues to
    # clip to the requested/effective coordinate range.
    if crosses:
        center_uw = lon_min + lon_span_req / 2.0
        center_lon = ((center_uw + 180.0) % 360.0) - 180.0
        map_crs = ccrs.PlateCarree(central_longitude=center_lon)
        view_extent = [eff_lo - center_uw, eff_hi - center_uw,
                       eff_lat_lo, eff_lat_hi]
        img_extent = [-lon_span_req / 2.0, lon_span_req / 2.0,
                      lat_min, lat_max]
    else:
        map_crs = ccrs.PlateCarree()
        view_extent = [eff_lo, eff_hi, eff_lat_lo, eff_lat_hi]
        img_extent = [lon_min, lon_max, lat_min, lat_max]

    ax = fig.add_axes([0.04, bottom_pad, map_w, map_h], projection=map_crs)
    ax.set_facecolor(DARK_BG)
    ax.set_extent(view_extent, crs=map_crs)

    mesh = None
    # Plot with the (lats, lons) arrays we computed via inverse projection.
    if is_rgb:
        if is_m2_truecolor:
            import truecolor

            plan, device_pixel_bounds = _m2_plan_axes_raster(
                fig,
                ax,
                cmi.shape[:2],
                origin="upper",
                extent=m2_truecolor_edge_extent,
                transform=map_crs,
                policy=policy,
            )
            delivered_state = _resample_truecolor_physical_state(
                data.truecolor_physical, plan
            )
            delivery = truecolor.finalize_truecolor_m2(
                delivered_state,
                background_display_rgb=to_rgb(DARK_BG),
            )
            rgb = plan.delivered_values(delivery.rgb)
            _m2_place_verified(
                fig,
                ax,
                rgb,
                plan,
                device_pixel_bounds,
            )
        else:
            # Legacy true-color RGB is on a regular lat/lon grid and remains
            # available as the exact M0/M1 rollback path.
            rgb = np.clip(
                np.nan_to_num(cmi, nan=0.0).astype(np.float32), 0.0, 1.0
            )
            _imshow_with_render_policy(
                ax,
                rgb,
                policy=policy,
                origin="upper",
                extent=img_extent,
                transform=map_crs,
                zorder=1,
            )
    else:
        # Masked / non-finite geolocation (off-disk limb; early-era ABI
        # sectors) would raise inside pcolormesh -- same guard as the
        # backdrop path (one guard, both call sites).
        if crosses:
            lons = _unwrap_lons(lons, lon_min)
        # The strict M2 binder already proved exact plain, finite coordinates.
        # Avoid the legacy guard's float64 coercion so a second strict bind at
        # the planning boundary can prove the same source, not a lookalike.
        if not m2_regular_scalar:
            lons, lats, plot_field = _guard_mesh_coords(lons, lats, plot_field)
        # ---- REGULAR-GRID FAST PATH (Time Machine archive tiers) --------
        # GridSat-B1 / GridSat-GOES / MergIR (and GOES fetch_regular) are
        # uniform lat/lon grids that the fetchers meshgrid into 2-D coords;
        # pcolormesh then transforms MILLIONS of quad vertices per frame
        # (seconds each -- the dominant server cost of a 25-frame archive
        # window). A separable uniform grid renders identically via imshow
        # with a cell-EDGE extent (shading="auto" places edges at centers
        # ± half-step; matched exactly here) in milliseconds. The native
        # geos path (floaters / meso / live sectors -- genuinely
        # curvilinear 2-D coords) can never satisfy the detector, so the
        # frozen-renderer guarantee is untouched: this branch simply never
        # fires for it. Antimeridian-crossing boxes stay on pcolormesh
        # (axes central_longitude=180 vs a PlateCarree(0) image transform
        # would trigger cartopy's warp -- the same reason the true-color
        # imshow path is 422-gated for crossing).
        if policy.resample_in_reflectance:
            axes1d = None if crosses else _m2_regular_grid_axes(lons, lats)
            if axes1d is None or m2_scalar_axes is None:
                raise RuntimeError(
                    "registered M2 policy lost its strict scalar-grid binding "
                    "before raster planning"
                )
            if not (
                np.array_equal(axes1d[0], m2_scalar_axes[0])
                and np.array_equal(axes1d[1], m2_scalar_axes[1])
            ):
                raise RuntimeError(
                    "registered M2 scalar-grid binding changed before raster planning"
                )
        else:
            axes1d = None if crosses else _regular_grid_axes(lons, lats)
        if axes1d is not None:
            lon1, lat1 = axes1d
            dlon = (lon1[-1] - lon1[0]) / max(len(lon1) - 1, 1)
            dlat = (lat1[-1] - lat1[0]) / max(len(lat1) - 1, 1)
            ext_img = [
                lon1[0] - dlon / 2.0, lon1[-1] + dlon / 2.0,
                min(lat1[0], lat1[-1]) - abs(dlat) / 2.0,
                max(lat1[0], lat1[-1]) + abs(dlat) / 2.0,
            ]
            scalar_origin = "lower" if dlat > 0 else "upper"
            if policy.resample_in_reflectance:
                plan, device_pixel_bounds = _m2_plan_axes_raster(
                    fig,
                    ax,
                    cmi.shape[:2],
                    origin=scalar_origin,
                    extent=ext_img,
                    transform=ccrs.PlateCarree(),
                    policy=policy,
                )
                source_coverage = (
                    np.isfinite(cmi)
                    & np.isfinite(lats)
                    & np.isfinite(lons)
                ).astype(np.float32)
                rgba, delivered_scalar = _m2_scalar_rgba(
                    cmi,
                    source_coverage,
                    plan,
                    is_visible=is_visible,
                    source_units=data.units,
                    cmap=plot_cmap,
                    norm=plot_cnorm,
                )
                _m2_place_verified(
                    fig,
                    ax,
                    rgba,
                    plan,
                    device_pixel_bounds,
                )
                # Colorbar semantics are independent of the 1:1 imagery
                # layer: it maps the same delivered scalar units/cmap without
                # relying on a pre-colour spatial artist.
                mesh = matplotlib.cm.ScalarMappable(
                    norm=plot_cnorm,
                    cmap=plot_cmap,
                )
                mesh.set_array(delivered_scalar[np.isfinite(delivered_scalar)])
            else:
                mesh = _imshow_with_render_policy(
                    ax,
                    np.ma.masked_invalid(plot_field),
                    policy=policy,
                    origin=scalar_origin,
                    extent=ext_img,
                    cmap=plot_cmap,
                    norm=plot_cnorm,
                    transform=ccrs.PlateCarree(),
                    zorder=1,
                )
        else:
            if policy.resample_in_reflectance:
                raise RuntimeError(
                    "registered M2 policy rejects curvilinear or "
                    "antimeridian-crossing scalar fallback; caller must "
                    "select the registered legacy policy"
                )
            mesh = ax.pcolormesh(
                lons,
                lats,
                plot_field,
                cmap=plot_cmap,
                norm=plot_cnorm,
                shading="auto",
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )

    # Coastlines + borders. Resolution scales with bbox; zorder explicitly
    # above pcolormesh (which defaults to ~1.5 in cartopy) so cyan coast
    # never gets painted over by hot cloud tops; full alpha for legibility.
    if coastlines:
        coast_scale = _coast_resolution(max(lon_span, lat_span))
        # halo under-strokes first (same geometry, slightly wider, below)…
        ax.add_feature(
            cfeature.COASTLINE.with_scale(coast_scale),
            linewidth=2.2, edgecolor=LINE_HALO, alpha=0.9, zorder=3,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale(coast_scale),
            linewidth=1.6, edgecolor=LINE_HALO, alpha=0.9, zorder=3,
        )
        # …then the legible light strokes on top
        ax.add_feature(
            cfeature.COASTLINE.with_scale(coast_scale),
            linewidth=1.0, edgecolor=COAST_COLOR, alpha=1.0, zorder=3.02,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale(coast_scale),
            linewidth=0.7, edgecolor=BORDER_COLOR, alpha=1.0, zorder=3.02,
        )
        # State/province (admin_1) boundary LINES — the internal-boundary-only
        # dataset (not the admin_1 *lakes* polygons), so it never re-traces the
        # coastline. Same halo'd off-white as the country borders, a touch
        # thinner so US/MX/AU state lines read as subtle landfall context.
        # Loaded from the vendored geojson (see _state_lines_feature) so it
        # works on the deploy host. Guarded: a missing asset degrades to "no
        # state lines" rather than a failed frame.
        try:
            states = _state_lines_feature()
            if states is not None:
                ax.add_feature(
                    states, linewidth=1.1, edgecolor=LINE_HALO,
                    facecolor="none", alpha=0.9, zorder=3,
                )
                ax.add_feature(
                    states, linewidth=0.45, edgecolor=BORDER_COLOR,
                    facecolor="none", alpha=1.0, zorder=3.02,
                )
        except Exception as e:  # noqa: BLE001 — never let admin_1 break a frame
            log.warning("state borders skipped: %s", e)

    # Dashed gridlines auto-spaced (toggleable) over the EFFECTIVE extent;
    # crossing frames lay xlocs on the unwrapped range then wrap to ±180.
    if gridlines:
        # Dashed gridlines auto-spaced over the EFFECTIVE extent. Crossing frames
        # lay xlocs out on the unwrapped lon range then wrap to ±180 (true
        # longitudes, so the gridliner labels them correctly in the re-centered
        # frame); plain frames keep the raw values — wrapping would map a bbox
        # edge at exactly 180 to -180 and silently drop that meridian's gridline.
        step = _gridline_step(max(lon_span, lat_span))
        xlocs = np.arange(
            np.floor(eff_lo / step) * step, eff_hi + step, step
        )
        if crosses:
            xlocs = ((xlocs + 180.0) % 360.0) - 180.0
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            linestyle="--",
            color=GRID_COLOR,
            alpha=0.7,
            xlocs=xlocs,
            ylocs=np.arange(np.floor(eff_lat_lo / step) * step,
                            eff_lat_hi + step, step),
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"color": TEXT_COLOR, "size": 8}
        gl.ylabel_style = {"color": TEXT_COLOR, "size": 8}

    # Right-side colorbar (every scalar product). Lives in the reserved right
    # margin; physical °C ticks for IR/WV, reflectance % for visible.
    if show_cbar and mesh is not None:
        cbar_ax = fig.add_axes([0.905, bottom_pad + 0.04, 0.016, map_h - 0.08])
        cbar = fig.colorbar(mesh, cax=cbar_ax)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
        cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8, length=3)
        cbar.outline.set_edgecolor(GRID_COLOR)
        cbar.set_label(cbar_label, color=MUTED_COLOR, fontsize=8)

    # Title strip
    title_ax = fig.add_axes([0, 1.0 - title_h, 1.0, title_h])
    title_ax.set_facecolor(DARK_BG)
    title_ax.axis("off")
    # Sensor label: read off FetchResult so it works for both ABI (GOES) and
    # AHI (Himawari) without per-family branching here. The GridSat-B1 deep
    # archive gets an HONEST era title — actual source, channel, cadence and
    # resolution — so an old frame can never imply modern imagery.
    is_gridsat = data.bucket.startswith("noaa-cdr-gridsat")
    is_mergir = data.bucket == "gesdisc-mergir"
    is_gg = data.bucket == "ncei-gridsat-goes"
    if is_gg:
        # per-satellite GOES-era tier: name the ACTUAL satellite + channel +
        # cadence + resolution (native 1 km GVAR is order-staged only — never
        # imply it; the visible channel says its 4 km is subsampled)
        sensor_label = "GOES Imager (GridSat-GOES)"
        gg_chan = {1: "0.65 µm visible (4 km, from 1 km)",
                   2: "3.9 µm shortwave IR",
                   3: "6.5 µm water vapor",
                   4: "10.7 µm IR window"}.get(channel, f"ch{channel}")
        center_title = (
            f"{data.sat_name} · GridSat-GOES · {gg_chan} · hourly · ~4 km "
            f"· {time_str} UTC")
    elif is_mergir:
        sensor_label = "merged geostationary IR"
        center_title = (
            f"NASA MergIR · 11 µm IR window · 30-min · ~4 km · {time_str} UTC")
    elif is_gridsat:
        sensor_label = "geostationary IR composite"
        gs_chan = "11 µm IR window" if channel == 1 else "6.7 µm water vapor"
        center_title = (
            f"GridSat-B1 · {gs_chan} · 3-hourly · ~8 km · {time_str} UTC")
    else:
        sensor_label = "AHI" if data.bucket.startswith("noaa-himawari") else "ABI"
        center_title = (
            f"{data.sat_name} {sensor_label} True Color · {time_str} UTC"
            if is_rgb
            else f"{data.sat_name} {sensor_label} Channel {channel:02d} · {time_str} UTC"
        )
    title_ax.text(
        0.5, 0.5,
        center_title,
        ha="center", va="center",
        color=TEXT_COLOR, fontsize=14, fontweight="bold",
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.99, 0.5,
        f"{data.product} · {'true color' if is_rgb else enhancement}",
        ha="right", va="center",
        color=ACCENT_COLOR, fontsize=9,
        transform=title_ax.transAxes,
    )

    # Brightness-temperature min/max, in the HEADER beside the storm stats.
    # Two rows INSIDE the existing 6% strip -- title_h must not change: the
    # explorer's objfix georeferencing pins its data-axes constants to this
    # layout (objfix_sources.js LAYOUT.dataY1), so growing the strip would
    # silently shift every floater fix.
    bt_tag = None if (is_rgb or is_visible) else _bt_band_tag(data, channel)
    show_bt = bt_tag is not None and bt_min_c is not None and bt_max_c is not None

    # Storm badge (left of title strip) — only when /render is called with
    # storm context (poller path). Format:
    #   JANGMI · TS · 35 kt · 998 mb
    # Color-coded by Saffir-Simpson category as a tinted background pill.
    if storm:
        name = (storm.get("name") or "").upper()[:18]
        wind_kt = storm.get("wind_kt")
        pressure_mb = storm.get("pressure_mb")
        nature = storm.get("nature")
        cat = _ss_category(nature, wind_kt)
        cat_color = _SS_COLORS.get(cat, _SS_COLORS["EX"])
        parts = [name, cat]
        if wind_kt is not None:
            parts.append(f"{int(round(wind_kt))} kt")
        if pressure_mb is not None:
            parts.append(f"{int(round(pressure_mb))} mb")
        badge_text = "  ·  ".join(p for p in parts if p)
        title_ax.text(
            0.01, 0.72 if show_bt else 0.5,
            badge_text,
            ha="left", va="center",
            color=TEXT_COLOR, fontsize=10, fontweight="bold",
            transform=title_ax.transAxes,
            bbox=dict(
                facecolor=cat_color, alpha=0.22, edgecolor=cat_color,
                linewidth=1.0, boxstyle="round,pad=0.35",
            ),
        )

    if show_bt:
        # Second header row under the storm badge (or vertically centred when
        # there is no badge). This USED to sit at the bottom-left of the MAP
        # axes, where its opaque backing box overwrote real data pixels --
        # exactly the pixels the explorer's objfix reads brightness
        # temperature back out of. In the header it occludes nothing.
        title_ax.text(
            0.01, 0.22 if storm else 0.5,
            f"{bt_tag} BT   min {bt_min_c:+.1f} °C   max {bt_max_c:+.1f} °C",
            ha="left", va="center",
            color=ACCENT_COLOR, fontsize=8,
            transform=title_ax.transAxes,
        )

    # Watermark: top-left of the map axes, mirroring the title strip's
    # right-aligned product label so the two corners balance visually.
    # Translucent dark backing rect keeps it legible over hot pixels.
    source_label = ("NASA" if is_mergir
                    else "NOAA NCEI" if is_gg
                    else "NOAA CDR" if is_gridsat
                    else "JMA" if data.bucket.startswith("noaa-himawari")
                    else "NOAA")
    ax.text(
        0.01, 0.99,
        f"@WeathermanAAA_  ·  {source_label} {data.sat_name} {sensor_label}",
        ha="left", va="top",
        color=ACCENT_COLOR, fontsize=9,
        transform=ax.transAxes,
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=4),
        zorder=10,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _erode1(mask: np.ndarray) -> np.ndarray:
    """Erode a boolean mask by one 4-connected cell (True = keep). A kept cell
    that touches a dropped cell in any of the 4 directions is itself dropped --
    so the surviving cells never border the off-disk fill region (see
    _guard_mesh_coords). Pure numpy (no scipy)."""
    e = mask.copy()
    e[1:, :] &= mask[:-1, :]
    e[:-1, :] &= mask[1:, :]
    e[:, 1:] &= mask[:, :-1]
    e[:, :-1] &= mask[:, 1:]
    return e


def _regular_grid_axes(lons, lats):
    """(lon1, lat1) 1-D center axes when the 2-D coord arrays are a UNIFORM,
    SEPARABLE lat/lon meshgrid — else None.

    The archive tiers (GridSat-B1 0.07°, GridSat-GOES 0.04°, MergIR ~0.036°)
    and GOES fetch_regular all build their 2-D coords as np.meshgrid of
    uniform 1-D axes; the render fast path (imshow) is exact only for that
    shape. Full-array separability checks (every lons row identical, every
    lats column identical) cost ~ms on a 5M-cell grid — cheap insurance that
    a curvilinear geos grid (which varies per row/column by construction)
    can NEVER slip through. Uniform-step tolerance absorbs float32 coord
    wobble (~1e-5°) while rejecting real curvature (degrees). Masked
    geolocation = limb handling = never regular. Ascending lons only (the
    crop functions emit ascending, antimeridian-unwrapped axes; unwrapped
    >180 grids are handled by the caller's `crossing` gate)."""
    if getattr(lons, "ndim", 0) != 2 or getattr(lats, "ndim", 0) != 2:
        return None
    if lons.shape != lats.shape or lons.shape[0] < 2 or lons.shape[1] < 2:
        return None
    if isinstance(lons, np.ma.MaskedArray) or isinstance(lats, np.ma.MaskedArray):
        return None
    lon1 = np.asarray(lons[0], dtype=np.float64)
    lat1 = np.asarray(lats[:, 0], dtype=np.float64)
    if not (np.isfinite(lon1).all() and np.isfinite(lat1).all()):
        return None
    if not ((lons == lons[0]).all() and (lats == lats[:, :1]).all()):
        return None
    dlon = np.diff(lon1)
    dlat = np.diff(lat1)
    if not (dlon > 0).all():
        return None
    if not ((dlat > 0).all() or (dlat < 0).all()):
        return None
    tol = 1e-3
    if (dlon.max() - dlon.min()) > tol or (np.abs(dlat).max() - np.abs(dlat).min()) > tol:
        return None
    return lon1, lat1


def _float32_axes_equal_within_ulps(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    max_ulps: int = 4,
) -> bool:
    """Compare a float32 archive axis with its endpoint linear fit.

    Archive coordinates originate as uniform float64 netCDF vectors and are
    cast to float32 only after cropping/mesh construction.  Comparing each
    value against *its local* ULP is invalid near zero and rejects genuine
    grids after thousands of accumulated steps.  Bound the residual by four
    float32 ULPs at the largest coordinate magnitude plus a tiny step-relative
    term.  This admits only representational quantisation (tens of microdegrees
    at global longitude), remaining orders of magnitude below the gradual
    0.125-degree warp mutant.
    """

    if (
        actual.dtype != np.dtype(np.float32)
        or expected.dtype != np.dtype(np.float32)
        or actual.shape != expected.shape
        or max_ulps < 0
    ):
        return False
    expected64 = expected.astype(np.float64)
    scale_value = np.float32(
        max(float(np.max(np.abs(expected64))), np.finfo(np.float32).tiny)
    )
    scale_ulp = float(
        np.nextafter(scale_value, np.float32(np.inf), dtype=np.float32)
        - scale_value
    )
    nominal_step = abs(
        float(expected64[-1] - expected64[0]) / (expected64.size - 1)
    )
    tolerance = max(
        float(max_ulps) * scale_ulp,
        nominal_step * 4.0 * float(np.finfo(np.float32).eps),
    )
    difference = np.abs(actual.astype(np.float64) - expected64)
    return bool(np.all(difference <= tolerance))


def _m2_regular_grid_axes(lons, lats):
    """Strict M2 binder for the two registered archive scalar families.

    Unlike the legacy speed-path detector, this accepts only exact plain
    float32 meshgrid arrays whose complete axes agree with their endpoint
    ``linspace`` within the bounded archive-quantisation tolerance above.  A
    gradually warped axis can have
    almost-uniform local steps and still move physical pixel centres; it must
    never acquire M2 authority.
    """

    if type(lons) is not np.ndarray or type(lats) is not np.ndarray:
        return None
    if lons.dtype != np.dtype(np.float32) or lats.dtype != np.dtype(np.float32):
        return None
    if lons.ndim != 2 or lats.ndim != 2:
        return None
    if lons.shape != lats.shape or min(lons.shape) < 2:
        return None
    if not np.all(np.isfinite(lons)) or not np.all(np.isfinite(lats)):
        return None
    lon_axis = lons[0, :]
    lat_axis = lats[:, 0]
    if not np.array_equal(lons, np.broadcast_to(lon_axis, lons.shape)):
        return None
    if not np.array_equal(lats, np.broadcast_to(lat_axis[:, None], lats.shape)):
        return None
    if not np.all(np.diff(lon_axis.astype(np.float64)) > 0.0):
        return None
    lat_steps = np.diff(lat_axis.astype(np.float64))
    if not (np.all(lat_steps > 0.0) or np.all(lat_steps < 0.0)):
        return None
    expected_lon = np.linspace(
        lon_axis[0], lon_axis[-1], lon_axis.size, dtype=np.float32
    )
    expected_lat = np.linspace(
        lat_axis[0], lat_axis[-1], lat_axis.size, dtype=np.float32
    )
    if not _float32_axes_equal_within_ulps(lon_axis, expected_lon):
        return None
    if not _float32_axes_equal_within_ulps(lat_axis, expected_lat):
        return None
    return lon_axis.copy(), lat_axis.copy()


def _guard_mesh_coords(lons, lats, plot_field):
    """Make (lons, lats, plot_field) safe for pcolormesh when the geolocation
    carries masked / non-finite cells.

    Geostationary lat/lon grids carry NaN at the OFF-DISK limb (and early-era
    ABI sectors -- e.g. GOES-16 CONUS from the 89.5W checkout slot in 2017 --
    carry off-earth pixels INSIDE the sector), and pcolormesh REJECTS
    non-finite values in its x/y (coord) arrays ("x and y arguments ... cannot
    have non-finite values or be of type numpy.ma.MaskedArray"). Mask the
    field at those cells (-> transparent), replace the bad coords with a
    finite in-extent fill so pcolormesh accepts the grid, and erode the valid
    region by one cell so an on-disk edge cell never stretches a quad out to a
    filled coord (shading="auto" averages neighbouring centres -> limb streaks
    otherwise). Coords are coerced to plain NaN-filled float arrays FIRST -- a
    fetch may hand back MASKED lat/lon, and pcolormesh rejects both masked
    coords and non-finite values, so np.where alone (which can stay masked) is
    not enough. Shared by the main scalar render and the backdrop render (one
    guard, both call sites)."""
    lons = np.asarray(np.ma.filled(lons, np.nan), dtype=float)
    lats = np.asarray(np.ma.filled(lats, np.nan), dtype=float)
    xy_ok = np.isfinite(lons) & np.isfinite(lats)
    if not xy_ok.all():
        valid = _erode1(xy_ok & ~np.ma.getmaskarray(plot_field))
        plot_field = np.ma.masked_array(np.ma.getdata(plot_field), mask=~valid)
        fill_lon = float(np.nanmean(lons)) if np.isfinite(lons).any() else 0.0
        fill_lat = float(np.nanmean(lats)) if np.isfinite(lats).any() else 0.0
        lons = np.where(xy_ok, lons, fill_lon)
        lats = np.where(xy_ok, lats, fill_lat)
    return lons, lats, plot_field


def render_backdrop_webp(
    data,
    bbox,
    *,
    enhancement: str = "grayscale",
    downsample: int = 1,
    dpi: int = 110,
    quality: int = 82,
) -> bytes:
    """Bare GRAYSCALE Vis/SWIR satellite backdrop cutout for the ASCAT + MW viewers.

    Day scenes arrive as a VISIBLE channel (reflectance, units "1") and render via
    the sqrt-stretched grayscale recipe; night scenes arrive as SHORT-WAVE IR (or
    clean IR, brightness temperature) and render via the gray BT table. The
    day/night CHOICE of band is made upstream (pick_backdrop_band); only the units
    distinguish the two paths here. ZERO baked chrome either way: ONE full-bleed
    PlateCarree axes (set_aspect('auto') so the data fills the frame edge-to-edge),
    and no coastlines / gridlines / colorbar / title strip / storm badge /
    watermark / min-max overlay. Returns an OPAQUE WebP georeferenced to ``bbox``
    ([W, S, E, N]); the consumer draws it into those exact WGS84 corner bounds and
    owns the single shared graticule, coastline, colorbar, legend and watermark.
    Grayscale ONLY so the colored barbs / MW over it stay legible. Raises
    RuntimeError on a mostly-NaN (degenerate / partial-fetch) field and ValueError
    on a non-gray enhancement on the thermal path.
    """
    cmi = data.cmi
    lats = data.lats
    lons = data.lons
    if downsample > 1:
        cmi = cmi[::downsample, ::downsample]
        lats = lats[::downsample, ::downsample]
        lons = lons[::downsample, ::downsample]

    if getattr(data, "units", "") == "1":
        # VIS (day): visible reflectance -> sqrt-stretched grayscale (mirror of
        # render_png's is_visible branch).
        field = np.asarray(normalize_visible(cmi), dtype=float)
        plot_field = np.ma.masked_invalid(field)
        plot_cmap = plt.get_cmap("gray")
        plot_cnorm = Normalize(vmin=0.0, vmax=1.0)
    else:
        # SWIR / clean IR (night): brightness temperature in °C on the gray table.
        enh = get_enhancement(enhancement)
        if enh.get("kind") != "gray":
            raise ValueError(
                f"render_backdrop_webp requires a grayscale enhancement on the "
                f"thermal path (got {enhancement!r}, kind={enh.get('kind')!r})"
            )
        bt = cmi
        if data.units in ("C", "celsius", "degC"):
            bt = bt + 273.15
        field = np.asarray(bt - 273.15, dtype=float)
        plot_field = np.ma.masked_invalid(field)
        plot_cmap = enh["cmap"]
        plot_cnorm = enhancement_norm(enhancement)  # fresh, not shared

    nan_frac = float(np.isnan(field).mean()) if field.size else 1.0
    if nan_frac > 0.55:
        raise RuntimeError(
            f"backdrop render produced a mostly-NaN field (nan={nan_frac:.0%}) "
            "— bailing so a partial fetch never publishes a near-empty backdrop"
        )

    # Off-disk / masked geolocation guard -- shared with the main scalar
    # render (see _guard_mesh_coords; without it every basin backdrop 500s).
    if bbox[2] < bbox[0]:   # antimeridian-crossing: continuous lons first
        lons = _unwrap_lons(lons, bbox[0])
    lons, lats, plot_field = _guard_mesh_coords(lons, lats, plot_field)

    lon_min, lat_min, lon_max, lat_max = bbox
    proj, extent, lon_span, _crossing = _map_geometry(bbox)
    lat_span = lat_max - lat_min
    aspect = lon_span / max(lat_span, 1e-6)

    # Pixel proportions track the bbox aspect; set_aspect("auto") then fills the
    # axes edge-to-edge so the image corners ARE the bbox corners (no letterbox).
    fig_w = 10.0
    fig_h = max(2.0, fig_w / max(aspect, 0.2))
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_facecolor(DARK_BG)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_aspect("auto")
    ax.axis("off")
    ax.pcolormesh(
        lons, lats, plot_field,
        cmap=plot_cmap, norm=plot_cnorm, shading="auto",
        transform=ccrs.PlateCarree(), rasterized=True,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    return encode_webp(buf.getvalue(), quality)


def encode_webp(png: bytes, quality: int) -> bytes:
    """Re-encode a rendered PNG as lossy WebP at its NATIVE size (no resize).

    The custom-zoom "low" tier uses this for a small download while keeping the
    chrome crisp: the figure is already rendered small (low dpi), so there is no
    bitmap downscale of the composited plot (that is what used to pixelate the
    title/coastlines/colorbar). Only the codec changes. Distinct from
    ``transcode_frame``, which DOWNSCALES the 1320 px loop render to the fixed
    WEBP_FRAME_WIDTH and is the floater/meso poller path -- untouched here.
    """
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    out = io.BytesIO()
    im.save(out, "WEBP", quality=quality, method=6)
    return out.getvalue()


def transcode_frame(png: bytes, width: int, quality: int) -> bytes:
    """Downscale a rendered PNG and re-encode it as lossy WebP loop-frame bytes.

    The figure is still rendered at the full 1320 px and downscaled here, NOT
    rendered natively at the target width: Lanczos from the oversampled render
    is exactly what browsers were already doing client-side (1320 -> 1050
    device px inside the 525 CSS px frame box), so this path changes the codec
    and the transfer weight, not the displayed look. Frames are opaque
    (DARK_BG facecolor) -- encode RGB, no alpha.
    """
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    if width < im.width:
        height = max(1, round(im.height * width / im.width))
        # Loop frames are square-by-design products (the 12deg storm floater, the
        # square meso sectors). The upstream cartopy figure height occasionally
        # rounds 1px off (a sub-0.1% bbox-aspect wobble as the storm drifts),
        # flipping a frame between e.g. 1056x1056 and 1056x1055 for a multi-hour
        # block. With the live player resizing its <canvas> to each frame's
        # native size, that made the loop visibly jump every cycle (the
        # "seizure"). Snap a near-square result to EXACTLY square so every frame
        # of a loop matches: a <=3px nudge is sub-0.3% (imperceptible), while a
        # genuinely non-square product (>3px off) is left untouched.
        if abs(height - width) <= 3:
            height = width
        im = im.resize((width, height), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "WEBP", quality=quality, method=6)
    return out.getvalue()
