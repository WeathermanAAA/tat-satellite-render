"""Read-only adapter from the V-gate harness to TAT's production modules.

The adapter deliberately calls the real rendering and colour functions.  It
contains instrumentation and test-image construction, not a second colour
recipe.  Keeping that boundary explicit prevents a harness port from passing
while production silently does something different.
"""

from __future__ import annotations

import ast
import contextlib
import datetime as dt
import hashlib
import importlib
import inspect
import io
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping
from unittest import mock

import numpy as np

from .metrics import forbidden_sharpening_findings


UTC = dt.timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_MODULE_NAMES = (
    "app",
    "poller_framework",
    "render",
    "s2_recipes",
    "satellites",
    "truecolor",
)


def production_modules() -> dict[str, object]:
    """Import and return the current checkout's production modules."""

    modules = {name: importlib.import_module(name) for name in PRODUCTION_MODULE_NAMES}
    for name, module in modules.items():
        path = Path(inspect.getfile(module)).resolve()
        if REPO_ROOT not in path.parents:
            raise RuntimeError(
                f"validation imported {name} from {path}, not target repo {REPO_ROOT}"
            )
    return modules


def source_sha256(function) -> str:
    """Hash a function body exactly as imported from production."""

    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def cira_source_sha256() -> str:
    return source_sha256(production_modules()["truecolor"].cira_stretch)


@dataclass(frozen=True)
class ImshowCall:
    line: int
    interpolation: str | None
    branch: str


def imshow_calls(policy_version: str | None = None) -> tuple[ImshowCall, ...]:
    """Return the effective literal kernel at both regular-grid render sites.

    M1 keeps literal ``nearest`` and ``hanning`` calls in a shared dispatcher
    so the previous M0 generation remains selectable for rollback.  Merely
    enumerating every ``Axes.imshow`` node would therefore report dispatcher
    alternatives instead of the policy used at the RGB and scalar production
    sites.  Resolve the requested immutable policy, prove its dispatcher arm
    contains the matching literal, then report that effective value at each
    call site.  The default stays M0 for the M0 baseline adapter; later-stage
    validators must name their policy explicitly.
    """

    render_policy = importlib.import_module("render_policy")
    selected_name = (
        render_policy.M0_POLICY.algo_version
        if policy_version is None
        else policy_version
    )
    policy = render_policy.resolve_render_policy(selected_name)
    path = REPO_ROOT / "render.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    dispatcher_literals: dict[str, str | None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, str)
            and isinstance(test.left, ast.Attribute)
            and test.left.attr == "resample_kernel"
        ):
            continue
        declared = test.comparators[0].value
        literal: str | None = None
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if not (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "imshow"
            ):
                continue
            interpolation = next(
                (keyword.value for keyword in child.keywords if keyword.arg == "interpolation"),
                None,
            )
            if isinstance(interpolation, ast.Constant) and isinstance(
                interpolation.value, str
            ):
                literal = interpolation.value
        dispatcher_literals[declared] = literal

    effective = dispatcher_literals.get(policy.resample_kernel)
    if effective != policy.resample_kernel:
        effective = None

    calls: list[ImshowCall] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_imshow_with_render_policy"
        ):
            continue
        branch = "scalar" if any(k.arg == "cmap" for k in node.keywords) else "rgb"
        calls.append(ImshowCall(node.lineno, effective, branch))
    return tuple(sorted(calls, key=lambda item: item.line))


def assemble_call_order() -> tuple[str, ...]:
    """Pointwise/spatial call order in the real ``assemble_truecolor`` body."""

    tc = production_modules()["truecolor"]
    source = inspect.getsource(tc.assemble_truecolor)
    tree = ast.parse(source)
    wanted = {"self_sharpen", "cira_stretch", "glint_tame", "night_blend"}
    found: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else None
            )
            if name in wanted:
                found.append((node.lineno, node.col_offset, name))
    return tuple(name for _, _, name in sorted(found))


def forbidden_operator_findings(paths: tuple[Path, ...] | None = None) -> tuple[str, ...]:
    """AST scan for prohibited enhancement operators in the render path."""

    paths = paths or tuple(
        REPO_ROOT / name
        for name in ("app.py", "render.py", "satellites.py", "truecolor.py")
    )
    findings: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        findings.extend(
            f"{path.name}:{finding.line}:{finding.identifier}"
            for finding in forbidden_sharpening_findings(source)
        )
    return tuple(sorted(findings))


@dataclass(frozen=True)
class PortFidelityResult:
    case_id: str
    matplotlib_version: str
    sampling_model: str
    bbox_degrees: tuple[float, float]
    geographic_aspect_ratio: float
    source_shape: tuple[int, int, int]
    figure_shape: tuple[int, int, int]
    delivered_shape: tuple[int, int, int]
    device_origin: tuple[int, int]
    sampling_phase: tuple[float, float]
    boundary_exclusion_px: int
    minimum_scored_identity: float
    exact_identity_required: bool
    full_pixel_identity: float
    scored_pixel_identity: float
    numeric_core_exclusion_px: int
    noise_variance_ratio: float
    delivered_sha256: str
    expected_sha256: str
    production_png_sha256: str
    png_mode: str
    png_info_keys: tuple[str, ...]
    png_icc_profile_present: bool
    png_gamma_present: bool
    png_srgb_present: bool
    quantization_model: str
    floor_quantization_scored_identity: float
    round_half_up_mutant_scored_identity: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PortFidelityCase:
    """One geographic-aspect case defined by the research port audit."""

    case_id: str
    lon_span_deg: float
    lat_span_deg: float
    source_width_px: int
    boundary_exclusion_px: int
    minimum_scored_identity: float
    exact_identity_required: bool

    @property
    def source_height_px(self) -> int:
        return int(round(self.source_width_px * self.lat_span_deg / self.lon_span_deg))


# "1:1" and "2:1" in the canonical research are geographic bbox aspect
# ratios (4 x 4 degrees and 4 x 2 degrees), not source/destination resize
# ratios.  The 2:1 case excludes the two-pixel Cartopy clip boundary before
# requiring exact identity; all unmasked pixels remain represented by the
# separately reported full-frame score.
PORT_FIDELITY_CASES: Final[Mapping[str, PortFidelityCase]] = MappingProxyType({
    "bbox_1x1": PortFidelityCase(
        case_id="bbox_1x1",
        lon_span_deg=4.0,
        lat_span_deg=4.0,
        source_width_px=1600,
        boundary_exclusion_px=1,
        minimum_scored_identity=0.998,
        exact_identity_required=False,
    ),
    "bbox_2x1": PortFidelityCase(
        case_id="bbox_2x1",
        lon_span_deg=4.0,
        lat_span_deg=2.0,
        source_width_px=1600,
        boundary_exclusion_px=2,
        minimum_scored_identity=1.0,
        exact_identity_required=True,
    ),
})


@dataclass(frozen=True)
class AnalyticFigureGeometry:
    """Independent device-space port of the production figure layout."""

    figure_shape: tuple[int, int]
    delivered_shape: tuple[int, int]
    device_origin: tuple[int, int]
    sampling_phase: tuple[float, float]
    sampling_model: str


_LEGACY_FRACTIONAL_CLIP_MODEL = "legacy_fractional_clip_ceil"
_AGG_ANCHOR_ROUNDING_MODEL = "agg_anchor_round_half"


def matplotlib_image_sampling_model() -> str:
    """Identify the installed Matplotlib ``AxesImage`` bbox algorithm.

    Matplotlib 3.9 and 3.10 keep the fractional clipped-bbox origin, ceil the
    output dimensions, and compensate by scaling the image transform.  That
    makes nearest-neighbour sampling start at an ideal half-pixel phase.

    Matplotlib 3.11 instead rounds the magnified bbox to the Agg anchor before
    constructing the transform.  That preserves the axes' fractional device
    offset in the sampling phase.  Detect the implementation itself, rather
    than fitting a phase to production pixels or assuming a version string;
    this also handles downstream backports honestly.
    """

    import matplotlib.image as mpl_image

    source = inspect.getsource(mpl_image._ImageBase._make_image)
    anchor_markers = (
        "magnified_extents = clipped_bbox.extents * magnification",
        "magnified_bbox = Bbox.from_extents",
        "Round to the nearest output pixel",
    )
    legacy_markers = (
        "out_width_base = clipped_bbox.width * magnification",
        "math.ceil(out_width_base)",
        ".translate(-clipped_bbox.x0, -clipped_bbox.y0)",
    )
    if all(marker in source for marker in anchor_markers):
        return _AGG_ANCHOR_ROUNDING_MODEL
    if all(marker in source for marker in legacy_markers):
        return _LEGACY_FRACTIONAL_CLIP_MODEL
    raise RuntimeError(
        "unsupported Matplotlib AxesImage bbox algorithm; port fidelity must "
        "be re-derived before this dependency can be accepted"
    )


def analytic_figure_geometry(
    lon_span_deg: float,
    lat_span_deg: float,
    *,
    dpi: int = 110,
    sampling_model: str | None = None,
) -> AnalyticFigureGeometry:
    """Derive the GeoAxes raster bounds without importing renderer geometry.

    Agg samples at device-pixel centres.  Matplotlib 3.11's Agg-anchor bbox
    rounding therefore retains the fractional offset between the image edge
    and the first covered device-pixel centre.  The legacy algorithm used by
    production Matplotlib 3.9.3 subtracts the fractional clipped origin before
    resampling and scales the transform to its ceiled output dimensions, so
    both axes start at the ideal half-pixel phase.
    """

    if lon_span_deg <= 0.0 or lat_span_deg <= 0.0 or dpi <= 0:
        raise ValueError("port geometry spans and dpi must be positive")
    aspect = lon_span_deg / lat_span_deg
    figure_width_in = 12.0
    figure_height_in = max(4.0, figure_width_in / max(aspect, 0.3))
    figure_width_px = figure_width_in * dpi
    figure_height_px = figure_height_in * dpi
    axes_width_px = 0.92 * figure_width_px
    axes_height_px = (1.0 - 0.06 - 0.04) * figure_height_px

    if axes_width_px / axes_height_px > aspect:
        draw_height = axes_height_px
        draw_width = draw_height * aspect
    else:
        draw_width = axes_width_px
        draw_height = draw_width / aspect

    left = 0.04 * figure_width_px + (axes_width_px - draw_width) / 2.0
    axes_bottom = 0.04 * figure_height_px
    top = figure_height_px - (
        axes_bottom + (axes_height_px - draw_height) / 2.0 + draw_height
    )
    first_col = int(math.ceil(left - 0.5))
    first_row = int(math.ceil(top - 0.5))
    last_col_exclusive = int(math.ceil(left + draw_width - 0.5))
    last_row_exclusive = int(math.ceil(top + draw_height - 0.5))
    model = sampling_model or matplotlib_image_sampling_model()
    if model == _AGG_ANCHOR_ROUNDING_MODEL:
        sampling_phase = (first_row + 0.5 - top, first_col + 0.5 - left)
    elif model == _LEGACY_FRACTIONAL_CLIP_MODEL:
        sampling_phase = (0.5, 0.5)
    else:
        raise ValueError(f"unknown Matplotlib image sampling model: {model!r}")
    return AnalyticFigureGeometry(
        figure_shape=(int(round(figure_height_px)), int(round(figure_width_px))),
        delivered_shape=(
            last_row_exclusive - first_row,
            last_col_exclusive - first_col,
        ),
        device_origin=(first_row, first_col),
        sampling_phase=sampling_phase,
        sampling_model=model,
    )


def _probe_source(case: PortFidelityCase) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0x544154 + int(case.lon_span_deg * 10 + case.lat_span_deg))
    # A stationary, zero-mean-ish noise field makes V18's variance clause
    # discriminating: nearest preserves its variance while a sharpening
    # operator increases it. Values remain away from clip endpoints.
    expected_u8 = np.clip(
        np.rint(
            rng.normal(
                loc=128.0,
                scale=24.0,
                size=(case.source_height_px, case.source_width_px, 3),
            )
        ),
        16,
        239,
    ).astype(np.uint8)
    # Values are intentionally three quarters of a DN above each floor
    # boundary.  They remain well away from floating-point ambiguity while
    # discriminating production floor conversion from round-half-up: a
    # round-half-up mutation changes every source channel by exactly +1 DN.
    rgb = (expected_u8.astype(np.float32) + np.float32(0.75)) / np.float32(255.0)
    return rgb, expected_u8


def _analytic_nearest_port(
    source_u8: np.ndarray, geometry: AnalyticFigureGeometry
) -> np.ndarray:
    output_height, output_width = geometry.delivered_shape
    phase_y, phase_x = geometry.sampling_phase
    rows = np.clip(
        np.floor(
            (np.arange(output_height, dtype=np.float64) + phase_y)
            * source_u8.shape[0]
            / output_height
        ).astype(np.int64),
        0,
        source_u8.shape[0] - 1,
    )
    columns = np.clip(
        np.floor(
            (np.arange(output_width, dtype=np.float64) + phase_x)
            * source_u8.shape[1]
            / output_width
        ).astype(np.int64),
        0,
        source_u8.shape[1] - 1,
    )
    return source_u8[rows[:, None], columns[None, :]]


@lru_cache(maxsize=2)
def port_fidelity_probe(case_id: str) -> PortFidelityResult:
    """Compare a real Cartopy render with the independent analytic port."""

    try:
        case = PORT_FIDELITY_CASES[case_id]
    except KeyError as exc:
        raise ValueError(f"unknown port-fidelity case: {case_id!r}") from exc

    modules = production_modules()
    render = modules["render"]
    satellites = modules["satellites"]
    from PIL import Image
    from cartopy.mpl.geoaxes import GeoAxes

    geometry = analytic_figure_geometry(case.lon_span_deg, case.lat_span_deg)
    rgb, source_u8 = _probe_source(case)
    bbox = [
        -case.lon_span_deg / 2.0,
        -case.lat_span_deg / 2.0,
        case.lon_span_deg / 2.0,
        case.lat_span_deg / 2.0,
    ]
    lons, lats = np.meshgrid(
        np.linspace(bbox[0], bbox[2], source_u8.shape[1], dtype=np.float32),
        np.linspace(bbox[3], bbox[1], source_u8.shape[0], dtype=np.float32),
    )
    data = satellites.FetchResult(
        cmi=rgb,
        lats=lats,
        lons=lons,
        channel=2,
        generic_channel="true_color",
        scan_start=dt.datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
        product="M0-PORT-FIDELITY",
        bucket="noaa-goes19",
        sat_name="GOES-19",
        sub_sat_lon=-75.2,
        units="rgb",
    )

    retained_figures: list[object] = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(render.plt, "close", retained_figures.append))
        # Suppress only the watermark drawn inside the map.  The title lives on
        # a normal Axes and is outside the compared map crop.
        stack.enter_context(mock.patch.object(GeoAxes, "text", lambda *a, **k: None))
        png = render.render_png(
            data,
            bbox,
            2,
            "2026-08-20 12:00",
            "true_color",
            1,
            coastlines=False,
            gridlines=False,
            dpi=110,
            policy=importlib.import_module("render_policy").M0_POLICY,
        )
    try:
        with Image.open(io.BytesIO(png)) as image:
            png_mode = image.mode
            png_info_keys = tuple(sorted(str(key) for key in image.info))
            png_icc_profile_present = bool(image.info.get("icc_profile"))
            png_gamma_present = "gamma" in image.info
            png_srgb_present = "srgb" in image.info
            figure_rgb = np.asarray(image.convert("RGB"))
    finally:
        for figure in retained_figures:
            render.plt.close(figure)

    expected_figure_shape = (*geometry.figure_shape, 3)
    if tuple(figure_rgb.shape) != expected_figure_shape:
        raise RuntimeError(
            f"{case.case_id} figure geometry drifted: "
            f"expected {expected_figure_shape}, got {figure_rgb.shape}"
        )
    first_row, first_col = geometry.device_origin
    delivered_height, delivered_width = geometry.delivered_shape
    delivered = figure_rgb[
        first_row:first_row + delivered_height,
        first_col:first_col + delivered_width,
    ]
    if tuple(delivered.shape) != (*geometry.delivered_shape, 3):
        raise RuntimeError(f"map crop drifted: {delivered.shape}")
    expected = _analytic_nearest_port(source_u8, geometry)
    equal = np.all(delivered == expected, axis=-1)
    boundary = case.boundary_exclusion_px
    scored = equal[boundary:-boundary, boundary:-boundary]
    # The source fixture is +0.75 DN above its floor values, so round-half-up
    # would increment every channel.  Comparing the actual PNG against both
    # analytic rasters makes the baseline quantizer discriminating without
    # reimplementing any production colour transform.
    round_half_up_mutant = np.minimum(
        expected.astype(np.uint16) + np.uint16(1), np.uint16(255)
    ).astype(np.uint8)
    mutant_equal = np.all(delivered == round_half_up_mutant, axis=-1)
    mutant_scored = mutant_equal[boundary:-boundary, boundary:-boundary]
    numeric_boundary = 2
    delivered_numeric = delivered[
        numeric_boundary:-numeric_boundary,
        numeric_boundary:-numeric_boundary,
    ].astype(np.float64)
    expected_numeric = expected[
        numeric_boundary:-numeric_boundary,
        numeric_boundary:-numeric_boundary,
    ].astype(np.float64)
    reference_variance = float(np.var(expected_numeric))
    if reference_variance <= np.finfo(np.float64).tiny:
        raise RuntimeError("port-fidelity noise fixture has zero variance")
    return PortFidelityResult(
        case_id=case.case_id,
        matplotlib_version=str(render.matplotlib.__version__),
        sampling_model=geometry.sampling_model,
        bbox_degrees=(case.lon_span_deg, case.lat_span_deg),
        geographic_aspect_ratio=case.lon_span_deg / case.lat_span_deg,
        source_shape=tuple(rgb.shape),
        figure_shape=tuple(figure_rgb.shape),
        delivered_shape=tuple(delivered.shape),
        device_origin=geometry.device_origin,
        sampling_phase=geometry.sampling_phase,
        boundary_exclusion_px=boundary,
        minimum_scored_identity=case.minimum_scored_identity,
        exact_identity_required=case.exact_identity_required,
        full_pixel_identity=float(equal.mean()),
        scored_pixel_identity=float(scored.mean()),
        numeric_core_exclusion_px=numeric_boundary,
        noise_variance_ratio=float(np.var(delivered_numeric) / reference_variance),
        delivered_sha256=hashlib.sha256(delivered.tobytes()).hexdigest(),
        expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        production_png_sha256=hashlib.sha256(png).hexdigest(),
        png_mode=png_mode,
        png_info_keys=png_info_keys,
        png_icc_profile_present=png_icc_profile_present,
        png_gamma_present=png_gamma_present,
        png_srgb_present=png_srgb_present,
        quantization_model="float_nearest_then_floor_u8",
        floor_quantization_scored_identity=float(scored.mean()),
        round_half_up_mutant_scored_identity=float(mutant_scored.mean()),
    )


def port_probe_png_sha256(case_id: str = "bbox_1x1") -> str:
    """Small public worker used by the three-process repeatability gate."""

    return port_fidelity_probe(case_id).production_png_sha256
