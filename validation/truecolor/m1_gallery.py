"""Durable, original-scale M1 review artifacts outside every Git worktree."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .m1 import M1Run, M1_REQUIRED_GATE_IDS, canonical_radial_power_profile
from .report import stable_json_bytes, validate_artifact_directory


GALLERY_SCHEMA = "tat_truecolor_m1_gallery/v2"


@dataclass(frozen=True, slots=True)
class GalleryArtifact:
    filename: str
    sha256: str
    bytes: int
    media_type: str
    width: int | None
    height: int | None
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "description": self.description,
        }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_rendered_raster_bindings(run: M1Run) -> None:
    """Bind gallery pixels to the exact rasters measured by canonical gates."""

    if run.canonical is None or run.rendered is None:
        raise RuntimeError("canonical real M1 evidence is required for the gallery")
    bindings = (
        (
            "production_nearest",
            run.rendered.production_nearest,
            run.canonical.production_nearest_raster_sha256,
        ),
        (
            "explicit_hanning",
            run.rendered.explicit_hanning,
            run.canonical.explicit_hanning_raster_sha256,
        ),
        (
            "bandlimited_reference",
            run.rendered.bandlimited_reference,
            run.canonical.bandlimited_reference_raster_sha256,
        ),
    )
    for name, raster, expected_sha256 in bindings:
        if not isinstance(raster, np.ndarray):
            raise RuntimeError(f"{name} gallery raster is not a NumPy array")
        if raster.dtype != np.uint8 or raster.ndim != 3 or raster.shape[-1] != 3:
            raise RuntimeError(
                f"{name} gallery raster must be uint8 HxWx3, got "
                f"dtype={raster.dtype}, shape={raster.shape}"
            )
        actual_sha256 = _sha256(raster.tobytes(order="C"))
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{name} gallery raster SHA-256 does not match canonical evidence: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )


def _atomic_new_file(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite M1 review artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _png_bytes(array: np.ndarray) -> bytes:
    from PIL import Image

    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _full_scale_sheet_bytes(
    nearest: np.ndarray,
    hanning: np.ndarray,
    reference: np.ndarray,
) -> tuple[bytes, tuple[int, int]]:
    """Concatenate three untouched 1188px panels beneath a label strip."""

    from PIL import Image, ImageDraw

    panel_width = 1188
    panel_height = 1188
    label_height = 60
    sheet = Image.new("RGB", (panel_width * 3, panel_height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    panels = (
        ("Production nearest (M0)", nearest),
        ("Explicit Hanning (M1)", hanning),
        ("Band-limited exact-area reference", reference),
    )
    for index, (label, array) in enumerate(panels):
        left = index * panel_width
        draw.text((left + 16, 20), label, fill="black")
        sheet.paste(Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB"), (left, label_height))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue(), sheet.size


def _signed_luma_error(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return (
        0.299 * difference[..., 0]
        + 0.587 * difference[..., 1]
        + 0.114 * difference[..., 2]
    )


def _signed_error_map(error: np.ndarray, *, scale: float) -> np.ndarray:
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("signed-error scale must be positive and finite")
    per_pixel = np.asarray(error, dtype=np.float64)
    if per_pixel.ndim != 2:
        raise ValueError("signed-error map expects a two-dimensional luma error")
    normalized = np.clip(per_pixel / scale, -1.0, 1.0)
    magnitude = np.abs(normalized)
    neutral = 245.0 * (1.0 - magnitude)
    red = neutral + np.clip(normalized, 0.0, 1.0) * 255.0
    blue = neutral + np.clip(-normalized, 0.0, 1.0) * 255.0
    green = neutral
    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(np.floor(rgb + 0.5), 0.0, 255.0).astype(np.uint8)


def _error_comparison_sheet_bytes(
    nearest_error_map: np.ndarray,
    hanning_error_map: np.ndarray,
    *,
    shared_scale: float,
) -> tuple[bytes, tuple[int, int]]:
    """Label two untouched error rasters above a shared signed-DN legend."""

    from PIL import Image, ImageDraw

    panel = 1188
    header = 80
    sheet = Image.new("RGB", (panel * 2, panel + header), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), "Production nearest signed Rec.601 luma error", fill="black")
    draw.text((panel + 16, 12), "Explicit Hanning signed Rec.601 luma error", fill="black")
    legend_left = panel - 180
    legend_right = panel + 180
    legend_top = 48
    for x in range(legend_left, legend_right):
        normalized = 2.0 * (x - legend_left) / (legend_right - legend_left - 1) - 1.0
        magnitude = abs(normalized)
        neutral = 245.0 * (1.0 - magnitude)
        color = (
            int(np.clip(neutral + max(normalized, 0.0) * 255.0, 0, 255)),
            int(np.clip(neutral, 0, 255)),
            int(np.clip(neutral + max(-normalized, 0.0) * 255.0, 0, 255)),
        )
        draw.line((x, legend_top, x, legend_top + 12), fill=color)
    draw.text((legend_left - 56, legend_top), f"-{shared_scale:.2f}", fill="black")
    draw.text((legend_right + 8, legend_top), f"+{shared_scale:.2f} DN", fill="black")
    sheet.paste(Image.fromarray(nearest_error_map, mode="RGB"), (0, header))
    sheet.paste(Image.fromarray(hanning_error_map, mode="RGB"), (panel, header))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue(), sheet.size


def _signed_error_summary_csv_bytes(
    nearest_error: np.ndarray,
    hanning_error: np.ndarray,
    *,
    shared_scale: float,
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "path",
            "shared_symmetric_scale_dn_p99_9_max",
            "mean_signed_dn",
            "rms_dn",
            "min_dn",
            "max_dn",
            "p99_abs_dn",
            "p99_9_abs_dn",
            "negative_fraction",
            "zero_fraction",
            "positive_fraction",
        ]
    )
    for name, error in (
        ("production_nearest", nearest_error),
        ("explicit_hanning", hanning_error),
    ):
        values = np.asarray(error, dtype=np.float64)
        absolute = np.abs(values)
        writer.writerow(
            [
                name,
                f"{shared_scale:.12g}",
                f"{np.mean(values):.12g}",
                f"{np.sqrt(np.mean(np.square(values))):.12g}",
                f"{np.min(values):.12g}",
                f"{np.max(values):.12g}",
                f"{np.percentile(absolute, 99):.12g}",
                f"{np.percentile(absolute, 99.9):.12g}",
                f"{np.mean(values < 0):.12g}",
                f"{np.mean(values == 0):.12g}",
                f"{np.mean(values > 0):.12g}",
            ]
        )
    return buffer.getvalue().encode("ascii")


def _signed_error_values_npy_bytes(
    nearest_error: np.ndarray,
    hanning_error: np.ndarray,
) -> bytes:
    values = np.stack([nearest_error, hanning_error]).astype(np.float32)
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def radial_power_profile(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The exact canonical V03 profile used by the gate and review artifacts."""

    return canonical_radial_power_profile(values, bins=64)


def _spectral_plot_bytes(
    nearest: np.ndarray,
    hanning: np.ndarray,
    reference: np.ndarray,
    *,
    nearest_ratio: float,
    hanning_ratio: float,
) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12.0, 7.0), dpi=150)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    for label, raster, color, width in (
        ("Production nearest", nearest, "#c1121f", 1.0),
        ("Explicit Hanning", hanning, "#2364aa", 1.8),
        ("Exact-area reference", reference, "#2a9d8f", 2.2),
    ):
        frequency, power = radial_power_profile(raster)
        axes.semilogy(frequency, np.maximum(power, 1e-12), label=label, color=color, lw=width)
    axes.axvline(0.3, color="#333333", linestyle="--", linewidth=1.0)
    axes.text(0.305, axes.get_ylim()[1] / 4.0, "V03 integration begins", fontsize=9)
    axes.set_xlim(0.0, 0.5)
    axes.set_xlabel("Radial spatial frequency (cycles / delivered pixel)")
    axes.set_ylabel("Radially averaged luma power")
    axes.set_title(
        "M1 canonical V03 spectrum — Rec.601 luma, 2-D Hann, 64 bins\n"
        f">0.3 cyc/px ratio: nearest {nearest_ratio:.4f}×, "
        f"Hanning {hanning_ratio:.4f}×"
    )
    axes.grid(True, alpha=0.2)
    axes.legend(frameon=False)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    return buffer.getvalue()


def _spectral_csv_bytes(
    nearest: np.ndarray,
    hanning: np.ndarray,
    reference: np.ndarray,
) -> bytes:
    profiles = [radial_power_profile(value) for value in (nearest, hanning, reference)]
    frequencies = profiles[0][0]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "radial_frequency_cycles_per_delivered_pixel",
            "production_nearest_luma_power",
            "explicit_hanning_luma_power",
            "exact_area_reference_luma_power",
        ]
    )
    for index, frequency in enumerate(frequencies):
        writer.writerow(
            [
                f"{frequency:.12g}",
                f"{profiles[0][1][index]:.12g}",
                f"{profiles[1][1][index]:.12g}",
                f"{profiles[2][1][index]:.12g}",
            ]
        )
    return buffer.getvalue().encode("ascii")


def _artifact(
    directory: Path,
    filename: str,
    payload: bytes,
    *,
    media_type: str,
    width: int | None,
    height: int | None,
    description: str,
) -> GalleryArtifact:
    path = directory / filename
    _atomic_new_file(path, payload)
    return GalleryArtifact(
        filename=filename,
        sha256=_sha256(payload),
        bytes=len(payload),
        media_type=media_type,
        width=width,
        height=height,
        description=description,
    )


def write_m1_gallery_images(
    run: M1Run,
    artifact_directory: str | os.PathLike[str] | Path,
) -> dict[str, dict[str, Any]]:
    """Write original-scale comparisons but not the final manifest/report."""

    _validate_rendered_raster_bindings(run)
    assert run.canonical is not None and run.rendered is not None
    directory = validate_artifact_directory(artifact_directory)
    directory.mkdir(parents=True, exist_ok=True)
    nearest = run.rendered.production_nearest
    hanning = run.rendered.explicit_hanning
    reference = run.rendered.bandlimited_reference
    if not (nearest.shape == hanning.shape == reference.shape == (1188, 1188, 3)):
        raise RuntimeError("M1 original-scale gallery rasters are not 1188x1188 RGB")

    nearest_luma_error = _signed_luma_error(nearest, reference)
    hanning_luma_error = _signed_luma_error(hanning, reference)
    shared_error_scale = max(
        1.0,
        float(np.percentile(np.abs(nearest_luma_error), 99.9)),
        float(np.percentile(np.abs(hanning_luma_error), 99.9)),
    )
    nearest_error = _signed_error_map(
        nearest_luma_error,
        scale=shared_error_scale,
    )
    hanning_error = _signed_error_map(
        hanning_luma_error,
        scale=shared_error_scale,
    )
    artifacts: list[GalleryArtifact] = []
    for filename, raster, description in (
        (
            "01_production_nearest_1188px.png",
            nearest,
            "M0 production nearest-neighbour map raster, exact 100% delivered scale.",
        ),
        (
            "02_explicit_hanning_1188px.png",
            hanning,
            "M1 explicit-Hanning map raster, exact 100% delivered scale.",
        ),
        (
            "03_bandlimited_reference_1188px.png",
            reference,
            "Exact-area same-display-array reference on the captured Agg internal grid, then the identity-proved M0 canvas slice; exact 100% scale.",
        ),
        (
            "04_production_nearest_error_1188px.png",
            nearest_error,
            f"Signed Rec.601 luma error versus exact-area reference; red positive, blue negative, shared symmetric scale {shared_error_scale:.3f} DN.",
        ),
        (
            "05_explicit_hanning_error_1188px.png",
            hanning_error,
            f"Signed Rec.601 luma error versus exact-area reference; red positive, blue negative, shared symmetric scale {shared_error_scale:.3f} DN.",
        ),
    ):
        artifacts.append(
            _artifact(
                directory,
                filename,
                _png_bytes(raster),
                media_type="image/png",
                width=1188,
                height=1188,
                description=description,
            )
        )

    error_sheet, error_sheet_dimensions = _error_comparison_sheet_bytes(
        nearest_error,
        hanning_error,
        shared_scale=shared_error_scale,
    )
    artifacts.append(
        _artifact(
            directory,
            "09_error_comparison_shared_scale_100pct.png",
            error_sheet,
            media_type="image/png",
            width=error_sheet_dimensions[0],
            height=error_sheet_dimensions[1],
            description=(
                "Labeled signed Rec.601 error comparison and shared legend; "
                "both 1188x1188 raster regions are embedded pixel-for-pixel."
            ),
        )
    )
    artifacts.append(
        _artifact(
            directory,
            "10_signed_luma_error_summary.csv",
            _signed_error_summary_csv_bytes(
                nearest_luma_error,
                hanning_luma_error,
                shared_scale=shared_error_scale,
            ),
            media_type="text/csv",
            width=None,
            height=None,
            description=(
                "Machine-readable signed Rec.601 luma error statistics using "
                "the gallery's one shared symmetric scale."
            ),
        )
    )
    artifacts.append(
        _artifact(
            directory,
            "11_signed_luma_error_values_float32.npy",
            _signed_error_values_npy_bytes(
                nearest_luma_error,
                hanning_luma_error,
            ),
            media_type="application/x-npy",
            width=1188,
            height=1188,
            description=(
                "Machine-readable float32 signed Rec.601 errors with shape "
                "[2,1188,1188], ordered production-nearest then explicit-Hanning."
            ),
        )
    )

    spectrum = _spectral_plot_bytes(
        nearest,
        hanning,
        reference,
        nearest_ratio=run.canonical.production_nearest_high_frequency_ratio,
        hanning_ratio=run.canonical.explicit_hanning_high_frequency_ratio,
    )
    from PIL import Image

    with Image.open(io.BytesIO(spectrum)) as spectral_image:
        spectral_dimensions = spectral_image.size
    artifacts.append(
        _artifact(
            directory,
            "06_spectral_comparison.png",
            spectrum,
            media_type="image/png",
            width=spectral_dimensions[0],
            height=spectral_dimensions[1],
            description=(
                "Canonical V03 Rec.601/2-D-Hann/64-bin spectra and cutoff for "
                "all three delivered rasters."
            ),
        )
    )
    full_scale_sheet, sheet_dimensions = _full_scale_sheet_bytes(
        nearest,
        hanning,
        reference,
    )
    artifacts.append(
        _artifact(
            directory,
            "08_kernel_before_hanning_reference_sheet_100pct.png",
            full_scale_sheet,
            media_type="image/png",
            width=sheet_dimensions[0],
            height=sheet_dimensions[1],
            description=(
                "Labeled before/Hanning/reference sheet; each 1188x1188 panel is "
                "embedded pixel-for-pixel with no downscale."
            ),
        )
    )
    artifacts.append(
        _artifact(
            directory,
            "07_spectral_comparison.csv",
            _spectral_csv_bytes(nearest, hanning, reference),
            media_type="text/csv",
            width=None,
            height=None,
            description=(
                "Machine-readable canonical V03 Rec.601 luma power profiles "
                "underlying the spectral plot."
            ),
        )
    )

    gate_outcomes = {gate.gate_id: gate.outcome for gate in run.gates}
    outcome_lines = "\n".join(
        f"- {gate_id}: {gate_outcomes.get(gate_id, 'MISSING')}"
        for gate_id in sorted(M1_REQUIRED_GATE_IDS | gate_outcomes.keys())
    )
    if run.all_thresholds_passed:
        scientific_status = "All assigned M1 thresholds PASS."
    elif run.stage_advancement_passed:
        failures = ", ".join(run.threshold_fail_gate_ids) or "none"
        scientific_status = (
            f"M1 partial progress; {failures} remains FAIL and M2 closure is "
            "required. The research-defined M1 partial-progress rule permits "
            "stage advancement but does not convert that threshold to PASS."
        )
    else:
        remaining = tuple(
            gate.gate_id for gate in run.gates if gate.blocking
        ) or run.unmeasured_gate_ids
        scientific_status = (
            "M1 is blocked; unresolved blocking/unmeasured gates: "
            + (", ".join(remaining) if remaining else "gate-set contract invalid")
            + "."
        )
    review = (
        "# M1 visual-review instructions\n\n"
        f"Scientific status at publication: {scientific_status}\n\n"
        "Andrew verdict: PENDING. Visual acceptance cannot close or re-label a "
        "scientific gate. This file and the gallery manifest are immutable "
        "members of the checksummed evidence set; do not edit them.\n\n"
        "M1 is only the explicit-kernel stopgap. Even when every assigned M1 "
        "threshold passes, it does not repair upstream striding, inter-band "
        "co-registration, native-resolution caps, or display-domain spatial "
        "averaging; M2 and later staged work remain required.\n\n"
        "Review the three 1188 px originals at 100% scale and confirm storm "
        "structure, cirrus texture, coastline appearance, and absence of a new "
        "ringing or softness concern.\n\n"
        "After publication, create a separate append-only file named "
        "`ANDREW_VERDICT_<UTC_TIMESTAMP>.md`. Record ACCEPT / REJECT / REVISE, "
        "comments, reviewer identity/time, and the immutable "
        "`gallery_manifest.json` SHA-256 obtained with "
        "`sha256sum gallery_manifest.json`. Never alter the manifest or root "
        "checksum index to insert the verdict.\n\n"
        "Gate outcomes:\n\n"
        f"{outcome_lines}\n"
    ).encode("utf-8")
    artifacts.append(
        _artifact(
            directory,
            "REVIEW_INSTRUCTIONS.md",
            review,
            media_type="text/markdown",
            width=None,
            height=None,
            description=(
                "Immutable instructions for a separate append-only Andrew verdict; "
                "the verdict is intentionally pending."
            ),
        )
    )
    readme = (
        "# M1 kernel-correction review group\n\n"
        f"Scientific status: {scientific_status}\n\n"
        "Andrew visual verdict: PENDING. Visual acceptance cannot close a "
        "scientific gate; follow REVIEW_INSTRUCTIONS.md and do not edit any "
        "checksummed member.\n\n"
        "M1 is only the explicit-Hanning stopgap. It does not repair upstream "
        "striding, inter-band co-registration, native-resolution caps, or "
        "display-domain spatial averaging; M2 and later stages remain required "
        "even if all M1-assigned thresholds pass.\n\n"
        "Integrity convention: `SHA256SUMS` lists every immutable payload file "
        "except the two checksum-index files; `SHA256SUMS.sha256` binds "
        "`SHA256SUMS`. Verify both with `sha256sum -c`. A later supplemental "
        "Andrew verdict is intentionally not inserted into or allowed to mutate "
        "this immutable run.\n\n"
        "Gate outcomes:\n\n"
        f"{outcome_lines}\n\n"
        "All raster comparisons are extracted map pixels, not the 1320 px figure "
        "with chrome, and are stored at the exact 1188 x 1188 delivered scale. "
        "No image in this directory is a downscaled substitute.\n\n"
        f"Reference convention: {run.canonical.reference_convention}\n\n"
        f"Captured Agg internal shape: {run.canonical.agg_internal_shape}\n\n"
        f"Delivered internal pixel bounds: {run.canonical.delivered_internal_pixel_bounds}\n\n"
        f"Reference mapping scored identity: {run.canonical.reference_identity_scored:.9f}\n\n"
        f"Production nearest RMS: {run.canonical.production_nearest_rms_dn:.6f} DN\n\n"
        f"Explicit Hanning RMS: {run.canonical.explicit_hanning_rms_dn:.6f} DN\n\n"
        f"Nearest >0.3 cyc/px ratio: {run.canonical.production_nearest_high_frequency_ratio:.6f}x\n\n"
        f"Hanning >0.3 cyc/px ratio: {run.canonical.explicit_hanning_high_frequency_ratio:.6f}x\n"
    ).encode("utf-8")
    artifacts.append(
        _artifact(
            directory,
            "README.md",
            readme,
            media_type="text/markdown",
            width=None,
            height=None,
            description="Review-group index and exact measurement convention.",
        )
    )
    return {item.filename: item.as_dict() for item in artifacts}


def write_m1_gallery_manifest(
    run: M1Run,
    artifact_directory: str | os.PathLike[str] | Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    report_filename: str,
    report_sha256: str,
) -> GalleryArtifact:
    if run.canonical is None:
        raise RuntimeError("canonical evidence is required for the gallery manifest")
    _validate_rendered_raster_bindings(run)
    assert run.rendered is not None
    directory = validate_artifact_directory(artifact_directory)
    if (
        not report_filename
        or Path(report_filename).name != report_filename
        or "/" in report_filename
        or "\\" in report_filename
    ):
        raise ValueError(f"unsafe M1 report filename: {report_filename!r}")
    report_path = directory / report_filename
    if not report_path.is_file():
        raise FileNotFoundError(f"M1 report is absent before manifest binding: {report_path}")
    actual_report_sha256 = _sha256(report_path.read_bytes())
    if actual_report_sha256 != report_sha256:
        raise RuntimeError(
            "M1 report SHA-256 does not match manifest input: "
            f"expected {report_sha256}, got {actual_report_sha256}"
        )
    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"M1 report is not valid JSON: {report_path}") from exc
    declared_artifacts = dict(artifacts)
    if not declared_artifacts:
        raise RuntimeError("M1 manifest refuses an empty gallery artifact set")
    if report_payload.get("gallery_artifacts") != declared_artifacts:
        raise RuntimeError(
            "M1 report gallery_artifacts do not exactly match manifest input"
        )
    for artifact_name, declaration in declared_artifacts.items():
        if (
            not artifact_name
            or Path(artifact_name).name != artifact_name
            or "/" in artifact_name
            or "\\" in artifact_name
            or declaration.get("filename") != artifact_name
        ):
            raise ValueError(f"unsafe or inconsistent gallery artifact: {artifact_name!r}")
        artifact_path = directory / artifact_name
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"declared M1 gallery artifact is absent: {artifact_path}"
            )
        payload = artifact_path.read_bytes()
        if declaration.get("bytes") != len(payload):
            raise RuntimeError(
                f"declared byte count does not match gallery artifact {artifact_name}"
            )
        if declaration.get("sha256") != _sha256(payload):
            raise RuntimeError(
                f"declared SHA-256 does not match gallery artifact {artifact_name}"
            )
    nearest_luma_error = _signed_luma_error(
        run.rendered.production_nearest,
        run.rendered.bandlimited_reference,
    )
    hanning_luma_error = _signed_luma_error(
        run.rendered.explicit_hanning,
        run.rendered.bandlimited_reference,
    )
    shared_error_scale = max(
        1.0,
        float(np.percentile(np.abs(nearest_luma_error), 99.9)),
        float(np.percentile(np.abs(hanning_luma_error), 99.9)),
    )
    manifest = {
        "schema": GALLERY_SCHEMA,
        "scene_id": run.canonical.scene_id,
        "reference_convention": run.canonical.reference_convention,
        "source_shape": run.canonical.source_shape,
        "figure_shape": run.canonical.figure_shape,
        "map_pixel_bounds": run.canonical.map_pixel_bounds,
        "delivered_shape": run.canonical.delivered_shape,
        "sampling_model": run.canonical.sampling_model,
        "sampling_phase": run.canonical.sampling_phase,
        "agg_internal_shape": run.canonical.agg_internal_shape,
        "agg_transform_matrix": run.canonical.agg_transform_matrix,
        "agg_make_image_geometry": run.canonical.agg_make_image_geometry,
        "delivered_internal_pixel_bounds": (
            run.canonical.delivered_internal_pixel_bounds
        ),
        "internal_mapping_candidate_scores": (
            run.canonical.internal_mapping_candidate_scores
        ),
        "reference_identity_scored": run.canonical.reference_identity_scored,
        "gate_outcomes": {
            gate.gate_id: gate.outcome for gate in run.gates
        },
        "stage_advancement_passed": run.stage_advancement_passed,
        "all_thresholds_passed": run.all_thresholds_passed,
        "threshold_fail_gate_ids": list(run.threshold_fail_gate_ids),
        "error_map_encoding": {
            "quantity": "signed Rec.601 luma error in DN (candidate - reference)",
            "rec601_coefficients_rgb": [0.299, 0.587, 0.114],
            "shared_scale_percentile": 99.9,
            "shared_symmetric_limit_dn": shared_error_scale,
            "shared_scale_definition": (
                "max(1 DN, nearest absolute-error p99.9, Hanning "
                "absolute-error p99.9)"
            ),
            "color_mapping": (
                "zero=RGB(245,245,245); positive=red; negative=blue; "
                "linear magnitude through the shared limit, then clipped"
            ),
            "individual_error_raster_bounds_xyxy": [0, 0, 1188, 1188],
            "comparison_sheet": {
                "artifact": "09_error_comparison_shared_scale_100pct.png",
                "nearest_raster_bounds_xyxy": [0, 80, 1188, 1268],
                "hanning_raster_bounds_xyxy": [1188, 80, 2376, 1268],
                "legend_location": "header outside both raster regions",
            },
            "raw_values_artifact": "11_signed_luma_error_values_float32.npy",
            "raw_values_order": ["production_nearest", "explicit_hanning"],
        },
        "report": {"filename": report_filename, "sha256": report_sha256},
        "artifacts": dict(artifacts),
        "andrew_review": {
            "verdict": "PENDING",
            "instructions": "REVIEW_INSTRUCTIONS.md",
            "supplement_pattern": "ANDREW_VERDICT_<UTC_TIMESTAMP>.md",
            "supplement_must_bind_gallery_manifest_sha256": True,
            "manifest_is_immutable": True,
        },
    }
    payload = stable_json_bytes(manifest)
    return _artifact(
        directory,
        "gallery_manifest.json",
        payload,
        media_type="application/json",
        width=None,
        height=None,
        description="Checksummed M1 gallery manifest bound to the validation report.",
    )
