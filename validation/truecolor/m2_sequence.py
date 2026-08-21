"""Offline, mutation-resistant V30 production sequence capture.

The acquisition package is only an input manifest.  This module verifies its
bytes, invokes the real ABI physical loader and M2 renderer for every same-scan
four-band group, and constructs the timestamped array binding consumed by the
M2 oracle.  No network resolver or nearest-scan fallback is reachable here.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import tempfile
import types
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np


V30_CAPTURE_SCHEMA: Final = "tat_truecolor_m2_v30_production_capture/v1"
V30_BINDING_SCHEMA: Final = "tat_truecolor_m2_timestamped_sequence/v2"
V30_CANONICAL_MANIFEST_SHA256: Final = (
    "432e1c7974fb7dd47467739b60d56eb183a4ca652ea71fc60de5304b323fd12e"
)
V30_CANONICAL_ACQUISITION_INDEX_SHA256: Final = (
    "10fa14933bc4afefab139e03569196f415f5128e5d09e794448a5f0f3fddcc38"
)
V30_NATURAL_EARTH_ZIP_SHA256: Final = (
    "e547d749445eaa0964aba76738090ec88f5e63c4585122170f98c67a7ea922dc"
)
V30_BBOX: Final = (-91.0, 31.2, -87.0, 35.2)
V30_PATCH_WORLD_BOUNDS: Final = (
    -90.158249,
    31.327946,
    -89.835017,
    31.651178,
)
V30_DPI: Final = 110
V30_PATCH_EDGE_BINDING_TOLERANCE_PX: Final = 1.0e-3
V30_ROLE_BY_CHANNEL: Final = {
    "C01": "blue",
    "C02": "red",
    "C03": "veggie",
    "C13": "ir",
}
V30_ROLE_ORDER: Final = ("red", "blue", "veggie", "ir")


def _v30_logical_s3_key(manifest_key: str) -> str:
    """Bind an unprefixed acquisition key to the exact NOAA bucket."""

    if (
        type(manifest_key) is not str
        or not manifest_key.startswith("ABI-L2-CMIPM/")
        or Path(manifest_key).is_absolute()
        or ".." in Path(manifest_key).parts
        or any(not part for part in manifest_key.split("/"))
    ):
        raise RuntimeError("V30 manifest object key is not in the bound NOAA namespace")
    return f"noaa-goes19/{manifest_key}"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parse_utc(value: str) -> dt.datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f"timestamp is not explicit UTC: {value!r}")
    return parsed.astimezone(dt.timezone.utc)


def _parse_checksum_index(path: Path) -> Mapping[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"invalid checksum row in {path}: {line!r}")
        digest, raw_name = fields
        name = raw_name.lstrip(" *")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name
            or Path(name).name != name
            or name in rows
        ):
            raise RuntimeError(f"unsafe checksum row in {path}: {line!r}")
        rows[name] = digest
    if not rows:
        raise RuntimeError(f"empty checksum index: {path}")
    return rows


@dataclass(frozen=True, slots=True)
class V30Object:
    nominal_slot_utc: str
    exact_scan_start_utc: str
    exact_scan_end_utc: str
    date_created_utc: str
    channel: str
    role: str
    key: str
    filename: str
    bytes: int
    sha256: str
    band_id: int

    def as_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class V30Corpus:
    acquisition_directory: Path
    raw_directory: Path
    manifest_path: Path
    land_zip_path: Path
    objects: tuple[V30Object, ...]
    acquisition_index_sha256: str
    manifest_sha256: str
    land_zip_sha256: str
    land_binding: Mapping[str, Any]

    @property
    def groups(self) -> tuple[tuple[V30Object, ...], ...]:
        by_slot: dict[str, dict[str, V30Object]] = {}
        for item in self.objects:
            by_slot.setdefault(item.nominal_slot_utc, {})[item.role] = item
        return tuple(
            tuple(by_slot[slot][role] for role in V30_ROLE_ORDER)
            for slot in sorted(by_slot, key=_parse_utc)
        )

    def summary_dict(self) -> Mapping[str, Any]:
        return {
            "acquisition_directory": str(self.acquisition_directory),
            "raw_directory": str(self.raw_directory),
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
            },
            "acquisition_index_sha256": self.acquisition_index_sha256,
            "land_zip": {
                "path": str(self.land_zip_path),
                "sha256": self.land_zip_sha256,
            },
            "land_binding": dict(self.land_binding),
            "object_count": len(self.objects),
            "group_count": len(self.groups),
            "total_bytes": sum(item.bytes for item in self.objects),
            "inventory_sha256": _sha256_bytes(
                "".join(
                    f"{item.sha256}  {item.filename}\n"
                    for item in sorted(self.objects, key=lambda row: row.filename)
                ).encode("ascii")
            ),
        }


@dataclass(frozen=True, slots=True)
class V30SequenceCapture:
    schema: str
    binding: Any
    corpus: V30Corpus
    frame_provenance: tuple[Mapping[str, Any], ...]
    screening: tuple[Mapping[str, Any], ...]
    request_identity_sha256: str
    grid_identity_sha256: str
    raster_sha256: tuple[str, ...]
    frozen_output_mutant_killed: bool

    def summary_dict(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "corpus": self.corpus.summary_dict(),
            "request_identity_sha256": self.request_identity_sha256,
            "grid_identity_sha256": self.grid_identity_sha256,
            "raster_sha256": list(self.raster_sha256),
            "distinct_raster_count": len(set(self.raster_sha256)),
            "frozen_output_mutant_killed": self.frozen_output_mutant_killed,
            "patch_bounds": list(self.binding.patch_bounds),
            "stationary_world_bounds": list(self.binding.stationary_world_bounds),
            "stationary_class": self.binding.stationary_class,
            "stationarity_method": self.binding.stationarity_method,
            "frames": [dict(item) for item in self.frame_provenance],
            "screening": [dict(item) for item in self.screening],
        }


def _verify_acquisition_directory(root: Path) -> tuple[Path, Mapping[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"V30 acquisition root is not a real directory: {root}")
    checksum_path = root / "SHA256SUMS"
    if _sha256_file(checksum_path) != V30_CANONICAL_ACQUISITION_INDEX_SHA256:
        raise RuntimeError("V30 acquisition checksum-index identity changed")
    rows = _parse_checksum_index(checksum_path)
    actual_files = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(rows) != actual_files:
        raise RuntimeError("V30 acquisition checksum index is incomplete or overbroad")
    for name, expected in rows.items():
        path = root / name
        if path.is_symlink() or _sha256_file(path) != expected:
            raise RuntimeError(f"V30 acquisition payload failed SHA-256: {name}")
    summary = json.loads((root / "ACQUISITION_SUMMARY.json").read_text("utf-8"))
    if summary.get("status") != "ACQUISITION_COMPLETE_CANDIDATE_ONLY_NOT_V30_EVIDENCE":
        raise RuntimeError("V30 acquisition summary overclaims or changed status")
    return root / "alternative_m2_objects.csv", summary


def _verify_land_patch(zip_path: Path) -> Mapping[str, Any]:
    if not zip_path.is_file() or zip_path.is_symlink():
        raise RuntimeError("Natural Earth input must be a regular ZIP file")
    digest = _sha256_file(zip_path)
    if digest != V30_NATURAL_EARTH_ZIP_SHA256:
        raise RuntimeError("Natural Earth land ZIP identity changed")
    from shapely.geometry import box
    from shapely.ops import unary_union
    from cartopy.io import shapereader

    temporary = Path(tempfile.mkdtemp(prefix="tat-m2-v30-land-"))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            required = {
                "ne_10m_land.shp",
                "ne_10m_land.shx",
                "ne_10m_land.dbf",
            }
            if not required <= names or any(
                Path(name).name != name or Path(name).is_absolute() for name in names
            ):
                raise RuntimeError("Natural Earth ZIP layout is unsafe or incomplete")
            for name in required | ({"ne_10m_land.prj"} & names):
                (temporary / name).write_bytes(archive.read(name))
        geometries = tuple(shapereader.Reader(temporary / "ne_10m_land.shp").geometries())
        land = unary_union(geometries)
        patch = box(*V30_PATCH_WORLD_BOUNDS)
        if not land.covers(patch):
            raise RuntimeError("predeclared V30 stationary patch is not fully land")
        distance = float(land.boundary.distance(patch))
        if not math.isfinite(distance) or distance < 0.5:
            raise RuntimeError("V30 patch is not safely interior to the land polygon")
        return {
            "dataset": "Natural Earth 1:10m land",
            "zip_sha256": digest,
            "patch_fully_covered": True,
            "distance_to_land_boundary_deg": distance,
            "minimum_required_distance_deg": 0.5,
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def verify_v30_corpus(
    acquisition_directory: str | os.PathLike[str] | Path,
    raw_directory: str | os.PathLike[str] | Path,
    land_zip_path: str | os.PathLike[str] | Path,
) -> V30Corpus:
    """Verify the canonical 19x4 acquisition without invoking the renderer."""

    acquisition = Path(acquisition_directory).expanduser().resolve(strict=True)
    raw = Path(raw_directory).expanduser().resolve(strict=True)
    land_zip = Path(land_zip_path).expanduser().resolve(strict=True)
    manifest, summary = _verify_acquisition_directory(acquisition)
    if _sha256_file(manifest) != V30_CANONICAL_MANIFEST_SHA256:
        raise RuntimeError("canonical V30 object manifest identity changed")
    if not raw.is_dir() or raw.is_symlink():
        raise RuntimeError("V30 raw corpus is not a real directory")
    if list(raw.glob("*.part")):
        raise RuntimeError("V30 raw corpus contains an interrupted .part file")

    objects: list[V30Object] = []
    seen_keys: set[str] = set()
    seen_names: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            channel = row["channel"]
            role = V30_ROLE_BY_CHANNEL.get(channel)
            if role is None:
                raise RuntimeError(f"unexpected V30 channel: {channel}")
            key = row["key"]
            filename = Path(key).name
            if Path(key).is_absolute() or ".." in Path(key).parts or not filename:
                raise RuntimeError(f"unsafe V30 object key: {key!r}")
            if key in seen_keys or filename in seen_names:
                raise RuntimeError("duplicate V30 key or local filename")
            seen_keys.add(key)
            seen_names.add(filename)
            expected_band = int(channel[1:])
            if (
                row["corpus"] != "alternative_m2_candidate"
                or row["sector"] != "M2"
                or row["platform_id"] != "G19"
                or row["scene_id"] != "Mesoscale"
                or int(row["band_id"]) != expected_band
                or int(row["catalog_size"]) != int(row["local_size"])
            ):
                raise RuntimeError(f"V30 catalog identity mismatch for {filename}")
            digest = row["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise RuntimeError(f"invalid V30 SHA-256 for {filename}")
            objects.append(
                V30Object(
                    nominal_slot_utc=row["nominal_slot_utc"],
                    exact_scan_start_utc=row["time_coverage_start"],
                    exact_scan_end_utc=row["time_coverage_end"],
                    date_created_utc=row["date_created"],
                    channel=channel,
                    role=role,
                    key=key,
                    filename=filename,
                    bytes=int(row["local_size"]),
                    sha256=digest,
                    band_id=expected_band,
                )
            )
    if len(objects) != 76:
        raise RuntimeError(f"V30 manifest must contain 76 objects, got {len(objects)}")
    actual_names = {path.name for path in raw.iterdir() if path.is_file()}
    if actual_names != seen_names or any(path.is_symlink() for path in raw.iterdir()):
        raise RuntimeError("V30 raw directory does not exactly match its manifest")
    for item in objects:
        path = raw / item.filename
        if path.stat().st_size != item.bytes or _sha256_file(path) != item.sha256:
            raise RuntimeError(f"V30 raw object failed identity: {item.filename}")

    by_slot: dict[str, list[V30Object]] = {}
    for item in objects:
        by_slot.setdefault(item.nominal_slot_utc, []).append(item)
    slots = sorted((_parse_utc(slot) for slot in by_slot), key=lambda value: value)
    expected_start = dt.datetime(2025, 10, 28, 14, tzinfo=dt.timezone.utc)
    expected_slots = [expected_start + index * dt.timedelta(minutes=10) for index in range(19)]
    if slots != expected_slots:
        raise RuntimeError("V30 nominal cadence is not the canonical 19x10-minute sequence")
    for slot, group in by_slot.items():
        if {item.role for item in group} != set(V30_ROLE_ORDER) or len(group) != 4:
            raise RuntimeError(f"V30 scan group is not exact four-band same-scan: {slot}")
        starts = {_parse_utc(item.exact_scan_start_utc) for item in group}
        if len(starts) != 1:
            raise RuntimeError(f"V30 group has mixed exact scan starts: {slot}")
        if next(iter(starts)) - _parse_utc(slot) != dt.timedelta(seconds=54.9):
            raise RuntimeError(f"V30 exact scan start differs from canonical +54.9s: {slot}")

    alternative = summary.get("alternative_m2", {})
    integrity = alternative.get("integrity", {})
    if (
        alternative.get("product") != "ABI-L2-CMIPM"
        or alternative.get("sector") != "M2"
        or alternative.get("platform") != "G19"
        or integrity.get("objects") != 76
        or integrity.get("scan_groups") != 19
        or integrity.get("bytes") != sum(item.bytes for item in objects)
    ):
        raise RuntimeError("V30 acquisition summary differs from verified objects")
    land_binding = _verify_land_patch(land_zip)
    return V30Corpus(
        acquisition_directory=acquisition,
        raw_directory=raw,
        manifest_path=manifest,
        land_zip_path=land_zip,
        objects=tuple(objects),
        acquisition_index_sha256=V30_CANONICAL_ACQUISITION_INDEX_SHA256,
        manifest_sha256=V30_CANONICAL_MANIFEST_SHA256,
        land_zip_sha256=V30_NATURAL_EARTH_ZIP_SHA256,
        land_binding=land_binding,
    )


def _plan_identity(plan: Any) -> str:
    payload = {
        "source_shape": list(plan.source_shape),
        "destination_shape": list(plan.destination_shape),
        "source_extent": list(plan.source_extent),
        "view_extent": list(plan.view_extent),
        "device_bounds": list(plan.device_bounds),
        "affine_matrix": [list(row) for row in plan.affine_matrix],
        "sampling_phase_yx": list(plan.sampling_phase_yx),
        "delivered_internal_bounds": list(plan.delivered_internal_bounds),
        "delivered_device_bounds": list(plan.delivered_device_bounds),
        "internal_to_canvas_flip_y": plan.internal_to_canvas_flip_y,
        "internal_to_canvas_flip_x": plan.internal_to_canvas_flip_x,
        "kernel": plan.kernel,
        "domain": plan.domain,
    }
    return _sha256_bytes(_stable_json_bytes(payload))


def _patch_edge_coordinates_from_plan(plan: Any) -> tuple[float, float, float, float]:
    if plan.delivered_shape is None:
        raise RuntimeError("V30 plan omitted delivered shape")
    height, width = (int(value) for value in plan.delivered_shape)
    left, right, bottom, top = (float(value) for value in plan.view_extent)
    west, south, east, north = V30_PATCH_WORLD_BOUNDS
    return (
        (top - north) / (top - bottom) * height,
        (top - south) / (top - bottom) * height,
        (west - left) / (right - left) * width,
        (east - left) / (right - left) * width,
    )


def _patch_bounds_from_plan(plan: Any) -> tuple[int, int, int, int]:
    coordinates = _patch_edge_coordinates_from_plan(plan)
    height, width = (int(value) for value in plan.delivered_shape)
    rounded = tuple(int(round(value)) for value in coordinates)
    if any(
        abs(value - integer) > V30_PATCH_EDGE_BINDING_TOLERANCE_PX
        for value, integer in zip(coordinates, rounded)
    ):
        raise RuntimeError(
            "V30 world patch does not bind to integer delivered pixel edges: "
            f"{coordinates!r}"
        )
    row_start, row_stop, column_start, column_stop = rounded
    if (
        row_stop - row_start < 96
        or column_stop - column_start < 96
        or not (0 <= row_start < row_stop <= height)
        or not (0 <= column_start < column_stop <= width)
    ):
        raise RuntimeError("V30 world patch does not select a valid >=96px region")
    return rounded  # type: ignore[return-value]


def _request_identity(delivered_shape: Sequence[int]) -> str:
    import render
    import render_policy

    payload = {
        "schema": V30_BINDING_SCHEMA,
        "bbox": list(V30_BBOX),
        "dpi": V30_DPI,
        "delivered_shape": [int(item) for item in delivered_shape],
        "channel": "true_color",
        "zone": 2,
        "coastlines": False,
        "gridlines": False,
        "background": render.DARK_BG,
        "policy": render_policy.M2_POLICY.algo_version,
        "product": "CMIPM2",
        "platform": "GOES-19",
        "stationary_world_bounds": list(V30_PATCH_WORLD_BOUNDS),
    }
    return _sha256_bytes(_stable_json_bytes(payload))


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    source = np.asarray(axis, dtype=np.float64)
    target = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size < 2 or np.any(~np.isfinite(source)):
        raise RuntimeError("V30 raw coordinate axis is invalid")
    increasing = source[-1] > source[0]
    ordered = source if increasing else source[::-1]
    flat = target.ravel()
    maximum_step = float(np.max(np.abs(np.diff(ordered))))
    if (
        not math.isfinite(maximum_step)
        or maximum_step <= 0.0
        or np.any(flat < ordered[0] - 0.5 * maximum_step)
        or np.any(flat > ordered[-1] + 0.5 * maximum_step)
    ):
        raise RuntimeError("V30 patch lies outside a raw coordinate axis")
    insertion = np.searchsorted(ordered, flat, side="left")
    insertion = np.clip(insertion, 1, ordered.size - 1)
    before = insertion - 1
    choose_after = np.abs(ordered[insertion] - flat) < np.abs(ordered[before] - flat)
    selected = np.where(choose_after, insertion, before)
    if not increasing:
        selected = source.size - 1 - selected
    return selected.reshape(target.shape)


def _screen_raw_patch(
    group: Sequence[V30Object],
    raw_directory: Path,
    plan: Any,
    patch_bounds: tuple[int, int, int, int],
) -> Mapping[str, Any]:
    import satellites
    import xarray as xr

    height, width = plan.delivered_shape
    row_start, row_stop, column_start, column_stop = patch_bounds
    left, right, bottom, top = (float(value) for value in plan.view_extent)
    columns = np.arange(column_start, column_stop, dtype=np.float64) + 0.5
    rows = np.arange(row_start, row_stop, dtype=np.float64) + 0.5
    lons = left + columns / float(width) * (right - left)
    lats = top - rows / float(height) * (top - bottom)
    patch_lons, patch_lats = np.meshgrid(lons, lats)
    dqf_zero: dict[str, float] = {}
    sampled_cmi: dict[str, np.ndarray] = {}
    for item in group:
        path = raw_directory / item.filename
        with xr.open_dataset(path, engine="h5netcdf", decode_times=False) as dataset:
            attrs = dataset.attrs
            if (
                attrs.get("dataset_name") != item.filename
                or attrs.get("platform_ID") != "G19"
                or attrs.get("scene_id") != "Mesoscale"
                or attrs.get("time_coverage_start") != item.exact_scan_start_utc
                or attrs.get("time_coverage_end") != item.exact_scan_end_utc
                or attrs.get("date_created") != item.date_created_utc
                or int(np.asarray(dataset["band_id"]).reshape(-1)[0]) != item.band_id
            ):
                raise RuntimeError(f"V30 NetCDF identity changed: {item.filename}")
            projection = dataset["goes_imager_projection"].attrs
            semi_major_axis = float(projection["semi_major_axis"])
            x_values, y_values = satellites._latlon_to_xy_visible(  # noqa: SLF001
                patch_lats,
                patch_lons,
                float(projection["longitude_of_projection_origin"]),
                float(projection["perspective_point_height"]) + semi_major_axis,
                semi_major_axis,
                float(projection["semi_minor_axis"]),
            )
            xi = _nearest_axis_indices(np.asarray(dataset["x"]), x_values)
            yi = _nearest_axis_indices(np.asarray(dataset["y"]), y_values)
            cmi = np.asarray(dataset["CMI"].values)[yi, xi]
            dqf = np.asarray(dataset["DQF"].values)[yi, xi]
            if cmi.shape != (row_stop - row_start, column_stop - column_start):
                raise RuntimeError("V30 raw patch screen shape drifted")
            dqf_zero[item.role] = float(np.mean(np.isfinite(dqf) & (dqf == 0)))
            sampled_cmi[item.role] = np.asarray(cmi, dtype=np.float32)
    if any(value != 1.0 for value in dqf_zero.values()):
        raise RuntimeError(f"V30 patch contains nonzero/invalid DQF: {dqf_zero}")
    red = sampled_cmi["red"]
    infrared = sampled_cmi["ir"]
    cloud_proxy = (red > np.float32(0.30)) | (infrared < np.float32(270.0))
    proxy_fraction = float(np.mean(cloud_proxy | ~np.isfinite(red) | ~np.isfinite(infrared)))
    warm_fraction = float(np.mean(np.isfinite(infrared) & (infrared > 280.0)))
    if proxy_fraction != 0.0 or warm_fraction != 1.0:
        raise RuntimeError("V30 predeclared clear-land patch failed raw screen")
    return {
        "nominal_slot_utc": group[0].nominal_slot_utc,
        "dqf_zero_fraction_by_role": dqf_zero,
        "cloud_proxy_fraction": proxy_fraction,
        "c13_above_280k_fraction": warm_fraction,
        "cloud_proxy_definition": "C02>0.30 or C13<270K; acquisition screen only",
        "selection_policy": "predeclared_before_renderer_metric_evaluation",
    }


def capture_v30_sequence(corpus: V30Corpus) -> V30SequenceCapture:
    """Run all 19 verified frames through the real physical loader/renderer."""

    import render_resample
    import satellites
    import truecolor
    from validation.truecolor import m2_production

    if m2_production.M2_SEQUENCE_SCHEMA != V30_BINDING_SCHEMA:
        raise RuntimeError("V30 producer/evaluator sequence schemas differ")

    frames: list[Any] = []
    provenance: list[Mapping[str, Any]] = []
    screens: list[Mapping[str, Any]] = []
    plan_identities: list[str] = []
    request_identities: list[str] = []
    raster_hashes: list[str] = []
    patch_bounds: tuple[int, int, int, int] | None = None

    for group in corpus.groups:
        by_role = {item.role: item for item in group}
        logical_key_by_role = {
            role: _v30_logical_s3_key(item.key) for role, item in by_role.items()
        }
        item_by_key = {
            logical_key_by_role[role]: item for role, item in by_role.items()
        }
        local_by_key = {
            logical_key_by_role[role]: corpus.raw_directory / item.filename
            for role, item in by_role.items()
        }
        satellite = satellites.GOESEastSatellite()

        def open_local(self, resolved):
            item = item_by_key.get(resolved.s3_key)
            path = local_by_key.get(resolved.s3_key)
            if item is None or path is None:
                raise RuntimeError(f"production requested an unlisted V30 object: {resolved.s3_key}")
            if (
                resolved.bucket != "noaa-goes19"
                or resolved.product != "CMIPM2"
                or resolved.sat_name != "GOES-19"
                or float(resolved.sub_sat_lon) != -75.2
                or Path(resolved.s3_key).name != item.filename
                or resolved.scan_start != satellites._parse_scan_start(item.filename)  # noqa: SLF001
            ):
                raise RuntimeError("production V30 ResolvedFile identity changed")
            dataset = satellites.xr.open_dataset(
                path, engine="h5netcdf", decode_times=False
            )
            cleanup = tempfile.mkdtemp(prefix="tat-m2-v30-local-open-")
            return dataset, cleanup

        satellite.open = types.MethodType(open_local, satellite)
        parsed_starts = {
            satellites._parse_scan_start(item.filename) for item in group  # noqa: SLF001
        }
        if len(parsed_starts) != 1:
            raise RuntimeError("V30 production filename starts are not same-scan")
        production_start = next(iter(parsed_starts))
        band_files = {
            role: satellites.ResolvedFile(
                bucket="noaa-goes19",
                s3_key=logical_key_by_role[role],
                product="CMIPM2",
                scan_start=production_start,
                sat_name="GOES-19",
                sub_sat_lon=-75.2,
            )
            for role in V30_ROLE_ORDER
        }
        source_manifest = satellites.TrueColorPhysicalSourceManifest(
            sensor="abi",
            bucket=band_files["red"].bucket,
            product=band_files["red"].product,
            scan_start=band_files["red"].scan_start,
            sat_name=band_files["red"].sat_name,
            resolved_anchor_s3_key=band_files["red"].s3_key,
            bbox=tuple(float(value) for value in V30_BBOX),
            sources=tuple(
                satellites.TrueColorPhysicalSource(
                    role=role,
                    band=band,
                    object_keys=(band_files[role].s3_key,),
                    complete_scan_object_keys=(band_files[role].s3_key,),
                )
                for role, band in satellite._physical_truecolor_role_bands()  # noqa: SLF001
            ),
        )
        result = satellite._compose_true_color_sync(  # noqa: SLF001
            band_files,
            list(V30_BBOX),
            band_files["red"],
            1,
            physical=True,
            physical_source_manifest=source_manifest,
        )
        state = result.truecolor_physical
        if (
            type(state) is not truecolor.TrueColorPhysicalState
            or result.cmi is not state.day_rgb_reflectance
            or result.units != "truecolor_physical"
            or state.sensor != "abi"
            or tuple(state.extent) != V30_BBOX
            or state.ir_bt_kelvin is None
            or state.ir_coverage is None
            or state.grid_metadata.get("point_stride_applied") is not False
            or "physical_source_cap" not in state.grid_metadata
            or "native_band_crops" not in state.grid_metadata
            or state.grid_metadata.get("physical_source_manifest")
            != source_manifest.as_dict()
        ):
            raise RuntimeError("V30 physical loader result is not fully bound")
        exact_start = _parse_utc(group[0].exact_scan_start_utc)
        title = (
            exact_start.strftime("%Y-%m-%d %H:%M:%S.%f")
            .rstrip("0")
            .rstrip(".")
            + " UTC"
        )
        capture = m2_production.capture_production_m2_render(
            result,
            bbox=V30_BBOX,
            dpi=V30_DPI,
            title_text=title,
        )
        if capture.trace_issues or not np.array_equal(
            capture.figure_image_rgb, capture.delivered_floor_rgb
        ):
            raise RuntimeError("V30 frame is not bound to the M2 renderer boundary")
        plan_identity = _plan_identity(capture.plan)
        if plan_identity != m2_production._plan_identity(capture.plan):  # noqa: SLF001
            raise RuntimeError("independent V30 plan identity disagrees with adapter")
        current_patch = _patch_bounds_from_plan(capture.plan)
        if patch_bounds is None:
            patch_bounds = current_patch
        elif current_patch != patch_bounds:
            raise RuntimeError("V30 world patch maps to different pixels across frames")
        delivered = capture.delivered_state
        daylight = (
            np.asarray(delivered.geom_valid, dtype=np.bool_)
            & (np.asarray(delivered.day_coverage) > np.float32(render_resample.DEFAULT_COVERAGE_THRESHOLD))
            & (np.asarray(delivered.day_weight) == np.float32(1.0))
        )
        if daylight.shape != capture.figure_image_rgb.shape[:2]:
            raise RuntimeError("V30 daylight mask is not delivered-grid bound")
        request_identity = _request_identity(capture.plan.delivered_shape)
        raster = np.asarray(capture.figure_image_rgb)
        raster_sha = m2_production._array_sha256(raster)  # noqa: SLF001
        png_sha = _sha256_bytes(capture.png)
        object_hashes = tuple(by_role[role].sha256 for role in V30_ROLE_ORDER)
        frames.append(
            m2_production.TimestampedRendererFrame(
                timestamp_utc=group[0].exact_scan_start_utc,
                raster_rgb_u8=raster.copy(),
                full_daylight_mask=daylight.copy(),
                request_identity_sha256=request_identity,
                grid_identity_sha256=plan_identity,
                source_object_sha256=object_hashes,
                renderer_png_sha256=png_sha,
                renderer_raster_sha256=raster_sha,
            )
        )
        screens.append(
            _screen_raw_patch(group, corpus.raw_directory, capture.plan, current_patch)
        )
        provenance.append(
            {
                "nominal_slot_utc": group[0].nominal_slot_utc,
                "exact_noaa_scan_start_utc": group[0].exact_scan_start_utc,
                "production_scan_start_utc": production_start.isoformat(),
                "source_objects": [by_role[role].as_dict() for role in V30_ROLE_ORDER],
                "request_identity_sha256": request_identity,
                "grid_identity_sha256": plan_identity,
                "renderer_png_sha256": png_sha,
                "renderer_raster_sha256": raster_sha,
                "trace_identity_sha256": capture.trace_identity_sha256,
                "delivered_shape": list(capture.plan.delivered_shape),
                "patch_bounds": list(current_patch),
                "world_patch_max_pixel_edge_residual": max(
                    abs(value - integer)
                    for value, integer in zip(
                        _patch_edge_coordinates_from_plan(capture.plan),
                        current_patch,
                    )
                ),
                "world_patch_pixel_edge_tolerance": (
                    V30_PATCH_EDGE_BINDING_TOLERANCE_PX
                ),
            }
        )
        plan_identities.append(plan_identity)
        request_identities.append(request_identity)
        raster_hashes.append(raster_sha)
        del result, state, capture, delivered, daylight, raster

    if patch_bounds is None or len(frames) != 19:
        raise RuntimeError("V30 renderer did not produce exactly 19 frames")
    if len(set(plan_identities)) != 1 or len(set(request_identities)) != 1:
        raise RuntimeError("V30 fixed request/grid identity changed across frames")
    source_groups = [tuple(frame.source_object_sha256) for frame in frames]
    if len(set(source_groups)) != 19 or len({item for group in source_groups for item in group}) != 76:
        raise RuntimeError("V30 source groups are duplicated or incomplete")
    frozen_killed = len(set(raster_hashes)) == len(raster_hashes)
    if not frozen_killed:
        raise RuntimeError("V30 frozen-output mutant was not discriminated")
    binding = m2_production.TimestampedSequenceBinding(
        schema=m2_production.M2_SEQUENCE_SCHEMA,
        frames=tuple(frames),
        patch_bounds=patch_bounds,
        stationary_world_bounds=V30_PATCH_WORLD_BOUNDS,
        stationary_class="clear_land",
        stationarity_method=m2_production.M2_STATIONARITY_METHOD,
        daylight_mask_producer_binding=(
            m2_production.M2_DAYLIGHT_MASK_PRODUCER_BINDING
        ),
    )
    return V30SequenceCapture(
        schema=V30_CAPTURE_SCHEMA,
        binding=binding,
        corpus=corpus,
        frame_provenance=tuple(provenance),
        screening=tuple(screens),
        request_identity_sha256=request_identities[0],
        grid_identity_sha256=plan_identities[0],
        raster_sha256=tuple(raster_hashes),
        frozen_output_mutant_killed=frozen_killed,
    )
