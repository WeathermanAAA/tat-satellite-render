"""High-level AHI loader: discover segments, fetch from S3, stitch full disk.

Architecture:

  ``load_band(bucket, sat_short, dt_floor10, band, fs)`` lists the band's
  segment objects under the AWS s3://noaa-himawari{8,9}/AHI-L1b-FLDK/...
  prefix for the given 10-minute slot, downloads each, decompresses bz2,
  parses with vendor.ahi_hsd, and stitches the per-segment count arrays
  back into a single full-disk array. Calibration (BT for IR/WV,
  reflectance for visible) is applied to the stitched array, not per
  segment, so the seam logic stays format-agnostic.

  Returns a CalibratedDisk dataclass holding:
    - the (n_lines, n_columns) calibrated array
    - the geos projection params (sub_lon, CFAC, LFAC, COFF, LOFF) so
      satellites.HimawariPacificSatellite can do the inverse projection
    - meta (sat name, band, scan time)

NOAA actually distributes two segment layouts in the same buckets:

  (A) Native FLDK 10-segment, file names ``..._S0110.DAT.bz2`` ...
      ``..._S1010.DAT.bz2``. Each segment is N_columns × (full_disk/10)
      rows. This is what noaa-himawari9 uses everywhere we've checked,
      and what noaa-himawari8 uses for recent dates.

  (B) NOAA-stitched single-segment, file name ``..._S0101.DAT.bz2``.
      One file per band per timestep, already containing the full disk.
      noaa-himawari8 uses (B) for older data (the Yutu 2018 / Hagibis
      2019 era).

The stitcher reads ``total_segments`` and ``first_line_number`` from each
segment's Block #7 and writes into the right rows of the full-disk array,
so both layouts work without any per-bucket branching.

Resolution selector
-------------------
Per HSD spec Table 3, AHI bands have native ground sampling distance:

  * Band  3      : 0.5 km  (R05 file suffix)
  * Bands 1,2,4  : 1.0 km  (R10)
  * Bands 5..16  : 2.0 km  (R20)

The R-suffix in the filename matches native; only one R-version per band
exists in the H-9 bucket (which is at-native), but H-8's older data has
multiple R-suffixes per band as resampled views. We pick the at-native R
for each band and ignore the resampled extras.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import s3fs

from .ahi_hsd import (
    HSDSegment,
    counts_to_brightness_temperature,
    counts_to_reflectance,
    decompress_bz2,
    parse_hsd_segment,
    COUNT_ERROR_PIXEL,
    COUNT_OUTSIDE_SCAN,
)

log = logging.getLogger("tat-satellite.ahi_loader")


# Native resolution suffix per band (per HSD spec Table 3)
BAND_RES_SUFFIX = {
    1: "R10", 2: "R10", 3: "R05", 4: "R10",
    5: "R20", 6: "R20", 7: "R20", 8: "R20", 9: "R20",
    10: "R20", 11: "R20", 12: "R20", 13: "R20", 14: "R20",
    15: "R20", 16: "R20",
}


@dataclass
class CalibratedDisk:
    """Calibrated AHI image (full disk OR a row-banded subset) + projection params.

    ``data`` may cover only a subset of the full disk's lines if the loader
    pre-filtered segments by bbox — in which case ``line_offset`` is the
    global (full-disk) 0-based line index of local row 0, and ``n_lines``
    is the local span. Columns always span the full disk.

    ``coff`` / ``loff`` / ``cfac`` / ``lfac`` are read off the segment
    Block #3 and ALWAYS describe the full disk's coordinate system, so
    forward projection produces global indices. Subtract ``line_offset``
    when indexing into ``data``.
    """

    sat_name: str            # "Himawari-8" / "Himawari-9"
    band_number: int
    central_wavelength_um: float

    sub_lon: float
    cfac: int
    lfac: int
    coff: float
    loff: float

    n_columns: int           # full-disk width
    n_lines: int             # local span of ``data`` (may be < full_lines)
    line_offset: int         # 0-based global line index of local row 0

    # Calibrated array, float32, shape (n_lines, n_columns).
    # IR/WV bands: brightness_temperature [K]. Visible/NIR bands: reflectance [0..1].
    data: np.ndarray
    units: str               # "K" or "1"

    obs_start_mjd: float


def _list_segments(
    fs: s3fs.S3FileSystem, bucket: str, dt_floor10, band: int
) -> list[str]:
    """List S3 object paths for a (bucket, time slot, band) at native resolution."""
    res_suffix = BAND_RES_SUFFIX[band]
    # Path: {bucket}/AHI-L1b-FLDK/{Y}/{m}/{d}/{HHMM}/HS_{sat}_{Ymd}_{HHMM}_B{NN}_FLDK_{R..}_S....DAT.bz2
    prefix = (
        f"{bucket}/AHI-L1b-FLDK/"
        f"{dt_floor10.year:04d}/{dt_floor10.month:02d}/{dt_floor10.day:02d}/"
        f"{dt_floor10.hour:02d}{dt_floor10.minute:02d}/"
    )
    band_token = f"_B{band:02d}_FLDK_{res_suffix}_S"
    try:
        files = fs.ls(prefix)
    except (FileNotFoundError, OSError):
        return []
    matches = [f for f in files if band_token in f and f.endswith(".DAT.bz2")]
    return sorted(matches)


def _segment_seq_from_path(path: str) -> tuple[int, int]:
    """Pull (segment_seq, total_segments) from the ``_SkkLL.DAT.bz2`` filename token."""
    base = path.rsplit("/", 1)[-1]
    # ...HS_H08_..._S0510.DAT.bz2 -> S0510
    s_part = base.rsplit("_S", 1)[1].split(".", 1)[0]  # "0510"
    return int(s_part[:2]), int(s_part[2:])


def _filter_segments_for_bbox(
    paths: list[str], bbox: tuple[float, float, float, float], band: int
) -> list[str]:
    """Drop segments that don't intersect the bbox's line range.

    Why this exists: B03 (0.5 km visible) on a 10-segment FLDK is ~300 MB
    compressed per segment, so loading all 10 for a small TC bbox blows
    Railway's memory budget. We forward-project the bbox lat range to
    line indices and keep only the segments whose line bands overlap.

    Single-segment files (S0101) always pass through unchanged.
    """
    if not paths:
        return paths
    seq_total = [_segment_seq_from_path(p) for p in paths]
    totals = {t for _, t in seq_total}
    if totals == {1}:
        return paths

    # The full-disk total lines is uniform per band (per HSD spec Table 3):
    #   B3        : 22000 lines  (R05)
    #   B1,B2,B4  : 11000 lines  (R10)
    #   B5..B16   :  5500 lines  (R20)
    full_lines = {3: 22000}.get(band, 11000 if band in (1, 2, 4) else 5500)
    total = next(iter(totals))
    lines_per_seg = full_lines // total

    lon_min, lat_min, lon_max, lat_max = bbox
    if lon_max < lon_min:
        # Antimeridian crossing — sampling on the unwrapped grid is fine,
        # we only care about lat in this filter step.
        pass

    # Forward-project the bbox lat range only (line is the y axis). We use
    # the standard AHI 2km grid params and then rescale to the band's actual
    # full_lines. AHI uses LFAC*2^-16 = 312.39 lines/deg at 2km; ratio
    # full_lines / 5500 scales it to other resolutions.
    res_factor = full_lines / 5500.0
    coff = 2750.5 * res_factor
    loff = 2750.5 * res_factor
    cfac = 20466275 * res_factor
    lfac = 20466275 * res_factor

    # Sample lat extremes at a few longitudes spanning the bbox; line index
    # depends on lat (and weakly on lon via the WGS84 ellipsoid term).
    sample_lats = [lat_min, (lat_min + lat_max) / 2, lat_max]
    sample_lons = [lon_min, (lon_min + lon_max) / 2, lon_max]
    lines = []
    for lat in sample_lats:
        for lon in sample_lons:
            _, line = _forward_for_filter(lat, lon, 140.7, cfac, lfac, coff, loff)
            if np.isfinite(line):
                lines.append(line)
    if not lines:
        # bbox doesn't project into any segment's lat band — fall back to all
        # segments (the higher-level fetch will surface a clearer error).
        return paths
    line_lo = max(1, int(np.floor(min(lines))) - 5)
    line_hi = min(full_lines, int(np.ceil(max(lines))) + 5)

    keep = []
    for path, (seq, _t) in zip(paths, seq_total):
        seg_first = (seq - 1) * lines_per_seg + 1
        seg_last = seq * lines_per_seg
        # Overlap test
        if not (seg_last < line_lo or seg_first > line_hi):
            keep.append(path)

    return keep if keep else paths


def _forward_for_filter(
    lat_deg: float, lon_deg: float, sub_lon: float,
    cfac: float, lfac: float, coff: float, loff: float,
) -> tuple[float, float]:
    """Lightweight scalar forward projection — only used by segment filter."""
    R_s = 42164.0; r_eq = 6378.1370; r_pol = 6356.7523
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    sub_lon_r = np.deg2rad(sub_lon)
    c_lat = np.arctan((r_pol * r_pol) / (r_eq * r_eq) * np.tan(lat))
    R_l = r_pol / np.sqrt(1 - (1 - (r_pol * r_pol) / (r_eq * r_eq)) * np.cos(c_lat) ** 2)
    R1 = R_s - R_l * np.cos(c_lat) * np.cos(lon - sub_lon_r)
    R2 = -R_l * np.cos(c_lat) * np.sin(lon - sub_lon_r)
    R3 = R_l * np.sin(c_lat)
    R_n = np.sqrt(R1 * R1 + R2 * R2 + R3 * R3)
    if R1 <= 0 or not np.isfinite(R_n):
        return float("nan"), float("nan")
    x = np.arctan2(-R2, R1)
    y = np.arcsin(-R3 / R_n)
    col = coff + cfac / (2 ** 16) * np.rad2deg(x)
    line = loff + lfac / (2 ** 16) * np.rad2deg(y)
    return col, line


def _download_and_parse(fs: s3fs.S3FileSystem, s3_path: str) -> HSDSegment:
    with fs.open(s3_path, mode="rb") as f:
        compressed = f.read()
    raw = decompress_bz2(compressed)
    return parse_hsd_segment(raw)


def _stitch(segments: list[HSDSegment]) -> tuple[np.ndarray, int, int, int]:
    """Stitch per-segment count arrays into a row-banded uint16 array.

    Spans from the first downloaded segment's start line to the last
    downloaded segment's end line. Gaps between non-contiguous segments
    are filled with the OUTSIDE_SCAN sentinel.

    Returns (counts, n_lines_local, n_columns, line_offset_global) where
    line_offset_global is the 0-based full-disk line index of local row 0.
    """
    if not segments:
        raise ValueError("no segments provided")

    segments = sorted(segments, key=lambda s: s.first_line_number)

    n_columns = segments[0].n_columns
    if any(s.n_columns != n_columns for s in segments):
        raise ValueError("segments disagree on n_columns")
    total = segments[0].total_segments
    if any(s.total_segments != total for s in segments):
        raise ValueError("segments disagree on total_segments")

    first_global = segments[0].first_line_number  # 1-based
    last_global_inclusive = segments[-1].first_line_number + segments[-1].n_lines - 1
    n_lines_local = last_global_inclusive - first_global + 1
    line_offset_global = first_global - 1  # 0-based global line for local row 0

    full = np.full((n_lines_local, n_columns), COUNT_OUTSIDE_SCAN, dtype=np.uint16)
    for s in segments:
        local_start = s.first_line_number - first_global
        local_end = local_start + s.n_lines
        full[local_start:local_end, :] = s.counts

    return full, n_lines_local, n_columns, line_offset_global


def _calibrate(meta: HSDSegment, counts_full: np.ndarray) -> tuple[np.ndarray, str]:
    """Calibrate stitched counts using metadata from any segment.

    All segments of a (band, time) share Slope/Intercept/Planck coefs, so
    we can re-use one segment's metadata as a calibration prototype and
    swap its ``counts`` for the stitched array.
    """
    proto = HSDSegment(
        sat_name=meta.sat_name,
        band_number=meta.band_number,
        central_wavelength_um=meta.central_wavelength_um,
        sub_lon=meta.sub_lon,
        cfac=meta.cfac,
        lfac=meta.lfac,
        coff=meta.coff,
        loff=meta.loff,
        total_segments=meta.total_segments,
        segment_seq=0,
        first_line_number=1,
        n_columns=counts_full.shape[1],
        n_lines=counts_full.shape[0],
        slope=meta.slope,
        intercept=meta.intercept,
        planck_c0=meta.planck_c0,
        planck_c1=meta.planck_c1,
        planck_c2=meta.planck_c2,
        speed_of_light=meta.speed_of_light,
        planck_const=meta.planck_const,
        boltzmann_const=meta.boltzmann_const,
        albedo_coef=meta.albedo_coef,
        obs_start_mjd=meta.obs_start_mjd,
        counts=counts_full,
    )
    if proto.is_visible:
        arr = counts_to_reflectance(proto)
        units = "1"
    else:
        arr = counts_to_brightness_temperature(proto)
        units = "K"
    return arr, units


def load_band_sync(
    fs: s3fs.S3FileSystem,
    bucket: str,
    dt_floor10,
    band: int,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> CalibratedDisk:
    """Download + parse + stitch + calibrate one (bucket, time slot, band).

    If ``bbox`` is provided, segments outside the bbox's line range are
    skipped before download — critical for B03 (0.5 km visible) where each
    segment is ~300 MB compressed. Stitched output covers only the
    downloaded segments; downstream consumers must rely on first_line_number
    metadata, not assume the array starts at line 1.

    Synchronous; call via ``run_in_executor`` from async code.
    """
    paths = _list_segments(fs, bucket, dt_floor10, band)
    if not paths:
        raise FileNotFoundError(
            f"no AHI segments found at s3://{bucket}/AHI-L1b-FLDK/"
            f"{dt_floor10:%Y/%m/%d/%H%M}/ for band {band}"
        )
    n_listed = len(paths)
    if bbox is not None and n_listed > 1:
        paths = _filter_segments_for_bbox(paths, bbox, band)
    log.info(
        "B%02d at %s: %d segment(s) listed, %d to download",
        band, dt_floor10, n_listed, len(paths),
    )

    segments = [_download_and_parse(fs, p) for p in paths]

    counts_local, n_lines_local, n_columns, line_offset = _stitch(segments)
    data, units = _calibrate(segments[0], counts_local)

    seg0 = segments[0]
    return CalibratedDisk(
        sat_name=seg0.sat_name,
        band_number=seg0.band_number,
        central_wavelength_um=seg0.central_wavelength_um,
        sub_lon=seg0.sub_lon,
        cfac=seg0.cfac,
        lfac=seg0.lfac,
        coff=seg0.coff,
        loff=seg0.loff,
        n_columns=n_columns,
        n_lines=n_lines_local,
        line_offset=line_offset,
        data=data,
        units=units,
        obs_start_mjd=seg0.obs_start_mjd,
    )


async def load_band(
    fs: s3fs.S3FileSystem, bucket: str, dt_floor10, band: int
) -> CalibratedDisk:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_band_sync, fs, bucket, dt_floor10, band)
