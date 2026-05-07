"""Minimal Himawari Standard Data (HSD) format reader.

Clean-room implementation written from the public JMA spec; no satpy code is
copied or derived. Scoped narrowly to what tat-satellite-render needs:

  - Parse one HSD segment file (already decompressed bz2 -> raw bytes).
  - Extract the metadata blocks we depend on (#1, #2, #3, #5, #7) and the raw
    pixel-count data block (#12).
  - Calibrate counts -> brightness_temperature for IR/WV bands (7-16) using
    the per-file Slope/Intercept and the c0/c1/c2 Planck-correction coefs.
  - Calibrate counts -> reflectance for the visible/NIR bands (1-6) using the
    Slope/Intercept and the c' radiance->albedo coefficient.

Everything else from the satpy/CSPP HSD readers — Scene integration, dask,
pyspectral, composite generation, navigation correction, observation-time
interpolation, error masks beyond fill-value handling, GSICS bias correction
— is intentionally omitted. We don't need it.

Format reference
----------------
  Himawari-8/9 Himawari Standard Data User's Guide
  Version 1.3, 3 July 2017
  Japan Meteorological Agency
  https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v13.pdf

Block layout (see spec Table 5/6):
  #1  Basic information (282 B)
  #2  Data information (50 B) — N_columns, N_lines
  #3  Projection information (127 B) — sub_lon, CFAC, LFAC, COFF, LOFF
  #4  Navigation information (139 B)
  #5  Calibration information (147 B; structure differs IR vs visible)
  #6  Inter-calibration information (259 B)
  #7  Segment information (47 B) — total_segments, segment_seq, first_line
  #8  Navigation correction (variable)
  #9  Observation time (variable)
  #10 Error information (variable)
  #11 Spare (259 B)
  #12 Data block — N_columns × N_lines × uint16

Block lengths are encoded in each block header so we don't hard-code them;
total_header_length lives in block #1 field 13 and lets us jump straight
to the data block.

License: identical to tat-satellite-render's project license. © 2026 TAT.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np


# Spec-defined sentinel pixel values (Block #5 fields 6, 7).
COUNT_ERROR_PIXEL = 65535
COUNT_OUTSIDE_SCAN = 65534


@dataclass
class HSDSegment:
    """Parsed HSD segment ready for calibration."""

    # Identity
    sat_name: str            # "Himawari-8" / "Himawari-9"
    band_number: int         # 1..16
    central_wavelength_um: float

    # Geometry / projection
    sub_lon: float           # degrees east, e.g. 140.7
    cfac: int
    lfac: int
    coff: float
    loff: float

    # Segment placement in the full disk
    total_segments: int      # 1 (single-file repack) or 10 (native FLDK)
    segment_seq: int         # 1-based
    first_line_number: int   # 1-based, line index in the full disk where this segment starts

    # Per-segment dimensions (Block #2). For a single-file repack this is the
    # full disk; for a native segment this is the segment's slice.
    n_columns: int
    n_lines: int

    # Calibration — count -> radiance (linear)
    slope: float
    intercept: float

    # IR Planck-correction (zero for visible)
    planck_c0: float
    planck_c1: float
    planck_c2: float
    speed_of_light: float
    planck_const: float
    boltzmann_const: float

    # Visible-only: radiance -> albedo (zero for IR)
    albedo_coef: float

    # Time
    obs_start_mjd: float

    # Raw pixel counts as uint16, shape (n_lines, n_columns)
    counts: np.ndarray

    @property
    def is_visible(self) -> bool:
        return self.band_number <= 6


# ---------------------------------------------------------------------------
# Low-level reader
# ---------------------------------------------------------------------------
class _Reader:
    """Tiny offset-tracking helper around a bytes buffer.

    Endianness is stamped once via set_endian(); all subsequent reads use it.
    HSD field types from the spec map onto struct codes below.
    """

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0
        self.endian = "<"  # default; updated after reading Block 1 byte_order

    def set_endian(self, byte_order_byte: int) -> None:
        # Spec Block #1 field 4: 0=Little Endian, 1=Big Endian.
        self.endian = ">" if byte_order_byte == 1 else "<"

    def seek(self, abs_pos: int) -> None:
        self.pos = abs_pos

    def skip(self, n: int) -> None:
        self.pos += n

    def _unpack(self, fmt: str, size: int):
        v = struct.unpack(self.endian + fmt, self.buf[self.pos : self.pos + size])[0]
        self.pos += size
        return v

    def i1(self) -> int:
        return self._unpack("B", 1)

    def i2(self) -> int:
        return self._unpack("H", 2)

    def i4(self) -> int:
        return self._unpack("I", 4)

    def r4(self) -> float:
        return self._unpack("f", 4)

    def r8(self) -> float:
        return self._unpack("d", 8)

    def cstr(self, n: int) -> str:
        s = self.buf[self.pos : self.pos + n].rstrip(b"\x00 ").decode("ascii", errors="replace")
        self.pos += n
        return s


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------
def parse_hsd_segment(buf: bytes) -> HSDSegment:
    """Parse one HSD segment from a bytes buffer (already decompressed).

    Returns an HSDSegment with all metadata + a uint16 ``counts`` array of
    shape ``(n_lines, n_columns)``. Use ``counts_to_brightness_temperature``
    or ``counts_to_reflectance`` to calibrate.
    """
    if len(buf) < 64:
        raise ValueError(f"buffer too short to be HSD ({len(buf)} bytes)")

    r = _Reader(buf)

    # ---- Block #1: Basic information (282 B fixed) -----------------------
    # Layout we care about (spec Table 6, #1):
    #   1: header_block_number (I1) = 1
    #   2: block_length (I2) = 282
    #   3: total_n_header_blocks (I2) = 11
    #   4: byte_order (I1) — must be read before any I2/I4/R4/R8 above
    #   5: satellite_name (C[16])
    #   ...
    #  10: observation_start_time (R8) [MJD]
    #  ...
    #  13: total_header_length (I4)
    block1_start = r.pos
    block1_header_num = r.i1()
    if block1_header_num != 1:
        raise ValueError(f"expected block #1 header number 1, got {block1_header_num}")
    # read block_length and total_n_header_blocks ASSUMING little-endian for now;
    # we'll override below once byte_order_byte is known. Block #1 length is fixed at 282
    # in both orderings, so we sanity-check after.
    saved = r.pos
    _ = struct.unpack("<H", buf[r.pos : r.pos + 2])[0]  # tentative LE block_length
    r.skip(2)
    _ = struct.unpack("<H", buf[r.pos : r.pos + 2])[0]  # tentative LE total_n_header_blocks
    r.skip(2)
    byte_order_byte = r.i1()  # I1, no endianness ambiguity
    r.set_endian(byte_order_byte)

    # Re-read block_length / total_n_header_blocks with confirmed endianness
    r.seek(saved)
    block_length = r.i2()
    if block_length != 282:
        raise ValueError(f"block #1 length {block_length} != 282 (bad endian byte? = {byte_order_byte})")
    total_n_header_blocks = r.i2()
    if total_n_header_blocks not in (11,):
        # Spec says fixed 11; if a future revision changes this we'd want to know.
        raise ValueError(f"unexpected total_n_header_blocks={total_n_header_blocks}")
    r.skip(1)  # byte_order, already consumed

    sat_name = r.cstr(16)
    _ = r.cstr(16)  # processing_center_name
    _ = r.cstr(4)   # observation_area
    _ = r.cstr(2)   # other_observation_information
    _ = r.i2()      # observation_timeline (hhmm)
    obs_start_mjd = r.r8()
    _ = r.r8()      # obs_end
    _ = r.r8()      # file_creation
    total_header_length = r.i4()
    _ = r.i4()      # total_data_length
    # Skip remainder of block #1 (quality flags + format version + filename + spare)
    # We've consumed block1_start + (1+2+2+1+16+16+4+2+2+8+8+8+4+4) = block1_start + 86
    # Block ends at block1_start + 282.
    r.seek(block1_start + 282)

    # ---- Block #2: Data information (50 B fixed) -------------------------
    block2_start = r.pos
    if r.i1() != 2:
        raise ValueError("expected block #2")
    block2_len = r.i2()
    bits_per_pixel = r.i2()  # always 16
    if bits_per_pixel != 16:
        raise ValueError(f"unexpected bits_per_pixel={bits_per_pixel}")
    n_columns = r.i2()
    n_lines = r.i2()
    _compression = r.i1()  # spec: data block compression. 0 in NOAA-distributed files.
    r.seek(block2_start + block2_len)

    # ---- Block #3: Projection information (127 B fixed) ------------------
    block3_start = r.pos
    if r.i1() != 3:
        raise ValueError("expected block #3")
    block3_len = r.i2()
    sub_lon = r.r8()
    cfac = r.i4()
    lfac = r.i4()
    coff = r.r4()
    loff = r.r4()
    _Rs = r.r8()
    _r_eq = r.r8()
    _r_pol = r.r8()
    # Remaining projection constants in block #3 are derivatives (req2/rpol2 etc.)
    # We rely on WGS84 fixed values when we project, so skip past block end.
    r.seek(block3_start + block3_len)

    # ---- Block #4: Navigation information ----
    block4_start = r.pos
    if r.i1() != 4:
        raise ValueError("expected block #4")
    block4_len = r.i2()
    r.seek(block4_start + block4_len)

    # ---- Block #5: Calibration information ------------------------------
    block5_start = r.pos
    if r.i1() != 5:
        raise ValueError("expected block #5")
    block5_len = r.i2()
    band_number = r.i2()
    central_wavelength_um = r.r8()
    _valid_bits = r.i2()
    _err_count = r.i2()
    _outside_count = r.i2()
    slope = r.r8()
    intercept = r.r8()

    if band_number >= 7:
        # IR/WV calibration
        c0 = r.r8()
        c1 = r.r8()
        c2 = r.r8()
        _C0 = r.r8()
        _C1 = r.r8()
        _C2 = r.r8()
        c_speed = r.r8()
        h_planck = r.r8()
        k_boltzmann = r.r8()
        albedo_coef = 0.0
    else:
        # Visible / NIR calibration
        albedo_coef = r.r8()
        _update_time_mjd = r.r8()
        # Updated calibration values (post-launch). Prefer these if non-zero.
        cal_slope = r.r8()
        cal_intercept = r.r8()
        if cal_slope != 0.0:
            slope = cal_slope
            intercept = cal_intercept
        c0 = c1 = c2 = 0.0
        c_speed = h_planck = k_boltzmann = 0.0

    r.seek(block5_start + block5_len)

    # ---- Block #6: Inter-calibration (259 B) ----
    block6_start = r.pos
    if r.i1() != 6:
        raise ValueError("expected block #6")
    block6_len = r.i2()
    r.seek(block6_start + block6_len)

    # ---- Block #7: Segment information (47 B) ----
    block7_start = r.pos
    if r.i1() != 7:
        raise ValueError("expected block #7")
    block7_len = r.i2()
    total_segments = r.i1()
    segment_seq = r.i1()
    first_line_number = r.i2()
    r.seek(block7_start + block7_len)

    # Skip directly to data block via total_header_length from Block #1.
    r.seek(total_header_length)

    # ---- Block #12: Data ------------------------------------------------
    n_pixels = n_columns * n_lines
    expected_bytes = n_pixels * 2
    end = r.pos + expected_bytes
    if end > len(buf):
        raise ValueError(
            f"data block truncated: need {expected_bytes} bytes from offset {r.pos}, have {len(buf) - r.pos}"
        )
    dtype = np.dtype("<u2") if r.endian == "<" else np.dtype(">u2")
    counts = np.frombuffer(buf, dtype=dtype, count=n_pixels, offset=r.pos).reshape(n_lines, n_columns)

    return HSDSegment(
        sat_name=sat_name,
        band_number=band_number,
        central_wavelength_um=central_wavelength_um,
        sub_lon=sub_lon,
        cfac=cfac,
        lfac=lfac,
        coff=coff,
        loff=loff,
        total_segments=total_segments,
        segment_seq=segment_seq,
        first_line_number=first_line_number,
        n_columns=n_columns,
        n_lines=n_lines,
        slope=slope,
        intercept=intercept,
        planck_c0=c0,
        planck_c1=c1,
        planck_c2=c2,
        speed_of_light=c_speed,
        planck_const=h_planck,
        boltzmann_const=k_boltzmann,
        albedo_coef=albedo_coef,
        obs_start_mjd=obs_start_mjd,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def counts_to_radiance(seg: HSDSegment) -> np.ndarray:
    """Convert raw counts to spectral radiance [W / (m² sr μm)].

    Pixels with sentinel values (error / outside-scan) become NaN.
    """
    counts = seg.counts
    mask = (counts == COUNT_ERROR_PIXEL) | (counts == COUNT_OUTSIDE_SCAN)
    # Cast to float32 for the calibration math; uint16 arithmetic would overflow.
    rad = counts.astype(np.float32) * np.float32(seg.slope) + np.float32(seg.intercept)
    rad[mask] = np.nan
    return rad


def counts_to_brightness_temperature(seg: HSDSegment) -> np.ndarray:
    """Convert IR/WV counts to brightness temperature [K].

    Algorithm (per HSD spec Block #5 fields 10-18, IR variant):
      1. radiance = slope * count + intercept     [W / (m² sr μm)]
      2. effective BT Te = (h c / (λ k)) / ln(1 + 2 h c² / (I λ⁵))
         using λ in meters and I in SI [W / (m² sr m_wavelength)]
      3. corrected BT Tb = c0 + c1*Te + c2*Te²
    """
    if seg.band_number < 7:
        raise ValueError(f"counts_to_brightness_temperature called on visible band {seg.band_number}")

    rad_um = counts_to_radiance(seg)  # W / (m² sr μm)

    # Convert to per-meter radiance (W / (m² sr m)) for SI Planck math.
    # Per-μm × (1e6 μm/m) = per-m for the radiance density.
    rad_si = rad_um.astype(np.float64) * 1.0e6

    lam_m = seg.central_wavelength_um * 1.0e-6
    h = seg.planck_const
    c = seg.speed_of_light
    k = seg.boltzmann_const

    # Inverse Planck. argument of log must be > 0; nan-safe via numpy errstate.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        denom = np.log1p(2.0 * h * c * c / (rad_si * lam_m**5))
        te = (h * c) / (lam_m * k * denom)
        tb = seg.planck_c0 + seg.planck_c1 * te + seg.planck_c2 * te * te

    # Pixels that were NaN in radiance (sentinels) propagate.
    return tb.astype(np.float32)


def counts_to_reflectance(seg: HSDSegment) -> np.ndarray:
    """Convert visible/NIR counts to top-of-atmosphere reflectance (0..1).

    Algorithm (per HSD spec Block #5 visible variant):
      1. radiance = slope * count + intercept    [W / (m² sr μm)]
      2. reflectance A = c' * radiance           [unitless]
    """
    if seg.band_number >= 7:
        raise ValueError(f"counts_to_reflectance called on IR band {seg.band_number}")
    rad = counts_to_radiance(seg)
    return (rad * np.float32(seg.albedo_coef)).astype(np.float32)


# ---------------------------------------------------------------------------
# Bz2 decompression helper
# ---------------------------------------------------------------------------
def decompress_bz2(data: bytes) -> bytes:
    """Decompress a bzip2-compressed HSD segment.

    Some NOAA repackaged files are concatenations of multiple bz2 streams
    (one per native segment), so we use BZ2Decompressor in a loop.
    """
    import bz2

    decomp = bz2.BZ2Decompressor()
    out = io.BytesIO()
    pos = 0
    while pos < len(data):
        chunk = decomp.decompress(data[pos:])
        out.write(chunk)
        if decomp.eof:
            tail = decomp.unused_data
            if not tail:
                break
            decomp = bz2.BZ2Decompressor()
            pos = len(data) - len(tail)
        else:
            break
    return out.getvalue()
