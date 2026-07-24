"""Offline locks for s2_gk2a (GK-2A AMI L1B ingestion).

Every calibration pin below is hand-computed from the REAL global/variable
attrs of live bucket files (gk2a_ami_le1b_ir105_fd020ge_202607230600.nc and
gk2a_ami_le1b_vi005_fd010ge_202607230610.nc, captured in gk2a_attrs.json):
the gain/offset/Planck/Teff-polynomial constants and the DN samples are
byte-for-byte those files' values. No network anywhere -- fetch paths run
against monkeypatched requests.
"""
import datetime as dt
import os

import numpy as np
import pytest
import xarray as xr

import s2_gk2a as G

UTC = dt.timezone.utc

# ---------------------------------------------------------------------------
# real-file attrs (verbatim from gk2a_attrs.json)
# ---------------------------------------------------------------------------

IR105_ATTRS = {
    "satellite_name": "GK-2A",
    "instrument_name": "AMI",
    "channel_spatial_resolution": "2.0",
    "channel_center_wavelength": "10.5",
    "scene_acquisition_time": "20260723_060032",
    "file_name": "gk2a_ami_le1b_ir105_fd020ge_202607230600.nc",
    "DN_to_Radiance_Gain": -0.0198196955025196,
    "DN_to_Radiance_Offset": 161.580139160156,
    "Teff_to_Tbb_c0": -0.142866448475177,
    "Teff_to_Tbb_c1": 1.00064069572049,
    "Teff_to_Tbb_c2": -5.50443294960498e-07,
    "light_speed": 299792458.0,
    "Plank_constant_h": 6.62606957e-34,
    "Boltzmann_constant_k": 1.3806488e-23,
    "number_of_columns": "5500",
    "number_of_lines": "5500",
    "observation_mode": "FD",
    "observation_start_time": 838058432.1166971,
    "projection_type": "GEOS",
    "sub_longitude": 2.2375121010567303,        # RADIANS in the file
    "cfac": 20425338.903339352,
    "lfac": -20425338.903339352,                # negative: row 1 = north
    "coff": 2750.5,
    "loff": 2750.5,
    "nominal_satellite_height": 42164000.0,     # geocentric, meters
    "earth_equatorial_radius": 6378137.0,
    "earth_polar_radius": 6356752.3,
    "image_upperleft_x": -0.153972,             # radians, col 1.0
    "image_upperleft_y": 0.153972,              # radians, row 1.0
}
IR105_VAR = {
    "channel_name": "IR105",
    "number_of_total_bits_per_pixel": "16",
    "number_of_data_quality_flag_bits_per_pixel": "2",
    "number_of_valid_bits_per_pixel": "13",
}

VI005_ATTRS = {
    "satellite_name": "GK-2A",
    "instrument_name": "AMI",
    "channel_spatial_resolution": "1.0",
    "channel_center_wavelength": "0.51",
    "scene_acquisition_time": "20260723_061032",
    "file_name": "gk2a_ami_le1b_vi005_fd010ge_202607230610.nc",
    "DN_to_Radiance_Gain": 0.343625485897064,
    "DN_to_Radiance_Offset": -6.87249755859375,
    "Radiance_to_Albedo_c": 0.0016595767,
    "observation_start_time": 838059032.1161231,
    "projection_type": "GEOS",
    "sub_longitude": 2.2375121010567303,
    "cfac": 40850677.806678705,
    "lfac": -40850677.806678705,
    "coff": 5500.5,
    "loff": 5500.5,
    "nominal_satellite_height": 42164000.0,
    "earth_equatorial_radius": 6378137.0,
    "earth_polar_radius": 6356752.3,
    "image_upperleft_x": -0.153986,
    "image_upperleft_y": 0.153986,
}
VI005_VAR = {
    "channel_name": "VI005",
    "number_of_total_bits_per_pixel": "16",
    "number_of_data_quality_flag_bits_per_pixel": "2",
    "number_of_valid_bits_per_pixel": "11",
}

SUB_LON_DEG = 128.2   # degrees(2.2375121010567303), exact to <1e-9


def _make_ds(counts, attrs, var_attrs):
    da = xr.DataArray(np.asarray(counts, np.uint16),
                      dims=("dim_image_y", "dim_image_x"), attrs=dict(var_attrs))
    return xr.Dataset({"image_pixel_values": da}, attrs=dict(attrs))


# ---------------------------------------------------------------------------
# calibration: pinned to hand-computed values from the real constants
# ---------------------------------------------------------------------------

class TestVisCalibration:
    def test_pinned_albedo(self):
        # DN 539: rad = 539*0.343625485897064 - 6.87249755859375
        #             = 178.34163933992374 W m-2 sr-1 um-1
        #        alb = rad * 0.0016595767 = 0.29597162928834087
        # DN 146 -> 0.07185439926940904
        disk = G.ami_disk_from_dataset(
            _make_ds([[539, 146], [17, 32768]], VI005_ATTRS, VI005_VAR))
        assert disk.kind == "albedo"
        assert disk.units == "1"
        assert disk.data.dtype == np.float32
        assert disk.data[0, 0] == pytest.approx(0.2959716293, abs=2e-6)
        assert disk.data[0, 1] == pytest.approx(0.0718543993, abs=2e-6)

    def test_negative_albedo_clipped_to_zero(self):
        # DN 17 (the file's min_pixel_value): rad = -1.0309 -> albedo would
        # be -0.00171 -> clipped to exactly 0, NOT NaN (it is a valid pixel).
        disk = G.ami_disk_from_dataset(
            _make_ds([[17]], VI005_ATTRS, VI005_VAR))
        assert disk.data[0, 0] == 0.0

    def test_fill_dn_is_nan(self):
        # 32768 = 0b10<<14: DQF "out of scan area" (the corner space pixel
        # of the real file's DN sample row).
        disk = G.ami_disk_from_dataset(
            _make_ds([[539, 32768]], VI005_ATTRS, VI005_VAR))
        assert np.isnan(disk.data[0, 1])
        assert np.isfinite(disk.data[0, 0])


class TestIrCalibration:
    def test_pinned_bt(self):
        # Inverse Planck with the file's own h/c/k and nu = 1/10.5um,
        # radiance in mW m-2 sr-1 (cm-1)-1 (x1e-5 -> SI), then the
        # Teff_to_Tbb polynomial. Hand-computed:
        #   DN 3508 -> 289.9783358 K   (rad 92.0526 mW ...)
        #   DN 3127 -> 294.8541096 K
        #   DN 6198 -> 245.2956070 K
        disk = G.ami_disk_from_dataset(
            _make_ds([[3508, 3127], [6198, 32768]], IR105_ATTRS, IR105_VAR))
        assert disk.kind == "bt"
        assert disk.units == "K"
        assert disk.data[0, 0] == pytest.approx(289.9783358, abs=1e-3)
        assert disk.data[0, 1] == pytest.approx(294.8541096, abs=1e-3)
        assert disk.data[1, 0] == pytest.approx(245.2956070, abs=1e-3)
        assert np.isnan(disk.data[1, 1])          # DQF out-of-scan

    def test_negative_radiance_is_nan(self):
        # DN 8191 (top of the 13 valid bits): rad = -0.76 mW -> no Planck
        # inversion exists -> NaN, never a fabricated temperature.
        disk = G.ami_disk_from_dataset(
            _make_ds([[8191]], IR105_ATTRS, IR105_VAR))
        assert np.isnan(disk.data[0, 0])


class TestDnMasking:
    def test_dqf_bits(self):
        raw = np.array([[3508,                # DQF 0: good
                         0x4000 | 3508,       # DQF 1: conditionally usable
                         0x8000 | 3508,       # DQF 2: out of scan
                         0xC000 | 3508]],     # DQF 3: error
                       np.uint16)
        dn = G._mask_dn(raw, IR105_VAR)
        assert dn[0, 0] == 3508.0
        assert dn[0, 1] == 3508.0             # kept (module-doc departure)
        assert np.isnan(dn[0, 2])
        assert np.isnan(dn[0, 3])

    def test_padding_bits_stripped(self):
        # ir105 has 13 valid bits in 16: bit 13 is padding, not data --
        # a set padding bit with DQF 0 must not corrupt the count.
        dn = G._mask_dn(np.array([[8192 + 100]], np.uint16), IR105_VAR)
        assert dn[0, 0] == 100.0

    def test_vis_valid_bits(self):
        # vi005: 11 valid bits -> count mask 0x7FF.
        dn = G._mask_dn(np.array([[0x4000 | 539]], np.uint16), VI005_VAR)
        assert dn[0, 0] == 539.0

    def test_float_counts_rejected(self):
        with pytest.raises(ValueError):
            G._mask_dn(np.array([[1.0]]), IR105_VAR)


# ---------------------------------------------------------------------------
# scan time: the J2000-NOON epoch discovery
# ---------------------------------------------------------------------------

class TestScanTime:
    def test_obs_start_epoch_is_j2000_noon(self):
        # 838058432.1166971 s after 2000-01-01T12:00Z (NOT midnight -- that
        # reads 12 h early) == scene_acquisition_time "20260723_060032".
        t = G._scan_start(IR105_ATTRS)
        want = dt.datetime(2026, 7, 23, 6, 0, 32, 116697, tzinfo=UTC)
        assert abs((t - want).total_seconds()) < 1e-3
        assert t.tzinfo is not None

    def test_vi005_slot(self):
        t = G._scan_start(VI005_ATTRS)
        assert t.replace(microsecond=0) == dt.datetime(2026, 7, 23, 6, 10, 32,
                                                       tzinfo=UTC)

    def test_epoch_mismatch_falls_back_to_string(self):
        attrs = dict(IR105_ATTRS)
        attrs["observation_start_time"] += 100000.0     # broken epoch
        t = G._scan_start(attrs)
        assert t == dt.datetime(2026, 7, 23, 6, 0, 32, tzinfo=UTC)

    def test_string_only(self):
        attrs = {"scene_acquisition_time": "20260723_060032"}
        assert G._scan_start(attrs) == dt.datetime(2026, 7, 23, 6, 0, 32,
                                                   tzinfo=UTC)

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            G._scan_start({})


# ---------------------------------------------------------------------------
# GEOS navigation
# ---------------------------------------------------------------------------

NAV2KM = dict(cfac=IR105_ATTRS["cfac"], lfac=IR105_ATTRS["lfac"],
              coff=IR105_ATTRS["coff"], loff=IR105_ATTRS["loff"])


class TestGeos:
    def test_sub_lon_degrees(self):
        assert np.degrees(IR105_ATTRS["sub_longitude"]) == pytest.approx(
            SUB_LON_DEG, abs=1e-9)

    def test_disk_center_maps_to_coff_loff(self):
        x, y = G._ami_latlon_to_xy_deg(0.0, SUB_LON_DEG, SUB_LON_DEG)
        assert float(x) == pytest.approx(0.0, abs=1e-12)
        assert float(y) == pytest.approx(0.0, abs=1e-12)
        col, row = G._ami_xy_deg_to_colline(x, y, **NAV2KM)
        assert float(col) == pytest.approx(2750.5, abs=1e-9)
        assert float(row) == pytest.approx(2750.5, abs=1e-9)

    def test_upperleft_scan_angle_is_pixel_one(self):
        # The real file's image_upperleft_x/y land on 1-based pixel-center
        # (1.0, 1.0) -- the check that array index = col - 1.
        col, row = G._ami_xy_deg_to_colline(
            np.degrees(IR105_ATTRS["image_upperleft_x"]),
            np.degrees(IR105_ATTRS["image_upperleft_y"]), **NAV2KM)
        assert float(col) == pytest.approx(1.0, abs=0.01)
        assert float(row) == pytest.approx(1.0, abs=0.01)
        # and on the 1 km grid (independent cfac/coff)
        col, row = G._ami_xy_deg_to_colline(
            np.degrees(VI005_ATTRS["image_upperleft_x"]),
            np.degrees(VI005_ATTRS["image_upperleft_y"]),
            VI005_ATTRS["cfac"], VI005_ATTRS["lfac"],
            VI005_ATTRS["coff"], VI005_ATTRS["loff"])
        assert float(col) == pytest.approx(1.0, abs=0.01)
        assert float(row) == pytest.approx(1.0, abs=0.01)

    def test_orientation_north_up_west_left(self):
        x, y = G._ami_latlon_to_xy_deg(np.array([20.0, 0.0]),
                                       np.array([SUB_LON_DEG, 120.0]),
                                       SUB_LON_DEG)
        col, row = G._ami_xy_deg_to_colline(x, y, **NAV2KM)
        assert row[0] < 2750.5            # north of center -> smaller row
        assert col[1] < 2750.5            # west of center -> smaller col

    def test_roundtrip_under_a_tenth_pixel(self):
        lats, lons = np.meshgrid(np.linspace(-55.0, 55.0, 9),
                                 np.linspace(75.0, 175.0, 9))
        x, y = G._ami_latlon_to_xy_deg(lats, lons, SUB_LON_DEG)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        col1, row1 = G._ami_xy_deg_to_colline(x, y, **NAV2KM)
        lat2, lon2 = G._ami_xy_deg_to_latlon(x, y, SUB_LON_DEG)
        np.testing.assert_allclose(lat2, lats, atol=1e-6)
        np.testing.assert_allclose(lon2, lons, atol=1e-6)
        x2, y2 = G._ami_latlon_to_xy_deg(lat2, lon2, SUB_LON_DEG)
        col2, row2 = G._ami_xy_deg_to_colline(x2, y2, **NAV2KM)
        assert np.max(np.abs(col2 - col1)) < 0.1
        assert np.max(np.abs(row2 - row1)) < 0.1

    def test_far_side_is_nan(self):
        x, y = G._ami_latlon_to_xy_deg(0.0, SUB_LON_DEG + 180.0, SUB_LON_DEG)
        assert np.isnan(float(x)) and np.isnan(float(y))
        # just past the horizon too (0N, ~86 deg east of sub-lon)
        x, y = G._ami_latlon_to_xy_deg(0.0, SUB_LON_DEG + 86.0, SUB_LON_DEG)
        assert np.isnan(float(x))


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def _plane_disk(stride=50):
    """Real 2 km nav, decimated to a 110x110 grid holding an exact plane in
    LOCAL index space -- bilinear interpolation of a plane is exact, so any
    sampling offset shows up as a hard error."""
    n = 5500 // stride
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    data = (3.0 + 2.0 * ii + 0.5 * jj).astype(np.float32)
    return G.AmiDisk(
        band="ir105", kind="bt", data=data, units="K",
        sub_lon=SUB_LON_DEG, stride=stride,
        scan_start=dt.datetime(2026, 7, 23, 6, 0, 32, tzinfo=UTC),
        sat_name="GK-2A", wavelength_um=10.5, resolution_km=2.0, **NAV2KM)


class TestSampling:
    def test_bilinear_matches_plane(self):
        disk = _plane_disk()
        lats = np.array([10.0, -20.0, 35.0])
        lons = np.array([130.0, 110.0, 150.0])
        got = disk.sample(lats, lons)
        x, y = G._ami_latlon_to_xy_deg(lats, lons, SUB_LON_DEG)
        col, row = G._ami_xy_deg_to_colline(x, y, **NAV2KM)
        want = 3.0 + 2.0 * (row - 1.0) / 50 + 0.5 * (col - 1.0) / 50
        np.testing.assert_allclose(got, want, atol=2e-3)

    def test_sample_xy_shared_trig_path_agrees(self):
        disk = _plane_disk()
        lats = np.array([[5.0, -12.0]])
        lons = np.array([[140.0, 100.0]])
        x, y = G._ami_latlon_to_xy_deg(lats, lons, disk.sub_lon)
        np.testing.assert_array_equal(disk.sample_xy(x, y),
                                      disk.sample(lats, lons))

    def test_off_disk_is_nan(self):
        disk = _plane_disk()
        got = disk.sample(np.array([0.0, 0.0]),
                          np.array([SUB_LON_DEG - 180.0, SUB_LON_DEG + 86.0]))
        assert np.isnan(got).all()

    def test_nan_neighbor_poisons_sample(self):
        # SeviriDisk sentinel semantics: -1e9 sentinel vs -1e8 threshold
        # guarantees poisoning at bilinear weight >= 0.1; the NEAREST cell
        # always carries weight >= 0.25, so poison that one.
        disk = _plane_disk()
        x, y = G._ami_latlon_to_xy_deg(10.0, 130.0, SUB_LON_DEG)
        col, row = G._ami_xy_deg_to_colline(x, y, **NAV2KM)
        r0 = int(round((float(row) - 1) / 50))
        c0 = int(round((float(col) - 1) / 50))
        disk.data[r0, c0] = np.nan
        got = disk.sample(np.array([10.0]), np.array([130.0]))
        assert np.isnan(got[0])


# ---------------------------------------------------------------------------
# keys, listing, slot selection
# ---------------------------------------------------------------------------

class TestKeys:
    def test_slot_key_pins(self):
        slot = dt.datetime(2026, 7, 23, 6, 0, tzinfo=UTC)
        assert G.slot_key("ir105", slot) == (
            "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_ir105_fd020ge_202607230600.nc")
        assert G.slot_key("vi005", slot) == (
            "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_vi005_fd010ge_202607230600.nc")
        assert G.slot_key("vi006", slot) == (
            "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_vi006_fd005ge_202607230600.nc")

    def test_hour_prefix(self):
        assert G._hour_prefix(dt.datetime(2026, 7, 23, 6, 44, tzinfo=UTC)) == (
            "AMI/L1B/FD/202607/23/06/")

    def test_truecolor_roles(self):
        assert G.TRUECOLOR_BANDS == {"red": "vi006", "green": "vi005",
                                     "blue": "vi004", "veggie": "vi008",
                                     "ir": "ir105"}
        assert G.AMI_PLATFORM == "GEO-KOMPSAT-2A"
        assert G.AMI_SENSOR == "ami"


_NS = "http://s3.amazonaws.com/doc/2006-03-01/"


def _listing_xml(keys, truncated=False, token=None):
    contents = "".join(f"<Contents><Key>{k}</Key><Size>1</Size></Contents>"
                       for k in keys)
    tok = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<ListBucketResult xmlns="{_NS}"><Name>noaa-gk2a-pds</Name>'
            f'<KeyCount>{len(keys)}</KeyCount>'
            f'<IsTruncated>{"true" if truncated else "false"}</IsTruncated>'
            f'{tok}{contents}</ListBucketResult>').encode()


class _FakeResp:
    def __init__(self, content=b"", chunks=None):
        self.content = content
        self._chunks = list(chunks or [])

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestListing:
    def test_parse_and_band_filter(self, monkeypatch):
        keys = [
            "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_ir105_fd020ge_202607230600.nc",
            "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_ir105_fd020ge_202607230610.nc",
            "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_vi005_fd010ge_202607230600.nc",
            "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_vi006_fd005ge_202607230600.nc",
        ]
        monkeypatch.setattr(G.requests, "get",
                            lambda url, **kw: _FakeResp(_listing_xml(keys)))
        got = G._list_keys("AMI/L1B/FD/202607/23/06/")
        assert got == keys
        slots = G._band_slots(got, "ir105")
        assert sorted(slots) == [dt.datetime(2026, 7, 23, 6, 0, tzinfo=UTC),
                                 dt.datetime(2026, 7, 23, 6, 10, tzinfo=UTC)]
        assert slots[dt.datetime(2026, 7, 23, 6, 0, tzinfo=UTC)] == keys[0]
        assert list(G._band_slots(got, "vi006")) == [
            dt.datetime(2026, 7, 23, 6, 0, tzinfo=UTC)]

    def test_continuation_token_followed(self, monkeypatch):
        k1 = "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_ir105_fd020ge_202607230600.nc"
        k2 = "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_ir105_fd020ge_202607230610.nc"

        def fake_get(url, params=None, **kw):
            if params and params.get("continuation-token") == "T1":
                return _FakeResp(_listing_xml([k2]))
            return _FakeResp(_listing_xml([k1], truncated=True, token="T1"))

        monkeypatch.setattr(G.requests, "get", fake_get)
        assert G._list_keys("AMI/L1B/FD/202607/23/06/") == [k1, k2]


class TestPickSlot:
    S = {dt.datetime(2026, 7, 23, 5, 50, tzinfo=UTC): "a",
         dt.datetime(2026, 7, 23, 6, 0, tzinfo=UTC): "b",
         dt.datetime(2026, 7, 23, 6, 10, tzinfo=UTC): "c"}

    def test_nearest(self):
        got = G._pick_slot(self.S, dt.datetime(2026, 7, 23, 6, 4, tzinfo=UTC),
                           nearest=True)
        assert got == dt.datetime(2026, 7, 23, 6, 0, tzinfo=UTC)

    def test_tie_prefers_earlier(self):
        got = G._pick_slot(self.S, dt.datetime(2026, 7, 23, 6, 5, tzinfo=UTC),
                           nearest=True)
        assert got == dt.datetime(2026, 7, 23, 6, 0, tzinfo=UTC)

    def test_exact_slot_required(self):
        got = G._pick_slot(self.S, dt.datetime(2026, 7, 23, 6, 4, tzinfo=UTC),
                           nearest=False)
        assert got == dt.datetime(2026, 7, 23, 6, 0, tzinfo=UTC)
        with pytest.raises(RuntimeError):
            G._pick_slot(self.S, dt.datetime(2026, 7, 23, 6, 25, tzinfo=UTC),
                         nearest=False)

    def test_empty_raises(self):
        with pytest.raises(RuntimeError):
            G._pick_slot({}, dt.datetime(2026, 7, 23, tzinfo=UTC), nearest=True)


# ---------------------------------------------------------------------------
# fetch path (requests fully mocked; decode stubbed)
# ---------------------------------------------------------------------------

class TestFetch:
    HOUR_KEYS = {
        "AMI/L1B/FD/202607/23/05/": [
            "AMI/L1B/FD/202607/23/05/gk2a_ami_le1b_ir105_fd020ge_202607230550.nc",
            "AMI/L1B/FD/202607/23/05/gk2a_ami_le1b_vi005_fd010ge_202607230550.nc",
        ],
        "AMI/L1B/FD/202607/23/06/": [
            "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_ir105_fd020ge_202607230600.nc",
            "AMI/L1B/FD/202607/23/06/gk2a_ami_le1b_ir105_fd020ge_202607230610.nc",
        ],
        "AMI/L1B/FD/202607/23/07/": [],
    }

    def _wire(self, monkeypatch):
        calls = {"objects": [], "decoded": [], "dirs": []}

        def fake_get(url, params=None, stream=False, **kw):
            if params and "list-type" in params:
                assert url == G.S3_BASE + "/"
                return _FakeResp(_listing_xml(
                    self.HOUR_KEYS.get(params["prefix"], [])))
            calls["objects"].append(url)
            return _FakeResp(chunks=[b"NCDF", b"BYTES"])

        def fake_decode(path, band=None, stride=1):
            calls["decoded"].append((path, band, stride))
            calls["dirs"].append(os.path.dirname(path))
            assert os.path.exists(path)          # streamed file is on disk
            with open(path, "rb") as fh:
                assert fh.read() == b"NCDFBYTES"
            return "SENTINEL-DISK"

        monkeypatch.setattr(G.requests, "get", fake_get)
        monkeypatch.setattr(G, "_decode_path", fake_decode)
        return calls

    def test_nearest_slot_url_and_cleanup(self, monkeypatch):
        calls = self._wire(monkeypatch)
        got = G.fetch_ami_disk(
            "ir105", time=dt.datetime(2026, 7, 23, 6, 4, tzinfo=UTC),
            nearest=True, timeout=5, stride=3)
        assert got == "SENTINEL-DISK"
        assert calls["objects"] == [G.S3_BASE + (
            "/AMI/L1B/FD/202607/23/06/"
            "gk2a_ami_le1b_ir105_fd020ge_202607230600.nc")]
        (path, band, stride), = calls["decoded"]
        assert band == "ir105" and stride == 3
        assert path.endswith("gk2a_ami_le1b_ir105_fd020ge_202607230600.nc")
        assert not os.path.exists(calls["dirs"][0])   # tempdir hygiene

    def test_previous_hour_reachable(self, monkeypatch):
        calls = self._wire(monkeypatch)
        G.fetch_ami_disk("vi005",
                         time=dt.datetime(2026, 7, 23, 6, 1, tzinfo=UTC),
                         nearest=True, timeout=5)
        assert calls["objects"][0].endswith(
            "gk2a_ami_le1b_vi005_fd010ge_202607230550.nc")

    def test_exact_slot_missing_raises(self, monkeypatch):
        self._wire(monkeypatch)
        with pytest.raises(RuntimeError):
            G.fetch_ami_disk(
                "ir105", time=dt.datetime(2026, 7, 23, 6, 25, tzinfo=UTC),
                nearest=False, timeout=5)

    def test_no_slots_raises(self, monkeypatch):
        self._wire(monkeypatch)
        with pytest.raises(RuntimeError):
            G.fetch_ami_disk(
                "vi006", time=dt.datetime(2026, 7, 23, 6, 4, tzinfo=UTC),
                timeout=5)

    def test_unknown_band_rejected(self):
        with pytest.raises(ValueError):
            G.fetch_ami_disk("b13")


# ---------------------------------------------------------------------------
# dataset decode metadata + stride
# ---------------------------------------------------------------------------

class TestDecodeMeta:
    def test_metadata(self):
        disk = G.ami_disk_from_dataset(
            _make_ds([[3508, 3127], [6198, 32768]], IR105_ATTRS, IR105_VAR))
        assert disk.band == "ir105"           # from channel_name "IR105"
        assert disk.sat_name == "GK-2A"
        assert disk.sub_lon == pytest.approx(128.2, abs=1e-9)
        assert (disk.cfac, disk.lfac) == (IR105_ATTRS["cfac"], IR105_ATTRS["lfac"])
        assert (disk.coff, disk.loff) == (2750.5, 2750.5)
        assert disk.h_km == pytest.approx(42164.0)
        assert disk.r_eq_km == pytest.approx(6378.137)
        assert disk.r_pol_km == pytest.approx(6356.7523)
        assert disk.wavelength_um == pytest.approx(10.5)
        assert disk.resolution_km == pytest.approx(2.0)
        assert disk.scan_start.replace(microsecond=0) == dt.datetime(
            2026, 7, 23, 6, 0, 32, tzinfo=UTC)
        assert disk.stride == 1

    def test_stride_decimates_but_keeps_native_nav(self):
        counts = np.full((8, 8), 3508, np.uint16)
        disk = G.ami_disk_from_dataset(
            _make_ds(counts, IR105_ATTRS, IR105_VAR), stride=2)
        assert disk.data.shape == (4, 4)
        assert disk.stride == 2
        assert disk.cfac == IR105_ATTRS["cfac"]   # nav stays full-res

    def test_unknown_calibration_kind_raises(self):
        attrs = {k: v for k, v in VI005_ATTRS.items()
                 if k != "Radiance_to_Albedo_c"}
        with pytest.raises(ValueError):
            G.ami_disk_from_dataset(_make_ds([[1]], attrs, VI005_VAR))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
