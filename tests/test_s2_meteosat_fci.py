"""Offline tests for the MTG FCI true-color member (s2_meteosat FCI half).

No network, no satpy: the Data Store client pieces are exercised with mocked
requests and zip fixtures, the sampler with a stub pyresample-like area. The
end-to-end fetch needs creds + satpy and is validated at activation time
(the module's honest-degrade contract is exactly that it raises until then).
"""
import datetime as dt
import io
import os
import zipfile

import numpy as np
import pytest

import s2_meteosat as MET
import truecolor as tc

UTC = dt.timezone.utc


def test_fci_roles_match_shared_pipeline_band_names():
    # the fetcher's dataset names ARE the satpy names truecolor passes to
    # pyspectral for the sensor-specific effective wavelength -- one table
    # drifting from the other silently breaks the Rayleigh band resolution
    want = tc.SENSOR_BANDS["fci"]
    assert MET.FCI_TRUECOLOR_ROLES["red"] == want["red"] == "vis_06"
    assert MET.FCI_TRUECOLOR_ROLES["green"] == want["green"] == "vis_05"
    assert MET.FCI_TRUECOLOR_ROLES["blue"] == want["blue"] == "vis_04"
    assert MET.FCI_TRUECOLOR_ROLES["veggie"] == want["veggie"] == "vis_08"
    assert MET.FCI_TRUECOLOR_ROLES["ir"] == MET.FCI_IR_DATASET == "ir_105"
    assert set(MET.FCI_VIS_DATASETS) == {"vis_04", "vis_05", "vis_06", "vis_08"}


def test_fci_collection_and_platform_constants():
    assert MET.COLLECTION_FCI == "EO:EUM:DAT:0662"   # MTG-I1 FCI L1C FDHSI
    assert MET.FCI_PLATFORM == "Meteosat-12"


def test_available_false_without_creds(monkeypatch):
    monkeypatch.delenv("EUMETSAT_CONSUMER_KEY", raising=False)
    monkeypatch.delenv("EUMETSAT_CONSUMER_SECRET", raising=False)
    assert MET.available() is False


class _Resp:
    def __init__(self, content):
        self._c = content
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        for i in range(0, len(self._c), n):
            yield self._c[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_product_extracts_nc_chunks(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("prod/chunk_0001.nc", b"a")
        zf.writestr("prod/chunk_0002.nc", b"b")
        zf.writestr("prod/manifest.xml", b"m")
    monkeypatch.setattr(MET, "_token", lambda: "tok")
    monkeypatch.setattr(MET.requests, "get",
                        lambda *a, **k: _Resp(buf.getvalue()))
    paths = MET._download_product("EO:EUM:DAT:0662", "PID", str(tmp_path),
                                  pattern="*.nc")
    assert [os.path.basename(p) for p in paths] == ["chunk_0001.nc", "chunk_0002.nc"]


def test_download_product_seviri_default_still_nat(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("prod/scene.nat", b"n")
    monkeypatch.setattr(MET, "_token", lambda: "tok")
    monkeypatch.setattr(MET.requests, "get",
                        lambda *a, **k: _Resp(buf.getvalue()))
    paths = MET._download_product("EO:EUM:DAT:MSG:HRSEVIRI", "PID", str(tmp_path))
    assert len(paths) == 1 and paths[0].endswith(".nat")
    with pytest.raises(RuntimeError):
        MET._download_product("EO:EUM:DAT:0662", "PID2", str(tmp_path),
                              pattern="*.nc")


class _StubArea:
    """pyresample-like area: identity mapping lon->col, lat->row over a
    10-deg/px grid anchored at (lon0, lat0), masked outside the grid."""

    def __init__(self, lon0=-10.0, lat0=10.0, n=5):
        self.lon0, self.lat0, self.n = lon0, lat0, n

    def get_array_coordinates_from_lonlat(self, lons, lats):
        cols = (np.asarray(lons) - self.lon0) / 10.0
        rows = (self.lat0 - np.asarray(lats)) / 10.0
        return np.ma.masked_invalid(cols), np.ma.masked_invalid(rows)


def test_sample_area_bilinear_and_offdisk_nan():
    area = _StubArea()
    vals = np.arange(25, dtype=np.float32).reshape(5, 5)
    TLAT = np.array([[10.0, 5.0, 200.0]])   # 200 -> far outside -> NaN
    TLON = np.array([[-10.0, -5.0, -10.0]])
    out = MET._sample_area(area, vals, TLAT, TLON)
    assert out[0, 0] == pytest.approx(0.0)         # exact grid point
    assert out[0, 1] == pytest.approx(3.0)         # bilinear midpoint (0.5, 0.5)
    assert np.isnan(out[0, 2])                     # off-grid stays NaN


def test_sample_area_space_pixel_sentinel_poisons():
    area = _StubArea()
    vals = np.full((5, 5), 7.0, np.float32)
    vals[0, 1] = np.nan                            # a space pixel
    TLAT = np.array([[10.0]])
    TLON = np.array([[-5.0]])                      # bilinear touches the NaN
    out = MET._sample_area(area, vals, TLAT, TLON)
    assert np.isnan(out[0, 0])


def test_fci_disk_sample_uses_own_area_per_dataset():
    a1, a2 = _StubArea(), _StubArea(lon0=0.0)
    disk = MET.FciDisk(
        values={"vis_06": np.arange(25, dtype=np.float32).reshape(5, 5),
                "ir_105": np.full((5, 5), 250.0, np.float32)},
        areas={"vis_06": a1, "ir_105": a2},
        scan_end=dt.datetime(2026, 7, 24, tzinfo=UTC),
        sat_name="Meteosat-12", collection=MET.COLLECTION_FCI)
    v = disk.sample("vis_06", np.array([[10.0]]), np.array([[-10.0]]))
    assert v[0, 0] == pytest.approx(0.0)
    b = disk.sample("ir_105", np.array([[10.0]]), np.array([[0.0]]))
    assert b[0, 0] == pytest.approx(250.0)


def test_search_latest_rechecks_end_time(monkeypatch):
    """The dtend filter matches on START; the honest end<=not_after re-check
    must skip a still-scanning newest product (same contract SEVIRI relies
    on, now shared by the FCI fetch)."""
    class _R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"features": [
                {"properties": {"identifier": "TOO_NEW",
                                "date": "2026-07-24T00:20:00Z/2026-07-24T00:29:35Z"}},
                {"properties": {"identifier": "GOOD",
                                "date": "2026-07-24T00:10:00Z/2026-07-24T00:19:35Z"}},
            ]}

    monkeypatch.setattr(MET.requests, "get", lambda *a, **k: _R())
    feat = MET._search_latest("EO:EUM:DAT:0662",
                              dt.datetime(2026, 7, 24, 0, 25, tzinfo=UTC))
    assert feat["properties"]["identifier"] == "GOOD"
