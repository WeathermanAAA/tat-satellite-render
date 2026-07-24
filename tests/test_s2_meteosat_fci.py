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
    def __init__(self, content, status=200, headers=None, text=""):
        self._c = content
        self.status_code = status
        self.headers = {"Content-Length": str(len(content))} if headers is None \
            else headers
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise MET.requests.HTTPError(f"HTTP {self.status_code}")

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


def _fci_zip(n_body=40, trailer=True, skip=()):
    """A fake FDHSI product zip: n_body CHK-BODY strips + one TRAIL, all
    numbered _NNNN.nc like the real Data Store product."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        k = 0
        for i in range(1, n_body + 1):
            if i in skip:
                continue
            k += 1
            zf.writestr(f"prod/W_XX-CHK-BODY_{i:04d}.nc", b"b")
        if trailer:
            zf.writestr(f"prod/W_XX-CHK-TRAIL_{n_body + 1:04d}.nc", b"t")
    return buf.getvalue()


def test_fci_chunk_gate_accepts_complete_set(tmp_path, monkeypatch):
    monkeypatch.setattr(MET, "_token", lambda: "tok")
    monkeypatch.setattr(MET.requests, "get",
                        lambda *a, **k: _Resp(_fci_zip()))
    paths = MET._download_product(MET.COLLECTION_FCI, "PID", str(tmp_path),
                                  pattern="*.nc")
    assert len(paths) == MET.FCI_EXPECTED_CHUNKS == 41
    MET._verify_fci_chunks(paths, "PID")   # must not raise


def test_fci_chunk_gate_rejects_short_set(tmp_path, monkeypatch):
    monkeypatch.setattr(MET, "_token", lambda: "tok")
    monkeypatch.setattr(MET.requests, "get",
                        lambda *a, **k: _Resp(_fci_zip(n_body=39)))
    paths = MET._download_product(MET.COLLECTION_FCI, "PID", str(tmp_path),
                                  pattern="*.nc")
    with pytest.raises(RuntimeError, match="incomplete FCI chunk set"):
        MET._verify_fci_chunks(paths, "PID")


def test_fci_chunk_gate_rejects_gapped_numbering(tmp_path, monkeypatch):
    # 41 files but body strip 0007 missing (an extra strip pushes the count
    # back up) -- the contiguity check must still refuse
    monkeypatch.setattr(MET, "_token", lambda: "tok")
    monkeypatch.setattr(MET.requests, "get",
                        lambda *a, **k: _Resp(_fci_zip(n_body=41, skip=(7,))))
    paths = MET._download_product(MET.COLLECTION_FCI, "PID", str(tmp_path),
                                  pattern="*.nc")
    assert len(paths) == 41
    with pytest.raises(RuntimeError, match="missing"):
        MET._verify_fci_chunks(paths, "PID")


def test_download_product_resumes_with_range(tmp_path, monkeypatch):
    """A dropped stream mid-transfer must resume from the byte on disk via
    a Range request, not restart (an ~800 MB product makes restarts a
    reliability cliff)."""
    payload = _fci_zip()
    cut = len(payload) // 2
    calls = []

    class _DropResp(_Resp):
        def iter_content(self, n):
            yield payload[:cut]
            raise IOError("connection reset")

    def fake_get(url, headers=None, stream=True, timeout=0):
        calls.append(dict(headers or {}))
        if len(calls) == 1:
            return _DropResp(payload)
        rng = (headers or {}).get("Range", "")
        assert rng == f"bytes={cut}-", f"expected resume, got {rng!r}"
        rest = payload[cut:]
        return _Resp(rest, status=206,
                     headers={"Content-Length": str(len(rest))})

    monkeypatch.setattr(MET, "_token", lambda: "tok")
    monkeypatch.setattr(MET.requests, "get", fake_get)
    paths = MET._download_product(MET.COLLECTION_FCI, "PID", str(tmp_path),
                                  pattern="*.nc")
    assert len(calls) == 2
    assert len(paths) == 41


def test_download_product_4xx_is_fatal_with_server_message(tmp_path, monkeypatch):
    """The licence gate's 403 must surface EUMETSAT's own message verbatim
    and must NOT burn retries (a 4xx never heals)."""
    calls = []

    def fake_get(url, headers=None, stream=True, timeout=0):
        calls.append(1)
        return _Resp(b"", status=403,
                     text='{"exceptionText":"GeneralLicense required to '
                          'access this collection"}')

    monkeypatch.setattr(MET, "_token", lambda: "tok")
    monkeypatch.setattr(MET.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="GeneralLicense required"):
        MET._download_product(MET.COLLECTION_FCI, "PID", str(tmp_path),
                              pattern="*.nc")
    assert len(calls) == 1


def test_newest_fci_slot_clamps_to_slot_plus_half_cadence(monkeypatch):
    """The suite pin for a backfill slot must search end <= slot + cadence/2
    (the emit grid's covered-tolerance), never the raw wall clock."""
    seen = {}

    def fake_search(collection, not_after, lookback_h=6.0):
        seen["not_after"] = not_after
        return {"properties": {"identifier": "PID",
                               "date": "2026-07-24T16:30:07Z/2026-07-24T16:39:35Z"}}

    monkeypatch.setattr(MET, "_search_latest", fake_search)
    slot = dt.datetime(2026, 7, 24, 16, 40, tzinfo=UTC)
    got = MET.newest_fci_slot(time=slot)
    assert got == dt.datetime(2026, 7, 24, 16, 39, 35, tzinfo=UTC)
    assert seen["not_after"] == slot + dt.timedelta(minutes=MET.FCI_CADENCE_MIN / 2)


def test_newest_fci_slot_none_when_window_empty(monkeypatch):
    def fake_search(collection, not_after, lookback_h=6.0):
        raise RuntimeError("no product")

    monkeypatch.setattr(MET, "_search_latest", fake_search)
    assert MET.newest_fci_slot(time=dt.datetime(2026, 7, 24, tzinfo=UTC)) is None


def test_fetch_fci_disk_slot_tolerance_rejects_wrong_cycle(monkeypatch):
    """With a pinned time + tolerance, a product from a different repeat
    cycle must be refused BEFORE any download (never render the wrong
    cycle under a backfill slot's stamp)."""
    def fake_search(collection, not_after, lookback_h=6.0):
        return {"properties": {"identifier": "OLD",
                               "date": "2026-07-24T15:30:07Z/2026-07-24T15:39:35Z"}}

    def no_download(*a, **k):
        raise AssertionError("download must not be attempted")

    monkeypatch.setenv("EUMETSAT_CONSUMER_KEY", "k")
    monkeypatch.setenv("EUMETSAT_CONSUMER_SECRET", "s")
    monkeypatch.setattr(MET, "_search_latest", fake_search)
    monkeypatch.setattr(MET, "_download_product", no_download)
    with pytest.raises(RuntimeError, match="no repeat cycle within"):
        MET.fetch_fci_disk(time=dt.datetime(2026, 7, 24, 17, 0, tzinfo=UTC),
                           slot_tolerance_min=MET.FCI_CADENCE_MIN)


def test_fci_band_tokens_align_with_recipes():
    """The numeric convention (1/2/3/4/13) must resolve to the exact satpy
    dataset names the shared pipeline expects -- registry rows, recipes and
    the fetcher all speak through this one table."""
    assert MET.FCI_BAND_TOKENS == {1: "vis_04", 2: "vis_05", 3: "vis_06",
                                   4: "vis_08", 13: "ir_105"}


def test_mtgi1_registry_rows_and_recipes():
    import s2_recipes as rx
    import s2_registry as R
    tc_recipe = rx.FCI_RECIPES_BY_KEY["truecolor"]
    assert tc_recipe.bands == (1, 2, 3, 4, 13)
    assert tc_recipe.finest_km == 1.0            # FDHSI: no 0.5 km band
    assert rx.recipe_for("mtgi1", "irbd").enhancement == "dvorak"
    ids = [e.product_id for e in R.REGISTRY if e.sat_key == "mtgi1"]
    assert ids == ["mtgi1-fd-truecolor", "mtgi1-fd-ir", "mtgi1-fd-irbd"]
    e = R.REGISTRY_BY_ID["mtgi1-fd-truecolor"]
    assert e.family == "mtgi1" and e.sector_key == "fd" and e.tiled
    assert e.sector_bbox == (-80.0, -60.0, 80.0, 60.0)
    idx = R.build_products_index("mtgi1", "fd",
                                 dt.datetime(2026, 7, 24, tzinfo=UTC))
    assert idx["count"] == 3
    assert idx["products"][0]["path"] == "sat/mtgi1/fd/truecolor"


def test_fci_disk_cache_is_negative_too(monkeypatch):
    """Within one suite pass, a failed slot download must not be retried by
    the sibling products -- the failure is cached alongside successes."""
    import s2_imagery as I
    slot = dt.datetime(2026, 7, 24, 16, 39, 35, tzinfo=UTC)
    calls = {"n": 0}

    def failing_fetch(time=None, slot_tolerance_min=None):
        calls["n"] += 1
        raise RuntimeError("HTTP 403: GeneralLicense required")

    monkeypatch.setattr(MET, "newest_fci_slot", lambda time=None: slot)
    monkeypatch.setattr(MET, "fetch_fci_disk", failing_fetch)
    cache = {}
    with pytest.raises(RuntimeError, match="GeneralLicense"):
        I.fetch_fci_disk_cached(time=slot, cache=cache)
    with pytest.raises(RuntimeError, match="already failed this pass"):
        I.fetch_fci_disk_cached(time=slot, cache=cache)
    assert calls["n"] == 1


def test_newest_fci_slot_swallows_transport_errors(monkeypatch):
    def fake_search(collection, not_after, lookback_h=6.0):
        raise MET.requests.ConnectionError("reset by peer")

    monkeypatch.setattr(MET, "_search_latest", fake_search)
    assert MET.newest_fci_slot() is None


def test_pin_suite_scan_mtgi1_uses_fci_slot(monkeypatch):
    import s2_pyramid_emit as E
    import s2_registry as R
    want = dt.datetime(2026, 7, 24, 16, 39, 35, tzinfo=UTC)
    monkeypatch.setattr(MET, "newest_fci_slot", lambda time=None: want)
    entries = [e for e in R.REGISTRY if e.sat_key == "mtgi1"]
    got = E._pin_suite_scan(entries, dt.datetime(2026, 7, 24, 16, 40, tzinfo=UTC))
    assert got == want


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
