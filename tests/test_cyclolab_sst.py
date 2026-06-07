"""Tests for cyclolab_sst - the storm-centered SST hero layers
(FINAL-GATE R2 #1).

The POSITIVE registration contract: a synthetic field with exactly one
hot cell at the storm position must paint at the canvas pixel center
of the rendered PNG. Plus: panel-aspect output, dateline-straddling
box reads, lat-descending (SSTA) normalization, the writer lifecycle
(move/date triggers, png-before-meta ordering, the write_png guard
pair), and the pinned house-colormap mirror.
"""
from __future__ import annotations

import datetime as dt
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import cyclolab_sst as cs  # noqa: E402

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:  # noqa: BLE001
    HAVE_PIL = False

try:
    import netCDF4
    HAVE_NC = True
except Exception:  # noqa: BLE001
    HAVE_NC = False


def _grid(clat, clon, *, step=0.05):
    lats = np.arange(clat - cs.HW_LAT - 0.5, clat + cs.HW_LAT + 0.5, step)
    lons = np.arange(clon - cs.HW_LON - 0.5, clon + cs.HW_LON + 0.5, step)
    return lats, lons


@unittest.skipUnless(HAVE_PIL, "Pillow unavailable")
class TestRegistration(unittest.TestCase):
    """The storm sits at the EXACT pixel center of every layer PNG."""

    def test_hot_cell_at_storm_position_paints_at_canvas_center(self):
        clat, clon = 12.1, -134.9
        lats, lons = _grid(clat, clon)
        data = np.full((lats.size, lons.size), np.nan, dtype=np.float32)
        # one non-NaN cell at the grid node nearest the storm center
        ii = int(np.argmin(np.abs(lats - clat)))
        jj = int(np.argmin(np.abs(lons - clon)))
        data[ii, jj] = 28.0
        png = cs.render_hero_layer(cs.LAYERS[0], data, lats, lons,
                                   basin="EP", clat=clat, clon=clon)
        im = Image.open(io.BytesIO(png)).convert("RGB")
        self.assertEqual(im.size, (1200, 690))
        px = np.asarray(im, dtype=np.int16)
        land = np.array([0x5f, 0x6b, 0x7a], dtype=np.int16)
        colored = (np.abs(px - land).sum(axis=2) > 40)
        ys, xs = np.nonzero(colored)
        self.assertGreater(ys.size, 0, "the hot cell did not paint")
        cx, cy = xs.mean(), ys.mean()
        # one 0.05-degree cell is ~3.3 px wide; grid-node snap adds up
        # to half a cell each axis. 5 px tolerance pins the contract.
        self.assertLess(abs(cx - 600), 5.0,
                        f"hot-cell centroid x={cx:.1f}, want 600")
        self.assertLess(abs(cy - 345), 5.0,
                        f"hot-cell centroid y={cy:.1f}, want 345")

    def test_render_is_panel_aspect_both_layers(self):
        clat, clon = 12.1, -134.9
        lats, lons = _grid(clat, clon)
        rng = np.random.default_rng(7)
        for layer, base in ((cs.LAYERS[0], 27.5), (cs.LAYERS[1], 0.6)):
            data = np.full((lats.size, lons.size), base, dtype=np.float32)
            data += rng.normal(0, 0.5, data.shape).astype(np.float32)
            png = cs.render_hero_layer(layer, data, lats, lons,
                                       basin="EP", clat=clat, clon=clon)
            im = Image.open(io.BytesIO(png))
            self.assertEqual(im.size, (1200, 690), layer["slug"])
            self.assertGreater(len(png), 20_000, layer["slug"])


@unittest.skipUnless(HAVE_NC, "netCDF4 unavailable")
class TestBoxReader(unittest.TestCase):
    """Partial box reads: dateline wrap + lat-descending handling."""

    def _write_nc(self, path, lats, lons, fn, *, time_dim=True):
        ds = netCDF4.Dataset(str(path), "w")
        ds.createDimension("lat", lats.size)
        ds.createDimension("lon", lons.size)
        dims = ("lat", "lon")
        if time_dim:
            ds.createDimension("time", 1)
            dims = ("time", "lat", "lon")
        ds.createVariable("lat", "f8", ("lat",))[:] = lats
        ds.createVariable("lon", "f8", ("lon",))[:] = lons
        v = ds.createVariable("analysed_sst", "f4", dims)
        LON2, LAT2 = np.meshgrid(lons, lats)
        field = fn(LAT2, LON2).astype(np.float32)
        v[:] = field[None, ...] if time_dim else field
        ds.close()

    def test_dateline_box_is_contiguous_and_seamless(self):
        # global -180..180 grid, smooth function of folded lon - a box
        # at clon=179.8 must read as ONE monotone stitched block.
        lats = np.arange(-20, 20.0001, 0.5)
        lons = np.arange(-180, 180, 0.5)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "g.nc"
            self._write_nc(
                p, lats, lons,
                lambda la, lo: np.cos(np.radians(lo - 179.8)) * 10 + la)
            data, ola, olo = cs.read_crw_box(p, "analysed_sst",
                                             5.0, 179.8)
        self.assertTrue(np.all(np.diff(olo) > 0),
                        "display lons must be monotone across the wrap")
        self.assertGreater(olo.max(), 180.0,
                           "the box should extend past +180 in the "
                           "display frame")
        # seamless: the stitched values equal the smooth function
        LON2, LAT2 = np.meshgrid(olo, ola)
        want = np.cos(np.radians(LON2 - 179.8)) * 10 + LAT2
        self.assertLess(float(np.nanmax(np.abs(data - want))), 1e-3)

    def test_descending_lat_is_normalized_ascending(self):
        # the SSTA product stores lat DESCENDING - rows must flip.
        lats = np.arange(30, -30.0001, -0.5)     # descending
        lons = np.arange(-180, 180, 0.5)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "g.nc"
            self._write_nc(p, lats, lons, lambda la, lo: la,
                           time_dim=False)
            data, ola, olo = cs.read_crw_box(p, "analysed_sst",
                                             5.0, -135.0)
        self.assertTrue(np.all(np.diff(ola) > 0), "lats must ascend")
        # row values == their own latitude (proves rows flipped WITH
        # the axis, not independently)
        self.assertLess(
            float(np.nanmax(np.abs(data - ola[:, None]))), 1e-3)


class _PngSink:
    """write + write_png recorder preserving write ORDER."""

    def __init__(self):
        self.log = []          # (kind, key)
        self.json = {}
        self.png = {}

    def write(self, key, payload):
        self.log.append(("json", key))
        self.json[key] = payload

    def write_png(self, key, data, cache=None):
        self.log.append(("png", key))
        self.png[key] = data


class _JsonOnlySink:
    def __init__(self):
        self.json = {}

    def write(self, key, payload):
        self.json[key] = payload


def _fake_io(available=("sst",), climo=True):
    """(fetch_day, read_box, fetch_climo) fakes - no network, no netCDF.

    final-gate-3 #6a: the anomaly layer now reads the SST file + the
    house DOY-climo grid (no official-SSTA product). The climo box must
    share the SST box's lat/lon vectors so the subtraction lands.
    """
    def fetch_day(product, d, cache_dir, session=None, read_timeout=None):
        if product in available and d == dt.date(2026, 6, 5):
            return Path(f"/fake/crw_{product}.nc")
        return None

    def fetch_climo(d, cache_dir=None, session=None, read_timeout=None):
        return Path("/fake/crw_doy_climo.nc") if climo else None

    def read_box(path, var, clat, clon, hw_lat=cs.HW_LAT,
                 hw_lon=cs.HW_LON):
        lats, lons = _grid(clat, clon, step=0.5)
        # SST ~27 C, climo ~26.4 C -> anomaly ~+0.6 C; same grid so the
        # writer's alignment guard passes.
        base = 26.4 if "climo" in str(path) else 27.0
        rng = np.random.default_rng(3 if "climo" in str(path) else 4)
        data = rng.normal(base, 0.3,
                          (lats.size, lons.size)).astype(np.float32)
        return data, lats, lons

    return fetch_day, read_box, fetch_climo


def _storm(lat=12.1, lon=-134.9):
    return {"sid": "NHC_EP012026", "name": "AMANDA",
            "current_category": "TD",
            "points": [{"t": "2026-06-07T00:00:00", "lat": lat,
                        "lon": lon, "wind_kt": 30}]}


@unittest.skipUnless(HAVE_PIL, "Pillow unavailable")
class TestWriterLifecycle(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)

    def _writer(self, sink, available=("sst",), climo=True,
                climo_ready=True):
        fetch_day, read_box, fetch_climo = _fake_io(available, climo)
        return cs.SstHeroWriter(
            sink, prefix="shadow/cyclolab", cache_dir=Path(self._td.name),
            fetch_day=fetch_day, read_box=read_box, fetch_climo=fetch_climo,
            climo_ready=climo_ready, today=lambda: dt.date(2026, 6, 6))

    def test_writes_layers_then_meta_in_order(self):
        sink = _PngSink()
        w = self._writer(sink)
        w.maybe_render("NHC_EP012026", _storm(), "EP")
        kinds = [k for k, _ in sink.log]
        self.assertEqual(kinds, ["png", "png", "json"],
                         "PNGs must write BEFORE meta (the commit)")
        keys = [k for _, k in sink.log]
        self.assertEqual(keys, [
            "shadow/cyclolab/NHC_EP012026/sst/actual.png",
            "shadow/cyclolab/NHC_EP012026/sst/anomaly.png",
            "shadow/cyclolab/NHC_EP012026/sst/meta.json"])
        self.assertTrue(sink.png[keys[0]].startswith(b"\x89PNG"))
        meta = sink.json[keys[2]]
        self.assertEqual([x["slug"] for x in meta["layers"]],
                         ["actual", "anomaly"])
        self.assertEqual(meta["valid_date"], "2026-06-05")
        self.assertEqual(meta["layers"][0]["valid"], "2026-06-05")
        # final-gate-3 #6a, ONE CANON: the anomaly layer carries the
        # site-wide baseline in its caption and NO divergence note.
        anom = meta["layers"][1]
        self.assertNotIn("note", anom)
        self.assertIn("1991", anom["field"])
        self.assertIn("1991", anom["source"])
        self.assertNotIn("official CRW climatology", anom["source"])
        self.assertEqual(meta["center"], {"lat": 12.1, "lon": -134.9})
        self.assertEqual(meta["px"], [1200, 690])

    def test_anomaly_hidden_until_the_bake_lands(self):
        # the bake-complete GATE: no manifest -> the anomaly layer is
        # simply not written (hidden, never a wrong-baseline render).
        sink = _PngSink()
        w = self._writer(sink, climo_ready=False)
        w.maybe_render("NHC_EP012026", _storm(), "EP")
        meta = sink.json["shadow/cyclolab/NHC_EP012026/sst/meta.json"]
        self.assertEqual([x["slug"] for x in meta["layers"]], ["actual"])

    def test_anomaly_hidden_when_climo_grid_missing(self):
        # gate is open but the DOY grid 404s (off-belt / not yet on R2):
        # still hidden, never broken.
        sink = _PngSink()
        w = self._writer(sink, climo=False, climo_ready=True)
        w.maybe_render("NHC_EP012026", _storm(), "EP")
        meta = sink.json["shadow/cyclolab/NHC_EP012026/sst/meta.json"]
        self.assertEqual([x["slug"] for x in meta["layers"]], ["actual"])

    def test_unmoved_same_day_skips_then_move_rerenders(self):
        sink = _PngSink()
        w = self._writer(sink)
        w.maybe_render("NHC_EP012026", _storm(), "EP")
        n = len(sink.log)
        w.maybe_render("NHC_EP012026", _storm(lat=12.15), "EP")
        self.assertEqual(len(sink.log), n, "0.05 deg is not a move")
        w.maybe_render("NHC_EP012026", _storm(lat=12.6), "EP")
        self.assertGreater(len(sink.log), n, "0.5 deg IS a move")

    def test_missing_sst_file_writes_nothing(self):
        # the SST file is the source of BOTH layers now - absent -> no
        # family at all (replaces the old ssta-missing case, which is
        # covered by the two climo-gate tests above).
        sink = _PngSink()
        w = self._writer(sink, available=())
        w.maybe_render("NHC_EP012026", _storm(), "EP")
        self.assertEqual(sink.json, {})

    def test_json_only_sink_writes_nothing(self):
        sink = _JsonOnlySink()
        w = self._writer(sink)
        w.maybe_render("NHC_EP012026", _storm(), "EP")
        self.assertEqual(sink.json, {},
                         "no write_png -> no family at all (a meta "
                         "without PNGs would advertise 404s)")

    def test_never_raises_on_io_failure(self):
        sink = _PngSink()

        def boom(product, d, cache_dir, session=None):
            raise RuntimeError("upstream down")

        with tempfile.TemporaryDirectory() as td:
            w = cs.SstHeroWriter(sink, prefix="shadow/cyclolab",
                                 cache_dir=Path(td), fetch_day=boom,
                                 today=lambda: dt.date(2026, 6, 6))
            w.maybe_render("NHC_EP012026", _storm(), "EP")   # must not raise
        self.assertEqual(sink.log, [])


class TestShellFixtureMirrorsWriter(unittest.TestCase):
    """The jsdom suite's SST_META fixture must stay a faithful mirror
    of what SstHeroWriter actually writes (adversarial-review find: a
    hand-fixture can drift and the picker tests would keep passing
    against a shape the poller never produces)."""

    def test_fixture_layers_match_writer_meta(self):
        from test_cyclolab_shell import SST_META
        sink = _PngSink()
        fetch_day, read_box, fetch_climo = _fake_io()
        with tempfile.TemporaryDirectory() as td:
            w = cs.SstHeroWriter(
                sink, prefix="shadow/cyclolab", cache_dir=Path(td),
                fetch_day=fetch_day, read_box=read_box,
                fetch_climo=fetch_climo, climo_ready=True,
                today=lambda: dt.date(2026, 6, 6))
            w.maybe_render("NHC_EP012026", _storm(), "EP")
        meta = sink.json["shadow/cyclolab/NHC_EP012026/sst/meta.json"]
        fix_layers = SST_META["layers"]
        real_layers = meta["layers"]
        self.assertEqual([x["slug"] for x in fix_layers],
                         [x["slug"] for x in real_layers])
        for fx, rl in zip(fix_layers, real_layers):
            for key in ("label", "title", "field", "file", "source"):
                self.assertEqual(fx[key], rl[key],
                                 f"{fx['slug']}.{key} drifted")
            self.assertEqual(fx.get("note"), rl.get("note"),
                             f"{fx['slug']}.note drifted")
        # the shell fixture's top-level keys the picker reads
        for key in ("valid_date", "updated_utc"):
            self.assertIn(key, meta)


class TestColormapContract(unittest.TestCase):
    """The ramps are byte-pinned MIRRORS of the house generator
    (generate_sst_plots.py _sst_actual_cmap/_sst_anom_cmap). If the
    house ramp ever changes, change BOTH and this pin."""

    def _close(self, got, want, label, delta=3):
        # the 256-bin LUT quantizes stop positions by up to ~2/channel;
        # ±3 still uniquely pins the ramp identity.
        for g, w, ch in zip(got, want, "rgb"):
            self.assertAlmostEqual(
                g, w, delta=delta,
                msg=f"{label} channel {ch}: {got} vs {want}")

    def test_actual_endpoint_and_midpoint_colors(self):
        cm = cs.CMAP_ACTUAL
        # stop 0 = #2c0b4a, stop 1.0 = #6b0d18, 0.50 = #6bd98e
        for x, want in ((0.0, (0x2c, 0x0b, 0x4a)),
                        (0.5, (0x6b, 0xd9, 0x8e)),
                        (1.0, (0x6b, 0x0d, 0x18))):
            got = tuple(int(round(c * 255)) for c in cm(x)[:3])
            self._close(got, want, f"sst_actual stop {x}")

    def test_anom_center_is_white_and_bad_is_land_gray(self):
        cm = cs.CMAP_ANOM
        # the white stop (0.50) sits between 256-bin LUT centers (its
        # neighbors are only 0.005 away), so the sampled bin averages
        # toward its neighbors - same quantization as the house
        # generator. Near-white pins the diverging center.
        got = tuple(int(round(c * 255)) for c in cm(0.5)[:3])
        self.assertGreaterEqual(min(got), 245, f"center not white: {got}")
        bad = tuple(int(round(c * 255)) for c in cm.get_bad()[:3])
        self.assertEqual(bad, (0x5f, 0x6b, 0x7a))


if __name__ == "__main__":
    unittest.main(verbosity=2)
