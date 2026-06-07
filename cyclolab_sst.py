"""CycloLab storm-centered SST hero layers (FINAL-GATE R2 #1).

The Overview hero used to client-crop the site's full-basin CRW PNG -
unacceptable resolution once zoomed to a storm box. This module renders
DEDICATED per-storm products from the SOURCE data instead: the same
NOAA Coral Reef Watch CoralTemp v3.1 5 km daily NetCDF the house SST
pipeline consumes (generate_sst_plots.py in the main repo), cropped to
a storm-centered box and drawn with the house recipe at native 5 km
detail - house colormaps, NaN-gray land, white Natural-Earth coasts,
clean isotherms WITH inline clabel'd °C values (the hero has no
colorbar; the labels are the scale).

REGISTRATION CONTRACT: the figure is FULL-BLEED (axes span the whole
canvas, no ticks/margins) and the extent is exactly storm-centered, so
the storm sits at the precise pixel center of every layer PNG. The
shell pins the spinning category glyph at 50%/50% - no client crop
math exists anymore. The box aspect (18° x 10.35°) equals the panel's
16/9.2, so object-fit:cover is a 1:1 mapping. Pinned by the
registration test (a synthetic field with one hot cell at the storm
position must paint at the canvas center).

LAYERS: 'actual' = analysed_sst from the daily CoralTemp file - the
house recipe verbatim (0-32 °C, sst_actual ramp). 'anomaly' = the
OFFICIAL CRW SSTA daily product (sea_surface_temperature_anomaly,
±5 °C, sst_anom ramp). HONESTY NOTE: the house full-basin anomaly maps
recompute anomaly against a 1991-2020 same-DOY climatology built from
30 historical daily files - infeasible inside the poller (≈30 × ~80 MB
downloads per day). The official CRW SSTA file is a single daily
download of the SAME CoralTemp field against CRW's OWN climatology;
the difference is disclosed in the layer's caption note, never hidden.

COLORMAPS are byte-pinned MIRRORS of the house generator
(generate_sst_plots.py _sst_actual_cmap/_sst_anom_cmap lines 758-812
@ main) - the stops are the cross-repo contract, pinned in
tests/test_cyclolab_sst.py. Candidate post-go-live cleanup: move both
into the shared tat_palettes package and import them on both sides.

Writes (SstHeroWriter, riding CycloLabPageWriter's per-fix cadence):
    {prefix}/{sid}/sst/{layer}.png    R2Sink.write_png
    {prefix}/{sid}/sst/meta.json      Sink.write   (written LAST - the
                                      shell treats meta as the commit)
Kill switch: CYCLOLAB_SST (default on). Best-effort by contract.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap, Normalize)
import matplotlib.patheffects as pe  # noqa: E402

from cyclolab_basemap import basemap_for  # noqa: E402

log = logging.getLogger("cyclolab-sst")

SST_ENABLED = (os.environ.get("CYCLOLAB_SST") or "1").lower() \
    not in ("0", "false", "no")

# ---------------------------------------------------------------------------
# Source contract (the house CRW pipeline's, mirrored)
# ---------------------------------------------------------------------------
CRW_BASE = ("https://www.star.nesdis.noaa.gov/pub/sod/mecb/crw/data/5km/"
            "v3.1_op/nc/v1.0/daily")
# Site CDN: the house 1991-2020 day-of-year SST climatology (final-gate-3
# #6a, ONE CANON). 366 grids baked by build_crw_doy_climatology.py to
# R2 sst/climo/, native CRW grid + -180..180 lon (so the box-read math
# below is unchanged), int16 packed -> netCDF4 auto-unpacks to degC.
# The manifest is the GATE: present only when all 366 grids exist, so a
# partial bake can never serve a wrong-baseline anomaly.
CDN_BASE = "https://cdn.triple-a-tropics.com"
CLIMO_BASE = f"{CDN_BASE}/sst/climo"
CLIMO_MANIFEST_URL = f"{CLIMO_BASE}/manifest.json"
CLIMO_FILE = "crw_doy_climo_1991_2020_{doy:03d}.nc"
CLIMO_VAR = "sst_climo"
CLIMO_PROBE_TTL_S = 3600          # re-check the bake-complete gate hourly
FETCH_CONNECT_TIMEOUT = 8
FETCH_TIMEOUT = 45
LATENCY_TRIES = 7
LATEST_PROBE_TTL_S = 1800        # re-probe upstream at most every 30 min
CACHE_KEEP_DAYS = 3
# HARD per-call wall-clock budget (adversarial-review find): the writer
# runs INLINE on the single poll thread - a cold/slow CRW upstream must
# never stall the other sources or the health heartbeat for minutes.
CALL_BUDGET_S = 90
PARTIAL_RETRY_S = 600            # re-try a missing layer (SSTA lag) after

# ---------------------------------------------------------------------------
# Render geometry: full-bleed, panel-aspect, storm-centered
# ---------------------------------------------------------------------------
FIG_W_IN, FIG_H_IN, DPI = 8.0, 4.6, 150      # -> 1200 x 690 px
HW_LON = 9.0                                  # box half-width, degrees
HW_LAT = HW_LON * (FIG_H_IN / FIG_W_IN)       # 5.175 - equal deg/px x and y
MOVE_TRIG_DEG = 0.25                          # re-render when moved this far

LAND_COLOR = "#5f6b7a"                        # house NaN/land gray
COAST_COLOR = "#ffffff"                       # house coastline white


def _sst_actual_cmap() -> LinearSegmentedColormap:
    """MIRROR of the house sst_actual ramp (generate_sst_plots.py
    _sst_actual_cmap) - same 11 stops, same order. Keep in sync."""
    stops = [
        (0.00, "#2c0b4a"), (0.08, "#2a1794"), (0.18, "#2f4bc4"),
        (0.28, "#2e8bd0"), (0.38, "#2fc4c9"), (0.50, "#6bd98e"),
        (0.62, "#e7ee5f"), (0.72, "#f5b23d"), (0.82, "#e84b2a"),
        (0.92, "#b01a26"), (1.00, "#6b0d18"),
    ]
    cm = LinearSegmentedColormap.from_list("sst_actual", stops, N=256)
    cm.set_bad(color=LAND_COLOR, alpha=1.0)
    return cm


def _sst_anom_cmap() -> LinearSegmentedColormap:
    """MIRROR of the house sst_anom ramp (generate_sst_plots.py
    _sst_anom_cmap) - same 17 stops, same order. Keep in sync."""
    stops = [
        (0.00, "#1a0c5f"), (0.08, "#1a2b9e"), (0.18, "#2261c7"),
        (0.30, "#4695db"), (0.40, "#8bc0ea"), (0.47, "#cde5f5"),
        (0.495, "#f2f7fb"), (0.50, "#ffffff"), (0.506, "#fdf4ea"),
        (0.53, "#f8d5b8"), (0.58, "#efac86"), (0.65, "#df815f"),
        (0.73, "#cc4836"), (0.82, "#9f1e26"), (0.90, "#6d1321"),
        (0.96, "#3f0c23"), (1.00, "#ef37b8"),
    ]
    cm = LinearSegmentedColormap.from_list("sst_anom", stops, N=256)
    cm.set_bad(color=LAND_COLOR, alpha=1.0)
    return cm


CMAP_ACTUAL = _sst_actual_cmap()
CMAP_ANOM = _sst_anom_cmap()

# The hero layer registry. file/slug are the R2 key leaves; note is the
# per-layer disclosure the shell appends to the caption.
LAYERS = [
    {
        "slug": "actual",
        "label": "SST",
        "title": "Sea surface temperature",
        "field": "sea-surface temperature (°C)",
        "file": "actual.png",
        "product": "sst",
        "var": "analysed_sst",
        "source": "NOAA Coral Reef Watch CoralTemp v3.1 (5 km)",
    },
    {
        "slug": "anomaly",
        "label": "Anomaly",
        "title": "SST anomaly",
        "field": "SST anomaly vs the site-wide 1991–2020 baseline (°C)",
        "file": "anomaly.png",
        # final-gate-3 #6a, ONE CANON: today's CoralTemp SST MINUS the
        # site's own 1991-2020 day-of-year climatology (the same baseline
        # /sst/ uses). Computed in the writer from the SST file + the
        # house climo grid - no official-SSTA product, no divergence note.
        "compute": "house_anomaly",
        "product": "sst",
        "var": "analysed_sst",
        "source": ("NOAA Coral Reef Watch CoralTemp v3.1 (5 km) minus the "
                   "Triple-A-Tropics 1991–2020 day-of-year "
                   "climatology"),
    },
]


def crw_url_for(product: str, d: dt.date) -> str:
    """House URL contract (generate_sst_plots.crw_url_for)."""
    if product == "sst":
        fname = f"coraltemp_v3.1_{d:%Y%m%d}.nc"
    else:
        fname = f"ct5km_{product}_v3.1_{d:%Y%m%d}.nc"
    return f"{CRW_BASE}/{product}/{d:%Y}/{fname}"


def fetch_crw_day(product: str, d: dt.date, cache_dir: Path,
                  session=None, read_timeout: float = FETCH_TIMEOUT
                  ) -> Path | None:
    """Download one CRW daily file, disk-cached (house contract:
    >100 KB validity guard, 404 -> None; connect timeout is short so a
    black-holed upstream can't eat the poll thread's budget)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"crw_{product}.{d:%Y%m%d}.nc"
    if cp.exists() and cp.stat().st_size > 100_000:
        return cp
    if session is None:
        import requests as session  # noqa: PLC0415
    url = crw_url_for(product, d)
    try:
        r = session.get(url, timeout=(FETCH_CONNECT_TIMEOUT,
                                      max(5.0, read_timeout)))
    except Exception as e:  # noqa: BLE001
        log.warning("CRW %s %s fetch error: %s", product, d, e)
        return None
    if r.status_code != 200 or len(r.content) < 100_000:
        return None
    tmp = cp.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(cp)
    return cp


def climo_doy(d: dt.date) -> int:
    """Day-of-year index for the house climo bake (final-gate-3 #6a).
    EXACTLY the bake's contract: a leap-keyed (month, day) -> 1..366 map
    so Feb-29 has its own slot and every other day is stable across
    years. Pinned to build_crw_doy_climatology._doy_of."""
    return dt.date(2000, d.month, d.day).timetuple().tm_yday


def fetch_climo_doy(d: dt.date, cache_dir: Path, session=None,
                    read_timeout: float = FETCH_TIMEOUT) -> Path | None:
    """Download the house 1991-2020 climatology grid for ``d``'s
    day-of-year from R2, disk-cached. Same validity/404/timeout contract
    as fetch_crw_day; returns None on any miss so the anomaly layer
    degrades cleanly to 'not shown'."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    doy = climo_doy(d)
    cp = cache_dir / f"crw_doy_climo.{doy:03d}.nc"
    if cp.exists() and cp.stat().st_size > 100_000:
        return cp
    if session is None:
        import requests as session  # noqa: PLC0415
    url = f"{CLIMO_BASE}/{CLIMO_FILE.format(doy=doy)}"
    try:
        r = session.get(url, timeout=(FETCH_CONNECT_TIMEOUT,
                                      max(5.0, read_timeout)))
    except Exception as e:  # noqa: BLE001
        log.warning("climo DOY %03d fetch error: %s", doy, e)
        return None
    if r.status_code != 200 or len(r.content) < 100_000:
        return None
    tmp = cp.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(cp)
    return cp


def prune_cache(cache_dir: Path, *, today: dt.date) -> None:
    cutoff = today - dt.timedelta(days=CACHE_KEEP_DAYS)
    for p in list(cache_dir.glob("crw_*.nc")) + \
            list(cache_dir.glob("crw_*.part")):
        try:
            # interrupted downloads leave .part orphans - age them out
            # with the same date stamp (crw_{prod}.{YYYYMMDD}.{ext}).
            stamp = dt.datetime.strptime(
                p.name.rsplit(".", 2)[-2], "%Y%m%d").date()
            if stamp < cutoff:
                p.unlink()
        except Exception:  # noqa: BLE001 - unparseable names: leave them
            pass


def read_crw_box(path: Path, var_name: str, clat: float, clon: float,
                 hw_lat: float = HW_LAT, hw_lon: float = HW_LON):
    """Partial-read a storm-centered box from a CRW daily file.

    Returns (data, lats, lons) with lats ASCENDING and lons in the
    storm's CONTIGUOUS display frame (clon ± hw_lon, may cross the
    antimeridian - the native -180..180 vector is re-framed and a
    wrapping box is read as two index blocks). Only the box is read
    from disk - never the 7200x3600 global grid.
    """
    import netCDF4  # local import: keep module import light

    ds = netCDF4.Dataset(str(path))
    try:
        if var_name in ds.variables:
            v = ds.variables[var_name]
        else:
            # NO loose fallback (adversarial-review find): the CRW
            # files also carry mask/analysis_error/sea_ice_fraction -
            # silently rendering one of those as the field would be a
            # lie. Fail loudly; the per-layer isolation logs + skips.
            raise KeyError(
                f"variable {var_name!r} missing from {path.name} "
                f"(has: {', '.join(ds.variables)})")
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)

        flipped = lat[0] > lat[-1]            # SSTA stores lat descending
        lat_mask = (lat >= clat - hw_lat) & (lat <= clat + hw_lat)
        ii = np.nonzero(lat_mask)[0]
        if ii.size == 0:
            raise ValueError("storm box outside the CRW lat range")
        i0, i1 = int(ii[0]), int(ii[-1]) + 1

        # display frame: degrees east of clon, folded into (-180, 180]
        d = ((lon - clon + 180.0) % 360.0) - 180.0
        lon_mask = np.abs(d) <= hw_lon
        jj = np.nonzero(lon_mask)[0]
        if jj.size == 0:
            raise ValueError("storm box selects no CRW lons")
        # contiguous index blocks (a dateline-straddling box wraps)
        breaks = np.nonzero(np.diff(jj) > 1)[0]
        blocks = np.split(jj, breaks + 1)

        def _read(j0: int, j1: int) -> np.ndarray:
            arr = v[0, i0:i1, j0:j1] if v.ndim == 3 else v[i0:i1, j0:j1]
            return np.ma.filled(
                np.ma.masked_invalid(np.ma.asarray(arr)),
                np.nan).astype(np.float32)

        # order blocks by display lon so the stitched array is monotone
        blocks.sort(key=lambda b: d[b[0]])
        data = np.hstack([_read(int(b[0]), int(b[-1]) + 1)
                          for b in blocks])
        lons = clon + np.hstack([d[b] for b in blocks])
        lats = lat[i0:i1]
        if flipped:
            data = data[::-1, :]
            lats = lats[::-1]
        return data, lats, lons
    finally:
        ds.close()


def render_hero_layer(layer: dict, data: np.ndarray, lats: np.ndarray,
                      lons: np.ndarray, *, basin: str, clat: float,
                      clon: float) -> bytes:
    """Render ONE storm-centered hero layer PNG (full-bleed; the storm
    at the exact canvas center). Pure - returns PNG bytes. The figure
    is ALWAYS closed (try/finally) - a raise mid-render must not leak
    figures into the long-lived poller process."""
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    try:
        return _render_hero_layer(fig, layer, data, lats, lons,
                                  basin=basin, clat=clat, clon=clon)
    finally:
        plt.close(fig)


def _render_hero_layer(fig, layer: dict, data: np.ndarray,
                       lats: np.ndarray, lons: np.ndarray, *,
                       basin: str, clat: float, clon: float) -> bytes:
    ax = fig.add_axes([0, 0, 1, 1])           # FULL BLEED: registration
    ax.set_facecolor(LAND_COLOR)              # no-data == land gray
    LON2, LAT2 = np.meshgrid(lons, lats)

    if layer["slug"] == "actual":
        norm = Normalize(vmin=0.0, vmax=32.0)
        cmap = CMAP_ACTUAL
    else:
        norm = Normalize(vmin=-5.0, vmax=5.0)
        cmap = CMAP_ANOM
    ax.pcolormesh(LON2, LAT2, data, cmap=cmap, norm=norm,
                  shading="auto", zorder=1, rasterized=True)

    # isotherms WITH inline labels (final-gate-2 #1: "clean isotherms
    # with inline °C labels") - the house clabel treatment (black thin
    # lines, bold labels with a white halo), level steps tuned to the
    # storm box where there is no colorbar to read against.
    halo = [pe.withStroke(linewidth=1.6, foreground="#ffffff")]
    try:
        if layer["slug"] == "actual":
            cs = ax.contour(LON2, LAT2, data, levels=np.arange(0, 33, 1),
                            colors="#000000", linewidths=0.45,
                            alpha=0.65, zorder=1.5)
            labels = ax.clabel(cs, inline=True, inline_spacing=3,
                               fontsize=8, fmt="%d°", colors="#000000")
        else:
            ax.contour(LON2, LAT2, data, levels=[0.0], colors="#ffffff",
                       linewidths=0.5, alpha=0.5, zorder=1.6)
            lv = [x / 2.0 for x in range(-10, 11) if x != 0]
            cs = ax.contour(LON2, LAT2, data, levels=lv,
                            colors="#000000", linewidths=0.45,
                            alpha=0.65, zorder=1.65)
            labels = ax.clabel(cs, inline=True, inline_spacing=3,
                               fontsize=7, fmt="%+.1f", colors="#000000")
        for t in labels:
            t.set_fontweight("bold")
            t.set_path_effects(halo)
    except Exception as e:  # noqa: BLE001 - flat fields can't contour
        log.debug("contours skipped (%s): %s", layer["slug"], e)

    # house white coastlines from the vendored Natural Earth land
    # (cyclolab_basemap clips + lon-normalizes around the same center,
    # so the rings land in this plot's display frame).
    try:
        bm = basemap_for(clat, clon, basin,
                         dlat=HW_LAT + 0.6, dlon=HW_LON + 0.6)
        for ring in bm["land"]:
            xs = [p[0] for p in ring] + [ring[0][0]]
            ys = [p[1] for p in ring] + [ring[0][1]]
            ax.plot(xs, ys, color=COAST_COLOR, linewidth=0.8, zorder=3)
    except Exception as e:  # noqa: BLE001
        log.warning("coastlines skipped: %s", e)

    # exact storm-centered extent == exact pixel-center registration
    ax.set_xlim(clon - HW_LON, clon + HW_LON)
    ax.set_ylim(clat - HW_LAT, clat + HW_LAT)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    return buf.getvalue()       # caller's finally closes the figure


class SstHeroWriter:
    """Per-storm SST hero layer lifecycle (best-effort by contract).

    Rides CycloLabPageWriter's per-fix cadence: maybe_render(sid, storm)
    re-renders when the storm has MOVED >= MOVE_TRIG_DEG, the CRW day
    has advanced, or nothing has been written yet. PNGs write first,
    meta.json LAST (the shell's commit point), so a half-written family
    is never advertised.
    """

    def __init__(self, sink, *, prefix: str, cache_dir: Path | None = None,
                 fetch_day=fetch_crw_day, read_box=read_crw_box,
                 fetch_climo=fetch_climo_doy, climo_ready=None,
                 today=None):
        self.sink = sink
        self.prefix = prefix.rstrip("/")
        self.cache_dir = cache_dir or (
            Path(tempfile.gettempdir()) / "cyclolab-crw-cache")
        self._fetch_day = fetch_day
        self._read_box = read_box
        # final-gate-3 #6a injection points (tests drive these offline):
        # the house-climo fetch and the bake-complete gate override.
        self._fetch_climo = fetch_climo
        self._climo_ready_override = climo_ready
        self._today = today or (lambda: dt.datetime.now(dt.UTC).date())
        # sid -> {"date": iso, "lat": f, "lon": f,
        #         "layers": [slugs], "t": monotonic}
        self._state: dict[str, dict] = {}
        # product -> (probed_monotonic, date|None) latest-day cache
        self._latest: dict[str, tuple[float, dt.date | None]] = {}
        self._deadline = 0.0     # per-call wall-clock budget marker

    def _remaining(self) -> float:
        import time
        return self._deadline - time.monotonic()

    def _fetch_budgeted(self, product: str, d: dt.date) -> Path | None:
        rem = self._remaining()
        if rem <= 2:
            return None          # budget spent: the next poll retries
        try:
            return self._fetch_day(product, d, self.cache_dir,
                                   read_timeout=min(FETCH_TIMEOUT, rem))
        except TypeError:        # injected test fakes without the kwarg
            return self._fetch_day(product, d, self.cache_dir)

    def _climo_ready(self) -> bool:
        """The house-climo bake-complete GATE (final-gate-3 #6a): the
        manifest exists only when all 366 grids are on R2. Hourly-cached
        + best-effort - until the bake lands, the anomaly layer is simply
        not written (hidden, never a wrong-baseline render)."""
        if self._climo_ready_override is not None:
            return bool(self._climo_ready_override)
        import time
        hit = getattr(self, "_climo_gate", None)
        if hit and time.monotonic() - hit[0] < CLIMO_PROBE_TTL_S:
            return hit[1]
        ok = False
        try:
            import requests
            r = requests.get(CLIMO_MANIFEST_URL,
                             timeout=(FETCH_CONNECT_TIMEOUT, 15))
            ok = (r.status_code == 200
                  and int((r.json() or {}).get("doy_count", 0)) >= 366)
        except Exception as e:  # noqa: BLE001
            log.debug("climo manifest probe failed: %s", e)
        self._climo_gate = (time.monotonic(), ok)
        return ok

    def _house_anomaly_box(self, sst_path, sst_day, clat, clon):
        """today's SST box minus the house DOY-climo box, on the SAME
        native CRW grid (both -180..180, 0.05deg). Returns (anom, lats,
        lons) or None when the climo grid is unavailable / off-belt /
        misaligned - the caller then omits the layer."""
        try:
            climo_path = self._fetch_climo(sst_day, self.cache_dir)
        except TypeError:        # injected fakes without the kwargs
            climo_path = self._fetch_climo(sst_day)
        if climo_path is None:
            return None
        sst, lats, lons = self._read_box(sst_path, "analysed_sst",
                                         clat, clon)
        cli, clats, clons = self._read_box(climo_path, CLIMO_VAR,
                                           clat, clon)
        if (sst.shape != cli.shape
                or not np.allclose(lats, clats, atol=1e-4)
                or not np.allclose(lons, clons, atol=1e-4)):
            log.warning("house anomaly: SST/climo grids misaligned "
                        "(%s vs %s) - layer skipped", sst.shape, cli.shape)
            return None
        return np.asarray(sst) - np.asarray(cli), lats, lons

    def _latest_day(self, product: str) -> dt.date | None:
        import time
        hit = self._latest.get(product)
        if hit and time.monotonic() - hit[0] < LATEST_PROBE_TTL_S:
            return hit[1]
        found = None
        d = self._today() - dt.timedelta(days=1)
        for _ in range(LATENCY_TRIES):
            if self._remaining() <= 2:
                # budget spent mid-walk-back: DON'T negative-cache the
                # probe - the next call re-tries with a fresh budget.
                return None
            if self._fetch_budgeted(product, d) is not None:
                found = d
                break
            d -= dt.timedelta(days=1)
        self._latest[product] = (time.monotonic(), found)
        return found

    def maybe_render(self, sid: str, storm: dict, basin: str) -> None:
        """NEVER raises (best-effort contract, like the page writer)."""
        try:
            self._maybe_render(sid, storm, basin)
        except Exception as e:  # noqa: BLE001
            log.warning("sst hero render failed (%s): %s", sid, e)

    def _maybe_render(self, sid: str, storm: dict, basin: str) -> None:
        if not SST_ENABLED or not hasattr(self.sink, "write_png"):
            return
        import time
        pts = storm.get("points") or []
        last = pts[-1] if pts else {}
        if last.get("lat") is None or last.get("lon") is None:
            return
        clat, clon = float(last["lat"]), float(last["lon"])
        st = self._state.get(sid)
        unmoved = (st is not None
                   and abs(clat - st["lat"]) < MOVE_TRIG_DEG
                   and abs(((clon - st["lon"] + 180) % 360) - 180)
                   < MOVE_TRIG_DEG)
        complete = (st is not None
                    and len(st.get("layers") or []) == len(LAYERS))
        # a PARTIAL family (the SSTA-lags-SST case) retries its missing
        # layer after a cooldown instead of waiting for the storm to
        # move or the day to roll (adversarial-review find); within the
        # cooldown, don't even probe upstream.
        if (unmoved and st is not None and not complete
                and time.monotonic() - st.get("t", 0) < PARTIAL_RETRY_S):
            return
        self._deadline = time.monotonic() + CALL_BUDGET_S
        sst_day = self._latest_day("sst")   # TTL-cached: cheap when warm
        if sst_day is None:
            return
        if unmoved and st["date"] == sst_day.isoformat() and complete:
            return

        prune_cache(self.cache_dir, today=self._today())
        written = []
        for layer in LAYERS:
            try:
                day = (sst_day if layer["product"] == "sst"
                       else self._latest_day(layer["product"]))
                if day is None:
                    continue
                path = self._fetch_budgeted(layer["product"], day)
                if path is None:
                    continue
                if layer.get("compute") == "house_anomaly":
                    # ONE CANON: SST - house DOY climo. Gated on the
                    # bake-complete manifest; off-belt / pre-bake / grid
                    # mismatch -> the layer is simply not advertised.
                    if not self._climo_ready():
                        continue
                    res = self._house_anomaly_box(path, day, clat, clon)
                    if res is None:
                        continue
                    data, lats, lons = res
                else:
                    data, lats, lons = self._read_box(
                        path, layer["var"], clat, clon)
                png = render_hero_layer(layer, data, lats, lons,
                                        basin=basin, clat=clat, clon=clon)
                self.sink.write_png(
                    f"{self.prefix}/{sid}/sst/{layer['file']}", png)
                entry = {k: layer[k] for k in
                         ("slug", "label", "title", "field", "file",
                          "source") if k in layer}
                if layer.get("note"):
                    entry["note"] = layer["note"]
                entry["valid"] = day.isoformat()
                written.append(entry)
            except Exception as e:  # noqa: BLE001 - per-layer isolation
                log.warning("sst hero layer %s failed (%s): %s",
                            layer["slug"], sid, e)
        if not written:
            return
        now = dt.datetime.now(dt.UTC)
        self.sink.write(f"{self.prefix}/{sid}/sst/meta.json", {
            "sid": sid,
            "center": {"lat": round(clat, 2), "lon": round(clon, 2)},
            "box": {"hw_lon": HW_LON, "hw_lat": round(HW_LAT, 4)},
            "px": [int(FIG_W_IN * DPI), int(FIG_H_IN * DPI)],
            "valid_date": sst_day.isoformat(),
            "updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "layers": written,
        })
        self._state[sid] = {"date": sst_day.isoformat(),
                            "lat": clat, "lon": clon,
                            "layers": [x["slug"] for x in written],
                            "t": time.monotonic()}
        log.info("sst hero %s: %d layer(s) @ %.1f,%.1f (%s)",
                 sid, len(written), clat, clon, sst_day)
