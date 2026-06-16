"""kml_advisories - NHC advisory KMZ -> cached-JSON parser (CYCLOLAB_DESIGN.md §8.3).

CycloLab never ships raw KML to the browser. The advisories poller (§9)
unzips the per-storm ``..._CONE_latest.kmz`` / ``..._TRACK_latest.kmz``
products NHC publishes per advisory, validates them, and PUTs the §8.3
cached contract; the page animates pre-parsed geometry only.

Dissected from AMANDA's REAL advisory #13 artifacts (EP012026,
``tests/fixtures/cyclolab/``):

KMZ container
    A plain zip carrying a single ``*.kml`` member (``doc.kml`` by
    NetworkLink convention; NHC names it ``ep012026_013adv_CONE.kml``).
    We read the first member whose name ends ``.kml`` - no fixed name.

CONE kml  (namespace ``http://earth.google.com/kml/2.1``)
    One ``Placemark`` -> ``Polygon`` -> ``outerBoundaryIs`` ->
    ``LinearRing`` -> ``coordinates``. Coordinates are whitespace-
    separated ``lon,lat[,alt]`` tuples (lon FIRST). Amanda adv #13: a
    closed 1,272-vertex ring (first == last). We close it if upstream
    ever leaves it open and note that in provenance.

TRACK kml (namespace ``http://www.opengis.net/kml/2.2``)
    A ``Folder`` of ``Placemark`` elements of two shapes:
      * ``LineString`` placemarks (the 72 h / 120 h connector lines) -
        NOT forecast points; we read their ``ExtendedData`` only for
        ``advisoryNum``, ``stormType`` and ``pubAdvTime``.
      * ``Point`` placemarks - the forecast points. Each carries a
        ``styleUrl`` (``#initial_point`` / ``#s_point`` / ``#d_point`` /
        ``#xd_point`` / ...), a ``Point/coordinates`` ``lon,lat``, and a
        CDATA ``description`` table whose rows give the lead label
        (``Advisory Information`` == tau 0, or ``N hr Forecast``) and
        ``Maximum Wind: N knots``. Amanda: 9 points at tau
        0/12/24/36/48/60/72/96/120.

    dev_label comes from the point style id, ``x``-prefix == post-
    tropical (NHC's extratropical/remnant glyphs):
        s -> TS   d -> TD   h -> HU   m -> MH   l -> L
        xs -> STS xd -> PTC xh -> PTC xm -> PTC
    ``initial_point`` (tau 0) has no type of its own; it inherits the
    run's ``stormType`` ExtendedData (Amanda: TS).

    issued_utc is parsed FROM the document (clock-free, the §9 source-
    freshness rule): ``pubAdvTime`` "1100 AM HST Fri Jun 05 2026" ->
    21:00 UTC. valid_utc per point = issued_utc + tau hours.

The module is deliberately CLOCK-FREE: ``build_advisory_json`` leaves
``provenance.parsed_utc = None`` for the caller to stamp, so tests are
deterministic. stdlib only (zipfile / xml.etree / re / datetime).
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET


class AdvisoryParseError(ValueError):
    """Raised when a KMZ/advisory is malformed or fails §8.3 validation."""


# Point-style id -> developmental label. ``x`` prefix == post-tropical.
_DEV_LABEL = {
    "s": "TS", "d": "TD", "h": "HU", "m": "MH", "l": "L",
    "xs": "STS", "xd": "PTC", "xh": "PTC", "xm": "PTC", "xl": "L",
}

# US time-zone abbreviations seen in NHC pubAdvTime, offset hours from UTC.
# (NHC issues local standard/daylight per basin; these cover AL/EP/CP.)
_TZ_OFFSET = {
    "UTC": 0, "GMT": 0,
    "AST": -4,                      # Atlantic standard
    "EST": -5, "EDT": -4,           # Eastern
    "CST": -6, "CDT": -5,           # Central
    "MST": -7, "MDT": -6,           # Mountain
    "PST": -8, "PDT": -7,           # Pacific
    "HST": -10, "HDT": -9,          # Hawaii (CPHC)
    "SST": -11,                     # Samoa
}


def _localname(tag: str) -> str:
    """Strip any ``{namespace}`` prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def _read_kml(kmz_bytes: bytes) -> str:
    """Return the doc.kml text from a KMZ (zip with one ``*.kml`` member)."""
    if not isinstance(kmz_bytes, (bytes, bytearray)):
        raise AdvisoryParseError("kmz_bytes must be bytes")
    try:
        zf = zipfile.ZipFile(io.BytesIO(kmz_bytes))
    except zipfile.BadZipFile as exc:
        raise AdvisoryParseError(f"not a valid KMZ/zip: {exc}") from exc
    names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
    if not names:
        raise AdvisoryParseError("KMZ contains no .kml member")
    # Prefer doc.kml, else the first .kml.
    name = next((n for n in names if _localname(n).lower() == "doc.kml"),
                names[0])
    try:
        raw = zf.read(name)
    except (zipfile.BadZipFile, RuntimeError) as exc:  # truncated/encrypted
        raise AdvisoryParseError(f"cannot read {name}: {exc}") from exc
    return raw.decode("utf-8", "replace")


def _parse_xml(text: str) -> ET.Element:
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise AdvisoryParseError(f"malformed KML XML: {exc}") from exc


def _iter(root: ET.Element, name: str):
    """Yield all descendants whose localname == ``name`` (namespace-agnostic)."""
    for el in root.iter():
        if _localname(el.tag) == name:
            yield el


def _first(el: ET.Element, name: str):
    return next(_iter(el, name), None)


def _parse_coord_blob(text: str) -> list[list[float]]:
    """Parse a KML ``coordinates`` blob into ``[[lon, lat], ...]``.

    Tuples are whitespace-separated ``lon,lat[,alt]`` (lon first). alt,
    if present, is dropped.
    """
    pts: list[list[float]] = []
    for tok in (text or "").split():
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        try:
            lon, lat = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise AdvisoryParseError(f"bad coordinate {tok!r}: {exc}") from exc
        pts.append([lon, lat])
    return pts


def _validate_lonlat(pts, where: str) -> None:
    for lon, lat in pts:
        if not (-180.0 <= lon <= 180.0):
            raise AdvisoryParseError(f"{where}: lon {lon} out of range")
        if not (-90.0 <= lat <= 90.0):
            raise AdvisoryParseError(f"{where}: lat {lat} out of range")


# ---------------------------------------------------------------------------
# CONE
# ---------------------------------------------------------------------------

def parse_cone_kmz(kmz_bytes: bytes) -> list[list[float]]:
    """KMZ bytes -> closed cone ring ``[[lon, lat], ...]`` (first == last).

    Cone geometry lives at
    ``Placemark/Polygon/outerBoundaryIs/LinearRing/coordinates``. Matches
    on localname so the KML namespace is irrelevant. Validates >= 4
    vertices and all lon/lat in range; closes the ring if upstream left
    it open (the caller records that via the returned ring's endpoints).
    """
    root = _parse_xml(_read_kml(kmz_bytes))

    ring_el = None
    for poly in _iter(root, "Polygon"):
        outer = _first(poly, "outerBoundaryIs")
        if outer is None:
            continue
        lr = _first(outer, "LinearRing")
        if lr is None:
            continue
        coords = _first(lr, "coordinates")
        if coords is not None and (coords.text or "").strip():
            ring_el = coords
            break
    if ring_el is None:
        raise AdvisoryParseError(
            "cone KML has no Polygon/outerBoundaryIs/LinearRing/coordinates")

    pts = _parse_coord_blob(ring_el.text)
    if len(pts) < 4:
        raise AdvisoryParseError(
            f"cone ring has {len(pts)} vertices, need >= 4")
    _validate_lonlat(pts, "cone")

    if pts[0] != pts[-1]:                       # close an open ring
        pts.append(list(pts[0]))
    return pts


# ---------------------------------------------------------------------------
# TRACK
# ---------------------------------------------------------------------------

def _extended_data(placemark: ET.Element) -> dict[str, str]:
    """Read ``ExtendedData/Data[name]/value`` into a flat dict."""
    out: dict[str, str] = {}
    ed = _first(placemark, "ExtendedData")
    if ed is None:
        return out
    for data in _iter(ed, "Data"):
        key = data.get("name")
        val = _first(data, "value")
        if key is not None and val is not None:
            out[key] = (val.text or "").strip()
    return out


def _dev_label(style_id: str, storm_type: str | None) -> str:
    """Map a point ``styleUrl`` id to a developmental label."""
    sid = (style_id or "").lstrip("#").strip()
    base = sid[:-len("_point")] if sid.endswith("_point") else sid
    if base == "initial":
        return (storm_type or "").upper() or "TS"
    return _DEV_LABEL.get(base, (storm_type or base).upper())


_TAU_RE = re.compile(r"(\d+)\s*hr\s*Forecast", re.I)
_INIT_RE = re.compile(r"Advisory\s+Information", re.I)
_WIND_RE = re.compile(r"Maximum\s+Wind:\s*(\d+)\s*knots", re.I)


def _placemark_text(placemark: ET.Element) -> str:
    """Concatenate a placemark's description text (CDATA included)."""
    desc = _first(placemark, "description")
    return (desc.text or "") if desc is not None else ""


def _parse_pub_adv_time(s: str) -> datetime:
    """Parse NHC ``pubAdvTime`` ("1100 AM HST Fri Jun 05 2026") -> aware UTC.

    Format: ``HHMM AM|PM TZ Dow Mon DD YYYY``. The TZ abbreviation maps to
    a fixed UTC offset (``_TZ_OFFSET``); the result is normalized to UTC.
    """
    m = re.match(
        r"\s*(\d{1,2})(\d{2})\s+(AM|PM)\s+([A-Z]{3})\s+\w+\s+"
        r"(\w{3})\s+(\d{1,2})\s+(\d{4})",
        s.strip(), re.I)
    if not m:
        raise AdvisoryParseError(f"unparseable pubAdvTime {s!r}")
    hh, mm, ampm, tz, mon, dd, yyyy = m.groups()
    hh, mm = int(hh), int(mm)
    if ampm.upper() == "PM" and hh != 12:
        hh += 12
    elif ampm.upper() == "AM" and hh == 12:
        hh = 0
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    try:
        month = months.index(mon.lower()) + 1
    except ValueError as exc:
        raise AdvisoryParseError(f"bad month in pubAdvTime {s!r}") from exc
    if tz.upper() not in _TZ_OFFSET:
        raise AdvisoryParseError(f"unknown timezone {tz!r} in pubAdvTime")
    local = datetime(int(yyyy), month, int(dd), hh, mm,
                     tzinfo=timezone(timedelta(hours=_TZ_OFFSET[tz.upper()])))
    return local.astimezone(timezone.utc)


def _iso_z(dt: datetime) -> str:
    """aware datetime -> ``YYYY-MM-DDTHH:MM:SSZ`` (UTC, no microseconds)."""
    return dt.astimezone(timezone.utc).replace(
        microsecond=0, tzinfo=None).isoformat() + "Z"


def parse_track_kmz(kmz_bytes: bytes) -> dict:
    """KMZ bytes -> ``{advisory, issued_utc, points}``.

    Iterates ``Placemark`` elements in document order. LineString
    placemarks contribute ``advisoryNum`` / ``stormType`` / ``pubAdvTime``
    (ExtendedData); Point placemarks become forecast points with tau,
    valid_utc, lat, lon, intensity_kt and dev_label. issued_utc is parsed
    from ``pubAdvTime`` and each point's valid_utc = issued_utc + tau.
    """
    root = _parse_xml(_read_kml(kmz_bytes))

    advisory: int | None = None
    storm_type: str | None = None
    issued: datetime | None = None
    raw_points: list[dict] = []

    for pm in _iter(root, "Placemark"):
        ed = _extended_data(pm)
        if ed:
            if advisory is None and ed.get("advisoryNum"):
                try:
                    advisory = int(ed["advisoryNum"])
                except ValueError:
                    pass
            if storm_type is None and ed.get("stormType"):
                storm_type = ed["stormType"]
            if issued is None and ed.get("pubAdvTime"):
                issued = _parse_pub_adv_time(ed["pubAdvTime"])

        point = _first(pm, "Point")
        if point is None:
            continue                            # LineString / non-point

        coords_el = _first(point, "coordinates")
        coords = _parse_coord_blob(coords_el.text if coords_el is not None
                                   else "")
        if not coords:
            raise AdvisoryParseError("track Point has empty coordinates")
        lon, lat = coords[0]

        text = _placemark_text(pm)
        if _INIT_RE.search(text):
            tau = 0
        else:
            mt = _TAU_RE.search(text)
            if not mt:
                raise AdvisoryParseError(
                    "track Point missing forecast-hour label")
            tau = int(mt.group(1))

        style = _first(pm, "styleUrl")
        style_id = (style.text or "") if style is not None else ""

        mw = _WIND_RE.search(text)
        intensity = int(mw.group(1)) if mw else None

        raw_points.append({
            "tau_h": tau,
            "lat": lat,
            "lon": lon,
            "intensity_kt": intensity,
            # dev_label is resolved in the SECOND pass below: in the real
            # NHC document the Point placemarks precede the LineString
            # that carries the stormType ExtendedData, so resolving here
            # would always see storm_type=None for the tau-0 initial
            # point and silently mislabel a tau-0 HU/TD/STS as "TS"
            # (cross-review-found; Amanda masked it by genuinely being
            # a TS).
            "style_id": style_id,
        })

    if advisory is None:
        raise AdvisoryParseError("track KML carries no advisoryNum")
    if issued is None:
        raise AdvisoryParseError("track KML carries no pubAdvTime")
    if not raw_points:
        raise AdvisoryParseError("track KML has no forecast Point placemarks")

    points = []
    for p in raw_points:
        valid = issued + timedelta(hours=p["tau_h"])
        points.append({
            "tau_h": p["tau_h"],
            "valid_utc": _iso_z(valid),
            "lat": p["lat"],
            "lon": p["lon"],
            "intensity_kt": p["intensity_kt"],
            # Second-pass resolution: storm_type is final here, so the
            # tau-0 initial point inherits the run's true stormType.
            "dev_label": _dev_label(p["style_id"], storm_type),
        })

    _validate_lonlat([(p["lon"], p["lat"]) for p in points], "track")

    return {
        "advisory": advisory,
        "issued_utc": _iso_z(issued),
        "points": points,
    }


# ---------------------------------------------------------------------------
# ASSEMBLY
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WATCHES / WARNINGS (Phase 4) - the coastal tropical-cyclone watch/warning
# line segments from the advisory GIS package's WW KMZ (CurrentStorms
# windWatchesWarnings.kmzFile). NHC AL/EP/CP only; never JTWC/WP (we never
# fabricate watches/warnings). Each Placemark/LineString connects the
# breakpoints delimiting one watch/warning.
# ---------------------------------------------------------------------------

# NHC's watch/warning vocabulary -> our canonical type code. Keyed first by the
# Placemark <name> (explicit, human-readable), then by the styleUrl id
# (TWA/TWR/HWA/HWR observed in the wind WW KMZ; SSA/SSR for storm surge). An
# unknown segment keeps its raw name so it is never silently dropped (the
# renderer falls back to a neutral color).
_WW_NAME_TO_TYPE = {
    "TROPICAL STORM WATCH": "TS_WATCH",
    "TROPICAL STORM WARNING": "TS_WARNING",
    "HURRICANE WATCH": "HU_WATCH",
    "HURRICANE WARNING": "HU_WARNING",
    "STORM SURGE WATCH": "SS_WATCH",
    "STORM SURGE WARNING": "SS_WARNING",
}
_WW_STYLE_TO_TYPE = {
    "TWA": "TS_WATCH", "TWR": "TS_WARNING",
    "HWA": "HU_WATCH", "HWR": "HU_WARNING",
    "SSA": "SS_WATCH", "SSR": "SS_WARNING",
}


def _ww_type(name: str, style_id: str) -> str:
    n = (name or "").strip().upper()
    if n in _WW_NAME_TO_TYPE:
        return _WW_NAME_TO_TYPE[n]
    sid = (style_id or "").lstrip("#").strip().upper()
    if sid in _WW_STYLE_TO_TYPE:
        return _WW_STYLE_TO_TYPE[sid]
    return n or sid or "UNKNOWN"


def parse_ww_kmz(kmz_bytes: bytes) -> list[dict]:
    """WW KMZ bytes -> coastal watch/warning segments::

        [{"type": "TS_WATCH", "geometry": [[lon, lat], ...]}, ...]

    Each segment is ONE Placemark/LineString (NHC connects the breakpoints
    delimiting a watch/warning). ``type`` is the canonical code (see
    ``_WW_NAME_TO_TYPE``). NO watches/warnings in effect -> ``[]``. Raises
    :class:`AdvisoryParseError` on a malformed document so the caller degrades
    gracefully (the cone is written WITHOUT a ``ww`` overlay rather than not at
    all). Namespace-agnostic (localname matching) like the cone/track parsers.
    """
    root = _parse_xml(_read_kml(kmz_bytes))
    out: list[dict] = []
    for pm in _iter(root, "Placemark"):
        ls = _first(pm, "LineString")
        if ls is None:
            continue                            # skip non-line placemarks
        coords = _first(ls, "coordinates")
        if coords is None or not (coords.text or "").strip():
            continue
        pts = _parse_coord_blob(coords.text)
        if len(pts) < 2:                        # a segment needs >= 2 points
            continue
        _validate_lonlat(pts, "ww")
        name_el = _first(pm, "name")
        style_el = _first(pm, "styleUrl")
        seg_type = _ww_type(
            name_el.text if name_el is not None else "",
            style_el.text if style_el is not None else "")
        out.append({"type": seg_type, "geometry": pts})
    return out


def build_advisory_json(sid: str, cone_kmz_bytes: bytes,
                        track_kmz_bytes: bytes,
                        text_urls: dict | None = None,
                        ww_kmz_bytes: bytes | None = None) -> dict:
    """Assemble the §8.3 cached-advisory contract from CONE + TRACK (+ WW) KMZs.

    Returns a json-serializable dict with the contract keys::

        {sid, advisory, issued_utc, source, method, cone, points, text,
         ww, provenance}

    source == "nhc", method == "official-cone". Validation (raises
    :class:`AdvisoryParseError`): the cone ring closes (>= 4 verts),
    points >= 2, taus strictly increasing, issued_utc parses. provenance
    carries the raw-bytes hashes/sizes; ``parsed_utc`` is left None for
    the caller to stamp (clock-free for testability).

    ``ww`` (Phase 4) is the coastal watch/warning overlay from the optional
    ``ww_kmz_bytes`` (CurrentStorms windWatchesWarnings.kmzFile): a list of
    ``{type, geometry}`` segments, or ``[]`` when no WW KMZ is supplied OR it
    fails to parse. ADDITIVE + GRACEFUL by construction: a missing/malformed
    WW layer never raises here, so it can never break the cone the page needs.
    """
    cone = parse_cone_kmz(cone_kmz_bytes)
    track = parse_track_kmz(track_kmz_bytes)
    points = track["points"]

    # --- validation -------------------------------------------------------
    if len(cone) < 4 or cone[0] != cone[-1]:
        raise AdvisoryParseError("cone ring does not close")
    if len(points) < 2:
        raise AdvisoryParseError(
            f"need >= 2 forecast points, got {len(points)}")
    taus = [p["tau_h"] for p in points]
    if any(b <= a for a, b in zip(taus, taus[1:])):
        raise AdvisoryParseError(f"taus not strictly increasing: {taus}")
    # intensity_kt is a REQUIRED int per the contract (the cone's
    # forecast-point icons are intensity-driven); NHC always supplies the
    # "Maximum Wind" row, so a windless point means a malformed document
    # - fail loud rather than render a colorless icon.
    missing_wind = [p["tau_h"] for p in points
                    if not isinstance(p["intensity_kt"], int)]
    if missing_wind:
        raise AdvisoryParseError(
            f"points missing intensity_kt at taus {missing_wind}")
    try:
        datetime.strptime(track["issued_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise AdvisoryParseError(
            f"issued_utc does not parse: {track['issued_utc']!r}") from exc

    text = None
    if text_urls:
        text = {
            "tcp_url": text_urls.get("tcp_url"),
            "tcd_url": text_urls.get("tcd_url"),
        }

    # Watches/warnings overlay - ADDITIVE + GRACEFUL. A malformed/absent WW
    # layer degrades to [] and NEVER raises, so it can't break the cone.
    ww: list[dict] = []
    if ww_kmz_bytes:
        try:
            ww = parse_ww_kmz(ww_kmz_bytes)
        except AdvisoryParseError:
            ww = []

    provenance = {
        "cone_sha256": hashlib.sha256(cone_kmz_bytes).hexdigest(),
        "track_sha256": hashlib.sha256(track_kmz_bytes).hexdigest(),
        "cone_bytes": len(cone_kmz_bytes),
        "track_bytes": len(track_kmz_bytes),
        "parsed_utc": None,                     # caller stamps the clock
    }
    if ww_kmz_bytes:
        provenance["ww_sha256"] = hashlib.sha256(ww_kmz_bytes).hexdigest()
        provenance["ww_bytes"] = len(ww_kmz_bytes)

    return {
        "sid": sid,
        "advisory": track["advisory"],
        "issued_utc": track["issued_utc"],
        "source": "nhc",
        "method": "official-cone",
        "cone": cone,
        "points": points,
        "text": text,
        "ww": ww,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# NEXT ADVISORY (Public Advisory / TCP footer countdown - CYCLOLAB_DESIGN
# advisory-countdown, source-of-truth rule)
# ---------------------------------------------------------------------------
#
# NHC Public Advisories (TCP) state the next advisory time EXPLICITLY in a
# footer block, e.g.::
#
#     NEXT ADVISORY
#     -------------
#     Next complete advisory at 1100 PM HST.
#
# or, while intermediate ("A") advisories are active (3-hourly, watches /
# warnings in effect)::
#
#     Next intermediate advisory at 700 PM CDT.
#     Next complete advisory at 1000 PM CDT.
#
# The countdown shell MUST display the STATED time - never a cadence guessed
# from the wall clock. The footer times carry no date, so we resolve each
# stated local time against ``issued_utc``'s date IN THAT ZONE, rolling
# forward one day when the resolved instant is <= issued (an advisory never
# states a time already in the past).
#
# Ending storms drop the block entirely or replace it with "This is the last
# public advisory ..."; that is the graceful no-countdown case (stated False).

# Pull the local-zone time off a "Next ... advisory at HHMM AM|PM TZ." line.
# Minutes are optional in the wild ("at noon" never occurs; NHC always emits
# HMM/HHMM), TZ is the same 3-letter abbreviation _TZ_OFFSET already covers.
_NEXT_ADV_RE = re.compile(
    r"Next\s+(intermediate|complete)\s+advisory\s+at\s+"
    r"(\d{1,2})(\d{2})\s+(AM|PM)\s+([A-Z]{3})",
    re.I,
)

# Footer phrasing for a final advisory (real NHC text). Presence of this
# anywhere in the NEXT ADVISORY region means the storm is ending.
_LAST_ADV_RE = re.compile(
    r"This\s+is\s+the\s+last\s+(?:public\s+)?advisory", re.I)

_PRE_RE = re.compile(r"<pre\b[^>]*>(.*?)</pre>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Return the advisory product text from a .shtml page or a bare product.

    NHC serves the product wrapped in a ``<pre>`` block inside boilerplate
    HTML; the raw product has no tags at all. We use the first ``<pre>``
    block when present, else strip any stray tags, and unescape the handful
    of entities NHC emits (``&amp;`` / ``&lt;`` / ``&gt;`` / ``&#176;``...).
    Bare text passes through unchanged.
    """
    m = _PRE_RE.search(text)
    body = m.group(1) if m else _TAG_RE.sub("", text)
    # Minimal entity unescape (stdlib html.unescape would also work, but the
    # module is intentionally dependency-light and these are all NHC emits).
    return (body
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'"))


def _resolve_footer_time(hh: int, mm: int, ampm: str, tz: str,
                         issued: datetime) -> datetime:
    """Resolve a dateless footer time against ``issued``'s date in its zone.

    ``issued`` is an aware UTC datetime (the advisory's own issuance, parsed
    upstream - the clock-free source-freshness rule). We anchor the stated
    HH:MM in ``tz`` on the SAME calendar date ``issued`` falls on within that
    zone, then roll forward whole days until the instant is strictly after
    ``issued`` (NHC never states a time already past). Returns aware UTC.
    """
    tz = tz.upper()
    if tz not in _TZ_OFFSET:
        raise AdvisoryParseError(f"unknown timezone {tz!r} in NEXT ADVISORY")
    if ampm.upper() == "PM" and hh != 12:
        hh += 12
    elif ampm.upper() == "AM" and hh == 12:
        hh = 0
    zone = timezone(timedelta(hours=_TZ_OFFSET[tz]))
    issued_local = issued.astimezone(zone)
    cand = issued_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    # Roll forward a day at a time until strictly after issuance. A footer
    # time equal to issuance is also a future advisory (>= would keep it),
    # but NHC never restates the current time, so strict > is correct and
    # the loop runs at most once in practice.
    while cand <= issued_local:
        cand += timedelta(days=1)
    return cand.astimezone(timezone.utc)


def product_text(text: str) -> str:
    """Public face of the product extractor: the plain advisory text from
    a .shtml wrapper page or a bare product (see _strip_html). Used to
    ship TCP/TCD panels in the advisory JSON (§7.4)."""
    return _strip_html(text)


def parse_next_advisory(tcp_text: str, issued_utc: str) -> dict:
    """Parse a Public Advisory (TCP) footer into the next-advisory countdown.

    ``tcp_text`` is the RAW advisory product OR the ``.shtml`` page that wraps
    it (we strip HTML / read the ``<pre>`` block automatically). ``issued_utc``
    is the advisory's own issuance instant as ``YYYY-MM-DDTHH:MM:SSZ`` - the
    dateless footer times resolve against THIS, never the wall clock.

    Returns::

        {"next_advisory_utc": iso-z | None,
         "kind": "intermediate" | "complete" | None,
         "next_complete_utc": iso-z | None,
         "stated": bool}

    When both an intermediate and a complete time are stated, the
    intermediate (earlier) wins ``next_advisory_utc``/``kind`` and the
    complete time is also returned in ``next_complete_utc``. With only a
    complete time stated, ``next_complete_utc`` mirrors ``next_advisory_utc``.

    A final advisory ("This is the last public advisory ...") or any product
    with no parseable NEXT ADVISORY footer is the graceful no-countdown case:
    all fields ``None`` / ``stated`` False (the shell shows no countdown - the
    storm is ending). :class:`AdvisoryParseError` is raised ONLY for garbage
    input (empty / None / non-string, or an unparseable ``issued_utc``); an
    absent footer is NOT an error.
    """
    if not tcp_text or not isinstance(tcp_text, str):
        raise AdvisoryParseError("tcp_text is empty or not a string")
    try:
        issued = datetime.strptime(
            issued_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise AdvisoryParseError(
            f"issued_utc must be ISO-Z, got {issued_utc!r}") from exc

    none_result = {
        "next_advisory_utc": None,
        "kind": None,
        "next_complete_utc": None,
        "stated": False,
    }

    body = _strip_html(tcp_text)

    # Last advisory -> no countdown (graceful), even if the NEXT ADVISORY
    # header is present (NHC keeps the header and prints the last-advisory
    # paragraph in place of a time).
    if _LAST_ADV_RE.search(body):
        return none_result

    intermediate_utc = None
    complete_utc = None
    for m in _NEXT_ADV_RE.finditer(body):
        kind, hh, mm, ampm, tz = m.groups()
        resolved = _resolve_footer_time(
            int(hh), int(mm), ampm, tz, issued)
        if kind.lower() == "intermediate":
            # Keep the earliest if NHC ever repeated the line (it doesn't).
            if intermediate_utc is None or resolved < intermediate_utc:
                intermediate_utc = resolved
        else:
            if complete_utc is None or resolved < complete_utc:
                complete_utc = resolved

    if intermediate_utc is None and complete_utc is None:
        return none_result

    # The SOONEST upcoming advisory wins next_advisory_utc/kind. Normally the
    # intermediate is the nearer one, but NHC's footer states the next
    # intermediate AND complete by clock time relative to the product's OWN
    # issuance, while issued_utc here is the SYNOPTIC instant - so a stated
    # intermediate that equals the synoptic time resolves +24 h (e.g. THREE-E
    # adv 2: "Next intermediate ... 1200 PM CST" == 1800Z issuance -> next
    # day), leaving the complete ("300 PM CST" = 2100Z, +3 h) the real next.
    # Pick the earlier of the two rather than blindly trusting the
    # intermediate (which would show a ~24 h countdown on an active storm).
    candidates = [(t, k) for t, k in
                  ((intermediate_utc, "intermediate"),
                   (complete_utc, "complete")) if t is not None]
    candidates.sort(key=lambda c: c[0])
    next_utc, kind = candidates[0]
    return {
        "next_advisory_utc": _iso_z(next_utc),
        "kind": kind,
        "next_complete_utc": (_iso_z(complete_utc)
                              if complete_utc is not None else None),
        "stated": True,
    }
