"""Antimeridian (dateline-crossing) bbox support + 429 backoff, end to end.

Run from the repo root:

    .venv/bin/python tests/test_antimeridian.py

Covers the 2026-07-14 fix chain:
  * app.py `_v_bbox` accepts BOTH crossing input forms (unwrapped E>180 like
    the swpac basin backdrop [140,-35,200,5]; pre-wrapped E<W like the
    poller's norm_lon storm box) and normalizes to the internal E<W
    convention — previously both 422'd ("invalid longitude range") and
    dateline-straddling floaters were silently skipped.
  * compute_downsample_factor / _bbox_overlaps / _bbox_inside crossing math.
  * pick_satellite routes crossing boxes (Himawari / GOES-West).
  * render_png + render_backdrop_webp actually RENDER a crossing bbox with
    the field on the correct side of the seam (geographic placement proven
    by flipping a half-plane field and watching the image flip).
  * floater_poller.call_render 429 handling: Retry-After honored, exponential
    floor, own retry budget, RenderError after exhaustion.
  * app.py real_ip / rate_limit_for: trusted-internal detection.

Exits non-zero on any failure. No pytest dependency — assertions only.
"""

import datetime as dt
import io
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satellites import (  # noqa: E402
    FetchResult,
    GOES_WEST,
    HIMAWARI_PACIFIC,
    _bbox_inside,
    _bbox_overlaps,
    pick_satellite,
)
from app import (  # noqa: E402
    RATE_LIMIT,
    RATE_LIMIT_INTERNAL,
    RenderRequest,
    compute_downsample_factor,
    rate_limit_for,
)
import floater_poller as fp  # noqa: E402
from render import render_backdrop_webp, render_png  # noqa: E402


NOW = dt.datetime(2026, 7, 14, 4, 0, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# 1. RenderRequest bbox validator
# ---------------------------------------------------------------------------

def _bbox_of(raw):
    return RenderRequest(bbox=raw, time="latest", channel="clean_ir").bbox


def _bbox_rejected(raw) -> bool:
    try:
        RenderRequest(bbox=raw, time="latest", channel="clean_ir")
        return False
    except Exception:
        return True


def test_bbox_normalization():
    # The swpac basin backdrop: unwrapped E=200 -> crossing convention E=-160.
    assert _bbox_of([140, -35, 200, 5]) == [140.0, -35.0, -160.0, 5.0]
    # Pre-wrapped crossing storm box passes through untouched.
    assert _bbox_of([172, -6, -176, 6]) == [172.0, -6.0, -176.0, 6.0]
    # Exact-dateline east edge stays NON-crossing (edge-preserving wrap).
    assert _bbox_of([100, 0, 180, 45]) == [100.0, 0.0, 180.0, 45.0]
    # 0-360-style non-crossing box normalizes both edges.
    assert _bbox_of([190, 0, 220, 10]) == [-170.0, 0.0, -140.0, 10.0]
    # E=-180 flips to +180 (same meridian, non-crossing form).
    assert _bbox_of([170, 0, -180, 5]) == [170.0, 0.0, 180.0, 5.0]
    # A plain box is byte-identical.
    assert _bbox_of([-100, 5, -80, 25]) == [-100.0, 5.0, -80.0, 25.0]
    # Degenerate / nonsense still refused.
    assert _bbox_rejected([0, 0, 360, 10])      # full circle -> W == E
    assert _bbox_rejected([10, 5, 10, 10])      # zero lon span
    assert _bbox_rejected([400, 0, 20, 10])     # out of the accepted domain
    assert _bbox_rejected([10, 20, 20, 5])      # lat inverted
    print("bbox normalization ok")


def test_downsample_crossing_span():
    # A 40-deg crossing span must budget exactly like a 40-deg plain span.
    plain = compute_downsample_factor([0, -10, 40, 10], "visible_red")
    crossing = compute_downsample_factor([160, -10, -160, 10], "visible_red")
    assert plain == crossing, (plain, crossing)
    print("downsample crossing span ok")


def test_bbox_fractional_lons_pass_byte_identical():
    # In-range fractional longitudes are NOT perturbed by wrap arithmetic --
    # existing cache keys depend on it (adversarial-review catch: float mod
    # turns -76.3 into -76.30000000000001).
    assert _bbox_of([-76.3, 10.2, -60.1, 20.7]) == [-76.3, 10.2, -60.1, 20.7]
    assert _bbox_of([139.7, -8.4, 179.9, 8.1]) == [139.7, -8.4, 179.9, 8.1]
    print("fractional lon byte-identity ok")


def test_full_globe_span_keeps_pixel_budget():
    # [-180, S, 180, N] is legal + non-crossing; % 360 must not zero its
    # span (pixel-budget bypass -> unstrided full-disk render,
    # adversarial-review catch).
    f_globe = compute_downsample_factor([-180, -60, 180, 60], "visible_red")
    f_half = compute_downsample_factor([-180, -60, 0, 60], "visible_red")
    assert f_globe >= f_half > 1, (f_globe, f_half)
    print("full-globe pixel budget ok")


def test_reversed_edges_rejected():
    # A reversed-typo "crossing" box spanning >180 deg is refused loudly
    # (pre-diff these 422'd; a silently mis-framed near-world 200 is worse).
    assert _bbox_rejected([50, 0, 30, 10])      # 340-deg "crossing"
    assert _bbox_rejected([30, 0, 10, 10])
    # Genuine crossing boxes stay accepted.
    assert _bbox_of([172, -6, -176, 6]) == [172.0, -6.0, -176.0, 6.0]
    print("reversed-edge rejection ok")


# ---------------------------------------------------------------------------
# 2. Overlap / inside helpers
# ---------------------------------------------------------------------------

def test_bbox_helpers_crossing():
    goes_east_disk = (-135.0, -75.0, -5.0, 75.0)
    # 100E..100W crossing box genuinely reaches into GOES-East's window.
    assert _bbox_overlaps([100, -10, -100, 10], goes_east_disk)
    # 150E..160W does not.
    assert not _bbox_overlaps([150, -10, -160, 10], goes_east_disk)
    # Plain-box behavior unchanged.
    assert _bbox_overlaps([-100, -10, -90, 10], goes_east_disk)
    assert not _bbox_overlaps([0, -10, 20, 10], goes_east_disk)
    # A crossing box is never "inside" a non-crossing sector footprint.
    pacus = (-152.0, 14.0, -77.0, 51.0)
    assert not _bbox_inside([176, 15, -172, 25], pacus)
    print("bbox helpers ok")


def test_pick_satellite_crossing():
    # swpac (normalized): Himawari is the closest disk.
    assert pick_satellite([140, -35, -160, 5], NOW) is HIMAWARI_PACIFIC
    # A dateline-straddling CPac storm box: GOES-West edges out Himawari
    # (sub-sat -137.2 vs center ~-178 -> 40.8 deg vs 41.3 deg).
    assert pick_satellite([176, 10, -172, 22], NOW) is GOES_WEST
    print("pick_satellite crossing ok")


# ---------------------------------------------------------------------------
# 3. Real crossing renders (cartopy + matplotlib)
# ---------------------------------------------------------------------------

def _synthetic_crossing_fetch(cold_east: bool) -> FetchResult:
    """A 20x40-deg field straddling the dateline (170E..170W), WRAPPED source
    longitudes like the AHI inverse projection emits. One half-plane is cold
    (200 K), the other warm (290 K), split exactly at the dateline."""
    lon_uw = np.linspace(168.0, 192.0, 240)          # unwrapped 168..192
    lat = np.linspace(-11.0, 11.0, 220)
    LONuw, LAT = np.meshgrid(lon_uw, lat)
    LON = ((LONuw + 180.0) % 360.0) - 180.0          # wrapped, jump at 180
    east_of_dateline = LONuw > 180.0                 # the ...W side
    cold = east_of_dateline if cold_east else ~east_of_dateline
    bt = np.where(cold, 200.0, 290.0)
    return FetchResult(
        cmi=bt.astype(np.float32),
        lats=LAT.astype(np.float32),
        lons=LON.astype(np.float32),
        channel=13,
        generic_channel="clean_ir",
        scan_start=NOW,
        product="FLDK",
        bucket="noaa-himawari9",
        sat_name="Himawari-9",
        sub_sat_lon=140.7,
        units="K",
    )


def _halves_mean(png_or_webp: bytes):
    from PIL import Image
    im = Image.open(io.BytesIO(png_or_webp)).convert("L")
    a = np.asarray(im, dtype=float)
    w = a.shape[1] // 2
    return float(a[:, :w].mean()), float(a[:, w:].mean())


def test_backdrop_crossing_render():
    bbox = [170.0, -10.0, -170.0, 10.0]
    webp_ce = render_backdrop_webp(_synthetic_crossing_fetch(cold_east=True), bbox)
    webp_cw = render_backdrop_webp(_synthetic_crossing_fetch(cold_east=False), bbox)
    assert len(webp_ce) > 2000 and webp_ce[:4] == b"RIFF"
    l_ce, r_ce = _halves_mean(webp_ce)
    l_cw, r_cw = _halves_mean(webp_cw)
    # The two halves must differ (both lobes carry data, not background)...
    assert abs(l_ce - r_ce) > 15.0, (l_ce, r_ce)
    # ...and flipping the cold half-plane must flip the image: geographic
    # placement across the seam is real, not a wrap-around artifact.
    assert (l_ce - r_ce) * (l_cw - r_cw) < 0, (l_ce, r_ce, l_cw, r_cw)
    print("backdrop crossing render ok "
          f"(cold-east halves {l_ce:.0f}/{r_ce:.0f}, cold-west {l_cw:.0f}/{r_cw:.0f})")


def test_render_png_crossing():
    bbox = [172.0, -6.0, -176.0, 6.0]   # the poller's norm_lon storm-box form
    png = render_png(
        _synthetic_crossing_fetch(cold_east=True), bbox,
        channel=13, time_str="2026-07-14T04:00Z", enhancement="rainbow_ir",
    )
    assert len(png) > 10_000 and png[:8] == b"\x89PNG\r\n\x1a\n"
    print("render_png crossing ok")


def test_render_png_plain_unchanged():
    # Control: a plain box still renders (the shared geometry helper is a
    # no-op for non-crossing input).
    lon = np.linspace(-100.0, -80.0, 200)
    lat = np.linspace(5.0, 25.0, 200)
    LON, LAT = np.meshgrid(lon, lat)
    fr = FetchResult(
        cmi=np.full(LON.shape, 250.0, np.float32), lats=LAT.astype(np.float32),
        lons=LON.astype(np.float32), channel=13, generic_channel="clean_ir",
        scan_start=NOW, product="CMIPF", bucket="noaa-goes19",
        sat_name="GOES-19", sub_sat_lon=-75.2, units="K",
    )
    png = render_png(fr, [-100, 5, -80, 25], channel=13,
                     time_str="2026-07-14T04:00Z", enhancement="rainbow_ir")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    print("render_png plain control ok")


# ---------------------------------------------------------------------------
# 4. call_render 429 backoff
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code = status
        self.content = content
        self.headers = headers or {}
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _Session:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def post(self, *a, **kw):
        self.calls += 1
        return self._responses.pop(0)


def _patched_sleeps():
    sleeps = []
    real = fp.time.sleep
    fp.time.sleep = lambda s: sleeps.append(s)
    return sleeps, real


def test_429_recovers_and_honors_floor():
    sleeps, real = _patched_sleeps()
    try:
        s = _Session([
            _Resp(429, headers={"Retry-After": "7"}),
            _Resp(429),
            _Resp(200, b"FRAME", {"Content-Type": "image/webp"}),
        ])
        content, headers = fp.call_render(s, [140, -35, 200, 5], "shortwave_ir",
                                          "grayscale", backdrop=True)
    finally:
        fp.time.sleep = real
    assert content == b"FRAME" and s.calls == 3
    # Retry-After of 7 s is BELOW the 15 s floor -> floor wins; second wait
    # doubles. Jitter adds at most 2 s.
    assert fp.RENDER_429_BASE_S <= sleeps[0] <= fp.RENDER_429_BASE_S + 2, sleeps
    assert fp.RENDER_429_BASE_S * 2 <= sleeps[1] <= fp.RENDER_429_BASE_S * 2 + 2, sleeps
    print(f"429 recovery ok (sleeps {[round(x, 1) for x in sleeps]})")


def test_429_honors_long_retry_after():
    sleeps, real = _patched_sleeps()
    try:
        s = _Session([
            _Resp(429, headers={"Retry-After": "45"}),
            _Resp(200, b"F", {"Content-Type": "image/webp"}),
        ])
        fp.call_render(s, [0, 0, 10, 10], "clean_ir", "grayscale")
    finally:
        fp.time.sleep = real
    assert 45.0 <= sleeps[0] <= 47.0, sleeps   # hint beats the 15 s floor
    print("429 Retry-After honored ok")


def test_429_budget_exhausts():
    sleeps, real = _patched_sleeps()
    try:
        s = _Session([_Resp(429)] * (fp.RENDER_429_RETRIES + 1))
        try:
            fp.call_render(s, [0, 0, 10, 10], "clean_ir", "grayscale")
            raise AssertionError("expected RenderError")
        except fp.RenderError as e:
            assert "429" in str(e)
    finally:
        fp.time.sleep = real
    assert s.calls == fp.RENDER_429_RETRIES + 1
    assert len(sleeps) == fp.RENDER_429_RETRIES
    # Waits are capped: no single sleep beyond RENDER_429_MAX_WAIT_S + jitter.
    assert max(sleeps) <= fp.RENDER_429_MAX_WAIT_S + 2.0
    print("429 budget exhaustion ok")


def test_429_budget_separate_from_transport_budget():
    # A 429 must not consume the transport-error retries: 1x429 then 2
    # connection errors still leaves a final successful attempt.
    class _FlakySession:
        def __init__(self):
            self.calls = 0

        def post(self, *a, **kw):
            self.calls += 1
            if self.calls == 1:
                return _Resp(429)
            if self.calls <= 3:
                raise ConnectionError("boom")
            return _Resp(200, b"OK", {"Content-Type": "image/png"})

    sleeps, real = _patched_sleeps()
    try:
        s = _FlakySession()
        content, _ = fp.call_render(s, [0, 0, 10, 10], "clean_ir", "grayscale")
    finally:
        fp.time.sleep = real
    assert content == b"OK" and s.calls == 4
    print("429/transport budget separation ok")


# ---------------------------------------------------------------------------
# 5. Trusted-internal rate limiting
# ---------------------------------------------------------------------------

def _fake_request(headers=None, peer="8.8.4.4"):
    from app import real_ip
    req = types.SimpleNamespace()
    req.headers = headers or {}
    req.client = types.SimpleNamespace(host=peer)
    return real_ip(req)


def test_internal_rate_key():
    # Compose-network poller: private peer, no XFF -> internal key + big limit.
    key = _fake_request(peer="172.18.0.4")
    assert key.startswith("internal|")
    assert rate_limit_for(key) == RATE_LIMIT_INTERNAL
    # GH-stopgap localhost uvicorn.
    assert _fake_request(peer="127.0.0.1").startswith("internal|")
    # Public client via caddy: XFF present -> public limit even if the
    # claimed first hop is a private address (spoof attempt).
    key = _fake_request(headers={"x-forwarded-for": "10.0.0.1, 8.8.8.8"})
    assert key == "10.0.0.1" and rate_limit_for(key) == RATE_LIMIT
    # Direct public hit (no proxy, no XFF): public limit.
    key = _fake_request(peer="8.8.8.8")
    assert key == "8.8.8.8" and rate_limit_for(key) == RATE_LIMIT
    print("internal rate key ok")


def test_xff_sentinel_injection_rejected():
    # CRITICAL adversarial-review catch: a forged "internal|..." first hop
    # must NOT reach the internal bucket. A non-IP first hop falls back to
    # the last (caddy-appended) hop, then the socket peer -- every XFF path
    # yields a PUBLIC key.
    key = _fake_request(
        headers={"x-forwarded-for": "internal|127.0.0.1, 8.8.8.8"})
    assert not key.startswith("internal|")
    assert key == "8.8.8.8" and rate_limit_for(key) == RATE_LIMIT
    # Pure junk header: falls to the socket peer, still never internal.
    key = _fake_request(headers={"x-forwarded-for": "garbage, more-garbage"},
                        peer="172.18.0.9")
    assert key == "172.18.0.9" and rate_limit_for(key) == RATE_LIMIT
    # Legit proxied client keeps first-hop semantics.
    key = _fake_request(headers={"x-forwarded-for": "203.0.113.7, 8.8.8.8"})
    assert key == "203.0.113.7"
    print("XFF sentinel injection rejected ok")


def test_widen_bbox_crossing_stays_on_dateline():
    # Pre-wrapped crossing storm box (storm near 178E): the widened backdrop
    # box must stay centered at the dateline, not its ANTIPODE in the Gulf
    # of Guinea (adversarial-review catch, two lenses independently).
    out = fp.widen_bbox_to_view([171.0, -6.0, -175.0, 6.0])
    w, s, e, n = out
    e_uw = e + 360.0 if e < w else e
    center = (((w + e_uw) / 2.0 + 180.0) % 360.0) - 180.0
    assert abs(abs(center) - 178.0) < 1.0, out
    assert (s, n) == (-6.0, 6.0)
    assert 0 < e_uw - w <= 150.0 + 1e-6, out
    # Non-crossing boxes keep their old behavior exactly.
    plain = fp.widen_bbox_to_view([-100.0, 0.0, -5.0, 55.0])
    assert plain[0] <= -100.0 and plain[2] >= -5.0
    print("widen_bbox_to_view crossing ok")


def main():
    test_bbox_normalization()
    test_downsample_crossing_span()
    test_bbox_fractional_lons_pass_byte_identical()
    test_full_globe_span_keeps_pixel_budget()
    test_reversed_edges_rejected()
    test_bbox_helpers_crossing()
    test_pick_satellite_crossing()
    test_backdrop_crossing_render()
    test_render_png_crossing()
    test_render_png_plain_unchanged()
    test_429_recovers_and_honors_floor()
    test_429_honors_long_retry_after()
    test_429_budget_exhausts()
    test_429_budget_separate_from_transport_budget()
    test_internal_rate_key()
    test_xff_sentinel_injection_rejected()
    test_widen_bbox_crossing_stays_on_dateline()
    print("\nall antimeridian + 429 tests passed")


if __name__ == "__main__":
    main()
