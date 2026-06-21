"""POST /export -- server-side loop export (mp4 primary + global-palette gif).

Covers the request validation + frame bounds + format switch, the shared ffmpeg
-vf chain (even-pad for mp4, dwell always, optional motion interpolation), the
SSRF host guard, and a REAL encode proving the mp4 is valid + the gif carries ONE
global palette (no per-frame shimmer). The webp LOOP poller path + /render are
untouched (not exercised here).
"""
import os
import shutil
import sys
import tempfile

import pytest
from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402

FFMPEG = shutil.which("ffmpeg") is not None
CDN = "https://cdn.triple-a-tropics.com"
TWO = [CDN + "/a.png", CDN + "/b.png"]


# ---- request validation ---------------------------------------------------
def test_defaults():
    r = app.ExportRequest(frames=TWO)
    assert r.fps == 10 and r.skip == 0 and r.format == "mp4"
    assert r.interpolate is False and r.target_fps == 0


def test_format_must_be_mp4_or_gif():
    app.ExportRequest(frames=TWO, format="gif")     # ok
    app.ExportRequest(frames=TWO, format="mp4")     # ok
    with pytest.raises(ValidationError):
        app.ExportRequest(frames=TWO, format="avi")


def test_fps_and_skip_bounds():
    for bad in ({"fps": 0}, {"fps": 99}, {"skip": -1}, {"skip": 99},
                {"target_fps": 999}):
        with pytest.raises(ValidationError):
            app.ExportRequest(frames=TWO, **bad)


def test_frame_count_bounds():
    with pytest.raises(ValidationError):           # need >= 2
        app.ExportRequest(frames=[CDN + "/a.png"])
    big = [CDN + "/%d.png" % i for i in range(app.EXPORT_MAX_FRAMES + 1)]
    with pytest.raises(ValidationError):           # <= EXPORT_MAX_FRAMES
        app.ExportRequest(frames=big)


# ---- ffmpeg -vf chain -----------------------------------------------------
def test_vf_mp4_pads_even_gif_does_not():
    mp4 = app._export_vf(0.6, False, 10, for_mp4=True)
    gif = app._export_vf(0.6, False, 10, for_mp4=False)
    assert "pad=ceil(iw/2)*2:ceil(ih/2)*2" in mp4   # even dims for yuv420p
    assert "pad=ceil" not in gif                    # gif needs no even-dim pad
    # dwell (last-frame hold) is ALWAYS present in both -> chain never empty
    assert "tpad=stop_mode=clone" in mp4 and "tpad=stop_mode=clone" in gif


def test_vf_interpolate_toggle():
    assert "minterpolate" not in app._export_vf(0.6, False, 10, for_mp4=True)
    on = app._export_vf(0.6, True, 24, for_mp4=False)
    assert "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir" in on


# ---- SSRF host guard ------------------------------------------------------
def test_fetch_rejects_disallowed_host():
    tmp = tempfile.mkdtemp()
    try:
        with pytest.raises(app.HTTPException) as ei:
            app._fetch_frames_to_dir(["https://evil.example.com/a.png"], tmp)
        assert ei.value.status_code == 400
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fetch_rejects_redirect(monkeypatch):
    # An allowlisted host that 3xx-redirects (to an internal target) must be
    # REJECTED, not followed (SSRF). allow_redirects=False -> we see the 302.
    import requests

    class _Resp:
        status_code = 302
        def raise_for_status(self):
            pass
        def iter_content(self, n):
            return iter([])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    tmp = tempfile.mkdtemp()
    try:
        with pytest.raises(app.HTTPException) as ei:
            app._fetch_frames_to_dir([CDN + "/a.png"], tmp)
        assert ei.value.status_code == 502
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- real encode: format switch + global palette --------------------------
def _write_frames(tmp, n=6, w=201, h=151):   # odd dims exercise the even-pad
    for i in range(n):
        im = Image.new("RGB", (w, h), (8, 12, 20))
        x = int((w - 16) * i / max(1, n - 1))
        for yy in range(h // 2 - 8, h // 2 + 8):
            for xx in range(x, x + 16):
                im.putpixel((xx, yy), (240, 200, 120))
        im.save(os.path.join(tmp, "f_%05d.png" % i))


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")
def test_encode_format_switch_and_global_palette():
    tmp = tempfile.mkdtemp()
    try:
        _write_frames(tmp, n=30)   # a realistic loop length (x264 wins at scale)
        mp4, mt = app._encode_export(tmp, "mp4", 10, False, 10)
        assert mt == "video/mp4" and mp4[4:8] == b"ftyp"     # valid MP4 box
        gif, gt = app._encode_export(tmp, "gif", 10, False, 10)
        assert gt == "image/gif" and gif[:6] in (b"GIF89a", b"GIF87a")
        # one GLOBAL color table (logical-screen-descriptor flag bit 7) => every
        # frame shares ONE palette -> no per-frame NeuQuant shimmer.
        assert bool(gif[10] & 0x80)
        assert len(mp4) < len(gif)                           # mp4 smaller than gif
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")
def test_interpolate_adds_frames():
    tmp = tempfile.mkdtemp()
    try:
        _write_frames(tmp, n=10)

        def _count(data):
            p = os.path.join(tmp, "probe.mp4")
            with open(p, "wb") as f:
                f.write(data)
            import subprocess
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", p], capture_output=True, text=True).stdout
            return int(out.strip())
        plain, _ = app._encode_export(tmp, "mp4", 10, False, 10)
        interp, _ = app._encode_export(tmp, "mp4", 10, True, 20)
        assert _count(interp) > _count(plain)   # 2x interpolation ~doubles frames
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
