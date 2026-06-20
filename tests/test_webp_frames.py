#!/usr/bin/env python3
"""Offline proofs for the 1056 px WebP loop-frame path (format=webp).

The contract under test, end to end:
  TRANSCODE  - render.transcode_frame Lanczos-downscales the 1320 px render to
               the requested width and re-encodes lossy WebP; never upscales.
  REQUEST    - RenderRequest accepts format png|webp, defaults png, rejects
               anything else (the draw-a-box panel and legacy callers are
               untouched by construction).
  CACHE      - _request_key for the WEBP LOOP path is byte-identical to its
               pre-resolution-tier key (poller frame-cache continuity); the png
               /custom-zoom path keys on the resolution tier so the tiers cache
               separately (the default tier's size changed, so it must NOT reuse
               the old tier-less key).
  POLLER     - floater_poller derives the uploaded extension + content-type
               from the /render RESPONSE, case-insensitively, falling back to
               .png for an old service that ignored the format param.

Run:  python tests/test_webp_frames.py -v
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import floater_poller as fp          # noqa: E402
from render import transcode_frame   # noqa: E402


def _png_bytes(w: int, h: int) -> bytes:
    from PIL import Image

    im = Image.new("RGB", (w, h))
    # A gradient, so lossy encoders have real work to do and the size
    # assertions below aren't comparing degenerate flat-color outputs.
    px = im.load()
    for x in range(w):
        for y in range(0, h, max(1, h // 64)):
            for yy in range(y, min(h, y + max(1, h // 64))):
                px[x, yy] = (x * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1), 128)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _dims(webp: bytes) -> tuple[int, int]:
    from PIL import Image

    return Image.open(io.BytesIO(webp)).size


class TestTranscodeFrame(unittest.TestCase):
    def test_emits_webp_at_requested_width(self):
        out = transcode_frame(_png_bytes(1320, 1101), 1056, 90)
        self.assertEqual(out[:4], b"RIFF")
        self.assertEqual(out[8:12], b"WEBP")
        w, h = _dims(out)
        self.assertEqual(w, 1056)
        # Aspect preserved: 1101 * 1056/1320 = 880.8 -> 881
        self.assertEqual(h, round(1101 * 1056 / 1320))

    def test_never_upscales(self):
        out = transcode_frame(_png_bytes(800, 600), 1056, 90)
        self.assertEqual(_dims(out), (800, 600))
        self.assertEqual(out[8:12], b"WEBP")

    def test_near_square_snaps_to_uniform_square(self):
        # The "seizure" root cause: the cartopy floater render occasionally
        # emits a 1px-short height (1320x1319), which downscales to 1056x1055
        # (round(1319*1056/1320)=1055) and made the loop resize the canvas mid
        # cycle. A near-square result must snap to a UNIFORM 1056x1056 so every
        # frame of the loop matches.
        self.assertEqual(_dims(transcode_frame(_png_bytes(1320, 1320), 1056, 90)),
                         (1056, 1056))
        self.assertEqual(_dims(transcode_frame(_png_bytes(1320, 1319), 1056, 90)),
                         (1056, 1056))            # the actual flip seen live
        self.assertEqual(_dims(transcode_frame(_png_bytes(1320, 1316), 1056, 90)),
                         (1056, 1056))            # within the +/-3px snap band

    def test_non_square_aspect_is_not_snapped(self):
        # A genuinely non-square frame (>3px off square) keeps its true aspect.
        self.assertEqual(_dims(transcode_frame(_png_bytes(1320, 1101), 1056, 90)),
                         (1056, round(1101 * 1056 / 1320)))

    def test_quality_knob_orders_sizes(self):
        src = _png_bytes(1320, 1101)
        lo = transcode_frame(src, 1056, 60)
        hi = transcode_frame(src, 1056, 95)
        self.assertLess(len(lo), len(hi))


class TestRenderRequestFormat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app

        cls.app = app

    def _req(self, **kw):
        base = dict(bbox=[-99.0, 34.0, -82.0, 49.0], channel="clean_ir")
        base.update(kw)
        return self.app.RenderRequest(**base)

    def test_default_is_png(self):
        self.assertEqual(self._req().format, "png")

    def test_webp_accepted_garbage_rejected(self):
        self.assertEqual(self._req(format="webp").format, "webp")
        with self.assertRaises(Exception):
            self._req(format="gif")

    def test_webp_loop_cache_key_is_byte_identical_to_legacy(self):
        # The WEBP loop path (the floater/meso poller frames) MUST keep its
        # pre-resolution-tier key so the durable frame cache stays continuous
        # across this deploy. quality is irrelevant there (no q-part).
        body = self._req(format="webp")
        key = self.app._request_key(body, "clean_ir", "2026-06-12T06:00:00", "noaa-goes19")
        legacy_raw = (
            f"{body.bbox}|2026-06-12T06:00:00|clean_ir|{body.enhancement}|noaa-goes19|fmt=webp"
        )
        self.assertEqual(key, hashlib.sha256(legacy_raw.encode()).hexdigest())

    def test_png_cache_key_carries_resolution_tier(self):
        # The png/custom-zoom path now keys on the resolution tier so the three
        # tiers cache separately. The default tier's output size changed
        # (full-res -> ~1500px), so it INTENTIONALLY no longer matches the old
        # tier-less key -- the in-memory cache must not serve a stale full-res
        # frame as the new default.
        body = self._req()  # default png
        key = self.app._request_key(body, "clean_ir", "2026-06-12T06:00:00", "noaa-goes19")
        legacy_raw = (
            f"{body.bbox}|2026-06-12T06:00:00|clean_ir|{body.enhancement}|noaa-goes19"
        )
        self.assertNotEqual(key, hashlib.sha256(legacy_raw.encode()).hexdigest())
        self.assertEqual(key, hashlib.sha256((legacy_raw + "|q=default").encode()).hexdigest())

    def test_webp_keys_separately_from_png(self):
        png_key = self.app._request_key(
            self._req(), "clean_ir", "2026-06-12T06:00:00", "noaa-goes19"
        )
        webp_key = self.app._request_key(
            self._req(format="webp"), "clean_ir", "2026-06-12T06:00:00", "noaa-goes19"
        )
        self.assertNotEqual(png_key, webp_key)


class TestPollerExtensionFromResponse(unittest.TestCase):
    def test_webp_response(self):
        self.assertEqual(
            fp.frame_ext({"Content-Type": "image/webp"}), (".webp", "image/webp")
        )

    def test_case_insensitive_and_params_stripped(self):
        self.assertEqual(
            fp.frame_ext({"content-type": "IMAGE/WEBP; charset=binary"}),
            (".webp", "image/webp"),
        )

    def test_old_service_or_rollback_falls_back_to_png(self):
        self.assertEqual(
            fp.frame_ext({"Content-Type": "image/png"}), (".png", "image/png")
        )
        self.assertEqual(fp.frame_ext({}), (".png", "image/png"))

    def test_frame_key_carries_extension(self):
        ts = dt.datetime(2026, 6, 12, 6, 0, tzinfo=dt.timezone.utc)
        self.assertTrue(
            fp.frame_key("wp06", "ir", ts, ".webp").endswith("/20260612T0600Z.webp")
        )
        # Default stays .png -- pre-existing callers/fixtures unaffected.
        self.assertTrue(fp.frame_key("wp06", "ir", ts).endswith("/20260612T0600Z.png"))

    def test_call_render_sends_format(self):
        sent = {}

        class _Resp:
            status_code = 200
            content = b"x"
            headers = {"Content-Type": "image/webp"}

            def raise_for_status(self):
                return None

        class _Session:
            def post(self, url, json=None, timeout=None):
                sent.update(json)
                return _Resp()

        fp.call_render(_Session(), [-99.0, 34.0, -82.0, 49.0], "ir", "rainbow_ir")
        self.assertEqual(sent.get("format"), fp.FRAME_FORMAT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
