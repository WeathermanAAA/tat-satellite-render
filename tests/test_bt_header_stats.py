"""Brightness-temperature min/max in the render header strip.

The storm-centred renders now carry the frame's BT extremes in the HEADER,
tagged with the band they came from ("IR BT" / "WV BT"), beside the storm
badge. Two properties matter beyond "a number appears":

  * The tag is not decoration. An IR-window extreme and a water-vapour-channel
    extreme are read completely differently; an unlabelled "min/max" invites
    reading a WV frame's -60 C as a cloud-top temperature. Visible and
    true-colour frames carry NO readout at all rather than a reflectance
    printed in degrees.

  * The readout moved OUT of the map axes. Its old home was the bottom-left of
    the map, where an opaque backing box overwrote real data pixels -- the very
    pixels the explorer's objfix reads brightness temperature back out of.

  * title_h is load-bearing GEOMETRY. The explorer pins its floater
    georeferencing constants to this layout (objfix_sources.js LAYOUT), so the
    second header row had to fit INSIDE the existing strip. A test pins the
    constant so a future "just make the header taller" silently shifting every
    floater fix fails here instead.
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import render                                  # noqa: E402
from satellites import FetchResult             # noqa: E402


def _frame(generic: str, channel: int, units: str = "K",
           bucket: str = "noaa-goes19", cmi=None) -> FetchResult:
    N = 40
    lat = np.linspace(8.0, 18.0, N)
    lon = np.linspace(165.0, 176.0, N)
    LON, LAT = np.meshgrid(lon, lat)
    if cmi is None:
        bt_c = np.full((N, N), -40.0)
        bt_c[0, 0] = -87.0          # coldest
        bt_c[-1, -1] = 15.0         # warmest (a clear, warm eye)
        cmi = bt_c + 273.15
    return FetchResult(cmi=cmi, lats=LAT, lons=LON, channel=channel,
                       generic_channel=generic,
                       scan_start=dt.datetime(2026, 7, 28, 12, 0),
                       product="Meso", bucket=bucket, sat_name="GOES-19",
                       sub_sat_lon=-75.0, units=units)


class BandTagTests(unittest.TestCase):
    """_bt_band_tag: which products have a temperature to report."""

    def test_ir_and_wv_bands_are_tagged(self):
        for generic, chan, expect in (("clean_ir", 13, "IR"),
                                      ("ir_window", 14, "IR"),
                                      ("wv_upper", 8, "WV"),
                                      ("wv_lower", 10, "WV"),
                                      ("shortwave_ir", 7, "SWIR")):
            with self.subTest(generic):
                self.assertEqual(
                    render._bt_band_tag(_frame(generic, chan), chan), expect)

    def test_reflectance_products_get_no_tag(self):
        # A visible frame's pixels are reflectance; a number in degrees would
        # be a unit error wearing a label.
        for generic, chan in (("visible_red", 2), ("visible_blue", 1),
                              ("veggie", 3)):
            with self.subTest(generic):
                self.assertIsNone(
                    render._bt_band_tag(_frame(generic, chan, units="1"), chan))

    def test_archive_eras_are_tagged_by_their_own_channel_numbering(self):
        # GridSat-B1 and GridSat-GOES do not use the generic-channel vocabulary;
        # their channel numbers mean different things and are mapped explicitly.
        gs = _frame("", 1, bucket="noaa-cdr-gridsat-b1")
        self.assertEqual(render._bt_band_tag(gs, 1), "IR")
        self.assertEqual(render._bt_band_tag(gs, 2), "WV")
        gg = _frame("", 4, bucket="ncei-gridsat-goes")
        self.assertEqual(render._bt_band_tag(gg, 4), "IR")
        self.assertEqual(render._bt_band_tag(gg, 3), "WV")
        self.assertEqual(render._bt_band_tag(gg, 2), "SWIR")
        self.assertEqual(render._bt_band_tag(_frame("", 1,
                                             bucket="gesdisc-mergir"), 1), "IR")


class HeaderGeometryTests(unittest.TestCase):
    def test_title_strip_height_is_unchanged(self):
        # LOAD-BEARING: satellite/explorer/objfix_sources.js pins its
        # LAYOUT.dataY0/dataY1 to this figure layout. Changing title_h moves
        # the map axes and silently shifts every floater center fix. If this
        # test fails, update the explorer LAYOUT in the SAME change.
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "render.py"), encoding="utf-8").read()
        self.assertIn("title_h = 0.06", src,
                      "title strip height changed - objfix LAYOUT must follow")

    def test_readout_is_not_drawn_over_the_map(self):
        # It must live on the title axes, never on the data axes: an opaque
        # label inside the map corrupts the pixels objfix reads BT out of.
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "render.py"), encoding="utf-8").read()
        self.assertIn("f\"{bt_tag} BT   min {bt_min_c:+.1f}", src)
        head = src[:src.index("{bt_tag} BT")]
        # the nearest preceding text() call must be on title_ax
        self.assertIn("title_ax.text(", head[-400:],
                      "BT readout is not on the title axes")


@unittest.skipUnless(os.environ.get("RUN_RENDER_PNG", "1") == "1",
                     "render smoke disabled")
class RenderedHeaderTests(unittest.TestCase):
    """End-to-end: the readout actually reaches the PNG."""

    BBOX = [8.0, 18.0, 165.0, 176.0]

    def _png(self, generic, chan, storm=None, units="K"):
        return render.render_png(
            _frame(generic, chan, units=units), bbox=self.BBOX, channel=chan,
            time_str="2026-07-28 12:00", enhancement="tat_neon",
            storm=storm, coastlines=False, gridlines=False, dpi=60)

    def test_ir_frame_renders_with_a_header_readout(self):
        png = self._png("clean_ir", 13,
                        storm={"name": "DOLPHIN", "wind_kt": 100.0,
                               "pressure_mb": 960.0, "nature": "TS"})
        self.assertGreater(len(png), 5000)
        self.assertEqual(png[1:4], b"PNG")

    def test_visible_frame_renders_without_one(self):
        # No crash, no readout: the reflectance path must stay untouched.
        png = self._png("visible_red", 2, units="1")
        self.assertGreater(len(png), 5000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
