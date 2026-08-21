"""Exact device geometry and 1:1 imagery-layer placement for M2."""

from __future__ import annotations

import io

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import render


def _figure(width=4.0, height=3.0, dpi=100):
    fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor=render.DARK_BG)
    ax = fig.add_axes([0.11, 0.17, 0.77, 0.66], projection=ccrs.PlateCarree())
    ax.set_extent([-81.0, -65.0, 8.0, 20.0], crs=ccrs.PlateCarree())
    fig.canvas.draw()
    return fig, ax


def test_device_bounds_follow_resolved_fractional_axes_not_a_fixed_offset():
    fig, ax = _figure(width=4.13, height=3.07, dpi=97)
    try:
        top, bottom, left, right = render._axes_device_pixel_bounds(fig, ax)
        bbox = ax.get_window_extent(fig.canvas.get_renderer())
        h = int(round(fig.bbox.height))
        assert left == int(np.ceil(bbox.x0 - 0.5))
        assert right == int(np.ceil(bbox.x1 - 0.5))
        assert top == int(np.ceil(h - bbox.y1 - 0.5))
        assert bottom == int(np.ceil(h - bbox.y0 - 0.5))
        assert (bottom - top) > 0 and (right - left) > 0
        # This deliberately non-square/fractional fixture guards against the
        # legacy assumption that one fixed +1 row describes every backend.
        assert (bottom - top) != (right - left)
    finally:
        plt.close(fig)


def test_figimage_places_exact_raster_without_a_second_spatial_transform():
    fig, ax = _figure()
    try:
        bounds = render._axes_device_pixel_bounds(fig, ax)
        top, bottom, left, right = bounds
        raster = np.zeros((bottom - top, right - left, 3), np.float32)
        raster[..., 0] = 0.8
        raster[..., 1] = 0.2
        raster[..., 2] = 0.1
        artist = render._place_device_raster(fig, ax, raster, bounds)
        assert artist.get_array().shape == raster.shape
        assert artist.ox == left
        assert artist.oy == int(round(fig.bbox.height)) - bottom
        assert ax.patch.get_alpha() == 0.0

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.dpi, facecolor=render.DARK_BG)
        decoded = np.asarray(Image.open(buf).convert("RGB"))
        sample = decoded[(top + bottom) // 2, (left + right) // 2]
        np.testing.assert_allclose(sample, np.array([204, 51, 26]), atol=1)
    finally:
        plt.close(fig)


def test_device_layer_rejects_shape_proxy_and_nonfinite_pixels():
    fig, ax = _figure()
    try:
        bounds = render._axes_device_pixel_bounds(fig, ax)
        top, bottom, left, right = bounds
        wrong = np.zeros((bottom - top + 1, right - left, 3), np.float32)
        with pytest.raises(ValueError, match="does not match bounds"):
            render._place_device_raster(fig, ax, wrong, bounds)

        bad = np.zeros((bottom - top, right - left, 3), np.float32)
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="must be finite"):
            render._place_device_raster(fig, ax, bad, bounds)
    finally:
        plt.close(fig)
