#!/usr/bin/env python

"""Tests for `seegull.data.image` package."""

import PIL
import pytest

from seegull.data import image


def test_download_image(im):
    assert im is not None
    assert im.image is not None


def test_resize(im):
    im2 = im.resize(100, 200)
    assert im2.width == 100
    assert im2.height == 200

    im3 = im.resize(0.01, min_width=200, min_height=200)
    assert im3.width >= 200
    assert im3.height >= 200
    assert im3.width < im.width
    assert im3.height < im.height
    assert (im.width / im.height) == pytest.approx(im3.width / im3.height, 1e-2)

    im4 = im.resize(10.0, max_width=2000, max_height=2000)
    assert im4.width <= 2000
    assert im4.height <= 2000
    assert im4.width > im.width
    assert im4.height > im.height
    assert (im.width / im.height) == pytest.approx(im4.width / im4.height, 1e-2)


def test_crop_with_padding(im):
    im2 = im.crop([0, 20, 100, 100], 15)

    assert im2.width == 115
    assert im2.height == 110


def test_load_images(annotation_df):
    annotation_df["image"] = image.load_images(annotation_df)

    row = annotation_df.iloc[0]
    assert isinstance(row.image, image.Image)
    assert row.image.width == row.image_width


def test_load_images_with_options(annotation_df):
    annotation_df["image"] = image.load_images(
        annotation_df, crop=True, return_type="PIL"
    )

    row = annotation_df.iloc[0]
    assert isinstance(row.image, PIL.Image.Image)
    assert row.image.width == (int(row.x2) - int(row.x1))


def test_get_image_df(annotation_df):
    image_df = image.get_image_df(annotation_df)
    assert len(image_df) < len(annotation_df)
    assert len(image_df) == image_df["image_id"].nunique()
    assert "image" in image_df.columns
