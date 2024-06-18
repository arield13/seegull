#!/usr/bin/env python

"""Tests for `seegull.data.image` package."""

import shutil
from pathlib import Path

import pandas as pd
import PIL
import pytest
from seegull.data import image

TEST_IMAGE = "https://storage.googleapis.com/seegull-test-data/image-recycling-26061-651438125879274ed7cf6047-7d65f0894860efb7-573076577817.jpg"
TEST_CSV = "tests/data/images.csv"
IMAGE_DIR = Path("/tmp/seegull_test_images")


@pytest.fixture(scope="session", autouse=True)
def clear_and_create_image_directory(request):
    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)
    IMAGE_DIR.mkdir()


@pytest.fixture
def im():
    return image.Image(url=TEST_IMAGE, path=IMAGE_DIR / "test.jpg")


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


@pytest.fixture
def df():
    df = pd.read_csv(TEST_CSV)
    df["path"] = df["image_id"].apply(lambda i: IMAGE_DIR / f"{i}.jpg")
    return df


def test_load_images(df):
    df["image"] = image.load_images(df)

    row = df.iloc[0]
    assert isinstance(row.image, image.Image)
    assert row.image.width == row.image_width


def test_load_images_with_options(df):
    df["image"] = image.load_images(df, crop=True, return_type="PIL")

    row = df.iloc[0]
    assert isinstance(row.image, PIL.Image.Image)
    assert row.image.width == (int(row.x2) - int(row.x1))
