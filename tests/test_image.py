#!/usr/bin/env python

"""Tests for `seegull.data.image` package."""

from seegull.data import image


def test_download_image():
    im = image.Image(
        url="https://storage.googleapis.com/bower-eu-production-bucket/production/package-picture-48675-61b47b94718eda00112d31e5-5701092108869-front.jpg"
    )

    assert im is not None
    assert im.image is not None
