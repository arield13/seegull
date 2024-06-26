import shutil
from pathlib import Path

import pandas as pd
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


@pytest.fixture
def annotation_df():
    df = pd.read_csv(TEST_CSV)
    df["path"] = df["image_id"].apply(lambda i: IMAGE_DIR / f"{i}.jpg")
    return df
