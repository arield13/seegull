import numpy as np
import pytest
from seegull.data.bounding_box import box_area, iou


def test_box_area():
    assert box_area([20, 10, 30, 20]) == 100

    a = box_area(
        np.array(
            [
                [20, 100],
                [10, 150],
                [30, 300],
                [20, 160],
            ]
        )
    )

    assert a[0] == 100
    assert a[1] == 200 * 10


def test_iou():
    i = iou(
        np.array([[50, 100, 200, 300], [0, 0, 10, 10], [0, 0, 10, 10]]),
        np.array([[80, 120, 220, 310], [20, 20, 30, 30], [0, 0, 10, 10]]),
    )

    assert i[0] == pytest.approx(0.617, 1e-2)
    assert i[1] == 0
    assert i[2] == 1
