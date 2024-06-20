"""Utilities for working with bounding boxes."""

import numpy as np


def box_area(xyxy: np.ndarray) -> np.ndarray:
    """Return the area of bounding boxes in xyxy format.

    Args:
        xyxy: Either a single array of 4 elements or a 2D
            array of shape (4, n)

    Returns:
        The area of the bounding box(es)
    """
    return (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])


def iou(boxes_true: np.ndarray, boxes_pred: np.ndarray) -> np.ndarray:
    """Return the element-wise intersection-over-union (IoU) between
    two arrays of bounding boxes.

    Returns the IoU of each bounding box in boxes_true with the corresponding
    bounding box in boxes_pred. So the 0th element of the return value will be
    the IoU of boxes_true[0] with boxes_pred[0] and so on.

    Args:
        boxes_true: A numpy array of bounding boxes in xyxy format with shape (n, 4)
        boxes_pred: A numpy array of bounding boxes in xyxy format with shape (n, 4)

    Returns:
        A 1-dimensional array of floats with the IoU of each pair of bounding boxes
    """
    area_pred = box_area(boxes_pred.T)
    area_true = box_area(boxes_true.T)

    top_left = np.maximum(boxes_true[:, :2], boxes_pred[:, :2])
    bottom_right = np.minimum(boxes_true[:, 2:], boxes_pred[:, 2:])

    area_inter = np.prod(
        np.clip(bottom_right - top_left, a_min=0, a_max=None), axis=1
    )

    return area_inter / (area_true + area_pred - area_inter)
