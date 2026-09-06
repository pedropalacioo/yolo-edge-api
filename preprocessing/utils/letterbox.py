"""Aspect-ratio preserving resize helpers."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    size: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize ``image`` to a square and add symmetric padding.

    Returns the image, uniform scale and the left/top padding. Any odd
    remainder is placed on the right/bottom.
    """
    if size <= 0:
        raise ValueError("size must be greater than zero")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image must have non-zero dimensions")

    scale = min(size / width, size / height)
    resized_w = min(size, round(width * scale))
    resized_h = min(size, round(height * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = size - resized_w
    pad_h = size - resized_h
    left = pad_w // 2
    top = pad_h // 2
    output = cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return output, scale, (left, top)


def adjust_bboxes(
    boxes_xyxy: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    pad_w: int = 0,
    pad_h: int = 0,
    original_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Map xyxy boxes from processed coordinates to original coordinates."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("scales must be greater than zero")
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale_y
    if original_size is not None:
        height, width = original_size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes
