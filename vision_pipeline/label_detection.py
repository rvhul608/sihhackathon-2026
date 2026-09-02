"""Simple contour-based label crop for clean, front-facing photographs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_image(image_path: str | Path) -> Any:
    """Decode an image without silently resizing it."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required; install requirements-vision.txt.") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    return image


def clahe_bgr(image: Any) -> Any:
    """Improve local luminance contrast while preserving colour for OCR."""
    import cv2

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, a_channel, b_channel = cv2.split(lab)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance)
    return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def laplacian_sharpness(image: Any) -> float:
    import cv2

    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def glare_ratio(image: Any) -> float:
    """Fraction covered by bright, low-saturation reflection-like pixels."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 0, 245)), np.array((180, 80, 255)))
    return float(cv2.countNonZero(mask)) / float(mask.size)


def detect_and_crop_label(image_path: str | Path) -> Any:
    """Return the best rectangular label proposal, or the complete image.

    This is intentionally a baseline, not an object detector: it suits today's
    clean photos without requiring a labelled training set.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required; install requirements-vision.txt.") from exc

    image = load_image(image_path)
    return detect_and_crop_label_image(image)


def detect_and_crop_label_image(image: Any) -> Any:
    """Find a rectangular label and perspective-correct it, with full-image fallback."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required; install requirements-vision.txt.") from exc

    height, width = image.shape[:2]
    image_area = height * width
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best: tuple[float, Any] | None = None
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        if not 0.12 <= area_ratio <= 0.92:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, rect_width, rect_height = cv2.boundingRect(polygon)
        rectangularity = area / max(rect_width * rect_height, 1)
        if rectangularity < 0.60:
            continue
        # Favour a large, rectangular proposal but avoid treating the whole photo as a label.
        score = area_ratio * rectangularity
        if best is None or score > best[0]:
            best = (score, polygon.reshape(4, 2).astype("float32"))

    if best is None:
        return image
    return _four_point_crop(image, best[1], cv2, np)


def _four_point_crop(image: Any, points: Any, cv2: Any, np: Any) -> Any:
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    top_left, bottom_right = points[np.argmin(sums)], points[np.argmax(sums)]
    top_right, bottom_left = points[np.argmin(diffs)], points[np.argmax(diffs)]

    def distance(a: Any, b: Any) -> float:
        return float(np.linalg.norm(a - b))

    target_width = int(max(distance(bottom_right, bottom_left), distance(top_right, top_left)))
    target_height = int(max(distance(top_right, bottom_right), distance(top_left, bottom_left)))
    if target_width < 20 or target_height < 20:
        return image
    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"), destination
    )
    return cv2.warpPerspective(image, transform, (target_width, target_height))
