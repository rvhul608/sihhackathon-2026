"""Full-resolution server OCR pipeline for the fixed vision JSON contract."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Iterable, Literal

from .label_detection import (
    clahe_bgr,
    detect_and_crop_label_image,
    glare_ratio,
    laplacian_sharpness,
    load_image,
)
from .parsing import ParsedLine, parse_declarations

ProductCategory = Literal[
    "packaged_food", "cosmetics", "electronics", "household", "other"
]
VALID_CATEGORIES = {"packaged_food", "cosmetics", "electronics", "household", "other"}
DEFAULT_FIELD_CONFIDENCE_THRESHOLD = 0.50
GLARE_WARNING_RATIO = 0.025


def extract_label_declarations(
    image_path: str | Path,
    *,
    image_id: str | None = None,
    product_category: ProductCategory = "other",
    field_confidence_threshold: float = DEFAULT_FIELD_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Extract label declarations from one full-resolution product image.

    Input is never downscaled. CLAHE is applied before contour localisation and
    again to the perspective-corrected crop. Glare degrades OCR-line confidence
    before the field threshold is applied, so a reflection cannot become a
    confidently-present declaration merely because the recognizer guessed text.
    """
    _validate_request(product_category, field_confidence_threshold)
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    source = load_image(path)
    label_image = detect_and_crop_label_image(clahe_bgr(source))
    reflection = glare_ratio(label_image)
    lines = _run_paddle_ocr(clahe_bgr(label_image))
    lines = _apply_glare_penalty(lines, reflection)
    raw_ocr_text = "\n".join(line.text for line in lines)

    return {
        "image_id": image_id or path.stem,
        "product_category": product_category,
        "extracted_fields": parse_declarations(
            lines, min_confidence=field_confidence_threshold
        ),
        "raw_ocr_text": raw_ocr_text,
    }


def extract_sharpest_label_declarations(
    image_paths: Iterable[str | Path],
    *,
    image_id: str | None = None,
    product_category: ProductCategory = "other",
    field_confidence_threshold: float = DEFAULT_FIELD_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Select the sharpest submitted frame before running the costly OCR pass."""
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("At least one image is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
    sharpest = max(paths, key=lambda path: laplacian_sharpness(load_image(path)))
    return extract_label_declarations(
        sharpest,
        image_id=image_id or sharpest.stem,
        product_category=product_category,
        field_confidence_threshold=field_confidence_threshold,
    )


def _validate_request(product_category: str, field_confidence_threshold: float) -> None:
    if product_category not in VALID_CATEGORIES:
        raise ValueError(
            "product_category must be one of: " + ", ".join(sorted(VALID_CATEGORIES))
        )
    if not 0.0 <= field_confidence_threshold <= 1.0:
        raise ValueError("field_confidence_threshold must be between 0.0 and 1.0")


def _apply_glare_penalty(lines: list[ParsedLine], reflection: float) -> list[ParsedLine]:
    if reflection <= GLARE_WARNING_RATIO:
        return lines
    # At 10% reflection, retain at most 30% of OCR confidence. This deliberately
    # routes uncertain declarations below the normal 0.5 server threshold.
    multiplier = max(0.30, 1.0 - (reflection - GLARE_WARNING_RATIO) / 0.075)
    return [ParsedLine(line.text, line.confidence * multiplier) for line in lines]


@lru_cache(maxsize=1)
def _paddle_ocr() -> Any:
    """Load the configured server model once; never silently choose CPU."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Use Python 3.10–3.12 and install "
            "requirements-vision.txt plus a matching paddlepaddle-gpu wheel."
        ) from exc

    use_gpu = os.environ.get("PADDLEOCR_USE_GPU", "true").lower() in {"1", "true", "yes"}
    options: dict[str, Any] = {
        "use_angle_cls": True,
        "lang": "en",
        "show_log": False,
        "use_gpu": use_gpu,
    }
    # Point these at the downloaded PP-OCR server inference directories. Leaving
    # them unset permits PaddleOCR's own model management during local bootstrap.
    for argument, variable in (
        ("det_model_dir", "PADDLEOCR_SERVER_DET_MODEL_DIR"),
        ("rec_model_dir", "PADDLEOCR_SERVER_REC_MODEL_DIR"),
        ("cls_model_dir", "PADDLEOCR_CLS_MODEL_DIR"),
    ):
        if model_dir := os.environ.get(variable):
            options[argument] = model_dir
    return PaddleOCR(**options)


def _run_paddle_ocr(image: Any) -> list[ParsedLine]:
    """Run PaddleOCR 2.x and retain per-line recognition confidence."""
    result = _paddle_ocr().ocr(image, cls=True)
    lines: list[ParsedLine] = []
    for page in result or []:
        if not page:
            continue
        for item in page:
            if not item or len(item) < 2:
                continue
            text, confidence = item[1]
            clean_text = str(text).strip()
            if clean_text:
                lines.append(ParsedLine(clean_text, float(confidence)))
    return lines
