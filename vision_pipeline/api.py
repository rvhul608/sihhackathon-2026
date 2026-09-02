"""FastAPI adapter for the fixed Group 1 -> Group 2 vision contract."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .extractor import (
    DEFAULT_FIELD_CONFIDENCE_THRESHOLD,
    VALID_CATEGORIES,
    extract_sharpest_label_declarations,
)

app = FastAPI(title="SIH26034 Vision Extractor", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    file: Annotated[UploadFile | None, File()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    image_id: Annotated[str | None, Form()] = None,
    product_category: Annotated[str, Form()] = "other",
) -> dict:
    """Extract one image or choose the sharpest from repeated `files` uploads.

    The default response always contains only the agreed four top-level keys.
    `file` is kept for the existing browser fallback client; `files` is an
    optional repeated multipart field for 2-3 capture frames.
    """
    if product_category not in VALID_CATEGORIES:
        raise HTTPException(status_code=422, detail="Unsupported product_category")
    uploads = ([file] if file is not None else []) + (files or [])
    if not uploads:
        raise HTTPException(status_code=422, detail="Provide file or files")

    with TemporaryDirectory(prefix="sih26034-") as temp_dir:
        paths: list[Path] = []
        for index, upload in enumerate(uploads):
            if upload.content_type and not upload.content_type.startswith("image/"):
                raise HTTPException(status_code=415, detail="Only image uploads are supported")
            suffix = Path(upload.filename or "image.jpg").suffix or ".jpg"
            path = Path(temp_dir) / f"upload-{index}{suffix}"
            path.write_bytes(await upload.read())
            paths.append(path)
        try:
            return extract_sharpest_label_declarations(
                paths,
                image_id=image_id,
                product_category=product_category,
                field_confidence_threshold=DEFAULT_FIELD_CONFIDENCE_THRESHOLD,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
