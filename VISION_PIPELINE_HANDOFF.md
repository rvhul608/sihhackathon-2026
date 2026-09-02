# Vision pipeline handoff

The callable boundary is:

```python
from vision_pipeline import extract_label_declarations

result = extract_label_declarations(
    "/path/to/label.jpg",
    image_id="img_0001",
    product_category="packaged_food",
)
```

`result` always has the agreed top-level keys: `image_id`, `product_category`,
`extracted_fields`, and `raw_ocr_text`. Every required `extracted_fields` key is
always present. An absent declaration is exactly
`{"value": null, "confidence": 0.0, "present": false}`.

## FastAPI service

The service now lives at `vision_pipeline.api:app`:

```bash
uvicorn vision_pipeline.api:app --host 0.0.0.0 --port 8000
```

`POST /extract` accepts the existing multipart `file` field, plus optional
`image_id` and `product_category` form fields. It also accepts repeated
multipart `files` fields and selects the sharpest frame before OCR. The response
is returned directly and has only the agreed keys: `image_id`,
`product_category`, `extracted_fields`, and `raw_ocr_text`.

## Environment

Use Python 3.10–3.12. The current workspace Python 3.14 cannot install the
required Paddle runtime. Install the project requirements, then install exactly
one PaddlePaddle wheel matching the host CUDA runtime. For GPU production use,
this must be `paddlepaddle-gpu`, not the CPU package:

```bash
python -m pip install -r requirements-vision.txt
# install the CUDA-matched paddlepaddle-gpu wheel from PaddlePaddle's index
```

On this workstation, a local Python 3.12 runtime is available at
`.tools/python/bin/python3.12` and NVIDIA reports CUDA 13.2. Create the
isolated environment and use Paddle's CUDA 13.0 wheel (the installed driver is
backwards-compatible with it):

```bash
.tools/python/bin/python3.12 -m venv .vision-venv
.vision-venv/bin/python -m pip install -r requirements-vision.txt
.vision-venv/bin/python -m pip install paddlepaddle-gpu==3.3.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

Download and unpack PP-OCR server inference models, then point the runtime at
them before starting Uvicorn:

```bash
./scripts/download-server-models.sh
export PADDLEOCR_USE_GPU=true
export PADDLEOCR_SERVER_DET_MODEL_DIR="$PWD/vision_pipeline/models/en_PP-OCRv3_det_infer"
export PADDLEOCR_SERVER_REC_MODEL_DIR="$PWD/vision_pipeline/models/en_PP-OCRv3_rec_infer"
```

The server pipeline preserves full resolution, applies CLAHE before label
localisation, uses a four-point perspective correction where a rectangular
label is found, and penalizes OCR confidence when bright low-saturation glare
overlaps the corrected crop. It deliberately rejects fields below 0.50 rather
than presenting them as trustworthy. Curved wraparound labels remain a known
limitation.

Before deployment, verify GPU execution with `paddle.utils.run_check()` and a
real extraction while observing `nvidia-smi`. Do not set `PADDLEOCR_USE_GPU`
to false for the online fallback without separately approving CPU latency.
