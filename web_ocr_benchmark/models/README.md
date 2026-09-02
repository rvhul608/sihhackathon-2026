# Model assets

This directory is intentionally empty in Git: models are binary assets and are downloaded locally.

From this directory's parent, run:

```bash
./scripts/download-and-convert-models.sh
```

Use a Python 3.10–3.12 environment with `paddle2onnx` installed. The script downloads PaddleOCR's PP-OCRv3 English mobile inference models and converts them to the dynamic-shape ONNX pair consumed by `benchmark.js`.

## Fast benchmark-only route

If a compatible Python conversion environment is unavailable, run
`../scripts/download-preconverted-benchmark-models.sh` from this directory's parent.
It downloads an already-converted PP-OCRv3 English pair and prints their SHA-256 values. This is acceptable for the Phase 0 latency gate only; use the reproducible Paddle2ONNX script above for the final handoff assets.
