#!/usr/bin/env bash
set -euo pipefail

# Fast path only for the Phase 0 device gate. These are pre-converted ONNX files
# for the same PP-OCRv3 English mobile detector/recognizer named in the benchmark.
# The normal, reproducible Paddle2ONNX route remains download-and-convert-models.sh.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"

curl -fL --retry 3 -o "$MODELS/en_PP-OCRv3_det.onnx" "https://huggingface.co/deepghs/paddleocr/resolve/main/det/en_PP-OCRv3_det/model.onnx"
curl -fL --retry 3 -o "$MODELS/en_PP-OCRv3_rec.onnx" "https://huggingface.co/deepghs/paddleocr/resolve/main/rec/en_PP-OCRv3_rec/model.onnx"
curl -fL --retry 3 -o "$MODELS/en_dict.txt" "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/en_dict.txt"

echo "Expected SHA-256 (for source verification):"
echo "det: 69d10a2f151e0561e7e6c948ff0207a5fb84789fa6a4591d1d08138e3d82f1f9"
echo "rec: 18fae12e175a4e8616b57fb001ed51270eb81ce126c652d0f88e297a8de53f49"
sha256sum "$MODELS/en_PP-OCRv3_det.onnx" "$MODELS/en_PP-OCRv3_rec.onnx"
