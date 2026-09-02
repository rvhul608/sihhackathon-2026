#!/usr/bin/env bash
set -euo pipefail

# Run this using Python 3.10–3.12, not this workspace's Python 3.14.
# It follows PaddleOCR's documented PP-OCRv3 English inference-model → ONNX path.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
WORK="$MODELS/.paddle-inference"
mkdir -p "$WORK" "$MODELS"

command -v paddle2onnx >/dev/null || {
  echo "paddle2onnx is required. In a Python 3.10–3.12 environment: python -m pip install paddle2onnx"
  exit 1
}

curl -fL --retry 3 -o "$WORK/det.tar" "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
curl -fL --retry 3 -o "$WORK/rec.tar" "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar"
tar -xf "$WORK/det.tar" -C "$WORK"
tar -xf "$WORK/rec.tar" -C "$WORK"

paddle2onnx --model_dir "$WORK/en_PP-OCRv3_det_infer" --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file "$MODELS/en_PP-OCRv3_det.onnx" --opset_version 11 --enable_onnx_checker True
paddle2onnx --model_dir "$WORK/en_PP-OCRv3_rec_infer" --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file "$MODELS/en_PP-OCRv3_rec.onnx" --opset_version 11 --enable_onnx_checker True
curl -fL --retry 3 -o "$MODELS/en_dict.txt" "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/en_dict.txt"
rm -rf "$WORK"
echo "Models written to $MODELS"
