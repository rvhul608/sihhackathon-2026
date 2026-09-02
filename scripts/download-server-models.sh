#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/vision_pipeline/models"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$MODELS"
curl -fL --retry 3 -o "$WORK/det.tar" \
  "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
curl -fL --retry 3 -o "$WORK/rec.tar" \
  "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar"
tar -xf "$WORK/det.tar" -C "$MODELS"
tar -xf "$WORK/rec.tar" -C "$MODELS"

printf 'Downloaded server models to %s\n' "$MODELS"
