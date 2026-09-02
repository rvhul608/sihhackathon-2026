# Phase 0: Android on-device OCR benchmark

This is the benchmark gate for the offline tier. It uses PP-OCRv3's English mobile detector and recognizer in ONNX Runtime Web/WASM. No API request is made during inference.

## Prepare models

The ONNX files are not checked into Git. In a Python 3.10–3.12 environment, install `paddle2onnx`, then run:

```bash
cd web_ocr_benchmark
./scripts/download-and-convert-models.sh
```

For today's latency gate only, when a compatible conversion Python is not available:

```bash
./scripts/download-preconverted-benchmark-models.sh
```

This uses an already-converted PP-OCRv3 English ONNX pair and prints checksums. Replace it with the reproducibly converted pair before final integration.

## Run on an Android phone

Serve the repository over HTTP from the laptop; camera/file access and model fetches will not work reliably from a `file://` URL.

```bash
cd web_ocr_benchmark
npx serve -l 8080
```

Connect the phone to the same Wi-Fi, then open `http://<laptop-LAN-IP>:8080` on the phone. Choose a real label photo (or use the camera picker), tap **Run benchmark**, and send back the displayed `Total` latency.

The gate is 1–3 seconds. At 6+ seconds, stop before Phase 1 and assess WebGPU, a smaller model, or a server-only demo path.

## Notes

- The benchmark measures detection plus sequential recognition of detected text regions, including browser preprocessing/postprocessing.
- Camera images are downscaled to a 1280-pixel longest side before OCR. This prevents out-of-memory failures on phones while retaining more detail than the detector's 960-pixel input.
- It is deliberately standalone: Hrishi and Devika can later integrate the Phase 1 module into their PWA without adopting a framework.
- The model conversion follows PaddleOCR's documented Paddle2ONNX process for PP-OCRv3 English detector and recognizer.
