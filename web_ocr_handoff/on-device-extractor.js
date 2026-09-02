import { parseDeclarations } from "./field-parser.js";

/** Thrown instead of silently OCR'ing through material glare. */
export class CaptureQualityError extends Error {
  constructor(code, message) { super(message); this.name = "CaptureQualityError"; this.code = code; }
}

const MAX_SOURCE_SIDE = 1280;
const MAX_DET_SIDE = 960;
const DET_THRESHOLD = .3;
const BOX_THRESHOLD = .55;
const waitForOpenCv = () => new Promise((resolve) => { const check = () => globalThis.cv?.Mat ? resolve() : setTimeout(check, 50); check(); });
const orderQuad = (points) => {
  const sorted = [...points].sort((a, b) => a[1] - b[1]);
  const [tl, tr] = sorted.slice(0, 2).sort((a, b) => a[0] - b[0]);
  const [bl, br] = sorted.slice(2).sort((a, b) => a[0] - b[0]);
  return [tl, tr, br, bl];
};

export class OnDeviceLabelExtractor {
  /**
   * @param {{detModelUrl: string, recModelUrl: string, dictionaryUrl: string, fallbackUrl?: string, confidenceThreshold?: number, timeoutMs?: number}} config
   */
  constructor(config) {
    this.config = { confidenceThreshold: .5, timeoutMs: 7000, ...config };
    this.detSession = null; this.recSession = null; this.dictionary = [];
  }

  async load() {
    await waitForOpenCv();
    if (!globalThis.ort) throw new Error("Load onnxruntime-web before OnDeviceLabelExtractor.");
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    [this.dictionary, this.detSession, this.recSession] = await Promise.all([
      fetch(this.config.dictionaryUrl).then((r) => r.ok ? r.text() : Promise.reject(new Error("OCR character dictionary unavailable."))).then((t) => t.trim().split(/\r?\n/)),
      ort.InferenceSession.create(this.config.detModelUrl, { executionProviders: ["wasm"] }),
      ort.InferenceSession.create(this.config.recModelUrl, { executionProviders: ["wasm"] }),
    ]);
  }

  /** Capture three quick frames and retain the sharpest (variance of Laplacian). */
  async captureBestFrame(video, count = 3, intervalMs = 180) {
    if (!video.videoWidth) throw new CaptureQualityError("CAMERA_NOT_READY", "Camera has not produced a frame yet.");
    let best = null;
    for (let index = 0; index < count; index += 1) {
      const canvas = document.createElement("canvas"); const scale = Math.min(1, MAX_SOURCE_SIDE / Math.max(video.videoWidth, video.videoHeight)); canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      const sharpness = this.#sharpness(canvas);
      if (!best || sharpness > best.sharpness) best = { canvas, sharpness };
      if (index < count - 1) await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
    return best.canvas;
  }

  async extractFromFile(file, options = {}) { return this.#extract(await this.#canvasFromFile(file), file, options); }
  async extractFromCanvas(canvas, options = {}) { return this.#extract(this.#scaleCanvas(canvas), null, options); }

  async #extract(canvas, fallbackFile, { imageId = `img_${Date.now()}`, productCategory = "other", allowFallback = true, onFallback } = {}) {
    if (!this.detSession) throw new Error("Call extractor.load() before extract.");
    if (this.#hasGlare(canvas)) throw new CaptureQualityError("GLARE_DETECTED", "Glare covers part of the label. Tilt the package slightly and scan again.");
    const lines = await this.#ocr(this.#clahe(canvas));
    const extractedFields = parseDeclarations(lines, { minConfidence: this.config.confidenceThreshold });
    const confidenceValues = Object.values(extractedFields).filter((field) => field.present).map((field) => field.confidence);
    const overallConfidence = confidenceValues.length ? confidenceValues.reduce((a, b) => a + b, 0) / confidenceValues.length : 0;
    let result = { image_id: imageId, product_category: productCategory, extracted_fields: extractedFields, raw_ocr_text: lines.map((line) => line.text).join("\n") };
    if (allowFallback && overallConfidence < this.config.confidenceThreshold && fallbackFile && navigator.onLine && this.config.fallbackUrl) {
      onFallback?.({ state: "started", overallConfidence });
      try { result = await this.#serverFallback(fallbackFile, imageId, productCategory); onFallback?.({ state: "succeeded" }); }
      catch { onFallback?.({ state: "failed" }); }
    }
    return result; // Fixed vision JSON contract only: no additive keys until Group 2 approves them.
  }

  async #serverFallback(file, imageId, productCategory) {
    const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), this.config.timeoutMs);
    try {
      const form = new FormData(); form.append("file", file); form.append("image_id", imageId); form.append("product_category", productCategory);
      const response = await fetch(this.config.fallbackUrl, { method: "POST", body: form, signal: controller.signal });
      if (!response.ok) throw new Error(`Fallback failed (${response.status})`);
      return await response.json();
    } finally { clearTimeout(timer); }
  }

  async #canvasFromFile(file) { const bitmap = await createImageBitmap(file); const canvas = document.createElement("canvas"); const scale = Math.min(1, MAX_SOURCE_SIDE / Math.max(bitmap.width, bitmap.height)); canvas.width = Math.round(bitmap.width * scale); canvas.height = Math.round(bitmap.height * scale); canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height); bitmap.close(); return canvas; }
  #scaleCanvas(canvas) { const scale = Math.min(1, MAX_SOURCE_SIDE / Math.max(canvas.width, canvas.height)); if (scale === 1) return canvas; const out = document.createElement("canvas"); out.width = Math.round(canvas.width * scale); out.height = Math.round(canvas.height * scale); out.getContext("2d").drawImage(canvas, 0, 0, out.width, out.height); return out; }
  #sharpness(canvas) { const src = cv.imread(canvas), gray = new cv.Mat(), lap = new cv.Mat(), mean = new cv.Mat(), deviation = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY); cv.Laplacian(gray, lap, cv.CV_64F); cv.meanStdDev(lap, mean, deviation); const value = deviation.doubleAt(0, 0) ** 2; src.delete(); gray.delete(); lap.delete(); mean.delete(); deviation.delete(); return value; }
  #hasGlare(canvas) { const src = cv.imread(canvas), hsv = new cv.Mat(), mask = new cv.Mat(); cv.cvtColor(src, hsv, cv.COLOR_RGBA2HSV); const lower = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 0, 245, 0]), upper = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [180, 80, 255, 255]); cv.inRange(hsv, lower, upper, mask); const ratio = cv.countNonZero(mask) / (canvas.width * canvas.height); src.delete(); hsv.delete(); mask.delete(); lower.delete(); upper.delete(); return ratio > .025; }
  #clahe(canvas) { const src = cv.imread(canvas), lab = new cv.Mat(), channels = new cv.MatVector(), out = new cv.Mat(); cv.cvtColor(src, lab, cv.COLOR_RGBA2Lab); cv.split(lab, channels); const luminance = channels.get(0), clahe = cv.createCLAHE(2, new cv.Size(8, 8)); clahe.apply(luminance, luminance); luminance.delete(); cv.merge(channels, lab); cv.cvtColor(lab, out, cv.COLOR_Lab2RGBA); const result = document.createElement("canvas"); cv.imshow(result, out); src.delete(); lab.delete(); channels.delete(); out.delete(); clahe.delete(); return result; }
  #tensor(canvas, width, height, mean, scale) { const target = document.createElement("canvas"); target.width = width; target.height = height; const context = target.getContext("2d", { willReadFrequently: true }); context.drawImage(canvas, 0, 0, width, height); const rgba = context.getImageData(0, 0, width, height).data; const data = new Float32Array(3 * width * height); for (let i = 0; i < width * height; i += 1) for (let c = 0; c < 3; c += 1) data[c * width * height + i] = (rgba[i * 4 + c] / 255 - mean[c]) / scale[c]; return new ort.Tensor("float32", data, [1, 3, height, width]); }
  #boxes(probability, mapWidth, mapHeight, sourceWidth, sourceHeight) { const score = new cv.Mat(mapHeight, mapWidth, cv.CV_32FC1); score.data32F.set(probability); const binary = new cv.Mat(); cv.threshold(score, binary, DET_THRESHOLD, 255, cv.THRESH_BINARY); binary.convertTo(binary, cv.CV_8UC1); const contours = new cv.MatVector(), hierarchy = new cv.Mat(); cv.findContours(binary, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE); const boxes = []; for (let i = 0; i < contours.size(); i += 1) { const contour = contours.get(i), rect = cv.minAreaRect(contour); const points = cv.RotatedRect.points(rect); contour.delete(); if (rect.size.width * rect.size.height < 20) continue; const quad = orderQuad(points.map((p) => [p.x, p.y])); const mask = cv.Mat.zeros(mapHeight, mapWidth, cv.CV_8UC1), polygon = cv.matFromArray(4, 1, cv.CV_32SC2, quad.flatMap(([x, y]) => [Math.round(x), Math.round(y)])), polygons = new cv.MatVector(); polygons.push_back(polygon); cv.fillPoly(mask, polygons, new cv.Scalar(255)); const averageScore = cv.mean(score, mask)[0]; mask.delete(); polygon.delete(); polygons.delete(); if (averageScore < BOX_THRESHOLD) continue; boxes.push(quad.map(([x, y]) => [x * sourceWidth / mapWidth, y * sourceHeight / mapHeight])); } score.delete(); binary.delete(); contours.delete(); hierarchy.delete(); return boxes.sort((a, b) => a[0][1] - b[0][1]).slice(0, 40); }
  #crop(canvas, quad) { const [tl, tr, br, bl] = quad; const width = Math.max(8, Math.round(Math.max(Math.hypot(tr[0]-tl[0], tr[1]-tl[1]), Math.hypot(br[0]-bl[0], br[1]-bl[1])))); const height = Math.max(8, Math.round(Math.max(Math.hypot(bl[0]-tl[0], bl[1]-tl[1]), Math.hypot(br[0]-tr[0], br[1]-tr[1])))); const src = cv.imread(canvas), dst = new cv.Mat(), from = cv.matFromArray(4, 1, cv.CV_32FC2, quad.flat()), to = cv.matFromArray(4, 1, cv.CV_32FC2, [0,0,width,0,width,height,0,height]), transform = cv.getPerspectiveTransform(from, to); cv.warpPerspective(src, dst, transform, new cv.Size(width, height), cv.INTER_CUBIC, cv.BORDER_REPLICATE); const result = document.createElement("canvas"); cv.imshow(result, dst); src.delete(); dst.delete(); from.delete(); to.delete(); transform.delete(); return result; }
  async #ocr(canvas) { const scale = Math.min(1, MAX_DET_SIDE / Math.max(canvas.width, canvas.height)); const width = Math.max(32, Math.round(canvas.width * scale / 32) * 32), height = Math.max(32, Math.round(canvas.height * scale / 32) * 32); const det = await this.detSession.run({ [this.detSession.inputNames[0]]: this.#tensor(canvas, width, height, [.485,.456,.406], [.229,.224,.225]) }); const output = det[this.detSession.outputNames[0]], boxes = this.#boxes(output.data, output.dims.at(-1), output.dims.at(-2), canvas.width, canvas.height); const lines = []; for (const box of boxes) { const crop = this.#crop(canvas, box), ratio = 48 / crop.height, cropWidth = Math.min(320, Math.max(16, Math.ceil(crop.width * ratio))); const rec = await this.recSession.run({ [this.recSession.inputNames[0]]: this.#tensor(crop, cropWidth, 48, [.5,.5,.5], [.5,.5,.5]) }); const tensor = rec[this.recSession.outputNames[0]], classes = tensor.dims.at(-1), steps = tensor.dims.at(-2); let text = "", total = 0, count = 0, previous = -1; for (let t = 0; t < steps; t += 1) { let best = 0, score = -Infinity; for (let c = 0; c < classes; c += 1) if (tensor.data[t * classes + c] > score) { score = tensor.data[t * classes + c]; best = c; } if (best && best !== previous) { text += this.dictionary[best - 1] || ""; total += score; count += 1; } previous = best; } if (text) lines.push({ text, confidence: count ? total / count : 0 }); } return lines; }
}
