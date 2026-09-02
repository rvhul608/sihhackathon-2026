const DET_MODEL = "./models/en_PP-OCRv3_det.onnx";
const REC_MODEL = "./models/en_PP-OCRv3_rec.onnx";
const DICT_URL = "./models/en_dict.txt";
const MAX_DET_SIDE = 960;
// A 12–50 MP camera image creates several large Canvas/OpenCV copies. OCR does
// not benefit from retaining that full resolution after 960 px DB detection.
const MAX_SOURCE_SIDE = 1280;
const DET_SCORE_THRESHOLD = 0.3;
const BOX_SCORE_THRESHOLD = 0.55;

const $ = (id) => document.getElementById(id);
const status = (message) => { $("status").textContent = message; };
const waitForOpenCv = () => new Promise((resolve) => {
  const ready = () => globalThis.cv?.Mat ? resolve() : setTimeout(ready, 50);
  ready();
});

let detSession;
let recSession;
let dictionary;
let modelsReady = false;

async function bootstrap() {
  status("Loading OpenCV…");
  await waitForOpenCv();
  status("Loading PP-OCR mobile ONNX models (first run can take a moment)…");
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
  ort.env.wasm.simd = true;
  [dictionary, detSession, recSession] = await Promise.all([
    fetch(DICT_URL).then((r) => r.ok ? r.text() : Promise.reject(new Error("Missing models/en_dict.txt. Run scripts/download-and-convert-models.sh first."))).then((t) => t.trim().split(/\r?\n/)),
    ort.InferenceSession.create(DET_MODEL, { executionProviders: ["wasm"] }),
    ort.InferenceSession.create(REC_MODEL, { executionProviders: ["wasm"] }),
  ]);
  modelsReady = true;
  $("run").disabled = !$("image").files[0];
  status("Ready. Choose a photo, then run the benchmark.");
}

function canvasFor(file) {
  return createImageBitmap(file).then((bitmap) => {
    const canvas = document.createElement("canvas");
    const ratio = Math.min(1, MAX_SOURCE_SIDE / Math.max(bitmap.width, bitmap.height));
    canvas.width = Math.round(bitmap.width * ratio);
    canvas.height = Math.round(bitmap.height * ratio);
    // Scale, do not crop: the canvas is deliberately smaller than the camera
    // bitmap to keep mobile memory bounded.
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    return canvas;
  });
}

function toTensor(canvas, width, height, mean, scale) {
  const resized = document.createElement("canvas");
  resized.width = width; resized.height = height;
  const context = resized.getContext("2d", { willReadFrequently: true });
  context.drawImage(canvas, 0, 0, width, height);
  const rgba = context.getImageData(0, 0, width, height).data;
  const data = new Float32Array(3 * width * height);
  for (let i = 0; i < width * height; i++) {
    for (let c = 0; c < 3; c++) data[c * width * height + i] = (rgba[i * 4 + c] / 255 - mean[c]) / scale[c];
  }
  return new ort.Tensor("float32", data, [1, 3, height, width]);
}

function detectionSize(width, height) {
  const ratio = Math.min(1, MAX_DET_SIDE / Math.max(width, height));
  return { width: Math.max(32, Math.round(width * ratio / 32) * 32), height: Math.max(32, Math.round(height * ratio / 32) * 32) };
}

function orderQuad(points) {
  const sorted = [...points].sort((a, b) => a[1] - b[1]);
  const [topA, topB] = sorted.slice(0, 2).sort((a, b) => a[0] - b[0]);
  const [bottomA, bottomB] = sorted.slice(2).sort((a, b) => a[0] - b[0]);
  return [topA, topB, bottomB, bottomA];
}

function boxesFromDb(probability, mapWidth, mapHeight, sourceWidth, sourceHeight) {
  // Do not use the Mat(..., scalar) overload here: model data is a score map,
  // not a four-value OpenCV Scalar.
  const score = new cv.Mat(mapHeight, mapWidth, cv.CV_32FC1);
  score.data32F.set(probability);
  const binary = new cv.Mat();
  cv.threshold(score, binary, DET_SCORE_THRESHOLD, 255, cv.THRESH_BINARY);
  binary.convertTo(binary, cv.CV_8UC1);
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(binary, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
  const boxes = [];
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    const rect = cv.minAreaRect(contour);
    const vertices = cv.RotatedRect.points(rect);
    const area = rect.size.width * rect.size.height;
    contour.delete();
    if (area < 20) continue;
    const quad = orderQuad(vertices.map((p) => [p.x, p.y]));
    const mask = cv.Mat.zeros(mapHeight, mapWidth, cv.CV_8UC1);
    const polygon = cv.matFromArray(4, 1, cv.CV_32SC2, quad.flatMap(([x, y]) => [Math.round(x), Math.round(y)]));
    const polygons = new cv.MatVector(); polygons.push_back(polygon);
    cv.fillPoly(mask, polygons, new cv.Scalar(255));
    const mean = cv.mean(score, mask)[0];
    mask.delete(); polygon.delete(); polygons.delete();
    if (mean < BOX_SCORE_THRESHOLD) continue;
    boxes.push(quad.map(([x, y]) => [x * sourceWidth / mapWidth, y * sourceHeight / mapHeight]));
  }
  score.delete(); binary.delete(); contours.delete(); hierarchy.delete();
  return boxes.sort((a, b) => (a[0][1] - b[0][1]) || (a[0][0] - b[0][0])).slice(0, 40);
}

function cropQuad(canvas, quad) {
  const [tl, tr, br, bl] = quad;
  const width = Math.max(8, Math.round(Math.max(Math.hypot(tr[0]-tl[0], tr[1]-tl[1]), Math.hypot(br[0]-bl[0], br[1]-bl[1]))));
  const height = Math.max(8, Math.round(Math.max(Math.hypot(bl[0]-tl[0], bl[1]-tl[1]), Math.hypot(br[0]-tr[0], br[1]-tr[1]))));
  const src = cv.imread(canvas); const dst = new cv.Mat();
  const from = cv.matFromArray(4, 1, cv.CV_32FC2, quad.flat());
  const to = cv.matFromArray(4, 1, cv.CV_32FC2, [0,0, width,0, width,height, 0,height]);
  const transform = cv.getPerspectiveTransform(from, to);
  cv.warpPerspective(src, dst, transform, new cv.Size(width, height), cv.INTER_CUBIC, cv.BORDER_REPLICATE);
  const target = document.createElement("canvas"); cv.imshow(target, dst);
  src.delete(); dst.delete(); from.delete(); to.delete(); transform.delete();
  return target;
}

function decodeRecognition(output) {
  // PP-OCRv3 recognition returns [batch, timeSteps, classes] in this ONNX
  // export. Reading from the tail also tolerates exports with an extra axis.
  const classes = output.dims[output.dims.length - 1];
  const timeSteps = output.dims[output.dims.length - 2];
  let text = ""; let confidenceSum = 0; let count = 0; let previous = -1;
  for (let t = 0; t < timeSteps; t++) {
    let best = 0; let probability = -Infinity;
    for (let c = 0; c < classes; c++) {
      const value = output.data[t * classes + c];
      if (value > probability) { probability = value; best = c; }
    }
    if (best !== 0 && best !== previous) { text += dictionary[best - 1] || ""; confidenceSum += probability; count++; }
    previous = best;
  }
  return { text, confidence: count ? confidenceSum / count : 0 };
}

async function recognize(crop) {
  const ratio = 48 / crop.height;
  const width = Math.min(320, Math.max(16, Math.ceil(crop.width * ratio)));
  const tensor = toTensor(crop, width, 48, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
  const result = await recSession.run({ [recSession.inputNames[0]]: tensor });
  return decodeRecognition(result[recSession.outputNames[0]]);
}

async function run(file) {
  const image = await canvasFor(file);
  $("preview").src = image.toDataURL("image/jpeg", .85);
  const started = performance.now();
  const size = detectionSize(image.width, image.height);
  const tensor = toTensor(image, size.width, size.height, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
  const detectionStarted = performance.now();
  const detResult = await detSession.run({ [detSession.inputNames[0]]: tensor });
  const detectionMs = performance.now() - detectionStarted;
  const output = detResult[detSession.outputNames[0]];
  const boxes = boxesFromDb(output.data, output.dims[3], output.dims[2], image.width, image.height);
  const recognitionStarted = performance.now();
  const lines = [];
  for (const box of boxes) {
    const result = await recognize(cropQuad(image, box));
    if (result.text) lines.push(result);
  }
  const recognitionMs = performance.now() - recognitionStarted;
  const totalMs = performance.now() - started;
  status(`BENCHMARK RESULT\nTotal: ${totalMs.toFixed(0)} ms\nDetection: ${detectionMs.toFixed(0)} ms\nRecognition: ${recognitionMs.toFixed(0)} ms\nText regions: ${boxes.length}\n\nOCR text:\n${lines.map((line) => `${line.text}  (${line.confidence.toFixed(2)})`).join("\n") || "No text regions found"}`);
}

$("image").addEventListener("change", () => { $("run").disabled = !(modelsReady && $("image").files[0]); });
$("run").addEventListener("click", async () => {
  try { $("run").disabled = true; status("Running detection and recognition…"); await run($("image").files[0]); }
  catch (error) { console.error(error); status(`Benchmark failed: ${error.message}`); }
  finally { $("run").disabled = false; }
});
bootstrap().catch((error) => { console.error(error); status(`Model setup failed: ${error.message}`); });
