// Simple YOLOX demo for running character.onnx in the browser
// - Input: image (H x W x 3, uint8)
// - Inference: ONNX Runtime Web (WebGPU / WASM)
// - Output: bbox + confidence on image (assumes single class "character")

const ORT_SCRIPT_CANDIDATES = [
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.webgpu.min.js",
  "https://unpkg.com/onnxruntime-web@1.23.2/dist/ort.webgpu.min.js",
  "./vendor/ort.webgpu.min.js", // Optional: use when placed locally
];

let ortApi = null;
let session = null;
let inputName = null;
let outputName = null;

const CLASS_NAMES = ["character"]; // Assumes single class
const INPUT_HEIGHT = 640;
const INPUT_WIDTH = 640;
const HEIF_EXTS = [".heic", ".heif", ".hif"];
const LIBHEIF_SCRIPT_CANDIDATES = [
  "https://unpkg.com/@petamoriken/libheif-js@1.19.5/libheif-bundle.js",
  "https://cdn.jsdelivr.net/npm/libheif-js@1.18.2/libheif-wasm/libheif-bundle.js",
  "./vendor/libheif-bundle.js",
];

let libHeifReadyPromise = null;

// Check URL parameter first (?lang=en or ?lang=ja), then fall back to browser language
// For browser language, only 'ja' displays Japanese, all others display English
function detectLocale() {
  const urlParams = new URLSearchParams(window.location.search);
  const langParam = urlParams.get('lang');
  if (langParam !== null) {
    // lang parameter exists: return 'ja' only if explicitly 'ja', otherwise 'en'
    return langParam === 'ja' ? 'ja' : 'en';
  }
  // No lang parameter, use browser language: only 'ja' displays Japanese, all others display English
  const browserLang = typeof navigator !== "undefined" ? (navigator.language || "") : "";
  return /^ja\b/i.test(browserLang) ? "ja" : "en";
}

const currentLocale = detectLocale();
const MESSAGES = {
  heic2anyStatus: {
    ja: "HEIC/HEIF を PNG に変換中... (heic2any)",
    en: "Converting HEIC/HEIF to PNG... (heic2any)",
  },
  heicLibStatus: {
    ja: "HEIC/HEIF を PNG に変換中... (libheif-js)",
    en: "Converting HEIC/HEIF to PNG... (libheif-js)",
  },
  heicConvertFailed: {
    ja: "HEIC/HEIF を変換できませんでした: {reason}",
    en: "Failed to convert HEIC/HEIF: {reason}",
  },
  libheifNotReady: {
    ja: "libheif-js が読み込まれていません。CDN ブロック時は web/vendor/ に libheif-bundle.js を配置してください。",
    en: "libheif-js is not loaded. If CDN access is blocked, place libheif-bundle.js under web/vendor/.",
  },
  libDecodeFailed: {
    ja: "libheif-js で画像をデコードできませんでした",
    en: "Failed to decode image with libheif-js",
  },
  libDisplayEmpty: {
    ja: "libheif-js: display() の結果が空です",
    en: "libheif-js: display() returned empty data",
  },
  libCanvasError: {
    ja: "libheif-js: Canvas 2D コンテキストを取得できません",
    en: "libheif-js: unable to get Canvas 2D context",
  },
  libPngFailed: {
    ja: "libheif-js: PNG変換に失敗しました",
    en: "libheif-js: failed to convert to PNG",
  },
  ortLoading: {
    ja: "ONNX Runtime Web をロードしています...",
    en: "Loading ONNX Runtime Web...",
  },
  ortLoadFailed: {
    ja: "ONNX Runtime Web の読み込みに失敗しました。ネットワークまたは web/vendor/ort-webgpu.min.js の配置を確認してください。試行: {attempts}",
    en: "Failed to load ONNX Runtime Web. Check your network or the web/vendor/ort-webgpu.min.js fallback. Attempts: {attempts}",
  },
  modelLoadFailed: {
    ja: "モデル読み込みに失敗しました: {reason}",
    en: "Failed to load model: {reason}",
  },
  modelLoadSuccess: {
    ja: "モデル読み込み完了 ({providers}){note}",
    en: "Model loaded ({providers}){note}",
  },
  webgpuNote: {
    ja: " (WebGPU を使うには対応ブラウザで --enable-unsafe-webgpu 等のフラグが必要です)",
    en: " (Enable WebGPU in your browser, e.g., via --enable-unsafe-webgpu.)",
  },
  modelNotReady: {
    ja: "モデルの初期化が完了していません。",
    en: "Model is not initialized yet.",
  },
  imageLoading: {
    ja: "画像読み込み中...",
    en: "Loading image...",
  },
  invalidFile: {
    ja: "画像ファイルを選択してください。",
    en: "Please select an image file.",
  },
  inferenceError: {
    ja: "画像の読み込みまたは推論に失敗しました: {reason}",
    en: "Failed to load the image or run inference: {reason}",
  },
  canvasContextMissing: {
    ja: "Canvas 2D コンテキストを取得できません",
    en: "Failed to get Canvas 2D context",
  },
  unexpectedOutput: {
    ja: "想定外の出力形状: {shape}",
    en: "Unexpected output shape: {shape}",
  },
  ortNotInitialized: {
    ja: "ONNX Runtime が初期化されていません",
    en: "ONNX Runtime is not initialized",
  },
  noDetections: {
    ja: "検出結果はありません (スコア閾値を下げてみてください)。",
    en: "No detections (try lowering the score threshold).",
  },
  infoPlaceholder: {
    ja: "画像をドロップするとここに検出結果の一覧を表示します。",
    en: "Drop an image to display detection results here.",
  },
  runningInference: {
    ja: "推論中...",
    en: "Running inference...",
  },
  inferenceDone: {
    ja: "推論完了 (約 {ms} ms, 検出数: {count})",
    en: "Inference finished (~{ms} ms, detections: {count})",
  },
  pageTitle: {
    ja: "character.onnx Web デモ",
    en: "character.onnx Web Demo",
  },
  headerDescription: {
    ja: "画像をドラッグ&ドロップすると、ブラウザ内(WebGPU / WASM)で推論して bbox とスコアを表示します。",
    en: "Drag & drop an image to run inference in the browser (WebGPU / WASM) and display bboxes with confidence scores.",
  },
  loadingModel: {
    ja: "モデル読み込み中...",
    en: "Loading model...",
  },
  dropzoneLabel: {
    ja: "ここに画像ファイルをドラッグ&ドロップ<br>またはクリックしてファイルを選択<br>貼り付けも可能",
    en: "Drag & drop image here<br>or click to select file<br>or paste from clipboard",
  },
  scoreThresholdLabel: {
    ja: "スコア閾値",
    en: "Score Threshold",
  },
  nmsThresholdLabel: {
    ja: "NMS IoU閾値",
    en: "NMS IoU Threshold",
  },
  footerNote: {
    ja: "処理はすべてクライアント側で行われ、画像が外部にアップロードされることはありません。<br><a href=\"https://x.com/ksasao\">@ksasao</a> / <a href=\"https://github.com/ksasao/anime-character-detector\">GitHub</a>",
    en: "All processing runs client-side; your images are never uploaded.<br><a href=\"https://x.com/ksasao\">@ksasao</a> / <a href=\"https://github.com/ksasao/anime-character-detector\">GitHub</a>",
  },
};

function msg(key, vars = {}) {
  const entry = MESSAGES[key];
  const template = entry ? entry[currentLocale] ?? entry.en ?? "" : "";
  return template.replace(/\{(\w+)\}/g, (_, name) => (vars[name] ?? `{${name}}`));
}

function localizeHtml() {
  document.querySelectorAll("[data-msg]").forEach((el) => {
    const key = el.getAttribute("data-msg");
    if (key && MESSAGES[key]) {
      el.innerHTML = msg(key);
    }
  });
}

function getScoreThreshold() {
  const el = document.getElementById("score-thr");
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : 0.3;
}

function getNmsThreshold() {
  const el = document.getElementById("nms-thr");
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : 0.45;
}

function setStatus(text) {
  const el = document.getElementById("status");
  if (el) el.textContent = text;
}

function setInfo(text) {
  const el = document.getElementById("info");
  if (el) el.textContent = text;
}

function isHeicFile(file) {
  const name = (file && file.name ? file.name : "").toLowerCase();
  const extMatch = HEIF_EXTS.some((ext) => name.endsWith(ext));
  const typeMatch = typeof file?.type === "string" && file.type.toLowerCase().includes("heic");
  return extMatch || typeMatch;
}

async function normalizeImageFile(file) {
  if (!isHeicFile(file)) {
    return file;
  }
  const newName = file.name ? file.name.replace(/\.(heic|heif|hif)$/i, ".png") : "converted.png";
  const heicLib = globalThis.heic2any;
  if (heicLib) {
    try {
      setStatus(msg("heic2anyStatus"));
      const converted = await heicLib({
        blob: file,
        toType: "image/png",
        quality: 1,
      });
      const blob = Array.isArray(converted) ? converted[0] : converted;
      return new File([blob], newName, { type: "image/png" });
    } catch (err) {
      console.warn("heic2any conversion failed", err);
    }
  }
  try {
    setStatus(msg("heicLibStatus"));
    return await convertHeicWithLibHeif(file, newName);
  } catch (err) {
    console.error("libheif-js conversion failed", err);
    throw new Error(msg("heicConvertFailed", { reason: err.message }));
  }
}

async function resolveLibHeifCandidate(candidate) {
  if (!candidate) {
    return null;
  }
  if (typeof candidate === "function") {
    // Some bundles expose a factory function returning a promise/module
    return resolveLibHeifCandidate(candidate());
  }
  if (candidate && typeof candidate.then === "function") {
    const resolved = await candidate;
    return resolveLibHeifCandidate(resolved);
  }
  if (candidate && candidate.ready && typeof candidate.ready.then === "function") {
    const resolved = await candidate.ready;
    return resolveLibHeifCandidate(resolved);
  }
  if (candidate && typeof candidate.HeifDecoder === "function") {
    return candidate;
  }
  return null;
}

function ensureLibHeifReady() {
  if (libHeifReadyPromise) {
    return libHeifReadyPromise;
  }
  const gatherCandidates = () => [
    globalThis.libheif,
    globalThis.libheifModule,
    globalThis.LibHeifModule,
    globalThis.LibHeif,
    globalThis.libheif_default,
    globalThis.LibHeifModuleFactory,
  ];

  const hasCandidate = () => gatherCandidates().some((c) => Boolean(c));

  libHeifReadyPromise = (async () => {
    if (!hasCandidate()) {
      for (const url of LIBHEIF_SCRIPT_CANDIDATES) {
        try {
          await loadScriptOnce(url);
          if (hasCandidate()) {
            break;
          }
        } catch (err) {
          console.warn("Failed to load libheif script", url, err);
        }
      }
    }

    const candidates = gatherCandidates();
    for (const candidate of candidates) {
      try {
        const module = await resolveLibHeifCandidate(candidate);
        if (module) {
          return module;
        }
      } catch (err) {
        console.warn("libheif candidate failed", err);
      }
    }
    throw new Error(msg("libheifNotReady"));
  })();

  return libHeifReadyPromise;
}

async function convertHeicWithLibHeif(file, newName) {
  const lib = await ensureLibHeifReady();
  const buffer = await file.arrayBuffer();
  const decoder = new lib.HeifDecoder();
  const images = decoder.decode(new Uint8Array(buffer));
  if (!images || images.length === 0) {
    throw new Error(msg("libDecodeFailed"));
  }
  const image = images[0];
  const width = image.get_width();
  const height = image.get_height();
  const rgba = new Uint8ClampedArray(width * height * 4);
  const target = { data: rgba, width, height, stride: width * 4 };

  await new Promise((resolve, reject) => {
    try {
      image.display(target, (filled) => {
        if (!filled || !filled.data) {
          reject(new Error(msg("libDisplayEmpty")));
          return;
        }
        resolve(filled);
      });
    } catch (err) {
      reject(err);
    }
  });

  const canvas = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(width, height)
    : (() => {
        const el = document.createElement("canvas");
        el.width = width;
        el.height = height;
        return el;
      })();
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error(msg("libCanvasError"));
  }
  const imageData = new ImageData(rgba, width, height);
  ctx.putImageData(imageData, 0, 0);

  const useOffscreen = typeof OffscreenCanvas !== "undefined" && canvas instanceof OffscreenCanvas;
  const blob = useOffscreen
    ? await canvas.convertToBlob({ type: "image/png" })
    : await new Promise((resolve, reject) =>
        canvas.toBlob(
          (b) => (b ? resolve(b) : reject(new Error(msg("libPngFailed")))),
          "image/png",
          1
        )
      );

  if (typeof image.free === "function") {
    image.free();
  }
  images.slice(1).forEach((img) => {
    if (img && typeof img.free === "function") {
      img.free();
    }
  });

  return new File([blob], newName, { type: "image/png" });
}

function loadScriptOnce(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.async = false;
    script.onload = () => resolve(url);
    script.onerror = () => reject(new Error("Failed to load " + url));
    document.head.appendChild(script);
  });
}

async function ensureOrtLoaded() {
  if (globalThis.ort) {
    return globalThis.ort;
  }
  setStatus(msg("ortLoading"));
  const tried = [];
  for (const url of ORT_SCRIPT_CANDIDATES) {
    tried.push(url);
    try {
      await loadScriptOnce(url);
      if (globalThis.ort) {
        return globalThis.ort;
      }
    } catch (err) {
      console.warn(err);
    }
  }
  throw new Error(msg("ortLoadFailed", { attempts: tried.join(", ") }));
}

async function initOrt() {
  try {
    ortApi = await ensureOrtLoaded();
  } catch (err) {
    console.error(err);
    setStatus(err.message);
    return;
  }

  const providerPlans = [
    [
      {
        name: 'webgpu',
        preferredLayout: 'NHWC',
        enableGraphCapture: true,
      },
      'wasm'
    ],
    ['wasm'],
  ];
  let lastError = null;
  let usedProviders = null;

  for (const providers of providerPlans) {
    try {
      session = await ortApi.InferenceSession.create("character.onnx", {
        executionProviders: providers,
      });
      usedProviders = Array.isArray(providers) 
        ? providers.map(p => typeof p === 'object' ? p.name : p)
        : [providers];
      break;
    } catch (err) {
      lastError = err;
      console.warn("Failed to init providers", providers, err);
    }
  }

  if (!session) {
    console.error(lastError);
    setStatus(msg("modelLoadFailed", { reason: lastError ? lastError.message : "unknown" }));
    return;
  }

  inputName = session.inputNames[0];
  outputName = session.outputNames[0];
  const providerLabel = usedProviders.join(", ");
  const webgpuNote = usedProviders[0] === "webgpu" ? "" : msg("webgpuNote");
  setStatus(msg("modelLoadSuccess", { providers: providerLabel, note: webgpuNote }));
}

function setupDragAndDrop() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");

  if (!dropZone || !fileInput) return;

  const prevent = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, prevent, false);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add("dragover"), false);
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove("dragover"), false);
  });

  dropZone.addEventListener(
    "drop",
    (e) => {
      const dt = e.dataTransfer;
      const files = dt?.files;
      if (files && files.length > 0) {
        handleFile(files[0]);
      }
    },
    false
  );

  dropZone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  // Setup paste event listener
  document.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.startsWith("image/")) {
        e.preventDefault();
        const blob = item.getAsFile();
        if (blob) {
          handleFile(blob);
        }
        break;
      }
    }
  });
}

function createImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(err);
    };
    img.src = url;
  });
}

async function handleFile(file) {
  if (!session) {
    setStatus(msg("modelNotReady"));
    return;
  }
  const mime = typeof file.type === "string" ? file.type : "";
  if (!mime.startsWith("image/") && !isHeicFile(file)) {
    setStatus(msg("invalidFile"));
    return;
  }

  try {
    setStatus(msg("imageLoading"));
    const preparedFile = await normalizeImageFile(file);
    const img = await createImageFromFile(preparedFile);
    await runInference(img);
  } catch (err) {
    console.error(err);
    setStatus(msg("inferenceError", { reason: err.message }));
  }
}

function prepareInput(imgEl) {
  const offCanvas = document.createElement("canvas");
  offCanvas.width = INPUT_WIDTH;
  offCanvas.height = INPUT_HEIGHT;
  const ctx = offCanvas.getContext("2d");
  if (!ctx) throw new Error(msg("canvasContextMissing"));

  // Calculate ratio and apply letterbox to fit the longer side
  const ratio = Math.min(INPUT_HEIGHT / imgEl.naturalHeight, INPUT_WIDTH / imgEl.naturalWidth);
  const newW = Math.round(imgEl.naturalWidth * ratio);
  const newH = Math.round(imgEl.naturalHeight * ratio);

  ctx.fillStyle = "rgb(114,114,114)"; // YOLOX default padding color
  ctx.fillRect(0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  
  // Use high-quality interpolation for better small object detection
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(imgEl, 0, 0, newW, newH);

  const imageData = ctx.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  const { data } = imageData; // RGBA format

  const size = INPUT_WIDTH * INPUT_HEIGHT;
  const inputData = new Float32Array(3 * size);

  // Optimized single loop with precalculated offsets
  const size2 = size * 2;
  for (let i = 0, len = size; i < len; i++) {
    const idx4 = i * 4;
    inputData[i] = data[idx4];
    inputData[size + i] = data[idx4 + 1];
    inputData[size2 + i] = data[idx4 + 2];
  }

  if (!ortApi) {
    throw new Error(msg("ortNotInitialized"));
  }
  const tensor = new ortApi.Tensor("float32", inputData, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);
  return { tensor, ratio, letterboxCanvas: offCanvas };
}

function demoPostprocessInPlace(preds, numAnchors, numVals, inputShape, p6 = false) {
  const [h, w] = inputShape;
  const strides = p6 ? [8, 16, 32, 64] : [8, 16, 32];
  const hsizes = strides.map((s) => Math.floor(h / s));
  const wsizes = strides.map((s) => Math.floor(w / s));

  let anchorIdx = 0;
  for (let sIdx = 0; sIdx < strides.length; sIdx++) {
    const stride = strides[sIdx];
    const hsize = hsizes[sIdx];
    const wsize = wsizes[sIdx];
    for (let y = 0; y < hsize; y++) {
      for (let x = 0; x < wsize; x++) {
        if (anchorIdx >= numAnchors) return;
        const base = anchorIdx * numVals;
        const gx = x;
        const gy = y;
        preds[base + 0] = (preds[base + 0] + gx) * stride;
        preds[base + 1] = (preds[base + 1] + gy) * stride;
        preds[base + 2] = Math.exp(preds[base + 2]) * stride;
        preds[base + 3] = Math.exp(preds[base + 3]) * stride;
        anchorIdx++;
      }
    }
  }
}

function nms(boxes, scores, iouThr) {
  // Create sorted indices array
  const indices = Array.from({ length: scores.length }, (_, i) => i);
  indices.sort((a, b) => scores[b] - scores[a]);

  const keep = [];
  const suppressed = new Set();

  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    if (suppressed.has(idx)) continue;
    
    keep.push(idx);

    // Suppress overlapping boxes
    for (let j = i + 1; j < indices.length; j++) {
      const jdx = indices[j];
      if (suppressed.has(jdx)) continue;
      
      const iou = computeIoU(boxes[idx], boxes[jdx]);
      if (iou > iouThr) {
        suppressed.add(jdx);
      }
    }
  }

  return keep;
}

function computeIoU(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  
  // Early exit if no overlap
  if (x2 <= x1 || y2 <= y1) return 0;
  
  const w = x2 - x1 + 1;
  const h = y2 - y1 + 1;
  const inter = w * h;
  const areaA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  const areaB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  const union = areaA + areaB - inter;
  return inter / union;
}

function decodeYolox(output, ratio, scoreThr, nmsThr) {
  const dims = output.dims;
  if (dims.length !== 3 || dims[0] !== 1) {
    throw new Error(msg("unexpectedOutput", { shape: JSON.stringify(dims) }));
  }
  const numAnchors = dims[1];
  const numVals = dims[2];
  // Use output.data directly if it's already a Float32Array, otherwise copy
  const preds = output.data instanceof Float32Array ? output.data : new Float32Array(output.data);

  demoPostprocessInPlace(preds, numAnchors, numVals, [INPUT_HEIGHT, INPUT_WIDTH], false);

  const numClasses = numVals - 5;

  const boxes = [];
  const boxesLetterbox = [];
  const scores = [];
  const clsIds = [];

  for (let i = 0; i < numAnchors; i++) {
    const base = i * numVals;
    const cx = preds[base + 0];
    const cy = preds[base + 1];
    const w = preds[base + 2];
    const h = preds[base + 3];
    const obj = preds[base + 4];

    let bestScore = 0;
    let bestCls = 0;
    for (let c = 0; c < numClasses; c++) {
      const clsProb = preds[base + 5 + c];
      const conf = obj * clsProb;
      if (conf > bestScore) {
        bestScore = conf;
        bestCls = c;
      }
    }

    if (bestScore < scoreThr) continue;

    const x0Letter = cx - w / 2;
    const y0Letter = cy - h / 2;
    const x1Letter = cx + w / 2;
    const y1Letter = cy + h / 2;

    boxesLetterbox.push([x0Letter, y0Letter, x1Letter, y1Letter]);

    const x0 = x0Letter / ratio;
    const y0 = y0Letter / ratio;
    const x1 = x1Letter / ratio;
    const y1 = y1Letter / ratio;

    boxes.push([x0, y0, x1, y1]);
    scores.push(bestScore);
    clsIds.push(bestCls);
  }

  if (boxes.length === 0) {
    return { boxes: [], boxesLetterbox: [], scores: [], clsIds: [] };
  }

  const keep = nms(boxes, scores, nmsThr);

  return {
    boxes: keep.map((idx) => boxes[idx]),
    boxesLetterbox: keep.map((idx) => boxesLetterbox[idx]),
    scores: keep.map((idx) => scores[idx]),
    clsIds: keep.map((idx) => clsIds[idx]),
  };
}

function drawDetections(letterboxCanvas, detections) {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  canvas.width = letterboxCanvas.width;
  canvas.height = letterboxCanvas.height;
  ctx.drawImage(letterboxCanvas, 0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 3;
  ctx.font = "14px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI'";

  const infoLines = [];

  const drawBoxes = detections.boxesLetterbox && detections.boxesLetterbox.length ? detections.boxesLetterbox : detections.boxes;

  drawBoxes.forEach((box, i) => {
    const score = detections.scores[i];
    const clsId = detections.clsIds[i];
    const label = CLASS_NAMES[clsId] || `cls_${clsId}`;
    const [x0, y0, x1, y1] = box;

    const color = "#f97316"; // Orange color
    ctx.strokeStyle = color;
    ctx.fillStyle = "rgba(15, 23, 42, 0.75)";

    ctx.beginPath();
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.stroke();

    const text = `${label}:${(score * 100).toFixed(1)}%`;
    const textMetrics = ctx.measureText(text);
    const textW = textMetrics.width + 10;
    const textH = 18;

    const tx = x0;
    const ty = y0 - textH < 0 ? y0 + textH : y0;

    ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
    ctx.fillRect(tx, ty - textH + 2, textW, textH);

    ctx.fillStyle = "#e5e7eb";
    ctx.fillText(text, tx + 5, ty - 4);

    infoLines.push(
      `${i + 1}. ${label}  score=${(score * 100).toFixed(1)}%  bbox=(${detections.boxes[i][0].toFixed(
        1
      )}, ${detections.boxes[i][1].toFixed(1)}, ${detections.boxes[i][2].toFixed(1)}, ${detections.boxes[i][3].toFixed(1)})`
    );
  });

  if (detections.boxes.length === 0) {
    setInfo(msg("noDetections"));
  } else {
    setInfo(infoLines.join("\n"));
  }
}

async function runInference(imgEl) {
  setStatus(msg("runningInference"));

  const { tensor, ratio, letterboxCanvas } = prepareInput(imgEl);
  const feeds = {};
  feeds[inputName] = tensor;

  const t0 = performance.now();
  const results = await session.run(feeds);
  const t1 = performance.now();

  const output = results[outputName];
  const scoreThr = getScoreThreshold();
  const nmsThr = getNmsThreshold();

  const dets = decodeYolox(output, ratio, scoreThr, nmsThr);
  drawDetections(letterboxCanvas, dets);

  const ms = (t1 - t0).toFixed(1);
  setStatus(msg("inferenceDone", { ms, count: dets.boxes.length }));
}

window.addEventListener("DOMContentLoaded", async () => {
  localizeHtml();
  setupDragAndDrop();
  await initOrt();
  setInfo(msg("infoPlaceholder"));
});
