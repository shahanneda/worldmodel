const ORT_VERSION = "1.24.3";
const DEFAULT_MANIFEST_URL = "./model/manifest.json";
const STORAGE_KEY = "worldmodel.webgpu.manifestUrl";
const CDN_WASM_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const manifestUrlInput = document.getElementById("manifestUrlInput");
const loadManifestButton = document.getElementById("loadManifestButton");
const rerunButton = document.getElementById("rerunButton");
const inputStage = document.getElementById("inputStage");
const crosshair = document.getElementById("crosshair");
const outputCanvas = document.getElementById("outputCanvas");
const outputPlaceholder = document.getElementById("outputPlaceholder");
const downloadProgress = document.getElementById("downloadProgress");
const secureContextNotice = document.getElementById("secureContextNotice");
const statusLine = document.getElementById("statusLine");
const sessionStateValue = document.getElementById("sessionStateValue");
const providerValue = document.getElementById("providerValue");
const precisionValue = document.getElementById("precisionValue");
const imageSizeValue = document.getElementById("imageSizeValue");
const modelSizeValue = document.getElementById("modelSizeValue");
const latencyValue = document.getElementById("latencyValue");
const xValue = document.getElementById("xValue");
const yValue = document.getElementById("yValue");

const outputContext = outputCanvas.getContext("2d");

const state = {
  session: null,
  provider: null,
  manifest: null,
  currentPoint: { xNorm: 0.5, yNorm: 0.5 },
  inFlight: false,
  pendingPoint: null,
};

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.style.color = isError ? "#9f3518" : "";
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 100 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function basenameFromUrl(url) {
  return new URL(url, window.location.href).pathname.split("/").pop();
}

function resolveUrl(baseUrl, maybeRelativeUrl) {
  return new URL(maybeRelativeUrl, baseUrl).toString();
}

function updateCrosshair(xNorm, yNorm) {
  crosshair.style.left = `${xNorm * 100}%`;
  crosshair.style.top = `${yNorm * 100}%`;
  xValue.textContent = xNorm.toFixed(3);
  yValue.textContent = yNorm.toFixed(3);
}

function updateRuntimeSummary() {
  const manifest = state.manifest;
  providerValue.textContent = state.provider || "-";
  precisionValue.textContent = manifest?.precision || "-";
  imageSizeValue.textContent = manifest ? `${manifest.imageSize} x ${manifest.imageSize}` : "-";
  modelSizeValue.textContent = manifest ? formatBytes(manifest.totalModelBytes) : "-";
}

function setSessionState(message) {
  sessionStateValue.textContent = message;
}

function normalizeManifest(rawManifest, manifestUrl) {
  if (!rawManifest || typeof rawManifest !== "object") {
    throw new Error("Manifest JSON must be an object.");
  }

  const graphUrl = rawManifest.graphUrl || rawManifest.modelUrl;
  if (typeof graphUrl !== "string" || graphUrl.length === 0) {
    throw new Error("Manifest is missing graphUrl.");
  }

  const externalData = Array.isArray(rawManifest.externalData)
    ? rawManifest.externalData.map((entry) => {
        if (typeof entry === "string") {
          return {
            path: basenameFromUrl(entry),
            url: resolveUrl(manifestUrl, entry),
            bytes: null,
          };
        }

        if (!entry || typeof entry !== "object" || typeof entry.url !== "string") {
          throw new Error("Each externalData item must be a URL string or { path, url } object.");
        }

        return {
          path: entry.path || basenameFromUrl(entry.url),
          url: resolveUrl(manifestUrl, entry.url),
          bytes: Number.isFinite(entry.bytes) ? entry.bytes : null,
        };
      })
    : [];

  const imageSize =
    rawManifest.imageSize ??
    rawManifest.output?.shape?.[2] ??
    rawManifest.outputShape?.[2] ??
    128;

  return {
    modelName: rawManifest.modelName || "CoordinateToImageUNet",
    graphUrl: resolveUrl(manifestUrl, graphUrl),
    externalData,
    inputName: rawManifest.input?.name || rawManifest.inputName || "coords",
    outputName: rawManifest.output?.name || rawManifest.outputName || "image",
    imageSize: Number(imageSize),
    precision: rawManifest.precision || "fp32",
    totalModelBytes: Number(rawManifest.totalModelBytes) || null,
    preferredExecutionProviders: Array.isArray(rawManifest.preferredExecutionProviders)
      ? rawManifest.preferredExecutionProviders
      : ["webgpu", "wasm"],
  };
}

async function fetchBinary(url, label, expectedBytes = null) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${label}: ${response.status} ${response.statusText}`);
  }

  const contentLengthHeader = response.headers.get("content-length");
  const totalBytes =
    expectedBytes ||
    (contentLengthHeader ? Number.parseInt(contentLengthHeader, 10) : Number.NaN);

  if (!response.body || !Number.isFinite(totalBytes) || totalBytes <= 0) {
    setStatus(`Downloading ${label}...`);
    return new Uint8Array(await response.arrayBuffer());
  }

  const reader = response.body.getReader();
  const merged = new Uint8Array(totalBytes);
  let loadedBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    merged.set(value, loadedBytes);
    loadedBytes += value.byteLength;
    downloadProgress.value = Math.min(1, loadedBytes / totalBytes);
    setStatus(`Downloading ${label}... ${formatBytes(loadedBytes)} / ${formatBytes(totalBytes)}`);
  }

  if (loadedBytes !== totalBytes) {
    return merged.slice(0, loadedBytes);
  }
  return merged;
}

function orderedProviders(preferredProviders) {
  const requested = Array.isArray(preferredProviders) ? [...preferredProviders] : ["webgpu", "wasm"];
  const unique = [];
  for (const provider of requested) {
    if (!unique.includes(provider)) unique.push(provider);
  }
  if (!unique.includes("wasm")) unique.push("wasm");
  return unique.filter((provider) => provider !== "webgpu" || Boolean(navigator.gpu));
}

async function createSession(manifest) {
  if (!window.ort) {
    throw new Error("onnxruntime-web failed to load.");
  }

  ort.env.wasm.wasmPaths = CDN_WASM_BASE;
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
  ort.env.logLevel = "warning";

  downloadProgress.value = 0;
  const graphBuffer = await fetchBinary(manifest.graphUrl, "ONNX graph");

  const externalData = [];
  for (const entry of manifest.externalData) {
    downloadProgress.value = 0;
    const data = await fetchBinary(entry.url, entry.path, entry.bytes);
    externalData.push({ path: entry.path, data });
  }

  const providers = orderedProviders(manifest.preferredExecutionProviders);
  let lastError = null;

  for (const provider of providers) {
    try {
      setStatus(`Creating ${provider} session...`);
      const session = await ort.InferenceSession.create(graphBuffer, {
        executionProviders: [provider],
        externalData,
        graphOptimizationLevel: "all",
      });
      return { session, provider };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error("No execution provider could create the session.");
}

function renderTensor(tensor) {
  const dims = tensor.dims.length === 4 ? tensor.dims.slice(1) : tensor.dims;
  if (dims.length !== 3 || dims[0] !== 3) {
    throw new Error(`Expected output tensor shaped [1, 3, H, W] but received ${tensor.dims}.`);
  }

  const [, height, width] = dims;
  const channelSize = height * width;
  const pixels = new Uint8ClampedArray(width * height * 4);
  const values = tensor.data;

  for (let index = 0; index < channelSize; index += 1) {
    const pixelOffset = index * 4;
    pixels[pixelOffset] = Math.round(Math.min(1, Math.max(0, values[index])) * 255);
    pixels[pixelOffset + 1] = Math.round(
      Math.min(1, Math.max(0, values[channelSize + index])) * 255,
    );
    pixels[pixelOffset + 2] = Math.round(
      Math.min(1, Math.max(0, values[channelSize * 2 + index])) * 255,
    );
    pixels[pixelOffset + 3] = 255;
  }

  outputCanvas.width = width;
  outputCanvas.height = height;
  outputContext.putImageData(new ImageData(pixels, width, height), 0, 0);
  outputPlaceholder.style.display = "none";
}

async function runInference(point = state.currentPoint) {
  if (!state.session || !state.manifest) {
    return;
  }

  if (state.inFlight) {
    state.pendingPoint = point;
    return;
  }

  state.inFlight = true;
  rerunButton.disabled = true;
  latencyValue.textContent = "-";

  try {
    setStatus(`Running ${state.provider} inference...`);
    const tensor = new ort.Tensor("float32", Float32Array.from([point.xNorm, point.yNorm]), [1, 2]);
    const startedAt = performance.now();
    const outputs = await state.session.run({ [state.manifest.inputName]: tensor });
    const elapsedMs = performance.now() - startedAt;
    const outputTensor = outputs[state.manifest.outputName];
    if (!outputTensor) {
      throw new Error(`Missing output "${state.manifest.outputName}" from session.`);
    }
    renderTensor(outputTensor);
    latencyValue.textContent = `${elapsedMs.toFixed(2)} ms`;
    setStatus("Inference complete.");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message, true);
  } finally {
    state.inFlight = false;
    rerunButton.disabled = !state.session;

    if (state.pendingPoint) {
      const nextPoint = state.pendingPoint;
      state.pendingPoint = null;
      void runInference(nextPoint);
    }
  }
}

function pointFromEvent(event) {
  const rect = inputStage.getBoundingClientRect();
  const xNorm = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  const yNorm = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
  return { xNorm, yNorm };
}

function updatePointAndInfer(event) {
  state.currentPoint = pointFromEvent(event);
  updateCrosshair(state.currentPoint.xNorm, state.currentPoint.yNorm);
  void runInference(state.currentPoint);
}

async function loadManifest(manifestUrl) {
  setSessionState("Loading");
  rerunButton.disabled = true;
  loadManifestButton.disabled = true;
  downloadProgress.value = 0;

  try {
    const response = await fetch(manifestUrl);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch manifest: ${response.status} ${response.statusText}. Run the export step first.`,
      );
    }

    const rawManifest = await response.json();
    const manifest = normalizeManifest(rawManifest, manifestUrl);
    const { session, provider } = await createSession(manifest);

    state.session = session;
    state.provider = provider;
    state.manifest = manifest;

    localStorage.setItem(STORAGE_KEY, manifestUrl);
    manifestUrlInput.value = manifestUrl;
    updateRuntimeSummary();
    setSessionState(`Ready on ${provider}`);
    rerunButton.disabled = false;
    setStatus("Model loaded.");
    void runInference();
  } catch (error) {
    state.session = null;
    state.provider = null;
    state.manifest = null;
    updateRuntimeSummary();
    const message = error instanceof Error ? error.message : String(error);
    setSessionState("Load failed");
    setStatus(message, true);
  } finally {
    loadManifestButton.disabled = false;
    downloadProgress.value = 0;
  }
}

function initialManifestUrl() {
  const queryManifest = new URLSearchParams(window.location.search).get("manifest");
  if (queryManifest) {
    return resolveUrl(window.location.href, queryManifest);
  }

  const savedManifest = localStorage.getItem(STORAGE_KEY);
  if (savedManifest) {
    return savedManifest;
  }

  return resolveUrl(window.location.href, DEFAULT_MANIFEST_URL);
}

function setupInputStage() {
  let dragging = false;

  inputStage.addEventListener("pointerdown", (event) => {
    dragging = true;
    inputStage.setPointerCapture(event.pointerId);
    updatePointAndInfer(event);
  });

  inputStage.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    updatePointAndInfer(event);
  });

  inputStage.addEventListener("pointerup", () => {
    dragging = false;
  });

  inputStage.addEventListener("pointerleave", () => {
    dragging = false;
  });
}

function setupSecureContextNotice() {
  if (window.location.protocol === "file:") {
    secureContextNotice.textContent =
      "Open this through an HTTP server. file:// will not reliably load the model or WebGPU.";
    secureContextNotice.style.display = "inline-flex";
    return;
  }

  if (window.isSecureContext || window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost") {
    secureContextNotice.style.display = "none";
  }
}

loadManifestButton.addEventListener("click", () => {
  void loadManifest(manifestUrlInput.value.trim());
});

rerunButton.addEventListener("click", () => {
  void runInference();
});

updateCrosshair(state.currentPoint.xNorm, state.currentPoint.yNorm);
setupInputStage();
setupSecureContextNotice();
manifestUrlInput.value = initialManifestUrl();
setStatus("Waiting for manifest.");
void loadManifest(manifestUrlInput.value);
