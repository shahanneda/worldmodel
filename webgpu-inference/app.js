const ORT_VERSION = "1.24.3";
const DEFAULT_MANIFEST_URL = "./model/manifest.json";
const STORAGE_KEY = "worldmodel.webgpu.manifestUrl";
const STAGE_LAYOUT_KEY = "worldmodel.webgpu.stageLayout";
const CACHE_INDEX_KEY = "worldmodel.webgpu.assetCacheIndex";
const CUSTOM_CHECKPOINT_ID = "__custom__";
const MODEL_ASSET_CACHE_NAME = "worldmodel.webgpu.modelAssets.v1";
const DEFAULT_MAX_CACHED_CHECKPOINTS = 2;
const CDN_WASM_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const RUNTIME_CONFIG = window.WORLDMODEL_WEBGPU_CONFIG || {};

const checkpointSelect = document.getElementById("checkpointSelect");
const manifestUrlInput = document.getElementById("manifestUrlInput");
const loadManifestButton = document.getElementById("loadManifestButton");
const rerunButton = document.getElementById("rerunButton");
const stageGrid = document.getElementById("stageGrid");
const stageLayoutToggle = document.getElementById("stageLayoutToggle");
const inputStage = document.getElementById("inputStage");
const stageOutputLayer = document.getElementById("stageOutputLayer");
const crosshair = document.getElementById("crosshair");
const outputFrame = document.getElementById("outputFrame");
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
  checkpoints: [],
  stageLayout: "split",
  currentPoint: { xNorm: 0.5, yNorm: 0.5 },
  inFlight: false,
  pendingPoint: null,
  lastLoadCacheHits: 0,
  lastLoadCacheWrites: 0,
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

function safeLocalStorageGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeLocalStorageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore storage failures in private mode or quota-limited contexts.
  }
}

function normalizeStageLayout(value) {
  return value === "stacked" ? "stacked" : "split";
}

function setOutputSurfaceParent(parent) {
  if (!parent) return;
  if (outputCanvas.parentElement !== parent) {
    parent.appendChild(outputCanvas);
  }
  if (outputPlaceholder.parentElement !== parent) {
    parent.appendChild(outputPlaceholder);
  }
}

function applyStageLayout(layout) {
  const nextLayout = normalizeStageLayout(layout);
  const stacked = nextLayout === "stacked";
  state.stageLayout = nextLayout;
  stageGrid.classList.toggle("is-stacked", stacked);
  stageLayoutToggle.checked = stacked;
  setOutputSurfaceParent(stacked ? stageOutputLayer : outputFrame);
  safeLocalStorageSet(STAGE_LAYOUT_KEY, nextLayout);
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

function checkpointManifestUrl(entry) {
  if (!entry || typeof entry !== "object") return null;
  const manifestUrl = entry.manifestUrl || entry.url;
  if (typeof manifestUrl !== "string" || manifestUrl.length === 0) return null;
  return resolveUrl(window.location.href, manifestUrl);
}

function buildCheckpointOptions() {
  const rawCheckpoints = Array.isArray(RUNTIME_CONFIG.checkpoints) ? RUNTIME_CONFIG.checkpoints : [];
  const options = [];
  const seenManifestUrls = new Set();

  rawCheckpoints.forEach((entry, index) => {
    const manifestUrl = checkpointManifestUrl(entry);
    if (!manifestUrl || seenManifestUrls.has(manifestUrl)) return;
    seenManifestUrls.add(manifestUrl);
    options.push({
      id:
        typeof entry.id === "string" && entry.id.length > 0
          ? entry.id
          : `checkpoint-${index + 1}`,
      label:
        typeof entry.label === "string" && entry.label.length > 0
          ? entry.label
          : `Checkpoint ${index + 1}`,
      manifestUrl,
    });
  });

  if (
    typeof RUNTIME_CONFIG.manifestUrl === "string" &&
    RUNTIME_CONFIG.manifestUrl.length > 0
  ) {
    const manifestUrl = resolveUrl(window.location.href, RUNTIME_CONFIG.manifestUrl);
    if (!seenManifestUrls.has(manifestUrl)) {
      options.unshift({
        id:
          typeof RUNTIME_CONFIG.defaultCheckpointId === "string" &&
          RUNTIME_CONFIG.defaultCheckpointId.length > 0
            ? RUNTIME_CONFIG.defaultCheckpointId
            : "default",
        label:
          typeof RUNTIME_CONFIG.defaultCheckpointLabel === "string" &&
          RUNTIME_CONFIG.defaultCheckpointLabel.length > 0
            ? RUNTIME_CONFIG.defaultCheckpointLabel
            : "Current default",
        manifestUrl,
      });
    }
  }

  if (options.length === 0) {
    options.push({
      id: "local-export",
      label: "Local export",
      manifestUrl: resolveUrl(window.location.href, DEFAULT_MANIFEST_URL),
    });
  }

  return options;
}

function syncCheckpointSelection(manifestUrl) {
  const resolvedManifestUrl = resolveUrl(window.location.href, manifestUrl);
  const selectedOption = state.checkpoints.find(
    (option) => option.manifestUrl === resolvedManifestUrl,
  );

  if (selectedOption) {
    checkpointSelect.value = selectedOption.id;
    manifestUrlInput.value = selectedOption.manifestUrl;
    manifestUrlInput.disabled = true;
    return;
  }

  checkpointSelect.value = CUSTOM_CHECKPOINT_ID;
  manifestUrlInput.disabled = false;
  manifestUrlInput.value = resolvedManifestUrl;
}

function populateCheckpointSelect(selectedManifestUrl) {
  state.checkpoints = buildCheckpointOptions();
  checkpointSelect.innerHTML = "";

  state.checkpoints.forEach((option) => {
    const element = document.createElement("option");
    element.value = option.id;
    element.textContent = option.label;
    checkpointSelect.appendChild(element);
  });

  const customOption = document.createElement("option");
  customOption.value = CUSTOM_CHECKPOINT_ID;
  customOption.textContent = "Custom manifest URL";
  checkpointSelect.appendChild(customOption);

  syncCheckpointSelection(selectedManifestUrl);
}

function selectedCheckpointManifestUrl() {
  if (checkpointSelect.value === CUSTOM_CHECKPOINT_ID) {
    return null;
  }

  const selectedOption = state.checkpoints.find((option) => option.id === checkpointSelect.value);
  return selectedOption ? selectedOption.manifestUrl : null;
}

function assetCacheAvailable() {
  return "caches" in window;
}

function maxCachedCheckpoints() {
  const configuredValue = Number.parseInt(RUNTIME_CONFIG.maxCachedCheckpoints, 10);
  if (Number.isFinite(configuredValue) && configuredValue > 0) {
    return configuredValue;
  }
  return DEFAULT_MAX_CACHED_CHECKPOINTS;
}

function loadAssetCacheIndex() {
  const raw = safeLocalStorageGet(CACHE_INDEX_KEY);
  if (!raw) {
    return { models: {} };
  }

  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || typeof parsed.models !== "object") {
      return { models: {} };
    }
    return parsed;
  } catch {
    return { models: {} };
  }
}

function saveAssetCacheIndex(index) {
  safeLocalStorageSet(CACHE_INDEX_KEY, JSON.stringify(index));
}

function cacheModelId(manifest) {
  return manifest.graphUrl;
}

function cacheAssetUrls(manifest) {
  return [manifest.graphUrl, ...manifest.externalData.map((entry) => entry.url)];
}

async function pruneCachedModels(limit) {
  if (!assetCacheAvailable()) {
    return;
  }

  const cache = await caches.open(MODEL_ASSET_CACHE_NAME);
  const index = loadAssetCacheIndex();
  const entries = Object.entries(index.models || {}).sort(
    ([, left], [, right]) => (right?.lastUsedAt || 0) - (left?.lastUsedAt || 0),
  );

  for (const [modelId, record] of entries.slice(Math.max(0, limit))) {
    const assetUrls = Array.isArray(record?.assetUrls) ? record.assetUrls : [];
    for (const assetUrl of assetUrls) {
      await cache.delete(assetUrl);
    }
    delete index.models[modelId];
  }

  saveAssetCacheIndex(index);
}

async function prepareCacheForManifest(manifest) {
  if (!assetCacheAvailable()) {
    return;
  }

  const index = loadAssetCacheIndex();
  if (index.models[cacheModelId(manifest)]) {
    return;
  }

  await pruneCachedModels(Math.max(0, maxCachedCheckpoints() - 1));
}

async function rememberCachedManifest(manifest) {
  if (!assetCacheAvailable()) {
    return;
  }

  const index = loadAssetCacheIndex();
  index.models[cacheModelId(manifest)] = {
    assetUrls: cacheAssetUrls(manifest),
    lastUsedAt: Date.now(),
  };
  saveAssetCacheIndex(index);
  await pruneCachedModels(maxCachedCheckpoints());
}

async function readCachedBytes(url) {
  if (!assetCacheAvailable()) {
    return null;
  }

  try {
    const cache = await caches.open(MODEL_ASSET_CACHE_NAME);
    const response = await cache.match(url);
    if (!response || !response.ok) {
      return null;
    }
    return new Uint8Array(await response.arrayBuffer());
  } catch {
    return null;
  }
}

async function writeCachedBytes(url, bytes, contentType) {
  if (!assetCacheAvailable()) {
    return false;
  }

  try {
    const headers = new Headers();
    if (contentType) {
      headers.set("content-type", contentType);
    }
    const cache = await caches.open(MODEL_ASSET_CACHE_NAME);
    await cache.put(url, new Response(bytes, { headers }));
    return true;
  } catch {
    return false;
  }
}

async function fetchBinary(url, label, expectedBytes = null) {
  const cachedBytes = await readCachedBytes(url);
  if (cachedBytes) {
    state.lastLoadCacheHits += 1;
    downloadProgress.value = 1;
    setStatus(`Loaded ${label} from local cache.`);
    return cachedBytes;
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${label}: ${response.status} ${response.statusText}`);
  }

  const contentType = response.headers.get("content-type");
  const contentLengthHeader = response.headers.get("content-length");
  const totalBytes =
    expectedBytes ||
    (contentLengthHeader ? Number.parseInt(contentLengthHeader, 10) : Number.NaN);

  let bytes;
  if (!response.body || !Number.isFinite(totalBytes) || totalBytes <= 0) {
    setStatus(`Downloading ${label}...`);
    bytes = new Uint8Array(await response.arrayBuffer());
  } else {
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

    bytes = loadedBytes !== totalBytes ? merged.slice(0, loadedBytes) : merged;
  }

  if (await writeCachedBytes(url, bytes, contentType)) {
    state.lastLoadCacheWrites += 1;
  }

  return bytes;
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
  const graphBuffer = await fetchBinary(manifest.graphUrl, "ONNX graph", manifest.graphBytes);

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
  const resolvedManifestUrl = resolveUrl(window.location.href, manifestUrl);
  syncCheckpointSelection(resolvedManifestUrl);
  setSessionState("Loading");
  rerunButton.disabled = true;
  checkpointSelect.disabled = true;
  loadManifestButton.disabled = true;
  downloadProgress.value = 0;
  state.lastLoadCacheHits = 0;
  state.lastLoadCacheWrites = 0;

  try {
    const response = await fetch(resolvedManifestUrl);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch manifest: ${response.status} ${response.statusText}. Run the export step first.`,
      );
    }

    const rawManifest = await response.json();
    const manifest = normalizeManifest(rawManifest, resolvedManifestUrl);
    await prepareCacheForManifest(manifest);
    const { session, provider } = await createSession(manifest);

    state.session = session;
    state.provider = provider;
    state.manifest = manifest;

    await rememberCachedManifest(manifest);
    safeLocalStorageSet(STORAGE_KEY, resolvedManifestUrl);
    syncCheckpointSelection(resolvedManifestUrl);
    updateRuntimeSummary();
    setSessionState(`Ready on ${provider}`);
    rerunButton.disabled = false;

    const cacheNotes = [];
    if (state.lastLoadCacheHits > 0) {
      cacheNotes.push(
        `reused ${state.lastLoadCacheHits} cached file${state.lastLoadCacheHits === 1 ? "" : "s"}`,
      );
    }
    if (state.lastLoadCacheWrites > 0) {
      cacheNotes.push(
        `stored ${state.lastLoadCacheWrites} file${state.lastLoadCacheWrites === 1 ? "" : "s"} locally`,
      );
    }
    setStatus(
      cacheNotes.length > 0 ? `Model loaded, ${cacheNotes.join(", ")}.` : "Model loaded.",
    );
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
    checkpointSelect.disabled = false;
    loadManifestButton.disabled = false;
    downloadProgress.value = 0;
  }
}

function initialManifestUrl() {
  const queryManifest = new URLSearchParams(window.location.search).get("manifest");
  if (queryManifest) {
    return resolveUrl(window.location.href, queryManifest);
  }

  const savedManifest = safeLocalStorageGet(STORAGE_KEY);
  if (savedManifest) {
    return savedManifest;
  }

  if (typeof RUNTIME_CONFIG.manifestUrl === "string" && RUNTIME_CONFIG.manifestUrl.length > 0) {
    return resolveUrl(window.location.href, RUNTIME_CONFIG.manifestUrl);
  }

  return resolveUrl(window.location.href, DEFAULT_MANIFEST_URL);
}

function initialStageLayout() {
  const queryLayout = new URLSearchParams(window.location.search).get("layout");
  if (queryLayout) {
    return normalizeStageLayout(queryLayout);
  }
  return normalizeStageLayout(safeLocalStorageGet(STAGE_LAYOUT_KEY));
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
      "Open this through an HTTP server. file:// will not reliably load the model, cache, or WebGPU.";
    secureContextNotice.style.display = "inline-flex";
    return;
  }

  if (
    window.isSecureContext ||
    window.location.hostname === "127.0.0.1" ||
    window.location.hostname === "localhost"
  ) {
    secureContextNotice.style.display = "none";
  }
}

checkpointSelect.addEventListener("change", () => {
  const selectedManifestUrl = selectedCheckpointManifestUrl();
  if (!selectedManifestUrl) {
    manifestUrlInput.disabled = false;
    manifestUrlInput.focus();
    manifestUrlInput.select();
    return;
  }

  manifestUrlInput.disabled = true;
  manifestUrlInput.value = selectedManifestUrl;
  void loadManifest(selectedManifestUrl);
});

loadManifestButton.addEventListener("click", () => {
  void loadManifest(manifestUrlInput.value.trim());
});

rerunButton.addEventListener("click", () => {
  void runInference();
});

stageLayoutToggle.addEventListener("change", () => {
  applyStageLayout(stageLayoutToggle.checked ? "stacked" : "split");
});

const startingManifestUrl = initialManifestUrl();
const startingStageLayout = initialStageLayout();
updateCrosshair(state.currentPoint.xNorm, state.currentPoint.yNorm);
applyStageLayout(startingStageLayout);
setupInputStage();
setupSecureContextNotice();
populateCheckpointSelect(startingManifestUrl);
setStatus("Waiting for manifest.");
void loadManifest(startingManifestUrl);
