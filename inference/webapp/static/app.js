const appShell = document.getElementById("appShell");
const inputStage = document.getElementById("inputStage");
const crosshair = document.getElementById("crosshair");
const rerunButton = document.getElementById("rerunButton");
const resultImage = document.getElementById("resultImage");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const spotlightBadge = document.getElementById("spotlightBadge");
const xValue = document.getElementById("xValue");
const yValue = document.getElementById("yValue");
const modelXValue = document.getElementById("modelXValue");
const modelYValue = document.getElementById("modelYValue");
const latencyValue = document.getElementById("latencyValue");
const modeValue = document.getElementById("modeValue");
const temperatureInput = document.getElementById("temperatureInput");
const temperatureValue = document.getElementById("temperatureValue");
const sampleCountInput = document.getElementById("sampleCountInput");
const sampleCountValue = document.getElementById("sampleCountValue");
const statusLine = document.getElementById("statusLine");
const galleryGrid = document.getElementById("galleryGrid");
const galleryEmpty = document.getElementById("galleryEmpty");
const latentPanel = document.getElementById("latentPanel");
const uncertaintyPanel = document.getElementById("uncertaintyPanel");
const deterministicPanel = document.getElementById("deterministicPanel");
const latentSummaryLine = document.getElementById("latentSummaryLine");
const topDimsList = document.getElementById("topDimsList");
const latentChart = document.getElementById("latentChart");
const uncertaintyImage = document.getElementById("uncertaintyImage");
const uncertaintyPlaceholder = document.getElementById("uncertaintyPlaceholder");

let currentPoint = { x_norm: 0.5, y_norm: 0.5 };
let currentResult = null;
let activeGalleryKey = null;
let inFlight = false;

function currentLatentSupport() {
  if (currentResult) {
    return Boolean(currentResult.supports_latent_exploration);
  }
  return appShell.dataset.modelKind === "pointing_cvae";
}

function setCrosshair(xNorm, yNorm) {
  crosshair.style.left = `${xNorm * 100}%`;
  crosshair.style.top = `${yNorm * 100}%`;
  xValue.textContent = xNorm.toFixed(3);
  yValue.textContent = yNorm.toFixed(3);
}

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.dataset.error = isError ? "true" : "false";
}

function syncControlLabels() {
  temperatureValue.textContent = Number.parseFloat(temperatureInput.value).toFixed(2);
  sampleCountValue.textContent = sampleCountInput.value;
}

function setLatentMode(enabled) {
  latentPanel.classList.toggle("hidden", !enabled);
  uncertaintyPanel.classList.toggle("hidden", !enabled);
  deterministicPanel.classList.toggle("hidden", enabled);
  temperatureInput.disabled = !enabled;
  sampleCountInput.disabled = !enabled;
}

function galleryItemSummary(item) {
  if (item.kind === "mean") {
    return `Deterministic decode from prior mean · ||mu|| ${item.latent_l2.toFixed(2)}`;
  }
  if (item.kind === "sample") {
    return `Delta from mean ${item.mean_absolute_delta.toFixed(3)} · ||z|| ${item.latent_l2.toFixed(2)}`;
  }
  return "Single deterministic decode";
}

function renderGallery(gallery) {
  galleryGrid.innerHTML = "";
  const items = Array.isArray(gallery) ? gallery : [];
  galleryEmpty.classList.toggle("hidden", items.length > 0);

  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "gallery-card";
    button.dataset.key = item.key;
    button.dataset.active = item.key === activeGalleryKey ? "true" : "false";

    const image = document.createElement("img");
    image.alt = item.label;
    image.src = `data:image/png;base64,${item.image_base64}`;

    const title = document.createElement("strong");
    title.textContent = item.label;

    const summary = document.createElement("span");
    summary.textContent = galleryItemSummary(item);

    button.append(image, title, summary);
    button.addEventListener("click", () => {
      setSpotlight(item.key);
    });
    galleryGrid.appendChild(button);
  }
}

function setSpotlight(key) {
  if (!currentResult) return;

  const gallery = Array.isArray(currentResult.gallery) ? currentResult.gallery : [];
  const item = gallery.find((candidate) => candidate.key === key) || gallery[0];
  if (!item) return;

  activeGalleryKey = item.key;
  resultImage.src = `data:image/png;base64,${item.image_base64}`;
  resultImage.style.display = "block";
  resultPlaceholder.style.display = "none";
  spotlightBadge.textContent = item.label;

  for (const card of galleryGrid.querySelectorAll(".gallery-card")) {
    card.dataset.active = card.dataset.key === activeGalleryKey ? "true" : "false";
  }
}

function renderTopDims(priorDistribution) {
  topDimsList.innerHTML = "";
  for (const entry of priorDistribution.top_uncertainty_dims || []) {
    const chip = document.createElement("div");
    chip.className = "metric-chip";

    const title = document.createElement("strong");
    title.textContent = `dim ${entry.dim}`;

    const details = document.createElement("span");
    details.textContent = `mu ${entry.mean.toFixed(3)} · sigma ${entry.std.toFixed(3)}`;

    chip.append(title, details);
    topDimsList.appendChild(chip);
  }
}

function renderLatentChart(priorDistribution) {
  const width = 720;
  const height = 240;
  const margin = { top: 18, right: 20, bottom: 34, left: 40 };
  const mean = priorDistribution.mean || [];
  const std = priorDistribution.std || [];
  const samples = (priorDistribution.samples || []).map((sample) => sample.values || []);

  if (mean.length === 0) {
    latentChart.innerHTML = "";
    return;
  }

  const extrema = [];
  for (let index = 0; index < mean.length; index += 1) {
    extrema.push(mean[index] - std[index], mean[index] + std[index]);
  }
  for (const sample of samples) {
    for (const value of sample) {
      extrema.push(value);
    }
  }

  const maxAbs = Math.max(0.25, ...extrema.map((value) => Math.abs(value))) * 1.15;
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const xStep = mean.length === 1 ? 0 : chartWidth / (mean.length - 1);
  const xAt = (index) => margin.left + (mean.length === 1 ? chartWidth / 2 : index * xStep);
  const yAt = (value) => margin.top + ((maxAbs - value) / (maxAbs * 2)) * chartHeight;

  const parts = [];
  parts.push(`<rect x="0" y="0" width="${width}" height="${height}" fill="transparent" />`);

  for (const tick of [-maxAbs, -maxAbs / 2, 0, maxAbs / 2, maxAbs]) {
    const y = yAt(tick);
    const label = tick.toFixed(1);
    parts.push(`<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(28,26,24,0.11)" stroke-width="1" />`);
    parts.push(`<text x="6" y="${y + 4}" font-size="11" fill="rgba(28,26,24,0.56)">${label}</text>`);
  }

  for (let index = 0; index < mean.length; index += 1) {
    const x = xAt(index);
    const yLow = yAt(mean[index] - std[index]);
    const yHigh = yAt(mean[index] + std[index]);
    const yMean = yAt(mean[index]);
    parts.push(`<line x1="${x}" y1="${yLow}" x2="${x}" y2="${yHigh}" stroke="rgba(236,107,45,0.45)" stroke-width="6" stroke-linecap="round" />`);
    parts.push(`<circle cx="${x}" cy="${yMean}" r="4" fill="#1c1a18" />`);

    for (const sample of samples) {
      const sampleValue = sample[index];
      if (typeof sampleValue !== "number") continue;
      parts.push(`<circle cx="${x}" cy="${yAt(sampleValue)}" r="2.6" fill="#ec6b2d" fill-opacity="0.78" />`);
    }

    if (index === 0 || index === mean.length - 1 || index % 4 === 0) {
      parts.push(`<text x="${x}" y="${height - 10}" text-anchor="middle" font-size="10" fill="rgba(28,26,24,0.56)">${index}</text>`);
    }
  }

  latentChart.innerHTML = parts.join("");
}

function renderPriorInspector(payload) {
  if (!payload.supports_latent_exploration || !payload.prior_distribution) {
    latentSummaryLine.textContent = "This checkpoint does not expose a coordinate-conditioned latent prior.";
    topDimsList.innerHTML = "";
    latentChart.innerHTML = "";
    return;
  }

  const priorDistribution = payload.prior_distribution;
  latentSummaryLine.textContent =
    `Prior mean across ${priorDistribution.mean.length} dims. Avg sigma ${priorDistribution.avg_std.toFixed(3)} · Max sigma ${priorDistribution.max_std.toFixed(3)} · Sample temperature ${payload.temperature.toFixed(2)}.`;
  renderTopDims(priorDistribution);
  renderLatentChart(priorDistribution);
}

function renderUncertainty(payload) {
  if (payload.uncertainty_heatmap_base64) {
    uncertaintyImage.src = `data:image/png;base64,${payload.uncertainty_heatmap_base64}`;
    uncertaintyImage.style.display = "block";
    uncertaintyPlaceholder.style.display = "none";
    return;
  }

  uncertaintyImage.style.display = "none";
  uncertaintyPlaceholder.style.display = "grid";
  uncertaintyPlaceholder.textContent =
    payload.supports_latent_exploration && payload.sample_count < 2
      ? "Need at least two prior samples to estimate spread."
      : "This checkpoint does not expose sampled prior uncertainty.";
}

function renderInference(payload) {
  currentResult = payload;
  modeValue.textContent = payload.model_kind;
  modelXValue.textContent = Number(payload.model_x).toFixed(3);
  modelYValue.textContent = Number(payload.model_y).toFixed(3);
  latencyValue.textContent = `${payload.latency_ms.toFixed(2)} ms`;
  setLatentMode(Boolean(payload.supports_latent_exploration));
  renderGallery(payload.gallery || []);
  renderPriorInspector(payload);
  renderUncertainty(payload);
  setSpotlight(payload.default_gallery_key || (payload.gallery || [])[0]?.key);
}

async function runInference() {
  if (inFlight) return;
  inFlight = true;
  rerunButton.disabled = true;
  temperatureInput.disabled = true;
  sampleCountInput.disabled = true;
  setStatus("Running server inference...");

  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ...currentPoint,
        sample_count: Number.parseInt(sampleCountInput.value, 10),
        temperature: Number.parseFloat(temperatureInput.value),
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Inference request failed.");
    }

    renderInference(payload);
    setStatus("Inference complete.");
  } catch (error) {
    latencyValue.textContent = "-";
    setStatus(error.message, true);
  } finally {
    inFlight = false;
    rerunButton.disabled = false;
    temperatureInput.disabled = !currentLatentSupport();
    sampleCountInput.disabled = !currentLatentSupport();
  }
}

function updatePointFromEvent(event) {
  const rect = inputStage.getBoundingClientRect();
  const xNorm = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  const yNorm = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
  currentPoint = { x_norm: xNorm, y_norm: yNorm };
  setCrosshair(xNorm, yNorm);
  void runInference();
}

inputStage.addEventListener("click", updatePointFromEvent);
rerunButton.addEventListener("click", () => {
  void runInference();
});
temperatureInput.addEventListener("input", syncControlLabels);
sampleCountInput.addEventListener("input", syncControlLabels);
temperatureInput.addEventListener("change", () => {
  if (currentResult?.supports_latent_exploration) {
    void runInference();
  }
});
sampleCountInput.addEventListener("change", () => {
  if (currentResult?.supports_latent_exploration) {
    void runInference();
  }
});

syncControlLabels();
setLatentMode(currentLatentSupport());
setCrosshair(currentPoint.x_norm, currentPoint.y_norm);
