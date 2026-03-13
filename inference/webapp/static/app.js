const inputStage = document.getElementById("inputStage");
const crosshair = document.getElementById("crosshair");
const rerunButton = document.getElementById("rerunButton");
const resultImage = document.getElementById("resultImage");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const xValue = document.getElementById("xValue");
const yValue = document.getElementById("yValue");
const latencyValue = document.getElementById("latencyValue");
const statusLine = document.getElementById("statusLine");

let currentPoint = { x_norm: 0.5, y_norm: 0.5 };
let inFlight = false;

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

async function runInference() {
  if (inFlight) return;
  inFlight = true;
  rerunButton.disabled = true;
  setStatus("Running server inference...");

  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(currentPoint),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Inference request failed.");
    }

    resultImage.src = `data:image/png;base64,${payload.image_base64}`;
    resultImage.style.display = "block";
    resultPlaceholder.style.display = "none";
    latencyValue.textContent = `${payload.latency_ms.toFixed(2)} ms`;
    setStatus("Inference complete.");
  } catch (error) {
    latencyValue.textContent = "-";
    setStatus(error.message, true);
  } finally {
    inFlight = false;
    rerunButton.disabled = false;
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

setCrosshair(currentPoint.x_norm, currentPoint.y_norm);
