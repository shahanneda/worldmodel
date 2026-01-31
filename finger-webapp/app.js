const startCameraBtn = document.getElementById("startCamera");
const recordBtn = document.getElementById("record5s");
const statusEl = document.getElementById("status");
const videoEl = document.getElementById("video");
const compositeEl = document.getElementById("composite");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const compositeCtx = compositeEl.getContext("2d");
const liveDataEl = document.getElementById("liveData");
const sparklineEl = document.getElementById("sparkline");
const sparkCtx = sparklineEl.getContext("2d");

let stream = null;
let camera = null;
let recorder = null;
let recordTimeout = null;
let recording = false;
let landmarksLog = [];
let lastHandsResult = null;
let liveHistory = [];
const maxHistory = 180;
let lastSegmentationResult = null;

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

const selfieSegmentation = new SelfieSegmentation({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7,
  selfieMode: false,
});

selfieSegmentation.setOptions({
  modelSelection: 1,
});

function setStatus(text) {
  statusEl.textContent = text;
}

async function startCamera() {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
    resizeCanvasToVideo();
    window.addEventListener("resize", resizeCanvasToVideo);
    recordBtn.disabled = false;
    setStatus("Camera ready");
    try {
      await hands.send({ image: videoEl });
    } catch (err) {
      console.error(err);
      setStatus("Hand model init error");
    }
    camera = new Camera(videoEl, {
      onFrame: async () => {
        try {
          await hands.send({ image: videoEl });
          await selfieSegmentation.send({ image: videoEl });
        } catch (err) {
          console.error(err);
          setStatus("Hand model error");
        }
      },
      width: 1280,
      height: 720,
    });
    camera.start();
  } catch (err) {
    console.error(err);
    setStatus("Camera access denied");
  }
}

function resizeCanvasToVideo() {
  const { videoWidth, videoHeight } = videoEl;
  if (videoWidth && videoHeight) {
    canvasEl.width = videoWidth;
    canvasEl.height = videoHeight;
    compositeEl.width = videoWidth;
    compositeEl.height = videoHeight;
  }
}

function drawComposite() {
  if (!lastSegmentationResult || !lastSegmentationResult.segmentationMask) {
    compositeCtx.clearRect(0, 0, compositeEl.width, compositeEl.height);
    compositeCtx.fillStyle = "#ffffff";
    compositeCtx.fillRect(0, 0, compositeEl.width, compositeEl.height);
    return;
  }

  const w = compositeEl.width;
  const h = compositeEl.height;

  compositeCtx.clearRect(0, 0, w, h);
  compositeCtx.drawImage(lastSegmentationResult.segmentationMask, 0, 0, w, h);
  compositeCtx.globalCompositeOperation = "source-in";
  compositeCtx.drawImage(videoEl, 0, 0, w, h);
  compositeCtx.globalCompositeOperation = "destination-over";
  compositeCtx.fillStyle = "#ffffff";
  compositeCtx.fillRect(0, 0, w, h);
  compositeCtx.globalCompositeOperation = "source-over";
}

function drawOverlay() {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (!lastHandsResult || !lastHandsResult.multiHandLandmarks) return;

  lastHandsResult.multiHandLandmarks.forEach((landmarks) => {
    const scaled = landmarks.map((lm) => ({
      x: lm.x * canvasEl.width,
      y: lm.y * canvasEl.height,
      z: lm.z,
    }));

    drawConnectors(ctx, scaled, HAND_CONNECTIONS, {
      color: "#7bdff2",
      lineWidth: 3,
    });
    drawLandmarks(ctx, scaled, {
      color: "#f7d6e0",
      lineWidth: 1,
      radius: 4,
    });

    const indexTip = scaled[8];
    if (indexTip) {
      ctx.fillStyle = "#ff3b30";
      ctx.beginPath();
      ctx.arc(indexTip.x, indexTip.y, 8, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

hands.onResults((results) => {
  lastHandsResult = results;
  drawOverlay();
  updateLiveData(results);
  if (recording) {
    const now = performance.now();
    const frame = {
      t: now,
      multiHandLandmarks: results.multiHandLandmarks || [],
      multiHandedness: results.multiHandedness || [],
    };
    landmarksLog.push(frame);
  } else if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    setStatus("No hand detected");
  } else if (stream) {
    setStatus("Tracking");
  }
});

selfieSegmentation.onResults((results) => {
  lastSegmentationResult = results;
  drawComposite();
});

function updateLiveData(results) {
  const firstHand = results.multiHandLandmarks?.[0];
  if (!firstHand) {
    liveHistory.push(null);
    if (liveHistory.length > maxHistory) liveHistory.shift();
    drawSparkline();
    liveDataEl.textContent = "No hand detected";
    return;
  }

  const indexTip = firstHand[8];
  const frame = {
    t: performance.now(),
    indexTip: {
      x: Number(indexTip.x.toFixed(4)),
      y: Number(indexTip.y.toFixed(4)),
      z: Number(indexTip.z.toFixed(4)),
    },
    landmarksCount: firstHand.length,
  };

  liveHistory.push(frame.indexTip.y);
  if (liveHistory.length > maxHistory) liveHistory.shift();
  drawSparkline();
  liveDataEl.textContent = JSON.stringify(frame, null, 2);
}

function drawSparkline() {
  const w = sparklineEl.width;
  const h = sparklineEl.height;
  sparkCtx.clearRect(0, 0, w, h);

  sparkCtx.fillStyle = "#0b1018";
  sparkCtx.fillRect(0, 0, w, h);

  const values = liveHistory.filter((v) => typeof v === "number");
  if (values.length < 2) return;

  sparkCtx.strokeStyle = "#7bdff2";
  sparkCtx.lineWidth = 2;
  sparkCtx.beginPath();

  const startIndex = liveHistory.length - values.length;
  const step = w / Math.max(values.length - 1, 1);
  let drawIndex = 0;

  for (let i = startIndex; i < liveHistory.length; i += 1) {
    const v = liveHistory[i];
    if (typeof v !== "number") continue;
    const x = drawIndex * step;
    const y = v * h;
    if (drawIndex === 0) sparkCtx.moveTo(x, y);
    else sparkCtx.lineTo(x, y);
    drawIndex += 1;
  }

  sparkCtx.stroke();

  sparkCtx.strokeStyle = "rgba(247, 214, 224, 0.7)";
  sparkCtx.lineWidth = 1;
  sparkCtx.beginPath();
  sparkCtx.moveTo(0, h / 2);
  sparkCtx.lineTo(w, h / 2);
  sparkCtx.stroke();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function startRecording() {
  if (!stream || recording) return;
  recording = true;
  landmarksLog = [];

  const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9"
    : "video/webm";

  recorder = new MediaRecorder(stream, { mimeType });
  const chunks = [];

  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };

  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: mimeType });
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    downloadBlob(blob, `finger-video-${ts}.webm`);

    const jsonBlob = new Blob([JSON.stringify(landmarksLog, null, 2)], {
      type: "application/json",
    });
    downloadBlob(jsonBlob, `finger-landmarks-${ts}.json`);

    setStatus("Recording saved");
    recordBtn.disabled = false;
    recording = false;
  };

  recorder.start(100);
  setStatus("Recording... 10s");
  recordBtn.disabled = true;

  recordTimeout = setTimeout(() => {
    recorder.stop();
    clearTimeout(recordTimeout);
  }, 10000);
}

startCameraBtn.addEventListener("click", startCamera);
recordBtn.addEventListener("click", startRecording);
