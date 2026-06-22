const tabs = document.querySelectorAll('.tab');
const views = document.querySelectorAll('.view');
const apiStatus = document.getElementById('api-status');
const databaseSize = document.getElementById('database-size');

const enrollVideo = document.getElementById('enroll-video');
const enrollCanvas = document.getElementById('enroll-canvas');
const detectVideo = document.getElementById('detect-video');
const detectCanvas = document.getElementById('detect-canvas');
const personName = document.getElementById('person-name');
const imageFile = document.getElementById('image-file');
const enrollMessage = document.getElementById('enroll-message');
const detectMessage = document.getElementById('detect-message');

const enrollFileBtn = document.getElementById('enroll-file-btn');
const enrollCameraBtn = document.getElementById('enroll-camera-btn');
const startDetectBtn = document.getElementById('start-detect-btn');
const stopDetectBtn = document.getElementById('stop-detect-btn');

let enrollStream = null;
let detectStream = null;
let detectTimer = null;
let latestFaces = [];
let smoothedFaces = [];
let recognizeBusy = false;
let drawAnimation = null;

const captureCanvas = document.createElement('canvas');
const SMOOTHING = 0.35;
const HOLD_FRAMES = 4;

function setMessage(element, text, type = '') {
  element.textContent = text;
  element.className = `message ${type}`.trim();
}

function switchView(viewId) {
  tabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.view === viewId));
  views.forEach((view) => view.classList.toggle('active', view.id === viewId));

  if (viewId === 'enroll-view') {
    startEnrollCamera();
  }
}

tabs.forEach((tab) => {
  tab.addEventListener('click', () => switchView(tab.dataset.view));
});

async function refreshStatus() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    apiStatus.textContent = data.status;
    databaseSize.textContent = `${data.database_size} embeddings`;
  } catch (error) {
    apiStatus.textContent = 'offline';
  }
}

async function startCamera(videoElement, currentStream) {
  if (currentStream) {
    return currentStream;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  videoElement.srcObject = stream;
  await videoElement.play();
  return stream;
}

async function startEnrollCamera() {
  try {
    enrollStream = await startCamera(enrollVideo, enrollStream);
  } catch (error) {
    setMessage(enrollMessage, 'Không mở được camera. Hãy kiểm tra quyền camera của trình duyệt.', 'error');
  }
}

function drawVideoToCanvas(videoElement, canvasElement) {
  const width = videoElement.videoWidth || 1280;
  const height = videoElement.videoHeight || 720;
  canvasElement.width = width;
  canvasElement.height = height;

  const ctx = canvasElement.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, width, height);
}

function canvasToBlob(canvasElement) {
  return new Promise((resolve) => {
    canvasElement.toBlob((blob) => resolve(blob), 'image/jpeg', 0.9);
  });
}

async function captureFrameBlob(videoElement) {
  drawVideoToCanvas(videoElement, captureCanvas);
  return canvasToBlob(captureCanvas);
}

async function capturePreviewBlob(videoElement, canvasElement) {
  drawVideoToCanvas(videoElement, canvasElement);
  return canvasToBlob(canvasElement);
}

async function enroll(blob, filename) {
  const name = personName.value.trim();
  if (!name) {
    setMessage(enrollMessage, 'Vui lòng nhập tên trước khi đăng ký.', 'error');
    return;
  }

  if (!blob) {
    setMessage(enrollMessage, 'Không có ảnh để đăng ký.', 'error');
    return;
  }

  const form = new FormData();
  form.append('name', name);
  form.append('file', blob, filename);

  setMessage(enrollMessage, 'Đang đăng ký khuôn mặt...');

  try {
    const response = await fetch('/enroll', { method: 'POST', body: form });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Đăng ký thất bại.');
    }

    setMessage(enrollMessage, `Đã đăng ký ${data.person.name}. Số mặt thấy trong ảnh: ${data.faces_detected}.`, 'ok');
    await refreshStatus();
  } catch (error) {
    setMessage(enrollMessage, error.message, 'error');
  }
}

enrollFileBtn.addEventListener('click', async () => {
  const file = imageFile.files[0];
  await enroll(file, file ? file.name : 'upload.jpg');
});

enrollCameraBtn.addEventListener('click', async () => {
  await startEnrollCamera();
  const blob = await captureFrameBlob(enrollVideo);
  await enroll(blob, 'camera.jpg');
});

function fitCanvasToVideo(videoElement, canvasElement) {
  const width = videoElement.videoWidth || 1280;
  const height = videoElement.videoHeight || 720;
  if (canvasElement.width !== width || canvasElement.height !== height) {
    canvasElement.width = width;
    canvasElement.height = height;
  }
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpArray(prev, next, t) {
  return next.map((value, index) => lerp(prev[index] ?? value, value, t));
}

function bboxCenter(bbox) {
  return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
}

function bboxDistance(a, b) {
  const [ax, ay] = bboxCenter(a);
  const [bx, by] = bboxCenter(b);
  return Math.hypot(ax - bx, ay - by);
}

function faceKey(face) {
  if (face.person_id !== null && face.person_id !== undefined) {
    return `id:${face.person_id}`;
  }
  if (face.name && face.name !== 'unknown') {
    return `name:${face.name}`;
  }
  return null;
}

function findPreviousFace(face, usedIndexes) {
  const key = faceKey(face);
  if (key) {
    const keyedIndex = smoothedFaces.findIndex((candidate, index) => !usedIndexes.has(index) && candidate.key === key);
    if (keyedIndex !== -1) {
      return keyedIndex;
    }
  }

  let bestIndex = -1;
  let bestDistance = Infinity;

  smoothedFaces.forEach((candidate, index) => {
    if (usedIndexes.has(index)) {
      return;
    }

    const distance = bboxDistance(candidate.bbox, face.bbox);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });

  return bestDistance < 120 ? bestIndex : -1;
}

function smoothIncomingFaces(faces) {
  const usedIndexes = new Set();
  const nextSmoothed = [];

  faces.forEach((face) => {
    const previousIndex = findPreviousFace(face, usedIndexes);
    const key = faceKey(face);

    if (previousIndex === -1) {
      nextSmoothed.push({ ...face, key, hold: HOLD_FRAMES });
      return;
    }

    usedIndexes.add(previousIndex);
    const previous = smoothedFaces[previousIndex];
    const landmarks = face.landmarks.map((point, index) => lerpArray(previous.landmarks[index] || point, point, SMOOTHING));

    nextSmoothed.push({
      ...face,
      key,
      bbox: lerpArray(previous.bbox, face.bbox, SMOOTHING),
      landmarks,
      score: lerp(previous.score, face.score, SMOOTHING),
      hold: HOLD_FRAMES,
    });
  });

  smoothedFaces.forEach((face, index) => {
    if (!usedIndexes.has(index) && face.hold > 0) {
      nextSmoothed.push({ ...face, hold: face.hold - 1 });
    }
  });

  smoothedFaces = nextSmoothed;
  latestFaces = smoothedFaces;
}

function displayBox(face) {
  const [x1, y1, x2, y2] = face.bbox;
  const width = x2 - x1;
  const height = y2 - y1;
  return [
    Math.max(0, x1 - width * 0.06),
    Math.max(0, y1 - height * 0.12),
    Math.min(detectCanvas.width, x2 + width * 0.06),
    Math.min(detectCanvas.height, y2 + height * 0.05),
  ];
}

function drawDetections() {
  fitCanvasToVideo(detectVideo, detectCanvas);
  const ctx = detectCanvas.getContext('2d');
  ctx.clearRect(0, 0, detectCanvas.width, detectCanvas.height);

  latestFaces.forEach((face) => {
    const [x1, y1, x2, y2] = displayBox(face);
    const known = face.name !== 'unknown';
    const color = known ? '#18b37f' : '#e03131';
    const label = `${face.name} ${face.score.toFixed(2)}`;

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.font = '22px Arial';
    const textWidth = ctx.measureText(label).width;
    const labelY = Math.max(0, y1 - 32);
    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, textWidth + 16, 30);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x1 + 8, labelY + 22);

    ctx.fillStyle = '#ffe066';
    face.landmarks.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

function drawLoop() {
  drawDetections();
  drawAnimation = requestAnimationFrame(drawLoop);
}

async function recognizeCurrentFrame() {
  if (!detectStream || recognizeBusy) {
    return;
  }

  recognizeBusy = true;

  try {
    const blob = await captureFrameBlob(detectVideo);
    const form = new FormData();
    form.append('file', blob, 'frame.jpg');

    const response = await fetch('/recognize', { method: 'POST', body: form });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Nhận diện thất bại.');
    }

    smoothIncomingFaces(data.faces || []);
    setMessage(detectMessage, `Đang nhận diện: ${latestFaces.length} khuôn mặt.`, 'ok');
    databaseSize.textContent = `${data.database_size} embeddings`;
  } catch (error) {
    setMessage(detectMessage, error.message, 'error');
  } finally {
    recognizeBusy = false;
  }
}

startDetectBtn.addEventListener('click', async () => {
  try {
    detectStream = await startCamera(detectVideo, detectStream);
    setMessage(detectMessage, 'Camera đã bật. Đang nhận diện...', 'ok');

    if (detectTimer) {
      clearInterval(detectTimer);
    }

    if (drawAnimation) {
      cancelAnimationFrame(drawAnimation);
    }

    latestFaces = [];
    smoothedFaces = [];
    drawAnimation = requestAnimationFrame(drawLoop);
    detectTimer = setInterval(recognizeCurrentFrame, 450);
    recognizeCurrentFrame();
  } catch (error) {
    setMessage(detectMessage, 'Không mở được camera. Hãy kiểm tra quyền camera của trình duyệt.', 'error');
  }
});

stopDetectBtn.addEventListener('click', () => {
  if (detectTimer) {
    clearInterval(detectTimer);
    detectTimer = null;
  }

  if (drawAnimation) {
    cancelAnimationFrame(drawAnimation);
    drawAnimation = null;
  }

  if (detectStream) {
    detectStream.getTracks().forEach((track) => track.stop());
    detectStream = null;
  }

  latestFaces = [];
  smoothedFaces = [];
  recognizeBusy = false;
  drawDetections();
  setMessage(detectMessage, 'Đã dừng camera.');
});

window.addEventListener('load', async () => {
  await refreshStatus();
  await startEnrollCamera();
});

setInterval(refreshStatus, 5000);



