from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import joblib
import textwrap
import os
from mediapipe import solutions as mp_solutions

app = Flask(__name__)

# ==============================
# Load model, scaler, encoder
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "isl_twohands_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler_twohands.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "models", "label_encoder_twohands.pkl")

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ==============================
# Mediapipe hand detector
# ==============================
mp_hands = mp_solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
)

def extract_twohand_landmarks_from_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_model.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None

    hands_dict = {}
    for lm, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
        hands_dict[handed.classification[0].label] = lm

    coords = []
    for side in ["Left", "Right"]:
        if side in hands_dict:
            for p in hands_dict[side].landmark:
                coords.extend([p.x, p.y, p.z])
        else:
            coords.extend([0.0] * 63)

    return coords

def predict_sign_from_image(image_bgr):
    landmarks = extract_twohand_landmarks_from_image(image_bgr)
    if landmarks is None:
        return None

    x = scaler.transform(np.array(landmarks).reshape(1, -1))
    pred_idx = clf.predict(x)[0]
    return label_encoder.inverse_transform([pred_idx])[0]

# ==============================
# Routes
# ==============================

@app.route("/")
def home():
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>ISL Two-Hand – Text + Speech</title>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui; }
body { background: radial-gradient(circle at top,#0f172a,#020617 60%); color:#e5e7eb; }
.container { max-width: 1000px; margin:auto; padding:1.5rem; }
.video-area { display:flex; gap:1.5rem; flex-wrap:wrap; }
.video-box, .text-box {
  flex:1 1 480px; background:#0f172a; border-radius:1rem;
  padding:1rem; border:1px solid #334155;
}
video {
  width:100%; height:420px; object-fit:cover;
  border-radius:0.75rem; background:black;
  transform:scaleX(-1);
}
.subtitle-bar {
  margin-top:0.5rem; padding:0.5rem;
  background:#020617; border-radius:999px; text-align:center;
}
.subtitle-text { font-weight:600; color:#22c55e; }
textarea {
  width:100%; height:180px; background:#020617;
  color:white; border-radius:0.75rem;
  padding:0.75rem; border:1px solid #334155;
}
.note { margin-top:0.5rem; font-size:0.8rem; color:#9ca3af; }
button {
  margin-top:0.5rem; padding:0.4rem 0.8rem;
  border-radius:999px; border:none; cursor:pointer;
  background:#22c55e; color:#020617; font-weight:600;
}
button:disabled {
  opacity:0.6; cursor:default;
}
</style>
</head>

<body>
<div class="container">
<h2 style="text-align:center">ISL Two-Hand – Auto Subtitles + Voice</h2>
<p style="text-align:center;color:#9ca3af">
Signs supported (from your dataset): A, HI, THANKYOU
</p>

<div class="video-area">
  <div class="video-box">
    <video id="video" autoplay muted></video>
    <div class="subtitle-bar">
      Current Sign:
      <span id="currentSign" class="subtitle-text">...</span>
    </div>
    <p class="note">Hold a sign steady for ~2 seconds to auto-add and speak it.</p>
  </div>

  <div class="text-box">
    <h3>Conversation Text</h3>
    <textarea id="fullText" readonly></textarea>
    <button id="clearBtn">Clear</button>
    <p class="note">Recognized signs are appended here and spoken aloud.</p>
  </div>
</div>
</div>

<canvas id="canvas" style="display:none;"></canvas>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const currentSignEl = document.getElementById("currentSign");
const fullTextEl = document.getElementById("fullText");
const clearBtn = document.getElementById("clearBtn");

// Auto-accept variables
let lastStableSign = null;
let signStartTime = null;
let lastAcceptTime = 0;

const HOLD_TIME = 2000;
const COOLDOWN_TIME = 1500;

// Text-to-speech
function speakText(text) {
  if (!("speechSynthesis" in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-IN";
  utterance.rate = 0.9;
  speechSynthesis.speak(utterance);
}

// Start camera
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => { video.srcObject = stream; })
  .catch(() => alert("Camera permission required"));

// Capture & predict
function captureAndSend() {
  if (!video.videoWidth) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    if (!blob) return;

    const formData = new FormData();
    formData.append("frame", blob, "frame.jpg");

    fetch("/predict", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        const now = Date.now();

        if (data.prediction) {
          const pred = data.prediction;
          currentSignEl.textContent = pred.toUpperCase();

          if (pred !== lastStableSign) {
            lastStableSign = pred;
            signStartTime = now;
          } else if (
            signStartTime &&
            now - signStartTime >= HOLD_TIME &&
            now - lastAcceptTime >= COOLDOWN_TIME
          ) {
            if (pred.length === 1) {
              fullTextEl.value += pred;
              speakText(pred);
            } else {
              fullTextEl.value += pred.toUpperCase() + " ";
              speakText(pred);
            }
            lastAcceptTime = now;
            signStartTime = null;
          }
        } else {
          currentSignEl.textContent = "...";
          lastStableSign = null;
          signStartTime = null;
        }
      })
      .catch(err => {
        console.error(err);
      });
  }, "image/jpeg", 0.8);
}

setInterval(captureAndSend, 800);

clearBtn.addEventListener("click", () => {
  fullTextEl.value = "";
});
</script>
</body>
</html>
"""
    return Response(textwrap.dedent(html), mimetype="text/html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "frame" not in request.files:
        return jsonify({"prediction": None})

    file_bytes = np.frombuffer(request.files["frame"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"prediction": None})

    pred = predict_sign_from_image(img)
    return jsonify({"prediction": pred})

if __name__ == "__main__":
    print("[INFO] ISL Web App running at http://localhost:5000")
    app.run(debug=True)