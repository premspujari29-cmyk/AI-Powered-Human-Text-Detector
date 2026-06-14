from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import io
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from scipy.sparse import hstack, csr_matrix

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# GLOBAL VARIABLES
# =====================================================

model_calibrated  = None   # calibrated SVC  → gives real probabilities
model_rf          = None   # random forest   → also gives probabilities
vectorizer        = None
AI_LABEL          = None   # whichever int label means "AI" in this dataset
HUMAN_LABEL       = None

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Words that appear far more often in AI-generated text
AI_VOCAB = {
    "overall", "moreover", "furthermore", "in conclusion", "therefore",
    "additionally", "however", "notably", "importantly", "essentially",
    "ultimately", "consequently", "nevertheless", "in summary",
    "it is worth", "it should be noted", "in this context",
    "a wide range", "a variety of", "plays a crucial role",
    "it is important to", "one of the most", "in today's world",
    "in recent years", "due to the fact", "as a result",
    "with that being said", "that being said", "to summarize",
    "to conclude", "as mentioned", "as previously", "in other words",
    "first and foremost", "last but not least", "on the other hand",
    "delve", "delves", "delving", "underscore", "underscores",
    "tapestry", "nuance", "nuanced", "multifaceted", "holistic",
    "paradigm", "leverage", "leveraging", "robust", "seamless",
    "cutting-edge", "state-of-the-art", "in-depth", "key takeaway",
    "best practices", "game-changing", "foster", "fosters"
}

def extract_features(text):
    """
    Returns a dense feature vector capturing statistical signals
    that distinguish AI text from human text.
    """
    raw   = str(text)
    lower = raw.lower()
    words = raw.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', raw) if s.strip()]

    word_count        = len(words)
    sentence_count    = max(len(sentences), 1)
    avg_word_len      = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len      = word_count / sentence_count
    unique_ratio      = len(set(w.lower() for w in words)) / max(word_count, 1)
    punct_count       = len(re.findall(r'[,.!?;:]', raw))
    uppercase_ratio   = sum(1 for c in raw if c.isupper()) / max(len(raw), 1)

    # Sentence-length variance — human writing is bursty; AI is uniform
    sent_lengths      = [len(s.split()) for s in sentences]
    sent_len_std      = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    sent_len_range    = (max(sent_lengths) - min(sent_lengths)) if sent_lengths else 0

    # AI vocabulary density
    ai_hits           = sum(1 for phrase in AI_VOCAB if phrase in lower)
    ai_vocab_density  = ai_hits / max(word_count / 10, 1)

    # Transition word ratio (AI loves them)
    transition_words  = [
        "however", "therefore", "furthermore", "moreover", "additionally",
        "consequently", "nevertheless", "nonetheless", "thus", "hence",
        "accordingly", "subsequently", "meanwhile", "conversely"
    ]
    transition_count  = sum(1 for w in words if w.lower() in transition_words)
    transition_ratio  = transition_count / max(word_count, 1)

    # Paragraph structure — AI rarely writes very short sentences
    short_sents       = sum(1 for l in sent_lengths if l < 5)
    short_sent_ratio  = short_sents / sentence_count

    # Exclamation / question marks — humans use them; AI rarely does
    exclamations      = raw.count("!")
    questions         = raw.count("?")
    informal_ratio    = (exclamations + questions) / max(sentence_count, 1)

    # Contraction usage — humans contract; AI often doesn't
    contractions      = len(re.findall(
        r"\b(don't|can't|won't|isn't|aren't|i'm|you're|they're|it's|"
        r"i've|we've|i'd|you'd|they'd|i'll|you'll|we'll|didn't|"
        r"doesn't|wasn't|weren't|couldn't|wouldn't|shouldn't)\b",
        lower
    ))
    contraction_ratio = contractions / max(word_count / 10, 1)

    # First-person usage — strong human signal
    first_person      = len(re.findall(
        r'\b(i|me|my|myself|mine|we|us|our|ours|ourselves)\b', lower
    ))
    first_person_ratio = first_person / max(word_count, 1)

    # Repetition: consecutive similar-length sentences (AI pattern)
    consec_similar    = 0
    for i in range(1, len(sent_lengths)):
        if abs(sent_lengths[i] - sent_lengths[i-1]) <= 2:
            consec_similar += 1
    consec_ratio      = consec_similar / max(sentence_count - 1, 1)

    return np.array([[
        word_count,
        sentence_count,
        avg_word_len,
        avg_sent_len,
        unique_ratio,
        punct_count,
        uppercase_ratio,
        sent_len_std,
        sent_len_range,
        ai_vocab_density,
        transition_ratio,
        short_sent_ratio,
        informal_ratio,
        contraction_ratio,
        first_person_ratio,
        consec_ratio,
        ai_hits,
        transition_count,
        exclamations,
        questions,
    ]])

# =====================================================
# LABEL ORIENTATION DETECTION
# =====================================================

# A strongly AI-sounding probe sentence used to calibrate label orientation
_AI_PROBE = (
    "furthermore it is important to note that in conclusion the overall "
    "multifaceted paradigm leverages a robust and seamless tapestry of "
    "nuanced best practices that underscore the holistic approach to "
    "state-of-the-art cutting-edge solutions additionally it should be "
    "noted that consequently this fosters a wide range of key takeaways"
)

# A strongly human-sounding probe sentence
_HUMAN_PROBE = (
    "i can't believe how tired i was after that, honestly didn't think "
    "i'd make it through the day. my legs were killing me and i just "
    "wanted to go home. you know what i mean? it's just one of those "
    "days where nothing goes right and i'm like, why am i even doing this"
)

def detect_ai_label_via_probe(fitted_vectorizer, fitted_calibrated, fitted_rf):
    """
    After training, send a known-AI sentence and a known-human sentence
    through the model. Whichever label the model assigns to the AI probe
    IS the AI label. This is immune to dataset label convention differences.

    Returns (ai_label, human_label).
    """
    def get_avg_proba(text):
        cleaned = clean_text(text)
        tv = fitted_vectorizer.transform([cleaned])
        fv = extract_features(cleaned)
        xi = hstack([tv, csr_matrix(fv)])
        p1 = fitted_calibrated.predict_proba(xi)[0]
        p2 = fitted_rf.predict_proba(xi)[0]
        return (p1 + p2) / 2   # shape: (n_classes,)

    classes    = fitted_calibrated.classes_          # e.g. [0, 1]
    ai_proba   = get_avg_proba(_AI_PROBE)
    hum_proba  = get_avg_proba(_HUMAN_PROBE)

    # The label that scores HIGHER for the AI probe is the AI label
    ai_label_idx    = int(np.argmax(ai_proba))
    human_label_idx = int(np.argmax(hum_proba))

    ai_label    = int(classes[ai_label_idx])
    human_label = int(classes[human_label_idx])

    # Sanity-check: if both probes map to the same label, fall back to
    # whichever class has a higher mean AI-vocab score in the training data
    if ai_label == human_label:
        # Means the model isn't confident either way — pick 1 as AI by default
        ai_label    = 1
        human_label = 0

    return ai_label, human_label

# =====================================================
# HTML PAGE
# =====================================================

HTML_PAGE = """
<!DOCTYPE html>
<html>

<head>
<title>AI Powered Human Text Detector</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/tsparticles@2/tsparticles.bundle.min.js"></script>

<style>
* { margin:0; padding:0; box-sizing:border-box; font-family:Arial; }

body { background:#050816; color:white; overflow-x:hidden; }

#tsparticles {
    position:fixed; width:100%; height:100%;
    z-index:-2; top:0; left:0;
}

.container {
    width:90%; max-width:1300px; margin:auto;
    padding-top:40px; padding-bottom:50px; text-align:center;
}

h1 {
    font-size:58px; margin-bottom:20px;
    background:linear-gradient(to right,#00d4ff,#7c3aed);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

.upload-box {
    background:rgba(255,255,255,0.08); padding:40px;
    border-radius:25px; margin-bottom:30px;
    backdrop-filter:blur(10px); transition:0.4s;
}
.upload-box:hover { transform:translateY(-5px); }

input[type=file] {
    width:100%; padding:18px; border-radius:15px;
    background:#111827; color:white; border:none; margin-bottom:20px;
}

textarea {
    width:100%; height:220px; border:none; outline:none;
    padding:20px; border-radius:20px; background:#0f172a;
    color:white; font-size:18px; resize:none;
    margin-top:20px; margin-bottom:20px;
}

button {
    padding:18px 45px; border:none; border-radius:15px;
    font-size:18px; cursor:pointer;
    background:linear-gradient(to right,#06b6d4,#7c3aed);
    color:white; transition:0.4s; font-weight:bold; margin:10px;
}
button:hover { transform:scale(1.05); box-shadow:0 0 25px #7c3aed; }
button:disabled { opacity:0.5; cursor:not-allowed; transform:none; box-shadow:none; }

.status { margin-top:20px; font-size:20px; font-weight:bold; }
.status.success { color:#06b6d4; }
.status.error   { color:#f87171; }

.result {
    margin-top:30px; background:rgba(255,255,255,0.08);
    padding:30px; border-radius:20px;
}

.cards {
    display:flex; justify-content:center;
    gap:25px; flex-wrap:wrap; margin-top:25px;
}

.card {
    background:#111827; padding:30px;
    border-radius:20px; width:220px; transition:0.4s;
}
.card:hover { transform:translateY(-10px); }
.card p { font-size:35px; font-weight:bold; margin-top:10px; }

.progress-container { margin-top:30px; }
.progress-title     { margin-bottom:12px; font-size:20px; }

.progress-bar {
    width:100%; height:25px; background:#1e293b;
    border-radius:20px; overflow:hidden;
}
.progress-fill {
    height:100%; width:0%;
    background:linear-gradient(to right,#06b6d4,#7c3aed); transition:1s;
}

.loader {
    width:60px; height:60px;
    border:6px solid rgba(255,255,255,0.2);
    border-top:6px solid #7c3aed;
    border-radius:50%; margin:20px auto;
    display:none; animation:spin 1s linear infinite;
}
@keyframes spin { 100% { transform:rotate(360deg); } }

.small-chart { width:260px; height:260px; margin:auto; margin-top:30px; }

.voice-controls { margin-top:15px; }
#voiceBtn.listening { animation:pulse 1s infinite; }
@keyframes pulse {
    0%   { box-shadow:0 0 0 0 rgba(124,58,237,0.7); }
    70%  { box-shadow:0 0 0 20px rgba(124,58,237,0); }
    100% { box-shadow:0 0 0 0 rgba(124,58,237,0); }
}

/* detail breakdown panel */
.breakdown {
    margin-top:25px; background:#0f172a;
    border-radius:15px; padding:20px; text-align:left;
}
.breakdown h3 { text-align:center; margin-bottom:15px; color:#06b6d4; }
.signal-row {
    display:flex; justify-content:space-between;
    align-items:center; margin-bottom:10px;
}
.signal-label { font-size:14px; color:#94a3b8; width:55%; }
.signal-bar-wrap { width:40%; height:10px; background:#1e293b; border-radius:10px; overflow:hidden; }
.signal-bar-fill { height:100%; border-radius:10px; transition:0.8s; }
.signal-val { font-size:13px; color:white; width:5%; text-align:right; }
</style>
</head>

<body>
<div id="tsparticles"></div>

<div class="container">

  <h1>AI Powered Human Text Detector</h1>

  <!-- TRAIN -->
  <div class="upload-box">
    <h2>Upload Dataset CSV</h2>
    <p style="color:#94a3b8;margin-bottom:15px;font-size:14px;">
      CSV must have two columns: <b>text</b> and <b>label</b> (0 or 1).
      The app auto-detects which label is AI vs Human.
    </p>
    <input type="file" id="datasetFile" accept=".csv">
    <button onclick="uploadDataset()" id="trainBtn">Train Advanced Model</button>
    <div class="status" id="trainStatus">No Dataset Uploaded</div>
  </div>

  <!-- ANALYZE -->
  <div class="upload-box">
    <h2>Paste Text For Detection</h2>
    <textarea id="textInput" placeholder="Paste AI or Human text here (minimum ~50 words for best results)..."></textarea>
    <div class="voice-controls">
      <button id="voiceBtn">Start Voice Typing</button>
    </div>
    <button onclick="analyzeText()" id="analyzeBtn">Analyze Text</button>
    <div id="loader" class="loader"></div>
  </div>

  <!-- RESULTS -->
  <div class="result">
    <h2 id="prediction">Waiting For Analysis...</h2>

    <div class="progress-container">
      <div class="progress-title">AI Probability</div>
      <div class="progress-bar">
        <div class="progress-fill" id="purityBar"></div>
      </div>
    </div>

    <div class="cards">
      <div class="card"><h3>Human Score</h3><p id="humanScore">0%</p></div>
      <div class="card"><h3>AI Score</h3>   <p id="aiScore">0%</p></div>
      <div class="card"><h3>Confidence</h3> <p id="confidence">0%</p></div>
    </div>

    <div class="small-chart">
      <canvas id="chart"></canvas>
    </div>

    <!-- Signal breakdown -->
    <div class="breakdown" id="breakdown" style="display:none;">
      <h3>Signal Breakdown</h3>
      <div id="signalRows"></div>
    </div>
  </div>

</div>

<script>

// ── Particles ────────────────────────────────────────────────────────────────
let chart;
tsParticles.load("tsparticles", {
  particles: {
    number: { value: 60 },
    color:  { value: "#7c3aed" },
    links:  { enable: true, distance: 120, color: "#06b6d4", opacity: 0.2 },
    move:   { enable: true, speed: 1 },
    size:   { value: 2 },
    opacity:{ value: 0.5 }
  }
});

// ── Train ────────────────────────────────────────────────────────────────────
async function uploadDataset() {
  const file = document.getElementById("datasetFile").files[0];
  if (!file) { alert("Please upload a CSV dataset"); return; }

  const statusEl  = document.getElementById("trainStatus");
  const trainBtn  = document.getElementById("trainBtn");
  trainBtn.disabled   = true;
  statusEl.className  = "status";
  statusEl.innerText  = "Training Advanced AI Model...";

  try {
    const fd = new FormData();
    fd.append("file", file);
    const res  = await fetch('/train', { method: 'POST', body: fd });
    const data = await res.json();
    statusEl.innerText  = data.message;
    statusEl.className  = (data.message.toLowerCase().includes("error") ||
                           data.message.toLowerCase().includes("must")  ||
                           data.message.toLowerCase().includes("small"))
                          ? "status error" : "status success";
  } catch(e) {
    statusEl.innerText = "Network error: " + e.message;
    statusEl.className = "status error";
  } finally {
    trainBtn.disabled = false;
  }
}

// ── Analyze ──────────────────────────────────────────────────────────────────
async function analyzeText() {
  const text = document.getElementById("textInput").value;
  if (text.trim() === "") { alert("Please enter text to analyze"); return; }

  const loader     = document.getElementById("loader");
  const analyzeBtn = document.getElementById("analyzeBtn");
  loader.style.display = "block";
  analyzeBtn.disabled  = true;

  try {
    const fd = new FormData();
    fd.append("text", text);
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();

    if (data.error) { alert(data.error); return; }

    document.getElementById("prediction").innerText  = data.prediction;
    document.getElementById("humanScore").innerText  = data.human_score + "%";
    document.getElementById("aiScore").innerText     = data.ai_score    + "%";
    document.getElementById("confidence").innerText  = data.confidence  + "%";
    document.getElementById("purityBar").style.width = data.ai_score    + "%";

    updateChart(data.human_score, data.ai_score);
    renderSignals(data.signals);
  } catch(e) {
    alert("Network error: " + e.message);
  } finally {
    loader.style.display = "none";
    analyzeBtn.disabled  = false;
  }
}

// ── Chart ─────────────────────────────────────────────────────────────────────
function updateChart(human, ai) {
  const ctx = document.getElementById("chart").getContext("2d");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Human','AI'],
      datasets: [{
        data: [human, ai],
        backgroundColor: ['#06b6d4','#7c3aed'],
        hoverOffset: 15, borderWidth: 2, borderColor: '#050816'
      }]
    },
    options: {
      responsive: true, cutout: '75%',
      plugins: { legend: { position:'bottom', labels:{ color:'white' } } }
    }
  });
}

// ── Signal breakdown ──────────────────────────────────────────────────────────
function renderSignals(signals) {
  if (!signals || signals.length === 0) return;
  const panel = document.getElementById("breakdown");
  const rows  = document.getElementById("signalRows");
  panel.style.display = "block";
  rows.innerHTML = "";

  signals.forEach(s => {
    const color = s.points_to === "AI" ? "#7c3aed" : "#06b6d4";
    rows.innerHTML += `
      <div class="signal-row">
        <span class="signal-label">${s.label}</span>
        <div class="signal-bar-wrap">
          <div class="signal-bar-fill"
               style="width:${s.strength}%;background:${color}"></div>
        </div>
        <span class="signal-val" style="color:${color}">${s.points_to}</span>
      </div>`;
  });
}

// ── Voice typing ──────────────────────────────────────────────────────────────
let recognition = null;
let isListening  = false;
const voiceBtn   = document.getElementById("voiceBtn");
voiceBtn.addEventListener("click", toggleVoiceTyping);

function toggleVoiceTyping() {
  isListening ? stopVoiceTyping() : startVoiceTyping();
}

function startVoiceTyping() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    alert("Voice recognition not supported. Please use Chrome."); return;
  }
  const SR  = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.continuous     = true;
  recognition.interimResults = true;
  recognition.lang           = "en-US";
  isListening = true;
  voiceBtn.classList.add("listening");
  voiceBtn.innerText = "Stop Voice Typing";

  let finalT = document.getElementById("textInput").value;
  if (finalT && !finalT.endsWith(" ")) finalT += " ";

  recognition.onresult = e => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      e.results[i].isFinal ? (finalT += t + " ") : (interim += t);
    }
    document.getElementById("textInput").value = finalT + interim;
  };
  recognition.onerror = () => stopVoiceTyping();
  recognition.onend   = () => { if (isListening) recognition.start(); };
  recognition.start();
}

function stopVoiceTyping() {
  isListening = false;
  if (recognition) { recognition.onend = null; recognition.stop(); recognition = null; }
  voiceBtn.classList.remove("listening");
  voiceBtn.innerText = "Start Voice Typing";
}

</script>
</body>
</html>
"""

# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

# =====================================================
# TRAIN
# =====================================================

@app.route("/train", methods=["POST"])
def train():
    global model_calibrated, model_rf, vectorizer, AI_LABEL, HUMAN_LABEL

    if "file" not in request.files:
        return jsonify({"message": "Please Upload CSV File"})

    file = request.files["file"]

    try:
        raw_text = file.stream.read().decode("utf-8", errors="ignore")

        if raw_text.startswith("{\\rtf"):
            raw_text = re.sub(r'\{\\.*?\}|\\[a-z]+[0-9]* ?', '', raw_text)
            raw_text = raw_text.replace("\\", "\n")

        data = pd.read_csv(
            io.StringIO(raw_text),
            sep=",", engine="python", on_bad_lines="skip"
        )

        if len(data.columns) < 2:
            return jsonify({"message": "Dataset must have at least 2 columns (text, label)"})

        data = data.iloc[:, :2]
        data.columns = ["text", "label"]
        data = data.dropna()
        data["text"]  = data["text"].astype(str)
        data["label"] = pd.to_numeric(data["label"], errors="coerce")
        data = data.dropna()
        data["label"] = data["label"].astype(int)
        data = data[data["label"].isin([0, 1])].drop_duplicates()
        data["text"]  = data["text"].apply(clean_text)

        n = len(data)
        if n < 20:
            return jsonify({"message": "Dataset too small — need at least 20 rows."})

        # ── Adaptive TF-IDF ───────────────────────────────────────────────────
        if n < 100:
            max_feat, ngram, max_df_v = 3000,  (1,1), 1.0
        elif n < 1000:
            max_feat, ngram, max_df_v = 10000, (1,2), 0.95
        else:
            max_feat, ngram, max_df_v = 30000, (1,3), 0.95

        X_train, X_test, y_train, y_test = train_test_split(
            data["text"], data["label"],
            test_size=0.2, random_state=42, stratify=data["label"]
        )

        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=ngram,
            max_features=max_feat, sublinear_tf=True,
            min_df=1, max_df=max_df_v
        )

        Xtr_tfidf = vectorizer.fit_transform(X_train)
        Xte_tfidf = vectorizer.transform(X_test)

        Xtr_feat = np.vstack([extract_features(t)[0] for t in X_train])
        Xte_feat = np.vstack([extract_features(t)[0] for t in X_test])

        Xtr = hstack([Xtr_tfidf, csr_matrix(Xtr_feat)])
        Xte = hstack([Xte_tfidf, csr_matrix(Xte_feat)])

        # ── Calibrated SVC ────────────────────────────────────────────────────
        base_svc = LinearSVC(max_iter=3000, C=1.0)
        model_calibrated = CalibratedClassifierCV(base_svc, cv=min(5, max(2, n // 50)))
        model_calibrated.fit(Xtr, y_train)

        # ── Random Forest ─────────────────────────────────────────────────────
        model_rf = RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        model_rf.fit(Xtr, y_train)

        # ── Auto-detect which label means AI (after training, using probes) ───
        AI_LABEL, HUMAN_LABEL = detect_ai_label_via_probe(
            vectorizer, model_calibrated, model_rf
        )

        # ── Ensemble accuracy ─────────────────────────────────────────────────
        prob_svc = model_calibrated.predict_proba(Xte)
        prob_rf  = model_rf.predict_proba(Xte)
        prob_avg = (prob_svc + prob_rf) / 2
        classes  = model_calibrated.classes_

        final_pred = [classes[np.argmax(p)] for p in prob_avg]
        accuracy   = round(accuracy_score(y_test, final_pred) * 100, 2)

        label_info = f"(label {AI_LABEL}=AI, label {HUMAN_LABEL}=Human)"

        return jsonify({
            "message": f"Model Trained Successfully {label_info} | Accuracy: {accuracy}%"
        })

    except Exception as e:
        return jsonify({"message": f"Training Error: {str(e)}"})


# =====================================================
# PREDICT
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():
    global model_calibrated, model_rf, vectorizer, AI_LABEL, HUMAN_LABEL

    if model_calibrated is None:
        return jsonify({"error": "Please upload and train a dataset first"})

    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"})

    cleaned = clean_text(text)

    tfidf_vec   = vectorizer.transform([cleaned])
    feat_vec    = extract_features(cleaned)
    final_input = hstack([tfidf_vec, csr_matrix(feat_vec)])

    # ── Calibrated probabilities from both models ─────────────────────────────
    prob_svc  = model_calibrated.predict_proba(final_input)[0]
    prob_rf   = model_rf.predict_proba(final_input)[0]
    prob_avg  = (prob_svc + prob_rf) / 2

    classes   = model_calibrated.classes_          # e.g. [0, 1]
    ai_idx    = list(classes).index(AI_LABEL)      # index of AI class
    human_idx = list(classes).index(HUMAN_LABEL)   # index of Human class

    ai_prob    = float(prob_avg[ai_idx])
    human_prob = float(prob_avg[human_idx])

    # ── Heuristic boost layer ─────────────────────────────────────────────────
    raw_text_lower = text.lower()
    words          = text.split()
    sentences      = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sent_lengths   = [len(s.split()) for s in sentences]
    burstiness     = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    ai_vocab_hits = sum(1 for p in AI_VOCAB if p in raw_text_lower)

    contractions = len(re.findall(
        r"\b(don't|can't|won't|isn't|aren't|i'm|you're|they're|it's|"
        r"i've|we've|i'd|you'd|they'd|i'll|you'll|we'll|didn't|"
        r"doesn't|wasn't|weren't|couldn't|wouldn't|shouldn't)\b",
        raw_text_lower
    ))
    first_person = len(re.findall(
        r'\b(i|me|my|myself|mine)\b', raw_text_lower
    ))

    # Signals that push probability toward AI
    heuristic_delta = 0.0

    if ai_vocab_hits >= 3:
        heuristic_delta += 0.06 * min(ai_vocab_hits, 8)

    if burstiness < 3 and len(sent_lengths) >= 3:
        heuristic_delta += 0.08   # very uniform sentence lengths → AI

    if contractions == 0 and len(words) > 50:
        heuristic_delta += 0.06   # no contractions in long text → AI

    if first_person == 0 and len(words) > 50:
        heuristic_delta += 0.05   # no first-person → likely AI

    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    if avg_sent_len > 22:
        heuristic_delta += 0.05   # very long avg sentence → AI

    # Signals that push toward Human
    if contractions >= 3:
        heuristic_delta -= 0.06
    if first_person >= 4:
        heuristic_delta -= 0.06
    if burstiness > 8:
        heuristic_delta -= 0.05   # very bursty → human

    # Apply delta and re-normalise
    ai_prob    = max(0.01, min(0.99, ai_prob    + heuristic_delta))
    human_prob = max(0.01, min(0.99, 1.0 - ai_prob))

    # ── Format scores ─────────────────────────────────────────────────────────
    ai_score_pct    = round(ai_prob    * 100, 1)
    human_score_pct = round(human_prob * 100, 1)
    confidence      = round(max(ai_prob, human_prob) * 100, 1)

    prediction_label = "AI Generated Text" if ai_prob > 0.5 else "Human Written Text"

    # ── Build signal breakdown for UI ─────────────────────────────────────────
    signals = [
        {
            "label":     "AI Vocabulary Density",
            "strength":  min(int(ai_vocab_hits / 8 * 100), 100),
            "points_to": "AI" if ai_vocab_hits >= 2 else "Human"
        },
        {
            "label":     "Sentence Length Uniformity",
            "strength":  max(0, int((10 - burstiness) / 10 * 100)),
            "points_to": "AI" if burstiness < 4 else "Human"
        },
        {
            "label":     "Contraction Usage",
            "strength":  min(int(contractions / 5 * 100), 100),
            "points_to": "Human" if contractions >= 2 else "AI"
        },
        {
            "label":     "First-Person Voice",
            "strength":  min(int(first_person / 8 * 100), 100),
            "points_to": "Human" if first_person >= 2 else "AI"
        },
        {
            "label":     "Average Sentence Length",
            "strength":  min(int(avg_sent_len / 30 * 100), 100),
            "points_to": "AI" if avg_sent_len > 20 else "Human"
        },
        {
            "label":     "Model Confidence",
            "strength":  int(confidence),
            "points_to": "AI" if ai_prob > 0.5 else "Human"
        },
    ]

    return jsonify({
        "prediction":  prediction_label,
        "human_score": human_score_pct,
        "ai_score":    ai_score_pct,
        "confidence":  confidence,
        "signals":     signals
    })


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
