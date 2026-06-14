from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import io
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from scipy.sparse import hstack

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# GLOBAL VARIABLES
# =====================================================

model1 = None
model2 = None
vectorizer = None

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)

    text = re.sub(r"\\s+", " ", text).strip()

    return text

# =====================================================
# ADVANCED NLP FEATURES
# =====================================================

def extract_advanced_features(text):

    text = str(text)

    words = text.split()

    sentences = re.split(r'[.!?]+', text)

    word_count = len(words)

    sentence_count = len(sentences)

    avg_word_length = np.mean(
        [len(w) for w in words]
    ) if words else 0

    avg_sentence_length = (
        word_count / max(sentence_count, 1)
    )

    unique_ratio = (
        len(set(words)) / max(word_count, 1)
    )

    punctuation_count = len(
        re.findall(r'[,.!?;:]', text)
    )

    uppercase_ratio = sum(
        1 for c in text if c.isupper()
    ) / max(len(text), 1)

    return np.array([[
        word_count,
        sentence_count,
        avg_word_length,
        avg_sentence_length,
        unique_ratio,
        punctuation_count,
        uppercase_ratio
    ]])

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

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:Arial;
}

body{
background:#050816;
color:white;
overflow-x:hidden;
}

#tsparticles{
position:fixed;
width:100%;
height:100%;
z-index:-2;
top:0;
left:0;
}

.container{
width:90%;
max-width:1300px;
margin:auto;
padding-top:40px;
padding-bottom:50px;
text-align:center;
}

h1{
font-size:58px;
margin-bottom:20px;
background:linear-gradient(to right,#00d4ff,#7c3aed);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.upload-box{
background:rgba(255,255,255,0.08);
padding:40px;
border-radius:25px;
margin-bottom:30px;
backdrop-filter:blur(10px);
transition:0.4s;
}

.upload-box:hover{
transform:translateY(-5px);
}

input[type=file]{
width:100%;
padding:18px;
border-radius:15px;
background:#111827;
color:white;
border:none;
margin-bottom:20px;
}

textarea{
width:100%;
height:220px;
border:none;
outline:none;
padding:20px;
border-radius:20px;
background:#0f172a;
color:white;
font-size:18px;
resize:none;
margin-top:20px;
margin-bottom:20px;
}

button{
padding:18px 45px;
border:none;
border-radius:15px;
font-size:18px;
cursor:pointer;
background:linear-gradient(to right,#06b6d4,#7c3aed);
color:white;
transition:0.4s;
font-weight:bold;
margin:10px;
}

button:hover{
transform:scale(1.05);
box-shadow:0 0 25px #7c3aed;
}

.status{
margin-top:20px;
font-size:20px;
font-weight:bold;
}

.result{
margin-top:30px;
background:rgba(255,255,255,0.08);
padding:30px;
border-radius:20px;
}

.cards{
display:flex;
justify-content:center;
gap:25px;
flex-wrap:wrap;
margin-top:25px;
}

.card{
background:#111827;
padding:30px;
border-radius:20px;
width:220px;
transition:0.4s;
}

.card:hover{
transform:translateY(-10px);
}

.card p{
font-size:35px;
font-weight:bold;
margin-top:10px;
}

.progress-container{
margin-top:30px;
}

.progress-title{
margin-bottom:12px;
font-size:20px;
}

.progress-bar{
width:100%;
height:25px;
background:#1e293b;
border-radius:20px;
overflow:hidden;
}

.progress-fill{
height:100%;
width:0%;
background:linear-gradient(to right,#06b6d4,#7c3aed);
transition:1s;
}

.loader{
width:60px;
height:60px;
border:6px solid rgba(255,255,255,0.2);
border-top:6px solid #7c3aed;
border-radius:50%;
margin:20px auto;
display:none;
animation:spin 1s linear infinite;
}

@keyframes spin{
100%{
transform:rotate(360deg);
}
}

.small-chart{
width:260px;
height:260px;
margin:auto;
margin-top:30px;
}

.voice-controls{
margin-top:15px;
}

#voiceBtn.listening{
animation:pulse 1s infinite;
}

@keyframes pulse{

0%{
box-shadow:0 0 0 0 rgba(124,58,237,0.7);
}

70%{
box-shadow:0 0 0 20px rgba(124,58,237,0);
}

100%{
box-shadow:0 0 0 0 rgba(124,58,237,0);
}

}

</style>

</head>

<body>

<div id="tsparticles"></div>

<div class="container">

<h1>AI Powered Human Text Detector</h1>

<div class="upload-box">

<h2>Upload Dataset CSV</h2>

<input type="file" id="datasetFile">

<button onclick="uploadDataset()">
Train Advanced Model
</button>

<div class="status" id="trainStatus">
No Dataset Uploaded
</div>

</div>

<div class="upload-box">

<h2>Paste Text For Detection</h2>

<textarea id="textInput"
placeholder="Paste AI or Human text here..."></textarea>

<div class="voice-controls">

<button id="voiceBtn"
onclick="startVoiceTyping()">

Start Voice Typing

</button>

</div>

<button onclick="analyzeText()">
Analyze Text
</button>

<div id="loader" class="loader"></div>

</div>

<div class="result">

<h2 id="prediction">
Waiting For Analysis...
</h2>

<div class="progress-container">

<div class="progress-title">
AI Probability
</div>

<div class="progress-bar">

<div class="progress-fill" id="purityBar">
</div>

</div>

</div>

<div class="cards">

<div class="card">
<h3>Human Score</h3>
<p id="humanScore">0%</p>
</div>

<div class="card">
<h3>AI Score</h3>
<p id="aiScore">0%</p>
</div>

<div class="card">
<h3>Confidence</h3>
<p id="confidence">0%</p>
</div>

</div>

<div class="small-chart">
<canvas id="chart"></canvas>
</div>

</div>

</div>

<script>

let chart;

tsParticles.load("tsparticles",{

particles:{

number:{value:60},

color:{value:"#7c3aed"},

links:{
enable:true,
distance:120,
color:"#06b6d4",
opacity:0.2
},

move:{
enable:true,
speed:1
},

size:{value:2},

opacity:{value:0.5}

}

});

async function uploadDataset(){

const fileInput =
document.getElementById("datasetFile");

const file = fileInput.files[0];

if(!file){
alert("Please upload CSV dataset");
return;
}

const formData = new FormData();

formData.append("file", file);

document.getElementById("trainStatus").innerText =
"Training Advanced AI Model...";

const response = await fetch('/train',{

method:'POST',
body:formData

});

const data = await response.json();

document.getElementById("trainStatus").innerText =
data.message;

}

async function analyzeText(){

const text =
document.getElementById("textInput").value;

if(text.trim() === ""){
alert("Please enter text");
return;
}

document.getElementById("loader").style.display =
"block";

const formData = new FormData();

formData.append("text", text);

const response = await fetch('/predict',{

method:'POST',
body:formData

});

const data = await response.json();

document.getElementById("loader").style.display =
"none";

if(data.error){
alert(data.error);
return;
}

document.getElementById("prediction").innerText =
data.prediction;

document.getElementById("humanScore").innerText =
data.human_score + "%";

document.getElementById("aiScore").innerText =
data.ai_score + "%";

document.getElementById("confidence").innerText =
data.confidence + "%";

document.getElementById("purityBar").style.width =
data.ai_score + "%";

updateChart(data.human_score, data.ai_score);

}

function updateChart(human, ai){

const ctx =
document.getElementById("chart").getContext("2d");

if(chart){
chart.destroy();
}

chart = new Chart(ctx,{

type:'doughnut',

data:{

labels:['Human','AI'],

datasets:[{

data:[human, ai],

backgroundColor:[
'#06b6d4',
'#7c3aed'
],

hoverOffset:15,

borderWidth:2,

borderColor:'#050816'

}]

},

options:{

responsive:true,

cutout:'75%',

plugins:{

legend:{
position:'bottom',

labels:{
color:'white'
}
}

}

}

});

}

let recognition;

function startVoiceTyping(){

    if(!('webkitSpeechRecognition' in window)){

        alert("Voice recognition not supported");

        return;

    }

    recognition = new webkitSpeechRecognition();

    recognition.continuous = true;

    recognition.interimResults = true;

    recognition.lang = "en-US";

    const button =
    document.getElementById("voiceBtn");

    button.classList.add("listening");

    button.innerText = "Listening...";

    let finalTranscript =
    document.getElementById("textInput").value + " ";

    recognition.onresult = function(event){

        let interimTranscript = "";

        for(
            let i = event.resultIndex;
            i < event.results.length;
            i++
        ){

            const transcript =
            event.results[i][0].transcript;

            if(event.results[i].isFinal){

                finalTranscript += transcript + " ";

            }else{

                interimTranscript += transcript;

            }

        }

        document.getElementById("textInput").value =
        finalTranscript + interimTranscript;

    };

    recognition.onerror = function(){

        button.classList.remove("listening");

        button.innerText = "Start Voice Typing";

    };

    recognition.onend = function(){

        button.classList.remove("listening");

        button.innerText = "Start Voice Typing";

    };

    recognition.start();

    button.onclick = stopVoiceTyping;

}

function stopVoiceTyping(){

    if(recognition){

        recognition.stop();

    }

    const button =
    document.getElementById("voiceBtn");

    button.classList.remove("listening");

    button.innerText = "Start Voice Typing";

    button.onclick = startVoiceTyping;

}


const button =
document.getElementById("voiceBtn");

button.classList.remove("listening");

button.innerText = "Start Voice Typing";

button.onclick = startVoiceTyping;

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
# TRAIN MODEL
# =====================================================

@app.route("/train", methods=["POST"])
def train():

    global model1
    global model2
    global vectorizer

    if "file" not in request.files:

        return jsonify({
            "message": "Please Upload CSV File"
        })

    file = request.files["file"]

    try:

        raw_text = file.stream.read().decode(
            "utf-8",
            errors="ignore"
        )

        if raw_text.startswith("{\\rtf"):

            raw_text = re.sub(
                r'{\\\\.*?}|\\\\[a-z]+[0-9]* ?',
                '',
                raw_text
            )

            raw_text = raw_text.replace("\\", "\n")

        data = pd.read_csv(

            io.StringIO(raw_text),

            sep=",",

            engine="python",

            on_bad_lines="skip"

        )

        if len(data.columns) < 2:

            return jsonify({
                "message":
                "Dataset must contain at least 2 columns"
            })

        data = data.iloc[:, :2]

        data.columns = ["text", "label"]

        data = data.dropna()

        data["text"] = data["text"].astype(str)

        data["label"] = pd.to_numeric(
            data["label"],
            errors="coerce"
        )

        data = data.dropna()

        data["label"] = data["label"].astype(int)

        data = data[
            data["label"].isin([0,1])
        ]

        data = data.drop_duplicates()

        data["text"] = data["text"].apply(
            clean_text
        )

        dataset_size = len(data)

        if dataset_size < 10:

            return jsonify({
                "message":
                "Dataset too small. Upload at least 10 rows."
            })

        if dataset_size < 100:

            max_features = 3000
            ngram = (1,1)

        elif dataset_size < 1000:

            max_features = 10000
            ngram = (1,2)

        else:

            max_features = 30000
            ngram = (1,3)

        X_train, X_test, y_train, y_test = train_test_split(

            data["text"],
            data["label"],

            test_size=0.2,

            random_state=42,

            stratify=data["label"]

        )

        vectorizer = TfidfVectorizer(

            stop_words="english",

            ngram_range=ngram,

            max_features=max_features,

            sublinear_tf=True,

            min_df=1,

            max_df=1.0

        )

        X_train_vec = vectorizer.fit_transform(
            X_train
        )

        X_test_vec = vectorizer.transform(
            X_test
        )

        train_extra = np.vstack([

            extract_advanced_features(t)[0]

            for t in X_train

        ])

        test_extra = np.vstack([

            extract_advanced_features(t)[0]

            for t in X_test

        ])

        X_train_final = hstack([

            X_train_vec,
            train_extra

        ])

        X_test_final = hstack([

            X_test_vec,
            test_extra

        ])

        model1 = LinearSVC()

        model2 = RandomForestClassifier(

            n_estimators=200,

            random_state=42

        )

        model1.fit(
            X_train_final,
            y_train
        )

        model2.fit(
            X_train_final,
            y_train
        )

        pred1 = model1.predict(
            X_test_final
        )

        pred2 = model2.predict(
            X_test_final
        )

        final_pred = []

        for p1, p2 in zip(pred1, pred2):

            if p1 == p2:
                final_pred.append(p1)
            else:
                final_pred.append(p1)

        accuracy = round(

            accuracy_score(
                y_test,
                final_pred
            ) * 100,

            2

        )

        return jsonify({

            "message":
            f"Advanced Model Trained Successfully | Accuracy: {accuracy}%"

        })

    except Exception as e:

        return jsonify({

            "message":
            f"Training Error: {str(e)}"

        })

# =====================================================
# PREDICT
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    global model1
    global model2
    global vectorizer

    if model1 is None:

        return jsonify({
            "error":
            "Please Upload And Train Dataset First"
        })

    text = request.form.get("text")

    cleaned_text = clean_text(text)

    vectorized_text = vectorizer.transform(
        [cleaned_text]
    )

    extra_features = extract_advanced_features(
        cleaned_text
    )

    final_input = hstack([

        vectorized_text,
        extra_features

    ])

    pred1 = model1.predict(
        final_input
    )[0]

    pred2 = model2.predict(
        final_input
    )[0]

    if pred1 == pred2:
        prediction = pred1
    else:
        prediction = pred1

    score = abs(

        model1.decision_function(
            final_input
        )[0]

    )

    confidence = round(
        min(score * 20, 99),
        2
    )

    ai_patterns = [

        "overall",
        "moreover",
        "furthermore",
        "in conclusion",
        "therefore",
        "additionally",
        "however"

    ]

    pattern_hits = sum(
        1 for p in ai_patterns
        if p in text.lower()
    )

    sentences = re.split(r'[.!?]+', text)

    sentence_lengths = [

        len(s.split())

        for s in sentences

        if s.strip()

    ]

    burstiness = np.std(sentence_lengths)

    if burstiness < 3 and pattern_hits >= 2:
        prediction = 1

    if prediction == 1:

        ai_score = min(
            60 + confidence / 2,
            99
        )

        human_score = round(
            100 - ai_score,
            2
        )

        result = "AI Generated Text"

    else:

        human_score = min(
            60 + confidence / 2,
            99
        )

        ai_score = round(
            100 - human_score,
            2
        )

        result = "Human Written Text"

    return jsonify({

        "prediction": result,

        "human_score": round(
            human_score,
            2
        ),

        "ai_score": round(
            ai_score,
            2
        ),

        "confidence": confidence

    })

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(

        host="0.0.0.0",

        port=port

    )
